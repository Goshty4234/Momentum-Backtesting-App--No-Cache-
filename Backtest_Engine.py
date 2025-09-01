import numpy as np
# Backtest_Engine.py
import streamlit as st
import matplotlib.pyplot as plt
import io
import contextlib
import json
import threading
import time
import re
from datetime import datetime, date, timezone, timedelta
from typing import List, Dict
import yfinance as yf
import pandas as pd
import matplotlib.dates as mdates
import mplcursors
from scipy.optimize import newton, brentq, root_scalar
import pandas_market_calendars as mcal
from warnings import warn
import plotly.graph_objects as go
import logging
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

# Set up logging to capture print statements
logging.basicConfig(level=logging.INFO, format='%(message)s')
console_output = io.StringIO()
logger = logging.getLogger(__name__)

def sync_date_widgets_with_imported_values():
    """Sync date widgets with imported values from JSON"""
    # Sync start date
    if st.session_state.get('start_date') is not None:
        # Ensure the widget key is set to the imported value
        st.session_state["start_date"] = st.session_state.start_date
    
    # Sync end date
    if st.session_state.get('end_date') is not None:
        # Ensure the widget key is set to the imported value
        st.session_state["end_date"] = st.session_state.end_date
    
    # Sync custom dates checkbox
    has_custom_dates = (st.session_state.get('start_date') is not None or 
                       st.session_state.get('end_date') is not None)
    st.session_state["use_custom_dates"] = has_custom_dates
    st.session_state["use_custom_dates_checkbox"] = has_custom_dates

# --- Print start dates for all tickers and benchmark at script start ---
def print_ticker_start_dates(asset_tickers, benchmark_ticker):
    # All print statements removed for performance
    pass

# Example usage at script start (replace with your actual tickers)
if __name__ == "__main__":
    # You may want to load these from config or session state
    asset_tickers = ["AAPL", "MSFT"] # Example, replace with your tickers
    benchmark_ticker = "^GSPC" # Example, replace with your benchmark
    print_ticker_start_dates(asset_tickers, benchmark_ticker)

# ==============================================================================
# Helpers for timezone-safe indices
# ==============================================================================

def _ensure_naive_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Return a copy of obj with a tz-naive DatetimeIndex."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        return obj
    idx = obj.index
    if getattr(idx, "tz", None) is not None:
        obj = obj.copy()
        obj.index = idx.tz_convert(None)
    else:
        # already naive
        import plotly.graph_objects as go
        import logging
def check_currency_warning(tickers):
    """
    Check if any tickers are non-USD and display a warning.
    """
    non_usd_suffixes = ['.TO', '.V', '.CN', '.AX', '.L', '.PA', '.AS', '.SW', '.T', '.HK', '.KS', '.TW', '.JP']
    non_usd_tickers = []
    
    for ticker in tickers:
        if any(ticker.endswith(suffix) for suffix in non_usd_suffixes):
            non_usd_tickers.append(ticker)
    
    if non_usd_tickers:
        st.warning(f"⚠️ **Currency Warning**: The following tickers are not in USD: {', '.join(non_usd_tickers)}. "
                  f"Currency conversion is not taken into account, which may affect allocation accuracy. "
                  f"Consider using USD equivalents for more accurate results.")

def get_trading_days(start_date, end_date):
    """
    Retrieves trading days between two dates. For international stocks (like Canadian .TO),
    this uses business days to avoid calendar mismatches with different market schedules.
    """
    # Ensure start_date and end_date are datetime objects before processing
    if not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())

    # Use business days instead of NYSE calendar to handle international stocks properly
    # This prevents issues with Canadian stocks (.TO) that trade on different holidays
    # Note: pd.bdate_range excludes weekends but includes US holidays
    return pd.bdate_range(start=start_date.date(), end=end_date.date())

def get_risk_free_rate(dates):
    """Downloads the risk-free rate (IRX) and aligns it to a given date range."""
    try:
        dates = pd.to_datetime(dates)
        if isinstance(dates, pd.DatetimeIndex):
            if getattr(dates, "tz", None) is not None:
                dates = dates.tz_convert(None)
        start_date = dates.min().strftime('%Y-%m-%d')
        # add 1 day to be safe with yfinance's exclusive end
        end_date = (dates.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        # Try several Yahoo symbols for short-term Treasury yields in order of preference
        symbols = ["^IRX", "^FVX", "^TNX"]
        hist = None
        last_exception = None

        for s in symbols:
            # attempt 1: Ticker.history
            try:
                t = yf.Ticker(s)
                h = t.history(start=start_date, end=end_date, raise_errors=False)
                if h is None or getattr(h, 'empty', True):
                    # attempt 2: yf.download as fallback
                    h = yf.download(s, start=start_date, end=end_date, progress=False)
                if h is None or getattr(h, 'empty', True):
                    # try expanding the window by 30 days each side once
                    try:
                        adj_start = (pd.to_datetime(start_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
                        adj_end = (pd.to_datetime(end_date) + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
                        h = yf.download(s, start=adj_start, end=adj_end, progress=False)
                    except Exception:
                        h = None
                if h is None or getattr(h, 'empty', True):
                    last_exception = f"no rows for {s}"
                    continue
                if 'Close' not in h.columns:
                    last_exception = f"no Close column for {s}"
                    continue
                h = _ensure_naive_index(h)
                h = h[h['Close'].notnull() & (h['Close'] > 0)]
                if h.empty:
                    last_exception = f"empty Close values for {s}"
                    continue
                hist = h
                logger.info(f"Fetched treasury data using symbol {s}, rows={len(hist)}")
                break
            except Exception as e:
                last_exception = str(e)
                logger.debug(f"symbol {s} fetch error: {e}")
                hist = None
                continue

        if hist is None or hist.empty:
            logger.warning(f"IRX fetch attempts failed ({last_exception}) - falling back to default rate")
            raise ValueError("No valid treasury yield data found from Yahoo symbols")

        # 'Close' from these tickers is typically in percent (e.g. 2.34 -> 2.34%)
        annual_rate = hist['Close'] / 100.0

        # convert annual nominal to approximate daily rate (252 trading days)
        with np.errstate(over='ignore', invalid='ignore'):
            daily_rate = (1 + annual_rate) ** (1 / 252.0) - 1.0

        # Align directly to the requested dates (avoids tz/index mismatches)
        try:
            target_index = pd.to_datetime(dates)
            if getattr(target_index, 'tz', None) is not None:
                target_index = target_index.tz_convert(None)
        except Exception:
            target_index = pd.to_datetime(dates)

        daily_rate = daily_rate.reindex(daily_rate.index.union(target_index)).ffill()
        result = daily_rate.reindex(target_index).ffill().fillna(0)
        return result
    except Exception as e:
        logger.warning(f"IRX error: {str(e)} - Using default 2% annual rate")
        # Use 2% annual default if IRX fetch fails
        default_daily = (1 + 0.02) ** (1 / 252) - 1
        return pd.Series(default_daily, index=pd.to_datetime(dates))

def calculate_cagr(values, dates):
    """Calculates the Compound Annual Growth Rate."""
    if len(values) < 2 or values.iloc[0] <= 0:
        return np.nan
    start_val = values.iloc[0]
    end_val = values.iloc[-1]
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1

def calculate_max_drawdown(series):
    """Calculates the maximum drawdown from a series of values."""
    if series.empty:
        return 0
    peak = series.expanding(min_periods=1).max()
    drawdown = (series / peak) - 1
    return drawdown.min()

def calculate_sharpe(returns, risk_free_rate):
    """Calculates the Sharpe ratio."""
    aligned_returns, aligned_rf = returns.align(risk_free_rate, join='inner')
    if aligned_returns.empty:
        return np.nan
    
    excess_returns = aligned_returns - aligned_rf
    if excess_returns.std() == 0:
        return np.nan
        
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

def calculate_sortino(returns, risk_free_rate):
    """Calculates the Sortino ratio."""
    aligned_returns, aligned_rf = returns.align(risk_free_rate, join='inner')
    if aligned_returns.empty:
        return np.nan
        
    downside_returns = aligned_returns[aligned_returns < aligned_rf]
    if downside_returns.empty or downside_returns.std() == 0:
        # If no downside returns, Sortino is infinite or undefined.
        # We can return nan or a very high value. nan is safer.
        return np.nan
    
    downside_std = downside_returns.std()
    
    return (aligned_returns.mean() - aligned_rf.mean()) / downside_std * np.sqrt(252)

def calculate_ulcer_index(series):
    """Calculates the Ulcer Index (average squared percent drawdown, then sqrt)."""
    if series.empty:
        return np.nan
    peak = series.expanding(min_periods=1).max()
    drawdown = (series - peak) / peak * 100  # percent drawdown
    drawdown_sq = drawdown ** 2
    return np.sqrt(drawdown_sq.mean())

def calculate_upi(cagr, ulcer_index):
    """Calculates the Ulcer Performance Index (UPI = CAGR / Ulcer Index, both as decimals)."""
    if ulcer_index is None or np.isnan(ulcer_index) or ulcer_index == 0:
        return np.nan
    return cagr / (ulcer_index / 100)

def calculate_beta(portfolio_returns, benchmark_returns):
    """Calculates Beta."""
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 2:
        return np.nan
    
    pr = portfolio_returns.reindex(common_idx).dropna()
    br = benchmark_returns.reindex(common_idx).dropna()
    
    # Re-align after dropping NAs
    common_idx = pr.index.intersection(br.index)
    if len(common_idx) < 2 or br.loc[common_idx].var() == 0:
        return np.nan
    
    cov = pr.loc[common_idx].cov(br.loc[common_idx])
    var = br.loc[common_idx].var()
    
    return cov / var

def calculate_mwrr(portfolio_values: pd.Series, cash_flows: pd.Series):
    """
    Calculate Money-Weighted Rate of Return (MWRR) using a simple and robust method.
    
    MWRR is the IRR (Internal Rate of Return) of all cash flows including:
    - Initial investment (negative cash flow)
    - Any additional investments (negative cash flows)
    - Final portfolio value (positive cash flow)
    
    Args:
        portfolio_values: Series of portfolio values over time
        cash_flows: Series of cash flows (negative = investment, positive = withdrawal)
    
    Returns:
        Annual MWRR as a decimal (e.g., 0.08 for 8%)
    """
    try:
        import numpy as np
        import pandas as pd
        from scipy.optimize import brentq
        
        # Convert to numpy arrays for easier handling
        values = portfolio_values.dropna()
        flows = cash_flows.reindex(values.index, fill_value=0.0)
        
        if len(values) < 2:
            return np.nan
        
        # Get dates and calculate time periods in years
        dates = pd.to_datetime(values.index)
        start_date = dates[0]
        time_periods = np.array([(d - start_date).days / 365.25 for d in dates])
        
        # Prepare cash flows for IRR calculation
        # Initial investment is negative (outflow)
        initial_investment = -values.iloc[0]
        
        # Find all non-zero cash flows during the period
        significant_flows = flows[flows != 0]
        
        # Build cash flow array: [initial_investment, additional_flows..., final_value]
        cash_flow_dates = [start_date]
        cash_flow_amounts = [initial_investment]
        cash_flow_times = [0.0]
        
        # Add intermediate cash flows
        for date, flow in significant_flows.items():
            if date != start_date and date != dates[-1]:
                cash_flow_dates.append(date)
                cash_flow_amounts.append(flow)
                cash_flow_times.append((date - start_date).days / 365.25)
        
        # Add final value as positive cash flow (return of investment)
        cash_flow_dates.append(dates[-1])
        cash_flow_amounts.append(values.iloc[-1])
        cash_flow_times.append((dates[-1] - start_date).days / 365.25)
        
        cash_flow_amounts = np.array(cash_flow_amounts)
        cash_flow_times = np.array(cash_flow_times)
        
        # Define NPV function for IRR calculation
        def npv(rate):
            """Calculate Net Present Value for given rate"""
            if rate <= -1:  # Avoid division by zero or negative denominators
                return float('inf')
            
            npv_value = 0
            for cf, t in zip(cash_flow_amounts, cash_flow_times):
                npv_value += cf / ((1 + rate) ** t)
            return npv_value
        
        # Find IRR using Brent's method
        # Try to find rate where NPV = 0
        try:
            irr = brentq(npv, -0.99, 5.0, maxiter=100)
            
            # Validate result
            if abs(npv(irr)) < 1e-6:  # Check if NPV is close to zero
                return irr
            else:
                return np.nan
                
        except (ValueError, RuntimeError):
            # If Brent's method fails, try a simple approximation
            # Calculate simple annualized return as fallback
            try:
                total_time = cash_flow_times[-1]
                if total_time > 0:
                    total_invested = abs(initial_investment) + abs(significant_flows.sum())
                    final_value = values.iloc[-1]
                    if total_invested > 0:
                        simple_return = (final_value / total_invested) ** (1 / total_time) - 1
                        return simple_return if not np.isnan(simple_return) else np.nan
                return np.nan
            except:
                return np.nan
                
    except Exception as e:
        # Print removed for performance
        return np.nan

def run_stats(portfolio_values, benchmark_returns, cash_flows, final_beta, mwrr_portfolio_values=None, mwrr_cash_flows=None):
    """Calculates various backtesting statistics from a portfolio value series.
    
    Args:
        portfolio_values: Portfolio values for calculating most statistics (typically without cash additions)
        benchmark_returns: Benchmark returns for calculations
        cash_flows: Cash flows matching portfolio_values
        final_beta: Beta value
        mwrr_portfolio_values: Optional separate portfolio values for MWRR (typically with cash additions)
        mwrr_cash_flows: Optional separate cash flows for MWRR calculation
    """
    if len(portfolio_values) < 2:
        return {}

    portfolio_returns = portfolio_values.pct_change().fillna(0)
    
    risk_free_rates = get_risk_free_rate(portfolio_returns.index)
    
    cagr = calculate_cagr(portfolio_values, portfolio_values.index)
    max_dd = calculate_max_drawdown(portfolio_values)
    vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = calculate_sharpe(portfolio_returns, risk_free_rates)
    sortino = calculate_sortino(portfolio_returns, risk_free_rates)
    ulcer = calculate_ulcer_index(portfolio_values)
    upi = calculate_upi(cagr, ulcer)
    
    # Use separate data for MWRR if provided (typically with cash additions)
    if mwrr_portfolio_values is not None and mwrr_cash_flows is not None:
        mwrr = calculate_mwrr(mwrr_portfolio_values, mwrr_cash_flows)
    else:
        mwrr = calculate_mwrr(portfolio_values, cash_flows)
    
    # All print statements removed for performance
    
    stats = {
        "CAGR": cagr,
        "MaxDrawdown": max_dd,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "UlcerIndex": ulcer,
        "UPI": upi,
        "Beta": final_beta,
        "MWRR": mwrr,
    }
    
    return stats
    

def calculate_periodic_performance(portfolio_values: pd.Series, period: str = "Year"):
    """
    Calculates periodic returns and end-of-period values from a portfolio value series.
    period: "Year" or "Month"
    """
    if portfolio_values.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = portfolio_values.to_frame(name="Value")
    df['Period'] = df.index.to_period('ME').astype(str) if period == "Month" else df.index.year
    if period not in ["Year", "Month"]:
        raise ValueError("period must be 'Year' or 'Month'")
    summary = []
    start_value = df['Value'].iloc[0]
    for p, group in df.groupby('Period'):
        start_of_period_value = start_value if p == df['Period'].iloc[0] else group.iloc[0]['Value']
        end_of_period_value = group.iloc[-1]['Value']
        start_date = group.index[0].strftime('%Y-%m-%d')
        end_date = group.index[-1].strftime('%Y-%m-%d')
        variation = (end_of_period_value - start_of_period_value) / start_of_period_value if start_of_period_value != 0 else np.nan
        gain_loss = end_of_period_value - start_of_period_value if start_of_period_value != 0 else np.nan
        summary.append({
            "Period": p,
            "Start Date": start_date,
            "End Date": end_date,
            "Variation (%)": variation * 100,
            "Gain/Loss": gain_loss,
            "Portfolio Value": end_of_period_value
        })
        start_value = end_of_period_value
    summary_df = pd.DataFrame(summary)
    def color_gradient_stock(val):
        """Applies a color gradient based on performance percentage."""
        if isinstance(val, (int, float)) and not pd.isna(val):
            style = 'background-color: {}; color: white;'
            if val > 50:
                return style.format('#004d00')
            elif val > 20:
                return style.format('#1e8449')
            elif val > 5:
                return style.format('#388e3c')
            elif val > 0:
                return style.format('#66bb6a')
            elif val < -50:
                return style.format('#7b0000')
            elif val < -20:
                return style.format('#b22222')
            elif val < -5:
                return style.format('#d32f2f')
            elif val < 0:
                return style.format('#ef5350')
        return '' # Default, no style
    def color_gain_loss(row):
        variation = row["Variation (%)"]
        return color_gradient_stock(variation)
    styled_df = summary_df.style.format({
        "Variation (%)": "{:.2f}%",
        "Gain/Loss": "${:,.2f}",
        "Portfolio Value": "${:,.2f}"
    })
    return styled_df, summary_df

def create_rebalance_tables(rebalance_metrics_list: List[Dict], all_tickers: List[str]):
    """
    Generates two tables: one for rebalancing allocations and one for metrics.
    """
    if not rebalance_metrics_list:
        return None, None

    # Table 1: Allocations at each rebalance
    allocations_data = []
    
    # Table 2: Metrics at each rebalance
    metrics_data = []
    
    # Process all data first to prepare the dataframes
    for rebalance in rebalance_metrics_list:
        date_str = rebalance['date'].strftime('%Y-%m-%d')
        
        allocations_row = {"Date": date_str}
        
        # Prepare allocations row
        for t in all_tickers:
            allocations_row[t] = rebalance['target_allocation'].get(t, 0.0)
        allocations_row["CASH"] = rebalance['target_allocation'].get("CASH", 0.0)
        
        allocations_data.append(allocations_row)
        
        # Prepare metrics rows
        for t in all_tickers:
            metrics_row = {
                "Date": date_str,
                "Ticker": t,
                "Momentum (%)": (rebalance['momentum_scores'].get(t, np.nan) or np.nan) * 100,
                "Volatility (%)": (rebalance['volatility'].get(t, np.nan) or np.nan) * 100,
                "Beta": rebalance['beta'].get(t, np.nan),
                "Allocation (%)": rebalance['target_allocation'].get(t, 0.0) * 100
            }
            metrics_data.append(metrics_row)
        # Add a CASH line if allocation to cash is nonzero
        cash_alloc = rebalance['target_allocation'].get("CASH", 0.0)
        if cash_alloc > 0:
            metrics_row = {
                "Date": date_str,
                "Ticker": "CASH",
                "Momentum (%)": np.nan,
                "Volatility (%)": np.nan,
                "Beta": np.nan,
                "Allocation (%)": cash_alloc * 100
            }
            metrics_data.append(metrics_row)
            
    allocations_df = pd.DataFrame(allocations_data)
    metrics_df = pd.DataFrame(metrics_data)

    # Styling with dark theme backgrounds
    color_even = '#0e1117' # Darker
    color_odd = '#262626'  # Lighter gray

    def highlight_rows_by_index(s):
        is_even_row = allocations_df.index.get_loc(s.name) % 2 == 0
        bg_color = color_even if is_even_row else color_odd
        return [f'background-color: {bg_color}; color: white;' for c in s]

    def percent_fmt(val):
        # Display whole-number percentages (e.g. 0.2 -> "20%")
        return f"{val*100:.0f}%" if pd.notna(val) else "N/A"

    styled_allocations = allocations_df.style.apply(highlight_rows_by_index, axis=1) \
        .format({t: percent_fmt for t in all_tickers + ["CASH"]}) \
        .format({"Date": "{}"})
    
    def highlight_metrics_rows(df):
        colors = []
        is_even_date_block = True
        current_date = None
        for idx, row in df.iterrows():
            date_val = row['Date']
            ticker_val = row['Ticker']
            if date_val != current_date:
                is_even_date_block = not is_even_date_block
                current_date = date_val
            # Distinct color for CASH line
            if ticker_val == "CASH":
                color = 'rgba(0,77,0,0.5)' # dark transparent green
            else:
                color = color_even if is_even_date_block else color_odd
            colors.append(color)

        styler_df = pd.DataFrame('', index=df.index, columns=df.columns)
        for col in df.columns:
            styler_df[col] = [f'background-color: {c}; color: white;' for c in colors]
        return styler_df

    def color_momentum(val):
        if isinstance(val, (int, float)) and not pd.isna(val):
            style = 'background-color: {};'
            if val > 50:
                return style.format('#004d00')
            elif val > 20:
                return style.format('#1e8449')
            elif val > 5:
                return style.format('#388e3c')
            elif val > 0:
                return style.format('#66bb6a')
            elif val < -50:
                return style.format('#7b0000')
            elif val < -20:
                return style.format('#b22222')
            elif val < -5:
                return style.format('#d32f2f')
            elif val < 0:
                return style.format('#ef5350')
        return ''

    # Format metrics styler with Allocation shown as whole-number percent
    styled_metrics = metrics_df.style.apply(highlight_metrics_rows, axis=None) \
                                     .map(color_momentum, subset=["Momentum (%)"]) \
                                     .format({"Momentum (%)": "{:.2f}%", 
                                              "Volatility (%)": "{:.2f}%", 
                                              "Beta": "{:.2f}",
                                              "Allocation (%)": "{:.0f}%"}, na_rep="N/A")

    # Create a guaranteed display DataFrame for allocations where values are strings like "20%"
    display_allocations_df = allocations_df.copy()
    for col in [c for c in display_allocations_df.columns if c != "Date"]:
        display_allocations_df[col] = display_allocations_df[col].apply(lambda v: f"{v*100:.0f}%" if pd.notna(v) else "N/A")

    return display_allocations_df, styled_metrics


# --- Main Backtest Logic ---

def _load_data(tickers: List[str], start_date: datetime, end_date: datetime):
    """
    Loads and preprocesses data for a list of tickers, handling incomplete data
    for the specified end date.
    """
    data = {}
    available_tickers = []
    invalid_tickers = []

    # Default to large range if missing
    if start_date is None:
        start_date = datetime(1900, 1, 1)
    if end_date is None:
        end_date = datetime.now()

    # Ensure datetime without tz
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())

    start_date = pd.Timestamp(start_date).tz_localize(None)
    end_date = pd.Timestamp(end_date).tz_localize(None)

    # yfinance end date is exclusive → add 1 day to include today
    yf_end_date = end_date + timedelta(days=1)


    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(start=start_date, end=yf_end_date, auto_adjust=False)

            if hist.empty:
                logger.warning(f"No data available for {t}")
                invalid_tickers.append(t)
                continue

            # Force tz-naive for hist
            hist = hist.copy()
            hist.index = hist.index.tz_localize(None)

            # Clip to end_date in case yf_end_date brought extra rows
            hist = hist[hist.index <= end_date]

            # Fetch dividends
            divs = ticker.dividends.copy()
            if getattr(divs.index, "tz", None) is not None:
                divs.index = divs.index.tz_convert(None)
            # Removed dividend debug prints for better performance

            # Map dividend payments to the next available trading day if not present
            divs_mapped = pd.Series(0.0, index=hist.index)
            for dt, val in divs.items():
                # Find the next available trading day in hist.index
                next_idx = hist.index[hist.index >= dt]
                if len(next_idx) > 0:
                    pay_date = next_idx[0]
                    divs_mapped[pay_date] += val
            hist["Dividend_per_share"] = divs_mapped

            # Price change calculation
            hist["Price_change"] = hist["Close"].pct_change(fill_method=None)

            # Drop rows with missing close prices
            data[t] = hist.dropna(subset=["Close"])
            available_tickers.append(t)
            # Removed data loading print for better performance

        except Exception as e:
            logger.error(f"Error for {t}: {e}")
            invalid_tickers.append(t)
            continue

    if not available_tickers:
        raise ValueError("No assets available with data.")

    return data, available_tickers, invalid_tickers


def _prepare_backtest_dates(start_date_user: date | None, end_date_user: date | None, start_with: str, data: Dict, portfolio_tickers: List[str], momentum_windows: List[Dict] | None = None):
    """Determines the final backtest date range and trading calendar based ONLY on portfolio tickers.

    If `start_with == 'all'` and `momentum_windows` is provided, the backtest start
    will be set to the latest ticker start date plus the largest momentum window
    (lookback + exclude). This ensures a full warm-up period so all tickers have
    the required history before the first allocation.
    """
    start_date_user_dt = datetime.combine(start_date_user, datetime.min.time()) if isinstance(start_date_user, date) else None
    end_date_user_dt = datetime.combine(end_date_user, datetime.min.time()) if isinstance(end_date_user, date) else None

    # Get the earliest and latest available data dates across ONLY the portfolio tickers
    idx_starts = [d.index[0] for t, d in data.items() if t in portfolio_tickers and not d.empty]
    idx_ends = [d.index[-1] for t, d in data.items() if t in portfolio_tickers and not d.empty]
    if not idx_starts or not idx_ends:
        raise ValueError("No assets available with data in the specified range.")

    oldest_data_start = min(idx_starts)
    latest_data_start = max(idx_starts)
    latest_data_end = min(idx_ends)

    # Base start: either oldest or latest depending on choice
    base_start = oldest_data_start if start_with == "oldest" else latest_data_start

    # If user provided a custom start date and it's after base_start, prefer it
    if start_date_user_dt and start_date_user_dt > base_start:
        base_start = start_date_user_dt

    # Handle first rebalance strategy
    first_rebalance_strategy = st.session_state.get('first_rebalance_strategy', 'rebalancing_date')
    
    # If user selected 'all' and momentum windows provided, check first rebalance strategy
    if start_with == "all" and momentum_windows:
        if first_rebalance_strategy == "momentum_window_complete":
            # Wait for momentum window to complete before first rebalance
            try:
                # largest window measured as lookback only (exclude is handled within the calculation)
                window_sizes = [int(w.get('lookback', 0)) for w in momentum_windows if w is not None]
                max_window_days = max(window_sizes) if window_sizes else 0
            except Exception:
                max_window_days = 0
            # Add momentum window delay so first rebalance happens when window is complete
            tentative_start = base_start + pd.Timedelta(days=max_window_days)
            backtest_start = tentative_start
        else:  # first_rebalance_strategy == "rebalancing_date"
            # Start immediately - momentum calculation will handle missing data
            backtest_start = base_start
    else:
        backtest_start = base_start

    backtest_end = latest_data_end
    if end_date_user_dt and end_date_user_dt < latest_data_end:
        backtest_end = end_date_user_dt

    if backtest_start >= backtest_end:
        raise ValueError("Start date must be before end date based on data availability and your settings.")

    # Build trading calendar from actual data instead of calendar to handle international stocks
    # This ensures we only use dates when ALL stocks actually have data (intersection, not union)
    all_trading_days = None
    for t, d in data.items():
        if t in portfolio_tickers and not d.empty:
            # Get trading days from this stock's data within our range
            stock_dates = set(d.index[(d.index >= backtest_start) & (d.index <= backtest_end)])
            if all_trading_days is None:
                all_trading_days = stock_dates
            else:
                # Use intersection to ensure ALL stocks have data on each date
                all_trading_days = all_trading_days.intersection(stock_dates)
    
    if all_trading_days is None or len(all_trading_days) == 0:
        raise ValueError("No overlapping trading days found for the selected date range.")
    
    all_dates = pd.DatetimeIndex(sorted(all_trading_days))
    # Ensure we start no earlier than backtest_start and not after end
    all_dates = all_dates[(all_dates >= backtest_start) & (all_dates <= backtest_end)]

    if len(all_dates) == 0:
        raise ValueError("No overlapping trading days found for the selected date range.")

    # final backtest_start is the first trading day in the calendar
    final_backtest_start = all_dates[0]

    return all_dates, final_backtest_start, backtest_end

def get_event_dates(trading_days, frequency):
    """Return event dates for a given frequency, mapped to previous available trading day."""
    if frequency == "Never":
        return pd.DatetimeIndex([])
    elif frequency == "Weekly":
        # First available trading day of each week
        weeks = trading_days.to_series().groupby([trading_days.year, trading_days.isocalendar().week]).first()
        return pd.DatetimeIndex(weeks.values)
    elif frequency == "Biweekly":
        weeks = trading_days.to_series().groupby([trading_days.year, trading_days.isocalendar().week]).first()
        return pd.DatetimeIndex(weeks.values[::2])
    elif frequency == "Monthly":
        # First available trading day of each month
        months = trading_days.to_series().groupby([trading_days.year, trading_days.month]).first()
        return pd.DatetimeIndex(months.values)
    elif frequency == "Quarterly":
        # First available trading day of each quarter
        quarters = trading_days.to_series().groupby([trading_days.year, trading_days.quarter]).first()
        return pd.DatetimeIndex(quarters.values)
    elif frequency == "Semiannually":
        # First available trading day of Jan and Jul each year
        semi = []
        for y in sorted(set(trading_days.year)):
            for m in [1, 7]:
                days = trading_days[(trading_days.year == y) & (trading_days.month == m)]
                if len(days) > 0:
                    semi.append(days[0])
        return pd.DatetimeIndex(semi)
    elif frequency == "Annually":
        # First available trading day of each year
        years = trading_days.to_series().groupby(trading_days.year).first()
        return pd.DatetimeIndex(years.values)
    else:
        return pd.DatetimeIndex([])

def map_to_prev_trading_day(dates, trading_days):
    """
    Maps each scheduled event date to the closest previous available trading day in trading_days.
    Ensures all mapped dates are present in trading_days and returns unique, sorted dates.
    """
    mapped = []
    for dt in dates:
        # Find the closest previous trading day
        prev_days = trading_days[trading_days <= dt]
        if len(prev_days) > 0:
            mapped.append(prev_days[-1])
        # If no previous trading day exists, skip (should not happen in normal use)
    return pd.DatetimeIndex(sorted(set(mapped)))

def _get_rebalancing_dates(all_dates: pd.DatetimeIndex, rebalancing_frequency: str):
    """Generates a list of rebalancing dates based on frequency."""
    if rebalancing_frequency == "Never":
        return pd.DatetimeIndex([])
    elif rebalancing_frequency == "Weekly":
        # Mondays
        return all_dates[all_dates.weekday == 0]
    elif rebalancing_frequency == "Biweekly":
        mondays = all_dates[all_dates.weekday == 0]
        return mondays[::2]
    elif rebalancing_frequency == "Monthly":
        return all_dates[all_dates.is_month_end]
    elif rebalancing_frequency == "Quarterly":
        return all_dates[all_dates.is_quarter_end]
    elif rebalancing_frequency == "Semiannually":
        # First trading day of Jan/Jul each year
        semi = [(y, m) for y in sorted(set(all_dates.year)) for m in [1, 7]]
        return pd.DatetimeIndex([all_dates[(all_dates.year == y) & (all_dates.month == m)][0] for y, m in semi if any((all_dates.year == y) & (all_dates.month == m))])
    elif rebalancing_frequency == "Annually":
        return all_dates[all_dates.is_year_end]
    elif rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"]:
        # Buy & Hold options don't have specific rebalancing dates - they rebalance immediately when cash is available
        return pd.DatetimeIndex([])
    else:
        return pd.DatetimeIndex([])

def _get_added_cash_dates(all_dates: pd.DatetimeIndex, added_frequency: str):
    """Generates a list of dates for periodic cash additions."""
    if added_frequency == "Never":
        return pd.DatetimeIndex([])

    if added_frequency == "Monthly":
        return all_dates[all_dates.is_month_end]
    elif added_frequency == "Quarterly":
        return all_dates[all_dates.is_quarter_end]
    elif added_frequency == "Annually":
        return all_dates[all_dates.is_year_end]
    else:
        return pd.DatetimeIndex([])

# =============================
# NEW MOMENTUM LOGIC
# =============================

def calculate_momentum(date, current_assets, momentum_windows, data_dict):
    cumulative_returns, valid_assets = {}, []
    filtered_windows = [w for w in momentum_windows if w["weight"] > 0]
    # Normalize weights so they sum to 1
    total_weight = sum(w["weight"] for w in filtered_windows)
    if total_weight == 0:
        normalized_weights = [0 for _ in filtered_windows]
    else:
        normalized_weights = [w["weight"] / total_weight for w in filtered_windows]
    
    # Bulletproof start dates calculation
    start_dates_config = {}
    for t in current_assets:
        if t in data_dict and not data_dict[t].empty:
            try:
                start_dates_config[t] = data_dict[t].first_valid_index()
            except:
                start_dates_config[t] = pd.Timestamp.max
    
    for t in current_assets:
        # Skip if ticker not in data
        if t not in data_dict or data_dict[t].empty:
            continue
            
        is_valid, asset_returns = True, 0.0
        for idx, window in enumerate(filtered_windows):
            lookback, exclude = window["lookback"], window["exclude"]
            weight = normalized_weights[idx]
            start_mom = date - pd.Timedelta(days=lookback)
            end_mom = date - pd.Timedelta(days=exclude)
            
            if start_dates_config.get(t, pd.Timestamp.max) > start_mom:
                is_valid = False
                break
                
            df_t = data_dict[t]
            
            # Bulletproof date access
            try:
                price_start_index = df_t.index.asof(start_mom)
                price_end_index = df_t.index.asof(end_mom)
                
                if pd.isna(price_start_index) or pd.isna(price_end_index):
                    is_valid = False
                    break
                    
                # Ensure indices exist in the dataframe
                if price_start_index not in df_t.index or price_end_index not in df_t.index:
                    is_valid = False
                    break
                    
                price_start = df_t.loc[price_start_index, "Close"]
                price_end = df_t.loc[price_end_index, "Close"]
                
                if pd.isna(price_start) or pd.isna(price_end) or price_start == 0:
                    is_valid = False
                    break
                    
                ret = (price_end - price_start) / price_start
                asset_returns += ret * weight
                
            except Exception:
                is_valid = False
                break
                
        if is_valid:
            cumulative_returns[t] = asset_returns
            valid_assets.append(t)
    return cumulative_returns, valid_assets


def calculate_momentum_weights(returns, valid_assets, date, negative_momentum_strategy):
    """
    Calculates weights based on momentum, with optional filtering for volatility and beta.
    """
    import pandas as pd
    import numpy as np

    if not valid_assets:
        return {}, {}
    rets = {t: returns[t] for t in valid_assets if not pd.isna(returns[t])}
    if not rets:
        return {}, {}
    beta_vals, vol_vals = {}, {}
    metrics = {t: {} for t in data.keys()}
    if calc_beta or calc_volatility:
        df_bench = data.get(benchmark_ticker)
        if calc_beta:
            start_beta = date - pd.Timedelta(days=beta_window_days)
            end_beta = date - pd.Timedelta(days=exclude_days_beta)
        if calc_volatility:
            start_vol = date - pd.Timedelta(days=vol_window_days)
            end_vol = date - pd.Timedelta(days=exclude_days_vol)
        for t in valid_assets:
            df_t = data[t]
            if calc_beta and df_bench is not None:
                mask_beta = (df_t.index >= start_beta) & (df_t.index <= end_beta)
                returns_t_beta = df_t.loc[mask_beta, "Price_change"]
                mask_bench_beta = (df_bench.index >= start_beta) & (df_bench.index <= end_beta)
                returns_bench_beta = df_bench.loc[mask_bench_beta, "Price_change"]
                if len(returns_t_beta) < 2 or len(returns_bench_beta) < 2:
                    beta_vals[t] = np.nan
                else:
                    # Align returns before calculating covariance
                    aligned_t, aligned_bench = returns_t_beta.align(returns_bench_beta, join='inner')
                    if len(aligned_t) < 2:
                        beta_vals[t] = np.nan
                    else:
                        covariance = np.cov(aligned_t, aligned_bench)[0, 1]
                        variance = np.var(aligned_bench)
                        if variance > 0:
                            beta_vals[t] = covariance / variance
                        else:
                            beta_vals[t] = np.nan
                metrics[t]['Beta'] = beta_vals.get(t, np.nan)
            if calc_volatility:
                mask_vol = (df_t.index >= start_vol) & (df_t.index <= end_vol)
                returns_t_vol = df_t.loc[mask_vol, "Price_change"]
                if len(returns_t_vol) < 2:
                    vol_vals[t] = np.nan
                else:
                    vol_vals[t] = returns_t_vol.std() * np.sqrt(252)
                metrics[t]['Volatility'] = vol_vals.get(t, np.nan)

    for t in rets:
        metrics[t]['Momentum'] = rets[t]

    weights = {}
    if not rets:
        return {}, {}

    all_negative = all(r <= 0 for r in rets.values())

    if all_negative:
        if negative_momentum_strategy == "Go to cash":
            weights = {t: 0 for t in rets}
            for t in metrics:
                metrics[t]['Calculated_Weight'] = 0
            return weights, metrics
        elif negative_momentum_strategy == "Equal weight":
            weights = {t: 1 / len(rets) for t in rets}
        elif negative_momentum_strategy == "Relative momentum":
            min_score = min(rets.values())
            offset = -min_score + 0.01
            shifted = {t: max(0.01, rets[t] + offset) for t in rets}
            sum_shifted = sum(shifted.values())
            weights = {t: shifted[t] / sum_shifted for t in shifted}
    else:
        if use_relative_momentum:
            min_score = min(rets.values())
            offset = -min_score + 0.01 if min_score < 0 else 0.01
            shifted = {t: max(0.01, rets[t] + offset) for t in rets}
            sum_shifted = sum(shifted.values())
            weights = {t: shifted[t] / sum_shifted for t in shifted}
        else: # Absolute momentum
            positive_scores = {t: s for t, s in rets.items() if s > 0}
            if positive_scores:
                sum_positive = sum(positive_scores.values())
                weights = {t: positive_scores[t] / sum_positive for t in positive_scores}
                for t in [t for t in rets if rets[t] <= 0]:
                    weights[t] = 0
            else:
                weights = {t: 0 for t in rets}

    # ==============================================================================
    # THE CORRECTED CODE SNIPPET STARTS HERE
    # ==============================================================================
    fallback_mode = all_negative and negative_momentum_strategy == "Equal weight"
    if weights and (calc_volatility or calc_beta) and not fallback_mode:
        filtered_weights = {}
        for t, weight in weights.items():
            if weight > 0:
                score = 1.0
                if calc_volatility and t in vol_vals and not np.isnan(vol_vals[t]):
                    # Inverse relationship: lower volatility -> higher score
                    score *= (1 / vol_vals[t]) if vol_vals[t] > 0 else 0
                if calc_beta and t in beta_vals and not np.isnan(beta_vals[t]):
                    # Penalize by absolute beta, but never exclude for negative beta
                    abs_beta = abs(beta_vals[t])
                    score *= (1 / abs_beta) if abs_beta > 0 else 1.0
                filtered_weights[t] = weight * score
        # Re-normalize weights
        total_filtered_weight = sum(filtered_weights.values())
        if total_filtered_weight > 0:
            weights = {t: w / total_filtered_weight for t, w in filtered_weights.items()}
        else:
            weights = {} # Go to cash if all filters eliminate assets
    # ==============================================================================
    # THE CORRECTED CODE SNIPPET ENDS HERE
    # ==============================================================================

    for t in metrics:
        metrics[t]['Calculated_Weight'] = weights.get(t, 0)

    return weights, metrics

def _rebalance_portfolio(
    current_date,
    total_portfolio_value,
    data_dict,
    tradable_tickers_today,
    use_momentum,
    momentum_windows,
    negative_momentum_strategy,
    use_relative_momentum_flag,
    allocations,
    use_beta_flag,
    beta_window_days_val,
    exclude_days_beta_val,
    benchmark_ticker_val,
    use_volatility_flag,
    vol_window_days_val,
    exclude_days_vol_val,
):
    """Rebalancing now uses the new, more robust momentum logic."""
    global data, calc_beta, calc_volatility, beta_window_days, exclude_days_beta, benchmark_ticker, vol_window_days, exclude_days_vol, use_relative_momentum
    data = data_dict
    calc_beta = use_beta_flag
    calc_volatility = use_volatility_flag
    beta_window_days = beta_window_days_val
    exclude_days_beta = exclude_days_beta_val
    benchmark_ticker = benchmark_ticker_val
    vol_window_days = vol_window_days_val
    exclude_days_vol = exclude_days_vol_val

    
    # The "Relative momentum" option for negative scores implies use_relative_momentum logic
    use_relative_momentum = use_relative_momentum_flag or (negative_momentum_strategy == "Relative momentum")

    target_allocation = {}
    rebalance_metrics = {
        "date": current_date,
        "momentum_scores": {},
        "volatility": {},
        "beta": {},
        "target_allocation": {},
    }

    if use_momentum:
        returns, valid_assets = calculate_momentum(current_date, set(tradable_tickers_today), momentum_windows, data)
        weights, metrics = calculate_momentum_weights(returns, valid_assets, date=current_date, negative_momentum_strategy=negative_momentum_strategy)
        
        rebalance_metrics["momentum_scores"] = {t: metrics.get(t, {}).get("Momentum", None) for t in tradable_tickers_today}
        rebalance_metrics["volatility"] = {t: metrics.get(t, {}).get("Volatility", None) for t in tradable_tickers_today}
        rebalance_metrics["beta"] = {t: metrics.get(t, {}).get("Beta", None) for t in tradable_tickers_today}

        if weights:
            # Normalize weights to ensure they sum to 1, handling potential float precision issues
            total_weight = sum(weights.values())
            if total_weight > 0:
                normalized_weights = {t: w / total_weight for t, w in weights.items()}
            else:
                normalized_weights = {}

            for t in tradable_tickers_today:
                target_allocation[t] = total_portfolio_value * normalized_weights.get(t, 0)
            
            total_val = sum(target_allocation.values())
            if total_val > 0:
                rebalance_metrics["target_allocation"] = {t: v / total_val for t, v in target_allocation.items()}
            else: # Go to cash if no positive weights
                 rebalance_metrics["target_allocation"]["CASH"] = 1.0
        else : # Go to cash if no valid assets or weights calculated
            target_allocation = {}
            rebalance_metrics["target_allocation"]["CASH"] = 1.0
            
        return target_allocation, rebalance_metrics

    else: # Rebalance to initial fixed allocations
        rebalance_metrics["target_allocation"]["CASH"] = 0.0
        # Renormalize only across tickers that are tradable today. This
        # ensures that when starting with 'oldest' the entire portfolio is
        # invested across available assets (e.g. first asset gets 100%, then
        # 50/50 when second appears, etc.). Allocations provided in the
        # `allocations` dict are assumed to be percentages (sum to 100).
        alloc_sum_available = sum(allocations.get(t, 0.0) for t in tradable_tickers_today)
        # Fallback to total allocation sum if nothing is available (shouldn't happen)
        if alloc_sum_available <= 0:
            alloc_sum_available = sum(allocations.values()) if sum(allocations.values()) > 0 else 1.0

        for t in tradable_tickers_today:
            normalized_alloc = allocations.get(t, 0.0) / alloc_sum_available
            target_allocation[t] = total_portfolio_value * normalized_alloc
            rebalance_metrics["target_allocation"][t] = normalized_alloc
        return target_allocation, rebalance_metrics

# =============================
# END OF NEW MOMENTUM LOGIC
# =============================


def run_backtest(
    tickers: List[str],
    benchmark_ticker: str,
    allocations: Dict[str, float],
    include_dividends: Dict[str, bool],
    initial_value: float,
    added_amount: float,
    added_frequency: str,
    rebalancing_frequency: str,
    start_date_user: date,
    end_date_user: date,
    start_with: str,
    use_momentum: bool,
    momentum_windows: List[Dict],
    initial_allocation_option: str,
    negative_momentum_strategy: str,
    use_relative_momentum: bool,
    calc_beta: bool,
    calc_volatility: bool,
    beta_window_days: int,
    exclude_days_beta: int,
    vol_window_days: int,
    exclude_days_vol: int,
):
    """
    Runs a momentum-based backtest on a portfolio of assets.
    """
    
    # Deduplicate tickers list first
    tickers = list(dict.fromkeys(tickers))
    
    # Deduplicate allocations by summing values for duplicate tickers
    deduplicated_allocations = {}
    for ticker, allocation in allocations.items():
        if ticker in deduplicated_allocations:
            deduplicated_allocations[ticker] += allocation
        else:
            deduplicated_allocations[ticker] = allocation
    
    # Deduplicate include_dividends by using True if any instance has it True
    deduplicated_include_dividends = {}
    for ticker, include_div in include_dividends.items():
        if ticker in deduplicated_include_dividends:
            deduplicated_include_dividends[ticker] = deduplicated_include_dividends[ticker] or include_div
        else:
            deduplicated_include_dividends[ticker] = include_div
    
    # Update the parameters to use deduplicated values
    allocations = deduplicated_allocations
    include_dividends = deduplicated_include_dividends
    
    # Convert user dates (may be None) -> datetime
    start_dt = datetime.combine(start_date_user, datetime.min.time()) if isinstance(start_date_user, date) else None
    end_dt = datetime.combine(end_date_user, datetime.min.time()) if isinstance(end_date_user, date) else None
    
    # Removed "Downloading data..." print for better performance
    # Tickers to download: portfolio + benchmark
    all_tickers_to_fetch = list(set(tickers + [benchmark_ticker] if benchmark_ticker else tickers))
    
    # BULLETPROOF VALIDATION: Wrap _load_data in try-catch to prevent crashes
    try:
        data, available_tickers, invalid_tickers = _load_data(all_tickers_to_fetch, start_dt, end_dt)
    except ValueError as e:
        # Handle the case where no assets are available
        raise ValueError("❌ **No valid tickers found!** No data could be downloaded for any of the specified tickers. Please check your ticker symbols and try again.")
    except Exception as e:
        # Handle any other unexpected errors
        raise ValueError(f"❌ **Error downloading data:** {str(e)}. Please check your ticker symbols and try again.")
    
    # Display invalid ticker warnings in Streamlit UI
    if invalid_tickers:
        # Separate portfolio tickers from benchmark ticker
        portfolio_invalid = [t for t in invalid_tickers if t in tickers]
        benchmark_invalid = [t for t in invalid_tickers if t == benchmark_ticker]
        
        if portfolio_invalid:
            st.warning(f"The following portfolio tickers are invalid and will be skipped: {', '.join(portfolio_invalid)}")
        if benchmark_invalid:
            st.warning(f"The benchmark ticker '{benchmark_ticker}' is invalid and will be skipped.")
    
    # BULLETPROOF VALIDATION: Check for valid tickers and raise exceptions if none
    if not tickers:
        raise ValueError("❌ **No valid tickers found!** No tickers were provided. Please add at least one ticker before running the backtest.")
    
    if not available_tickers:
        if invalid_tickers and len(invalid_tickers) == len(all_tickers_to_fetch):
            raise ValueError(f"❌ **No valid tickers found!** All tickers are invalid: {', '.join(invalid_tickers)}. Please check your ticker symbols and try again.")
        else:
            raise ValueError("❌ **No valid tickers found!** No data could be downloaded for any of the specified tickers. Please check your ticker symbols and try again.")
    
    # Filter to only valid tickers that exist in data
    tickers_with_data = [t for t in tickers if t in data]
    # Ensure tickers_with_data is also deduplicated
    tickers_with_data = list(dict.fromkeys(tickers_with_data))
    
    if not tickers_with_data:
        if invalid_tickers:
            raise ValueError(f"❌ **No valid tickers found!** None of your selected assets have data available. Invalid tickers: {', '.join(invalid_tickers)}")
        else:
            raise ValueError("❌ **No valid tickers found!** None of your selected assets have data available.")
    
    # Check if benchmark ticker is valid, if not, set it to None
    if benchmark_ticker and benchmark_ticker not in data:
        benchmark_ticker = None
        
    # Check for non-USD tickers and display currency warning
    check_currency_warning(tickers_with_data)
    
    # BULLETPROOF VALIDATION: Wrap _prepare_backtest_dates in try-catch to prevent crashes
    try:
        all_dates, backtest_start, backtest_end = _prepare_backtest_dates(
            start_date_user, end_date_user, start_with, data, tickers_with_data,
            momentum_windows=momentum_windows if use_momentum else None
        )
    except ValueError as e:
        raise ValueError(f"❌ **Date range error:** {str(e)}. Please check your date settings and try again.")
    except Exception as e:
        raise ValueError(f"❌ **Error preparing backtest dates:** {str(e)}. Please check your settings and try again.")
    # Removed backtest date range print for better performance

    # Get rebalancing dates ONLY from rebalancing frequency
    raw_rebalancing_dates = get_event_dates(all_dates, rebalancing_frequency)
    mapped_rebalancing_dates = map_to_prev_trading_day(raw_rebalancing_dates, all_dates)
    
    # Handle first rebalance strategy - replace first rebalance date if needed
    first_rebalance_strategy = st.session_state.get('first_rebalance_strategy', 'rebalancing_date')
    if first_rebalance_strategy == "momentum_window_complete" and use_momentum and momentum_windows:
        try:
            # Calculate when momentum window completes
            window_sizes = [int(w.get('lookback', 0)) for w in momentum_windows if w is not None]
            max_window_days = max(window_sizes) if window_sizes else 0
            momentum_completion_date = all_dates[0] + pd.Timedelta(days=max_window_days)
            
            # Find the closest trading day to momentum completion
            momentum_completion_trading_day = all_dates[all_dates >= momentum_completion_date][0] if len(all_dates[all_dates >= momentum_completion_date]) > 0 else all_dates[-1]
            
            # Replace the first rebalancing date with momentum completion date
            if len(mapped_rebalancing_dates) > 0:
                # Remove the first rebalancing date and add momentum completion date
                mapped_rebalancing_dates = mapped_rebalancing_dates[1:] if len(mapped_rebalancing_dates) > 1 else pd.DatetimeIndex([])
                mapped_rebalancing_dates = mapped_rebalancing_dates.insert(0, momentum_completion_trading_day)
        except Exception:
            pass  # Fall back to regular rebalancing dates
    
    # Always include the first date for initial investment
    first_date = all_dates[0]
    if rebalancing_frequency != "Never":
        if first_date not in mapped_rebalancing_dates:
            mapped_rebalancing_dates = mapped_rebalancing_dates.insert(0, first_date)
    # Remove duplicates and sort
    rebalancing_dates = pd.DatetimeIndex(sorted(set(mapped_rebalancing_dates)))

    # Get added cash dates ONLY from addition frequency
    raw_added_cash_dates = get_event_dates(all_dates, added_frequency)
    added_cash_dates = map_to_prev_trading_day(raw_added_cash_dates, all_dates)
    added_cash_dates = pd.DatetimeIndex(sorted(set(added_cash_dates)))
    # Do NOT merge or insert cash addition dates into rebalancing_dates

    portfolio_value_with_additions = pd.Series(0.0, index=all_dates)
    portfolio_value_without_additions = pd.Series(0.0, index=all_dates)
    
    # --- New variables to explicitly track cash and shares ---
    asset_shares_with_additions = pd.Series(0.0, index=tickers_with_data)
    cash_with_additions = initial_value
    
    asset_shares_without_additions = pd.Series(0.0, index=tickers_with_data)
    cash_without_additions = initial_value
    
    cash_flows_with_additions = pd.Series(0.0, index=all_dates)
    cash_flows_without_additions = pd.Series(0.0, index=all_dates)
    
    last_rebalance_allocations = {}
    current_allocations = {}

    # Store rebalance metrics and allocations for the new tables
    rebalance_metrics_list = []
    
    # Removed "Starting simulation..." print for better performance
    
    # Add initial cash flows
    # Always invest initial value on first date
    if len(all_dates) > 0:
        first_date = all_dates[0]
        cash_flows_with_additions.loc[first_date] = -initial_value
        cash_flows_without_additions.loc[first_date] = -initial_value
        if first_date not in rebalancing_dates:
            rebalancing_dates = rebalancing_dates.insert(0, first_date).drop_duplicates().sort_values()

    progress_bar = None
    progress_text = None
    start_time = time.time()
    total_steps = len(all_dates)
    if 'progress_bar' in st.session_state:
        del st.session_state['progress_bar']
    if 'progress_text' in st.session_state:
        del st.session_state['progress_text']
    progress_bar = st.progress(0, text="Starting backtest...")
    progress_text = st.empty()
    for i, current_date in enumerate(all_dates):
        
        # Only tickers that have price today - use more robust data access
        tradable_tickers_today = []
        for t in tickers_with_data:
            ticker_data = data.get(t, pd.DataFrame())
            if not ticker_data.empty and current_date in ticker_data.index:
                # Additional check: ensure we have valid price data
                try:
                    price = ticker_data.loc[current_date, "Close"]
                    if not pd.isna(price) and price > 0:
                        tradable_tickers_today.append(t)
                except (KeyError, IndexError):
                    continue
        
        # If no tickers have data for this date, carry forward the previous portfolio value
        if not tradable_tickers_today:
            # Carry forward the previous portfolio values
            if i > 0:
                prev_value_with = portfolio_value_with_additions.iloc[-1]
                prev_value_without = portfolio_value_without_additions.iloc[-1]
                # Only carry forward if we have valid previous values
                if not pd.isna(prev_value_with) and not pd.isna(prev_value_without):
                    portfolio_value_with_additions.loc[current_date] = prev_value_with
                    portfolio_value_without_additions.loc[current_date] = prev_value_without
            continue
        # Progress bar update
        elapsed = time.time() - start_time
        steps_left = total_steps - (i + 1)
        if i > 0:
            avg_time_per_step = elapsed / (i + 1)
            eta = avg_time_per_step * steps_left
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        else:
            eta_str = 'estimating...'
        percent_complete = (i + 1) / total_steps
        progress_bar.progress(percent_complete, text=f"Progress: {int(percent_complete*100)}% | ETA: {eta_str}")
        progress_text.write(f"Step {i+1}/{total_steps} | Time left: {eta_str}")
        
        # --- Update portfolio values for today based on yesterday's shares ---
        current_port_value_with_additions = cash_with_additions
        current_port_value_without_additions = cash_without_additions
        
        asset_values_with_additions = {}
        asset_values_without_additions = {}
        
        for t in asset_shares_with_additions.index:
            # Bulletproof data access - only process if we have valid data for this ticker and date
            ticker_data = data.get(t, pd.DataFrame())
            if ticker_data.empty or current_date not in ticker_data.index:
                continue
                
            try:
                price = float(ticker_data.loc[current_date, "Close"])
                if pd.isna(price) or price <= 0:
                    continue
                    
                # Safe dividend access
                dividend = 0.0
                if "Dividend_per_share" in ticker_data.columns:
                    try:
                        dividend_val = ticker_data.loc[current_date, "Dividend_per_share"]
                        dividend = float(dividend_val) if not pd.isna(dividend_val) else 0.0
                    except:
                        dividend = 0.0
                if pd.isna(dividend):
                    dividend = 0.0
                    
                shares_with = asset_shares_with_additions.loc[t]
                shares_without = asset_shares_without_additions.loc[t]
                
                # Only credit dividends if include_dividends is True
                if price > 0 and dividend > 0 and include_dividends.get(t, False):
                    dividend_cash = shares_with * dividend
                    dividend_cash_wo = shares_without * dividend
                    
                    # Check if dividends should be collected as cash instead of reinvested
                    collect_as_cash = st.session_state.get('collect_dividends_as_cash', False)
                    if collect_as_cash:
                        # Add dividend cash to cash instead of reinvesting
                        cash_with_additions += dividend_cash
                        cash_without_additions += dividend_cash_wo
                    else:
                        # Reinvest dividends (original behavior)
                        asset_shares_with_additions.loc[t] += dividend_cash / price
                        asset_shares_without_additions.loc[t] += dividend_cash_wo / price

                asset_value_with = asset_shares_with_additions.loc[t] * price
                asset_value_without = asset_shares_without_additions.loc[t] * price

                current_port_value_with_additions += asset_value_with
                current_port_value_without_additions += asset_value_without

                asset_values_with_additions[t] = asset_value_with
                asset_values_without_additions[t] = asset_value_without
                
            except Exception:
                # Skip this ticker if there's any error
                continue
        
        # Add periodic cash (if applicable)
        if current_date in added_cash_dates and current_date != all_dates[0] and added_amount > 0:
            current_port_value_with_additions += added_amount
            cash_with_additions += added_amount
            cash_flows_with_additions.loc[current_date] = -added_amount
        # Apply portfolio drag (annualized) if provided in session state
        try:
            drag_pct = float(st.session_state.get('portfolio_drag_pct', 0.0))
        except Exception:
            drag_pct = 0.0

        if drag_pct != 0.0:
            # safe compute daily multiplier: m = (1 - drag_pct/100) ** (1/252)
            with np.errstate(over='ignore', invalid='ignore'):
                base = 1.0 - (drag_pct / 100.0)
                # prevent negative base causing complex numbers
                if base <= 0:
                    m = 0.0
                else:
                    m = base ** (1.0 / 252.0)

            # For WITH additions: keep asset shares same, adjust cash to match new total after drag
            try:
                sum_assets_with = sum(asset_values_with_additions.values()) if asset_values_with_additions else 0.0
                total_with = sum_assets_with + cash_with_additions
                new_total_with = total_with * m
                cash_with_additions = new_total_with - sum_assets_with
            except Exception:
                pass

            # For WITHOUT additions: same logic
            try:
                sum_assets_wo = sum(asset_values_without_additions.values()) if asset_values_without_additions else 0.0
                total_wo = sum_assets_wo + cash_without_additions
                new_total_wo = total_wo * m
                cash_without_additions = new_total_wo - sum_assets_wo
            except Exception:
                pass

        # Record final values for the day
        portfolio_value_with_additions.loc[current_date] = sum(asset_values_with_additions.values()) + cash_with_additions
        portfolio_value_without_additions.loc[current_date] = sum(asset_values_without_additions.values()) + cash_without_additions
        
        # --- Rebalance portfolio on designated dates ---
        # Check if we should rebalance
        should_rebalance = False
        if current_date in rebalancing_dates:
            should_rebalance = True
        elif rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"]:
            # Buy & Hold: rebalance whenever there's cash available
            if cash_with_additions > 0:
                should_rebalance = True
        
        # Rebalance with additions
        if should_rebalance:
            # print(f"Rebalancing portfolio with additions on {current_date.date()}...")
            target_alloc_with, rebalance_metrics = _rebalance_portfolio(
                current_date, 
                current_port_value_with_additions, 
                data, 
                tradable_tickers_today,
                use_momentum, 
                momentum_windows, 
                negative_momentum_strategy,
                use_relative_momentum, 
                allocations,
                calc_beta,
                beta_window_days,
                exclude_days_beta,
                benchmark_ticker,
                calc_volatility,
                vol_window_days,
                exclude_days_vol
            )
            # Store rebalance metrics ONLY on true rebalancing dates
            rebalance_metrics_list.append(rebalance_metrics)
            
            # Record last rebalance allocation
            last_rebalance_allocations = rebalance_metrics["target_allocation"]

            # Reset shares to zero and calculate new shares, updating cash
            new_cash = current_port_value_with_additions
            new_shares = pd.Series(0.0, index=tickers_with_data)
            
            for t, target_val in target_alloc_with.items():
                # Use last available price on or before current_date
                df_t = data.get(t, pd.DataFrame())
                if not df_t.empty:
                    # Find the last available price before or on current_date
                    price_idx = df_t.index.asof(current_date)
                    if pd.notna(price_idx):
                        price = float(df_t.loc[price_idx, "Close"])
                        if price > 0:
                            new_shares.loc[t] = target_val / price
                            new_cash -= target_val
            
            asset_shares_with_additions = new_shares
            cash_with_additions = new_cash

        # Rebalance without additions
        if should_rebalance:
            # print(f"Rebalancing portfolio without additions on {current_date.date()}...")
            
            target_alloc_without, _ = _rebalance_portfolio(
                current_date, 
                current_port_value_without_additions, 
                data, 
                tradable_tickers_today,
                use_momentum, 
                momentum_windows, 
                negative_momentum_strategy,
                use_relative_momentum, 
                allocations,
                calc_beta,
                beta_window_days,
                exclude_days_beta,
                benchmark_ticker,
                calc_volatility,
                vol_window_days,
                exclude_days_vol
            )

            # Reset shares to zero and calculate new shares, updating cash
            new_cash = current_port_value_without_additions
            new_shares = pd.Series(0.0, index=tickers_with_data)
            
            for t, target_val in target_alloc_without.items():
                df_t = data.get(t, pd.DataFrame())
                if not df_t.empty:
                    price_idx = df_t.index.asof(current_date)
                    if pd.notna(price_idx):
                        price = float(df_t.loc[price_idx, "Close"])
                        if price > 0:
                            new_shares.loc[t] = target_val / price
                            new_cash -= target_val

            asset_shares_without_additions = new_shares
            cash_without_additions = new_cash
    
    # print("\nSimulation complete. Calculating final values...")
    
    # Calculate final (current) allocations
    final_port_value_with_additions = portfolio_value_with_additions.iloc[-1]
    current_allocs = {
        t: (asset_shares_with_additions[t] * data[t].loc[all_dates[-1], 'Close']) / final_port_value_with_additions
        for t in tickers_with_data if final_port_value_with_additions > 0 and asset_shares_with_additions[t] > 0
    }
    # Add cash to the current allocation
    cash_perc = cash_with_additions / final_port_value_with_additions if final_port_value_with_additions > 0 else 0
    if cash_perc > 0.0001: # Add cash if it's not just dust
        current_allocs["CASH"] = cash_perc
    

    # Compute stats (guard if no benchmark) - BULLETPROOF APPROACH
    bench_returns = None
    if benchmark_ticker and benchmark_ticker in data:
        try:
            # Only use dates that exist in both portfolio and benchmark data
            benchmark_df = data[benchmark_ticker]
            common_dates = all_dates.intersection(benchmark_df.index)
            if len(common_dates) > 0:
                bench_returns = benchmark_df.loc[common_dates, "Close"].pct_change()
                # Reindex to all_dates, filling missing values with 0
                bench_returns = bench_returns.reindex(all_dates, fill_value=0.0)
            else:
                bench_returns = pd.Series(0.0, index=all_dates)
        except Exception:
            bench_returns = pd.Series(0.0, index=all_dates)
    else:
        bench_returns = pd.Series(0.0, index=all_dates)
    
    final_beta = np.nan
    if calc_beta and benchmark_ticker and benchmark_ticker in data:
        try:
            # Use only common dates for beta calculation
            benchmark_df = data[benchmark_ticker]
            common_dates = all_dates.intersection(benchmark_df.index)
            if len(common_dates) > 1:
                benchmark_returns_series = benchmark_df.loc[common_dates, "Close"].pct_change().dropna()
                portfolio_returns = portfolio_value_with_additions.loc[common_dates].pct_change().dropna()
                final_beta = calculate_beta(portfolio_returns, benchmark_returns_series)
        except Exception:
            final_beta = np.nan
    
    
    # print("\n--- Results with Cash Additions ---")
    stats_with_additions = run_stats(
        portfolio_value_with_additions.dropna(),
        bench_returns,
        cash_flows_with_additions.fillna(0.0),
        final_beta
    )
    
    # print("\n--- Results without Cash Additions ---")
    stats_without_additions = run_stats(
        portfolio_value_without_additions.dropna(),
        bench_returns,
        cash_flows_without_additions.fillna(0.0),
        final_beta,
        mwrr_portfolio_values=portfolio_value_with_additions.dropna(),  # Use with-additions for MWRR
        mwrr_cash_flows=cash_flows_with_additions.fillna(0.0)  # Use with-additions cash flows for MWRR
    )

    # Add option to switch between yearly and monthly performance
    performance_period = st.session_state.get("performance_period", "Year")
    performance_table, _ = calculate_periodic_performance(portfolio_value_without_additions, period=performance_period)
    styled_allocs_table, styled_metrics_table = create_rebalance_tables(rebalance_metrics_list, tickers_with_data)

    plots = {}
    
    # Note: The main interactive figure was removed per user request.
    # Keep only the standard set of plots in `plots` returned to the UI.

    fig_value = go.Figure()
    fig_value.add_trace(go.Scatter(x=all_dates, y=portfolio_value_with_additions,
                                   mode='lines', name='Portfolio (with additions)'))
    fig_value.add_trace(go.Scatter(x=all_dates, y=portfolio_value_without_additions,
                                   mode='lines', name='Portfolio (without additions)'))
    fig_value.update_layout(
        title='Portfolio Value Over Time (includes added cash)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode="x unified"
    )
    plots["Portfolio Value (Counting Added Cash)"] = fig_value

    drawdown_series_with = (portfolio_value_with_additions / portfolio_value_with_additions.expanding(min_periods=1).max() - 1) * 100
    drawdown_series_without = (portfolio_value_without_additions / portfolio_value_without_additions.expanding(min_periods=1).max() - 1) * 100
    
    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(go.Scatter(x=all_dates, y=drawdown_series_with,
                                      mode='lines', name='Drawdown (with additions)'))
    fig_drawdown.add_trace(go.Scatter(x=all_dates, y=drawdown_series_without,
                                      mode='lines', name='Drawdown (without additions)'))
    fig_drawdown.update_layout(
        title='Drawdown History',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode="x unified"
    )
    plots["Drawdown History"] = fig_drawdown
    
    normalized_portfolio = portfolio_value_without_additions / portfolio_value_without_additions.iloc[0] * 100
    if benchmark_ticker and benchmark_ticker in data:
        try:
            # Only use dates that exist in benchmark data
            benchmark_df = data[benchmark_ticker]
            common_dates = all_dates.intersection(benchmark_df.index)
            if len(common_dates) > 0:
                benchmark_data = benchmark_df.loc[common_dates, "Close"]
                normalized_benchmark = benchmark_data / benchmark_data.iloc[0] * 100
                # Reindex to all_dates for plotting
                normalized_benchmark = normalized_benchmark.reindex(all_dates, method='ffill')
            else:
                normalized_benchmark = pd.Series(index=all_dates, dtype=float)
        except Exception:
            normalized_benchmark = pd.Series(index=all_dates, dtype=float)
    else:
        normalized_benchmark = pd.Series(index=all_dates, dtype=float)
    
    fig_benchmark = go.Figure()
    fig_benchmark.add_trace(go.Scatter(x=all_dates, y=normalized_portfolio,
                                       mode='lines', name='Portfolio (no additions)'))
    if not normalized_benchmark.empty:
        fig_benchmark.add_trace(go.Scatter(x=all_dates, y=normalized_benchmark,
                                           mode='lines', name=f"Benchmark ({benchmark_ticker})"))
    fig_benchmark.update_layout(
        title='Portfolio vs. Benchmark (Normalized)',
        xaxis_title='Date',
        yaxis_title='Normalized Value',
        hovermode="x unified"
    )
    plots["Portfolio vs. Benchmark"] = fig_benchmark
    # --- Drawdown compare for Portfolio (no additions) vs Benchmark (no additions)
    try:
        if benchmark_ticker and benchmark_ticker in data:
            # Only use dates that exist in benchmark data
            benchmark_df = data[benchmark_ticker]
            common_dates = all_dates.intersection(benchmark_df.index)
            if len(common_dates) > 0:
                bench_series = benchmark_df.loc[common_dates, "Close"]
                # Reindex to all_dates for plotting
                bench_series = bench_series.reindex(all_dates, method='ffill')
            else:
                bench_series = pd.Series(index=all_dates, dtype=float)
            drawdown_bench = (bench_series / bench_series.expanding(min_periods=1).max() - 1) * 100
        else:
            drawdown_bench = pd.Series(index=all_dates, dtype=float)
    except Exception:
        drawdown_bench = pd.Series(index=all_dates, dtype=float)

    # drawdown_series_without already computed earlier for portfolio without additions
    fig_bench_dd = go.Figure()
    fig_bench_dd.add_trace(go.Scatter(x=all_dates, y=drawdown_series_without,
                                      mode='lines', name='Portfolio Drawdown (no additions)'))
    if not drawdown_bench.empty:
        fig_bench_dd.add_trace(go.Scatter(x=all_dates, y=drawdown_bench,
                                          mode='lines', name=f"Benchmark Drawdown ({benchmark_ticker})"))
    fig_bench_dd.update_layout(
        title='Drawdown: Portfolio vs. Benchmark (no additions)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode="x unified"
    )
    plots["Drawdown: Portfolio vs. Benchmark"] = fig_bench_dd
    
    # --- Additional plot: normalized comparison that includes cash additions for both portfolio and benchmark
    # Build benchmark series that simulates buying the benchmark with each cash addition so it can be
    # compared fairly to `portfolio_value_with_additions` which already counts added cash.
    if benchmark_ticker and benchmark_ticker in data:
        try:
            # Only use dates that exist in benchmark data
            benchmark_df = data[benchmark_ticker]
            common_dates = all_dates.intersection(benchmark_df.index)
            if len(common_dates) > 0:
                bench_close = benchmark_df.loc[common_dates, "Close"]
                # Reindex to all_dates for consistent access
                bench_close = bench_close.reindex(all_dates, method='ffill')
                
                shares = 0.0
                bench_vals = []
                # cash_flows_with_additions uses negative values for cash invested into the portfolio
                for d in all_dates:
                    cf = 0.0
                    try:
                        cf = cash_flows_with_additions.loc[d]
                    except Exception:
                        cf = 0.0
                    # if cf is negative, it's an investment into the portfolio -> buy benchmark
                    if pd.notna(cf) and cf < 0:
                        invest = -float(cf)
                        try:
                            price = bench_close.loc[d]
                            if price and price > 0:
                                shares += invest / price
                        except Exception:
                            pass  # Skip if price not available
                    bench_vals.append(shares * bench_close.loc[d])
                benchmark_with_additions = pd.Series(bench_vals, index=all_dates)
            else:
                benchmark_with_additions = pd.Series(index=all_dates, dtype=float)
        except Exception:
            benchmark_with_additions = pd.Series(index=all_dates, dtype=float)
    else:
        benchmark_with_additions = pd.Series(index=all_dates, dtype=float)

    # Normalize both series to 100 at the start for visual comparison
    normalized_portfolio_with_additions = (portfolio_value_with_additions / portfolio_value_with_additions.iloc[0] * 100)
    if not benchmark_with_additions.empty and benchmark_with_additions.iloc[0] != 0:
        normalized_benchmark_with_additions = benchmark_with_additions / benchmark_with_additions.iloc[0] * 100
    else:
        normalized_benchmark_with_additions = pd.Series(index=all_dates, dtype=float)

    fig_benchmark_with_add = go.Figure()
    fig_benchmark_with_add.add_trace(go.Scatter(x=all_dates, y=normalized_portfolio_with_additions,
                                                mode='lines', name='Portfolio (with additions)'))
    if not normalized_benchmark_with_additions.empty:
        fig_benchmark_with_add.add_trace(go.Scatter(x=all_dates, y=normalized_benchmark_with_additions,
                                                   mode='lines', name=f"Benchmark ({benchmark_ticker}) (with additions)"))
    fig_benchmark_with_add.update_layout(
        title='Portfolio vs. Benchmark (Normalized, includes additions)',
        xaxis_title='Date',
        yaxis_title='Normalized Value',
        hovermode="x unified"
    )
    plots["Portfolio vs. Benchmark (with additions)"] = fig_benchmark_with_add
    # --- Drawdown compare for Portfolio (with additions) vs Benchmark (with additions)
    try:
        if not benchmark_with_additions.empty:
            drawdown_bench_with = (benchmark_with_additions / benchmark_with_additions.expanding(min_periods=1).max() - 1) * 100
        else:
            drawdown_bench_with = pd.Series(index=all_dates, dtype=float)
    except Exception:
        drawdown_bench_with = pd.Series(index=all_dates, dtype=float)

    # drawdown_series_with already computed earlier for portfolio with additions
    fig_bench_with_dd = go.Figure()
    fig_bench_with_dd.add_trace(go.Scatter(x=all_dates, y=drawdown_series_with,
                                          mode='lines', name='Portfolio Drawdown (with additions)'))
    if not drawdown_bench_with.empty:
        fig_bench_with_dd.add_trace(go.Scatter(x=all_dates, y=drawdown_bench_with,
                                              mode='lines', name=f"Benchmark Drawdown ({benchmark_ticker}) (with additions)"))
    fig_bench_with_dd.update_layout(
        title='Drawdown: Portfolio vs. Benchmark (with additions)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode="x unified"
    )
    plots["Drawdown: Portfolio vs. Benchmark (with additions)"] = fig_bench_with_dd
    
    return (
        plots,
        stats_with_additions,
        stats_without_additions,
        performance_table,
        last_rebalance_allocations,
        current_allocs,
        styled_allocs_table,
        styled_metrics_table,
        portfolio_value_with_additions,
        portfolio_value_without_additions,
        cash_flows_with_additions,
        cash_flows_without_additions,
    )


# ==============================================================================
# Streamlit App Logic
# ==============================================================================
st.set_page_config(page_title="Quantitative Portfolio Momentum Backtest & Analytics", layout="wide", page_icon="📈")

# Handle imported values from JSON - MUST BE AT THE VERY BEGINNING
if "_import_name" in st.session_state:
    st.session_state["portfolio_name"] = st.session_state.pop("_import_name")
    st.session_state["portfolio_name_input"] = st.session_state["portfolio_name"]
if "_import_tickers" in st.session_state:
    st.session_state["tickers"] = list(st.session_state.pop("_import_tickers"))
if "_import_allocs" in st.session_state:
    st.session_state["allocs"] = [float(a) for a in st.session_state.pop("_import_allocs")]
if "_import_divs" in st.session_state:
    st.session_state["divs"] = [bool(d) for d in st.session_state.pop("_import_divs")]
if "_import_initial_value" in st.session_state:
    val = st.session_state.pop("_import_initial_value")
    try:
        float_val = float(val)
        st.session_state["initial_value"] = float_val
        st.session_state["initial_value_input_int"] = int(float_val)
        st.session_state["initial_value_input"] = float_val
    except Exception:
        pass
if "_import_added_amount" in st.session_state:
    val = st.session_state.pop("_import_added_amount")
    try:
        float_val = float(val)
        st.session_state["added_amount"] = float_val
        st.session_state["added_amount_input_int"] = int(float_val)
        st.session_state["added_amount_input"] = float_val
    except Exception:
        pass
if "_import_rebalancing_frequency" in st.session_state:
    st.session_state["rebalancing_frequency"] = st.session_state.pop("_import_rebalancing_frequency")
    st.session_state["rebalancing_frequency_widget"] = st.session_state["rebalancing_frequency"]
if "_import_added_frequency" in st.session_state:
    st.session_state["added_frequency"] = st.session_state.pop("_import_added_frequency")
    st.session_state["added_frequency_widget"] = st.session_state["added_frequency"]
if "_import_use_custom_dates" in st.session_state:
    st.session_state["use_custom_dates"] = bool(st.session_state.pop("_import_use_custom_dates"))
    st.session_state["use_custom_dates_checkbox"] = st.session_state["use_custom_dates"]
if "_import_start_date" in st.session_state:
    sd = st.session_state.pop("_import_start_date")
    st.session_state["start_date"] = None if sd in (None, 'None', '') else pd.to_datetime(sd).date()
if "_import_end_date" in st.session_state:
    ed = st.session_state.pop("_import_end_date")
    st.session_state["end_date"] = None if ed in (None, 'None', '') else pd.to_datetime(ed).date()
if "_import_start_with" in st.session_state:
    st.session_state["start_with_radio_key"] = st.session_state.pop("_import_start_with")
if "_import_first_rebalance_strategy" in st.session_state:
    st.session_state["first_rebalance_strategy"] = st.session_state.pop("_import_first_rebalance_strategy")
    st.session_state["first_rebalance_strategy_radio_key"] = st.session_state["first_rebalance_strategy"]
if "_import_use_momentum" in st.session_state:
    st.session_state["use_momentum"] = bool(st.session_state.pop("_import_use_momentum"))
    st.session_state["use_momentum_checkbox"] = st.session_state["use_momentum"]
if "_import_momentum_strategy" in st.session_state:
    st.session_state["momentum_strategy"] = st.session_state.pop("_import_momentum_strategy")
    st.session_state["momentum_strategy_radio"] = st.session_state["momentum_strategy"]
if "_import_negative_momentum_strategy" in st.session_state:
    st.session_state["negative_momentum_strategy"] = st.session_state.pop("_import_negative_momentum_strategy")
    st.session_state["negative_momentum_strategy_radio"] = st.session_state["negative_momentum_strategy"]
if "_import_mom_windows" in st.session_state:
    # Convert momentum window weights from percentage to decimal format
    imported_windows = st.session_state.pop("_import_mom_windows")
    converted_windows = []
    for window in imported_windows:
        converted_window = window.copy()
        weight = window.get('weight', 0.0)
        if isinstance(weight, (int, float)):
            if weight > 1.0:
                # If weight is stored as percentage, convert to decimal
                converted_window['weight'] = weight / 100.0
            else:
                # Already in decimal format, use as is
                converted_window['weight'] = weight
        else:
            converted_window['weight'] = 0.0
        converted_windows.append(converted_window)
    st.session_state["mom_windows"] = converted_windows
if "_import_use_beta" in st.session_state:
    st.session_state["use_beta"] = bool(st.session_state.pop("_import_use_beta"))
    st.session_state["use_beta_checkbox"] = st.session_state["use_beta"]
if "_import_beta_window_days" in st.session_state:
    st.session_state["beta_window_days"] = int(st.session_state.pop("_import_beta_window_days"))
    st.session_state["beta_window_input"] = st.session_state["beta_window_days"]
if "_import_beta_exclude_days" in st.session_state:
    st.session_state["beta_exclude_days"] = int(st.session_state.pop("_import_beta_exclude_days"))
    st.session_state["beta_exclude_input"] = st.session_state["beta_exclude_days"]
if "_import_use_vol" in st.session_state:
    st.session_state["use_vol"] = bool(st.session_state.pop("_import_use_vol"))
    st.session_state["use_vol_checkbox"] = st.session_state["use_vol"]
if "_import_vol_window_days" in st.session_state:
    st.session_state["vol_window_days"] = int(st.session_state.pop("_import_vol_window_days"))
    st.session_state["vol_window_input"] = st.session_state["vol_window_days"]
if "_import_vol_exclude_days" in st.session_state:
    st.session_state["vol_exclude_days"] = int(st.session_state.pop("_import_vol_exclude_days"))
    st.session_state["vol_exclude_input"] = st.session_state["vol_exclude_days"]
if "_import_portfolio_drag_pct" in st.session_state:
    try:
        st.session_state["portfolio_drag_pct"] = float(st.session_state.pop("_import_portfolio_drag_pct"))
    except Exception:
        st.session_state["portfolio_drag_pct"] = st.session_state.pop("_import_portfolio_drag_pct")
if "_import_benchmark_ticker" in st.session_state:
    # This will be handled by the text_input widget directly when it renders
    pass
if "_import_collect_dividends_as_cash" in st.session_state:
    st.session_state["collect_dividends_as_cash"] = bool(st.session_state.pop("_import_collect_dividends_as_cash"))
    st.session_state["collect_dividends_as_cash_checkbox"] = st.session_state["collect_dividends_as_cash"]

# Custom CSS for a better layout, a distinct primary button, and the fixed 'Back to Top' button
st.markdown("""
<style>
    /* Global Styles for the App */
    .st-emotion-cache-1f87s81 {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .st-emotion-cache-1v0bb62 button {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .st-emotion-cache-1v0bb62 button:hover {
        background-color: #0056b3 !important;
        border-color: #0056b3 !important;
        transform: translateY(-2px);
    }
    /* Fix for the scrollable dataframe - forces it to be non-scrollable */
    div.st-emotion-cache-1ftv8z > div {
        overflow: visible !important;
        max-height: none !important;
    }
    /* Control sidebar width */
    .st-emotion-cache-1d391ky {
        width: 400px !important;
        min-width: 400px !important;
    }
    /* Ensure sidebar content is visible */
    .st-emotion-cache-1d391ky .st-emotion-cache-1r6slb0 {
        width: 100% !important;
        max-width: none !important;
    }
    /* Simple sidebar width control only - reduced since we're zooming out */
    .st-emotion-cache-1d391ky {
        width: 320px !important;
        min-width: 320px !important;
    }
    /* 65% zoom effect - more zoomed out */
    .st-emotion-cache-1wivap2 {
        font-size: 10px !important;
        zoom: 0.65 !important;
    }
    .st-emotion-cache-1r6slb0 {
        font-size: 10px !important;
        zoom: 0.65 !important;
    }
    /* Control input and button sizes for 65% zoom */
    .st-emotion-cache-1r6slb0 input, .st-emotion-cache-1r6slb0 select {
        font-size: 10px !important;
        zoom: 0.65 !important;
    }
    .st-emotion-cache-1r6slb0 button {
        font-size: 10px !important;
        padding: 3px 6px !important;
        zoom: 0.65 !important;
    }
    /* Control plot containers for 65% zoom */
    .plotly-graph-div {
        zoom: 0.65 !important;
    }
    /* Make the 'View Details' button more obvious */
    button[aria-label="View Details"] {
        background-color: #0ea5e9 !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: 0 4px 8px rgba(14,165,233,0.16) !important;
    }
    button[aria-label="View Details"]:hover {
        background-color: #0891b2 !important;
    }
</style>
<a id="top"></a>
<button id="back-to-top" onclick="window.scrollTo(0, 0);">⬆️</button>
<style>
    #back-to-top {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        opacity: 0.7;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 24px;
        cursor: pointer;
        display: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: opacity 0.3s;
    }
    #back-to-top:hover {
        opacity: 1;
    }
</style>
<script>
    window.onscroll = function() {
        var button = document.getElementById("back-to-top");
        if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {
            button.style.display = "block";
        } else {
            button.style.display = "none";
        }
    };
</script>
""", unsafe_allow_html=True)

st.title("Backtest Engine")

# Portfolio Name - will be created after JSON import processing

# -------------------------
# Session state defaults
# -------------------------
def _ss_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

_ss_default("tickers", ["SPY", "QQQ", "GLD", "TLT"])
_ss_default("allocs", [1.0 / len(st.session_state.tickers) if len(st.session_state.tickers) > 0 else 0.0 for _ in st.session_state.tickers])
_ss_default("divs", [True for _ in st.session_state.tickers])

_ss_default("mom_windows", [
    {"lookback": 365, "exclude": 30, "weight": 0.50},
    {"lookback": 180, "exclude": 30, "weight": 0.30},
    {"lookback": 120, "exclude": 30, "weight": 0.20},
])

_ss_default("use_momentum", True)
_ss_default("use_relative_momentum", False)
    # Removed use_decimals - no longer needed
_ss_default("momentum_strategy", "Classic momentum")
_ss_default("negative_momentum_strategy", "Go to cash")
_ss_default("use_beta", True)
_ss_default("beta_window_days", 365)
_ss_default("beta_exclude_days", 30)
_ss_default("use_vol", True)
_ss_default("vol_window_days", 365)
_ss_default("vol_exclude_days", 30)

_ss_default("use_custom_dates", False)
_ss_default("start_date", None)
_ss_default("end_date", date.today())  # allow today; loader will handle exclusivity

# Add portfolio settings to session state defaults
_ss_default("initial_value", 10000)
_ss_default("added_amount", 1000)
_ss_default("rebalancing_frequency", "Monthly")
_ss_default("added_frequency", "Monthly")
_ss_default("start_with_radio_key", "oldest")
_ss_default("collect_dividends_as_cash", False)

_ss_default("fig_dict", None)      
_ss_default("console", "")   
_ss_default("running", False)
_ss_default("last_run_time", None)
_ss_default("error", None)
_ss_default("yearly_table", None)
_ss_default("performance_table", None)
_ss_default("stats_with_additions", None)
_ss_default("stats_without_additions", None)
_ss_default("last_rebalance_allocs", None)
_ss_default("current_allocs", None)
_ss_default("rebalance_alloc_table", None)
_ss_default("rebalance_metrics_table", None)

# --- Apply any staged imports (from JSON) before widgets are created ---
# The JSON importer writes staging keys prefixed with `_import_` and sets
# `_import_pending` to True then reruns; this block applies them safely
# before any widgets instantiate (prevents StreamlitAPIException).
if st.session_state.get("_import_pending", False):
    try:
        # Map of staging key -> target session_state key or transformation
        if "_import_name" in st.session_state:
            st.session_state["portfolio_name"] = st.session_state.pop("_import_name")
            # Also update the widget key to ensure the UI reflects the imported name
            st.session_state["portfolio_name_input"] = st.session_state["portfolio_name"]
        if "_import_tickers" in st.session_state:
            st.session_state["tickers"] = list(st.session_state.pop("_import_tickers"))
        if "_import_allocs" in st.session_state:
            st.session_state["allocs"] = [float(a) for a in st.session_state.pop("_import_allocs")]
        if "_import_divs" in st.session_state:
            st.session_state["divs"] = [bool(d) for d in st.session_state.pop("_import_divs")]
        if "_import_initial_value" in st.session_state:
            val = st.session_state.pop("_import_initial_value")
            try:
                float_val = float(val)
                st.session_state["initial_value"] = float_val
                st.session_state["initial_value_input_int"] = int(float_val)
                st.session_state["initial_value_input"] = float_val
            except Exception:
                pass
        if "_import_added_amount" in st.session_state:
            val = st.session_state.pop("_import_added_amount")
            try:
                float_val = float(val)
                st.session_state["added_amount"] = float_val
                st.session_state["added_amount_input_int"] = int(float_val)
                st.session_state["added_amount_input"] = float_val
            except Exception:
                pass
        if "_import_rebalancing_frequency" in st.session_state:
            st.session_state["rebalancing_frequency"] = st.session_state.pop("_import_rebalancing_frequency")
            st.session_state["rebalancing_frequency_widget"] = st.session_state["rebalancing_frequency"]
        if "_import_added_frequency" in st.session_state:
            st.session_state["added_frequency"] = st.session_state.pop("_import_added_frequency")
            st.session_state["added_frequency_widget"] = st.session_state["added_frequency"]
        if "_import_use_custom_dates" in st.session_state:
            st.session_state["use_custom_dates"] = bool(st.session_state.pop("_import_use_custom_dates"))
            st.session_state["use_custom_dates_checkbox"] = st.session_state["use_custom_dates"]
        if "_import_start_date" in st.session_state:
            sd = st.session_state.pop("_import_start_date")
            start_date = None if sd in (None, 'None', '') else pd.to_datetime(sd).date()
            st.session_state["start_date"] = start_date
            if start_date is not None:
                st.session_state["use_custom_dates"] = True
                st.session_state["use_custom_dates_checkbox"] = True
        if "_import_end_date" in st.session_state:
            ed = st.session_state.pop("_import_end_date")
            end_date = None if ed in (None, 'None', '') else pd.to_datetime(ed).date()
            st.session_state["end_date"] = end_date
            if end_date is not None:
                st.session_state["use_custom_dates"] = True
                st.session_state["use_custom_dates_checkbox"] = True

        
        # Handle portfolio-specific JSON imports for main app
        if "_import_portfolio_config" in st.session_state:
            portfolio_config = st.session_state.pop("_import_portfolio_config")
            if isinstance(portfolio_config, dict):
                # Main app specific: ensure all required fields are present
                # and ignore fields that are specific to other pages
                main_app_config = {
                    'name': portfolio_config.get('name', 'Main Portfolio'),
                    'stocks': portfolio_config.get('stocks', []),
                    'benchmark_ticker': portfolio_config.get('benchmark_ticker', '^GSPC'),
                    'initial_value': portfolio_config.get('initial_value', 10000),
                    'added_amount': portfolio_config.get('added_amount', 1000),
                    'added_frequency': portfolio_config.get('added_frequency', 'month'),
                    'rebalancing_frequency': portfolio_config.get('rebalancing_frequency', 'month'),
                    'start_date_user': portfolio_config.get('start_date_user'),
                    'end_date_user': portfolio_config.get('end_date_user'),
                    'start_with': portfolio_config.get('start_with', 'all'),
                    'use_momentum': portfolio_config.get('use_momentum', True),
                    'use_relative_momentum': portfolio_config.get('use_relative_momentum', False),
            
                    'momentum_strategy': portfolio_config.get('momentum_strategy', 'Classic'),
                    'negative_momentum_strategy': portfolio_config.get('negative_momentum_strategy', 'Cash'),
                    'momentum_windows': portfolio_config.get('momentum_windows', []),
                    'calc_beta': portfolio_config.get('calc_beta', True),
                    'calc_volatility': portfolio_config.get('calc_volatility', True),
                    'beta_window_days': portfolio_config.get('beta_window_days', 365),
                    'exclude_days_beta': portfolio_config.get('exclude_days_beta', 30),
                    'vol_window_days': portfolio_config.get('vol_window_days', 365),
                    'exclude_days_vol': portfolio_config.get('exclude_days_vol', 30),
                    'collect_dividends_as_cash': portfolio_config.get('collect_dividends_as_cash', False),
                }
                
                # Apply the portfolio configuration to main app session state
                if 'name' in main_app_config:
                    st.session_state["portfolio_name"] = main_app_config['name']
                    # Also update the widget key to ensure the UI reflects the imported name
                    st.session_state["portfolio_name_input"] = main_app_config['name']
                if 'stocks' in main_app_config and main_app_config['stocks']:
                    # Clear existing tickers first
                    st.session_state["tickers"] = []
                    st.session_state["allocs"] = []
                    st.session_state["divs"] = []
                    
                    # Clear all ticker widget keys to prevent UI interference
                    for key in list(st.session_state.keys()):
                        if key.startswith("ticker_") or key.startswith("alloc_input_") or key.startswith("divs_checkbox_"):
                            del st.session_state[key]
                    
                    # Extract tickers and allocations from stocks array
                    stocks = main_app_config['stocks']
                    tickers = []
                    allocations = []
                    dividends = []
                    
                    for stock in stocks:
                        if stock.get('ticker'):
                            tickers.append(stock['ticker'])
                            # Get allocation (convert from percentage to decimal if needed)
                            allocation = stock.get('allocation', 0.0)
                            if isinstance(allocation, (int, float)):
                                if allocation > 1.0:
                                    # Convert percentage to decimal
                                    allocations.append(allocation / 100.0)
                                else:
                                    # Already in decimal format
                                    allocations.append(allocation)
                            else:
                                allocations.append(0.0)
                            # Get dividend setting
                            dividends.append(stock.get('include_dividends', True))
                    
                    st.session_state["tickers"] = tickers
                    st.session_state["allocs"] = allocations
                    st.session_state["divs"] = dividends
                
                if 'benchmark_ticker' in main_app_config:
                    st.session_state["_import_benchmark_ticker"] = main_app_config['benchmark_ticker']
                if 'initial_value' in main_app_config:
                    try:
                        val = float(main_app_config['initial_value'])
                        st.session_state["initial_value"] = val
                        st.session_state["initial_value_input_int"] = int(val)
                        st.session_state["initial_value_input"] = val
                    except Exception:
                        pass
                if 'added_amount' in main_app_config:
                    try:
                        val = float(main_app_config['added_amount'])
                        st.session_state["added_amount"] = val
                        st.session_state["added_amount_input_int"] = int(val)
                        st.session_state["added_amount_input"] = val
                    except Exception:
                        pass
                if 'rebalancing_frequency' in main_app_config:
                    st.session_state["rebalancing_frequency"] = main_app_config['rebalancing_frequency']
                    st.session_state["rebalancing_frequency_widget"] = main_app_config['rebalancing_frequency']
                if 'added_frequency' in main_app_config:
                    st.session_state["added_frequency"] = main_app_config['added_frequency']
                    st.session_state["added_frequency_widget"] = main_app_config['added_frequency']
                if 'collect_dividends_as_cash' in main_app_config:
                    st.session_state["collect_dividends_as_cash"] = bool(main_app_config['collect_dividends_as_cash'])
                    st.session_state["collect_dividends_as_cash_checkbox"] = bool(main_app_config['collect_dividends_as_cash'])
                if 'start_date_user' in main_app_config and main_app_config['start_date_user'] is not None:
                    # Parse date from string if needed
                    start_date = main_app_config['start_date_user']
                    if isinstance(start_date, str):
                        try:
                            start_date = pd.to_datetime(start_date).date()
                        except:
                            start_date = None
                    if start_date is not None:
                        st.session_state["start_date"] = start_date
                        st.session_state["use_custom_dates"] = True
                        st.session_state["use_custom_dates_checkbox"] = True
                if 'end_date_user' in main_app_config and main_app_config['end_date_user'] is not None:
                    # Parse date from string if needed
                    end_date = main_app_config['end_date_user']
                    if isinstance(end_date, str):
                        try:
                            end_date = pd.to_datetime(end_date).date()
                        except:
                            end_date = None
                    if end_date is not None:
                        st.session_state["end_date"] = end_date
                        st.session_state["use_custom_dates"] = True
                        st.session_state["use_custom_dates_checkbox"] = True
                
                # Ensure checkbox is enabled if either date is set
                if (main_app_config.get('start_date_user') is not None or 
                    main_app_config.get('end_date_user') is not None):
                    st.session_state["use_custom_dates"] = True
                    st.session_state["use_custom_dates_checkbox"] = True
                if 'start_with' in main_app_config:
                    # Handle start_with value mapping from other pages
                    start_with = main_app_config['start_with']
                    if start_with == 'first':
                        start_with = 'oldest'  # Map 'first' to 'oldest' (closest equivalent)
                    elif start_with not in ['all', 'oldest']:
                        start_with = 'all'  # Default fallback
                    st.session_state["start_with_radio_key"] = start_with
                if 'use_momentum' in main_app_config:
                    st.session_state["use_momentum"] = bool(main_app_config['use_momentum'])
                    st.session_state["use_momentum_checkbox"] = bool(main_app_config['use_momentum'])
                if 'momentum_strategy' in main_app_config:
                    # Handle momentum strategy value mapping from other pages
                    momentum_strategy = main_app_config['momentum_strategy']
                    if momentum_strategy == 'Classic':
                        momentum_strategy = 'Classic momentum'
                    elif momentum_strategy == 'Relative' or momentum_strategy == 'Relative Momentum':
                        momentum_strategy = 'Relative momentum'
                    elif momentum_strategy not in ['Classic momentum', 'Relative momentum']:
                        momentum_strategy = 'Classic momentum'  # Default fallback
                    st.session_state["momentum_strategy"] = momentum_strategy
                    st.session_state["momentum_strategy_radio"] = momentum_strategy
                if 'negative_momentum_strategy' in main_app_config:
                    # Handle negative momentum strategy value mapping from other pages
                    negative_momentum_strategy = main_app_config['negative_momentum_strategy']
                    if negative_momentum_strategy == 'Cash':
                        negative_momentum_strategy = 'Go to cash'
                    elif negative_momentum_strategy not in ['Go to cash', 'Equal weight', 'Relative momentum']:
                        negative_momentum_strategy = 'Go to cash'  # Default fallback
                    st.session_state["negative_momentum_strategy"] = negative_momentum_strategy
                    st.session_state["negative_momentum_strategy_radio"] = negative_momentum_strategy
                if 'momentum_windows' in main_app_config:
                    # Convert momentum window weights from percentage to decimal format
                    imported_windows = main_app_config['momentum_windows']
                    converted_windows = []
                    for window in imported_windows:
                        converted_window = window.copy()
                        weight = window.get('weight', 0.0)
                        if isinstance(weight, (int, float)):
                            if weight > 1.0:
                                # If weight is stored as percentage, convert to decimal
                                converted_window['weight'] = weight / 100.0
                            else:
                                # Already in decimal format, use as is
                                converted_window['weight'] = weight
                        else:
                            converted_window['weight'] = 0.0
                        converted_windows.append(converted_window)
                    st.session_state["mom_windows"] = converted_windows
                # Handle both Backtest_Engine.py format (use_beta, use_vol) and other pages format (calc_beta, calc_volatility)
                if 'calc_beta' in main_app_config:
                    st.session_state["use_beta"] = bool(main_app_config['calc_beta'])
                    st.session_state["use_beta_checkbox"] = bool(main_app_config['calc_beta'])
                elif 'use_beta' in main_app_config:
                    st.session_state["use_beta"] = bool(main_app_config['use_beta'])
                    st.session_state["use_beta_checkbox"] = bool(main_app_config['use_beta'])
                    
                if 'beta_window_days' in main_app_config:
                    st.session_state["beta_window_days"] = int(main_app_config['beta_window_days'])
                    st.session_state["beta_window_input"] = int(main_app_config['beta_window_days'])
                if 'exclude_days_beta' in main_app_config:
                    st.session_state["beta_exclude_days"] = int(main_app_config['exclude_days_beta'])
                    st.session_state["beta_exclude_input"] = int(main_app_config['exclude_days_beta'])
                    
                if 'calc_volatility' in main_app_config:
                    st.session_state["use_vol"] = bool(main_app_config['calc_volatility'])
                    st.session_state["use_vol_checkbox"] = bool(main_app_config['calc_volatility'])
                elif 'use_vol' in main_app_config:
                    st.session_state["use_vol"] = bool(main_app_config['use_vol'])
                    st.session_state["use_vol_checkbox"] = bool(main_app_config['use_vol'])
                    
                if 'vol_window_days' in main_app_config:
                    st.session_state["vol_window_days"] = int(main_app_config['vol_window_days'])
                    st.session_state["vol_window_input"] = int(main_app_config['vol_window_days'])
                if 'exclude_days_vol' in main_app_config:
                    st.session_state["vol_exclude_days"] = int(main_app_config['exclude_days_vol'])
                    st.session_state["vol_exclude_input"] = int(main_app_config['exclude_days_vol'])
    except Exception:
        # If any staging application fails, keep whatever was applied and continue
        pass
    
    # Sync date widgets with imported values
    sync_date_widgets_with_imported_values()
    
    # Clear the pending flag so widgets created afterwards won't be overwritten
    if "_import_pending" in st.session_state:
        del st.session_state["_import_pending"]
    
    # Force a rerun to update the UI widgets
    st.session_state.main_rerun_flag = True

# Handle rerun flag for smooth UI updates
if st.session_state.get('main_rerun_flag', False):
    st.session_state.main_rerun_flag = False
    st.rerun()

# Flag processing removed - buttons now execute actions immediately

# Beta and Volatility reset flag processing removed - buttons now execute immediately

# Initialize session state properly to avoid render-time modifications
if 'tickers' not in st.session_state:
    st.session_state.tickers = ["SPY", "QQQ", "GLD", "TLT"]
if 'allocs' not in st.session_state:
    num_tickers = len(st.session_state.get('tickers', ["SPY", "QQQ", "GLD", "TLT"]))
    st.session_state.allocs = [1.0 / num_tickers for _ in range(num_tickers)]  # Equal allocation as decimals (e.g., 0.25)
if 'divs' not in st.session_state:
    st.session_state.divs = [True, True, True, True]
if 'use_momentum' not in st.session_state:
    st.session_state.use_momentum = True
# Removed use_decimals initialization
if 'use_beta' not in st.session_state:
    st.session_state.use_beta = True
if 'use_vol' not in st.session_state:
    st.session_state.use_vol = True
if 'use_custom_dates' not in st.session_state:
    st.session_state.use_custom_dates = False
if 'portfolio_drag_pct' not in st.session_state:
    st.session_state.portfolio_drag_pct = 0.0
if 'mom_windows' not in st.session_state:
    st.session_state.mom_windows = [
        {'lookback': 365, 'exclude': 30, 'weight': 0.50},
        {'lookback': 180, 'exclude': 30, 'weight': 0.30},
        {'lookback': 120, 'exclude': 30, 'weight': 0.20},
    ]



# -------------------------
# Helpers
# -------------------------
def normalize_allocs():
    total = sum(st.session_state.allocs)
    if total > 0:
        # Normalize allocations to sum to 1.0 (decimal format like other pages)
        st.session_state.allocs = [a / total for a in st.session_state.allocs]

def equal_allocs():
    num_tickers = len(st.session_state.tickers)
    if num_tickers > 0:
        st.session_state.allocs = [1.0 / num_tickers for _ in range(num_tickers)]

def add_ticker_row(ticker: str = ""):
    st.session_state.tickers.append(ticker)
    st.session_state.allocs.append(0.0)
    st.session_state.divs.append(True)

def add_ticker_callback():
    """Callback for Add Ticker button to ensure immediate UI update"""
    add_ticker_row()

def remove_ticker_row(idx: int):
    if 0 <= idx < len(st.session_state.tickers):
        st.session_state.tickers.pop(idx)
        st.session_state.allocs.pop(idx)
        st.session_state.divs.pop(idx)
        # No rerun, no calculations
        # Removed 'break' as it is not valid outside a loop

def remove_ticker_callback(ticker: str):
    """Immediate ticker removal callback"""
    try:
        # Find and remove the ticker immediately
        if ticker in st.session_state.tickers:
            idx = st.session_state.tickers.index(ticker)
            st.session_state.tickers.pop(idx)
            st.session_state.allocs.pop(idx)
            st.session_state.divs.pop(idx)
            # If this was the last ticker, add an empty one
            if len(st.session_state.tickers) == 0:
                st.session_state.tickers.append("")
                st.session_state.allocs.append(0.0)
                st.session_state.divs.append(True)
            # Set rerun flag for smooth UI update
            st.session_state.main_rerun_flag = True
    except (ValueError, IndexError):
        pass

def update_ticker_callback(index: int):
    """Callback for ticker input to convert to uppercase"""
    try:
        key = f"ticker_{index}"
        val = st.session_state.get(key, None)
        if val is not None:
            # Convert the input value to uppercase
            upper_val = val.upper()
            st.session_state.tickers[index] = upper_val
            # Update the text box's state to show the uppercase value
            st.session_state[key] = upper_val
    except Exception:
        # Defensive: if index is out of range, skip silently
        pass

def add_mom_window(lookback=90, exclude=30, weight=0.0):
    st.session_state.mom_windows.append({"lookback": int(lookback), "exclude": int(exclude), "weight": float(weight)})
    # Force sync after adding a new window
    _sync_mom_widgets()

def remove_mom_window(idx: int):
    if 0 <= idx < len(st.session_state.mom_windows):
        st.session_state.mom_windows.pop(idx)

def _remove_mom_and_rerun(idx: int):
    """Helper used as an on_click callback to synchronously remove a
    momentum window, resync widget keys, and trigger a rerun. Using
    on_click avoids transient button/key mismatches when the UI changes
    rapidly and makes deletes register immediately.
    """
    try:
        remove_mom_window(idx)
    except Exception:
        pass
    try:
        _sync_mom_widgets()
    except Exception:
        pass
    # Trigger a rerun so the UI updates instantly
    try:
        st.rerun()
    except Exception:
        pass

def reset_beta_callback():
    """Callback for Reset Beta button to ensure immediate UI update"""
    st.session_state.use_beta = True
    st.session_state.beta_window_days = 365
    st.session_state.beta_exclude_days = 30
    # Update the checkbox state to match
    if "use_beta_checkbox" in st.session_state:
        st.session_state["use_beta_checkbox"] = True
    # Update the number input widget values
    st.session_state["beta_window_input"] = 365
    st.session_state["beta_exclude_input"] = 30

def reset_volatility_callback():
    """Callback for Reset Volatility button to ensure immediate UI update"""
    st.session_state.use_vol = True
    st.session_state.vol_window_days = 365
    st.session_state.vol_exclude_days = 30
    # Update the checkbox state to match
    if "use_vol_checkbox" in st.session_state:
        st.session_state["use_vol_checkbox"] = True
    # Update the number input widget values
    st.session_state["vol_window_input"] = 365
    st.session_state["vol_exclude_input"] = 30

def normalize_mom_weights():
    """Normalize momentum weights to sum to 1.0 (decimal format)"""
    total_weight = sum(w['weight'] for w in st.session_state.mom_windows)
    if total_weight > 0:
        for w in st.session_state.mom_windows:
            w['weight'] = w['weight'] / total_weight

def normalize_mom_weights_callback():
    """Callback for Normalize Weights button to ensure immediate UI update"""
    normalize_mom_weights()
    _sync_mom_widgets()

def reset_mom_windows_callback():
    """Callback for Reset Momentum Windows button to ensure immediate UI update"""
    reset_mom_windows()

def add_mom_window_callback():
    """Callback for Add Momentum Window button to ensure immediate UI update"""
    add_mom_window(lookback=90, exclude=30, weight=0.0)

def _sync_mom_widgets():
    """Ensure individual momentum window widget keys reflect current
    st.session_state.mom_windows values so number_input shows updated values."""
    try:
        for i, w in enumerate(st.session_state.get('mom_windows', [])):
            weight = float(w.get('weight', 0.0))
            # Convert decimal weight (0.0-1.0) to percentage (0-100) for display
            weight_percentage = min(100, max(0, int(round(weight * 100))))
            st.session_state[f"mom_weight_{i}"] = weight_percentage
            st.session_state[f"mom_lookback_{i}"] = int(w.get('lookback', 0))
            st.session_state[f"mom_exclude_{i}"] = int(w.get('exclude', 0))
    except Exception:
        pass

def _sync_alloc_widgets():
    """Sync allocation input widgets with st.session_state.allocs so the
    allocation number_inputs show updated values immediately.
    Convert decimal storage (0.25) to percentage display (25)."""
    try:
        for i in range(len(st.session_state.get('tickers', []))):
            key = f"alloc_input_{i}"
            if i < len(st.session_state.get('allocs', [])):
                decimal_val = st.session_state['allocs'][i]
                # Convert decimal (0.25) to percentage (25) for display
                percentage_val = decimal_val * 100
                st.session_state[key] = int(round(percentage_val))
    except Exception:
        pass

# Simple callback functions for momentum windows
def update_mom_lookback(idx):
    key = f"mom_lookback_{idx}"
    if key in st.session_state:
        st.session_state.mom_windows[idx]['lookback'] = st.session_state[key]

def update_mom_exclude(idx):
    key = f"mom_exclude_{idx}"
    if key in st.session_state:
        st.session_state.mom_windows[idx]['exclude'] = st.session_state[key]

def update_mom_weight(idx):
    key = f"mom_weight_{idx}"
    if key in st.session_state:
        st.session_state.mom_windows[idx]['weight'] = st.session_state[key] / 100.0

def update_start_with():
    # Session state automatically updates via the radio button key
    pass

def update_first_rebalance_strategy():
    st.session_state.first_rebalance_strategy = st.session_state.first_rebalance_strategy_radio_key

def update_momentum_strategy():
    st.session_state.momentum_strategy = st.session_state.momentum_strategy_radio

def update_negative_momentum_strategy():
    st.session_state.negative_momentum_strategy = st.session_state.negative_momentum_strategy_radio

def paste_json_callback():
    try:
        json_data = json.loads(st.session_state.backtest_engine_paste_json_text)
        
        # Clear widget keys for portfolio settings to force re-initialization
        widget_keys_to_clear = [
            "initial_value_input", "added_amount_input", "rebalancing_frequency_widget", 
            "added_frequency_widget", "collect_dividends_as_cash_checkbox", "first_rebalance_strategy_radio_key",
            "start_with_radio_key", "momentum_strategy_radio", "negative_momentum_strategy_radio"
        ]
        for key in widget_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Handle stocks field - convert from stocks format to legacy format
        stocks = json_data.get('stocks', [])
        if stocks:
            # Clear existing tickers and widget keys first
            st.session_state.tickers = []
            st.session_state.allocs = []
            st.session_state.divs = []
            
            # Clear all ticker widget keys to prevent UI interference
            for key in list(st.session_state.keys()):
                if key.startswith("ticker_") or key.startswith("alloc_input_") or key.startswith("divs_checkbox_"):
                    del st.session_state[key]
            
            tickers = []
            allocations = []
            dividends = []
            
            for stock in stocks:
                if stock.get('ticker'):
                    tickers.append(stock['ticker'])
                    # Get allocation (convert from percentage to decimal if needed)
                    allocation = stock.get('allocation', 0.0)
                    if isinstance(allocation, (int, float)):
                        if allocation > 1.0:
                            # Convert percentage to decimal
                            allocations.append(allocation / 100.0)
                        else:
                            # Already in decimal format
                            allocations.append(allocation)
                    else:
                        allocations.append(0.0)
                    # Get dividend setting
                    dividends.append(stock.get('include_dividends', True))
            
            st.session_state.tickers = tickers
            st.session_state.allocs = allocations
            st.session_state.divs = dividends
        
        # Handle basic portfolio settings
        if 'name' in json_data:
            st.session_state.portfolio_name = json_data['name']
            st.session_state.portfolio_name_input = json_data['name']
        if 'initial_value' in json_data:
            val = json_data['initial_value']
            try:
                float_val = float(val)
                st.session_state.initial_value = float_val
                st.session_state.initial_value_input_int = int(float_val)
                st.session_state.initial_value_input = float_val
            except Exception:
                pass
        if 'added_amount' in json_data:
            val = json_data['added_amount']
            try:
                float_val = float(val)
                st.session_state.added_amount = float_val
                st.session_state.added_amount_input_int = int(float_val)
                st.session_state.added_amount_input = float_val
            except Exception:
                pass
        if 'rebalancing_frequency' in json_data:
            st.session_state.rebalancing_frequency = json_data['rebalancing_frequency']
            st.session_state.rebalancing_frequency_widget = json_data['rebalancing_frequency']
        if 'added_frequency' in json_data:
            st.session_state.added_frequency = json_data['added_frequency']
            st.session_state.added_frequency_widget = json_data['added_frequency']
        if 'collect_dividends_as_cash' in json_data:
            st.session_state['_import_collect_dividends_as_cash'] = bool(json_data['collect_dividends_as_cash'])
        if 'start_date_user' in json_data:
            st.session_state['_import_start_date'] = json_data['start_date_user']
        if 'end_date_user' in json_data:
            st.session_state['_import_end_date'] = json_data['end_date_user']
        if 'benchmark_ticker' in json_data:
            st.session_state['_import_benchmark_ticker'] = json_data['benchmark_ticker']
        
        # Handle momentum settings
        if 'use_momentum' in json_data:
            st.session_state['_import_use_momentum'] = bool(json_data['use_momentum'])
        if 'momentum_strategy' in json_data:
            st.session_state['_import_momentum_strategy'] = json_data['momentum_strategy']
        if 'negative_momentum_strategy' in json_data:
            st.session_state['_import_negative_momentum_strategy'] = json_data['negative_momentum_strategy']
        if 'momentum_windows' in json_data:
            st.session_state['_import_mom_windows'] = json_data['momentum_windows']
        
        # Handle beta and volatility settings
        if 'calc_beta' in json_data:
            st.session_state['_import_use_beta'] = bool(json_data['calc_beta'])
        if 'beta_window_days' in json_data:
            st.session_state['_import_beta_window_days'] = int(json_data['beta_window_days'])
        if 'exclude_days_beta' in json_data:
            st.session_state['_import_beta_exclude_days'] = int(json_data['exclude_days_beta'])
        if 'calc_volatility' in json_data:
            st.session_state['_import_use_vol'] = bool(json_data['calc_volatility'])
        if 'vol_window_days' in json_data:
            st.session_state['_import_vol_window_days'] = int(json_data['vol_window_days'])
        if 'exclude_days_vol' in json_data:
            st.session_state['_import_vol_exclude_days'] = int(json_data['exclude_days_vol'])
        
        # Handle portfolio drag
        if 'portfolio_drag_pct' in json_data:
            st.session_state['_import_portfolio_drag_pct'] = float(json_data['portfolio_drag_pct'])
        
        # Handle global start_with setting from imported JSON
        if 'start_with' in json_data:
            # Handle start_with value mapping from other pages
            start_with = json_data['start_with']
            if start_with == 'first':
                start_with = 'oldest'  # Map 'first' to 'oldest' (closest equivalent)
            elif start_with not in ['all', 'oldest']:
                start_with = 'all'  # Default fallback
            st.session_state['_import_start_with'] = start_with
        
        # Handle first rebalance strategy from imported JSON
        if 'first_rebalance_strategy' in json_data:
            st.session_state['_import_first_rebalance_strategy'] = json_data['first_rebalance_strategy']
        
        # Set the pending flag to trigger import processing
        st.session_state['_import_pending'] = True
        
        st.success("Portfolio configuration updated from JSON (Backtest Engine page).")
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check the text and try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def clear_outputs():
    st.session_state.fig_dict = None
    st.session_state.console = ""
    st.session_state.error = None
    st.session_state.last_run_time = None
    st.session_state.yearly_table = None
    st.session_state.stats_with_additions = None
    st.session_state.stats_without_additions = None
    st.session_state.last_rebalance_allocs = None
    st.session_state.current_allocs = None
    st.session_state.rebalance_alloc_table = None
    st.session_state.rebalance_metrics_table = None
    st.session_state.performance_table = None
    st.session_state.portfolio_value_with_additions = None
    st.session_state.portfolio_value_without_additions = None

def reset_mom_windows():
    # Clear ALL existing momentum window widget keys first
    for k in list(st.session_state.keys()):
        try:
            if k.startswith("mom_weight_") or k.startswith("mom_lookback_") or k.startswith("mom_exclude_") or k.startswith("remove_mom_"):
                st.session_state.pop(k, None)
        except Exception:
            pass
    
    # Reset momentum windows to default
    st.session_state.mom_windows = [
        {"lookback": 365, "exclude": 30, "weight": 0.50},
        {"lookback": 180, "exclude": 30, "weight": 0.30},
        {"lookback": 120, "exclude": 30, "weight": 0.20},
    ]
    # Clear any pending momentum window flags to prevent conflicts
    st.session_state.add_mom_window_flag = False
    st.session_state.remove_mom_window_idx = None
    # Force sync to update widget display
    _sync_mom_widgets()

def clear_everything():
    # Clear common transient/run-related and staging keys first
    transient_prefixes = (
        "alloc_input_",
        "mom_weight_",
        "mom_lookback_",
        "mom_exclude_",
        "ticker_",
        "divs_",
        "remove_",
        "remove_mom_",
        "_import_",
    )
    for k in list(st.session_state.keys()):
        try:
            if any(k.startswith(p) for p in transient_prefixes):
                st.session_state.pop(k, None)
        except Exception:
            pass

    # Remove any pending or running flags and large objects
    for k in (
        "_pending_backtest_params", "_run_requested", "running", "_import_pending",
        "fig_dict", "console", "error", "last_run_time",
        "yearly_table", "performance_table", "last_rebalance_allocs",
        "current_allocs", "rebalance_alloc_table", "rebalance_metrics_table",
        "portfolio_value_with_additions", "portfolio_value_without_additions",
        "cash_flows_with_additions", "cash_flows_without_additions", "_last_ui_change",
    ):
        st.session_state.pop(k, None)

    # Reinitialize canonical input/default values
    st.session_state.tickers = ["SPY", "QQQ", "GLD", "TLT"]
    # Allocations stored as decimals (like other pages) 
    st.session_state.allocs = [1.0 / len(st.session_state.tickers) for _ in st.session_state.tickers]
    st.session_state.divs = [True for _ in st.session_state.tickers]
    st.session_state.mom_windows = [
        {"lookback": 365, "exclude": 30, "weight": 0.50},
        {"lookback": 180, "exclude": 30, "weight": 0.30},
        {"lookback": 120, "exclude": 30, "weight": 0.20},
    ]
    st.session_state.use_momentum = True
    st.session_state.use_relative_momentum = False
    st.session_state.momentum_strategy = "Classic momentum"
    st.session_state.negative_momentum_strategy = "Go to cash"
    st.session_state.use_beta = True
    st.session_state.beta_window_days = 365
    st.session_state.beta_exclude_days = 30
    st.session_state.use_vol = True
    st.session_state.vol_window_days = 365
    st.session_state.vol_exclude_days = 30
    st.session_state.use_custom_dates = False
    st.session_state.start_date = None
    st.session_state.end_date = date.today()
    # Removed use_decimals reset
    st.session_state.momentum_strategy_radio = "Classic momentum"
    st.session_state.negative_momentum_strategy_radio = "Go to cash"
    st.session_state.start_with_radio_key = "oldest"
    # Portfolio settings
    st.session_state.initial_value = 10000
    st.session_state.added_amount = 1000
    st.session_state.rebalancing_frequency = "Monthly"
    st.session_state.added_frequency = "Monthly"
    # Reset widget keys to match
    st.session_state["initial_value_input"] = 10000
    st.session_state["initial_value_input_int"] = 10000
    st.session_state["added_amount_input"] = 1000
    st.session_state["added_amount_input_int"] = 1000
    st.session_state["rebalancing_frequency_widget"] = "Monthly"
    st.session_state["added_frequency_widget"] = "Monthly"
    # Portfolio drag stored as float
    st.session_state.portfolio_drag_pct = 0.0

    # Ensure widget-backed keys reflect the new defaults
    try:
        _sync_mom_widgets()
    except Exception:
        pass
    try:
        _sync_alloc_widgets()
    except Exception:
        pass

def reset_assets_only():
    """Reset only asset-related inputs: tickers, allocations, and dividend flags.
    Remove widget-backed keys so checkboxes/text inputs are recreated cleanly.
    """
    # Remove widget-backed keys for tickers, allocations, divs, and remove buttons
    for k in list(st.session_state.keys()):
        try:
            if k.startswith("ticker_") or k.startswith("alloc_input_") or k.startswith("divs_checkbox_") or k.startswith("remove_"):
                st.session_state.pop(k, None)
        except Exception:
            pass

    # Reinitialize canonical asset inputs to their first-run defaults
    st.session_state.tickers = ["SPY", "QQQ", "GLD", "TLT"]
    # Allocations stored as decimals (like other pages)
    st.session_state.allocs = [1.0 / len(st.session_state.tickers) for _ in st.session_state.tickers]
    st.session_state.divs = [True for _ in st.session_state.tickers]

    # Reset any transient asset flags so the section behaves as on first run
    st.session_state.add_ticker_flag = False
    st.session_state.remove_ticker_idx = None

    # Also set widget-backed keys so checkboxes/text inputs reflect the reset immediately
    try:
        for i, t in enumerate(st.session_state.tickers):
            st.session_state[f"divs_checkbox_{i}"] = True
            st.session_state[f"ticker_{i}"] = t
            # Convert decimal allocation (0.25) to percentage for display (25)
            decimal_alloc = st.session_state.allocs[i] 
            percentage_alloc = decimal_alloc * 100
            st.session_state[f"alloc_input_{i}"] = int(round(percentage_alloc))
    except Exception:
        pass

    # Sync widgets so inputs show the new values immediately
    try:
        _sync_alloc_widgets()
    except Exception:
        pass
    try:
        _sync_mom_widgets()
    except Exception:
        pass

    # Update JSON preview
    try:
        # Create config using the new format
        config = {
            'name': st.session_state.get("portfolio_name", "Main Portfolio"),
            'stocks': [
                {
                    'ticker': ticker,
                    'allocation': alloc,
                    'include_dividends': div
                }
                for ticker, alloc, div in zip(
                    st.session_state.get("tickers", []),
                    st.session_state.get("allocs", []),
                    st.session_state.get("divs", [])
                )
            ],
            'benchmark_ticker': st.session_state.get("benchmark_ticker", "^GSPC"),
            'initial_value': st.session_state.get("initial_value", 10000),
            'added_amount': st.session_state.get("added_amount", 0),
            'added_frequency': st.session_state.get("added_frequency", "Monthly"),
            'rebalancing_frequency': st.session_state.get("rebalancing_frequency", "Monthly"),
            'start_date_user': st.session_state.get("start_date", None),
            'end_date_user': st.session_state.get("end_date", None),
            'start_with': st.session_state.get("start_with_radio_key", "oldest"),
            'first_rebalance_strategy': st.session_state.get("first_rebalance_strategy", "rebalancing_date"),
            'use_momentum': st.session_state.get("use_momentum", False),
            'momentum_strategy': st.session_state.get("momentum_strategy", "Classic momentum"),
            'negative_momentum_strategy': st.session_state.get("negative_momentum_strategy", "Go to cash"),
            'momentum_windows': st.session_state.get("mom_windows", []),
            'calc_beta': st.session_state.get("use_beta", False),
            'calc_volatility': st.session_state.get("use_vol", False),
            'beta_window_days': st.session_state.get("beta_window_days", 365),
            'exclude_days_beta': st.session_state.get("beta_exclude_days", 30),
            'vol_window_days': st.session_state.get("vol_window_days", 365),
            'exclude_days_vol': st.session_state.get("vol_exclude_days", 30),
            'collect_dividends_as_cash': st.session_state.get("collect_dividends_as_cash", False),
            'portfolio_drag_pct': float(st.session_state.get("portfolio_drag_pct", 0.0)),
        }
        st.session_state['backtest_json'] = json.dumps(config, indent=2, default=str)
    except Exception:
        st.session_state['backtest_json'] = '{}'
def clear_dates():
    st.session_state.start_date = None
    st.session_state.end_date = date.today()
    st.session_state.use_custom_dates = False
    # Update the checkbox state to match
    if "use_custom_dates_checkbox" in st.session_state:
        st.session_state["use_custom_dates_checkbox"] = False

def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# Helper functions removed - checkboxes now use the same pattern as working Multi backtest and Allocations pages



def create_pie_chart(allocations, title):
    if not allocations:
        return go.Figure().update_layout(title_text=title, annotations=[dict(text='No assets allocated', x=0.5, y=0.5, showarrow=False)])
    
    # Filter out allocations with 0%
    valid_allocations = {k: v for k, v in allocations.items() if v > 0.0001}
    
    if not valid_allocations:
        return go.Figure().update_layout(title_text=title, annotations=[dict(text='No assets allocated', x=0.5, y=0.5, showarrow=False)])

    labels = list(valid_allocations.keys())
    values = list(valid_allocations.values())
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title_text=title, showlegend=True)
    return fig


# -------------------------
# UI
# -------------------------

# Portfolio Name - created after JSON import processing
if 'portfolio_name' not in st.session_state:
    st.session_state.portfolio_name = "Main Portfolio"
# Initialize widget key with session state value
if "portfolio_name_input" not in st.session_state:
    st.session_state["portfolio_name_input"] = st.session_state.portfolio_name
portfolio_name = st.text_input("Portfolio Name", key="portfolio_name_input")
st.session_state.portfolio_name = portfolio_name

# Sidebar
with st.sidebar:
    st.title("Backtest Controls")

    # Small tolerance for allocation/weight checks (percentage points)
    # Tolerance for momentum weights (percentage points)
    # Use 1.0 => 1% tolerance for both momentum-window total and ticker allocation.
    _TOTAL_TOL = 1.0
    # Tolerance for ticker Total Allocation so rounding (e.g. 33.33*3=99.99)
    # does not block running; set to 1% as requested.
    _ALLOC_TOL = 1.0

    # Main Reset button removed per user request
    # 1. Use Momentum Strategy at the top - Completely isolated
    col_mom, col_dec = st.columns([2, 1])
    with col_mom:
        # Use the same pattern as working Multi backtest and Allocations pages
        if "use_momentum_checkbox" not in st.session_state:
            st.session_state["use_momentum_checkbox"] = st.session_state.use_momentum
        st.checkbox(
            "Use Momentum Strategy", 
            help="If unchecked, the backtest will rebalance to the initial allocations.",
            key="use_momentum_checkbox",
            on_change=lambda: setattr(st.session_state, 'use_momentum', st.session_state["use_momentum_checkbox"])
        )
        
    with col_dec:
        # Use the same pattern as working Multi backtest and Allocations pages
            # Removed use_decimals_checkbox initialization
        # Removed decimal checkbox - keeping layout consistent
        st.markdown("&nbsp;", unsafe_allow_html=True)  # Spacer to maintain layout


    # Session state initialization moved to top of script to avoid render-time conflicts

    # 2. Asset section
    st.header("Assets")
    col_asset_btns = st.columns([1, 1, 1])
    with col_asset_btns[0]:
        if st.button("Reset Assets", help="Reset asset tickers and allocations", key="reset_assets_btn"):
            reset_assets_only()
    if not st.session_state.use_momentum:
        with col_asset_btns[1]:
            if st.button("Equal Allocation", help="Set all allocations to equal weights", key="equal_alloc_btn"):
                equal_allocs()
                _sync_alloc_widgets()
        with col_asset_btns[2]:
            if st.button("Normalize Allocation", help="Normalize allocations to sum to 1", key="normalize_alloc_btn"):
                normalize_allocs()
                _sync_alloc_widgets()
    # Annualized portfolio drag (fees or negative = added benefit). Enter percent per year.

    def _on_portfolio_drag_changed():
        # Ensure the typed or clicked value is stored as float in session_state
        try:
            st.session_state.portfolio_drag_pct = float(st.session_state.get('portfolio_drag_pct', 0.0))
        except Exception:
            st.session_state.portfolio_drag_pct = 0.0
        # Update the JSON preview used for copy/paste/export so the change appears live
        try:
            # Create config using the new format
            config = {
                'name': st.session_state.get("portfolio_name", "Main Portfolio"),
                'stocks': [
                    {
                        'ticker': ticker,
                        'allocation': alloc,
                        'include_dividends': div
                    }
                    for ticker, alloc, div in zip(
                        st.session_state.get("tickers", []),
                        st.session_state.get("allocs", []),
                        st.session_state.get("divs", [])
                    )
                ],
                'benchmark_ticker': st.session_state.get("benchmark_ticker", "^GSPC"),
                'initial_value': st.session_state.get("initial_value", 10000),
                'added_amount': st.session_state.get("added_amount", 0),
                'added_frequency': st.session_state.get("added_frequency", "Monthly"),
                'rebalancing_frequency': st.session_state.get("rebalancing_frequency", "Monthly"),
                'start_date_user': st.session_state.get("start_date", None),
                'end_date_user': st.session_state.get("end_date", None),
                'start_with': st.session_state.get("start_with_radio_key", "oldest"),
                'first_rebalance_strategy': st.session_state.get("first_rebalance_strategy", "rebalancing_date"),
                'use_momentum': st.session_state.get("use_momentum", False),
                'momentum_strategy': st.session_state.get("momentum_strategy", "Classic momentum"),
                'negative_momentum_strategy': st.session_state.get("negative_momentum_strategy", "Go to cash"),
                'momentum_windows': st.session_state.get("mom_windows", []),
                'calc_beta': st.session_state.get("use_beta", False),
                'calc_volatility': st.session_state.get("use_vol", False),
                'beta_window_days': st.session_state.get("beta_window_days", 365),
                'exclude_days_beta': st.session_state.get("beta_exclude_days", 30),
                'vol_window_days': st.session_state.get("vol_window_days", 365),
                'exclude_days_vol': st.session_state.get("vol_exclude_days", 30),
                'collect_dividends_as_cash': st.session_state.get("collect_dividends_as_cash", False),
                'portfolio_drag_pct': float(st.session_state.get("portfolio_drag_pct", 0.0)),
            }
            st.session_state['backtest_json'] = json.dumps(config, indent=2, default=str)
        except Exception:
            # If config creation fails during initial runs, ignore
            pass

    with st.expander("Portfolio drag / fees (annual %)", expanded=False):
        # Simple layout: only keep the numeric input for manual edits.
        # The user requested removal of small +/- buttons, so no buttons are shown here.
        # Do not pass `value=` here because we manage the initial value via
        # `st.session_state['portfolio_drag_pct']` to avoid Streamlit warnings
        # about a widget being created with a default and also set via session state.
        st.number_input(
            "Annual drag (%) — positive = fee, negative = benefit",
            step=0.1, format="%.2f",
            help="Enter an annualized drag (e.g. 1.0 for 1% annual drag). Negative values act as additions.",
            key='portfolio_drag_pct', on_change=_on_portfolio_drag_changed
        )
    

    
    for i in range(len(st.session_state.tickers)):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 0.2])
        with col1:
            # Initialize ticker widget key with session state value
            ticker_key = f"ticker_{i}"
            if ticker_key not in st.session_state:
                st.session_state[ticker_key] = st.session_state.tickers[i]
            
            # Always update tickers from text_input
            ticker_val = st.text_input(
                f"Ticker {i+1}", key=ticker_key, on_change=update_ticker_callback, args=(i,)
            )
        with col3:
            # Use the same pattern as working Multi backtest and Allocations pages
            div_key = f"divs_checkbox_{i}"
            if div_key not in st.session_state:
                st.session_state[div_key] = st.session_state.divs[i]
            st.checkbox("Include Dividends", key=div_key, on_change=lambda i=i: setattr(st.session_state, f'divs[{i}]', st.session_state[div_key]))
        with col4:
            if st.button("x", key=f"remove_ticker_{i}", help="Remove this ticker", on_click=remove_ticker_callback, args=(st.session_state.tickers[i],)):
                pass
        # Only show allocation if NOT using momentum
        if not st.session_state.use_momentum:
            with col2:
                # Initialize allocation widget key with session state value
                alloc_key = f"alloc_input_{i}"
                if alloc_key not in st.session_state:
                    # Convert decimal storage (0.25) to percentage display (25)
                    decimal_alloc = st.session_state.allocs[i]
                    percentage_alloc = decimal_alloc * 100
                    # Ensure percentage is within valid range for the widget
                    percentage_alloc = max(0, min(100, percentage_alloc))
                    st.session_state[alloc_key] = percentage_alloc
                
                # Simple allocation input - always integer step
                st.number_input(
                    f"Allocation {i+1} (%)", min_value=0, max_value=100, step=1, key=alloc_key,
                    help="Enter allocation as a percentage.",
                    format="%d",
                    on_change=lambda: setattr(st.session_state, 'allocs', [float(st.session_state[f"alloc_input_{j}"]) / 100.0 if f"alloc_input_{j}" in st.session_state else st.session_state.allocs[j] for j in range(len(st.session_state.allocs))])
                )
    if not st.session_state.use_momentum:
        total_alloc = sum(st.session_state.allocs)  # This is now in decimal format (should sum to 1.0)
        total_alloc_percentage = total_alloc * 100   # Convert to percentage for display
        if abs(total_alloc - 1.0) <= (_ALLOC_TOL / 100.0):  # Convert tolerance to decimal
            # When effectively 100%, show a clean "100%" without decimal places
            st.markdown('<div style="background-color:#004d00;color:white;padding:8px;border-radius:6px;">Total Allocation: 100%</div>', unsafe_allow_html=True)
        else:
            st.info(f"Total Allocation: {total_alloc_percentage:.2f}%")

    if st.button("Add Ticker", help="Add another asset to the list", on_click=add_ticker_callback):
        pass

    # 3. Initial portfolio value and periodic cash addition
    st.markdown("---")
    st.header("Portfolio Settings")
    # Helper placeholder: widget on_change handlers will cause a rerun so the
    # JSON text_area (rendered directly each run) updates automatically.
    def update_backtest_json():
        return
    
    # Synchronize widget keys with persistent session state
    if "initial_value_input" not in st.session_state:
        st.session_state["initial_value_input"] = st.session_state.initial_value
    if "added_amount_input" not in st.session_state:
        st.session_state["added_amount_input"] = st.session_state.added_amount
    if "rebalancing_frequency_widget" not in st.session_state:
        st.session_state["rebalancing_frequency_widget"] = st.session_state.rebalancing_frequency
    if "added_frequency_widget" not in st.session_state:
        st.session_state["added_frequency_widget"] = st.session_state.added_frequency
    if "start_with_radio_key" not in st.session_state:
        st.session_state["start_with_radio_key"] = "oldest"
    if "momentum_strategy_radio" not in st.session_state:
        st.session_state["momentum_strategy_radio"] = st.session_state.get("momentum_strategy", "Classic momentum")
    if "negative_momentum_strategy_radio" not in st.session_state:
        st.session_state["negative_momentum_strategy_radio"] = st.session_state.get("negative_momentum_strategy", "Go to cash")
    if "use_momentum_checkbox" not in st.session_state:
        st.session_state["use_momentum_checkbox"] = st.session_state.use_momentum
    # Removed use_decimals_checkbox initialization
    if "use_beta_checkbox" not in st.session_state:
        st.session_state["use_beta_checkbox"] = st.session_state.use_beta
    if "use_vol_checkbox" not in st.session_state:
        st.session_state["use_vol_checkbox"] = st.session_state.use_vol
    if "use_custom_dates_checkbox" not in st.session_state:
        st.session_state["use_custom_dates_checkbox"] = st.session_state.use_custom_dates
    if "beta_window_input" not in st.session_state:
        st.session_state["beta_window_input"] = st.session_state.beta_window_days
    if "beta_exclude_input" not in st.session_state:
        st.session_state["beta_exclude_input"] = st.session_state.beta_exclude_days
    if "vol_window_input" not in st.session_state:
        st.session_state["vol_window_input"] = st.session_state.vol_window_days
    if "vol_exclude_input" not in st.session_state:
        st.session_state["vol_exclude_input"] = st.session_state.vol_exclude_days
    
    col_init, col_add = st.columns(2)
    with col_init:
        # Simple initial value input - always integer step
        initial_value = st.number_input(
            "Initial Portfolio Value", min_value=1000, step=1000, format="%d", key="initial_value_input",
            on_change=lambda: setattr(st.session_state, 'initial_value', st.session_state["initial_value_input"])
        )
    with col_add:
        # Simple added amount input - always integer step
        added_amount = st.number_input(
            "Periodic Cash Addition", min_value=0, step=1000, format="%d", key="added_amount_input",
            help="Set to 0 for no additions.", on_change=lambda: setattr(st.session_state, 'added_amount', st.session_state["added_amount_input"])
        )

    # 4. Rebalancing frequency and cash addition frequency side by side (switched order)
    st.markdown("")
    col_freq1, col_freq2 = st.columns(2)
    added_freq_options = ["Never", "Buy & Hold", "Buy & Hold (Target)", "Weekly", "Biweekly", "Monthly", "Quarterly", "Semiannually", "Annually"]
    rebalancing_freq_options = ["Never", "Buy & Hold", "Buy & Hold (Target)", "Weekly", "Biweekly", "Monthly", "Quarterly", "Semiannually", "Annually"]
    # Helper placeholder: widget on_change handlers will cause a rerun so the
    # JSON text_area (rendered directly each run) updates automatically.
    def update_backtest_json():
        return
    with col_freq1:
        rebalancing_frequency = st.selectbox(
            "Rebalancing Frequency", rebalancing_freq_options,
            help="How often to rebalance the portfolio. This is where the momentum strategy is applied. 'Buy & Hold' reinvests cash immediately using current proportions. 'Buy & Hold (Target)' reinvests cash immediately using target allocations. Cash from dividends (if 'Collect Dividends as Cash' is enabled) will be available for rebalancing.",
            key="rebalancing_frequency_widget", on_change=lambda: setattr(st.session_state, 'rebalancing_frequency', st.session_state["rebalancing_frequency_widget"])
        )
    with col_freq2:
        added_frequency = st.selectbox(
            "Cash Addition Frequency", added_freq_options,
            help="How often to add cash to the portfolio. 'Buy & Hold' reinvests cash immediately using current proportions. 'Buy & Hold (Target)' reinvests cash immediately using target allocations.",
            key="added_frequency_widget", on_change=lambda: setattr(st.session_state, 'added_frequency', st.session_state["added_frequency_widget"])
        )
    # Prevent cash from being added if frequency is Never
    if added_frequency == "Never":
        added_amount = 0

    # Dividend handling option
    if "collect_dividends_as_cash_checkbox" not in st.session_state:
        st.session_state["collect_dividends_as_cash_checkbox"] = st.session_state.collect_dividends_as_cash
    st.checkbox(
        "Collect Dividends as Cash", 
        key="collect_dividends_as_cash_checkbox",
        help="When enabled, dividends are collected as cash instead of being automatically reinvested in the stock. This cash will be available for rebalancing.",
        on_change=lambda: setattr(st.session_state, 'collect_dividends_as_cash', st.session_state["collect_dividends_as_cash_checkbox"])
    )

    st.markdown("---")
    # Time & Data Options section (only once, after portfolio settings)
    st.header("Time & Data Options")
    # Use the same pattern as working Multi backtest and Allocations pages
    # Ensure checkbox state is properly synchronized
    if "use_custom_dates_checkbox" not in st.session_state:
        st.session_state["use_custom_dates_checkbox"] = st.session_state.use_custom_dates
    else:
        # Keep checkbox in sync with the main state
        st.session_state["use_custom_dates_checkbox"] = st.session_state.use_custom_dates
    
    # Debug: Show current checkbox state
    if st.session_state.get('_import_pending', False):
        st.info(f"Debug: use_custom_dates={st.session_state.get('use_custom_dates', 'Not set')}, checkbox={st.session_state.get('use_custom_dates_checkbox', 'Not set')}")
    
    st.checkbox(
        "Use Custom Date Range", 
        key="use_custom_dates_checkbox",
        on_change=lambda: setattr(st.session_state, 'use_custom_dates', st.session_state["use_custom_dates_checkbox"])
    )

    start_date_user = None
    end_date_user = None

    if st.session_state.use_custom_dates:
        col_start, col_end, col_clear_dates = st.columns([1, 1, 0.5])
        with col_start:
            # Initialize widget key with session state value
            if "start_date" not in st.session_state:
                st.session_state["start_date"] = st.session_state.start_date if st.session_state.start_date else date(2010, 1, 1)
            # Let Streamlit manage `st.session_state['start_date']` via the widget key
            # Remove custom on_change to avoid preemptive reruns; Streamlit will
            # update session state automatically when the user selects a date.
            st.date_input(
                "Start Date",
                min_value=date(1900, 1, 1), key='start_date'
            )
        with col_end:
            # Initialize widget key with session state value
            if "end_date" not in st.session_state:
                st.session_state["end_date"] = st.session_state.end_date if st.session_state.end_date else date.today()
            # Let Streamlit manage `st.session_state['end_date']` via the widget key
            st.date_input(
                "End Date", key='end_date'
            )
        with col_clear_dates:
            st.markdown("<br>", unsafe_allow_html=True) # Spacer for alignment
            st.button("Clear Dates", on_click=clear_dates)


        start_date_user = st.session_state.start_date
        end_date_user = st.session_state.end_date


    st.markdown("---")

    # 5. How to handle assets with different start dates?
    start_with_options = ["all", "oldest"]
    if "start_with_radio_key" not in st.session_state:
        st.session_state["start_with_radio_key"] = "oldest"
    start_with = st.radio(
        "How to handle assets with different start dates?",
        start_with_options,
        format_func=lambda x: "Start when ALL assets are available" if x == "all" else "Start with OLDEST asset",
        help="""
        **All:** Starts the backtest when all selected assets are available.
        **Oldest:** Starts at the oldest date of any asset and adds assets as they become available.
        """,
        key="start_with_radio_key",
        on_change=update_start_with
    )

    # 6. When should the first rebalancing occur?
    first_rebalance_options = ["rebalancing_date", "momentum_window_complete"]
    if "first_rebalance_strategy_radio_key" not in st.session_state:
        st.session_state["first_rebalance_strategy_radio_key"] = st.session_state.get("first_rebalance_strategy", "rebalancing_date")
    
    first_rebalance_strategy = st.radio(
        "When should the first rebalancing occur?",
        first_rebalance_options,
        format_func=lambda x: "First rebalance on rebalancing date" if x == "rebalancing_date" else "First rebalance when momentum window complete",
        help="""
        **First rebalance on rebalancing date:** Start rebalancing immediately when possible.
        **First rebalance when momentum window complete:** Wait for the largest momentum window to complete before first rebalance.
        """,
        key="first_rebalance_strategy_radio_key",
        on_change=update_first_rebalance_strategy
    )

    # Always preserve radio selections in session state
    momentum_options = ["Classic momentum", "Relative momentum"]
    negative_options = ["Go to cash", "Equal weight", "Relative momentum"]
    if "momentum_strategy" not in st.session_state:
        st.session_state.momentum_strategy = momentum_options[0]
    if "negative_momentum_strategy" not in st.session_state:
        st.session_state.negative_momentum_strategy = negative_options[0]

    if st.session_state.use_momentum:
        st.header("Momentum & Rebalancing")
        # Ensure momentum_strategy_radio has a valid value
        if "momentum_strategy_radio" not in st.session_state:
            st.session_state["momentum_strategy_radio"] = st.session_state.get("momentum_strategy", "Classic momentum")
        
        # Validate and fix the value if it's not in the expected options
        current_value = st.session_state["momentum_strategy_radio"]
        valid_options = ["Classic momentum", "Relative momentum"]
        if current_value not in valid_options:
            # Map common variations to valid options
            if current_value in ["Classic", "Classic momentum"]:
                st.session_state["momentum_strategy_radio"] = "Classic momentum"
            elif current_value in ["Relative", "Relative Momentum", "Relative momentum"]:
                st.session_state["momentum_strategy_radio"] = "Relative momentum"
            else:
                # Default fallback
                st.session_state["momentum_strategy_radio"] = "Classic momentum"
        
        selected_momentum = st.radio(
            "Momentum Strategy (when not all assets have negative):",
            ["Classic momentum", "Relative momentum"],
            key="momentum_strategy_radio",
            help="Choose how to allocate when at least one asset has positive momentum.",
            on_change=update_momentum_strategy
        )
        # Ensure negative_momentum_strategy_radio has a valid value
        if "negative_momentum_strategy_radio" not in st.session_state:
            st.session_state["negative_momentum_strategy_radio"] = st.session_state.get("negative_momentum_strategy", "Go to cash")
        
        # Validate and fix the value if it's not in the expected options
        current_negative_value = st.session_state["negative_momentum_strategy_radio"]
        valid_negative_options = ["Go to cash", "Equal weight", "Relative momentum"]
        if current_negative_value not in valid_negative_options:
            # Map common variations to valid options
            if current_negative_value in ["Cash", "Go to cash"]:
                st.session_state["negative_momentum_strategy_radio"] = "Go to cash"
            elif current_negative_value in ["Equal weight", "Equal Weight"]:
                st.session_state["negative_momentum_strategy_radio"] = "Equal weight"
            elif current_negative_value in ["Relative momentum", "Relative Momentum"]:
                st.session_state["negative_momentum_strategy_radio"] = "Relative momentum"
            else:
                # Default fallback
                st.session_state["negative_momentum_strategy_radio"] = "Go to cash"
        
        selected_negative = st.radio(
            "If all assets have negative momentum:",
            ["Go to cash", "Equal weight", "Relative momentum"],
            key="negative_momentum_strategy_radio",
            help="Choose what to do when all assets have negative momentum.",
            on_change=update_negative_momentum_strategy
        )
        st.subheader("Momentum Windows")
        # Buttons for momentum window management
        mom_btn_col1, mom_btn_col2, mom_btn_col3 = st.columns(3)
        with mom_btn_col1:
            if st.button("Add Momentum Window", help="Add a new momentum window definition", key="add_mom_window_btn", on_click=add_mom_window_callback):
                pass
        with mom_btn_col2:
            if st.button("Normalize Weights", help="Normalize all momentum weights to sum to 100%", key="normalize_mom_weights_btn", on_click=normalize_mom_weights_callback):
                pass
        with mom_btn_col3:
            if st.button("Reset Momentum Windows", help="Reset momentum windows to original", key="reset_mom_windows_btn", on_click=reset_mom_windows_callback):
                pass

        col_l, col_e, col_w, col_x = st.columns([1, 1, 1, 0.2])
        with col_l: st.caption("Lookback (days)")
        with col_e: st.caption("Exclude (days)")
        with col_w: st.caption("Weight (%)")
        with col_x: st.caption("")
        # Ensure widget-backed session keys reflect current momentum window values
        try:
            _sync_mom_widgets()
        except Exception:
            pass

        # Render each window row
        for i, window in enumerate(st.session_state.mom_windows):
            col_win1, col_win2, col_win3, col_win4 = st.columns([1, 1, 1, 0.2])
            # Prepare widget-backed session_state keys to safe, clamped values so
            # st.number_input never receives a value below its min (avoids
            # StreamlitValueBelowMinError when windows are added/removed quickly).
            lb_key = f"mom_lookback_{i}"
            ex_key = f"mom_exclude_{i}"
            wt_key = f"mom_weight_{i}"
            # Lookback: minimum 1 day
            try:
                if lb_key not in st.session_state:
                    st.session_state[lb_key] = int(window.get('lookback', 1)) if window is not None else 1
                else:
                    # coerce and clamp
                    val = st.session_state[lb_key]
                    st.session_state[lb_key] = max(1, int(val) if val is not None else 1)
            except Exception:
                st.session_state[lb_key] = 1

            # Exclude: minimum 0 days
            try:
                if ex_key not in st.session_state:
                    st.session_state[ex_key] = int(window.get('exclude', 0)) if window is not None else 0
                else:
                    val = st.session_state[ex_key]
                    st.session_state[ex_key] = max(0, int(val) if val is not None else 0)
            except Exception:
                st.session_state[ex_key] = 0

            # Weight: simple initialization - always store as integer percentage
            if wt_key not in st.session_state:
                default_w = window.get('weight', 0.0) if window is not None else 0.0
                # Convert decimal weight to percentage and round to integer
                weight_percentage = min(100, max(0, int(round(default_w * 100))))
                st.session_state[wt_key] = weight_percentage

            with col_win1:
                # Lookback should always be integer days
                st.number_input(
                    "Lookback (days)", min_value=1, step=1, key=lb_key, on_change=update_mom_lookback, args=(i,), format="%d", label_visibility="collapsed"
                )
            with col_win2:
                # Exclude should always be integer days
                st.number_input(
                    "Exclude (days)", min_value=0, step=1, key=ex_key, on_change=update_mom_exclude, args=(i,), format="%d", label_visibility="collapsed"
                )
            with col_win3:
                # Simple weight input - always integer step
                st.number_input(
                    "Weight (%)", min_value=0, max_value=100, step=1, key=wt_key, on_change=update_mom_weight, args=(i,), format="%d", label_visibility="collapsed"
                )
            with col_win4:
                # Use on_click helper for immediate, reliable removal
                st.button("x", key=f"remove_mom_{i}", help="Remove this momentum window", on_click=_remove_mom_and_rerun, args=(i,))
            # Widget state handled by on_change callbacks above

        # Widget keys are initialized properly above, no need to sync again

    # Show total weight and warning if not 100% (allow a small tolerance for rounding)
    if st.session_state.get('use_momentum', False):
        # Convert decimal weights (0.0-1.0) to percentage (0-100) for display
        total_weight = sum(w['weight'] * 100 for w in st.session_state.mom_windows)
        if abs(total_weight - 100) <= _TOTAL_TOL:
            # When effectively 100%, show a clean "100%" without decimal places
            st.markdown('<div style="background-color:#004d00;color:white;padding:8px;border-radius:6px;">Total Weight: 100%</div>', unsafe_allow_html=True)
        else:
            st.warning(f"Total Weight: {total_weight:.2f}% (must sum to 100%)")

    # Prevent running if weights or allocations do not sum to 100% (use same tolerance)
    invalid_weights = False
    invalid_allocs = False
    if st.session_state.use_momentum:
        # Convert decimal weights (0.0-1.0) to percentage (0-100) for validation
        total_weight_percentage = sum(w['weight'] * 100 for w in st.session_state.mom_windows)
        if abs(total_weight_percentage - 100) > _TOTAL_TOL:
            invalid_weights = True
    else:
        # Convert decimal allocations (0.0-1.0) to percentage for validation
        total_alloc_decimal = sum(st.session_state.allocs)
        if abs(total_alloc_decimal - 1.0) > (_ALLOC_TOL / 100.0):
            invalid_allocs = True
    st.markdown("---")
    
    # Stat-specific options
    if st.session_state.use_momentum:
        st.header("Statistic Parameters")
        
        # Beta Section - Completely isolated
        col_beta, col_beta_btn = st.columns([3, 1])
        with col_beta:
            # Use the same pattern as working Multi backtest and Allocations pages
            if "use_beta_checkbox" not in st.session_state:
                st.session_state["use_beta_checkbox"] = st.session_state.use_beta
            st.checkbox(
                "Include Beta in momentum weighting", 
                key="use_beta_checkbox",
                on_change=lambda: setattr(st.session_state, 'use_beta', st.session_state["use_beta_checkbox"])
            )
            
        with col_beta_btn:
            if st.button("Reset Beta", help="Reset beta settings to original", key="reset_beta_btn", on_click=reset_beta_callback):
                pass
                
        # Show beta inputs only if beta is enabled
        if st.session_state.use_beta:
            st.number_input(
                "Beta Window (days)", 
                min_value=1, 
                step=30, 
                format="%d",
                key="beta_window_input",
                on_change=lambda: setattr(st.session_state, 'beta_window_days', st.session_state.beta_window_input)
            )
            st.number_input(
                "Beta Exclude (days)", 
                min_value=0, 
                step=1, 
                format="%d",
                key="beta_exclude_input",
                on_change=lambda: setattr(st.session_state, 'beta_exclude_days', st.session_state.beta_exclude_input)
            )
        
        # Volatility Section - Completely isolated
        col_vol, col_vol_btn = st.columns([3, 1])
        with col_vol:
            # Use the same pattern as working Multi backtest and Allocations pages
            if "use_vol_checkbox" not in st.session_state:
                st.session_state["use_vol_checkbox"] = st.session_state.use_vol
            st.checkbox(
                "Include Volatility in momentum weighting", 
                key="use_vol_checkbox",
                on_change=lambda: setattr(st.session_state, 'use_vol', st.session_state["use_vol_checkbox"])
            )
            
        with col_vol_btn:
            if st.button("Reset Volatility", help="Reset volatility settings to original", key="reset_vol_btn", on_click=reset_volatility_callback):
                pass
                
        # Show volatility inputs only if volatility is enabled
        if st.session_state.use_vol:
            st.number_input(
                "Volatility Window (days)", 
                min_value=1, 
                step=30, 
                format="%d",
                key="vol_window_input",
                on_change=lambda: setattr(st.session_state, 'vol_window_days', st.session_state.vol_window_input)
            )
            st.number_input(
                "Volatility Exclude (days)", 
                min_value=0, 
                step=1, 
                format="%d",
                key="vol_exclude_input",
                on_change=lambda: setattr(st.session_state, 'vol_exclude_days', st.session_state.vol_exclude_input)
            )
    # Handle imported benchmark ticker
    default_benchmark = "^GSPC"
    if "_import_benchmark_ticker" in st.session_state:
        default_benchmark = st.session_state.pop("_import_benchmark_ticker")
    
    benchmark_ticker = st.text_input(
                        "Benchmark Ticker (default: ^GSPC, used for beta calculation)", value=default_benchmark
    ).upper()



    # --- JSON Configuration Section (after Benchmark Ticker) ---
    st.markdown("---")
    
    # Create a single portfolio config for Backtest Engine (since it's single portfolio)
    backtest_engine_config = {
        'name': st.session_state.get("portfolio_name", "Main Portfolio"),
        'stocks': [
            {
                'ticker': ticker,
                'allocation': alloc,
                'include_dividends': div
            }
            for ticker, alloc, div in zip(
                st.session_state.get("tickers", []),
                st.session_state.get("allocs", []),
                st.session_state.get("divs", [])
            )
        ],
        'benchmark_ticker': benchmark_ticker,
        'initial_value': st.session_state.get("initial_value", 10000),
        'added_amount': st.session_state.get("added_amount", 0),
        'added_frequency': st.session_state.get("added_frequency", "Monthly"),
        'rebalancing_frequency': st.session_state.get("rebalancing_frequency", "Monthly"),
        'start_date_user': st.session_state.get("start_date", None),
        'end_date_user': st.session_state.get("end_date", None),
        'start_with': st.session_state.get("start_with_radio_key", "oldest"),
        'first_rebalance_strategy': st.session_state.get("first_rebalance_strategy", "rebalancing_date"),
        'use_momentum': st.session_state.get("use_momentum", False),
        'momentum_strategy': st.session_state.get("momentum_strategy", "Classic momentum"),
        'negative_momentum_strategy': st.session_state.get("negative_momentum_strategy", "Go to cash"),
        'momentum_windows': st.session_state.get("mom_windows", []),
        'calc_beta': st.session_state.get("use_beta", False),
        'calc_volatility': st.session_state.get("use_vol", False),
        'beta_window_days': st.session_state.get("beta_window_days", 365),
        'exclude_days_beta': st.session_state.get("beta_exclude_days", 30),
        'vol_window_days': st.session_state.get("vol_window_days", 365),
        'exclude_days_vol': st.session_state.get("vol_exclude_days", 30),
        'collect_dividends_as_cash': st.session_state.get("collect_dividends_as_cash", False),
        'portfolio_drag_pct': float(st.session_state.get("portfolio_drag_pct", 0.0)),
    }

with st.expander("JSON Configuration (Copy & Paste)", expanded=False):
    # Clean portfolio config for export by removing unused settings
    cleaned_config = backtest_engine_config.copy()
    # Update global settings from session state
    cleaned_config['start_with'] = st.session_state.get('start_with_radio_key', 'oldest')
    cleaned_config['first_rebalance_strategy'] = st.session_state.get('first_rebalance_strategy', 'rebalancing_date')
    config_json = json.dumps(cleaned_config, indent=4, default=str)
    st.code(config_json, language='json')
    # Fixed JSON copy button
    import streamlit.components.v1 as components
    copy_html = f"""
    <button onclick='navigator.clipboard.writeText({json.dumps(config_json)});' style='margin-bottom:10px;'>Copy to Clipboard</button>
    """
    components.html(copy_html, height=40)
    st.text_area("Paste JSON Here to Update Portfolio", key="backtest_engine_paste_json_text", height=200)
    st.button("Update with Pasted JSON", on_click=paste_json_callback)

    # --- Yahoo Finance Verification for Benchmark Ticker ---
    benchmark_error = None
    bench_start_date = None
    asset_start_dates = []  # List of (ticker, date)
    if benchmark_ticker:
        try:
            ticker_yf = yf.Ticker(benchmark_ticker)
            hist_yf = ticker_yf.history(period="max")
            if hist_yf.empty:
                benchmark_error = f"Benchmark ticker '{benchmark_ticker}' not found on Yahoo Finance. Please choose another ticker."
            else:
                bench_start_date = hist_yf.index.min().date()
                # Get start dates for all asset tickers
                for t in st.session_state.tickers:
                    try:
                        ticker_yf_asset = yf.Ticker(t)
                        hist_asset = ticker_yf_asset.history(period="max")
                        if not hist_asset.empty:
                            asset_start_dates.append((t, hist_asset.index.min().date()))
                    except Exception:
                        pass
                # Block if any asset ticker starts before benchmark
                if bench_start_date and asset_start_dates:
                    for asset_name, asset_date in asset_start_dates:
                        if asset_date < bench_start_date:
                            benchmark_error = f"Benchmark ticker '{benchmark_ticker}' starts on {bench_start_date} (Yahoo Finance), but asset ticker '{asset_name}' starts on {asset_date}. Please choose another benchmark or remove assets with earlier data."
                            break
        except Exception as e:
            benchmark_error = f"Error verifying benchmark ticker on Yahoo Finance: {e}"
    if benchmark_error:
        st.warning(benchmark_error)
    # Main page
st.markdown("---")
st.subheader("Run Backtest")

# If running is True, show a blocking info banner. The actual heavy work is
# executed only when both `running` and `_run_requested` are True — this
# prevents the app from performing heavy work in the same rerun that sets the
# running flag and avoids UI hang / infinite-loop scenarios.
if st.session_state.get("running", False):
    st.info("Do not interact when backtest is running.")

# Show the Run button (only if a run is not already requested). The button
# sets a _run_requested flag and triggers a rerun so the banner renders first.
if not st.session_state.get("_run_requested", False):
    col_run, col_clear = st.columns([1, 1])
    with col_run:
        if st.button("Run backtest", type="primary", use_container_width=True):
            # --- PRE-CHECK BLOCK ---
            print("DEBUG: Run Backtest button pressed. Checking pre-conditions...")
            # Print start date of each selected ticker and benchmark for debugging
            for t in st.session_state.tickers:
                try:
                    ticker_yf = yf.Ticker(t)
                    hist = ticker_yf.history(period="max")
                    if not hist.empty:
                        print(f"DEBUG: Asset Ticker {t} starts on {hist.index.min().date()}")
                    else:
                        print(f"DEBUG: Asset Ticker {t}: No data found.")
                except Exception as e:
                    print(f"DEBUG: Asset Ticker {t}: Error - {e}")
            if benchmark_ticker:
                try:
                    ticker_yf = yf.Ticker(benchmark_ticker)
                    hist = ticker_yf.history(period="max")
                    if not hist.empty:
                        print(f"DEBUG: Benchmark Ticker {benchmark_ticker} starts on {hist.index.min().date()}")
                    else:
                        print(f"DEBUG: Benchmark Ticker {benchmark_ticker}: No data found.")
                except Exception as e:
                    print(f"DEBUG: Benchmark Ticker {benchmark_ticker}: Error - {e}")

            # Validate basic pre-conditions; if any fail, ensure running flag is
            # not left set and abort the run immediately.
            if not st.session_state.tickers:
                st.session_state.running = False
                st.session_state._run_requested = False
                st.error("Please add at least one ticker.")
                st.stop()
            if invalid_weights:
                st.session_state.running = False
                st.session_state._run_requested = False
                st.warning("Momentum window weights must sum to 100%. Adjust the weights before running.")
                st.stop()
            if invalid_allocs:
                st.session_state.running = False
                st.session_state._run_requested = False
                st.warning("Ticker allocations must sum to 100%. Adjust the allocations before running.")
                st.stop()
            if benchmark_error:
                st.session_state.running = False
                st.session_state._run_requested = False
                st.warning(benchmark_error)
                st.stop()
            # Block if any asset ticker starts before benchmark
            if benchmark_ticker and bench_start_date and asset_start_dates:
                for asset_name, asset_date in asset_start_dates:
                    if asset_date < bench_start_date:
                        st.session_state.running = False
                        st.session_state._run_requested = False
                        st.warning(f"Benchmark ticker '{benchmark_ticker}' starts on {bench_start_date} (Yahoo Finance), but asset ticker '{asset_name}' starts on {asset_date}. Please choose another benchmark or remove assets with earlier data.")
                        st.stop()

            # Pre-checks passed: stage params and request a run on the next rerun so
            # the banner is rendered before heavy computation begins.
            st.info(f"Backtest period: {st.session_state.start_date} to {st.session_state.end_date}")
            st.session_state.running = True
            st.session_state._run_requested = True
            # Store params in session so the execution block can pick them up.
            st.session_state._pending_backtest_params = {
                "tickers": st.session_state.tickers,
                "benchmark_ticker": benchmark_ticker,
                "allocations": {t: a for t, a in zip(st.session_state.tickers, st.session_state.allocs)},
                "include_dividends": {t: d for t, d in zip(st.session_state.tickers, st.session_state.divs)},
                "initial_value": st.session_state.get("initial_value", 10000),
                "added_amount": st.session_state.get("added_amount", 0),
                "added_frequency": added_frequency,
                "rebalancing_frequency": rebalancing_frequency,
                "start_date_user": st.session_state.start_date,
                "end_date_user": st.session_state.end_date,
                "start_with": st.session_state.get("start_with_radio_key", "oldest"),
                "use_momentum": st.session_state.use_momentum,
                "momentum_windows": st.session_state.mom_windows,
                "initial_allocation_option": "cash",
                "use_relative_momentum": st.session_state.momentum_strategy == "Relative momentum",
                "negative_momentum_strategy": st.session_state.negative_momentum_strategy,
                "calc_beta": st.session_state.use_beta,
                "calc_volatility": st.session_state.use_vol,
                "beta_window_days": st.session_state.beta_window_days,
                "exclude_days_beta": st.session_state.beta_exclude_days,
                "vol_window_days": st.session_state.vol_window_days,
                "exclude_days_vol": st.session_state.vol_exclude_days,
            }
            clear_outputs()
            st.rerun()
    with col_clear:
        st.button("Clear outputs", on_click=clear_outputs, use_container_width=True)

# Execution block: perform the heavy backtest work only when a run has been
# requested and the running banner is visible. This ensures the UI updates
# before computation and avoids re-entrancy/infinite-loop problems.
if st.session_state.get("_run_requested", False) and st.session_state.get("running", False):
    params = st.session_state.get("_pending_backtest_params", None)
    # As a fallback, reconstruct params from session state if needed
    if params is None:
        params = {
            "tickers": st.session_state.tickers,
            "benchmark_ticker": benchmark_ticker,
            "allocations": {t: a for t, a in zip(st.session_state.tickers, st.session_state.allocs)},
            "include_dividends": {t: d for t, d in zip(st.session_state.tickers, st.session_state.divs)},
            "initial_value": st.session_state.get("initial_value", 10000),
            "added_amount": st.session_state.get("added_amount", 0),
            "added_frequency": added_frequency,
            "rebalancing_frequency": rebalancing_frequency,
            "start_date_user": st.session_state.start_date,
            "end_date_user": st.session_state.end_date,
            "start_with": st.session_state.get("start_with_radio_key", "oldest"),
            "use_momentum": st.session_state.use_momentum,
            "momentum_windows": st.session_state.mom_windows,
            "initial_allocation_option": "cash",
            "use_relative_momentum": st.session_state.momentum_strategy == "Relative momentum",
            "negative_momentum_strategy": st.session_state.negative_momentum_strategy,
            "calc_beta": st.session_state.use_beta,
            "calc_volatility": st.session_state.use_vol,
            "beta_window_days": st.session_state.beta_window_days,
            "exclude_days_beta": st.session_state.beta_exclude_days,
            "vol_window_days": st.session_state.vol_window_days,
            "exclude_days_vol": st.session_state.vol_exclude_days,
        }

    # BULLETPROOF VALIDATION: Check for empty tickers before starting the backtest
    if not params.get("tickers") or not any(params.get("tickers")):
        st.error("❌ **No valid tickers found!** Please add at least one ticker before running the backtest.")
        st.session_state.running = False
        st.session_state._run_requested = False
        if "_pending_backtest_params" in st.session_state:
            del st.session_state["_pending_backtest_params"]
        st.stop()

    with st.spinner("Running backtest..."):
        console_buf = io.StringIO()
        with contextlib.redirect_stdout(console_buf):
            try:
                (
                    fig_dict, 
                    stats_with_additions,
                    stats_without_additions,
                    performance_table, 
                    last_rebalance_allocs, 
                    current_allocs,
                    styled_allocs_table,
                    styled_metrics_table,
                    portfolio_value_with_additions,
                    portfolio_value_without_additions,
                    cash_flows_with_additions,
                    cash_flows_without_additions,
                ) = run_backtest(**params)
                st.session_state.fig_dict = fig_dict
                st.session_state.stats_with_additions = stats_with_additions
                st.session_state.stats_without_additions = stats_without_additions
                st.session_state.performance_table = performance_table
                st.session_state.last_rebalance_allocs = last_rebalance_allocs
                st.session_state.current_allocs = current_allocs
                st.session_state.rebalance_alloc_table = styled_allocs_table
                st.session_state.rebalance_metrics_table = styled_metrics_table
                st.session_state.portfolio_value_with_additions = portfolio_value_with_additions
                st.session_state.portfolio_value_without_additions = portfolio_value_without_additions
                st.session_state.cash_flows_with_additions = cash_flows_with_additions
                st.session_state.cash_flows_without_additions = cash_flows_without_additions
            except Exception as e:
                # Ensure flags are cleared so the UI does not remain in running state
                error_message = str(e)
                if "All tickers are invalid" in error_message or "Invalid tickers" in error_message:
                    st.session_state.error = error_message
                else:
                    st.session_state.error = f"An error occurred during backtest: {e}"
                logger.exception("Backtest failed")
            finally:
                st.session_state.console = strip_ansi(console_buf.getvalue())
                st.session_state.last_run_time = datetime.now()
                # Clear running/requested flags and pending params
                st.session_state.running = False
                st.session_state._run_requested = False
                if "_pending_backtest_params" in st.session_state:
                    del st.session_state["_pending_backtest_params"]
                # Rerun so the UI shows results
                st.rerun()


if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.last_run_time:
    st.info(f"Last run: {st.session_state.last_run_time.strftime('%Y-%m-%d %H:%M:%S')}")

if st.session_state.fig_dict:
    # Show backtest period before plots, styled as bold white on black, using actual first/last date
    all_dates = None
    if hasattr(st.session_state, 'portfolio_value_with_additions') and st.session_state.portfolio_value_with_additions is not None:
        all_dates = st.session_state.portfolio_value_with_additions.index
    elif hasattr(st.session_state, 'portfolio_value_without_additions') and st.session_state.portfolio_value_without_additions is not None:
        all_dates = st.session_state.portfolio_value_without_additions.index
    if all_dates is not None and len(all_dates) > 0:
        first_date = all_dates[0].strftime('%Y-%m-%d')
        last_date = all_dates[-1].strftime('%Y-%m-%d')
    else:
        first_date = str(st.session_state.start_date)
        last_date = str(st.session_state.end_date)
    st.markdown(
        f"<div style='background:#18181b;padding:10px;border-radius:6px;margin-bottom:10px;text-align:center;'>"
        f"<span style='color:white;font-weight:bold;font-size:1.1em;'>Backtest period: {first_date} to {last_date}</span>"
        "</div>", unsafe_allow_html=True
    )
    # Two clear return metrics:
    # 1) Return (no additions): final / initial - 1 using the 'without_additions' series
    # 2) Return (with additions): final / initial - 1 using the 'with_additions' series
    r_no_str = "N/A"
    r_with_str = "N/A"
    try:
        if hasattr(st.session_state, 'portfolio_value_without_additions') and st.session_state.portfolio_value_without_additions is not None:
            pv_no = st.session_state.portfolio_value_without_additions
            if len(pv_no) > 0:
                start_no = float(pv_no.iloc[0])
                end_no = float(pv_no.iloc[-1])
                r_no = (end_no / start_no - 1.0) * 100.0
                r_no_str = f"{r_no:,.2f}%"
        if hasattr(st.session_state, 'portfolio_value_with_additions') and st.session_state.portfolio_value_with_additions is not None:
            pv_with = st.session_state.portfolio_value_with_additions
            if len(pv_with) > 0:
                start_w = float(pv_with.iloc[0])
                end_w = float(pv_with.iloc[-1])
                r_w = (end_w / start_w - 1.0) * 100.0
                r_with_str = f"{r_w:,.2f}%"
    except Exception:
        r_no_str = r_no_str if r_no_str != "N/A" else "N/A"
        r_with_str = r_with_str if r_with_str != "N/A" else "N/A"

    # Show both metrics in a clear, high-contrast style for dark background
    final_value_str = "N/A"
    if hasattr(st.session_state, 'portfolio_value_with_additions') and st.session_state.portfolio_value_with_additions is not None:
        pv_with = st.session_state.portfolio_value_with_additions
        if len(pv_with) > 0:
            final_value_str = f"{float(pv_with.iloc[-1]):,.2f}"

    st.markdown(
        """
        <div style='text-align:center;margin-top:-6px;margin-bottom:6px;'>
          <div style='font-size:0.92em;color:#cfcfcf;margin-bottom:2px;'>Return (no additions)</div>
          <div style='font-size:1.28em;font-weight:700;color:#ffffff;margin-bottom:6px;'>{r_no}</div>
          <div style='font-size:0.88em;color:#cfcfcf;margin-bottom:2px;'>Final value</div>
          <div style='font-size:1.12em;font-weight:700;color:#ffffff;'>{final_value}</div>
        </div>
        """ .format(r_no=r_no_str, final_value=final_value_str), unsafe_allow_html=True
    )
    st.subheader("Plots")
    # Align all time-series plots to the same X start/end so their plots' left edges align
    common_start = None
    common_end = None
    if all_dates is not None and len(all_dates) > 0:
        common_start = all_dates[0]
        common_end = all_dates[-1]

    # Create a single, standalone professional main chart rendered at the top.
    # The chart shows Portfolio Value (with additions) and provides two simple
    # date inputs to compute the % variation for the selected range. This is
    # intentionally lightweight and does not rely on external selection
    # components so the UI remains robust across environments.
    try:
        pv = st.session_state.portfolio_value_with_additions
        if pv is not None and not pv.dropna().empty:
            # Build a clean, pro-style figure
            fig_main_pro = go.Figure()
            fig_main_pro.add_trace(go.Scatter(x=pv.index, y=pv.values,
                                              mode='lines', name='Portfolio Value (with additions)',
                                              line=dict(color='#0a84ff', width=2)))
            # subtle area fill
            fig_main_pro.update_traces(fill='tozeroy', fillcolor='rgba(10,132,255,0.06)')

            fig_main_pro.update_layout(
                title='MAIN: Portfolio Value (with additions)',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                template='plotly_white',
                margin=dict(l=80, r=20, t=60, b=60),
                xaxis=dict(rangeslider=dict(visible=True), type='date')
            )

            # Align the x-range with other charts if available
            try:
                if common_start is not None and common_end is not None:
                    fig_main_pro.update_xaxes(range=[common_start, common_end])
            except Exception:
                pass

            # Compute % variation across the whole available range (full plot range)
            try:
                pv_range = pv.dropna()
                if not pv_range.empty:
                    val_start = pv_range.iloc[0]
                    val_end = pv_range.iloc[-1]
                    variation_pct = (val_end / val_start - 1) * 100 if val_start != 0 else 0.0
                else:
                    variation_pct = 0.0
            except Exception:
                variation_pct = 0.0

            # Render the figure and the computed variation
            config = {"modeBarButtonsToRemove": ["zoom2d", "pan2d"], "displaylogo": False}
            st.plotly_chart(fig_main_pro, use_container_width=True, config=config)
    except Exception:
        # If anything fails, skip the main plot rendering gracefully
        pass

    # Use a fixed left margin (pixels) so plotting areas align even when labels differ in width.
    common_left_margin = 80
    for title, fig in st.session_state.fig_dict.items():
        if common_start is not None:
            try:
                fig.update_xaxes(range=[common_start, common_end])
            except Exception:
                pass
        try:
            fig.update_layout(margin=dict(l=common_left_margin, r=20, t=50, b=70))
            fig.update_yaxes(automargin=False)
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True)
        
if st.session_state.stats_without_additions is not None:
    st.markdown("---")
    st.subheader("Backtest Statistics")
    # Fix display names for Max Drawdown and Ulcer Index
    stats = st.session_state.stats_without_additions.copy()
    # Append final portfolio values (with and without cash additions) so they
    # appear in the statistics table.
    try:
        pv_with = st.session_state.portfolio_value_with_additions
        final_with = float(pv_with.dropna().iloc[-1]) if pv_with is not None and not pv_with.dropna().empty else np.nan
    except Exception:
        final_with = np.nan
    try:
        pv_without = st.session_state.portfolio_value_without_additions
        final_without = float(pv_without.dropna().iloc[-1]) if pv_without is not None and not pv_without.dropna().empty else np.nan
    except Exception:
        final_without = np.nan

    # Calculate return (no additions)
    initial_value = st.session_state.get("initial_value", 10000)
    return_no_additions = ((final_without - initial_value) / initial_value) if initial_value > 0 and not pd.isna(final_without) else np.nan
    
    # Use internal keys that map to friendly display names below
    stats["FinalValueWithAdditions"] = final_with
    stats["FinalValueWithoutAdditions"] = final_without
    stats["ReturnNoAdditions"] = return_no_additions
    display_names = {
        "MaxDrawdown": "Max Drawdown",
        "UlcerIndex": "Ulcer Index",
        "CAGR": "CAGR",
        "MWRR": "MWRR",
        "Volatility": "Volatility",
        "Sharpe": "Sharpe",
        "Sortino": "Sortino",
        "UPI": "UPI",
        "Beta": "Beta",
        "FinalValueWithAdditions": "Final Value (with additions)",
        "FinalValueWithoutAdditions": "Final Value (no additions)",
        "ReturnNoAdditions": "Return (no additions)"
    }
    # Create DataFrame and reorder to put Final Values at the top
    stats_df = pd.DataFrame({
        "Metric": [display_names.get(k, k) for k in stats.keys()],
        "Value": list(stats.values())
    })
    
    # Reorder to put Final Values at the top
    final_value_metrics = ["Final Value (with additions)", "Final Value (no additions)", "Return (no additions)"]
    other_metrics = [metric for metric in stats_df["Metric"] if metric not in final_value_metrics]
    
    # Reorder the DataFrame
    reordered_metrics = final_value_metrics + other_metrics
    stats_df = stats_df.set_index("Metric").reindex(reordered_metrics).reset_index()
    def format_value(row):
        val = row["Value"]
        if pd.isna(val):
            return "n/a"
        # Percentage metrics
        if row["Metric"] in ["CAGR", "MWRR", "Max Drawdown", "Volatility", "Return (no additions)"]:
            return f"{val:.2%}"
        # Simple float metrics
        elif row["Metric"] in ["Sharpe", "Sortino", "UPI", "Ulcer Index", "Beta"]:
            return f"{val:.2f}"
        # Currency metrics (final portfolio values)
        elif row["Metric"] in ["Final Value (with additions)", "Final Value (no additions)"]:
            try:
                return f"${val:,.2f}"
            except Exception:
                return val
        return val
    stats_df["Formatted Value"] = stats_df.apply(format_value, axis=1)
    # Use st.table with the already-formatted strings so Streamlit renders the
    # full table directly (avoids the internal scrolling behavior of st.dataframe).
    display_df = stats_df[["Metric", "Formatted Value"]].copy()
    st.table(display_df)

if st.session_state.last_rebalance_allocs is not None and st.session_state.current_allocs is not None:
    st.markdown("---")
    st.subheader("Portfolio Allocations")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_pie_chart(st.session_state.last_rebalance_allocs, "Last Rebalance Allocation"), use_container_width=True)
    with col2:
        st.plotly_chart(create_pie_chart(st.session_state.current_allocs, "Current Allocation"), use_container_width=True)





# --- Results Table Display ---
show_perf = False
show_alloc = st.session_state.rebalance_alloc_table is not None
show_metrics = st.session_state.rebalance_metrics_table is not None
show_perf_table = False


# Only show the period toggle if all tables are ready
if st.session_state.rebalance_alloc_table is not None:
    st.markdown("---")
    st.subheader("Rebalancing Allocations")
    st.dataframe(st.session_state.rebalance_alloc_table, use_container_width=True)

if st.session_state.rebalance_metrics_table is not None:
    st.markdown("---")
    st.subheader("Rebalancing Metrics Table")
    st.dataframe(st.session_state.rebalance_metrics_table, use_container_width=True)

    # --- Single Yearly Performance Table ---
if (
    hasattr(st.session_state, "portfolio_value_without_additions")
    and isinstance(st.session_state.portfolio_value_without_additions, pd.Series)
    and not st.session_state.portfolio_value_without_additions.empty
    and hasattr(st.session_state, "portfolio_value_with_additions")
    and isinstance(st.session_state.portfolio_value_with_additions, pd.Series)
    and not st.session_state.portfolio_value_with_additions.empty
):

    # Get last date for each year for period end
    pv_noadd = st.session_state.portfolio_value_without_additions.copy()
    pv_noadd.index = pd.to_datetime(pv_noadd.index)
    yearly_noadd = pv_noadd.groupby(pv_noadd.index.year).agg(['first', 'last'])
    years = yearly_noadd.index
    period_start = yearly_noadd['first'].index.map(lambda y: pv_noadd[pv_noadd.index.year == y].index.min().strftime('%Y-%m-%d'))
    period_end = yearly_noadd['last'].index.map(lambda y: pv_noadd[pv_noadd.index.year == y].index.max().strftime('%Y-%m-%d'))

    # Variation % (no additions)
    variation_pct = yearly_noadd['last'].pct_change() * 100
    # Calculate first year variation manually if needed
    if len(yearly_noadd) > 0:
        first_year = yearly_noadd.index[0]
        first_start = yearly_noadd.loc[first_year, 'first']
        first_end = yearly_noadd.loc[first_year, 'last']
        if pd.isna(variation_pct.iloc[0]):
            variation_pct.iloc[0] = ((first_end - first_start) / first_start) * 100 if first_start != 0 else 0

    # Variation Value (with additions, but exclude added cash)
    pv_add = st.session_state.portfolio_value_with_additions.copy()
    pv_add.index = pd.to_datetime(pv_add.index)
    yearly_add = pv_add.groupby(pv_add.index.year).agg(['first', 'last'])
    # Calculate total cash added per year
    cash_added = []
    for y in years:
        # Determine actual start/end dates for the period and count only cash
        # flows that occurred strictly after the period start (so the initial
        # deposit recorded at the period start is not double-counted).
        period_mask = (pv_add.index.year == y)
        if not period_mask.any():
            cash_added_year = 0.0
        else:
            period_dates = pv_add[period_mask].index
            period_start_dt = period_dates.min()
            period_end_dt = period_dates.max()
            cash_flow = st.session_state.get('cash_flows_with_additions', None)
            if cash_flow is not None:
                cf = cash_flow.copy()
                cf.index = pd.to_datetime(cf.index)
                # Count only additions that happened after the period start up
                # to avoid subtracting the opening initial deposit.
                added = cf[(cf.index > period_start_dt) & (cf.index <= period_end_dt) & (cf < 0)].sum()
                cash_added_year = -added if not pd.isna(added) else 0.0
            else:
                cash_added_year = 0.0
        cash_added.append(cash_added_year)

    # Variation value: (end - start) - cash added (confirmed: excludes added cash)
    variation_val = yearly_add['last'].values - yearly_add['first'].values - np.array(cash_added)

    df_single = pd.DataFrame({
        "Year": years,
        "Period": [f"{period_start[i]} to {period_end[i]}" for i in range(len(years))],
        "Variation (%)": variation_pct.values,
        "Variation Value": variation_val,
        "Portfolio Value (Counting Added Cash)": yearly_add['last'].values
    })
    # Make the index an explicit column named '#' so it appears in the header row
    try:
        df_single.insert(0, '#', df_single.index)
        df_single = df_single.reset_index(drop=True)
    except Exception:
        try:
            df_single.index.name = "#"
        except Exception:
            pass
    def color_gradient_stock(val):
        if isinstance(val, (int, float)) and not pd.isna(val):
            style = 'background-color: {}; color: white;'
            if val > 50:
                return style.format('#004d00')
            elif val > 20:
                return style.format('#1e8449')
            elif val > 5:
                return style.format('#388e3c')
            elif val > 0:
                return style.format('#66bb6a')
            elif val < -50:
                return style.format('#7b0000')
            elif val < -20:
                return style.format('#b22222')
            elif val < -5:
                return style.format('#d32f2f')
            elif val < 0:
                return style.format('#ef5350')
        return ''
    styler_single = df_single.style.map(color_gradient_stock, subset=['Variation (%)'])
    styler_single = styler_single.format({'Variation (%)': '{:,.2f}%'}, na_rep='N/A')
    styler_single = styler_single.format({'Variation Value': '${:,.2f}', 'Portfolio Value (Counting Added Cash)': '${:,.2f}'}, na_rep='N/A')
    try:
        # Hide the automatic DataFrame index (we inserted '#' as a real column)
        styler_single = styler_single.hide_index()
    except Exception:
        pass
    st.markdown("---")
    st.subheader("Yearly Portfolio Performance Table")
    # Render styled HTML so percentage formatting ("{:,.2f}%") is visible
    try:
        html_single = styler_single.to_html()
        # Force the rendered table to occupy full container width so it matches
        # the space used by the DataFrame-based Rebalancing Metrics Table.
        # Add a perf-table wrapper and CSS to force consistent column widths
        perf_css = '''
            <style>
            .perf-table table { width: 100% !important; border-collapse: collapse; }
            .perf-table th, .perf-table td { padding: 8px 10px; text-align: left; }
            /* Hide the second column (duplicate) */
            .perf-table table thead th:nth-child(2), .perf-table table tbody td:nth-child(2) { display: none; }
            /* Make the first column even smaller */
            .perf-table table thead th:nth-child(1), .perf-table table tbody td:nth-child(1) { width: 4%; text-align: center; }
            /* Remaining columns share remaining width roughly equally (5 columns -> ~19.2% each) */
            .perf-table table thead th:nth-child(3), .perf-table table tbody td:nth-child(3) { width: 19.2%; }
            .perf-table table thead th:nth-child(4), .perf-table table tbody td:nth-child(4) { width: 19.2%; }
            .perf-table table thead th:nth-child(5), .perf-table table tbody td:nth-child(5) { width: 19.2%; }
            .perf-table table thead th:nth-child(6), .perf-table table tbody td:nth-child(6) { width: 19.2%; }
            .perf-table table thead th:nth-child(7), .perf-table table tbody td:nth-child(7) { width: 19.2%; }
            .perf-table tbody tr:nth-child(odd) { background: rgba(255,255,255,0.01); }
            </style>
        '''
        html_single = (
            '<div class="perf-table" style="width:100%;">'
            + perf_css
            + html_single.replace('<table', '<table style="width:100%"', 1)
            + '</div>'
        )
        st.markdown(html_single, unsafe_allow_html=True)
    except Exception:
        st.write(styler_single)

    # --- Monthly Portfolio Performance Table ---
    pv_noadd_month = st.session_state.portfolio_value_without_additions.copy()
    pv_noadd_month.index = pd.to_datetime(pv_noadd_month.index)
    monthly_noadd = pv_noadd_month.groupby([pv_noadd_month.index.year, pv_noadd_month.index.month]).agg(['first', 'last'])
    months = monthly_noadd.index
    period_start_month = [pv_noadd_month[(pv_noadd_month.index.year == y) & (pv_noadd_month.index.month == m)].index.min().strftime('%Y-%m-%d') for y, m in months]
    period_end_month = [pv_noadd_month[(pv_noadd_month.index.year == y) & (pv_noadd_month.index.month == m)].index.max().strftime('%Y-%m-%d') for y, m in months]
    variation_pct_month = monthly_noadd['last'].pct_change() * 100
    # Calculate first month variation manually if needed
    if len(monthly_noadd) > 0:
        first_start = monthly_noadd.iloc[0]['first']
        first_end = monthly_noadd.iloc[0]['last']
        if pd.isna(variation_pct_month.iloc[0]):
            variation_pct_month.iloc[0] = ((first_end - first_start) / first_start) * 100 if first_start != 0 else 0

    pv_add_month = st.session_state.portfolio_value_with_additions.copy()
    pv_add_month.index = pd.to_datetime(pv_add_month.index)
    monthly_add = pv_add_month.groupby([pv_add_month.index.year, pv_add_month.index.month]).agg(['first', 'last'])
    # Calculate total cash added per month
    cash_added_month = []
    for (y, m) in months:
        # Determine period start/end for this month and count only cash added
        # after the period start (exclude opening deposit on the start date).
        mask = (pv_add_month.index.year == y) & (pv_add_month.index.month == m)
        if not mask.any():
            cash_added_m = 0.0
        else:
            period_dates = pv_add_month[mask].index
            period_start_dt = period_dates.min()
            period_end_dt = period_dates.max()
            cash_flow = st.session_state.get('cash_flows_with_additions', None)
            if cash_flow is not None:
                cf = cash_flow.copy()
                cf.index = pd.to_datetime(cf.index)
                added_m = cf[(cf.index > period_start_dt) & (cf.index <= period_end_dt) & (cf < 0)].sum()
                cash_added_m = -added_m if not pd.isna(added_m) else 0.0
            else:
                cash_added_m = 0.0
        cash_added_month.append(cash_added_m)
    variation_val_month = monthly_add['last'].values - monthly_add['first'].values - np.array(cash_added_month)

    df_monthly = pd.DataFrame({
        "Year-Month": [f"{y}-{m:02d}" for (y, m) in months],
        "Period": [f"{period_start_month[i]} to {period_end_month[i]}" for i in range(len(months))],
        "Variation (%)": variation_pct_month.values,
        "Variation Value": variation_val_month,
        "Portfolio Value (Counting Added Cash)": monthly_add['last'].values
    })
    try:
        df_monthly.insert(0, '#', df_monthly.index)
        df_monthly = df_monthly.reset_index(drop=True)
    except Exception:
        try:
            df_monthly.index.name = "#"
        except Exception:
            pass
    styler_monthly = df_monthly.style.map(color_gradient_stock, subset=['Variation (%)'])
    styler_monthly = styler_monthly.format({'Variation (%)': '{:,.2f}%'}, na_rep='N/A')
    styler_monthly = styler_monthly.format({'Variation Value': '${:,.2f}', 'Portfolio Value (Counting Added Cash)': '${:,.2f}'}, na_rep='N/A')
    try:
        styler_monthly = styler_monthly.hide_index()
    except Exception:
        pass
    st.markdown("---")
    st.subheader("Monthly Portfolio Performance Table")
    try:
        html_month = styler_monthly.to_html()
        perf_css = '''
            <style>
            .perf-table table { width: 100% !important; border-collapse: collapse; }
            .perf-table th, .perf-table td { padding: 8px 10px; text-align: left; }
            /* Hide the second column (duplicate) */
            .perf-table table thead th:nth-child(2), .perf-table table tbody td:nth-child(2) { display: none; }
            /* Make the first column even smaller */
            .perf-table table thead th:nth-child(1), .perf-table table tbody td:nth-child(1) { width: 4%; text-align: center; }
            /* Remaining columns share remaining width roughly equally (5 columns -> ~19.2% each) */
            .perf-table table thead th:nth-child(3), .perf-table table tbody td:nth-child(3) { width: 19.2%; }
            .perf-table table thead th:nth-child(4), .perf-table table tbody td:nth-child(4) { width: 19.2%; }
            .perf-table table thead th:nth-child(5), .perf-table table tbody td:nth-child(5) { width: 19.2%; }
            .perf-table table thead th:nth-child(6), .perf-table table tbody td:nth-child(6) { width: 19.2%; }
            .perf-table table thead th:nth-child(7), .perf-table table tbody td:nth-child(7) { width: 19.2%; }
            .perf-table tbody tr:nth-child(odd) { background: rgba(255,255,255,0.01); }
            </style>
        '''
        html_month = (
            '<div class="perf-table" style="width:100%;">'
            + perf_css
            + html_month.replace('<table', '<table style="width:100%"', 1)
            + '</div>'
        )
        st.markdown(html_month, unsafe_allow_html=True)
    except Exception:
        st.write(styler_monthly)


# Console Output Section (at the end, collapsed)
if st.session_state.console:
    st.markdown("---")
    with st.expander("Console Output", expanded=False):
        st.text_area(
            "Backtest Console",

            height=400,
            disabled=True,
            label_visibility="collapsed"
        )

