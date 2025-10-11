# NO_CACHE VERSION - All @st.cache_data decorators removed for maximum reliability
import streamlit as st
from datetime import datetime, timedelta, time, date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
import json
import io
import contextlib
import os
import plotly.io as pio
from numba import jit
import concurrent.futures
from functools import lru_cache
import signal
import sys
import threading
import logging

# Suppress specific Streamlit threading warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").propagate = False

# =============================================================================
# HARD KILL FUNCTIONS
# =============================================================================
def hard_kill_process():
    """Completely kill the current process and all background threads"""
    try:
        # Kill all background threads
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join(timeout=0.1)

        # Force garbage collection
        import gc
        gc.collect()

        # On Windows, use os._exit for immediate termination
        if os.name == 'nt':
            os._exit(1)
        else:
            # On Unix-like systems, use os.kill
            os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        # Last resort - force exit
        os._exit(1)

def check_kill_request():
    """Check if user has requested a hard kill"""
    if st.session_state.get('hard_kill_requested', False):
        st.error("🛑 **HARD KILL REQUESTED** - Terminating all processes...")
        st.stop()

def emergency_kill():
    """Emergency kill function that stops backtest without crashing the app"""
    st.error("🛑 **EMERGENCY KILL** - Forcing immediate backtest termination...")
    st.session_state.hard_kill_requested = True
    st.rerun()

# =============================================================================
# HELPER FUNCTIONS FOR FOCUSED ANALYSIS
# =============================================================================

def calculate_cagr(values, dates):
    if len(values) < 2:
        return np.nan
    start_val = values[0]
    end_val = values[-1]
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0 or start_val == 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1

def calculate_volatility(returns):
    # Annualized volatility - same as Backtest_Engine.py
    return returns.std() * np.sqrt(365) if len(returns) > 1 else np.nan

def calculate_sharpe(returns, risk_free_rate):
    """Calculates the Sharpe ratio."""
    # Create a constant risk-free rate series aligned with returns
    daily_rf_rate = risk_free_rate / 365.25
    rf_series = pd.Series(daily_rf_rate, index=returns.index)
    
    aligned_returns, aligned_rf = returns.align(rf_series, join='inner')
    if aligned_returns.empty:
        return np.nan
    
    excess_returns = aligned_returns - aligned_rf
    if excess_returns.std() == 0:
        return np.nan
        
    return excess_returns.mean() / excess_returns.std() * np.sqrt(365)

def calculate_sortino(returns, risk_free_rate):
    """Calculates the Sortino ratio."""
    # Create a constant risk-free rate series aligned with returns
    daily_rf_rate = risk_free_rate / 365.25
    rf_series = pd.Series(daily_rf_rate, index=returns.index)
    
    aligned_returns, aligned_rf = returns.align(rf_series, join='inner')
    if aligned_returns.empty:
        return np.nan
        
    downside_returns = aligned_returns[aligned_returns < aligned_rf]
    if downside_returns.empty or downside_returns.std() == 0:
        # If no downside returns, Sortino is infinite or undefined.
        # We can return nan or a very high value. nan is safer.
        return np.nan
    
    downside_std = downside_returns.std()
    
    return (aligned_returns.mean() - aligned_rf.mean()) / downside_std * np.sqrt(365)

def calculate_upi(cagr, ulcer_index):
    """Calculates the Ulcer Performance Index (UPI = CAGR / Ulcer Index, both as decimals)."""
    if ulcer_index is None or pd.isna(ulcer_index) or ulcer_index == 0:
        return np.nan
    return cagr / (ulcer_index / 100)

# =============================================================================
# TICKER ALIASES FUNCTIONS
# =============================================================================

def get_ticker_aliases():
    """Define ticker aliases for easier entry"""
    return {
        # Stock Market Indices
        'SPX': '^GSPC',           # S&P 500 (price only, no dividends) - 1927+
        'SPXTR': '^SP500TR',      # S&P 500 Total Return (with dividends) - 1988+
        'SP500': '^GSPC',         # S&P 500 (price only, no dividends) - 1927+
        'SP500TR': '^SP500TR',    # S&P 500 Total Return (with dividends) - 1988+
        'SPYTR': '^SP500TR',      # S&P 500 Total Return (with dividends) - 1988+
        'NASDAQ': '^IXIC',        # NASDAQ Composite (price only, no dividends) - 1971+
        'NDX': '^NDX',           # NASDAQ 100 (price only, no dividends) - 1985+
        'QQQTR': '^IXIC',        # NASDAQ Composite (price only, no dividends) - 1971+
        'DOW': '^DJI',           # Dow Jones Industrial Average (price only, no dividends) - 1992+
        
        # Treasury Yield Indices (LONGEST HISTORY - 1960s+)
        'TNX': '^TNX',           # 10-Year Treasury Yield (1962+) - Price only, no coupons
        'TYX': '^TYX',           # 30-Year Treasury Yield (1977+) - Price only, no coupons
        'FVX': '^FVX',           # 5-Year Treasury Yield (1962+) - Price only, no coupons
        'IRX': '^IRX',           # 3-Month Treasury Yield (1960+) - Price only, no coupons
        
        # Treasury Bond ETFs (MODERN - WITH COUPONS/DIVIDENDS)
        'TLTETF': 'TLT',          # 20+ Year Treasury Bond ETF (2002+) - With coupons
        'IEFETF': 'IEF',          # 7-10 Year Treasury Bond ETF (2002+) - With coupons
        'SHY': 'SHY',            # 1-3 Year Treasury Bond ETF (2002+) - With coupons
        'BIL': 'BIL',            # 1-3 Month T-Bill ETF (2007+) - With coupons
        'GOVT': 'GOVT',          # US Treasury Bond ETF (2012+) - With coupons
        'SPTL': 'SPTL',          # Long Term Treasury ETF (2007+) - With coupons
        'SPTS': 'SPTS',          # Short Term Treasury ETF (2011+) - With coupons
        'SPTI': 'SPTI',          # Intermediate Term Treasury ETF (2007+) - With coupons
        
        # Cash/Zero Return
        'ZEROX': 'ZEROX',        # Zero-cost portfolio (literally cash doing nothing)
        
        # Gold & Commodities
        'GOLDX': 'GOLDX',        # Fidelity Gold Fund (1994+) - With dividends
        'GLD': 'GLD',            # SPDR Gold Trust ETF (2004+) - With dividends
        'IAU': 'IAU',            # iShares Gold Trust ETF (2005+) - With dividends
        'GOLDF': 'GC=F',         # Gold Futures (2000+) - No dividends
        'GOLD50': 'GOLD_COMPLETE',  # Complete Gold Dataset (1975+) - Historical + GLD
        'ZROZ50': 'ZROZ_COMPLETE',  # Complete ZROZ Dataset (1962+) - Historical + ZROZ
        'TLT50': 'TLT_COMPLETE',  # Complete TLT Dataset (1962+) - Historical + TLT
        'BTC50': 'BTC_COMPLETE',  # Complete Bitcoin Dataset (2010+) - Historical + BTC-USD
        'TBILL': 'TBILL_COMPLETE',  # Complete TBILL Dataset (1948+) - Historical + SGOV
        'IEFTR': 'IEF_COMPLETE',  # Complete IEF Dataset (1962+) - Historical + IEF
        'TLTTR': 'TLT_COMPLETE',  # Complete TLT Dataset (1962+) - Historical + TLT
        'ZROZX': 'ZROZ_COMPLETE',  # Complete ZROZ Dataset (1962+) - Historical + ZROZ
        'GOLDX': 'GOLD_COMPLETE',  # Complete Gold Dataset (1975+) - Historical + GLD
        'SPYSIM': 'SPYSIM_COMPLETE',  # Complete S&P 500 Simulation (1885+) - Historical + SPYTR
        'GOLDSIM': 'GOLDSIM_COMPLETE',  # Complete Gold Simulation (1968+) - New Historical + GOLDX
        'KMLMX': 'KMLM_COMPLETE',  # Complete KMLM Dataset (1992+) - Historical + KMLM
        'DBMFX': 'DBMF_COMPLETE',  # Complete DBMF Dataset (2000+) - Historical + DBMF
        'BITCOINX': 'BTC_COMPLETE',  # Complete Bitcoin Dataset (2010+) - Historical + BTC-USD
        'IEF50': 'IEF_COMPLETE',  # Complete IEF Dataset (1962+) - Historical + IEF
        'KMLM50': 'KMLM_COMPLETE',  # Complete KMLM Dataset (1992+) - Historical + KMLM
        'DBMF50': 'DBMF_COMPLETE',  # Complete DBMF Dataset (2000+) - Historical + DBMF
        'TBILL50': 'TBILL_COMPLETE',  # Complete TBILL Dataset (1948+) - Historical + SGOV
        'SILVER': 'SI=F',        # Silver Futures (2000+) - No dividends
        'OIL': 'CL=F',           # Crude Oil Futures (2000+) - No dividends
        'NATGAS': 'NG=F',        # Natural Gas Futures (2000+) - No dividends
        'CORN': 'ZC=F',          # Corn Futures (2000+) - No dividends
        'SOYBEAN': 'ZS=F',       # Soybean Futures (2000+) - No dividends
        'COFFEE': 'KC=F',        # Coffee Futures (2000+) - No dividends
        'SUGAR': 'SB=F',         # Sugar Futures (2000+) - No dividends
        'COTTON': 'CT=F',        # Cotton Futures (2000+) - No dividends
        'COPPER': 'HG=F',        # Copper Futures (2000+) - No dividends
        'PLATINUM': 'PL=F',      # Platinum Futures (1997+) - No dividends
        'PALLADIUM': 'PA=F',     # Palladium Futures (1998+) - No dividends
        
        # Leveraged & Inverse ETFs (Synthetic Aliases)
        'TQQQTR': '^IXIC?L=3?E=0.95',    # 3x NASDAQ Composite (price only) - 1971+
        'SPXLTR': '^SP500TR?L=3?E=1.00', # 3x S&P 500 (with dividends)
        'UPROTR': '^SP500TR?L=3?E=0.91', # 3x S&P 500 (with dividends)
        'QLDTR': '^IXIC?L=2?E=0.95',     # 2x NASDAQ Composite (price only) - 1971+
        'SSOTR': '^SP500TR?L=2?E=0.91',  # 2x S&P 500 (with dividends)
        'SHTR': '^GSPC?L=-1?E=0.89',     # -1x S&P 500 (price only, no dividends) - 1927+
        'PSQTR': '^IXIC?L=-1?E=0.95',    # -1x NASDAQ Composite (price only, no dividends) - 1971+
        'SDSTR': '^GSPC?L=-2?E=0.91',    # -2x S&P 500 (price only, no dividends) - 1927+
        'QIDTR': '^IXIC?L=-2?E=0.95',    # -2x NASDAQ Composite (price only, no dividends) - 1971+
        'SPXUTR': '^GSPC?L=-3?E=1.00',   # -3x S&P 500 (price only, no dividends) - 1927+
        'SQQQTR': '^IXIC?L=-3?E=0.95',   # -3x NASDAQ Composite (price only, no dividends) - 1971+
        
        # Additional mappings for new aliases
        'SPYND': '^GSPC',         # S&P 500 (price only, no dividends) - 1927+
        'QQQND': '^IXIC',         # NASDAQ Composite (price only, no dividends) - 1971+
        
        # Sector Indices (No Dividends) - Using GICS codes
        'XLKND': '^SP500-45',    # S&P 500 Information Technology (1990+)
        'XLVND': '^SP500-35',    # S&P 500 Health Care (1990+)
        'XLPND': '^SP500-30',    # S&P 500 Consumer Staples (1990+)
        'XLFND': '^SP500-40',    # S&P 500 Financials (1990+)
        'XLEND': '^SP500-10',    # S&P 500 Energy (1990+)
        'XLIND': '^SP500-20',    # S&P 500 Industrials (1990+)
        'XLYND': '^SP500-25',    # S&P 500 Consumer Discretionary (1990+)
        'XLBND': '^SP500-15',    # S&P 500 Materials (1990+)
        'XLUND': '^SP500-55',    # S&P 500 Utilities (1990+)
        'XLREND': '^SP500-60',   # S&P 500 Real Estate (1990+)
        'XLCND': '^SP500-50',    # S&P 500 Communication Services (1990+)
        
        # Leveraged & Inverse ETFs (Synthetic Aliases) - NASDAQ-100 versions
        'TQQQND': '^NDX?L=3?E=0.95',     # 3x NASDAQ-100 (price only) - 1985+
        'QLDND': '^NDX?L=2?E=0.95',      # 2x NASDAQ-100 (price only) - 1985+
        'PSQND': '^NDX?L=-1?E=0.95',     # -1x NASDAQ-100 (price only, no dividends) - 1985+
        'QIDND': '^NDX?L=-2?E=0.95',     # -2x NASDAQ-100 (price only, no dividends) - 1985+
        'SQQQND': '^NDX?L=-3?E=0.95',    # -3x NASDAQ-100 (price only, no dividends) - 1985+
        
        # Leveraged & Inverse ETFs (Synthetic Aliases) - NASDAQ Composite versions (longer history)
        'TQQQIXIC': '^IXIC?L=3?E=0.95',  # 3x NASDAQ Composite (price only) - 1971+ ⚠️ Different from real TQQQ
        'QLDIXIC': '^IXIC?L=2?E=0.95',   # 2x NASDAQ Composite (price only) - 1971+ ⚠️ Different from real QLD
        'PSQIXIC': '^IXIC?L=-1?E=0.95',  # -1x NASDAQ Composite (price only, no dividends) - 1971+ ⚠️ Different from real PSQ
        'QIDIXIC': '^IXIC?L=-2?E=0.95',  # -2x NASDAQ Composite (price only, no dividends) - 1971+ ⚠️ Different from real QID
        'SQQQIXIC': '^IXIC?L=-3?E=0.95', # -3x NASDAQ Composite (price only, no dividends) - 1971+ ⚠️ Different from real SQQQ
        
        # S&P 500 leveraged/inverse (unchanged)
        'SPXLTR': '^SP500TR?L=3?E=1.00', # 3x S&P 500 (with dividends) - 1988+
        'UPROTR': '^SP500TR?L=3?E=0.91', # 3x S&P 500 (with dividends) - 1988+
        'SSOTR': '^SP500TR?L=2?E=0.91',  # 2x S&P 500 (with dividends) - 1988+
        'SHND': '^GSPC?L=-1?E=0.89',     # -1x S&P 500 (price only, no dividends) - 1927+
        'SDSND': '^GSPC?L=-2?E=0.91',    # -2x S&P 500 (price only, no dividends) - 1927+
        'SPXUND': '^GSPC?L=-3?E=1.00',   # -3x S&P 500 (price only, no dividends) - 1927+
        
        # Legacy aliases (kept for backward compatibility)
        'TQQQTR': '^NDX?L=3?E=0.95',     # Legacy - 3x NASDAQ-100 (price only) - 1985+
        'QLDTR': '^NDX?L=2?E=0.95',      # Legacy - 2x NASDAQ-100 (price only) - 1985+
        'SHTR': '^GSPC?L=-1?E=0.89',     # Legacy - -1x S&P 500 (price only, no dividends) - 1927+
        'PSQTR': '^NDX?L=-1?E=0.95',     # Legacy - -1x NASDAQ-100 (price only, no dividends) - 1985+
        'SDSTR': '^GSPC?L=-2?E=0.91',    # Legacy - -2x S&P 500 (price only, no dividends) - 1927+
        'QIDTR': '^NDX?L=-2?E=0.95',     # Legacy - -2x NASDAQ-100 (price only, no dividends) - 1985+
        'SPXUTR': '^GSPC?L=-3?E=1.00',   # Legacy - -3x S&P 500 (price only, no dividends) - 1927+
        'SQQQTR': '^NDX?L=-3?E=0.95',    # Legacy - -3x NASDAQ-100 (price only, no dividends) - 1985+
    }

def resolve_ticker_alias(ticker):
    """Resolve ticker alias to actual ticker symbol"""
    aliases = get_ticker_aliases()
    upper_ticker = ticker.upper()
    
    # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
    if upper_ticker == 'BRK.B':
        upper_ticker = 'BRK-B'
    elif upper_ticker == 'BRK.A':
        upper_ticker = 'BRK-A'
    
    return aliases.get(upper_ticker, upper_ticker)

# =============================================================================
# RISK-FREE RATE FUNCTIONS
# =============================================================================

def _get_default_risk_free_rate(dates):
    """Get default risk-free rate when all other methods fail."""
    default_daily = (1 + 0.02) ** (1 / 365.25) - 1
    result = pd.Series(default_daily, index=pd.to_datetime(dates))
    # Ensure the result is timezone-naive
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_convert(None)
    return result

def get_risk_free_rate_robust(dates):
    """Simple risk-free rate fetcher using Yahoo Finance treasury data."""
    try:
        dates = pd.to_datetime(dates)
        if isinstance(dates, pd.DatetimeIndex):
            if getattr(dates, "tz", None) is not None:
                dates = dates.tz_convert(None)
        
        # Get treasury data - use ^IRX (13-week treasury) as primary for leverage calculations
        # Fallback hierarchy: ^IRX → ^FVX → ^TNX → ^TYX
        symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]
        ticker = None
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="max", auto_adjust=False)
                if hist is not None and not hist.empty and 'Close' in hist.columns:
                    break
            except Exception:
                continue
        
        if ticker is None:
            # Final fallback to ^TNX
            ticker = yf.Ticker("^TNX")
        hist = ticker.history(period="max", auto_adjust=False)
        
        if hist is not None and not hist.empty and 'Close' in hist.columns:
            # Filter valid data
            valid_data = hist[hist['Close'].notnull() & (hist['Close'] > 0)]
            
            if not valid_data.empty:
                # Convert annual percentage to daily rate
                annual_rates = valid_data['Close'] / 100.0
                daily_rates = (1 + annual_rates) ** (1 / 365.25) - 1.0
                
                # Create series with timezone-naive index
                daily_rate_series = pd.Series(daily_rates.values, index=daily_rates.index)
                if getattr(daily_rate_series.index, "tz", None) is not None:
                    daily_rate_series.index = daily_rate_series.index.tz_convert(None)
                
                # For each target date, use the most recent available rate
                result = pd.Series(index=dates, dtype=float)
                
                for i, target_date in enumerate(dates):
                    # Find the most recent treasury date <= target_date
                    valid_dates = daily_rate_series.index[daily_rate_series.index <= target_date]
                    
                    if len(valid_dates) > 0:
                        closest_date = valid_dates.max()
                        result.iloc[i] = daily_rate_series.loc[closest_date]
                    else:
                        # If no data before target date, use the earliest available
                        result.iloc[i] = daily_rate_series.iloc[0]
                
                # Handle any remaining NaN values
                if result.isna().any():
                    result = result.fillna(method='ffill').fillna(method='bfill')
                    if result.isna().any():
                        result = result.fillna(0.000105)  # Default daily rate
                
                return result
        
        # Fallback to default if all else fails
        return _get_default_risk_free_rate(dates)
        
    except Exception:
        return _get_default_risk_free_rate(dates)

# =============================================================================
# PERFORMANCE OPTIMIZATION: CACHING FUNCTIONS
# =============================================================================

# =============================================================================
# STANDALONE LEVERAGE FUNCTIONS (Independent from Backtest_Engine)
# =============================================================================

def _ensure_naive_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Return a copy of obj with a tz-naive DatetimeIndex."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        return obj
    idx = obj.index
    if getattr(idx, "tz", None) is not None:
        obj = obj.copy()
        obj.index = idx.tz_convert(None)
    return obj

def _get_default_risk_free_rate(dates):
    """Get default risk-free rate when all other methods fail."""
    default_daily = (1 + 0.02) ** (1 / 365.25) - 1
    result = pd.Series(default_daily, index=pd.to_datetime(dates))
    # Ensure the result is timezone-naive
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_convert(None)
    return result

def get_risk_free_rate_robust(dates):
    """Simple risk-free rate fetcher using Yahoo Finance treasury data."""
    try:
        dates = pd.to_datetime(dates)
        if isinstance(dates, pd.DatetimeIndex):
            if getattr(dates, "tz", None) is not None:
                dates = dates.tz_convert(None)
        
        # Get treasury data - use ^IRX (13-week treasury) as primary for leverage calculations
        # Fallback hierarchy: ^IRX → ^FVX → ^TNX → ^TYX
        symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]
        ticker = None
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="max", auto_adjust=False)
                if hist is not None and not hist.empty and 'Close' in hist.columns:
                    break
            except Exception:
                continue
        
        if ticker is None:
            # Final fallback to ^TNX
            ticker = yf.Ticker("^TNX")
        hist = ticker.history(period="max", auto_adjust=False)
        
        if hist is not None and not hist.empty and 'Close' in hist.columns:
            # Filter valid data
            valid_data = hist[hist['Close'].notnull() & (hist['Close'] > 0)]
            
            if not valid_data.empty:
                # Convert annual percentage to daily rate
                annual_rates = valid_data['Close'] / 100.0
                daily_rates = (1 + annual_rates) ** (1 / 365.25) - 1.0
                
                # Create series with timezone-naive index
                daily_rate_series = pd.Series(daily_rates.values, index=daily_rates.index)
                if getattr(daily_rate_series.index, "tz", None) is not None:
                    daily_rate_series.index = daily_rate_series.index.tz_convert(None)
                
                # For each target date, use the most recent available rate
                result = pd.Series(index=dates, dtype=float)
                
                for i, target_date in enumerate(dates):
                    # Find the most recent treasury date <= target_date
                    valid_dates = daily_rate_series.index[daily_rate_series.index <= target_date]
                    
                    if len(valid_dates) > 0:
                        closest_date = valid_dates.max()
                        result.iloc[i] = daily_rate_series.loc[closest_date]
                    else:
                        # If no data before target date, use the earliest available
                        result.iloc[i] = daily_rate_series.iloc[0]
                
                # Handle any remaining NaN values
                if result.isna().any():
                    result = result.fillna(method='ffill').fillna(method='bfill')
                    if result.isna().any():
                        result = result.fillna(0.000105)  # Default daily rate
                
                return result
        
        # Fallback to default if all else fails
        return _get_default_risk_free_rate(dates)
        
    except Exception:
        return _get_default_risk_free_rate(dates)

def parse_ticker_parameters(ticker_symbol: str) -> tuple[str, float, float]:
    """
    Parse ticker symbol to extract base ticker, leverage multiplier, and expense ratio.
    
    Args:
        ticker_symbol: Ticker symbol with optional parameters (e.g., "SPY?L=3?E=0.84")
        
    Returns:
        tuple: (base_ticker, leverage_multiplier, expense_ratio)
        
    Examples:
        "SPY" -> ("SPY", 1.0, 0.0)
        "SPY?L=3" -> ("SPY", 3.0, 0.0)
        "QQQ?L=3?E=0.84" -> ("QQQ", 3.0, 0.84)
        "QQQ?E=1?L=2" -> ("QQQ", 2.0, 1.0)  # Order doesn't matter
    """
    # Convert commas to dots for decimal separators (like case conversion)
    ticker_symbol = ticker_symbol.replace(",", ".")
    
    base_ticker = ticker_symbol
    leverage = 1.0
    expense_ratio = 0.0
    
    # Parse leverage parameter
    if "?L=" in base_ticker:
        try:
            parts = base_ticker.split("?L=", 1)
            base_ticker = parts[0]
            leverage_part = parts[1]
            
            # Check if there are more parameters after leverage
            if "?" in leverage_part:
                leverage_str, remaining = leverage_part.split("?", 1)
                leverage = float(leverage_str)
                base_ticker += "?" + remaining  # Add back remaining parameters
            else:
                leverage = float(leverage_part)
            
            # Leverage validation removed - allow any leverage value for testing
                
        except (ValueError, IndexError) as e:
            # If parsing fails, treat as regular ticker with no leverage
            leverage = 1.0
    
    # Parse expense ratio parameter
    if "?E=" in base_ticker:
        try:
            parts = base_ticker.split("?E=", 1)
            base_ticker = parts[0]
            expense_part = parts[1]
            
            # Check if there are more parameters after expense ratio
            if "?" in expense_part:
                expense_str, remaining = expense_part.split("?", 1)
                expense_ratio = float(expense_str)
                base_ticker += "?" + remaining  # Add back remaining parameters
            else:
                expense_ratio = float(expense_part)
            
            # Expense ratio validation removed - allow any expense ratio value for testing
                
        except (ValueError, IndexError) as e:
            # If parsing fails, treat as regular ticker with no expense ratio
            expense_ratio = 0.0
    
    return base_ticker.strip(), leverage, expense_ratio

def parse_leverage_ticker(ticker_symbol: str) -> tuple[str, float]:
    """
    Parse ticker symbol to extract base ticker and leverage multiplier.
    This is a backward compatibility wrapper for the new parameter parsing function.
    
    Args:
        ticker_symbol: Ticker symbol, potentially with leverage (e.g., "SPY?L=3")
        
    Returns:
        tuple: (base_ticker, leverage_multiplier)
        
    Examples:
        "SPY" -> ("SPY", 1.0)
        "SPY?L=3" -> ("SPY", 3.0)
        "QQQ?L=2" -> ("QQQ", 2.0)
    """
    base_ticker, leverage, _ = parse_ticker_parameters(ticker_symbol)
    return base_ticker, leverage

def apply_daily_expense_ratio(price_data: pd.DataFrame, expense_ratio: float) -> pd.DataFrame:
    """
    Apply daily expense ratio drag to price data, simulating ETF management fees.
    
    Uses the same approach as leverage: build up the price series step by step
    with daily expense drag applied to each day's return.
    
    Args:
        price_data: DataFrame with 'Close' column containing price data
        expense_ratio: Annual expense ratio as a percentage (e.g., 0.84 for 0.84%)
        
    Returns:
        DataFrame with expense ratio drag applied to price data
    """
    if expense_ratio == 0.0:
        return price_data.copy()
    
    # Create a copy to avoid modifying original data
    result = price_data.copy()
    
    # Calculate daily expense drag: annual_expense_ratio / 365.25
    daily_expense_drag = expense_ratio / 100.0 / 365.25
    
    # Debug: Print the daily drag for verification
    print(f"DEBUG: Applying {expense_ratio}% annual expense ratio")
    print(f"DEBUG: Daily expense drag: {daily_expense_drag * 100:.6f}%")
    
    # Build up the price series step by step (same approach as leverage)
    adjusted_prices = pd.Series(index=price_data.index, dtype=float)
    first_price = price_data['Close'].iloc[0]
    if isinstance(first_price, pd.Series):
        first_price = first_price.iloc[0]
    adjusted_prices.iloc[0] = first_price
    
    # Apply expense drag to each day's return
    for i in range(1, len(price_data)):
        # Get scalar values from the Close column
        current_price = price_data['Close'].iloc[i]
        previous_price = price_data['Close'].iloc[i-1]
        
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]
        if isinstance(previous_price, pd.Series):
            previous_price = previous_price.iloc[0]
        
        if previous_price > 0:
            # Calculate the price change
            price_change = current_price / previous_price - 1
            
            # Apply expense drag to the price change (subtract daily drag)
            adjusted_price_change = price_change - daily_expense_drag
            
            # Apply the adjusted price change to the previous adjusted price
            adjusted_prices.iloc[i] = adjusted_prices.iloc[i-1] * (1 + adjusted_price_change)
        else:
            adjusted_prices.iloc[i] = adjusted_prices.iloc[i-1]
    
    # Update the Close price with adjusted prices
    result['Close'] = adjusted_prices
    
    # Recalculate price changes with the new adjusted prices
    result['Price_change'] = result['Close'].pct_change(fill_method=None)
    
    # Debug: Show the impact
    if len(result) > 1:
        original_return = (result['Close'].iloc[-1] / result['Close'].iloc[0] - 1) * 100
        print(f"DEBUG: Final return with {expense_ratio}% expense ratio: {original_return:.4f}%")
    
    return result

def apply_daily_leverage(price_data: pd.DataFrame, leverage: float, expense_ratio: float = 0.0) -> pd.DataFrame:
    """
    Apply daily leverage multiplier and expense ratio to price data, simulating leveraged ETF behavior.
    
    Leveraged ETFs reset daily, so we apply the leverage to daily returns and then
    compound the results to get the leveraged price series. Includes daily cost drag
    equivalent to (leverage - 1) × risk_free_rate plus daily expense ratio drag.
    
    Args:
        price_data: DataFrame with 'Close' column containing price data
        leverage: Leverage multiplier (e.g., 3.0 for 3x leverage)
        expense_ratio: Annual expense ratio in percentage (e.g., 1.0 for 1% annual expense)
        
    Returns:
        DataFrame with leveraged price data including cost drag and expense ratio drag
    """
    if leverage == 1.0 and expense_ratio == 0.0:
        return price_data.copy()
    
    # Create a copy to avoid modifying original data
    leveraged_data = price_data.copy()
    
    # Get time-varying risk-free rates for the entire period
    try:
        risk_free_rates = get_risk_free_rate_robust(price_data.index)
        # Ensure risk-free rates are timezone-naive to match price_data
        if getattr(risk_free_rates.index, "tz", None) is not None:
            risk_free_rates.index = risk_free_rates.index.tz_localize(None)
    except Exception as e:
        raise
    
    # Calculate daily cost drag: (leverage - 1) × risk_free_rate
    # risk_free_rates is already in daily format, so we don't need to divide by 365.25
    try:
        daily_cost_drag = (leverage - 1) * risk_free_rates
    except Exception as e:
        raise
    
    # Calculate daily expense ratio drag: expense_ratio / 100 / 365.25 (annual to daily)
    daily_expense_drag = expense_ratio / 100.0 / 365.25
    
    # VECTORIZED APPROACH - 100-1000x faster than for loop!
    # Calculate daily returns using vectorized operations
    prices = price_data['Close'].values  # Convert to NumPy array for speed
    daily_returns = np.zeros(len(prices))
    daily_returns[1:] = prices[1:] / prices[:-1] - 1  # Vectorized returns calculation
    
    # Apply leverage to returns and subtract cost drag and expense ratio drag
    leveraged_returns = (daily_returns * leverage) - daily_cost_drag.values - daily_expense_drag
    leveraged_returns[0] = 0  # First day has no return
    
    # Compound the leveraged returns to get prices (cumulative product)
    # Using np.cumprod for vectorized compounding
    leveraged_prices = prices[0] * np.cumprod(1 + leveraged_returns)
    
    # Convert back to pandas Series with proper index
    leveraged_prices = pd.Series(leveraged_prices, index=price_data.index)
    
    # Update the Close price with leveraged prices
    leveraged_data['Close'] = leveraged_prices
    
    # Recalculate price changes with the new leveraged prices
    leveraged_data['Price_change'] = leveraged_data['Close'].pct_change(fill_method=None)
    
    # IMPORTANT: Preserve all other columns (like Dividend_per_share) from original data
    # The dividends should remain at their original values (not leveraged) for leveraged ETFs
    # This is correct behavior - leveraged ETFs don't multiply dividends
    
    return leveraged_data

def generate_zero_return_data(period="max"):
    """Generate synthetic zero return data for ZEROX ticker"""
    try:
        ref_ticker = yf.Ticker("SPY")
        ref_hist = ref_ticker.history(period=period)
        if ref_hist.empty:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=365)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            dates = ref_hist.index
        zero_data = pd.DataFrame({
            'Close': [100.0] * len(dates),
            'Dividends': [0.0] * len(dates)
        }, index=dates)
        return zero_data
    except Exception:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        zero_data = pd.DataFrame({
            'Close': [100.0] * len(dates),
            'Dividends': [0.0] * len(dates)
        }, index=dates)
        return zero_data

def get_gold_complete_data(period="max"):
    """Get complete gold data from our custom gold ticker"""
    try:
        # Import our gold ticker
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from Complete_Tickers.GOLD_COMPLETE_TICKER import create_gold_complete_ticker
        
        # Get the complete gold data
        gold_data = create_gold_complete_ticker()
        
        if gold_data is None:
            # Fallback to GLD if our custom ticker fails
            ticker = yf.Ticker("GLD")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        
        # Convert to the expected format
        result = pd.DataFrame({
            'Close': gold_data['Close'],
            'Dividends': [0.0] * len(gold_data)  # Gold doesn't pay dividends
        }, index=gold_data.index)
        
        return result
    except Exception as e:
        # Fallback to GLD if anything fails
        try:
            ticker = yf.Ticker("GLD")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_zroz_complete_data(period="max"):
    """Get complete ZROZ data from our custom ZROZ ticker"""
    try:
        # Import our ZROZ ticker
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from Complete_Tickers.ZROZ_COMPLETE_TICKER import create_safe_zroz_ticker
        
        # Get the complete ZROZ data
        zroz_data = create_safe_zroz_ticker()
        
        if zroz_data is None:
            # Fallback to ZROZ if our custom ticker fails
            ticker = yf.Ticker("ZROZ")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        
        # Convert to the expected format
        result = pd.DataFrame({
            'Close': zroz_data['Close'],
            'Dividends': [0.0] * len(zroz_data)  # ZROZ doesn't pay dividends
        }, index=zroz_data.index)
        
        return result
    except Exception as e:
        # Fallback to ZROZ if anything fails
        try:
            ticker = yf.Ticker("ZROZ")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_tlt_complete_data(period="max"):
    """Get complete TLT data from our custom TLT ticker"""
    try:
        # Import our TLT ticker
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from Complete_Tickers.TLT_COMPLETE_TICKER import create_safe_tlt_ticker
        
        # Get the complete TLT data
        tlt_data = create_safe_tlt_ticker()
        
        if tlt_data is None:
            # Fallback to TLT if our custom ticker fails
            ticker = yf.Ticker("TLT")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        
        # Convert to the expected format
        result = pd.DataFrame({
            'Close': tlt_data['Close'],
            'Dividends': [0.0] * len(tlt_data)  # TLT doesn't pay dividends
        }, index=tlt_data.index)
        
        return result
    except Exception as e:
        # Fallback to TLT if anything fails
        try:
            ticker = yf.Ticker("TLT")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_bitcoin_complete_data(period="max"):
    """Get complete Bitcoin data from our custom Bitcoin ticker"""
    try:
        # Import our Bitcoin ticker
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from Complete_Tickers.BITCOIN_COMPLETE_TICKER import create_bitcoin_complete_ticker
        
        # Get the complete Bitcoin data
        bitcoin_data = create_bitcoin_complete_ticker()
        
        if bitcoin_data is None:
            # Fallback to BTC-USD if our custom ticker fails
            ticker = yf.Ticker("BTC-USD")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        
        # Convert to the expected format
        result = pd.DataFrame({
            'Close': bitcoin_data['Close'],
            'Dividends': [0.0] * len(bitcoin_data)  # Bitcoin doesn't pay dividends
        }, index=bitcoin_data.index)
        
        return result
    except Exception as e:
        # Fallback to BTC-USD if anything fails
        try:
            ticker = yf.Ticker("BTC-USD")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_spysim_complete_data(period="max"):
    """Get complete SPYSIM data from our custom SPYSIM ticker"""
    try:
        # Import our SPYSIM ticker
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from Complete_Tickers.SPYSIM_COMPLETE_TICKER import create_spysim_complete_ticker
        
        # Get the complete SPYSIM data
        spysim_data = create_spysim_complete_ticker()
        
        if spysim_data is None:
            # Fallback to SPYTR if our custom ticker fails
            ticker = yf.Ticker("^SP500TR")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        
        # Convert to the expected format
        # Handle both DataFrame and Series
        close_data = spysim_data['Close'] if isinstance(spysim_data, pd.DataFrame) else spysim_data
        result = pd.DataFrame({
            'Close': close_data,
            'Dividends': [0.0] * len(spysim_data)
        }, index=spysim_data.index)
        
        return result
        
    except Exception as e:
        # Fallback to SPYTR if anything fails
        try:
            ticker = yf.Ticker("^SP500TR")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_goldsim_complete_data(period="max"):
    """Get complete GOLDSIM data from our custom GOLDSIM ticker"""
    try:
        # Import our GOLDSIM ticker
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from Complete_Tickers.GOLDSIM_COMPLETE_TICKER import create_goldsim_complete_ticker
        
        # Get the complete GOLDSIM data
        goldsim_data = create_goldsim_complete_ticker()
        
        if goldsim_data is None:
            # Fallback to GLD if our custom ticker fails
            ticker = yf.Ticker("GLD")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        
        # Convert to the expected format
        # Handle both DataFrame and Series
        close_data = goldsim_data['Close'] if isinstance(goldsim_data, pd.DataFrame) else goldsim_data
        result = pd.DataFrame({
            'Close': close_data,
            'Dividends': [0.0] * len(goldsim_data)
        }, index=goldsim_data.index)
        
        return result
        
    except Exception as e:
        # Fallback to GLD if anything fails
        try:
            ticker = yf.Ticker("GLD")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_ticker_aliases():
    """Define ticker aliases for easier entry"""
    return {
        # Stock Market Indices
        'SPX': '^GSPC',           # S&P 500 (price only, no dividends) - 1927+
        'SPXTR': '^SP500TR',      # S&P 500 Total Return (with dividends) - 1988+
        'SP500': '^GSPC',         # S&P 500 (price only, no dividends) - 1927+
        'SP500TR': '^SP500TR',    # S&P 500 Total Return (with dividends) - 1988+
        'SPYTR': '^SP500TR',      # S&P 500 Total Return (with dividends) - 1988+
        'NASDAQ': '^IXIC',        # NASDAQ Composite (price only, no dividends) - 1971+
        'NDX': '^NDX',           # NASDAQ 100 (price only, no dividends) - 1985+
        'QQQTR': '^IXIC',        # NASDAQ Composite (price only, no dividends) - 1971+
        'DOW': '^DJI',           # Dow Jones Industrial Average (price only, no dividends) - 1992+
        
        # Treasury Yield Indices (LONGEST HISTORY - 1960s+)
        'TNX': '^TNX',           # 10-Year Treasury Yield (1962+) - Price only, no coupons
        'TYX': '^TYX',           # 30-Year Treasury Yield (1977+) - Price only, no coupons
        'FVX': '^FVX',           # 5-Year Treasury Yield (1962+) - Price only, no coupons
        'IRX': '^IRX',           # 3-Month Treasury Yield (1960+) - Price only, no coupons
        
        # Treasury Bond ETFs (MODERN - WITH COUPONS/DIVIDENDS)
        'TLTETF': 'TLT',          # 20+ Year Treasury Bond ETF (2002+) - With coupons
        'IEFETF': 'IEF',          # 7-10 Year Treasury Bond ETF (2002+) - With coupons
        'SHY': 'SHY',            # 1-3 Year Treasury Bond ETF (2002+) - With coupons
        'BIL': 'BIL',            # 1-3 Month T-Bill ETF (2007+) - With coupons
        'GOVT': 'GOVT',          # US Treasury Bond ETF (2012+) - With coupons
        'SPTL': 'SPTL',          # Long Term Treasury ETF (2007+) - With coupons
        'SPTS': 'SPTS',          # Short Term Treasury ETF (2011+) - With coupons
        'SPTI': 'SPTI',          # Intermediate Term Treasury ETF (2007+) - With coupons
        
        # Cash/Zero Return
        'ZEROX': 'ZEROX',        # Zero-cost portfolio (literally cash doing nothing)
        
        # Gold & Commodities
        'GOLDX': 'GOLDX',        # Fidelity Gold Fund (1994+) - With dividends
        'GLD': 'GLD',            # SPDR Gold Trust ETF (2004+) - With dividends
        'IAU': 'IAU',            # iShares Gold Trust ETF (2005+) - With dividends
        'GOLDF': 'GC=F',         # Gold Futures (2000+) - No dividends
        'GOLD50': 'GOLD_COMPLETE',  # Complete Gold Dataset (1975+) - Historical + GLD
        'ZROZ50': 'ZROZ_COMPLETE',  # Complete ZROZ Dataset (1962+) - Historical + ZROZ
        'TLT50': 'TLT_COMPLETE',  # Complete TLT Dataset (1962+) - Historical + TLT
        'BTC50': 'BTC_COMPLETE',  # Complete Bitcoin Dataset (2010+) - Historical + BTC-USD
        'TBILL': 'TBILL_COMPLETE',  # Complete TBILL Dataset (1948+) - Historical + SGOV
        'IEFTR': 'IEF_COMPLETE',  # Complete IEF Dataset (1962+) - Historical + IEF
        'TLTTR': 'TLT_COMPLETE',  # Complete TLT Dataset (1962+) - Historical + TLT
        'ZROZX': 'ZROZ_COMPLETE',  # Complete ZROZ Dataset (1962+) - Historical + ZROZ
        'GOLDX': 'GOLD_COMPLETE',  # Complete Gold Dataset (1975+) - Historical + GLD
        'SPYSIM': 'SPYSIM_COMPLETE',  # Complete S&P 500 Simulation (1885+) - Historical + SPYTR
        'GOLDSIM': 'GOLDSIM_COMPLETE',  # Complete Gold Simulation (1968+) - New Historical + GOLDX
        'KMLMX': 'KMLM_COMPLETE',  # Complete KMLM Dataset (1992+) - Historical + KMLM
        'DBMFX': 'DBMF_COMPLETE',  # Complete DBMF Dataset (2000+) - Historical + DBMF
        'BITCOINX': 'BTC_COMPLETE',  # Complete Bitcoin Dataset (2010+) - Historical + BTC-USD
        'IEF50': 'IEF_COMPLETE',  # Complete IEF Dataset (1962+) - Historical + IEF
        'KMLM50': 'KMLM_COMPLETE',  # Complete KMLM Dataset (1992+) - Historical + KMLM
        'DBMF50': 'DBMF_COMPLETE',  # Complete DBMF Dataset (2000+) - Historical + DBMF
        'TBILL50': 'TBILL_COMPLETE',  # Complete TBILL Dataset (1948+) - Historical + SGOV
        'SILVER': 'SI=F',        # Silver Futures (2000+) - No dividends
        'OIL': 'CL=F',           # Crude Oil Futures (2000+) - No dividends
        'NATGAS': 'NG=F',        # Natural Gas Futures (2000+) - No dividends
        'CORN': 'ZC=F',          # Corn Futures (2000+) - No dividends
        'SOYBEAN': 'ZS=F',       # Soybean Futures (2000+) - No dividends
        'COFFEE': 'KC=F',        # Coffee Futures (2000+) - No dividends
        'SUGAR': 'SB=F',         # Sugar Futures (2000+) - No dividends
        'COTTON': 'CT=F',        # Cotton Futures (2000+) - No dividends
        'COPPER': 'HG=F',        # Copper Futures (2000+) - No dividends
        'PLATINUM': 'PL=F',      # Platinum Futures (1997+) - No dividends
        'PALLADIUM': 'PA=F',     # Palladium Futures (1998+) - No dividends
        
        # Leveraged & Inverse ETFs (Synthetic Aliases)
        'TQQQTR': '^IXIC?L=3?E=0.95',    # 3x NASDAQ Composite (price only) - 1971+
        'SPXLTR': '^SP500TR?L=3?E=1.00', # 3x S&P 500 (with dividends)
        'UPROTR': '^SP500TR?L=3?E=0.91', # 3x S&P 500 (with dividends)
        'QLDTR': '^IXIC?L=2?E=0.95',     # 2x NASDAQ Composite (price only) - 1971+
        'SSOTR': '^SP500TR?L=2?E=0.91',  # 2x S&P 500 (with dividends)
        'SHTR': '^GSPC?L=-1?E=0.89',     # -1x S&P 500 (price only, no dividends) - 1927+
        'PSQTR': '^IXIC?L=-1?E=0.95',    # -1x NASDAQ Composite (price only, no dividends) - 1971+
        'SDSTR': '^GSPC?L=-2?E=0.91',    # -2x S&P 500 (price only, no dividends) - 1927+
        'QIDTR': '^IXIC?L=-2?E=0.95',    # -2x NASDAQ Composite (price only, no dividends) - 1971+
        'SPXUTR': '^GSPC?L=-3?E=1.00',   # -3x S&P 500 (price only, no dividends) - 1927+
        'SQQQTR': '^IXIC?L=-3?E=0.95',   # -3x NASDAQ Composite (price only, no dividends) - 1971+
        
        # Additional mappings for new aliases
        'SPYND': '^GSPC',         # S&P 500 (price only, no dividends) - 1927+
        'QQQND': '^IXIC',         # NASDAQ Composite (price only, no dividends) - 1971+
        
        # Sector Indices (No Dividends) - Using GICS codes
        'XLKND': '^SP500-45',    # S&P 500 Information Technology (1990+)
        'XLVND': '^SP500-35',    # S&P 500 Health Care (1990+)
        'XLPND': '^SP500-30',    # S&P 500 Consumer Staples (1990+)
        'XLFND': '^SP500-40',    # S&P 500 Financials (1990+)
        'XLEND': '^SP500-10',    # S&P 500 Energy (1990+)
        'XLIND': '^SP500-20',    # S&P 500 Industrials (1990+)
        'XLYND': '^SP500-25',    # S&P 500 Consumer Discretionary (1990+)
        'XLBND': '^SP500-15',    # S&P 500 Materials (1990+)
        'XLUND': '^SP500-55',    # S&P 500 Utilities (1990+)
        'XLREND': '^SP500-60',   # S&P 500 Real Estate (1990+)
        'XLCND': '^SP500-50',    # S&P 500 Communication Services (1990+)
        
        # Leveraged & Inverse ETFs (Synthetic Aliases) - NASDAQ-100 versions
        'TQQQND': '^NDX?L=3?E=0.95',     # 3x NASDAQ-100 (price only) - 1985+
        'QLDND': '^NDX?L=2?E=0.95',      # 2x NASDAQ-100 (price only) - 1985+
        'PSQND': '^NDX?L=-1?E=0.95',     # -1x NASDAQ-100 (price only, no dividends) - 1985+
        'QIDND': '^NDX?L=-2?E=0.95',     # -2x NASDAQ-100 (price only, no dividends) - 1985+
        'SQQQND': '^NDX?L=-3?E=0.95',    # -3x NASDAQ-100 (price only, no dividends) - 1985+
        
        # Leveraged & Inverse ETFs (Synthetic Aliases) - NASDAQ Composite versions (longer history)
        'TQQQIXIC': '^IXIC?L=3?E=0.95',  # 3x NASDAQ Composite (price only) - 1971+ ⚠️ Different from real TQQQ
        'QLDIXIC': '^IXIC?L=2?E=0.95',   # 2x NASDAQ Composite (price only) - 1971+ ⚠️ Different from real QLD
        'PSQIXIC': '^IXIC?L=-1?E=0.95',  # -1x NASDAQ Composite (price only, no dividends) - 1971+ ⚠️ Different from real PSQ
        'QIDIXIC': '^IXIC?L=-2?E=0.95',  # -2x NASDAQ Composite (price only, no dividends) - 1971+ ⚠️ Different from real QID
        'SQQQIXIC': '^IXIC?L=-3?E=0.95', # -3x NASDAQ Composite (price only, no dividends) - 1971+ ⚠️ Different from real SQQQ
        
        # S&P 500 leveraged/inverse (unchanged)
        'SPXLTR': '^SP500TR?L=3?E=1.00', # 3x S&P 500 (with dividends) - 1988+
        'UPROTR': '^SP500TR?L=3?E=0.91', # 3x S&P 500 (with dividends) - 1988+
        'SSOTR': '^SP500TR?L=2?E=0.91',  # 2x S&P 500 (with dividends) - 1988+
        'SHND': '^GSPC?L=-1?E=0.89',     # -1x S&P 500 (price only, no dividends) - 1927+
        'SDSND': '^GSPC?L=-2?E=0.91',    # -2x S&P 500 (price only, no dividends) - 1927+
        'SPXUND': '^GSPC?L=-3?E=1.00',   # -3x S&P 500 (price only, no dividends) - 1927+
        
        # Legacy aliases (kept for backward compatibility)
        'TQQQTR': '^NDX?L=3?E=0.95',     # Legacy - 3x NASDAQ-100 (price only) - 1985+
        'QLDTR': '^NDX?L=2?E=0.95',      # Legacy - 2x NASDAQ-100 (price only) - 1985+
        'SHTR': '^GSPC?L=-1?E=0.89',     # Legacy - -1x S&P 500 (price only, no dividends) - 1927+
        'PSQTR': '^NDX?L=-1?E=0.95',     # Legacy - -1x NASDAQ-100 (price only, no dividends) - 1985+
        'SDSTR': '^GSPC?L=-2?E=0.91',    # Legacy - -2x S&P 500 (price only, no dividends) - 1927+
        'QIDTR': '^NDX?L=-2?E=0.95',     # Legacy - -2x NASDAQ-100 (price only, no dividends) - 1985+
        'SPXUTR': '^GSPC?L=-3?E=1.00',   # Legacy - -3x S&P 500 (price only, no dividends) - 1927+
        'SQQQTR': '^NDX?L=-3?E=0.95',    # Legacy - -3x NASDAQ-100 (price only, no dividends) - 1985+
    }

def resolve_ticker_alias(ticker):
    """Resolve ticker alias to actual ticker symbol"""
    aliases = get_ticker_aliases()
    upper_ticker = ticker.upper()
    
    # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
    if upper_ticker == 'BRK.B':
        upper_ticker = 'BRK-B'
    elif upper_ticker == 'BRK.A':
        upper_ticker = 'BRK-A'
    
    return aliases.get(upper_ticker, upper_ticker)


def get_ticker_data_cached(base_ticker, leverage, expense_ratio, period="max", auto_adjust=False):
    """Get ticker data with proper cache keys including all parameters (NO_CACHE version)"""
    # Resolve ticker alias if it exists
    resolved_ticker = resolve_ticker_alias(base_ticker)
    
    # Special handling for ZEROX - generate zero return data
    if resolved_ticker == "ZEROX":
        return generate_zero_return_data(period)
    
    # Special handling for GOLD_COMPLETE - use our custom gold ticker
    if resolved_ticker == "GOLD_COMPLETE":
        return get_gold_complete_data(period)
    
    # Special handling for ZROZ_COMPLETE - use our custom ZROZ ticker
    if resolved_ticker == "ZROZ_COMPLETE":
        return get_zroz_complete_data(period)
    
    # Special handling for TLT_COMPLETE - use our custom TLT ticker
    if resolved_ticker == "TLT_COMPLETE":
        return get_tlt_complete_data(period)
    
    # Special handling for BTC_COMPLETE - use our custom Bitcoin ticker
    if resolved_ticker == "BTC_COMPLETE":
        return get_bitcoin_complete_data(period)
    
    # Special handling for SPYSIM_COMPLETE - use our custom SPYSIM ticker
    if resolved_ticker == "SPYSIM_COMPLETE":
        return get_spysim_complete_data(period)
    
    # Special handling for GOLDSIM_COMPLETE - use our custom GOLDSIM ticker
    if resolved_ticker == "GOLDSIM_COMPLETE":
        return get_goldsim_complete_data(period)
    
    ticker = yf.Ticker(resolved_ticker)
    hist = ticker.history(period=period, auto_adjust=auto_adjust)[["Close", "Dividends"]]
    
    if hist.empty:
        return hist
        
    # Apply leverage and expense ratio if specified
    if leverage != 1.0 or expense_ratio != 0.0:
        hist = apply_daily_leverage(hist, leverage, expense_ratio)
        
    return hist

def get_multiple_tickers_batch(ticker_list, period="max", auto_adjust=False):
    """
    Smart batch download with fallback to individual downloads.
    
    Strategy:
    1. Try batch download (fast - 1 API call for all tickers)
    2. If batch fails → fallback to individual downloads (reliable)
    3. Invalid tickers are skipped, others continue
    """
    if not ticker_list:
        return {}
    
    results = {}
    yahoo_tickers = []
    
    for ticker_symbol in ticker_list:
        # Parse ticker parameters
        base_ticker = ticker_symbol
        leverage = 1.0
        expense_ratio = 0.0
        
        if '?L=' in ticker_symbol or '?E=' in ticker_symbol:
            parts = ticker_symbol.split('?')
            base_ticker = parts[0]
            for part in parts[1:]:
                if part.startswith('L='):
                    try:
                        leverage = float(part[2:])
                    except:
                        pass
                elif part.startswith('E='):
                    try:
                        expense_ratio = float(part[2:])
                    except:
                        pass
        
        resolved = resolve_ticker_alias(base_ticker)
        yahoo_tickers.append((ticker_symbol, resolved, leverage, expense_ratio))
    
    # Extract unique resolved tickers (exclude ZEROX and _COMPLETE tickers)
    resolved_list = list(set([resolved for _, resolved, _, _ in yahoo_tickers if not resolved.endswith('_COMPLETE') and resolved != 'ZEROX']))
    
    try:
        # BATCH DOWNLOAD
        if len(resolved_list) > 1:
            batch_data = yf.download(
                resolved_list,
                period=period,
                auto_adjust=auto_adjust,
                progress=False,
                group_by='ticker'
            )
            
            if not batch_data.empty:
                for ticker_symbol, resolved, leverage, expense_ratio in yahoo_tickers:
                    # Skip _COMPLETE tickers and ZEROX (they will be handled in fallback section)
                    if resolved.endswith('_COMPLETE') or resolved == 'ZEROX':
                        continue
                    try:
                        if len(resolved_list) > 1:
                            ticker_data = batch_data[resolved][['Close', 'Dividends']] if resolved in batch_data else pd.DataFrame()
                        else:
                            ticker_data = batch_data[['Close', 'Dividends']]
                        
                        if not ticker_data.empty:
                            if leverage != 1.0 or expense_ratio != 0.0:
                                ticker_data = apply_daily_leverage(ticker_data, leverage, expense_ratio)
                            results[ticker_symbol] = ticker_data
                        else:
                            results[ticker_symbol] = pd.DataFrame()
                    except:
                        pass
            else:
                raise Exception("Batch download returned empty")
    except Exception:
        pass
    
    # FALLBACK - Individual downloads
    for ticker_symbol, resolved, leverage, expense_ratio in yahoo_tickers:
        if ticker_symbol not in results or results[ticker_symbol].empty:
            try:
                # Handle special tickers
                if resolved == "ZEROX":
                    hist = generate_zero_return_data(period)
                else:
                    ticker = yf.Ticker(resolved)
                    hist = ticker.history(period=period, auto_adjust=auto_adjust)[["Close", "Dividends"]]
                
                if not hist.empty:
                    if leverage != 1.0 or expense_ratio != 0.0:
                        hist = apply_daily_leverage(hist, leverage, expense_ratio)
                    results[ticker_symbol] = hist
                else:
                    results[ticker_symbol] = pd.DataFrame()
            except:
                results[ticker_symbol] = pd.DataFrame()
    
    return results

def get_ticker_data(ticker_symbol, period="max", auto_adjust=False):
    """Get ticker data (NO_CACHE version)
    
    Args:
        ticker_symbol: Stock ticker symbol (supports leverage and expense ratio format like SPY?L=3?E=0.84)
        period: Data period (used in cache key to prevent conflicts)
        auto_adjust: Auto-adjust setting (used in cache key to prevent conflicts)
    """
    try:
        # Parse parameters from ticker symbol
        base_ticker, leverage, expense_ratio = parse_ticker_parameters(ticker_symbol)
        
        # Debug: Print parsed parameters
        if leverage != 1.0 or expense_ratio != 0.0:
            print(f"DEBUG: Parsed {ticker_symbol} -> Base: {base_ticker}, Leverage: {leverage}, Expense: {expense_ratio}%")
        
        # Use cached function with parsed parameters as separate cache keys
        return get_ticker_data_cached(base_ticker, leverage, expense_ratio, period, auto_adjust)
    except Exception:
        return pd.DataFrame()

# Matplotlib configuration for high-quality PDF generation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

def plotly_to_matplotlib_figure(plotly_fig, title="", width_inches=8, height_inches=6):
    """
    Convert a Plotly figure to a matplotlib figure for PDF generation
    """
    try:
        # Extract data from Plotly figure
        fig_data = plotly_fig.data
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Define a color palette for different traces
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_index = 0
        
        # Process each trace
        for trace in fig_data:
            if trace.type == 'scatter':
                x_data = trace.x
                y_data = trace.y
                name = trace.name if hasattr(trace, 'name') and trace.name else f'Trace {color_index}'
                
                # Get color from trace or use palette
                if hasattr(trace, 'line') and hasattr(trace.line, 'color') and trace.line.color:
                    color = trace.line.color
                else:
                    color = colors[color_index % len(colors)]
                
                # Plot the line
                ax.plot(x_data, y_data, label=name, linewidth=2, color=color)
                color_index += 1
                
            elif trace.type == 'bar':
                x_data = trace.x
                y_data = trace.y
                name = trace.name if hasattr(trace, 'name') and trace.name else f'Bar {color_index}'
                
                # Get color from trace or use palette
                if hasattr(trace, 'marker') and hasattr(trace.marker, 'color') and trace.marker.color:
                    color = trace.marker.color
                else:
                    color = colors[color_index % len(colors)]
                
                # Plot the bars
                ax.bar(x_data, y_data, label=name, color=color, alpha=0.7)
                color_index += 1
                
            elif trace.type == 'pie':
                # Handle pie charts - SIMPLE PERFECT CIRCLE SOLUTION
                labels = trace.labels if hasattr(trace, 'labels') else []
                values = trace.values if hasattr(trace, 'values') else []
                
                if labels and values:
                    # Create a slightly wider figure to ensure perfect circle
                    fig_pie, ax_pie = plt.subplots(figsize=(8.5, 8))
                    
                    # Format long titles to break into multiple lines using textwrap
                    import textwrap
                    formatted_title = textwrap.fill(title, width=40, break_long_words=True, break_on_hyphens=False)
                    ax_pie.set_title(formatted_title, fontsize=14, fontweight='bold', pad=40, y=0.95)
                    
                    # Create pie chart with smart percentage display - hide small ones to prevent overlap
                    def smart_autopct(pct):
                        return f'{pct:.1f}%' if pct > 3 else ''  # Only show percentages > 3%
                    
                    wedges, texts, autotexts = ax_pie.pie(
                        values, 
                        labels=labels, 
                        autopct=smart_autopct,  # Use smart percentage display
                        startangle=90, 
                        colors=colors[:len(values)]
                    )
                    
                    # Create legend labels with percentages
                    legend_labels = []
                    for i, label in enumerate(labels):
                        percentage = (values[i] / sum(values)) * 100
                        legend_labels.append(f"{label} ({percentage:.1f}%)")
                    
                    # Add legend with SPECIFIC POSITIONING to prevent overlap
                    ax_pie.legend(wedges, legend_labels, title="Categories", 
                                loc="center left", bbox_to_anchor=(1.15, 0.5))
                    
                    # This is the magic - force perfect circle
                    ax_pie.axis('equal')
                    
                    # Add extra spacing to prevent overlap
                    plt.subplots_adjust(right=0.8)
                    
                    return fig_pie
        
        # Format the plot
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for dates if needed
        if fig_data and len(fig_data) > 0 and hasattr(fig_data[0], 'x') and fig_data[0].x is not None:
            try:
                # Try to parse as dates
                dates = pd.to_datetime(fig_data[0].x)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator(interval=2))  # Show every 2 years
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            except:
                pass
        
        # Use a fixed, generous bottom margin to ensure plot is never squished
        plt.subplots_adjust(bottom=0.35)
        
        # Return legend information separately for PDF placement
        legend_info = []
        if ax.get_legend_handles_labels()[0]:  # Only process if there are labels
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_info.append({
                    'handle': handle,
                    'label': label,
                    'color': handle.get_color() if hasattr(handle, 'get_color') else '#000000'
                })
        
        # Store legend info in figure object for later use
        fig.legend_info = legend_info
        
        return fig
        
    except Exception as e:
        pass
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.text(0.5, 0.5, f'Error converting plot: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig

def create_matplotlib_table(data, headers, title="", width_inches=10, height_inches=4):
    """
    Create a matplotlib table for PDF generation
    """
    try:
        # Ensure data is properly formatted
        if not data or not headers:
            raise ValueError("Data or headers are empty")
        
        # Convert data to strings and ensure proper format
        formatted_data = []
        for row in data:
            formatted_row = []
            for cell in row:
                if cell is None:
                    formatted_row.append('')
                else:
                    formatted_row.append(str(cell))
            formatted_data.append(formatted_row)
        
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=formatted_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternating rows
        for i in range(1, len(formatted_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        # Add title
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        pass


def create_legend_figure(legend_info, title="Legend", width_inches=10, height_inches=2):
    """
    Create a separate matplotlib figure for the legend to be placed below the main plot
    """
    try:
        if not legend_info:
            return None
        
        # Process labels to handle long portfolio names
        processed_labels = []
        max_label_length = 0
        
        for item in legend_info:
            label = item['label']
            # If label is very long, wrap it intelligently
            if len(label) > 40:
                # Split long labels at spaces or special characters
                words = label.split()
                if len(words) > 3:
                    # For very long names, create multiple lines
                    if len(words) <= 6:
                        # Split in the middle
                        mid = len(words) // 2
                        wrapped_label = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                    else:
                        # Split into thirds for extremely long names
                        third = len(words) // 3
                        wrapped_label = '\n'.join([
                            ' '.join(words[:third]),
                            ' '.join(words[third:2*third]),
                            ' '.join(words[2*third:])
                        ])
                else:
                    wrapped_label = label
            else:
                wrapped_label = label
            
            processed_labels.append(wrapped_label)
            # Calculate max height needed for wrapped labels
            lines = wrapped_label.count('\n') + 1
            max_label_length = max(max_label_length, lines)
        
        # Calculate dynamic height based on number of items and label complexity
        num_items = len(legend_info)
        base_height = 2.5  # Increased from 2.0 to 2.5 for larger text
        height_per_item = max(0.8, max_label_length * 0.25)  # Increased from 0.6/0.2 to 0.8/0.25 for larger text
        legend_height = max(base_height, min(6.0, num_items * height_per_item))  # Increased max from 5.0 to 6.0
        
        fig, ax = plt.subplots(figsize=(width_inches, legend_height))
        ax.axis('off')
        
        # Create legend handles
        handles = []
        
        for item in legend_info:
            # Create a line handle for the legend
            line = plt.Line2D([], [], color=item['color'], linewidth=3)
            handles.append(line)
        
        # Determine optimal number of columns based on content
        if num_items <= 3:
            ncol = num_items
        elif num_items <= 6:
            ncol = min(3, num_items)
        else:
            ncol = min(4, num_items)  # Max 4 columns for many items
        
        # Create legend with specific positioning and better text handling
        legend = ax.legend(handles, processed_labels, 
                           loc='center',
                           ncol=ncol,
                           frameon=True,
                           fancybox=True,
                           shadow=True,
                           fontsize=16,  # Increased from 12 to 16 for better readability
                           columnspacing=2.0,
                           labelspacing=1.2)  # Increased from 1.0 to 1.2 for better spacing
        
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=18, fontweight='bold', pad=15)  # Increased from 14 to 18
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(width_inches, 1))
        ax.text(0.5, 0.5, f'Error creating legend: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig


def create_paginated_legends(legend_info, title="Legend", width_inches=10, max_items_per_page=25):
    """
    Create multiple legend figures if the legend is very long, splitting across pages
    """
    if not legend_info:
        return []
    
    legends = []
    total_items = len(legend_info)
    
    # If legend is short enough, create single legend
    if total_items <= max_items_per_page:
        single_legend = create_legend_figure(legend_info, title, width_inches)
        if single_legend:
            legends.append(single_legend)
        return legends
    
    # Split legend into multiple pages
    num_pages = (total_items + max_items_per_page - 1) // max_items_per_page
    
    for page in range(num_pages):
        start_idx = page * max_items_per_page
        end_idx = min((page + 1) * max_items_per_page, total_items)
        
        page_legend_info = legend_info[start_idx:end_idx]
        page_title = f"{title} (Page {page + 1} of {num_pages})"
        
        # Calculate height for this page
        num_items = len(page_legend_info)
        base_height = 2.5  # Increased from 2.0 to 2.5 for larger text
        height_per_item = 0.8  # Increased from 0.6 to 0.8 for larger text
        page_height = max(base_height, min(6.0, num_items * height_per_item))  # Increased max from 5.0 to 6.0
        
        page_legend = create_legend_figure(page_legend_info, page_title, width_inches, page_height)
        if page_legend:
            legends.append(page_legend)
    
    return legends


def create_matplotlib_table(data, headers, title="", width_inches=10, height_inches=4):
    """
    Create a matplotlib table for PDF generation
    """
    try:
        # Ensure data is properly formatted
        if not data or not headers:
            raise ValueError("Data or headers are empty")
        
        # Convert data to strings and ensure proper format
        formatted_data = []
        for row in data:
            formatted_row = []
            for cell in row:
                if cell is None:
                    formatted_row.append('')
                else:
                    formatted_row.append(str(cell))
            formatted_data.append(formatted_row)
        
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=formatted_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternating rows
        for i in range(1, len(formatted_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        # Add title
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.text(0.5, 0.5, f'Error creating table: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig

# PDF Generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors as reportlab_colors

def generate_strategy_comparison_pdf_report(custom_name=""):
    """
    Generate a simple PDF report with exactly 4 sections using existing Streamlit data
    """
    try:
        # Show progress bar for PDF generation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Add proper PDF metadata
        if custom_name.strip():
            title = f"Strategy Comparison Report - {custom_name.strip()}"
            subject = f"Strategy Analysis Report: {custom_name.strip()}"
        else:
            title = "Strategy Comparison Report"
            subject = "Portfolio Strategy Analysis and Performance Report"
        
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            title=title,
            author="Portfolio Backtest System",
            subject=subject,
            creator="Strategy Comparison Application"
        )
        story = []
        
        # Update progress
        progress_bar.progress(10)
        status_text.text("📄 Initializing PDF document...")
        
        # Get styles
        styles = getSampleStyleSheet()
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,  # Keep original size for consistency
            spaceAfter=20,
            textColor=reportlab_colors.Color(0.2, 0.4, 0.6)
        )
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=reportlab_colors.Color(0.3, 0.5, 0.7)
        )
        
        # Update progress
        progress_bar.progress(20)
        status_text.text("📊 Adding portfolio configurations...")
        
        # Title page (no page break before)
        title_style = ParagraphStyle(
            'TitlePage',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=reportlab_colors.Color(0.2, 0.4, 0.6),
            alignment=1  # Center alignment
        )
        
        subtitle_style = ParagraphStyle(
            'SubtitlePage',
            parent=styles['Normal'],
            fontSize=16,
            spaceAfter=40,
            textColor=reportlab_colors.Color(0.4, 0.6, 0.8),
            alignment=1  # Center alignment
        )
        
        # Main title - use custom name if provided
        if custom_name.strip():
            main_title = f"Strategy Comparison Report - {custom_name.strip()}"
            subtitle = f"Investment Strategy Analysis: {custom_name.strip()}"
        else:
            main_title = "Strategy Comparison Report"
            subtitle = "Comprehensive Investment Strategy Analysis"
        
        story.append(Paragraph(main_title, title_style))
        story.append(Paragraph(subtitle, subtitle_style))
        
        # Document metadata is set in SimpleDocTemplate creation above
        
        # Report metadata
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated on: {current_time}", styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Get backtest period from actual portfolio results (not raw data which includes GSPC for beta)
        if 'strategy_comparison_all_results' in st.session_state:
            all_results = st.session_state.strategy_comparison_all_results
            if all_results:
                # Get first and last dates from actual portfolio backtest results
                all_dates = []
                for portfolio_name, portfolio_results in all_results.items():
                    if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                        # Use with_additions series for actual backtest dates
                        series = portfolio_results['with_additions']
                        if isinstance(series, pd.Series) and not series.empty:
                            all_dates.extend(series.index.tolist())
                    elif isinstance(portfolio_results, pd.Series) and not portfolio_results.empty:
                        # Direct series
                        all_dates.extend(portfolio_results.index.tolist())
                
                if all_dates:
                    start_date = min(all_dates)
                    end_date = max(all_dates)
                    total_days = (end_date - start_date).days
                    
                    story.append(Paragraph(f"Backtest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", styles['Normal']))
                    story.append(Paragraph(f"Total Days Analyzed: {total_days:,}", styles['Normal']))
                    story.append(Spacer(1, 20))
        
        # Table of contents first (correct order)
        toc_style = ParagraphStyle(
            'TOC',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=reportlab_colors.Color(0.3, 0.5, 0.7)
        )
        
        story.append(Paragraph("Table of Contents", toc_style))
        toc_points = [
            "Portfolio Configurations & Parameters",
            "Portfolio Value and Drawdown Comparison", 
            "Final Performance Statistics",
            "Portfolio Allocations & Rebalancing Timers"
        ]
        
        for i, point in enumerate(toc_points, 1):
            story.append(Paragraph(f"{i}. {point}", styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Report overview (after TOC, correct order)
        overview_style = ParagraphStyle(
            'Overview',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=reportlab_colors.Color(0.3, 0.5, 0.7)
        )
        
        story.append(Paragraph("Report Overview", overview_style))
        story.append(Paragraph("This report provides comprehensive analysis of investment strategies, including:", styles['Normal']))
        
        # Overview bullet points (non-personal, clear descriptions)
        overview_points = [
            "Detailed portfolio configurations with all parameters and strategies",
            "Performance analysis with value comparison and drawdown charts",
            "Comprehensive performance statistics and risk metrics",
            "Current allocations and rebalancing countdown timers"
        ]
        
        for point in overview_points:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        
        story.append(PageBreak())
        
        # SECTION 1: Portfolio Configurations & Parameters
        story.append(Paragraph("1. Portfolio Configurations & Parameters", heading_style))
        story.append(Spacer(1, 20))
        
        # Get portfolio configs from session state
        portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
        
        for i, config in enumerate(portfolio_configs):
            # Add page break for all portfolios except the first one
            if i > 0:
                story.append(PageBreak())
            
            story.append(Paragraph(f"Portfolio: {config.get('name', 'Unknown')}", subheading_style))
            story.append(Spacer(1, 10))
            
            # Create configuration table with all parameters
            config_data = [
                ['Parameter', 'Value', 'Description'],
                ['Initial Value', f"${config.get('initial_value', 0):,.2f}", 'Starting portfolio value'],
                ['Added Amount', f"${config.get('added_amount', 0):,.2f}", 'Regular contribution amount'],
                ['Added Frequency', config.get('added_frequency', 'N/A'), 'How often contributions are made'],
                ['Rebalancing Frequency', config.get('rebalancing_frequency', 'N/A'), 'How often portfolio is rebalanced'],
                ['Benchmark', config.get('benchmark_ticker', 'N/A'), 'Performance comparison index'],
                ['Use Momentum', 'Yes' if config.get('use_momentum', False) else 'No', 'Whether momentum strategy is enabled'],
                ['Momentum Strategy', config.get('momentum_strategy', 'N/A'), 'Type of momentum calculation'],
                ['Negative Momentum Strategy', config.get('negative_momentum_strategy', 'N/A'), 'How to handle negative momentum'],
                ['Use Relative Momentum', 'Yes' if config.get('use_relative_momentum', False) else 'No', 'Whether to use relative momentum'],
                ['Equal if All Negative', 'Yes' if config.get('equal_if_all_negative', False) else 'No', 'Equal weight when all momentum is negative'],
                ['Calculate Beta', 'Yes' if config.get('calc_beta', False) else 'No', 'Include beta in momentum weighting'],
                ['Calculate Volatility', 'Yes' if config.get('calc_volatility', False) else 'No', 'Include volatility in momentum weighting'],
                ['Start Strategy', config.get('start_with', 'N/A'), 'Initial allocation strategy'],
                ['First Rebalance Strategy', 
                 "rebalancing date" if config.get('first_rebalance_strategy', 'rebalancing_date') == 'rebalancing_date' else "momentum window complete", 
                 'Initial rebalancing approach'],
                ['Collect Dividends as Cash', 'Yes' if config.get('collect_dividends_as_cash', False) else 'No', 'Dividend handling method'],
                ['Beta Lookback', f"{config.get('beta_window_days', 0)} days", 'Days for beta calculation'],
                ['Beta Exclude', f"{config.get('exclude_days_beta', 0)} days", 'Days excluded from beta calculation'],
                ['Volatility Lookback', f"{config.get('vol_window_days', 0)} days", 'Days for volatility calculation'],
                ['Volatility Exclude', f"{config.get('exclude_days_vol', 0)} days", 'Days excluded from volatility calculation'],
                ['Minimal Threshold', f"{config.get('minimal_threshold_percent', 2.0):.1f}%" if config.get('use_minimal_threshold', False) else 'Disabled', 'Minimum allocation percentage threshold'],
                ['Max Allocation', f"{config.get('max_allocation_percent', 10.0):.1f}%" if config.get('use_max_allocation', False) else 'Disabled', 'Maximum allocation percentage per stock']
            ]
            
            # Add momentum windows if they exist
            momentum_windows = config.get('momentum_windows', [])
            if momentum_windows:
                for i, window in enumerate(momentum_windows, 1):
                    lookback = window.get('lookback', 0)
                    weight = window.get('weight', 0)
                    config_data.append([
                        f'Momentum Window {i}',
                        f"{lookback} days, {weight:.2f}",
                        f"Lookback: {lookback} days, Weight: {weight:.2f}"
                    ])
            
            # Create tables with proper column widths to prevent text overflow
            config_table = Table(config_data, colWidths=[2.2*inch, 1.8*inch, 2.5*inch])
            config_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('WORDWRAP', (0, 0), (-1, -1), True),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3)
            ]))
            
            story.append(config_table)
            story.append(PageBreak())
            # Show ticker allocations table, but hide Allocation % column if momentum is enabled
            if not config.get('use_momentum', True):
                story.append(Paragraph("Initial Ticker Allocations (Entered by User):", styles['Heading3']))
                story.append(Paragraph("Note: These are the initial allocations entered by the user, not rebalanced allocations.", styles['Normal']))
                story.append(Spacer(1, 10))
            else:
                story.append(Paragraph("Initial Ticker Allocations:", styles['Heading3']))
                story.append(Paragraph("Note: Momentum strategy is enabled - ticker allocations are calculated dynamically based on momentum scores.", styles['Normal']))
                story.append(Spacer(1, 10))
            
            # Ticker allocations table
            stocks = config.get('stocks', [])
            if stocks:
                if not config.get('use_momentum', True):
                    # Show full table with Allocation % column for non-momentum strategies
                    stocks_data = [['Ticker', 'Allocation %', 'Include Dividends']]
                    for stock in stocks:
                        ticker = stock.get('ticker', 'N/A')
                        allocation = stock.get('allocation', 0) * 100
                        include_div = "✓" if stock.get('include_dividends', False) else "✗"
                        stocks_data.append([ticker, f"{allocation:.1f}%", include_div])
                    
                    stocks_table = Table(stocks_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
                else:
                    # Hide Allocation % column for momentum strategies
                    stocks_data = [['Ticker', 'Include Dividends']]
                    for stock in stocks:
                        ticker = stock.get('ticker', 'N/A')
                        include_div = "✓" if stock.get('include_dividends', False) else "✗"
                        stocks_data.append([ticker, include_div])
                    
                    stocks_table = Table(stocks_data, colWidths=[2.25*inch, 2.25*inch])
                
                stocks_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                    ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ]))
                
                story.append(stocks_table)
                story.append(Spacer(1, 20))
        
        # Update progress
        progress_bar.progress(40)
        status_text.text("📈 Adding performance charts...")
        
        # SECTION 2: Portfolio Value and Drawdown Comparison
        story.append(PageBreak())
        story.append(Paragraph("2. Portfolio Value and Drawdown Comparison", heading_style))
        story.append(Spacer(1, 20))
        
        # Get performance plots from session state and convert to matplotlib
        fig1 = st.session_state.get('strategy_comparison_fig1')
        fig2 = st.session_state.get('strategy_comparison_fig2')
        
        if fig1:
            try:
                # Convert Plotly figure to matplotlib
                mpl_fig1 = plotly_to_matplotlib_figure(fig1, title="Portfolio Value Comparison", width_inches=10, height_inches=6)
                
                # Save matplotlib figure to buffer
                img_buffer1 = io.BytesIO()
                mpl_fig1.savefig(img_buffer1, format='png', dpi=300, bbox_inches='tight')
                img_buffer1.seek(0)
                plt.close(mpl_fig1)  # Close to free memory
                
                # Add to PDF
                story.append(Image(img_buffer1, width=7.5*inch, height=4.5*inch))
                story.append(Spacer(1, 20))
                
                # Add legend below the plot if available
                if hasattr(mpl_fig1, 'legend_info') and mpl_fig1.legend_info:
                    try:
                        legend_figures = create_paginated_legends(mpl_fig1.legend_info, "Portfolio Legend", width_inches=10)
                        for legend_fig in legend_figures:
                            legend_buffer = io.BytesIO()
                            legend_fig.savefig(legend_buffer, format='png', dpi=300, bbox_inches='tight')
                            legend_buffer.seek(0)
                            plt.close(legend_fig)
                            
                            # Add legend to PDF
                            story.append(Image(legend_buffer, width=7.5*inch, height=2*inch))
                            story.append(Spacer(1, 10))
                    except Exception as e:
                        story.append(Paragraph(f"Error creating legend: {str(e)}", styles['Normal']))
            except Exception as e:
                story.append(Paragraph("Performance comparison plot could not be generated.", styles['Normal']))
        
        if fig2:
            try:
                # Convert Plotly figure to matplotlib
                mpl_fig2 = plotly_to_matplotlib_figure(fig2, title="Portfolio Drawdown Comparison", width_inches=10, height_inches=6)
                
                # Save matplotlib figure to buffer
                img_buffer2 = io.BytesIO()
                mpl_fig2.savefig(img_buffer2, format='png', dpi=300, bbox_inches='tight')
                img_buffer2.seek(0)
                plt.close(mpl_fig2)  # Close to free memory
                
                # Add to PDF
                story.append(Image(img_buffer2, width=7.5*inch, height=4.5*inch))
                story.append(Spacer(1, 20))
                
                # Add legend below the plot if available
                if hasattr(mpl_fig2, 'legend_info') and mpl_fig2.legend_info:
                    try:
                        legend_figures = create_paginated_legends(mpl_fig2.legend_info, "Portfolio Legend", width_inches=10)
                        for legend_fig in legend_figures:
                            legend_buffer = io.BytesIO()
                            legend_fig.savefig(legend_buffer, format='png', dpi=300, bbox_inches='tight')
                            legend_buffer.seek(0)
                            plt.close(legend_fig)
                            
                            # Add legend to PDF
                            story.append(Image(legend_buffer, width=7.5*inch, height=2*inch))
                            story.append(Spacer(1, 10))
                    except Exception as e:
                        story.append(Paragraph(f"Error creating legend: {str(e)}", styles['Normal']))
            except Exception as e:
                story.append(Paragraph("Drawdown comparison plot could not be generated.", styles['Normal']))
        
        # Add Risk-Free Rate plot (Annualized)
        fig4 = st.session_state.get('strategy_comparison_fig4')
        if fig4:
            try:
                # Convert Plotly figure to matplotlib
                mpl_fig4 = plotly_to_matplotlib_figure(fig4, title="Annualized Risk-Free Rate (13-Week Treasury)", width_inches=10, height_inches=6)
                
                # Save matplotlib figure to buffer
                img_buffer4 = io.BytesIO()
                mpl_fig4.savefig(img_buffer4, format='png', dpi=300, bbox_inches='tight')
                img_buffer4.seek(0)
                plt.close(mpl_fig4)  # Close to free memory
                
                # Add to PDF
                story.append(Image(img_buffer4, width=7.5*inch, height=4.5*inch))
                story.append(Spacer(1, 20))
                
                # Add legend below the plot if available
                if hasattr(mpl_fig4, 'legend_info') and mpl_fig4.legend_info:
                    try:
                        legend_figures = create_paginated_legends(mpl_fig4.legend_info, "Risk-Free Rate Legend", width_inches=10)
                        for legend_fig in legend_figures:
                            legend_buffer = io.BytesIO()
                            legend_fig.savefig(legend_buffer, format='png', dpi=300, bbox_inches='tight')
                            legend_buffer.seek(0)
                            plt.close(legend_fig)
                            
                            # Add legend to PDF
                            story.append(Image(legend_buffer, width=7.5*inch, height=2*inch))
                            story.append(Spacer(1, 10))
                    except Exception as e:
                        story.append(Paragraph(f"Error creating legend: {str(e)}", styles['Normal']))
            except Exception as e:
                story.append(Paragraph(f"Error converting risk-free rate plot: {str(e)}", styles['Normal']))
        
        # Update progress
        progress_bar.progress(60)
        status_text.text("📋 Adding performance statistics...")
        
        # SECTION 3: Final Performance Statistics
        story.append(PageBreak())
        story.append(Paragraph("3. Final Performance Statistics", heading_style))
        story.append(Spacer(1, 15))
        
        # GUARANTEED statistics table creation - use multiple data sources
        table_created = False
        
        # Method 1: NUKE APPROACH - Extract from fig_stats with proper data handling
        if 'strategy_comparison_fig_stats' in st.session_state and not table_created:
            try:
                fig_stats = st.session_state.strategy_comparison_fig_stats
                if hasattr(fig_stats, 'data') and fig_stats.data:
                    for trace in fig_stats.data:
                        if trace.type == 'table':
                            # Get headers
                            if hasattr(trace, 'header') and trace.header and hasattr(trace.header, 'values'):
                                headers = trace.header.values
                            else:
                                headers = ['Portfolio', 'CAGR (%)', 'Max Drawdown (%)', 'Volatility (%)', 'Sharpe Ratio', 'Sortino Ratio']
                            
                            # Get cell data
                            if hasattr(trace, 'cells') and trace.cells and hasattr(trace.cells, 'values'):
                                cell_data = trace.cells.values
                                if cell_data and len(cell_data) > 0:
                                    # Convert to proper table format with header wrapping
                                    num_rows = len(cell_data[0]) if cell_data[0] else 0
                                    table_rows = []
                                    
                                    # Improved header wrapping with better line breaks and dynamic sizing
                                    wrapped_headers = []
                                    common_words = ['Portfolio', 'Volatility', 'Drawdown', 'Sharpe', 'Sortino', 'Ulcer', 'Index', 'Return', 'Value', 'Money', 'Added', 'Contributions']
                                    
                                    for header in headers:
                                        if len(header) > 8:  # More aggressive wrapping for better readability
                                            # Split on spaces and create multi-line header
                                            words = header.split()
                                            if len(words) > 1:
                                                # Smart splitting: try to balance lines
                                                if len(words) == 2:
                                                    wrapped_header = '\n'.join(words)
                                                elif len(words) == 3:
                                                    wrapped_header = '\n'.join([words[0], ' '.join(words[1:])])
                                                elif len(words) == 4:
                                                    wrapped_header = '\n'.join([' '.join(words[:2]), ' '.join(words[2:])])
                                                else:
                                                    # For longer headers, split more aggressively
                                                    mid = len(words) // 2
                                                    wrapped_header = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                            else:
                                                # Single long word - split more aggressively
                                                if header not in common_words and len(header) > 10:
                                                    mid = len(header) // 2
                                                    wrapped_header = header[:mid] + '\n' + header[mid:]
                                                else:
                                                    wrapped_header = header
                                        else:
                                            wrapped_header = header
                                        wrapped_headers.append(wrapped_header)
                                    
                                    for row_idx in range(num_rows):
                                        row = []
                                        for col_idx in range(len(cell_data)):
                                            if col_idx < len(cell_data) and row_idx < len(cell_data[col_idx]):
                                                value = cell_data[col_idx][row_idx]
                                                # Wrap long portfolio names in the first column
                                                if col_idx == 0 and len(str(value)) > 25:
                                                    # Split long portfolio names at spaces with balanced line breaks
                                                    words = str(value).split()
                                                    if len(words) > 5:
                                                        # For very long names, create 2-3 lines maximum
                                                        if len(words) <= 8:
                                                            # 2 lines: split in the middle
                                                            mid = len(words) // 2
                                                            wrapped_value = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                                        else:
                                                            # 3 lines: split into thirds for extremely long names
                                                            third = len(words) // 3
                                                            wrapped_value = '\n'.join([
                                                                ' '.join(words[:third]),
                                                                ' '.join(words[third:2*third]),
                                                                ' '.join(words[2*third:])
                                                            ])
                                                    elif len(words) > 3:
                                                        # Split in the middle for medium names
                                                        mid = len(words) // 2
                                                        wrapped_value = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                                    else:
                                                        wrapped_value = str(value)
                                                else:
                                                    wrapped_value = str(value) if value is not None else ''
                                                row.append(wrapped_value)
                                            else:
                                                row.append('')
                                        table_rows.append(row)
                                    
                                    # Create table with smart column widths for statistics table - WIDER TABLE WITH MONETARY COLUMNS
                                    page_width = 8.2*inch  # Increased from 7.5 to 8.2 inches for maximum width usage
                                    
                                    # Optimized column width distribution for statistics table - WITH WIDER MONETARY COLUMNS
                                    if len(headers) > 8:  # If we have many columns, use optimized widths
                                        # Portfolio column: increased from 1.4 to 2.1 inches for better text wrapping
                                        # Performance metrics get more space for better readability
                                        portfolio_width = 2.1*inch
                                        remaining_width = page_width - portfolio_width
                                        
                                        # Create custom column widths with wider monetary columns
                                        col_widths = [portfolio_width]
                                        for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                            header_lower = header.lower()
                                            # Give extra width to monetary value columns
                                            if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                                col_widths.append(1.6 * (remaining_width / (len(headers) - 1)))  # 60% wider for monetary columns
                                            else:
                                                col_widths.append(remaining_width / (len(headers) - 1))
                                        
                                        # Ensure total width equals page_width
                                        total_allocated = sum(col_widths)
                                        if total_allocated > page_width:
                                            # Scale down proportionally
                                            scale_factor = page_width / total_allocated
                                            col_widths = [w * scale_factor for w in col_widths]
                                            
                                    elif len(headers) > 6:  # Medium number of columns
                                        # Portfolio column: 2.3 inches for medium tables
                                        portfolio_width = 2.3*inch
                                        remaining_width = page_width - portfolio_width
                                        
                                        # Create custom column widths with wider monetary columns
                                        col_widths = [portfolio_width]
                                        for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                            header_lower = header.lower()
                                            # Give extra width to monetary value columns
                                            if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                                col_widths.append(1.7 * (remaining_width / (len(headers) - 1)))  # 70% wider for monetary columns
                                            else:
                                                col_widths.append(remaining_width / (len(headers) - 1))
                                        
                                        # Ensure total width equals page_width
                                        total_allocated = sum(col_widths)
                                        if total_allocated > page_width:
                                            # Scale down proportionally
                                            scale_factor = page_width / total_allocated
                                            col_widths = [w * scale_factor for w in col_widths]
                                            
                                    else:
                                        # Few columns: Portfolio gets 2.0 inches, others share remaining space
                                        portfolio_width = 2.0*inch
                                        remaining_width = page_width - portfolio_width
                                        
                                        # Create custom column widths with wider monetary columns
                                        col_widths = [portfolio_width]
                                        for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                            header_lower = header.lower()
                                            # Give extra width to monetary value columns
                                            if any(word in header_lower for word in ['value', 'portfolio', 'total']):
                                                col_widths.append(1.7 * (remaining_width / (len(headers) - 1)))  # 70% wider for monetary columns
                                            else:
                                                col_widths.append(remaining_width / (len(headers) - 1))
                                        
                                        # Ensure total width equals page_width
                                        total_allocated = sum(col_widths)
                                        if total_allocated > page_width:
                                            # Scale down proportionally
                                            scale_factor = page_width / total_allocated
                                            col_widths = [w * scale_factor for w in col_widths]
                                    
                                    stats_table = Table([wrapped_headers] + table_rows, colWidths=col_widths)
                                    # Dynamic font sizing based on number of columns and header complexity
                                    num_columns = len(headers)
                                    max_header_length = max(len(header) for header in headers)
                                    
                                    # EXACT SAME FONT SIZING AS METHOD 2 - QUICK PATCH
                                    font_size = 4 if len(headers) > 14 else 5 if len(headers) > 12 else 6 if len(headers) > 10 else 7 if len(headers) > 8 else 8
                                    
                                    # Adjust for very long headers - moderate reduction
                                    if max_header_length > 20:
                                        font_size = max(4, font_size - 1)  # Reduce font size moderately
                                    
                                    stats_table.setStyle(TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                        ('FONTSIZE', (0, 0), (-1, 0), font_size),  # Font size for headers
                                        ('FONTSIZE', (0, 1), (-1, -1), font_size + 2),  # Slightly larger font for data rows
                                        ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                                        ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                        ('LEFTPADDING', (0, 0), (-1, -1), 1),  # Reduced padding to maximize table width usage
                                        ('RIGHTPADDING', (0, 0), (-1, -1), 1),  # Reduced padding to maximize table width usage
                                        ('TOPPADDING', (0, 0), (-1, 0), 4),  # Increased padding for header row for better title visibility
                                        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),  # Increased padding for header row for better title visibility
                                        ('TOPPADDING', (0, 1), (-1, -1), 2),  # Padding for data rows
                                        ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
                                        ('WORDWRAP', (0, 0), (-1, -1), True)
                                    ]))
                                    story.append(stats_table)
                                    story.append(Spacer(1, 15))
                                    table_created = True
                                else:
                                    story.append(Paragraph("No statistics data available.", styles['Normal']))
            except Exception as e:
                pass
        
        # Method 2: Try to get from strategy_comparison_stats_df_display
        if 'strategy_comparison_stats_df_display' in st.session_state and not table_created:
            try:
                snapshot = st.session_state.strategy_comparison_snapshot_data
                all_results = snapshot.get('all_results', {})
                
                if all_results:
                    table_data = []
                    headers = ['Portfolio', 'CAGR (%)', 'Max Drawdown (%)', 'Volatility (%)', 'Sharpe Ratio', 'Sortino Ratio']
                    
                    # Wrap long headers to multiple lines
                    wrapped_headers = []
                    for header in headers:
                        if len(header) > 8:  # If header is long, wrap it
                            # Split on spaces and create multi-line header
                            words = header.split()
                            if len(words) > 1:
                                # Try to split in the middle
                                mid = len(words) // 2
                                wrapped_header = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                            else:
                                # Single long word, split in middle
                                mid = len(header) // 2
                                wrapped_header = header[:mid] + '\n' + header[mid:]
                        else:
                            wrapped_header = header
                        wrapped_headers.append(wrapped_header)
                    
                    for portfolio_name, result in all_results.items():
                        if isinstance(result, dict) and 'metrics' in result:
                            metrics = result['metrics']
                            row = [
                                portfolio_name,
                                f"{metrics.get('cagr', 0):.2f}",
                                f"{metrics.get('max_drawdown', 0):.2f}",
                                f"{metrics.get('volatility', 0):.2f}",
                                f"{metrics.get('sharpe_ratio', 0):.2f}",
                                f"{metrics.get('sortino_ratio', 0):.2f}"
                            ]
                            table_data.append(row)
                    
                    if table_data:
                        # Create table with smart formatting for statistics - WIDER TABLE WITH MONETARY COLUMNS
                        page_width = 8.2*inch  # Increased from 7.5 to 8.2 inches for maximum width usage
                        
                        # Optimized column width distribution for statistics table - WITH WIDER MONETARY COLUMNS
                        if len(headers) > 8:  # If we have many columns, use optimized widths
                            # Portfolio column: increased from 1.4 to 2.1 inches for better text wrapping
                            # Performance metrics get more space for better readability
                            portfolio_width = 2.1*inch
                            remaining_width = page_width - portfolio_width
                            
                            # Create custom column widths with wider monetary columns
                            col_widths = [portfolio_width]
                            for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                header_lower = header.lower()
                                # Give extra width to monetary value columns
                                if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                    col_widths.append(1.5 * (remaining_width / (len(headers) - 1)))  # 50% wider for monetary columns
                                else:
                                    col_widths.append(remaining_width / (len(headers) - 1))
                            
                            # Ensure total width equals page_width
                            total_allocated = sum(col_widths)
                            if total_allocated > page_width:
                                # Scale down proportionally
                                scale_factor = page_width / total_allocated
                                col_widths = [w * scale_factor for w in col_widths]
                                
                        elif len(headers) > 6:  # Medium number of columns
                            # Portfolio column: 2.3 inches for medium tables
                            portfolio_width = 2.3*inch
                            remaining_width = page_width - portfolio_width
                            
                            # Create custom column widths with wider monetary columns
                            col_widths = [portfolio_width]
                            for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                header_lower = header.lower()
                                # Give extra width to monetary value columns
                                if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                    col_widths.append(1.6 * (remaining_width / (len(headers) - 1)))  # 60% wider for monetary columns
                                else:
                                    col_widths.append(remaining_width / (len(headers) - 1))
                            
                            # Ensure total width equals page_width
                            total_allocated = sum(col_widths)
                            if total_allocated > page_width:
                                # Scale down proportionally
                                scale_factor = page_width / total_allocated
                                col_widths = [w * scale_factor for w in col_widths]
                                
                        else:
                            # Few columns: Portfolio gets 2.0 inches, others share remaining space
                            portfolio_width = 2.0*inch
                            remaining_width = page_width - portfolio_width
                            
                            # Create custom column widths with wider monetary columns
                            col_widths = [portfolio_width]
                            for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                header_lower = header.lower()
                                # Give extra width to monetary value columns
                                if any(word in header_lower for word in ['value', 'portfolio', 'total']):
                                    col_widths.append(1.7 * (remaining_width / (len(headers) - 1)))  # 70% wider for monetary columns
                                else:
                                    col_widths.append(remaining_width / (len(headers) - 1))
                            
                            # Ensure total width equals page_width
                            total_allocated = sum(col_widths)
                            if total_allocated > page_width:
                                # Scale down proportionally
                                scale_factor = page_width / total_allocated
                                col_widths = [w * scale_factor for w in col_widths]
                        
                        stats_table = Table([wrapped_headers] + table_data, colWidths=col_widths)
                        
                        # Smart font size based on number of columns - SLIGHTLY LARGER FOR BETTER READABILITY
                        font_size = 4 if len(headers) > 14 else 5 if len(headers) > 12 else 6 if len(headers) > 10 else 7 if len(headers) > 8 else 8
                        
                        stats_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), font_size),  # Font size for headers
                            ('FONTSIZE', (0, 1), (-1, -1), font_size + 2),  # Slightly larger font for data rows
                            ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 0),  # Zero padding for maximum space
                            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                            ('TOPPADDING', (0, 0), (-1, 0), 1),  # Minimal padding for header row
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 1),
                            ('TOPPADDING', (0, 1), (-1, -1), 0.5),  # Minimal padding for data rows
                            ('BOTTOMPADDING', (0, 1), (-1, -1), 0.5),
                            ('WORDWRAP', (0, 0), (-1, -1), True)
                        ]))
                        story.append(stats_table)
                        story.append(Spacer(1, 15))
                        table_created = True
            except Exception as e:
                pass
        
        # Method 3: Fallback - create simple table from any available data
        if not table_created:
            try:
                # Try to get any available portfolio data
                available_data = []
                if 'strategy_comparison_snapshot_data' in st.session_state:
                    snapshot = st.session_state.strategy_comparison_snapshot_data
                    portfolio_configs = snapshot.get('portfolio_configs', [])
                    if portfolio_configs:
                        headers = ['Portfolio', 'Status']
                        for config in portfolio_configs:
                            name = config.get('name', 'Unknown')
                            # Wrap long portfolio names with balanced line breaks for consistency
                            if len(name) > 25:
                                words = name.split()
                                if len(words) > 5:
                                    # For very long names, create 2-3 lines maximum
                                    if len(words) <= 8:
                                        # 2 lines: split in the middle
                                        mid = len(words) // 2
                                        wrapped_name = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                    else:
                                        # 3 lines: split into thirds for extremely long names
                                        third = len(words) // 3
                                        wrapped_name = '\n'.join([
                                            ' '.join(words[:third]),
                                            ' '.join(words[third:2*third]),
                                            ' '.join(words[2*third:])
                                        ])
                                elif len(words) > 3:
                                    # Split in the middle for medium names
                                    mid = len(words) // 2
                                    wrapped_name = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                else:
                                    wrapped_name = name
                            else:
                                wrapped_name = name
                            available_data.append([wrapped_name, 'Data Available'])
                
                if available_data:
                    fallback_table = Table([headers] + available_data, colWidths=[2.2*inch, 2.8*inch])
                    fallback_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                        ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black)
                    ]))
                    story.append(fallback_table)
                    story.append(Spacer(1, 15))
                    table_created = True
            except Exception as e:
                pass
        
        if not table_created:
            story.append(Paragraph("Statistics data not available. Please run the backtest first.", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # SECTION 3.1: Top 5 Best and Worst Performing Portfolios
        # Extract data from the Final Performance Statistics table that was just created
        if table_created and 'fig_stats' in st.session_state:
            try:
                fig_stats = st.session_state.fig_stats
                if hasattr(fig_stats, 'data') and fig_stats.data:
                    for trace in fig_stats.data:
                        if trace.type == 'table':
                            # Get headers and data from the existing table
                            if hasattr(trace, 'header') and trace.header and hasattr(trace.header, 'values'):
                                headers = trace.header.values
                            else:
                                headers = ['Portfolio', 'CAGR (%)', 'Max Drawdown (%)', 'Volatility (%)', 'Sharpe Ratio', 'Sortino Ratio']
                            
                            if hasattr(trace, 'cells') and trace.cells and hasattr(trace.cells, 'values'):
                                cell_data = trace.cells.values
                                if cell_data and len(cell_data) > 0 and len(cell_data[0]) > 0:
                                    # Convert to list of rows for easier processing
                                    num_rows = len(cell_data[0])
                                    table_rows = []
                                    for row_idx in range(num_rows):
                                        row = []
                                        for col_idx in range(len(cell_data)):
                                            if col_idx < len(cell_data) and row_idx < len(cell_data[col_idx]):
                                                value = cell_data[col_idx][row_idx]
                                                row.append(str(value) if value is not None else '')
                                            else:
                                                row.append('')
                                        table_rows.append(row)
                                    
                                    if len(table_rows) > 0:
                                        # Sort by Final Portfolio Value (column 1) to get best and worst performers
                                        # Convert Final Portfolio Value to float for sorting (remove $ and commas)
                                        def get_final_value(row):
                                            try:
                                                value_str = str(row[1]).replace('$', '').replace(',', '')
                                                return float(value_str)
                                            except:
                                                return 0.0
                                        
                                        # Sort by Final Portfolio Value descending (best first)
                                        sorted_rows = sorted(table_rows, key=get_final_value, reverse=True)
                                        
                                        # Get top 10 best and worst
                                        num_portfolios = len(sorted_rows)
                                        top_10_best = sorted_rows[:min(10, num_portfolios)]
                                        
                                        # For worst performers, get the actual worst (lowest final values)
                                        # Sort by final value ascending to get worst first, then take top 10
                                        worst_sorted = sorted(table_rows, key=get_final_value, reverse=False)
                                        top_10_worst = worst_sorted[:min(10, num_portfolios)]
                                        
                                        # Add page break before the new tables
                                        story.append(PageBreak())
                                        
                                        # Top 10 Best Performing Portfolios
                                        story.append(Paragraph("3.1. Top 10 Best Performing Portfolios by Final Value", heading_style))
                                        story.append(Spacer(1, 10))
                                        
                                        if len(top_10_best) > 0:
                                            # Create table data with headers - EXACT SAME TEXT WRAPPING AS FINAL PERFORMANCE STATISTICS
                                            # Wrap headers for better display
                                            wrapped_headers = []
                                            common_words = ['Portfolio', 'Volatility', 'Drawdown', 'Sharpe', 'Sortino', 'Ulcer', 'Index', 'Return', 'Value', 'Money', 'Added', 'Contributions']
                                            
                                            for header in headers:
                                                if len(header) > 8:  # More aggressive wrapping for better readability
                                                    # Split on spaces and create multi-line header
                                                    words = header.split()
                                                    if len(words) > 1:
                                                        # Smart splitting: try to balance lines
                                                        if len(words) == 2:
                                                            wrapped_header = '\n'.join(words)
                                                        elif len(words) == 3:
                                                            wrapped_header = '\n'.join([words[0], ' '.join(words[1:])])
                                                        elif len(words) == 4:
                                                            wrapped_header = '\n'.join([' '.join(words[:2]), ' '.join(words[2:])])
                                                        else:
                                                            # For longer headers, split more aggressively
                                                            mid = len(words) // 2
                                                            wrapped_header = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                                    else:
                                                        # Single long word - split more aggressively
                                                        if header not in common_words and len(header) > 10:
                                                            mid = len(header) // 2
                                                            wrapped_header = header[:mid] + '\n' + header[mid:]
                                                        else:
                                                            wrapped_header = header
                                                else:
                                                    wrapped_header = header
                                                wrapped_headers.append(wrapped_header)
                                            
                                            # Wrap portfolio names in the first column for best performers
                                            wrapped_best_rows = []
                                            for row in top_10_best:
                                                wrapped_row = row.copy()
                                                if len(str(row[0])) > 25:  # Wrap long portfolio names
                                                    words = str(row[0]).split()
                                                    if len(words) > 5:
                                                        if len(words) <= 8:
                                                            mid = len(words) // 2
                                                            wrapped_row[0] = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                                        else:
                                                            third = len(words) // 3
                                                            wrapped_row[0] = '\n'.join([
                                                                ' '.join(words[:third]),
                                                                ' '.join(words[third:2*third]),
                                                                ' '.join(words[2*third:])
                                                            ])
                                                    elif len(words) > 3:
                                                        mid = len(words) // 2
                                                        wrapped_row[0] = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                                wrapped_best_rows.append(wrapped_row)
                                            
                                            best_table_data = [wrapped_headers] + wrapped_best_rows
                                            
                                            # EXACT SAME COLUMN WIDTH LOGIC AS FINAL PERFORMANCE STATISTICS TABLE
                                            page_width = 8.2*inch  # Same as Final Performance Statistics
                                            
                                            # Optimized column width distribution - EXACT SAME LOGIC
                                            if len(headers) > 8:  # If we have many columns, use optimized widths
                                                portfolio_width = 2.1*inch
                                                remaining_width = page_width - portfolio_width
                                                
                                                col_widths = [portfolio_width]
                                                for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                                    header_lower = header.lower()
                                                    if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                                        col_widths.append(1.6 * (remaining_width / (len(headers) - 1)))
                                                    else:
                                                        col_widths.append(remaining_width / (len(headers) - 1))
                                                
                                                total_allocated = sum(col_widths)
                                                if total_allocated > page_width:
                                                    scale_factor = page_width / total_allocated
                                                    col_widths = [w * scale_factor for w in col_widths]
                                                    
                                            elif len(headers) > 6:  # Medium number of columns
                                                portfolio_width = 2.3*inch
                                                remaining_width = page_width - portfolio_width
                                                
                                                col_widths = [portfolio_width]
                                                for i, header in enumerate(headers[1:], 1):
                                                    header_lower = header.lower()
                                                    if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                                        col_widths.append(1.7 * (remaining_width / (len(headers) - 1)))
                                                    else:
                                                        col_widths.append(remaining_width / (len(headers) - 1))
                                                
                                                total_allocated = sum(col_widths)
                                                if total_allocated > page_width:
                                                    scale_factor = page_width / total_allocated
                                                    col_widths = [w * scale_factor for w in col_widths]
                                                    
                                            else:  # Few columns
                                                portfolio_width = 2.0*inch
                                                remaining_width = page_width - portfolio_width
                                                
                                                col_widths = [portfolio_width]
                                                for i, header in enumerate(headers[1:], 1):
                                                    header_lower = header.lower()
                                                    if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                                        col_widths.append(1.7 * (remaining_width / (len(headers) - 1)))
                                                    else:
                                                        col_widths.append(remaining_width / (len(headers) - 1))
                                                
                                                total_allocated = sum(col_widths)
                                                if total_allocated > page_width:
                                                    scale_factor = page_width / total_allocated
                                                    col_widths = [w * scale_factor for w in col_widths]
                                            
                                            # EXACT SAME TABLE CREATION AND STYLING AS FINAL PERFORMANCE STATISTICS
                                            best_table = Table(best_table_data, colWidths=col_widths)
                                            
                                            # Dynamic font sizing - EXACT SAME LOGIC
                                            num_columns = len(headers)
                                            max_header_length = max(len(header) for header in headers)
                                            
                                            if num_columns > 14:
                                                font_size = 5
                                            elif num_columns > 12:
                                                font_size = 6
                                            elif num_columns > 10:
                                                font_size = 7
                                            elif num_columns > 8:
                                                font_size = 8
                                            else:
                                                font_size = 9
                                            
                                            if max_header_length > 20:
                                                font_size = max(4, font_size - 1)
                                            
                                            best_table.setStyle(TableStyle([
                                                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                                                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                                ('FONTSIZE', (0, 0), (-1, 0), font_size),
                                                ('FONTSIZE', (0, 1), (-1, -1), font_size + 2),
                                                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                                                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                                                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                                ('LEFTPADDING', (0, 0), (-1, -1), 1),
                                                ('RIGHTPADDING', (0, 0), (-1, -1), 1),
                                                ('TOPPADDING', (0, 0), (-1, 0), 4),
                                                ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                                                ('TOPPADDING', (0, 1), (-1, -1), 2),
                                                ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
                                                ('WORDWRAP', (0, 0), (-1, -1), True)
                                            ]))
                                            story.append(best_table)
                                            story.append(Spacer(1, 15))
                                        
                                        # Top 10 Worst Performing Portfolios
                                        story.append(PageBreak())
                                        story.append(Paragraph("3.2. Top 10 Worst Performing Portfolios by Final Value", heading_style))
                                        story.append(Spacer(1, 10))
                                        
                                        if len(top_10_worst) > 0:
                                            # Create table data with headers - EXACT SAME TEXT WRAPPING AS FINAL PERFORMANCE STATISTICS
                                            # Wrap portfolio names in the first column for worst performers
                                            wrapped_worst_rows = []
                                            for row in top_10_worst:
                                                wrapped_row = row.copy()
                                                if len(str(row[0])) > 25:  # Wrap long portfolio names
                                                    words = str(row[0]).split()
                                                    if len(words) > 5:
                                                        if len(words) <= 8:
                                                            mid = len(words) // 2
                                                            wrapped_row[0] = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                                        else:
                                                            third = len(words) // 3
                                                            wrapped_row[0] = '\n'.join([
                                                                ' '.join(words[:third]),
                                                                ' '.join(words[third:2*third]),
                                                                ' '.join(words[2*third:])
                                                            ])
                                                    elif len(words) > 3:
                                                        mid = len(words) // 2
                                                        wrapped_row[0] = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                                wrapped_worst_rows.append(wrapped_row)
                                            
                                            worst_table_data = [wrapped_headers] + wrapped_worst_rows
                                            
                                            # EXACT SAME TABLE CREATION AND STYLING AS FINAL PERFORMANCE STATISTICS
                                            worst_table = Table(worst_table_data, colWidths=col_widths)
                                            worst_table.setStyle(TableStyle([
                                                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.7, 0.3, 0.3)),  # Red header for worst
                                                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                                ('FONTSIZE', (0, 0), (-1, 0), font_size),
                                                ('FONTSIZE', (0, 1), (-1, -1), font_size + 2),
                                                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                                                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                                                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                                ('LEFTPADDING', (0, 0), (-1, -1), 1),
                                                ('RIGHTPADDING', (0, 0), (-1, -1), 1),
                                                ('TOPPADDING', (0, 0), (-1, 0), 4),
                                                ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                                                ('TOPPADDING', (0, 1), (-1, -1), 2),
                                                ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
                                                ('WORDWRAP', (0, 0), (-1, -1), True)
                                            ]))
                                            story.append(worst_table)
                                            story.append(Spacer(1, 15))
                                        
                                        break  # Exit the loop after processing the first table
                    
            except Exception as e:
                story.append(Paragraph(f"Error creating top performers tables: {str(e)}", styles['Normal']))
                story.append(Spacer(1, 15))
        
        # Update progress
        progress_bar.progress(80)
        status_text.text("🎯 Adding allocation charts and timers...")
        
        # SECTION 4: Portfolio Allocations & Rebalancing Timers
        story.append(PageBreak())
        current_date_str = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"4. Portfolio Allocations & Rebalancing Timers ({current_date_str})", heading_style))
        story.append(Spacer(1, 20))
        
        # Get snapshot data for allocations
        snapshot_data = st.session_state.get('strategy_comparison_snapshot_data', {})
        today_weights_map = snapshot_data.get('today_weights_map', {})
        
        portfolio_count = 0
        for portfolio_config in portfolio_configs:
            portfolio_name = portfolio_config.get('name', 'Unknown')
            portfolio_count += 1
            
            # Add portfolio header
            story.append(Paragraph(f"Portfolio: {portfolio_name}", subheading_style))
            story.append(Spacer(1, 10))
            
            # Create pie chart for this portfolio (since we need ALL portfolios, not just the selected one)
            try:
                # Create labels and values for the plot
                today_weights = today_weights_map.get(portfolio_name, {})
                labels_today = [k for k, v in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
                vals_today = [float(today_weights[k]) * 100 for k in labels_today]
                
                if labels_today and vals_today:
                    # Create matplotlib pie chart (same format as sector/industry)
                    fig, ax_target = plt.subplots(1, 1, figsize=(10, 10))
                    
                    # Create pie chart with smart percentage display - hide small ones to prevent overlap
                    def smart_autopct(pct):
                        return f'{pct:.1f}%' if pct > 3 else ''  # Only show percentages > 3%
                    
                    wedges_target, texts_target, autotexts_target = ax_target.pie(vals_today, autopct=smart_autopct, 
                                                                                 startangle=90)
                    
                    # Add ticker names with percentages outside the pie chart slices for allocations > 1.8%
                    for i, (wedge, ticker, alloc) in enumerate(zip(wedges_target, labels_today, vals_today)):
                        # Only show tickers above 1.8%
                        if alloc > 1.8:
                            # Calculate position for the text (middle of the slice)
                            angle = (wedge.theta1 + wedge.theta2) / 2
                            # Convert angle to radians and calculate position
                            rad = np.radians(angle)
                            # Position text outside the pie chart at 1.4 radius (farther away)
                            x = 1.4 * np.cos(rad)
                            y = 1.4 * np.sin(rad)
                            
                            # Add ticker name with percentage under it (e.g., "ORLY 5%")
                            ax_target.text(x, y, f"{ticker}\n{alloc:.1f}%", ha='center', va='center', 
                                         fontsize=8, fontweight='bold', 
                                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                            
                            # Add leader line from slice edge to label
                            # Start from edge of pie chart (radius 1.0)
                            line_start_x = 1.0 * np.cos(rad)
                            line_start_y = 1.0 * np.sin(rad)
                            # End at label position
                            line_end_x = 1.25 * np.cos(rad)
                            line_end_y = 1.25 * np.sin(rad)
                            
                            ax_target.plot([line_start_x, line_end_x], [line_start_y, line_end_y], 
                                         'k-', linewidth=0.5, alpha=0.6)
                    
                    # Create legend with percentages - positioned farther to the right to avoid overlap
                    legend_labels = [f"{ticker} ({alloc:.1f}%)" for ticker, alloc in zip(labels_today, vals_today)]
                    ax_target.legend(wedges_target, legend_labels, title="Tickers", loc="center left", bbox_to_anchor=(1.15, 0, 0.5, 1), fontsize=10)
                    
                    # Wrap long titles to prevent them from going out of bounds
                    title_text = f'Target Allocation - {portfolio_name}'
                    # Use textwrap for proper word-based wrapping
                    import textwrap
                    wrapped_title = textwrap.fill(title_text, width=40, break_long_words=True, break_on_hyphens=False)
                    ax_target.set_title(wrapped_title, fontsize=14, fontweight='bold', pad=80)
                    # Force perfectly circular shape
                    ax_target.set_aspect('equal')
                    # Use tighter axis limits to make pie chart appear larger within its space
                    ax_target.set_xlim(-1.2, 1.2)
                    ax_target.set_ylim(-1.2, 1.2)
                    
                    # Adjust layout to accommodate legend and title (better spacing to prevent title cutoff)
                    # Use more aggressive spacing like sector/industry charts for bigger pie chart
                    plt.subplots_adjust(left=0.1, right=0.7, top=0.95, bottom=0.05)
                    
                    # Save to buffer
                    target_img_buffer = io.BytesIO()
                    fig.savefig(target_img_buffer, format='png', dpi=300, facecolor='white')
                    target_img_buffer.seek(0)
                    plt.close(fig)
                    
                    # Add to PDF - increased pie chart size for better visibility
                    story.append(Image(target_img_buffer, width=5.5*inch, height=5.5*inch))
                    
                    # Add Next Rebalance Timer information - simple text display
                    story.append(Paragraph(f"Next Rebalance Timer - {portfolio_name}", subheading_style))
                    story.append(Spacer(1, 1))
                    
                    # Try to get timer information from session state
                    timer_table_key = f"strategy_comparison_timer_table_{portfolio_name}"
                    if timer_table_key in st.session_state:
                        timer_fig = st.session_state[timer_table_key]
                        if timer_fig and hasattr(timer_fig, 'data') and timer_fig.data:
                            # Extract timer information from the figure data
                            for trace in timer_fig.data:
                                if trace.type == 'table' and hasattr(trace, 'cells') and trace.cells:
                                    cell_values = trace.cells.values
                                    if cell_values and len(cell_values) >= 2:
                                        # Format the timer information
                                        timer_info = []
                                        for i in range(len(cell_values[0])):
                                            if i < len(cell_values[0]) and i < len(cell_values[1]):
                                                param = cell_values[0][i]
                                                value = cell_values[1][i]
                                                timer_info.append(f"{param}: {value}")
                                        
                                        if timer_info:
                                            for info in timer_info:
                                                story.append(Paragraph(info, styles['Normal']))
                                        else:
                                            story.append(Paragraph("Next rebalance information not available", styles['Normal']))
                                        break
                            else:
                                story.append(Paragraph("Next rebalance information not available", styles['Normal']))
                        else:
                            story.append(Paragraph("Next rebalance information not available", styles['Normal']))
                    else:
                        story.append(Paragraph("Next rebalance information not available", styles['Normal']))
                    
                    # Add page break after pie plot + timer to separate from allocation table
                    story.append(PageBreak())
                    
                    # Now add the allocation table on the next page
                    story.append(Paragraph(f"Allocation Details for {portfolio_name}", subheading_style))
                    story.append(Spacer(1, 10))
                    
                    # NUKE APPROACH: Rebuild allocation table from scratch
                    alloc_table_key = f"strategy_comparison_fig_alloc_table_{portfolio_name}"
                    table_created = False
                    
                    if alloc_table_key in st.session_state:
                        try:
                            fig_alloc = st.session_state[alloc_table_key]
                            
                            # Method 1: Extract from Plotly figure data structure
                            if hasattr(fig_alloc, 'data') and fig_alloc.data:
                                for trace in fig_alloc.data:
                                    if trace.type == 'table':
                                        # Get headers
                                        if hasattr(trace, 'header') and trace.header and hasattr(trace.header, 'values'):
                                            headers = trace.header.values
                                        else:
                                            headers = ['Asset', 'Allocation %', 'Price ($)', 'Shares', 'Total Value ($)', '% of Portfolio']
                                        
                                        # Get cell data
                                        if hasattr(trace, 'cells') and trace.cells and hasattr(trace.cells, 'values'):
                                            cell_data = trace.cells.values
                                            if cell_data and len(cell_data) > 0:
                                                # Convert to proper table format
                                                num_rows = len(cell_data[0]) if cell_data[0] else 0
                                                table_rows = []
                                                
                                                for row_idx in range(num_rows):
                                                    row = []
                                                    for col_idx in range(len(cell_data)):
                                                        if col_idx < len(cell_data) and row_idx < len(cell_data[col_idx]):
                                                            value = cell_data[col_idx][row_idx]
                                                            row.append(str(value) if value is not None else '')
                                                        else:
                                                            row.append('')
                                                    table_rows.append(row)
                                                
                                                # Calculate total values for summary row
                                                total_alloc_pct = sum(float(row[1].rstrip('%')) for row in table_rows)
                                                total_value = sum(float(row[4].replace('$', '').replace(',', '')) for row in table_rows)
                                                total_port_pct = sum(float(row[5].rstrip('%')) for row in table_rows)

                                                # Add total row
                                                total_row = [
                                                    'TOTAL',
                                                    f"{total_alloc_pct:.2f}%",
                                                    '',
                                                    '',
                                                    f"${total_value:,.2f}",
                                                    f"{total_port_pct:.2f}%"
                                                ]

                                                # Create table with proper formatting
                                                page_width = 7.5*inch
                                                col_widths = [page_width/len(headers)] * len(headers)
                                                alloc_table = Table([headers] + table_rows + [total_row], colWidths=col_widths)
                                                alloc_table.setStyle(TableStyle([
                                                    ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                                                    ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                                                    ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                                                    ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                                                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                                    ('LEFTPADDING', (0, 0), (-1, -1), 3),
                                                    ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                                                    ('TOPPADDING', (0, 0), (-1, -1), 2),
                                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                                                    ('WORDWRAP', (0, 0), (-1, -1), True),
                                                    # Style the total row
                                                    ('BACKGROUND', (0, -1), (-1, -1), reportlab_colors.Color(0.2, 0.4, 0.6)),
                                                    ('TEXTCOLOR', (0, -1), (-1, -1), reportlab_colors.whitesmoke),
                                                    ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
                                                ]))
                                                story.append(alloc_table)
                                                story.append(Spacer(1, 5))
                                                table_created = True
                                                break
                        except Exception as e:
                            pass
                    
                    # Method 2: Create table from today_weights directly
                    if not table_created:
                        try:
                            if today_weights:
                                headers = ['Asset', 'Allocation %']
                                table_rows = []
                                
                                for asset, weight in today_weights.items():
                                    if float(weight) > 0:
                                        table_rows.append([asset, f"{float(weight)*100:.2f}%"])
                                
                                if table_rows:
                                    # Calculate total values for summary row
                                    total_alloc_pct = sum(float(row[1].rstrip('%')) for row in table_rows)

                                    # Add total row
                                    total_row = [
                                        'TOTAL',
                                        f"{total_alloc_pct:.2f}%"
                                    ]

                                    page_width = 7.5*inch
                                    col_widths = [page_width/len(headers)] * len(headers)
                                    alloc_table = Table([headers] + table_rows + [total_row], colWidths=col_widths)
                                    alloc_table.setStyle(TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                                        ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                                        ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                        ('LEFTPADDING', (0, 0), (-1, -1), 3),
                                        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                                        ('TOPPADDING', (0, 0), (-1, -1), 2),
                                        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                                        ('WORDWRAP', (0, 0), (-1, -1), True),
                                        # Style the total row
                                        ('BACKGROUND', (0, -1), (-1, -1), reportlab_colors.Color(0.2, 0.4, 0.6)),
                                        ('TEXTCOLOR', (0, -1), (-1, -1), reportlab_colors.whitesmoke),
                                        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
                                    ]))
                                    story.append(alloc_table)
                                    story.append(Spacer(1, 5))
                                    table_created = True
                                else:
                                    story.append(Paragraph("No allocation data available", styles['Normal']))
                            else:
                                story.append(Paragraph("No allocation data available", styles['Normal']))
                        except Exception as e2:
                            story.append(Paragraph(f"Error creating allocation table: {str(e2)}", styles['Normal']))
                    
                    story.append(Spacer(1, 5))
                    
                    # Page break between portfolios (but only if not the last one)
                    if portfolio_count < len(portfolio_configs):
                        story.append(PageBreak())
                        
            except Exception as e:
                pass
                story.append(Paragraph(f"Error creating pie chart for {portfolio_name}: {str(e)}", styles['Normal']))
        
        # Update progress
        progress_bar.progress(95)
        status_text.text("🔨 Building PDF document...")
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Update progress
        progress_bar.progress(100)
        status_text.text("✅ PDF generation complete! Downloading...")
        
        return pdf_bytes
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

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

# Initialize page-specific session state for Strategy Comparison page
if 'strategy_comparison_page_initialized' not in st.session_state:
    st.session_state.strategy_comparison_page_initialized = True
    # Initialize strategy-comparison specific session state
    st.session_state.strategy_comparison_portfolio_configs = [
        # 1) Equal weight portfolio (no momentum) - baseline strategy
        {
            'name': 'Equal Weight (Baseline)',
            'stocks': [
                {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
            ],
            'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
            'added_amount': 10000,
            'added_frequency': 'Annually',
            'rebalancing_frequency': 'Annually',
            'start_date_user': None,
            'end_date_user': None,
            'start_with': 'all',
            'use_momentum': False,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [],
            'calc_beta': False,
            'calc_volatility': False,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
            'use_minimal_threshold': False,
            'minimal_threshold_percent': 4.0,
            'use_max_allocation': False,
            'max_allocation_percent': 20.0,
        },
        # 2) Momentum-based portfolio with Volatility adjustments
        {
            'name': 'Momentum Strategy + Volatility',
            'stocks': [
                {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
            ],
            'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
            'added_amount': 10000,
            'added_frequency': 'Annually',
            'rebalancing_frequency': 'Annually',
            'start_date_user': None,
            'end_date_user': None,
            'start_with': 'all',
            'first_rebalance_strategy': 'momentum_window_complete',
            'use_momentum': True,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [
                {'lookback': 365, 'exclude': 30, 'weight': 0.5},
                {'lookback': 180, 'exclude': 30, 'weight': 0.3},
                {'lookback': 120, 'exclude': 30, 'weight': 0.2},
            ],
            'calc_beta': False,
            'calc_volatility': True,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
            'use_minimal_threshold': False,
            'minimal_threshold_percent': 4.0,
            'use_max_allocation': False,
            'max_allocation_percent': 20.0,
        },
        # 3) Pure momentum strategy (no beta/volatility adjustments)
        {
            'name': 'Pure Momentum Strategy',
            'stocks': [
                {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
            ],
            'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
            'added_amount': 10000,
            'added_frequency': 'Annually',
            'rebalancing_frequency': 'Annually',
            'start_date_user': None,
            'end_date_user': None,
            'start_with': 'all',
            'first_rebalance_strategy': 'momentum_window_complete',
            'use_momentum': True,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [
                {'lookback': 365, 'exclude': 30, 'weight': 0.5},
                {'lookback': 180, 'exclude': 30, 'weight': 0.3},
                {'lookback': 120, 'exclude': 30, 'weight': 0.2},
            ],
            'calc_beta': False,
            'calc_volatility': False,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
            'use_minimal_threshold': False,
            'minimal_threshold_percent': 4.0,
            'use_max_allocation': False,
            'max_allocation_percent': 20.0,
        },
    ]
    st.session_state.strategy_comparison_active_portfolio_index = 0
    st.session_state.strategy_comparison_rerun_flag = False
    # Clean up any existing portfolio configs to remove unused settings
if 'strategy_comparison_portfolio_configs' in st.session_state:
    for config in st.session_state.strategy_comparison_portfolio_configs:
        if isinstance(config, dict):
            config.pop('use_relative_momentum', None)
            config.pop('equal_if_all_negative', None)

# Note: portfolio selection is initialized later when the selector is created

# -----------------------
# Default JSON configs (for initialization)
# -----------------------
default_configs = [
    # 1) Equal weight portfolio (no momentum) - baseline strategy
    {
        'name': 'Equal Weight (Baseline)',
        'stocks': [
            {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
        ],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 10000,
        'added_frequency': 'Annually',
        'rebalancing_frequency': 'Annually',
        'start_date_user': None,
        'end_date_user': None,
        'start_with': 'all',
        'use_momentum': False,
        'momentum_strategy': 'Classic',
        'negative_momentum_strategy': 'Cash',
        'momentum_windows': [],
        'calc_beta': False,
        'calc_volatility': False,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
    },
    # 2) Momentum-based portfolio with Volatility adjustments
    {
        'name': 'Momentum Strategy + Volatility',
        'stocks': [
            {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
        ],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 10000,
        'added_frequency': 'Annually',
        'rebalancing_frequency': 'Annually',
        'start_date_user': None,
        'end_date_user': None,
        'start_with': 'all',
        'first_rebalance_strategy': 'momentum_window_complete',
        'use_momentum': True,
        'momentum_strategy': 'Classic',
        'negative_momentum_strategy': 'Cash',
        'momentum_windows': [
            {'lookback': 365, 'exclude': 30, 'weight': 0.5},
            {'lookback': 180, 'exclude': 30, 'weight': 0.3},
            {'lookback': 120, 'exclude': 30, 'weight': 0.2},
        ],
        'calc_beta': False,
        'calc_volatility': True,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
    },
    # 3) Pure momentum strategy (no beta/volatility adjustments)
    {
        'name': 'Pure Momentum Strategy',
        'stocks': [
            {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
        ],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 10000,
        'added_frequency': 'Annually',
        'rebalancing_frequency': 'Annually',
        'start_date_user': None,
        'end_date_user': None,
        'start_with': 'all',
        'first_rebalance_strategy': 'momentum_window_complete',
        'use_momentum': True,
        'momentum_strategy': 'Classic',
        'negative_momentum_strategy': 'Cash',
        'momentum_windows': [
            {'lookback': 365, 'exclude': 30, 'weight': 0.5},
            {'lookback': 180, 'exclude': 30, 'weight': 0.3},
            {'lookback': 120, 'exclude': 30, 'weight': 0.2},
        ],
        'calc_beta': False,
        'calc_volatility': False,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
    },
]

st.set_page_config(layout="wide", page_title="Strategy Performance Comparison", page_icon="📈")

# Initialize global date widgets on page load to maintain state across page navigation
def initialize_global_dates():
    """Initialize global date widgets to maintain state across page navigation"""
    from datetime import date
    if "strategy_comparison_start_date" not in st.session_state:
        st.session_state["strategy_comparison_start_date"] = date(2010, 1, 1)
    if "strategy_comparison_end_date" not in st.session_state:
        st.session_state["strategy_comparison_end_date"] = date.today()
    if "strategy_comparison_use_custom_dates" not in st.session_state:
        st.session_state["strategy_comparison_use_custom_dates"] = False

initialize_global_dates()

# Handle imported values from JSON - MUST BE AT THE VERY BEGINNING
if "_import_start_with" in st.session_state:
    st.session_state["strategy_comparison_start_with"] = st.session_state.pop("_import_start_with")
    st.session_state["strategy_comparison_start_with_radio"] = st.session_state["strategy_comparison_start_with"]
if "_import_first_rebalance_strategy" in st.session_state:
    st.session_state["strategy_comparison_first_rebalance_strategy"] = st.session_state.pop("_import_first_rebalance_strategy")
    st.session_state["strategy_comparison_first_rebalance_strategy_radio"] = st.session_state["strategy_comparison_first_rebalance_strategy"]
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



# ...rest of the code...

# Place rerun logic after first portfolio input widget
active_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] if 'strategy_comparison_portfolio_configs' in st.session_state and 'strategy_comparison_active_portfolio_index' in st.session_state else None
if active_portfolio:
    ## Removed duplicate Portfolio Name input field
    pass  # Removed rerun logic to prevent full page refresh
import numpy as np
import pandas as pd
def calculate_mwrr(values, cash_flows, dates):
    # Exact logic from app.py for MWRR calculation
    try:
        from scipy.optimize import brentq
        values = pd.Series(values).dropna()
        flows = pd.Series(cash_flows).reindex(values.index, fill_value=0.0)
        if len(values) < 2:
            return np.nan
        dates = pd.to_datetime(values.index)
        start_date = dates[0]
        time_periods = np.array([(d - start_date).days / 365.25 for d in dates])
        initial_investment = -values.iloc[0]
        significant_flows = flows[flows != 0]
        cash_flow_dates = [start_date]
        cash_flow_amounts = [initial_investment]
        cash_flow_times = [0.0]
        for date, flow in significant_flows.items():
            if date != dates[0] and date != dates[-1]:
                cash_flow_dates.append(pd.to_datetime(date))
                cash_flow_amounts.append(flow)
                cash_flow_times.append((pd.to_datetime(date) - start_date).days / 365.25)
        cash_flow_dates.append(dates[-1])
        cash_flow_amounts.append(values.iloc[-1])
        cash_flow_times.append((dates[-1] - start_date).days / 365.25)
        cash_flow_amounts = np.array(cash_flow_amounts)
        cash_flow_times = np.array(cash_flow_times)
        def npv(rate):
            return np.sum(cash_flow_amounts / (1 + rate) ** cash_flow_times)
        try:
            irr = brentq(npv, -0.999, 10)
            return irr * 100
        except (ValueError, RuntimeError):
            return np.nan
    except Exception:
        return np.nan
    # Exact logic from app.py for MWRR calculation
    try:
        from scipy.optimize import brentq
        values = pd.Series(values).dropna()
        flows = pd.Series(cash_flows).reindex(values.index, fill_value=0.0)
        if len(values) < 2:
            return np.nan
        dates = pd.to_datetime(values.index)
        start_date = dates[0]
        time_periods = np.array([(d - start_date).days / 365.25 for d in dates])
        initial_investment = -values.iloc[0]
        significant_flows = flows[flows != 0]
        cash_flow_dates = [start_date]
        cash_flow_amounts = [initial_investment]
        cash_flow_times = [0.0]
        for date, flow in significant_flows.items():
            if date != dates[0] and date != dates[-1]:
                cash_flow_dates.append(pd.to_datetime(date))
                cash_flow_amounts.append(flow)
                cash_flow_times.append((pd.to_datetime(date) - start_date).days / 365.25)
        cash_flow_dates.append(dates[-1])
        cash_flow_amounts.append(values.iloc[-1])
        cash_flow_times.append((dates[-1] - start_date).days / 365.25)
        cash_flow_amounts = np.array(cash_flow_amounts)
        cash_flow_times = np.array(cash_flow_times)
        def npv(rate):
            return np.sum(cash_flow_amounts / (1 + rate) ** cash_flow_times)
        try:
            irr = brentq(npv, -0.999, 10)
            return irr
        except (ValueError, RuntimeError):
            return np.nan
    except Exception:
        return np.nan
# Backtest_Engine.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io
import contextlib
import json
from datetime import datetime, timedelta
from warnings import warn
from scipy.optimize import newton, brentq, root_scalar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

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

st.set_page_config(layout="wide", page_title="Strategy Performance Comparison")

st.title("Strategy Comparison (NO_CACHE)")
st.markdown("Use the forms below to configure and run backtests for multiple portfolios.")

# Simple performance toggle
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 🚀 Performance Settings")
with col2:
    use_parallel = st.checkbox("Parallel Processing", value=True,
                              help="Process multiple portfolios simultaneously for faster execution")
    st.session_state.use_parallel_processing = use_parallel



# Handle rerun flag first (exact same as Multi Backtest)
active_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] if 'strategy_comparison_portfolio_configs' in st.session_state and 'strategy_comparison_active_portfolio_index' in st.session_state else None
if active_portfolio:
    if st.session_state.get('strategy_comparison_rerun_flag', False):
        st.session_state.strategy_comparison_rerun_flag = False
        st.rerun()

# -----------------------
# Helper functions
# -----------------------
def get_trading_days(start_date, end_date):
    return pd.bdate_range(start=start_date, end=end_date)

def sort_dataframe_numerically(df, column, ascending=False):
    """Sort DataFrame by a specific column numerically, handling percentage strings and N/A values"""
    if column not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    df_sorted = df.copy()
    
    # Extract numeric values for sorting
    def extract_numeric_value(value):
        if pd.isna(value) or value == 'N/A' or value == 'N/A%':
            return float('-inf')  # Put N/A values at the end
        
        # Handle percentage strings
        if isinstance(value, str) and value.endswith('%'):
            try:
                return float(value.replace('%', ''))
            except:
                return float('-inf')
        
        # Handle regular numbers
        try:
            return float(value)
        except:
            return float('-inf')
    
    # Create sorting key
    df_sorted['_sort_key'] = df_sorted[column].apply(extract_numeric_value)
    
    # Sort by the numeric key
    df_sorted = df_sorted.sort_values('_sort_key', ascending=ascending)
    
    # Drop the sorting key
    df_sorted = df_sorted.drop('_sort_key', axis=1)
    
    return df_sorted

def get_dates_by_freq(freq, start, end, market_days):
    market_days = sorted(market_days)
    
    # Ensure market_days are timezone-naive for consistent comparison
    market_days_naive = [d.tz_localize(None) if d.tz is not None else d for d in market_days]
    
    if freq == "market_day":
        return set(market_days)
    elif freq == "calendar_day":
        return set(pd.date_range(start=start, end=end, freq='D'))
    elif freq == "Weekly":
        base = pd.date_range(start=start, end=end, freq='W-MON')
    elif freq == "Biweekly":
        base = pd.date_range(start=start, end=end, freq='2W-MON')
    elif freq == "Monthly":
        # Fixed calendar dates: 1st of each month
        monthly = []
        for y in range(start.year, end.year + 1):
            for m in range(1, 13):
                monthly.append(pd.Timestamp(year=y, month=m, day=1))
        base = pd.DatetimeIndex(monthly)
    elif freq == "Quarterly":
        # Fixed calendar dates: 1st of each quarter (Jan 1, Apr 1, Jul 1, Oct 1)
        quarterly = []
        for y in range(start.year, end.year + 1):
            for m in [1, 4, 7, 10]:  # Q1, Q2, Q3, Q4
                quarterly.append(pd.Timestamp(year=y, month=m, day=1))
        base = pd.DatetimeIndex(quarterly)
    elif freq == "Semiannually":
        # First day of Jan and Jul each year
        semi = []
        for y in range(start.year, end.year + 1):
            for m in [1, 7]:
                semi.append(pd.Timestamp(year=y, month=m, day=1))
        base = pd.DatetimeIndex(semi)
    elif freq == "Annually" or freq == "year":
        base = pd.date_range(start=start, end=end, freq='YS')
    elif freq == "Never" or freq == "none" or freq is None:
        return set()
    elif freq == "Buy & Hold" or freq == "Buy & Hold (Target)":
        # Buy & Hold options don't have specific rebalancing dates - they rebalance immediately when cash is available
        return set()
    else:
        raise ValueError(f"Unknown frequency: {freq}")

    dates = []
    for d in base:
        # Ensure d is timezone-naive for comparison
        d_naive = d.tz_localize(None) if d.tz is not None else d
        idx = np.searchsorted(market_days_naive, d_naive, side='right')
        if idx > 0 and market_days_naive[idx-1] >= d_naive:
            dates.append(market_days[idx-1])  # Use original market_days for return
        elif idx < len(market_days_naive):
            dates.append(market_days[idx])  # Use original market_days for return
    return set(dates)

def get_cached_rebalancing_dates(portfolio_name, rebalancing_frequency, sim_index):
    """Get rebalancing dates with caching to avoid recalculation"""
    cache_key = f"rebalancing_dates_{portfolio_name}_{rebalancing_frequency}"
    portfolio_rebalancing_dates = st.session_state.get(cache_key)
    
    if portfolio_rebalancing_dates is None and sim_index is not None:
        # Calculate and cache the rebalancing dates
        portfolio_rebalancing_dates = get_dates_by_freq(rebalancing_frequency, sim_index[0], sim_index[-1], sim_index)
        st.session_state[cache_key] = portfolio_rebalancing_dates
    
    return portfolio_rebalancing_dates

def calculate_cagr(values, dates):
    if len(values) < 2:
        return np.nan
    start_val = values[0]
    end_val = values[-1]
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0 or start_val == 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1

def calculate_max_drawdown(values):
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / np.where(peak == 0, 1, peak)
    return np.nanmin(drawdowns), drawdowns

def calculate_volatility(returns):
    # Annualized volatility
    return np.std(returns) * np.sqrt(365) if len(returns) > 1 else np.nan

def calculate_beta(returns, benchmark_returns):
    # Use exact logic from app.py
    portfolio_returns = pd.Series(returns)
    benchmark_returns = pd.Series(benchmark_returns)
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

# FIXED: Correct Sortino Ratio calculation - EXACTLY like Backtest_Engine.py
def calculate_sortino(returns, risk_free_rate):
    """Calculates the Sortino ratio."""
    # Create a constant risk-free rate series aligned with returns
    daily_rf_rate = risk_free_rate / 365.25
    rf_series = pd.Series(daily_rf_rate, index=returns.index)
    
    aligned_returns, aligned_rf = returns.align(rf_series, join='inner')
    if aligned_returns.empty:
        return np.nan
        
    downside_returns = aligned_returns[aligned_returns < aligned_rf]
    if downside_returns.empty or downside_returns.std() == 0:
        # If no downside returns, Sortino is infinite or undefined.
        # We can return nan or a very high value. nan is safer.
        return np.nan
    
    downside_std = downside_returns.std()
    
    return (aligned_returns.mean() - aligned_rf.mean()) / downside_std * np.sqrt(365)

# FIXED: Correct Sharpe ratio calculation - EXACTLY like Backtest_Engine.py
def calculate_sharpe(returns, risk_free_rate):
    """Calculates the Sharpe ratio."""
    # Create a constant risk-free rate series aligned with returns
    daily_rf_rate = risk_free_rate / 365.25
    rf_series = pd.Series(daily_rf_rate, index=returns.index)
    
    aligned_returns, aligned_rf = returns.align(rf_series, join='inner')
    if aligned_returns.empty:
        return np.nan
    
    excess_returns = aligned_returns - aligned_rf
    if excess_returns.std() == 0:
        return np.nan
        
    return excess_returns.mean() / excess_returns.std() * np.sqrt(365)

# FIXED: Correct Ulcer Index calculation - EXACTLY like Backtest_Engine.py
def calculate_ulcer_index(series):
    """Calculates the Ulcer Index (average squared percent drawdown, then sqrt)."""
    if series.empty:
        return np.nan
    peak = series.expanding(min_periods=1).max()
    drawdown = (series - peak) / peak * 100  # percent drawdown
    drawdown_sq = drawdown ** 2
    return np.sqrt(drawdown_sq.mean())

# FIXED: Correct UPI calculation - EXACTLY like Backtest_Engine.py
def calculate_upi(cagr, ulcer_index):
    """Calculates the Ulcer Performance Index (UPI = CAGR / Ulcer Index, both as decimals)."""
    if ulcer_index is None or pd.isna(ulcer_index) or ulcer_index == 0:
        return np.nan
    return cagr / (ulcer_index / 100)

def calculate_total_money_added(config, start_date, end_date):
    """Calculate total money added to portfolio (initial + periodic additions)"""
    if start_date is None or end_date is None:
        return np.nan
    
    # Initial investment
    initial_value = config.get('initial_value', 0)
    
    # Calculate periodic additions
    added_amount = config.get('added_amount', 0)
    added_frequency = config.get('added_frequency', 'None')
    
    if added_frequency in ['None', 'Never', 'none', None] or added_amount == 0:
        return initial_value
    
    # Get dates when additions were made
    dates_added = get_dates_by_freq(added_frequency, start_date, end_date, pd.date_range(start_date, end_date, freq='D'))
    
    # Count additions (excluding the first date which is initial investment)
    num_additions = len([d for d in dates_added if d != start_date])
    total_additions = num_additions * added_amount
    
    return initial_value + total_additions


# -----------------------
# Single-backtest core (adapted from your code, robust)
# -----------------------
def single_backtest(config, sim_index, reindexed_data, _cache_version="v2_daily_allocations"):
    stocks_list = config['stocks']
    tickers = [s['ticker'] for s in stocks_list if s['ticker']]
    # Filter tickers to those present in reindexed_data to avoid KeyErrors for invalid tickers
    available_tickers = [t for t in tickers if t in reindexed_data]
    if len(available_tickers) < len(tickers):
        missing = set(tickers) - set(available_tickers)
        # Warning: Some tickers not found in price data
    tickers = available_tickers
    # Recompute allocations and include_dividends to only include valid tickers
    # Handle duplicate tickers by summing their allocations
    allocations = {}
    include_dividends = {}
    for s in stocks_list:
        if s.get('ticker') and s.get('ticker') in tickers:
            ticker = s['ticker']
            allocation = s.get('allocation', 0)
            include_div = s.get('include_dividends', False)
            
            if ticker in allocations:
                # If ticker already exists, add the allocation
                allocations[ticker] += allocation
                # For include_dividends, use True if any instance has it True
                include_dividends[ticker] = include_dividends[ticker] or include_div
            else:
                # First occurrence of this ticker
                allocations[ticker] = allocation
                include_dividends[ticker] = include_div
    
    # Update tickers to only include unique tickers after deduplication
    tickers = list(allocations.keys())
    benchmark_ticker = config['benchmark_ticker']
    initial_value = config.get('initial_value', 0)
    added_amount = config.get('added_amount', 0)
    added_frequency = config.get('added_frequency', 'none')
    rebalancing_frequency = config.get('rebalancing_frequency', 'none')
    use_momentum = config.get('use_momentum', True)
    momentum_windows = config.get('momentum_windows', [])
    calc_beta = config.get('calc_beta', False)
    calc_volatility = config.get('calc_volatility', False)
    beta_window_days = config.get('beta_window_days', 365)
    exclude_days_beta = config.get('exclude_days_beta', 30)
    vol_window_days = config.get('vol_window_days', 365)
    exclude_days_vol = config.get('exclude_days_vol', 30)
    current_data = {t: reindexed_data[t] for t in tickers + [benchmark_ticker] if t in reindexed_data}
    dates_added = get_dates_by_freq(added_frequency, sim_index[0], sim_index[-1], sim_index)
    
    # Get regular rebalancing dates
    dates_rebal = sorted(get_dates_by_freq(rebalancing_frequency, sim_index[0], sim_index[-1], sim_index))
    
    # Handle first rebalance strategy - ensure first rebalance happens after momentum window completes
    first_rebalance_strategy = st.session_state.get('strategy_comparison_first_rebalance_strategy', 'rebalancing_date')
    if use_momentum and momentum_windows:
        try:
            # Calculate when momentum window completes
            window_sizes = [int(w.get('lookback', 0)) for w in momentum_windows if w is not None]
            max_window_days = max(window_sizes) if window_sizes else 0
            momentum_completion_date = sim_index[0] + pd.Timedelta(days=max_window_days)
            
            # Find the closest trading day to momentum completion
            momentum_completion_trading_day = sim_index[sim_index >= momentum_completion_date][0] if len(sim_index[sim_index >= momentum_completion_date]) > 0 else sim_index[-1]
            
            if first_rebalance_strategy == "momentum_window_complete":
                # Replace the first rebalancing date with momentum completion date
                if len(dates_rebal) > 0:
                    # Remove the first rebalancing date and add momentum completion date
                    dates_rebal = dates_rebal[1:] if len(dates_rebal) > 1 else []
                    dates_rebal.insert(0, momentum_completion_trading_day)
                    dates_rebal = sorted(dates_rebal)
            else:  # first_rebalance_strategy == "rebalancing_date"
                # For rebalancing_date strategy, use the first available rebalancing date
                # Don't filter out early dates - let it start as soon as possible
                pass
        except Exception:
            pass  # Fall back to regular rebalancing dates

    # Dictionaries to store historical data for new tables
    historical_allocations = {}
    historical_metrics = {}

    def calculate_momentum(date, current_assets, momentum_windows):
        cumulative_returns, valid_assets = {}, []
        filtered_windows = [w for w in momentum_windows if w["weight"] > 0]
        # Normalize weights so they sum to 1 (same as app.py)
        total_weight = sum(w["weight"] for w in filtered_windows)
        if total_weight == 0:
            normalized_weights = [0 for _ in filtered_windows]
        else:
            normalized_weights = [w["weight"] / total_weight for w in filtered_windows]
        start_dates_config = {t: reindexed_data[t].first_valid_index() for t in tickers if t in reindexed_data}
        for t in current_assets:
            is_valid, asset_returns = True, 0.0
            for idx, window in enumerate(filtered_windows):
                lookback, exclude = window["lookback"], window["exclude"]
                weight = normalized_weights[idx]
                start_mom = date - pd.Timedelta(days=lookback)
                end_mom = date - pd.Timedelta(days=exclude)
                if start_dates_config.get(t, pd.Timestamp.max) > start_mom:
                    is_valid = False; break
                df_t = current_data[t]
                price_start_index = df_t.index.asof(start_mom)
                price_end_index = df_t.index.asof(end_mom)
                if pd.isna(price_start_index) or pd.isna(price_end_index):
                    is_valid = False; break
                price_start = df_t.loc[price_start_index, "Close"]
                price_end = df_t.loc[price_end_index, "Close"]
                if pd.isna(price_start) or pd.isna(price_end) or price_start == 0:
                    is_valid = False; break
                ret = (price_end - price_start) / price_start
                asset_returns += ret * weight
            if is_valid:
                cumulative_returns[t] = asset_returns
                valid_assets.append(t)
        return cumulative_returns, valid_assets

    def calculate_momentum_weights(returns, valid_assets, date, momentum_strategy='Classic', negative_momentum_strategy='Cash'):
        # Mirror approach used in allocations/app.py: compute weights from raw momentum
        # (Classic or Relative) and then optionally post-filter by inverse volatility
        # and inverse absolute beta (multiplicative), then renormalize. This avoids
        # dividing by beta directly which flips signs when beta is negative.
        if not valid_assets:
            return {}, {}
        # Keep only non-nan momentum values
        rets = {t: returns.get(t, np.nan) for t in valid_assets}
        rets = {t: rets[t] for t in rets if not pd.isna(rets[t])}
        if not rets:
            return {}, {}

        metrics = {t: {} for t in rets.keys()}

        # compute beta and volatility metrics when requested
        beta_vals = {}
        vol_vals = {}
        df_bench = current_data.get(benchmark_ticker)
        if calc_beta:
            start_beta = date - pd.Timedelta(days=beta_window_days)
            end_beta = date - pd.Timedelta(days=exclude_days_beta)
        if calc_volatility:
            start_vol = date - pd.Timedelta(days=vol_window_days)
            end_vol = date - pd.Timedelta(days=exclude_days_vol)

        for t in list(rets.keys()):
            df_t = current_data.get(t)
            if calc_beta and df_bench is not None and isinstance(df_t, pd.DataFrame):
                mask_beta = (df_t.index >= start_beta) & (df_t.index <= end_beta)
                returns_t_beta = df_t.loc[mask_beta, 'Price_change']
                mask_bench_beta = (df_bench.index >= start_beta) & (df_bench.index <= end_beta)
                returns_bench_beta = df_bench.loc[mask_bench_beta, 'Price_change']
                if len(returns_t_beta) < 2 or len(returns_bench_beta) < 2:
                    beta_vals[t] = np.nan
                else:
                    variance = np.var(returns_bench_beta)
                    beta_vals[t] = (np.cov(returns_t_beta, returns_bench_beta)[0,1] / variance) if variance > 0 else np.nan
                metrics[t]['Beta'] = beta_vals[t]
            if calc_volatility and isinstance(df_t, pd.DataFrame):
                mask_vol = (df_t.index >= start_vol) & (df_t.index <= end_vol)
                returns_t_vol = df_t.loc[mask_vol, 'Price_change']
                if len(returns_t_vol) < 2:
                    vol_vals[t] = np.nan
                else:
                    vol_vals[t] = returns_t_vol.std() * np.sqrt(365)
                metrics[t]['Volatility'] = vol_vals[t]

        # attach raw momentum
        for t in rets:
            metrics[t]['Momentum'] = rets[t]

        # Build initial weights from raw momentum (Classic or Relative)
        weights = {}
        rets_keys = list(rets.keys())
        all_negative = all(rets[t] <= 0 for t in rets_keys)
        relative_mode = isinstance(momentum_strategy, str) and momentum_strategy.lower().startswith('relat')

        if all_negative:
            if negative_momentum_strategy == 'Cash':
                weights = {t: 0.0 for t in rets_keys}
            elif negative_momentum_strategy == 'Equal weight':
                weights = {t: 1.0 / len(rets_keys) for t in rets_keys}
            else:  # Relative momentum
                min_score = min(rets[t] for t in rets_keys)
                offset = -min_score + 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
        else:
            if relative_mode:
                min_score = min(rets[t] for t in rets_keys)
                offset = -min_score + 0.01 if min_score < 0 else 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
            else:
                positive_scores = {t: rets[t] for t in rets_keys if rets[t] > 0}
                if positive_scores:
                    ssum = sum(positive_scores.values())
                    weights = {t: (positive_scores.get(t, 0.0) / ssum) for t in rets_keys}
                else:
                    weights = {t: 0.0 for t in rets_keys}

        # Post-filtering: multiply weights by inverse vol and inverse |beta| when requested
        if (calc_volatility or calc_beta) and weights:
            filter_scores = {}
            for t in weights:
                score = 1.0
                if calc_volatility:
                    v = metrics.get(t, {}).get('Volatility', np.nan)
                    if not pd.isna(v) and v > 0:
                        score *= 1.0 / v
                if calc_beta:
                    b = metrics.get(t, {}).get('Beta', np.nan)
                    if not pd.isna(b) and b != 0:
                        score *= 1.0 / abs(b)
                filter_scores[t] = score

            filtered = {t: weights.get(t, 0.0) * filter_scores.get(t, 1.0) for t in weights}
            ssum = sum(filtered.values())
            # If filtering removes all weight (sum==0), fall back to unfiltered weights
            if ssum > 0:
                weights = {t: filtered[t] / ssum for t in filtered}

        # Apply allocation filters in correct order: Max Allocation -> Min Threshold -> Max Allocation (two-pass system)
        use_max_allocation = config.get('use_max_allocation', False)
        max_allocation_percent = config.get('max_allocation_percent', 10.0)
        use_threshold = config.get('use_minimal_threshold', False)
        threshold_percent = config.get('minimal_threshold_percent', 2.0)
        
        if use_max_allocation and weights:
            max_allocation_decimal = max_allocation_percent / 100.0
            
            # FIRST PASS: Apply maximum allocation filter
            capped_weights = {}
            excess_weight = 0.0
            
            for ticker, weight in weights.items():
                # CASH is exempt from max_allocation limit to prevent money loss
                if ticker == 'CASH':
                    capped_weights[ticker] = weight
                elif weight > max_allocation_decimal:
                    # Cap the weight and collect excess
                    capped_weights[ticker] = max_allocation_decimal
                    excess_weight += (weight - max_allocation_decimal)
                else:
                    # Keep original weight
                    capped_weights[ticker] = weight
            
            # Redistribute excess weight proportionally among stocks that are below the cap
            if excess_weight > 0:
                # Find stocks that can receive more weight (below the cap)
                eligible_stocks = {ticker: weight for ticker, weight in capped_weights.items() 
                                 if weight < max_allocation_decimal}
                
                if eligible_stocks:
                    # Calculate total weight of eligible stocks
                    total_eligible_weight = sum(eligible_stocks.values())
                    
                    if total_eligible_weight > 0:
                        # Redistribute excess proportionally
                        for ticker in eligible_stocks:
                            proportion = eligible_stocks[ticker] / total_eligible_weight
                            additional_weight = excess_weight * proportion
                            new_weight = capped_weights[ticker] + additional_weight
                            
                            # Make sure we don't exceed the cap
                            capped_weights[ticker] = min(new_weight, max_allocation_decimal)
            
            weights = capped_weights
            
            # Final normalization to 100% in case not enough stocks to distribute excess
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
        
        # Apply minimal threshold filter if enabled
        if use_threshold and weights:
            threshold_decimal = threshold_percent / 100.0
            
            # Check which stocks are below threshold after max allocation redistribution
            filtered_weights = {}
            for ticker, weight in weights.items():
                if weight >= threshold_decimal:
                    # Keep stocks above or equal to threshold (remove stocks below threshold)
                    filtered_weights[ticker] = weight
            
            # Normalize remaining stocks to sum to 1.0
            if filtered_weights:
                total_weight = sum(filtered_weights.values())
                if total_weight > 0:
                    weights = {ticker: weight / total_weight for ticker, weight in filtered_weights.items()}
                else:
                    weights = {}
            else:
                # If no stocks meet threshold, keep original weights
                weights = weights
        
        # SECOND PASS: Apply maximum allocation filter again (in case normalization created new excess)
        if use_max_allocation and weights:
            max_allocation_decimal = max_allocation_percent / 100.0
            
            # Check if any stocks exceed the cap after threshold filtering and normalization
            capped_weights = {}
            excess_weight = 0.0
            
            for ticker, weight in weights.items():
                # CASH is exempt from max_allocation limit to prevent money loss
                if ticker == 'CASH':
                    capped_weights[ticker] = weight
                elif weight > max_allocation_decimal:
                    # Cap the weight and collect excess
                    capped_weights[ticker] = max_allocation_decimal
                    excess_weight += (weight - max_allocation_decimal)
                else:
                    # Keep original weight
                    capped_weights[ticker] = weight
            
            # Redistribute excess weight proportionally among stocks that are below the cap
            if excess_weight > 0:
                # Find stocks that can receive more weight (below the cap)
                eligible_stocks = {ticker: weight for ticker, weight in capped_weights.items() 
                                 if weight < max_allocation_decimal}
                
                if eligible_stocks:
                    # Calculate total weight of eligible stocks
                    total_eligible_weight = sum(eligible_stocks.values())
                    
                    if total_eligible_weight > 0:
                        # Redistribute excess proportionally
                        for ticker in eligible_stocks:
                            proportion = eligible_stocks[ticker] / total_eligible_weight
                            additional_weight = excess_weight * proportion
                            new_weight = capped_weights[ticker] + additional_weight
                            
                            # Make sure we don't exceed the cap
                            capped_weights[ticker] = min(new_weight, max_allocation_decimal)
            
            weights = capped_weights
            
            # Final normalization to 100% in case not enough stocks to distribute excess
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {ticker: weight / total_weight for ticker, weight in weights.items()}

        # Attach calculated weights to metrics and return
        for t in weights:
            metrics[t]['Calculated_Weight'] = weights.get(t, 0.0)

        # Debug print when beta/vol are used
        if calc_beta or calc_volatility:
            try:
                for t in rets_keys:
                    pass  # Momentum calculation completed
            except Exception:
                pass

        return weights, metrics
        # --- MODIFIED LOGIC END ---

    values = {t: [0.0] for t in tickers}
    unallocated_cash = [0.0]
    unreinvested_cash = [0.0]
    portfolio_no_additions = [initial_value]
    
    # Initial allocation and metric storage
    if not use_momentum:
        current_allocations = {t: allocations.get(t,0) for t in tickers}
        
        # Apply allocation filters in correct order: Max Allocation -> Min Threshold -> Max Allocation (two-pass system)
        use_max_allocation = config.get('use_max_allocation', False)
        max_allocation_percent = config.get('max_allocation_percent', 10.0)
        use_threshold = config.get('use_minimal_threshold', False)
        threshold_percent = config.get('minimal_threshold_percent', 2.0)
        
        if use_max_allocation and current_allocations:
            max_allocation_decimal = max_allocation_percent / 100.0
            
            # FIRST PASS: Apply maximum allocation filter
            capped_allocations = {}
            excess_allocation = 0.0
            
            for ticker, allocation in current_allocations.items():
                # CASH is exempt from max_allocation limit to prevent money loss
                if ticker == 'CASH':
                    capped_allocations[ticker] = allocation
                elif allocation > max_allocation_decimal:
                    # Cap the allocation and collect excess
                    capped_allocations[ticker] = max_allocation_decimal
                    excess_allocation += (allocation - max_allocation_decimal)
                else:
                    # Keep original allocation
                    capped_allocations[ticker] = allocation
            
            # Redistribute excess allocation proportionally among stocks that are below the cap
            if excess_allocation > 0:
                # Find stocks that can receive more allocation (below the cap)
                eligible_stocks = {ticker: allocation for ticker, allocation in capped_allocations.items() 
                                 if allocation < max_allocation_decimal}
                
                if eligible_stocks:
                    # Calculate total allocation of eligible stocks
                    total_eligible_allocation = sum(eligible_stocks.values())
                    
                    if total_eligible_allocation > 0:
                        # Redistribute excess proportionally
                        for ticker in eligible_stocks:
                            proportion = eligible_stocks[ticker] / total_eligible_allocation
                            additional_allocation = excess_allocation * proportion
                            new_allocation = capped_allocations[ticker] + additional_allocation
                            
                            # Make sure we don't exceed the cap
                            capped_allocations[ticker] = min(new_allocation, max_allocation_decimal)
            
            current_allocations = capped_allocations
            
            # Final normalization to 100% in case not enough stocks to distribute excess
            total_alloc = sum(current_allocations.values())
            if total_alloc > 0:
                current_allocations = {ticker: allocation / total_alloc for ticker, allocation in current_allocations.items()}
        
        # Apply minimal threshold filter for non-momentum strategies
        if use_threshold and current_allocations:
            threshold_decimal = threshold_percent / 100.0
            
            # First: Filter out stocks below threshold
            filtered_allocations = {}
            for ticker, allocation in current_allocations.items():
                if allocation >= threshold_decimal:
                    # Keep stocks above or equal to threshold
                    filtered_allocations[ticker] = allocation
            
            # Then: Normalize remaining stocks to sum to 1
            if filtered_allocations:
                total_allocation = sum(filtered_allocations.values())
                if total_allocation > 0:
                    current_allocations = {ticker: allocation / total_allocation for ticker, allocation in filtered_allocations.items()}
                else:
                    current_allocations = {}
            else:
                # If no stocks meet threshold, keep original allocations
                current_allocations = current_allocations
        
        # SECOND PASS: Apply maximum allocation filter again (in case normalization created new excess)
        if use_max_allocation and current_allocations:
            max_allocation_decimal = max_allocation_percent / 100.0
            
            # Check if any stocks exceed the cap after threshold filtering and normalization
            capped_allocations = {}
            excess_allocation = 0.0
            
            for ticker, allocation in current_allocations.items():
                # CASH is exempt from max_allocation limit to prevent money loss
                if ticker == 'CASH':
                    capped_allocations[ticker] = allocation
                elif allocation > max_allocation_decimal:
                    # Cap the allocation and collect excess
                    capped_allocations[ticker] = max_allocation_decimal
                    excess_allocation += (allocation - max_allocation_decimal)
                else:
                    # Keep original allocation
                    capped_allocations[ticker] = allocation
            
            # Redistribute excess allocation proportionally among stocks that are below the cap
            if excess_allocation > 0:
                # Find stocks that can receive more allocation (below the cap)
                eligible_stocks = {ticker: allocation for ticker, allocation in capped_allocations.items() 
                                 if allocation < max_allocation_decimal}
                
                if eligible_stocks:
                    # Calculate total allocation of eligible stocks
                    total_eligible_allocation = sum(eligible_stocks.values())
                    
                    if total_eligible_allocation > 0:
                        # Redistribute excess proportionally
                        for ticker in eligible_stocks:
                            proportion = eligible_stocks[ticker] / total_eligible_allocation
                            additional_allocation = excess_allocation * proportion
                            new_allocation = capped_allocations[ticker] + additional_allocation
                            
                            # Make sure we don't exceed the cap
                            capped_allocations[ticker] = min(new_allocation, max_allocation_decimal)
            
            current_allocations = capped_allocations
    else:
        returns, valid_assets = calculate_momentum(sim_index[0], set(tickers), momentum_windows)
        current_allocations, metrics_on_rebal = calculate_momentum_weights(
            returns, valid_assets, date=sim_index[0],
            momentum_strategy=config.get('momentum_strategy', 'Classic'),
            negative_momentum_strategy=config.get('negative_momentum_strategy', 'Cash')
        )
        historical_metrics[sim_index[0]] = metrics_on_rebal
    
    sum_alloc = sum(current_allocations.get(t,0) for t in tickers)
    if sum_alloc > 0:
        for t in tickers:
            values[t][0] = initial_value * current_allocations.get(t,0) / sum_alloc
        unallocated_cash[0] = 0
    else:
        unallocated_cash[0] = initial_value
    
    historical_allocations[sim_index[0]] = {t: values[t][0] / initial_value if initial_value > 0 else 0 for t in tickers}
    historical_allocations[sim_index[0]]['CASH'] = unallocated_cash[0] / initial_value if initial_value > 0 else 0
    
    for i in range(len(sim_index)):
        date = sim_index[i]
        if i == 0: continue
        
        date_prev = sim_index[i-1]
        total_unreinvested_dividends = 0
        total_portfolio_prev = sum(values[t][-1] for t in tickers) + unreinvested_cash[-1]
        daily_growth_factor = 1
        if total_portfolio_prev > 0:
            total_portfolio_current_before_changes = 0
            for t in tickers:
                df = reindexed_data[t]
                price_prev = df.loc[date_prev, "Close"]
                val_prev = values[t][-1]
                nb_shares = val_prev / price_prev if price_prev > 0 else 0
                # --- Dividend fix: find the correct trading day for dividend ---
                div = 0.0
                # CRITICAL FIX: For leveraged tickers, get dividends from the base ticker, not the leveraged ticker
                if "?L=" in t:
                    # For leveraged tickers, get dividend data from the base ticker
                    base_ticker, leverage = parse_leverage_ticker(t)
                    if base_ticker in reindexed_data:
                        base_df = reindexed_data[base_ticker]
                        if "Dividends" in base_df.columns:
                            if date in base_df.index:
                                div = base_df.loc[date, "Dividends"]
                            else:
                                # Find next trading day in index after 'date'
                                future_dates = base_df.index[base_df.index > date]
                                if len(future_dates) > 0:
                                    div = base_df.loc[future_dates[0], "Dividends"]
                else:
                    # For regular tickers, get dividend data normally
                    if "Dividends" in df.columns:
                        if date in df.index:
                            div = df.loc[date, "Dividends"]
                        else:
                            # Find next trading day in index after 'date'
                            future_dates = df.index[df.index > date]
                            if len(future_dates) > 0:
                                div = df.loc[future_dates[0], "Dividends"]
                var = df.loc[date, "Price_change"] if date in df.index else 0.0
                
                # Expense ratio is already applied in apply_daily_leverage() when data is fetched
                # No need to re-apply it here (would be double application + slow loop)
                
                if include_dividends.get(t, False):
                    # Check if dividends should be collected as cash instead of reinvested
                    collect_as_cash = config.get('collect_dividends_as_cash', False)
                    if collect_as_cash and div > 0:
                        # Calculate dividend cash and add to unreinvested cash
                        nb_shares = val_prev / price_prev if price_prev > 0 else 0
                        dividend_cash = nb_shares * div
                        total_unreinvested_dividends += dividend_cash
                        # Don't include dividend in rate of return
                        rate_of_return = var
                        val_new = val_prev * (1 + rate_of_return)
                    else:
                        # Reinvest dividends (original behavior)
                        # CRITICAL FIX: For leveraged tickers, dividends should be handled differently
                        # When simulating leveraged ETFs, the dividend RATE should be the same as the base asset
                        if "?L=" in t:
                            # For leveraged tickers, get the base ticker's dividend rate (not amount)
                            base_ticker, leverage = parse_leverage_ticker(t)
                            if base_ticker in reindexed_data:
                                base_df = reindexed_data[base_ticker]
                                base_price_prev = base_df.loc[date_prev, "Close"]
                                # Use the base ticker's dividend rate, not the leveraged amount
                                dividend_rate = div / base_price_prev if base_price_prev > 0 else 0
                                rate_of_return = var + dividend_rate
                            else:
                                # Fallback: use leveraged price (may cause issues)
                                rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                        else:
                            # For regular tickers, use normal dividend reinvestment
                            rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                        val_new = val_prev * (1 + rate_of_return)
                else:
                    val_new = val_prev * (1 + var)
                    # If dividends are not included, do NOT add to unreinvested cash or anywhere else
                total_portfolio_current_before_changes += val_new
            total_portfolio_current_before_changes += unreinvested_cash[-1] + total_unreinvested_dividends
            daily_growth_factor = total_portfolio_current_before_changes / total_portfolio_prev
        for t in tickers:
            df = reindexed_data[t]
            price_prev = df.loc[date_prev, "Close"]
            val_prev = values[t][-1]
            # --- Dividend fix: find the correct trading day for dividend ---
            div = 0.0
            # CRITICAL FIX: For leveraged tickers, get dividends from the base ticker, not the leveraged ticker
            if "?L=" in t:
                # For leveraged tickers, get dividend data from the base ticker
                base_ticker, leverage = parse_leverage_ticker(t)
                if base_ticker in reindexed_data:
                    base_df = reindexed_data[base_ticker]
                    if "Dividends" in base_df.columns:
                        if date in base_df.index:
                            div = base_df.loc[date, "Dividends"]
                        else:
                            # Find next trading day in index after 'date'
                            future_dates = base_df.index[base_df.index > date]
                            if len(future_dates) > 0:
                                div = base_df.loc[future_dates[0], "Dividends"]
            else:
                # For regular tickers, get dividend data normally
                if "Dividends" in df.columns:
                    if date in df.index:
                        div = df.loc[date, "Dividends"]
                    else:
                        # Find next trading day in index after 'date'
                        future_dates = df.index[df.index > date]
                        if len(future_dates) > 0:
                            div = df.loc[future_dates[0], "Dividends"]
            var = df.loc[date, "Price_change"] if date in df.index else 0.0
            
            # Expense ratio is already applied in apply_daily_leverage() when data is fetched
            # No need to re-apply it here (would be double application + slow loop)
            
            if include_dividends.get(t, False):
                # Check if dividends should be collected as cash instead of reinvested
                collect_as_cash = config.get('collect_dividends_as_cash', False)
                if collect_as_cash and div > 0:
                    # Calculate dividend cash and add to unreinvested cash
                    nb_shares = val_prev / price_prev if price_prev > 0 else 0
                    dividend_cash = nb_shares * div
                    total_unreinvested_dividends += dividend_cash
                    # Don't include dividend in rate of return
                    rate_of_return = var
                    val_new = val_prev * (1 + rate_of_return)
                else:
                    # Reinvest dividends (original behavior)
                    # CRITICAL FIX: For leveraged tickers, dividends should be handled differently
                    # When simulating leveraged ETFs, the dividend RATE should be the same as the base asset
                    if "?L=" in t:
                        # For leveraged tickers, get the base ticker's dividend rate (not amount)
                        base_ticker, leverage = parse_leverage_ticker(t)
                        if base_ticker in reindexed_data:
                            base_df = reindexed_data[base_ticker]
                            base_price_prev = base_df.loc[date_prev, "Close"]
                            # Use the base ticker's dividend rate, not the leveraged amount
                            dividend_rate = div / base_price_prev if base_price_prev > 0 else 0
                            rate_of_return = var + dividend_rate
                        else:
                            # Fallback: use leveraged price (may cause issues)
                            rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                    else:
                        # For regular tickers, use normal dividend reinvestment
                        rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                    val_new = val_prev * (1 + rate_of_return)
            else:
                val_new = val_prev * (1 + var)
            values[t].append(val_new)
        unallocated_cash.append(unallocated_cash[-1])
        if date in dates_added:
            unallocated_cash[-1] += added_amount
        unreinvested_cash.append(unreinvested_cash[-1] + total_unreinvested_dividends)
        portfolio_no_additions.append(portfolio_no_additions[-1] * daily_growth_factor)
        
        current_total = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
        
        # Check if we should rebalance
        should_rebalance = False
        # Normalize dates for comparison (remove timezone and time components)
        date_normalized = pd.Timestamp(date).normalize()
        dates_rebal_normalized = {pd.Timestamp(d).normalize() for d in dates_rebal}
        
        if date_normalized in dates_rebal_normalized and set(tickers):
            # If targeted rebalancing is enabled, check thresholds first - COPIED FROM PAGE 1
            if config.get('use_targeted_rebalancing', False):
                # Calculate current allocations as percentages
                current_total = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
                if current_total > 0:
                    current_allocations = {t: values[t][-1] / current_total for t in tickers}
                    
                    # Check if any ticker exceeds its targeted rebalancing thresholds
                    targeted_settings = config.get('targeted_rebalancing_settings', {})
                    threshold_exceeded = False
                    
                    for ticker in tickers:
                        if ticker in targeted_settings and targeted_settings[ticker].get('enabled', False):
                            current_allocation_pct = current_allocations.get(ticker, 0) * 100
                            max_threshold = targeted_settings[ticker].get('max_allocation', 100.0)
                            min_threshold = targeted_settings[ticker].get('min_allocation', 0.0)
                            
                            # Check if allocation exceeds max or falls below min threshold
                            if current_allocation_pct > max_threshold or current_allocation_pct < min_threshold:
                                threshold_exceeded = True
                                break
                    
                    # Only rebalance if thresholds are exceeded
                    should_rebalance = threshold_exceeded
                else:
                    # If no current value, don't rebalance
                    should_rebalance = False
            else:
                # Regular rebalancing - always rebalance on scheduled dates
                should_rebalance = True
        elif rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"] and set(tickers):
            # Buy & Hold: rebalance whenever there's cash available
            total_cash = unallocated_cash[-1] + unreinvested_cash[-1]
            if total_cash > 0:
                should_rebalance = True
        
        if should_rebalance:
            if use_momentum:
                returns, valid_assets = calculate_momentum(date, set(tickers), momentum_windows)
                if valid_assets:
                    weights, metrics_on_rebal = calculate_momentum_weights(
                        returns, valid_assets, date=date,
                        momentum_strategy=config.get('momentum_strategy', 'Classic'),
                        negative_momentum_strategy=config.get('negative_momentum_strategy', 'Cash')
                    )
                    historical_metrics[date] = metrics_on_rebal
                    if all(w == 0 for w in weights.values()):
                        # All cash: move total to unallocated_cash, set asset values to zero
                        for t in tickers:
                            values[t][-1] = 0
                        unallocated_cash[-1] = current_total
                        unreinvested_cash[-1] = 0
                    else:
                        # For Buy & Hold strategies with momentum, only distribute new cash
                        if rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"]:
                            # Calculate current proportions for Buy & Hold, or use momentum weights for Buy & Hold (Target)
                            if rebalancing_frequency == "Buy & Hold":
                                # Use current proportions from existing holdings
                                current_total_value = sum(values[t][-1] for t in tickers)
                                if current_total_value > 0:
                                    current_proportions = {t: values[t][-1] / current_total_value for t in tickers}
                                else:
                                    # If no current holdings, use equal weights
                                    current_proportions = {t: 1.0 / len(tickers) for t in tickers}
                            else:  # "Buy & Hold (Target)"
                                # Use momentum weights
                                current_proportions = weights
                            
                            # Only distribute the new cash (unallocated_cash + unreinvested_cash)
                            cash_to_distribute = unallocated_cash[-1] + unreinvested_cash[-1]
                            for t in tickers:
                                # Add new cash proportionally to existing holdings
                                values[t][-1] += cash_to_distribute * current_proportions.get(t, 0)
                            unreinvested_cash[-1] = 0
                            unallocated_cash[-1] = 0
                        else:
                            # Normal momentum rebalancing: replace all holdings
                            for t in tickers:
                                values[t][-1] = current_total * weights.get(t, 0)
                            unreinvested_cash[-1] = 0
                            unallocated_cash[-1] = 0
            else:
                # Apply allocation filters in correct order: Max Allocation -> Min Threshold -> Max Allocation (two-pass system)
                use_max_allocation = config.get('use_max_allocation', False)
                max_allocation_percent = config.get('max_allocation_percent', 10.0)
                use_threshold = config.get('use_minimal_threshold', False)
                threshold_percent = config.get('minimal_threshold_percent', 2.0)
                
                # Start with original allocations
                rebalance_allocations = {t: allocations.get(t, 0) for t in tickers}
                
                if use_max_allocation and rebalance_allocations:
                    max_allocation_decimal = max_allocation_percent / 100.0
                    
                    # FIRST PASS: Apply maximum allocation filter
                    capped_allocations = {}
                    excess_allocation = 0.0
                    
                    for ticker, allocation in rebalance_allocations.items():
                        if allocation > max_allocation_decimal:
                            # Cap the allocation and collect excess
                            capped_allocations[ticker] = max_allocation_decimal
                            excess_allocation += (allocation - max_allocation_decimal)
                        else:
                            # Keep original allocation
                            capped_allocations[ticker] = allocation
                    
                    # Redistribute excess allocation proportionally among stocks that are below the cap
                    if excess_allocation > 0:
                        # Find stocks that can receive more allocation (below the cap)
                        eligible_stocks = {ticker: allocation for ticker, allocation in capped_allocations.items() 
                                         if allocation < max_allocation_decimal}
                        
                        if eligible_stocks:
                            # Calculate total allocation of eligible stocks
                            total_eligible_allocation = sum(eligible_stocks.values())
                            
                            if total_eligible_allocation > 0:
                                # Redistribute excess proportionally
                                for ticker in eligible_stocks:
                                    proportion = eligible_stocks[ticker] / total_eligible_allocation
                                    additional_allocation = excess_allocation * proportion
                                    new_allocation = capped_allocations[ticker] + additional_allocation
                                    
                                    # Make sure we don't exceed the cap
                                    capped_allocations[ticker] = min(new_allocation, max_allocation_decimal)
                    
                    rebalance_allocations = capped_allocations
                
                # Apply minimal threshold filter for non-momentum strategies during rebalancing
                if use_threshold and rebalance_allocations:
                    threshold_decimal = threshold_percent / 100.0
                    
                    # First: Filter out stocks below threshold
                    filtered_allocations = {}
                    for t in tickers:
                        allocation = rebalance_allocations.get(t, 0)
                        if allocation >= threshold_decimal:
                            # Keep stocks above or equal to threshold
                            filtered_allocations[t] = allocation
                    
                    # Then: Normalize remaining stocks to sum to 1
                    if filtered_allocations:
                        total_allocation = sum(filtered_allocations.values())
                        if total_allocation > 0:
                            rebalance_allocations = {t: allocation / total_allocation for t, allocation in filtered_allocations.items()}
                        else:
                            rebalance_allocations = {}
                    else:
                        # If no stocks meet threshold, use original allocations
                        rebalance_allocations = {t: allocations.get(t, 0) for t in tickers}
                
                # SECOND PASS: Apply maximum allocation filter again (in case normalization created new excess)
                if use_max_allocation and rebalance_allocations:
                    max_allocation_decimal = max_allocation_percent / 100.0
                    
                    # Check if any stocks exceed the cap after threshold filtering and normalization
                    capped_allocations = {}
                    excess_allocation = 0.0
                    
                    for ticker, allocation in rebalance_allocations.items():
                        if allocation > max_allocation_decimal:
                            # Cap the allocation and collect excess
                            capped_allocations[ticker] = max_allocation_decimal
                            excess_allocation += (allocation - max_allocation_decimal)
                        else:
                            # Keep original allocation
                            capped_allocations[ticker] = allocation
                    
                    # Redistribute excess allocation proportionally among stocks that are below the cap
                    if excess_allocation > 0:
                        # Find stocks that can receive more allocation (below the cap)
                        eligible_stocks = {ticker: allocation for ticker, allocation in capped_allocations.items() 
                                         if allocation < max_allocation_decimal}
                        
                        if eligible_stocks:
                            # Calculate total allocation of eligible stocks
                            total_eligible_allocation = sum(eligible_stocks.values())
                            
                            if total_eligible_allocation > 0:
                                # Redistribute excess proportionally
                                for ticker in eligible_stocks:
                                    proportion = eligible_stocks[ticker] / total_eligible_allocation
                                    additional_allocation = excess_allocation * proportion
                                    new_allocation = capped_allocations[ticker] + additional_allocation
                                    
                                    # Make sure we don't exceed the cap
                                    capped_allocations[ticker] = min(new_allocation, max_allocation_decimal)
                    
                    rebalance_allocations = capped_allocations
                
                # Apply targeted rebalancing if enabled and thresholds are violated - COPIED FROM PAGE 1
                if config.get('use_targeted_rebalancing', False) and should_rebalance:
                    targeted_settings = config.get('targeted_rebalancing_settings', {})
                    current_asset_values = {t: values[t][-1] for t in tickers}
                    current_total_value = sum(current_asset_values.values())
                    
                    if current_total_value > 0:
                        current_allocations = {t: v / current_total_value for t, v in current_asset_values.items()}
                        
                        # For targeted rebalancing, rebalance TO THE THRESHOLD LIMITS, not to base allocations
                        target_allocations = {}
                        
                        # Calculate target allocations based on threshold limits
                        for t in tickers:
                            if t in targeted_settings and targeted_settings[t].get('enabled', False):
                                current_allocation_pct = (values[t][-1] / current_total_value) * 100 if current_total_value > 0 else 0
                                max_threshold = targeted_settings[t].get('max_allocation', 100.0)
                                min_threshold = targeted_settings[t].get('min_allocation', 0.0)
                                
                                # Rebalance to the threshold limit that was exceeded
                                if current_allocation_pct > max_threshold:
                                    target_allocations[t] = max_threshold / 100.0
                                elif current_allocation_pct < min_threshold:
                                    target_allocations[t] = min_threshold / 100.0
                                else:
                                    # Within bounds - keep current allocation
                                    target_allocations[t] = current_allocation_pct / 100.0
                            else:
                                # Not in targeted settings - use current allocation
                                target_allocations[t] = (values[t][-1] / current_total_value) if current_total_value > 0 else allocations.get(t, 0)
                        
                        # For targeted rebalancing, calculate the remaining allocation for non-targeted tickers
                        total_targeted = 0
                        targeted_count = 0
                        
                        for t in tickers:
                            if t in targeted_settings and targeted_settings[t].get('enabled', False):
                                total_targeted += target_allocations[t]
                                targeted_count += 1
                        
                        # Calculate remaining allocation for non-targeted tickers
                        remaining_allocation = 1.0 - total_targeted
                        non_targeted_tickers = [t for t in tickers if t not in targeted_settings or not targeted_settings[t].get('enabled', False)]
                        
                        if non_targeted_tickers and remaining_allocation > 0:
                            # Distribute remaining allocation PROPORTIONALLY to base allocations (not equally)
                            non_targeted_base_sum = sum(allocations.get(t, 0) for t in non_targeted_tickers)
                            if non_targeted_base_sum > 0:
                                # Distribute proportionally to base allocations
                                for t in non_targeted_tickers:
                                    base_proportion = allocations.get(t, 0) / non_targeted_base_sum
                                    target_allocations[t] = base_proportion * remaining_allocation
                            else:
                                # If no base allocations, distribute equally
                                allocation_per_ticker = remaining_allocation / len(non_targeted_tickers)
                                for t in non_targeted_tickers:
                                    target_allocations[t] = allocation_per_ticker
                        
                        # Use target allocations as rebalance allocations
                        rebalance_allocations = target_allocations
                
                sum_alloc = sum(rebalance_allocations.values())
                if sum_alloc > 0:
                    # For Buy & Hold strategies, only distribute new cash without touching existing holdings
                    if rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"]:
                        # Calculate current proportions for Buy & Hold, or use target allocations for Buy & Hold (Target)
                        if rebalancing_frequency == "Buy & Hold":
                            # Use current proportions from existing holdings
                            current_total_value = sum(values[t][-1] for t in tickers)
                            if current_total_value > 0:
                                current_proportions = {t: values[t][-1] / current_total_value for t in tickers}
                            else:
                                # If no current holdings, use equal weights
                                current_proportions = {t: 1.0 / len(tickers) for t in tickers}
                        else:  # "Buy & Hold (Target)"
                            # Use target allocations
                            current_proportions = {t: rebalance_allocations.get(t, 0) / sum_alloc for t in tickers}
                        
                        # Only distribute the new cash (unallocated_cash + unreinvested_cash)
                        cash_to_distribute = unallocated_cash[-1] + unreinvested_cash[-1]
                        for t in tickers:
                            # Add new cash proportionally to existing holdings
                            values[t][-1] += cash_to_distribute * current_proportions.get(t, 0)
                        unreinvested_cash[-1] = 0
                        unallocated_cash[-1] = 0
                    else:
                        # Normal rebalancing: replace all holdings
                        for t in tickers:
                            weight = rebalance_allocations.get(t, 0) / sum_alloc
                            values[t][-1] = current_total * weight
                        unreinvested_cash[-1] = 0
                        unallocated_cash[-1] = 0
            
            # Note: Daily allocations will be stored below after rebalancing
        
        # Store daily allocations for smooth allocation evolution charts (AFTER rebalancing)
        # For "start oldest" mode, only include tickers that have data available at this date
        available_tickers_at_date = []
        for t in tickers:
            if t in reindexed_data:
                price_value = reindexed_data[t].loc[date]
                # Handle case where loc returns a Series instead of scalar
                if isinstance(price_value, pd.Series):
                    price_value = price_value.iloc[0] if len(price_value) > 0 else np.nan
                if not pd.isna(price_value):
                    available_tickers_at_date.append(t)
        
        current_total_after_rebal = sum(values[t][-1] for t in available_tickers_at_date) + unallocated_cash[-1] + unreinvested_cash[-1]
        if current_total_after_rebal > 0:
            daily_allocs = {t: values[t][-1] / current_total_after_rebal for t in available_tickers_at_date}
            daily_allocs['CASH'] = (unallocated_cash[-1] + unreinvested_cash[-1]) / current_total_after_rebal
            historical_allocations[date] = daily_allocs

    # Store last allocation
    last_date = sim_index[-1]
    last_total = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
    if last_total > 0:
        historical_allocations[last_date] = {t: values[t][-1] / last_total for t in tickers}
        historical_allocations[last_date]['CASH'] = unallocated_cash[-1] / last_total if last_total > 0 else 0
    else:
        historical_allocations[last_date] = {t: 0 for t in tickers}
        historical_allocations[last_date]['CASH'] = 0
    
    # Store last metrics: always add a last-rebalance snapshot so the UI has a metrics row
    # If momentum is used, compute metrics; otherwise build metrics from the last allocation snapshot
    if use_momentum:
        returns, valid_assets = calculate_momentum(last_date, set(tickers), momentum_windows)
        weights, metrics_on_rebal = calculate_momentum_weights(
            returns, valid_assets, date=last_date,
            momentum_strategy=config.get('momentum_strategy', 'Classic'),
            negative_momentum_strategy=config.get('negative_momentum_strategy', 'Cash')
        )
        # Add CASH line to metrics
        cash_weight = 1.0 if all(w == 0 for w in weights.values()) else 0.0
        metrics_on_rebal['CASH'] = {'Calculated_Weight': cash_weight}
        historical_metrics[last_date] = metrics_on_rebal
    else:
        # Build a metrics snapshot from the last allocation so there's always a 'last rebalance' metrics entry
        if last_date in historical_allocations:
            alloc_snapshot = historical_allocations.get(last_date, {})
            metrics_on_rebal = {}
            for ticker_sym, alloc_val in alloc_snapshot.items():
                metrics_on_rebal[ticker_sym] = {'Calculated_Weight': alloc_val}
            # Ensure CASH entry exists
            if 'CASH' not in metrics_on_rebal:
                metrics_on_rebal['CASH'] = {'Calculated_Weight': alloc_snapshot.get('CASH', 0)}
            # Only set if not already present
            if last_date not in historical_metrics:
                historical_metrics[last_date] = metrics_on_rebal

    results = pd.DataFrame(index=sim_index)
    for t in tickers:
        results[f"Value_{t}"] = values[t]
    results["Unallocated_cash"] = unallocated_cash
    results["Unreinvested_cash"] = unreinvested_cash
    results["Total_assets"] = results[[f"Value_{t}" for t in tickers]].sum(axis=1)
    results["Total_with_dividends_plus_cash"] = results["Total_assets"] + results["Unallocated_cash"] + results["Unreinvested_cash"]
    results['Portfolio_Value_No_Additions'] = portfolio_no_additions

    return results["Total_with_dividends_plus_cash"], results['Portfolio_Value_No_Additions'], historical_allocations, historical_metrics


# -----------------------
# PAGE-SCOPED SESSION STATE INITIALIZATION - STRATEGY COMPARISON PAGE
# -----------------------
# Ensure complete independence from other pages by using page-specific session keys
if 'strategy_comparison_page_initialized' not in st.session_state:
    st.session_state.strategy_comparison_page_initialized = True
    # Clear any shared session state that might interfere with other pages
    keys_to_clear = [
        # Main app keys
        'main_portfolio_configs', 'main_active_portfolio_index', 'main_rerun_flag',
        'main_all_results', 'main_all_allocations', 'main_all_metrics',
        'main_drawdowns', 'main_stats_df', 'main_years_data', 'main_portfolio_map',
        'main_backtest_ran', 'main_raw_data', 'main_running', 'main_run_requested',
        'main_pending_backtest_params', 'main_tickers', 'main_allocs', 'main_divs',
        'main_use_momentum', 'main_mom_windows', 'main_use_beta', 'main_use_vol',
        'main_initial_value_input_decimals', 'main_initial_value_input_int',
        'main_added_amount_input_decimals', 'main_added_amount_input_int',
        'main_start_date', 'main_end_date', 'main_use_custom_dates',
        'main_momentum_strategy', 'main_negative_momentum_strategy',
        'main_beta_window_days', 'main_beta_exclude_days', 'main_vol_window_days', 'main_vol_exclude_days',
        # Allocations page keys
        'alloc_portfolio_configs', 'alloc_active_portfolio_index', 'alloc_rerun_flag',
        'alloc_all_results', 'alloc_all_allocations', 'alloc_all_metrics',
        'alloc_paste_json_text', 'allocations_page_initialized',
        # Any other potential shared keys
        'strategy_comparison_all_results', 'strategy_comparison_all_allocations', 'strategy_comparison_all_metrics',
        'all_drawdowns', 'stats_df_display', 'all_years', 'portfolio_key_map',
        'strategy_comparison_ran', 'raw_data'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# Main App Logic
# -----------------------

if 'strategy_comparison_portfolio_configs' not in st.session_state:
    st.session_state.strategy_comparison_portfolio_configs = default_configs
if 'strategy_comparison_active_portfolio_index' not in st.session_state:
    st.session_state.strategy_comparison_active_portfolio_index = 0

# Ensure all portfolios have threshold and maximum allocation settings
for portfolio in st.session_state.strategy_comparison_portfolio_configs:
    if 'use_minimal_threshold' not in portfolio:
        portfolio['use_minimal_threshold'] = False
    if 'minimal_threshold_percent' not in portfolio:
        portfolio['minimal_threshold_percent'] = 2.0
    if 'use_max_allocation' not in portfolio:
        portfolio['use_max_allocation'] = False
    if 'max_allocation_percent' not in portfolio:
        portfolio['max_allocation_percent'] = 10.0

if 'strategy_comparison_paste_json_text' not in st.session_state:
    st.session_state.strategy_comparison_paste_json_text = ""
if 'strategy_comparison_rerun_flag' not in st.session_state:
    st.session_state.strategy_comparison_rerun_flag = False
if 'strategy_comparison_global_tickers' not in st.session_state:
    st.session_state.strategy_comparison_global_tickers = [
        {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
    ]

# Sync global tickers to all portfolios on page load
def sync_global_tickers_to_all_portfolios():
    """Sync global tickers to all portfolios"""
    for portfolio in st.session_state.strategy_comparison_portfolio_configs:
        portfolio['stocks'] = st.session_state.strategy_comparison_global_tickers.copy()

# Initial sync
sync_global_tickers_to_all_portfolios()

# -----------------------
# Timer function for next rebalance date
# -----------------------
def calculate_next_rebalance_date(rebalancing_frequency, last_rebalance_date):
    """
    Calculate the next rebalance date based on rebalancing frequency and last rebalance date.
    Excludes today and yesterday as mentioned in the requirements.
    """
    if not last_rebalance_date or rebalancing_frequency == 'none':
        return None, None, None
    
    # Convert to datetime if it's a pandas Timestamp
    if hasattr(last_rebalance_date, 'to_pydatetime'):
        last_rebalance_date = last_rebalance_date.to_pydatetime()
    
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    # If last rebalance was today or yesterday, use the day before yesterday as base
    if last_rebalance_date.date() >= yesterday:
        base_date = yesterday - timedelta(days=1)
    else:
        base_date = last_rebalance_date.date()
    
    # Calculate next rebalance date based on frequency
    if rebalancing_frequency == 'market_day':
        # Next market day (simplified - just next day for now)
        next_date = base_date + timedelta(days=1)
    elif rebalancing_frequency == 'calendar_day':
        next_date = base_date + timedelta(days=1)
    elif rebalancing_frequency == 'week':
        next_date = base_date + timedelta(weeks=1)
    elif rebalancing_frequency == '2weeks':
        next_date = base_date + timedelta(weeks=2)
    elif rebalancing_frequency == 'month':
        # Add one month
        if base_date.month == 12:
            next_date = base_date.replace(year=base_date.year + 1, month=1)
        else:
            next_date = base_date.replace(month=base_date.month + 1)
    elif rebalancing_frequency == '3months':
        # Add three months
        new_month = base_date.month + 3
        new_year = base_date.year + (new_month - 1) // 12
        new_month = ((new_month - 1) % 12) + 1
        next_date = base_date.replace(year=new_year, month=new_month)
    elif rebalancing_frequency == '6months':
        # Add six months
        new_month = base_date.month + 6
        new_year = base_date.year + (new_month - 1) // 12
        new_month = ((new_month - 1) % 12) + 1
        next_date = base_date.replace(year=new_year, month=new_month)
    elif rebalancing_frequency == 'year':
        next_date = base_date.replace(year=base_date.year + 1)
    else:
        return None, None, None
    
    # Calculate time until next rebalance
    now = datetime.now()
    # Ensure both datetimes are offset-naive for comparison and subtraction
    if hasattr(next_date, 'tzinfo') and next_date.tzinfo is not None:
        next_date = next_date.replace(tzinfo=None)
    next_rebalance_datetime = datetime.combine(next_date, time(9, 30))  # Assume 9:30 AM market open
    if hasattr(next_rebalance_datetime, 'tzinfo') and next_rebalance_datetime.tzinfo is not None:
        next_rebalance_datetime = next_rebalance_datetime.replace(tzinfo=None)
    if hasattr(now, 'tzinfo') and now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    # If next rebalance is in the past, calculate the next one iteratively instead of recursively
    max_iterations = 10  # Prevent infinite loops
    iteration = 0
    while next_rebalance_datetime <= now and iteration < max_iterations:
        iteration += 1
        if rebalancing_frequency in ['market_day', 'calendar_day']:
            next_date = next_date + timedelta(days=1)
        elif rebalancing_frequency == 'week':
            next_date = next_date + timedelta(weeks=1)
        elif rebalancing_frequency == '2weeks':
            next_date = next_date + timedelta(weeks=2)
        elif rebalancing_frequency == 'month':
            # Add one month safely
            if next_date.month == 12:
                next_date = next_date.replace(year=next_date.year + 1, month=1)
            else:
                next_date = next_date.replace(month=next_date.month + 1)
            # Handle day overflow
            try:
                next_date = next_date.replace(day=min(next_date.day, 28))
            except ValueError:
                next_date = next_date.replace(day=1)
        elif rebalancing_frequency == '3months':
            # Add three months safely
            new_month = next_date.month + 3
            new_year = next_date.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            next_date = next_date.replace(year=new_year, month=new_month, day=1)
        elif rebalancing_frequency == '6months':
            # Add six months safely
            new_month = next_date.month + 6
            new_year = next_date.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            next_date = next_date.replace(year=new_year, month=new_month, day=1)
        elif rebalancing_frequency == 'year':
            next_date = next_date.replace(year=next_date.year + 1)
        
        next_rebalance_datetime = datetime.combine(next_date, time(9, 30))
    
    time_until = next_rebalance_datetime - now
    
    return next_date, time_until, next_rebalance_datetime

def format_time_until(time_until):
    """Format the time until next rebalance in a human-readable format."""
    if not time_until:
        return "Unknown"
    
    total_seconds = int(time_until.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    
    if days > 0:
        return f"{days} days, {hours} hours, {minutes} minutes"
    elif hours > 0:
        return f"{hours} hours, {minutes} minutes"
    else:
        return f"{minutes} minutes"

def ensure_unique_portfolio_name(proposed_name, existing_portfolios):
    """
    Ensures a portfolio name is unique by adding (1), (2), etc. if needed.
    This happens BEFORE the portfolio is added to prevent crashes.
    """
    existing_names = [p.get('name', '') for p in existing_portfolios]
    
    # If name is unique, return as-is
    if proposed_name not in existing_names:
        return proposed_name
    
    # Find the next available number
    counter = 1
    while f"{proposed_name} ({counter})" in existing_names:
        counter += 1
    
    return f"{proposed_name} ({counter})"

def add_portfolio_to_configs(portfolio):
    """
    CENTRAL function to add ANY portfolio to the configs.
    ALL portfolio additions MUST go through this function.
    Automatically ensures unique names no matter the source.
    """
    # Ensure unique name IMMEDIATELY
    unique_name = ensure_unique_portfolio_name(
        portfolio.get('name', 'Unnamed Portfolio'), 
        st.session_state.strategy_comparison_portfolio_configs
    )
    portfolio['name'] = unique_name
    
    # Add to configs
    st.session_state.strategy_comparison_portfolio_configs.append(portfolio)
    return portfolio

def ensure_all_portfolio_names_unique():
    """
    NUCLEAR OPTION: Ensures ALL existing portfolios have unique names.
    Call this at startup or after any bulk operations.
    """
    if 'strategy_comparison_portfolio_configs' not in st.session_state:
        return
    
    configs = st.session_state.strategy_comparison_portfolio_configs
    seen_names = set()
    
    for i, portfolio in enumerate(configs):
        original_name = portfolio.get('name', f'Unnamed Portfolio {i+1}')
        
        # If this name is already seen, make it unique
        if original_name in seen_names:
            counter = 1
            while f"{original_name} ({counter})" in seen_names:
                counter += 1
            portfolio['name'] = f"{original_name} ({counter})"
        
        seen_names.add(portfolio['name'])

# NUCLEAR OPTION: Ensure all portfolio names are unique at startup
# Called AFTER all functions are defined
ensure_all_portfolio_names_unique()

# CONTINUOUS MONITORING: Check for duplicates on every UI render
# This catches ANY duplicate that appears through ANY unknown method
def continuous_duplicate_check():
    """
    GOD-PROOF duplicate checking - runs on every UI render.
    Even if GOD spawns a duplicate portfolio, this catches it.
    """
    if 'strategy_comparison_portfolio_configs' not in st.session_state:
        return
    
    configs = st.session_state.strategy_comparison_portfolio_configs
    names = [p.get('name', '') for p in configs]
    
    # Check if there are any duplicates
    if len(names) != len(set(names)):
        # Duplicates found! Fix them immediately
        ensure_all_portfolio_names_unique()

# Call continuous check on every render
continuous_duplicate_check()

def add_portfolio_callback():
    # Create a completely blank portfolio with no default tickers and no momentum
    new_portfolio = {
        'name': f"New Portfolio {len(st.session_state.strategy_comparison_portfolio_configs) + 1}",
        'stocks': [],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 0,
        'added_frequency': 'none',
        'rebalancing_frequency': 'Monthly',
        'start_with': 'all',
        'first_rebalance_strategy': 'rebalancing_date',
        'use_momentum': False,
        'momentum_strategy': 'Classic',
        'negative_momentum_strategy': 'Cash',
        'momentum_windows': [
            {"lookback": 365, "exclude": 30, "weight": 1.0}
        ],
        'calc_beta': True,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'calc_volatility': True,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
        'use_minimal_threshold': False,
        'minimal_threshold_percent': 4.0,
        'use_max_allocation': False,
        'max_allocation_percent': 20.0,
        'collect_dividends_as_cash': False,
        'start_date_user': None,
        'end_date_user': None,
        'fusion_portfolio': {'enabled': False, 'selected_portfolios': [], 'allocations': {}}
    }
    
    # Use central function - automatically ensures unique name
    add_portfolio_to_configs(new_portfolio)
    st.session_state.strategy_comparison_active_portfolio_index = len(st.session_state.strategy_comparison_portfolio_configs) - 1
    st.session_state.strategy_comparison_rerun_flag = True

def remove_portfolio_callback():
    if len(st.session_state.strategy_comparison_portfolio_configs) > 1:
        st.session_state.strategy_comparison_portfolio_configs.pop(st.session_state.strategy_comparison_active_portfolio_index)
        st.session_state.strategy_comparison_active_portfolio_index = max(0, st.session_state.strategy_comparison_active_portfolio_index - 1)
        st.session_state.strategy_comparison_rerun_flag = True

def bulk_delete_portfolios_callback(portfolio_names_to_delete):
    """Delete multiple portfolios at once"""
    if len(st.session_state.strategy_comparison_portfolio_configs) <= 1:
        return  # Don't delete the last portfolio
    
    # Get indices of portfolios to delete
    indices_to_delete = []
    for name in portfolio_names_to_delete:
        for i, cfg in enumerate(st.session_state.strategy_comparison_portfolio_configs):
            if cfg['name'] == name:
                indices_to_delete.append(i)
                break
    
    # Sort indices in descending order to avoid index shifting issues
    indices_to_delete.sort(reverse=True)
    
    # Delete portfolios
    deleted_count = 0
    for idx in indices_to_delete:
        if len(st.session_state.strategy_comparison_portfolio_configs) > 1:
            st.session_state.strategy_comparison_portfolio_configs.pop(idx)
            deleted_count += 1
    
    # Clear all checkboxes after deletion
    st.session_state.strategy_comparison_portfolio_checkboxes = {}
    
    # Update active portfolio index if necessary
    if st.session_state.strategy_comparison_active_portfolio_index >= len(st.session_state.strategy_comparison_portfolio_configs):
        st.session_state.strategy_comparison_active_portfolio_index = max(0, len(st.session_state.strategy_comparison_portfolio_configs) - 1)
    
    # Set success message
    st.session_state.strategy_comparison_bulk_delete_success = f"Successfully deleted {deleted_count} portfolio(s)!"
    
    st.session_state.strategy_comparison_rerun_flag = True

def add_stock_callback():
    # Add stock without triggering full refresh - just set flag for processing
    st.session_state.strategy_comparison_add_stock_flag = True

def remove_stock_callback(ticker):
    """Immediate stock removal callback for individual portfolio - NO REFRESH"""
    try:
        active_idx = st.session_state.strategy_comparison_active_portfolio_index
        if (0 <= active_idx < len(st.session_state.strategy_comparison_portfolio_configs) and
            'stocks' in st.session_state.strategy_comparison_portfolio_configs[active_idx]):
            stocks = st.session_state.strategy_comparison_portfolio_configs[active_idx]['stocks']
            
            # Find and remove the stock with matching ticker
            for i, stock in enumerate(stocks):
                if stock['ticker'] == ticker:
                    stocks.pop(i)
                    # If this was the last stock, add an empty one
                    if len(stocks) == 0:
                        stocks.append({'ticker': '', 'allocation': 0.0, 'include_dividends': True})
                    # Update global tickers but don't trigger refresh
                    st.session_state.strategy_comparison_global_tickers = stocks.copy()
                    break
    except (ValueError, IndexError):
        pass

def normalize_stock_allocations_callback():
    if 'strategy_comparison_global_tickers' not in st.session_state:
        return
    stocks = st.session_state.strategy_comparison_global_tickers
    valid_stocks = [s for s in stocks if s['ticker']]
    total_alloc = sum(s['allocation'] for s in valid_stocks)
    if total_alloc > 0:
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] /= total_alloc
                alloc_key = f"strategy_comparison_global_alloc_{idx}"
                st.session_state[alloc_key] = int(s['allocation'] * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"strategy_comparison_global_alloc_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.strategy_comparison_global_tickers = stocks
    # Sync to all portfolios
    sync_global_tickers_to_all_portfolios()

def equal_stock_allocation_callback():
    if 'strategy_comparison_global_tickers' not in st.session_state:
        return
    stocks = st.session_state.strategy_comparison_global_tickers
    valid_stocks = [s for s in stocks if s['ticker']]
    if valid_stocks:
        equal_weight = 1.0 / len(valid_stocks)
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] = equal_weight
                alloc_key = f"strategy_comparison_global_alloc_{idx}"
                st.session_state[alloc_key] = int(equal_weight * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"strategy_comparison_global_alloc_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.strategy_comparison_global_tickers = stocks
    # Sync to all portfolios
    sync_global_tickers_to_all_portfolios()
    
def reset_portfolio_callback():
    current_name = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['name']
    default_cfg_found = next((cfg for cfg in default_configs if cfg['name'] == current_name), None)
    if default_cfg_found is None:
        default_cfg_found = default_configs[1].copy()
        default_cfg_found['name'] = current_name
    # Clear any saved momentum settings when resetting
    if 'saved_momentum_settings' in default_cfg_found:
        del default_cfg_found['saved_momentum_settings']
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] = default_cfg_found
    st.session_state.strategy_comparison_rerun_flag = True

def reset_stock_selection_callback():
    # Reset global tickers to default
    st.session_state.strategy_comparison_global_tickers = [
        {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
    ]
    # Sync to all portfolios
    sync_global_tickers_to_all_portfolios()

def reset_momentum_windows_callback():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['momentum_windows'] = [
        {"lookback": 365, "exclude": 30, "weight": 0.5},
        {"lookback": 180, "exclude": 30, "weight": 0.3},
        {"lookback": 120, "exclude": 30, "weight": 0.2},
    ]
    # Don't trigger immediate re-run for better performance
    # st.session_state.strategy_comparison_rerun_flag = True

def update_stock_allocation(index):
    try:
        active_index = st.session_state.strategy_comparison_active_portfolio_index
        portfolio_configs = st.session_state.strategy_comparison_portfolio_configs
        if (active_index < len(portfolio_configs) and 
            'stocks' in portfolio_configs[active_index] and 
            index < len(portfolio_configs[active_index]['stocks'])):
            key = f"strategy_comparison_alloc_input_{active_index}_{index}"
            val = st.session_state.get(key, None)
            if val is not None:
                portfolio_configs[active_index]['stocks'][index]['allocation'] = float(val) / 100.0
    except Exception:
        # Ignore transient errors (e.g., active_portfolio_index changed); UI will reflect state on next render
        return

def update_stock_ticker(index):
    try:
        active_index = st.session_state.strategy_comparison_active_portfolio_index
        portfolio_configs = st.session_state.strategy_comparison_portfolio_configs
        if (active_index < len(portfolio_configs) and 
            'stocks' in portfolio_configs[active_index] and 
            index < len(portfolio_configs[active_index]['stocks'])):
            key = f"strategy_comparison_ticker_{active_index}_{index}"
            val = st.session_state.get(key, None)
            if val is not None:
                # Convert the input value to uppercase
                upper_val = val.upper()
                
        # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
        if upper_val == 'BRK.B':
            upper_val = 'BRK-B'
        elif upper_val == 'BRK.A':
            upper_val = 'BRK-A'
        
        # CRITICAL: Resolve ticker alias BEFORE storing in portfolio config
        resolved_ticker = resolve_ticker_alias(upper_val)
        
        portfolio_configs[active_index]['stocks'][index]['ticker'] = resolved_ticker
        # Update the text box's state to show the resolved ticker (with leverage/expense visible)
        st.session_state[key] = resolved_ticker
        
        # Auto-disable dividends for negative leverage (inverse ETFs)
        if '?L=-' in resolved_ticker:
            portfolio_configs[active_index]['stocks'][index]['include_dividends'] = False
            # Also update the checkbox UI state
            div_key = f"strategy_comparison_div_{active_index}_{index}"
            st.session_state[div_key] = False
    except Exception:
        # Defensive: if portfolio index or structure changed, skip silently
        return

def update_stock_dividends(index):
    try:
        active_index = st.session_state.strategy_comparison_active_portfolio_index
        portfolio_configs = st.session_state.strategy_comparison_portfolio_configs
        if (active_index < len(portfolio_configs) and 
            'stocks' in portfolio_configs[active_index] and 
            index < len(portfolio_configs[active_index]['stocks'])):
            key = f"strategy_comparison_div_{active_index}_{index}"
            val = st.session_state.get(key, None)
            if val is not None:
                portfolio_configs[active_index]['stocks'][index]['include_dividends'] = bool(val)
    except Exception:
        return

# Global ticker management functions
def update_global_stock_ticker(index):
    try:
        global_tickers = st.session_state.strategy_comparison_global_tickers
        if index < len(global_tickers):
            key = f"strategy_comparison_global_ticker_{index}"
            val = st.session_state.get(key, None)
            if val is not None:
                # Convert commas to dots for decimal separators (like case conversion)
                converted_val = val.replace(",", ".")
                
                # Convert the input value to uppercase
                upper_val = converted_val.upper()
                
                # Resolve alias if it exists
                resolved_ticker = resolve_ticker_alias(upper_val)
                
                global_tickers[index]['ticker'] = resolved_ticker
                # Update the text box's state to show the resolved ticker
                st.session_state[key] = resolved_ticker
                # Sync to all portfolios
                sync_global_tickers_to_all_portfolios()
    except Exception:
        return

def update_global_stock_allocation(index):
    try:
        global_tickers = st.session_state.strategy_comparison_global_tickers
        if index < len(global_tickers):
            key = f"strategy_comparison_global_alloc_{index}"
            val = st.session_state.get(key, None)
            if val is not None:
                global_tickers[index]['allocation'] = float(val) / 100.0
                # Sync to all portfolios
                sync_global_tickers_to_all_portfolios()
    except Exception:
        return

def update_global_stock_dividends(index):
    try:
        global_tickers = st.session_state.strategy_comparison_global_tickers
        if index < len(global_tickers):
            key = f"strategy_comparison_global_div_{index}"
            val = st.session_state.get(key, None)
            if val is not None:
                global_tickers[index]['include_dividends'] = bool(val)
                # Sync to all portfolios
                sync_global_tickers_to_all_portfolios()
    except Exception:
        return

def remove_global_stock_callback(ticker):
    """Immediate stock removal callback - OPTIMIZED NO REFRESH"""
    try:
        global_tickers = st.session_state.strategy_comparison_global_tickers
        
        # Find and remove the stock with matching ticker
        for i, stock in enumerate(global_tickers):
            if stock['ticker'] == ticker:
                global_tickers.pop(i)
                # If this was the last stock, add an empty one
                if len(global_tickers) == 0:
                    global_tickers.append({'ticker': '', 'allocation': 0.0, 'include_dividends': True})
                
                # Clear session state keys for all remaining global tickers to force re-initialization
                for j in range(len(global_tickers)):
                    # Clear ticker keys
                    ticker_key = f"strategy_comparison_global_ticker_{j}"
                    if ticker_key in st.session_state:
                        del st.session_state[ticker_key]
                    # Clear allocation keys
                    alloc_key = f"strategy_comparison_global_alloc_{j}"
                    if alloc_key in st.session_state:
                        del st.session_state[alloc_key]
                    # Clear dividend keys
                    div_key = f"strategy_comparison_global_div_{j}"
                    if div_key in st.session_state:
                        del st.session_state[div_key]
                
                # Sync to all portfolios but don't trigger refresh
                sync_global_tickers_to_all_portfolios()
                break
    except (IndexError, KeyError):
        pass

def reset_beta_callback():
    # Reset beta lookback/exclude to defaults and enable beta calculation
    idx = st.session_state.strategy_comparison_active_portfolio_index
    st.session_state.strategy_comparison_portfolio_configs[idx]['beta_window_days'] = 365
    st.session_state.strategy_comparison_portfolio_configs[idx]['exclude_days_beta'] = 30
    # Ensure checkbox state reflects enabled
    st.session_state.strategy_comparison_portfolio_configs[idx]['calc_beta'] = True
    st.session_state['strategy_comparison_active_calc_beta'] = True
    # Update UI widget values to reflect reset
    st.session_state['strategy_comparison_active_beta_window'] = 365
    st.session_state['strategy_comparison_active_beta_exclude'] = 30
    # Trigger rerun to update UI
    st.session_state.strategy_comparison_rerun_flag = True

def reset_vol_callback():
    # Reset volatility lookback/exclude to defaults and enable volatility calculation
    idx = st.session_state.strategy_comparison_active_portfolio_index
    st.session_state.strategy_comparison_portfolio_configs[idx]['vol_window_days'] = 365
    st.session_state.strategy_comparison_portfolio_configs[idx]['exclude_days_vol'] = 30
    st.session_state.strategy_comparison_portfolio_configs[idx]['calc_volatility'] = True
    st.session_state['strategy_comparison_active_calc_vol'] = True
    # Update UI widget values to reflect reset
    st.session_state['strategy_comparison_active_vol_window'] = 365
    st.session_state['strategy_comparison_active_vol_exclude'] = 30
    # Trigger rerun to update UI
    st.session_state.strategy_comparison_rerun_flag = True

def sync_cashflow_from_first_portfolio_callback():
    """Sync initial value, added amount, and added frequency from first portfolio to all others"""
    try:
        if len(st.session_state.strategy_comparison_portfolio_configs) > 1:
            first_portfolio = st.session_state.strategy_comparison_portfolio_configs[0]
            
            # Get values from first portfolio
            initial_value = first_portfolio.get('initial_value', 10000)
            added_amount = first_portfolio.get('added_amount', 1000)
            added_frequency = first_portfolio.get('added_frequency', 'Monthly')
            
            # Update all other portfolios (skip those excluded from sync)
            updated_count = 0
            for i in range(1, len(st.session_state.strategy_comparison_portfolio_configs)):
                portfolio = st.session_state.strategy_comparison_portfolio_configs[i]
                if not portfolio.get('exclude_from_cashflow_sync', False):
                    # Only update if values are actually different
                    if (portfolio.get('initial_value') != initial_value or 
                        portfolio.get('added_amount') != added_amount or 
                        portfolio.get('added_frequency') != added_frequency):
                        portfolio['initial_value'] = initial_value
                        portfolio['added_amount'] = added_amount
                        portfolio['added_frequency'] = added_frequency
                        updated_count += 1
            
            # Only update UI and rerun if something actually changed
            if updated_count > 0:
                # Only update UI widgets if the current portfolio is NOT excluded from cash flow sync
                current_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
                if not current_portfolio.get('exclude_from_cashflow_sync', False):
                    # Update UI widget session states to reflect the changes
                    st.session_state['strategy_comparison_active_initial'] = initial_value
                    st.session_state['strategy_comparison_active_added_amount'] = added_amount
                    st.session_state['strategy_comparison_active_add_freq'] = added_frequency
                
                # Store success message in session state instead of showing it at top
                st.session_state['strategy_comparison_cashflow_sync_message'] = f"✅ Successfully synced cashflow settings to {updated_count} portfolio(s)"
                st.session_state['strategy_comparison_cashflow_sync_message_type'] = 'success'
                
                # Force immediate rerun to show changes
                st.session_state.strategy_comparison_rerun_flag = True
            else:
                # Store info message in session state
                st.session_state['strategy_comparison_cashflow_sync_message'] = "ℹ️ No portfolios were updated (all were excluded or already had matching values)"
                st.session_state['strategy_comparison_cashflow_sync_message_type'] = 'info'
    except Exception as e:
        # Store error message in session state
        st.session_state['strategy_comparison_cashflow_sync_message'] = f"❌ Error during cash flow sync: {str(e)}"
        st.session_state['strategy_comparison_cashflow_sync_message_type'] = 'error'

def sync_rebalancing_from_first_portfolio_callback():
    """Sync rebalancing frequency from first portfolio to all others"""
    try:
        if len(st.session_state.strategy_comparison_portfolio_configs) > 1:
            first_portfolio = st.session_state.strategy_comparison_portfolio_configs[0]
            
            # Get rebalancing frequency from first portfolio
            rebalancing_frequency = first_portfolio.get('rebalancing_frequency', 'Monthly')
            
            # Update all other portfolios (skip those excluded from sync)
            updated_count = 0
            for i in range(1, len(st.session_state.strategy_comparison_portfolio_configs)):
                portfolio = st.session_state.strategy_comparison_portfolio_configs[i]
                if not portfolio.get('exclude_from_rebalancing_sync', False):
                    # Only update if value is actually different
                    if portfolio.get('rebalancing_frequency') != rebalancing_frequency:
                        portfolio['rebalancing_frequency'] = rebalancing_frequency
                        updated_count += 1
            
            # Only update UI and rerun if something actually changed
            if updated_count > 0:
                # Only update UI widgets if the current portfolio is NOT excluded from rebalancing sync
                current_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
                if not current_portfolio.get('exclude_from_rebalancing_sync', False):
                    # Update UI widget session state to reflect the change
                    st.session_state['strategy_comparison_active_rebal_freq'] = rebalancing_frequency
                
                # Store success message in session state instead of showing it at top
                st.session_state['strategy_comparison_rebalancing_sync_message'] = f"✅ Successfully synced rebalancing frequency to {updated_count} portfolio(s)"
                st.session_state['strategy_comparison_rebalancing_sync_message_type'] = 'success'
                
                # Force immediate rerun to show changes
                st.session_state.strategy_comparison_rerun_flag = True
            else:
                # Store info message in session state
                st.session_state['strategy_comparison_rebalancing_sync_message'] = "ℹ️ No portfolios were updated (all were excluded or already had matching values)"
                st.session_state['strategy_comparison_rebalancing_sync_message_type'] = 'info'
    except Exception as e:
        # Store error message in session state
        st.session_state['strategy_comparison_rebalancing_sync_message'] = f"❌ Error during rebalancing sync: {str(e)}"
        st.session_state['strategy_comparison_rebalancing_sync_message_type'] = 'error'

def add_momentum_window_callback():
    st.session_state.strategy_comparison_add_momentum_window_flag = True

def remove_momentum_window_callback():
    st.session_state.strategy_comparison_remove_momentum_window_flag = True

def normalize_momentum_weights_callback():
    if 'strategy_comparison_portfolio_configs' not in st.session_state or 'strategy_comparison_portfolio_index' not in st.session_state:
        return
    active_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
    total_weight = sum(w['weight'] for w in active_portfolio['momentum_windows'])
    if total_weight > 0:
        for idx, w in enumerate(active_portfolio['momentum_windows']):
            w['weight'] /= total_weight
            weight_key = f"strategy_comparison_weight_input_active_{idx}"
            # Sanitize weight to prevent StreamlitValueAboveMaxError
            weight = w['weight']
            if isinstance(weight, (int, float)):
                # Convert decimal to percentage, ensuring it's within bounds
                weight_percentage = max(0.0, min(weight * 100.0, 100.0))
            else:
                # Invalid weight, set to default
                weight_percentage = 10.0
            st.session_state[weight_key] = int(weight_percentage)
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['momentum_windows'] = active_portfolio['momentum_windows']

def paste_json_callback():
    try:
        # Use the SAME parsing logic as successful PDF extraction
        raw_text = st.session_state.strategy_comparison_paste_json_text
        
        # STEP 1: Try the exact same approach as PDF extraction (simple strip + parse)
        try:
            cleaned_text = raw_text.strip()
            json_data = json.loads(cleaned_text)
            st.success("✅ JSON parsed successfully using PDF-style parsing!")
        except json.JSONDecodeError:
            # STEP 2: If that fails, apply our advanced cleaning (fallback)
            st.info("🔧 Simple parsing failed, applying advanced PDF extraction fixes...")
            
            json_text = raw_text
            import re
            
            # Fix common PDF extraction issues
            # Pattern to find broken portfolio name lines like: "name": "Some Name "stocks":
            broken_pattern = r'"name":\s*"([^"]*?)"\s*"stocks":'
            # Replace with proper JSON structure: "name": "Some Name", "stocks":
            json_text = re.sub(broken_pattern, r'"name": "\1", "stocks":', json_text)
            
            # Fix truncated names that end abruptly without closing quote
            # Pattern: "name": "Some text without closing quote "stocks":
            truncated_pattern = r'"name":\s*"([^"]*?)\s+"stocks":'
            json_text = re.sub(truncated_pattern, r'"name": "\1", "stocks":', json_text)
            
            # Fix missing opening brace for portfolio objects
            # Pattern: }, "name": should be }, { "name":
            missing_brace_pattern = r'(},)\s*("name":)'
            json_text = re.sub(missing_brace_pattern, r'\1 {\n \2', json_text)
            
            json_data = json.loads(json_text)
            st.success("✅ JSON parsed successfully using advanced cleaning!")
        
        # Add missing fields for compatibility if they don't exist
        if 'collect_dividends_as_cash' not in json_data:
            json_data['collect_dividends_as_cash'] = False
        if 'exclude_from_cashflow_sync' not in json_data:
            json_data['exclude_from_cashflow_sync'] = False
        if 'exclude_from_rebalancing_sync' not in json_data:
            json_data['exclude_from_rebalancing_sync'] = False
        if 'use_minimal_threshold' not in json_data:
            json_data['use_minimal_threshold'] = False
        if 'minimal_threshold_percent' not in json_data:
            json_data['minimal_threshold_percent'] = 2.0
        if 'use_max_allocation' not in json_data:
            json_data['use_max_allocation'] = False
        if 'max_allocation_percent' not in json_data:
            json_data['max_allocation_percent'] = 10.0
        
        # Clear widget keys to force re-initialization
        widget_keys_to_clear = [
            # Removed beta/volatility keys so they preserve values like momentum windows
        ]
        for key in widget_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        

        
        # Handle momentum strategy value mapping from other pages
        momentum_strategy = json_data.get('momentum_strategy', 'Classic')
        if momentum_strategy == 'Classic momentum':
            momentum_strategy = 'Classic'
        elif momentum_strategy == 'Relative momentum':
            momentum_strategy = 'Relative Momentum'
        elif momentum_strategy not in ['Classic', 'Relative Momentum']:
            momentum_strategy = 'Classic'  # Default fallback
        
        # Handle negative momentum strategy value mapping from other pages
        negative_momentum_strategy = json_data.get('negative_momentum_strategy', 'Cash')
        if negative_momentum_strategy == 'Go to cash':
            negative_momentum_strategy = 'Cash'
        elif negative_momentum_strategy not in ['Cash', 'Equal weight', 'Relative momentum']:
            negative_momentum_strategy = 'Cash'  # Default fallback
        
        # Handle stocks field - convert from legacy format if needed
        stocks = json_data.get('stocks', [])
        if not stocks and 'tickers' in json_data:
            # Convert legacy format (tickers, allocs, divs) to stocks format
            tickers = json_data.get('tickers', [])
            allocs = json_data.get('allocs', [])
            divs = json_data.get('divs', [])
            stocks = []
            
            # Ensure we have valid arrays
            if tickers and isinstance(tickers, list):
                for i in range(len(tickers)):
                    if tickers[i] and tickers[i].strip():  # Check for non-empty ticker
                        # Convert allocation from percentage (0-100) to decimal (0.0-1.0) format
                        allocation = 0.0
                        if i < len(allocs) and allocs[i] is not None:
                            alloc_value = float(allocs[i])
                            if alloc_value > 1.0:
                                # Already in percentage format, convert to decimal
                                allocation = alloc_value / 100.0
                            else:
                                # Already in decimal format, use as is
                                allocation = alloc_value
                        
                        stock = {
                            'ticker': tickers[i].strip(),
                            'allocation': allocation,
                            'include_dividends': bool(divs[i]) if i < len(divs) and divs[i] is not None else True
                        }
                        stocks.append(stock)
            

        
        # Sanitize momentum window weights to prevent StreamlitValueAboveMaxError
        momentum_windows = json_data.get('momentum_windows', [])
        for window in momentum_windows:
            if 'weight' in window:
                weight = window['weight']
                # If weight is a percentage (e.g., 50 for 50%), convert to decimal
                if isinstance(weight, (int, float)) and weight > 1.0:
                    # Cap at 100% and convert to decimal
                    weight = min(weight, 100.0) / 100.0
                elif isinstance(weight, (int, float)) and weight <= 1.0:
                    # Already in decimal format, ensure it's valid
                    weight = max(0.0, min(weight, 1.0))
                else:
                    # Invalid weight, set to default
                    weight = 0.1
                window['weight'] = weight
        
                        # Map frequency values from app.py format to Strategy Comparison format
        def map_frequency(freq):
            if freq is None:
                return 'Never'
            freq_map = {
                'Never': 'Never',
                'Buy & Hold': 'Buy & Hold',
                'Buy & Hold (Target)': 'Buy & Hold (Target)',
                'Weekly': 'Weekly',
                'Biweekly': 'Biweekly',
                'Monthly': 'Monthly',
                'Quarterly': 'Quarterly',
                'Semiannually': 'Semiannually',
                'Annually': 'Annually',
                # Legacy format mapping
                'none': 'Never',
                'week': 'Weekly',
                '2weeks': 'Biweekly',
                'month': 'Monthly',
                '3months': 'Quarterly',
                '6months': 'Semiannually',
                'year': 'Annually'
            }
            return freq_map.get(freq, 'Monthly')
        
                        # Strategy Comparison page specific: ensure all required fields are present
        # and ignore fields that are specific to other pages
        strategy_comparison_config = {
            'name': json_data.get('name', 'New Portfolio'),
            'stocks': stocks,
            'benchmark_ticker': json_data.get('benchmark_ticker', '^GSPC'),
            'initial_value': json_data.get('initial_value', 10000),
            'added_amount': json_data.get('added_amount', 1000),
            'added_frequency': map_frequency(json_data.get('added_frequency', 'Monthly')),
            'rebalancing_frequency': map_frequency(json_data.get('rebalancing_frequency', 'Monthly')),
            'start_date_user': parse_date_from_json(json_data.get('start_date_user')),
            'end_date_user': parse_date_from_json(json_data.get('end_date_user')),
            'start_with': json_data.get('start_with', 'all'),
            'first_rebalance_strategy': json_data.get('first_rebalance_strategy', 'rebalancing_date'),
            'use_momentum': json_data.get('use_momentum', True),
            'momentum_strategy': momentum_strategy,
            'negative_momentum_strategy': negative_momentum_strategy,
            'momentum_windows': momentum_windows,
            'use_minimal_threshold': json_data.get('use_minimal_threshold', False),
            'minimal_threshold_percent': json_data.get('minimal_threshold_percent', 2.0),
            'use_max_allocation': json_data.get('use_max_allocation', False),
            'max_allocation_percent': json_data.get('max_allocation_percent', 10.0),
            'calc_beta': json_data.get('calc_beta', True),
            'calc_volatility': json_data.get('calc_volatility', True),
            'beta_window_days': json_data.get('beta_window_days', 365),
            'exclude_days_beta': json_data.get('exclude_days_beta', 30),
            'vol_window_days': json_data.get('vol_window_days', 365),
            'exclude_days_vol': json_data.get('exclude_days_vol', 30),
            'collect_dividends_as_cash': json_data.get('collect_dividends_as_cash', False),
            'saved_momentum_settings': json_data.get('saved_momentum_settings', {}),
            # Preserve sync exclusion settings from imported JSON
            'exclude_from_cashflow_sync': json_data.get('exclude_from_cashflow_sync', False),
            'exclude_from_rebalancing_sync': json_data.get('exclude_from_rebalancing_sync', False),
            'use_targeted_rebalancing': json_data.get('use_targeted_rebalancing', False),
            'targeted_rebalancing_settings': json_data.get('targeted_rebalancing_settings', {}),
            # Note: Ignoring Backtest Engine specific fields like 'portfolio_drag_pct', 'use_custom_dates', etc.
        }
        
        # Fix: ensure proper defaults for beta/volatility windows regardless of calc_beta/calc_volatility state
        if strategy_comparison_config['beta_window_days'] <= 1:
            strategy_comparison_config['beta_window_days'] = 365
        if strategy_comparison_config['exclude_days_beta'] <= 0:
            strategy_comparison_config['exclude_days_beta'] = 30
        if strategy_comparison_config['vol_window_days'] <= 1:
            strategy_comparison_config['vol_window_days'] = 365
        if strategy_comparison_config['exclude_days_vol'] <= 0:
            strategy_comparison_config['exclude_days_vol'] = 30
        
        # Update the configuration with corrected values
        st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] = strategy_comparison_config
        
        # Update global date widgets to match imported portfolio dates (global date range)
        imported_start_date = parse_date_from_json(json_data.get('start_date_user'))
        imported_end_date = parse_date_from_json(json_data.get('end_date_user'))
        
        if imported_start_date is not None:
            st.session_state["strategy_comparison_start_date"] = imported_start_date
            # Update ALL portfolios with the imported start date
            for i, portfolio in enumerate(st.session_state.strategy_comparison_portfolio_configs):
                st.session_state.strategy_comparison_portfolio_configs[i]['start_date_user'] = imported_start_date
        
        if imported_end_date is not None:
            st.session_state["strategy_comparison_end_date"] = imported_end_date
            # Update ALL portfolios with the imported end date
            for i, portfolio in enumerate(st.session_state.strategy_comparison_portfolio_configs):
                st.session_state.strategy_comparison_portfolio_configs[i]['end_date_user'] = imported_end_date
        
        # Update custom dates checkbox based on imported dates
        has_imported_dates = imported_start_date is not None or imported_end_date is not None
        st.session_state["strategy_comparison_use_custom_dates"] = has_imported_dates
        
        # UPDATE UI WIDGET STATES TO REFLECT IMPORTED SETTINGS
        # Update portfolio name
        st.session_state['strategy_comparison_active_name'] = strategy_comparison_config['name']
        
        # Update basic portfolio settings
        st.session_state['strategy_comparison_active_initial'] = int(strategy_comparison_config['initial_value'])
        st.session_state['strategy_comparison_active_added_amount'] = int(strategy_comparison_config['added_amount'])
        st.session_state['strategy_comparison_active_rebal_freq'] = strategy_comparison_config['rebalancing_frequency']
        st.session_state['strategy_comparison_active_add_freq'] = strategy_comparison_config['added_frequency']
        st.session_state['strategy_comparison_active_benchmark'] = strategy_comparison_config['benchmark_ticker']
        
        # Update momentum settings
        st.session_state['strategy_comparison_active_use_momentum'] = strategy_comparison_config['use_momentum']
        st.session_state['strategy_comparison_active_momentum_strategy'] = strategy_comparison_config['momentum_strategy']
        st.session_state['strategy_comparison_active_negative_momentum_strategy'] = strategy_comparison_config['negative_momentum_strategy']
        
        # Update threshold settings
        st.session_state['strategy_comparison_active_use_threshold'] = strategy_comparison_config.get('use_minimal_threshold', False)
        st.session_state['strategy_comparison_active_threshold_percent'] = strategy_comparison_config.get('minimal_threshold_percent', 0.0)
        
        # Update maximum allocation settings
        st.session_state['strategy_comparison_active_use_max_allocation'] = strategy_comparison_config.get('use_max_allocation', False)
        st.session_state['strategy_comparison_active_max_allocation_percent'] = strategy_comparison_config.get('max_allocation_percent', 0.0)
        
        # Update targeted rebalancing settings
        st.session_state['strategy_comparison_active_use_targeted_rebalancing'] = strategy_comparison_config.get('use_targeted_rebalancing', False)
        
        # Update beta settings
        st.session_state['strategy_comparison_active_calc_beta'] = strategy_comparison_config['calc_beta']
        st.session_state['strategy_comparison_active_beta_window'] = strategy_comparison_config['beta_window_days']
        st.session_state['strategy_comparison_active_beta_exclude'] = strategy_comparison_config['exclude_days_beta']
        
        # Update volatility settings
        st.session_state['strategy_comparison_active_calc_vol'] = strategy_comparison_config['calc_volatility']
        st.session_state['strategy_comparison_active_vol_window'] = strategy_comparison_config['vol_window_days']
        st.session_state['strategy_comparison_active_vol_exclude'] = strategy_comparison_config['exclude_days_vol']
        
        # Update dividend settings
        st.session_state['strategy_comparison_active_collect_dividends_as_cash'] = strategy_comparison_config['collect_dividends_as_cash']
        
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
        
        # UPDATE GLOBAL TICKERS FROM IMPORTED JSON
        if stocks:
            # Clear existing global tickers
            st.session_state.strategy_comparison_global_tickers = []
            
            # Clear all ticker widget keys to prevent UI interference
            for key in list(st.session_state.keys()):
                if key.startswith("strategy_comparison_global_ticker_") or key.startswith("strategy_comparison_global_alloc_") or key.startswith("strategy_comparison_global_div_"):
                    del st.session_state[key]
            
            # Update global tickers from imported stocks
            for stock in stocks:
                if stock.get('ticker'):
                    st.session_state.strategy_comparison_global_tickers.append({
                        'ticker': stock['ticker'],
                        'allocation': stock.get('allocation', 0.0),
                        'include_dividends': stock.get('include_dividends', True)
                    })
            
            # DON'T sync global tickers to all portfolios - this would overwrite the imported settings
            # Instead, just update the current portfolio's stocks to match global tickers
            st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['stocks'] = st.session_state.strategy_comparison_global_tickers.copy()
        
        # Sync date widgets with the updated portfolio
        sync_date_widgets_with_portfolio()
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check the text and try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    st.session_state.strategy_comparison_rerun_flag = True

# Sidebar JSON export/import for ALL portfolios
def paste_all_json_callback():
    txt = st.session_state.get('strategy_comparison_paste_all_json_text', '')
    if not txt:
        st.warning('No JSON provided')
        return
    try:
        # Use the SAME parsing logic as successful PDF extraction
        raw_text = txt
        
        # STEP 1: Try the exact same approach as PDF extraction (simple strip + parse)
        try:
            cleaned_text = raw_text.strip()
            obj = json.loads(cleaned_text)
            st.success("✅ Multi-portfolio JSON parsed successfully using PDF-style parsing!")
        except json.JSONDecodeError:
            # STEP 2: If that fails, apply our advanced cleaning (fallback)
            st.info("🔧 Simple parsing failed, applying advanced PDF extraction fixes...")
            
            json_text = raw_text
            import re
            
            # Fix broken portfolio name lines
            broken_pattern = r'"name":\s*"([^"]*?)"\s*"stocks":'
            json_text = re.sub(broken_pattern, r'"name": "\1", "stocks":', json_text)
            
            # Fix truncated names
            truncated_pattern = r'"name":\s*"([^"]*?)\s+"stocks":'
            json_text = re.sub(truncated_pattern, r'"name": "\1", "stocks":', json_text)
            
            # Fix missing opening brace for portfolio objects
            missing_brace_pattern = r'(},)\s*("name":)'
            json_text = re.sub(missing_brace_pattern, r'\1 {\n \2', json_text)
            
            obj = json.loads(json_text)
            st.success("✅ Multi-portfolio JSON parsed successfully using advanced cleaning!")
        
        # Add missing fields for compatibility if they don't exist
        if isinstance(obj, list):
            for portfolio in obj:
                # Add missing fields with default values
                if 'collect_dividends_as_cash' not in portfolio:
                    portfolio['collect_dividends_as_cash'] = False
                if 'exclude_from_cashflow_sync' not in portfolio:
                    portfolio['exclude_from_cashflow_sync'] = False
                if 'exclude_from_rebalancing_sync' not in portfolio:
                    portfolio['exclude_from_rebalancing_sync'] = False
                if 'use_minimal_threshold' not in portfolio:
                    portfolio['use_minimal_threshold'] = False
                if 'minimal_threshold_percent' not in portfolio:
                    portfolio['minimal_threshold_percent'] = 2.0
                if 'use_max_allocation' not in portfolio:
                    portfolio['use_max_allocation'] = False
                if 'max_allocation_percent' not in portfolio:
                    portfolio['max_allocation_percent'] = 10.0
        
        if isinstance(obj, list):
            # Clear widget keys to force re-initialization
            widget_keys_to_clear = [
                "strategy_comparison_active_name", "strategy_comparison_active_initial", 
                "strategy_comparison_active_added_amount", "strategy_comparison_active_rebal_freq",
                "strategy_comparison_active_add_freq", "strategy_comparison_active_benchmark",
                "strategy_comparison_active_use_momentum", "strategy_comparison_active_collect_dividends_as_cash",
                "strategy_comparison_start_with_radio", "strategy_comparison_first_rebalance_strategy_radio"
                # Removed beta/volatility keys so they preserve values like momentum windows
            ]
            for key in widget_keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Process each portfolio configuration for Strategy Comparison page (existing logic)
            processed_configs = []
            for cfg in obj:
                if not isinstance(cfg, dict) or 'name' not in cfg:
                    st.error('Invalid portfolio configuration structure.')
                    return
                
                # Handle momentum strategy value mapping from other pages
                momentum_strategy = cfg.get('momentum_strategy', 'Classic')
                if momentum_strategy == 'Classic momentum':
                    momentum_strategy = 'Classic'
                elif momentum_strategy == 'Relative momentum':
                    momentum_strategy = 'Relative Momentum'
                elif momentum_strategy not in ['Classic', 'Relative Momentum']:
                    momentum_strategy = 'Classic'  # Default fallback
                
                # Handle negative momentum strategy value mapping from other pages
                negative_momentum_strategy = cfg.get('negative_momentum_strategy', 'Cash')
                if negative_momentum_strategy == 'Go to cash':
                    negative_momentum_strategy = 'Cash'
                elif negative_momentum_strategy not in ['Cash', 'Equal weight', 'Relative momentum']:
                    negative_momentum_strategy = 'Cash'  # Default fallback
                
                # Handle stocks field - convert from legacy format if needed
                stocks = cfg.get('stocks', [])
                if not stocks and 'tickers' in cfg:
                    # Convert legacy format (tickers, allocs, divs) to stocks format
                    tickers = cfg.get('tickers', [])
                    allocs = cfg.get('allocs', [])
                    divs = cfg.get('divs', [])
                    stocks = []
                    
                    # Ensure we have valid arrays
                    if tickers and isinstance(tickers, list):
                        for i in range(len(tickers)):
                            if tickers[i] and tickers[i].strip():  # Check for non-empty ticker
                                stock = {
                                    'ticker': tickers[i].strip(),
                                    'allocation': float(allocs[i]) if i < len(allocs) and allocs[i] is not None else 0.0,
                                    'include_dividends': bool(divs[i]) if i < len(divs) and divs[i] is not None else True
                                }
                                stocks.append(stock)
                
                # Sanitize momentum window weights to prevent StreamlitValueAboveMaxError
                momentum_windows = cfg.get('momentum_windows', [])
                for window in momentum_windows:
                    if 'weight' in window:
                        weight = window['weight']
                        # If weight is a percentage (e.g., 50 for 50%), convert to decimal
                        if isinstance(weight, (int, float)) and weight > 1.0:
                            # Cap at 100% and convert to decimal
                            weight = min(weight, 100.0) / 100.0
                        elif isinstance(weight, (int, float)) and weight <= 1.0:
                            # Already in decimal format, ensure it's valid
                            weight = max(0.0, min(weight, 1.0))
                        else:
                            # Invalid weight, set to default
                            weight = 0.1
                        window['weight'] = weight
                

                
                # Map frequency values from app.py format to Strategy Comparison format
                def map_frequency(freq):
                    if freq is None:
                        return 'Never'
                    freq_map = {
                        'Never': 'Never',
                        'Buy & Hold': 'Buy & Hold',
                        'Buy & Hold (Target)': 'Buy & Hold (Target)',
                        'Weekly': 'Weekly',
                        'Biweekly': 'Biweekly',
                        'Monthly': 'Monthly',
                        'Quarterly': 'Quarterly',
                        'Semiannually': 'Semiannually',
                        'Annually': 'Annually',
                        # Legacy format mapping
                        'none': 'Never',
                        'week': 'Weekly',
                        '2weeks': 'Biweekly',
                        'month': 'Monthly',
                        '3months': 'Quarterly',
                        '6months': 'Semiannually',
                        'year': 'Annually'
                    }
                    return freq_map.get(freq, 'Monthly')
                
                # Strategy Comparison page specific: ensure all required fields are present
                # and ignore fields that are specific to other pages
                strategy_comparison_config = {
                    'name': cfg.get('name', 'New Portfolio'),
                    'stocks': stocks,
                    'benchmark_ticker': cfg.get('benchmark_ticker', '^GSPC'),
                    'initial_value': cfg.get('initial_value', 10000),
                    'added_amount': cfg.get('added_amount', 1000),
                    'added_frequency': map_frequency(cfg.get('added_frequency', 'Monthly')),
                    'rebalancing_frequency': map_frequency(cfg.get('rebalancing_frequency', 'Monthly')),
                                    'start_date_user': parse_date_from_json(cfg.get('start_date_user')),
                'end_date_user': parse_date_from_json(cfg.get('end_date_user')),
                    'start_with': cfg.get('start_with', 'all'),
                    'first_rebalance_strategy': cfg.get('first_rebalance_strategy', 'rebalancing_date'),
                    'use_momentum': cfg.get('use_momentum', True),
                    'momentum_strategy': momentum_strategy,
                    'negative_momentum_strategy': negative_momentum_strategy,
                    'momentum_windows': momentum_windows,
                    'use_minimal_threshold': cfg.get('use_minimal_threshold', False),
                    'minimal_threshold_percent': cfg.get('minimal_threshold_percent', 2.0),
                    'use_max_allocation': cfg.get('use_max_allocation', False),
                    'max_allocation_percent': cfg.get('max_allocation_percent', 10.0),
                    'calc_beta': cfg.get('calc_beta', True),
                    'calc_volatility': cfg.get('calc_volatility', True),
                    'beta_window_days': cfg.get('beta_window_days', 365),
                    'exclude_days_beta': cfg.get('exclude_days_beta', 30),
                    'vol_window_days': cfg.get('vol_window_days', 365),
                    'exclude_days_vol': cfg.get('exclude_days_vol', 30),
                    'collect_dividends_as_cash': cfg.get('collect_dividends_as_cash', False),
                    'saved_momentum_settings': cfg.get('saved_momentum_settings', {}),
                    # Preserve sync exclusion settings from imported JSON
                    'exclude_from_cashflow_sync': cfg.get('exclude_from_cashflow_sync', False),
                    'exclude_from_rebalancing_sync': cfg.get('exclude_from_rebalancing_sync', False),
                    'use_targeted_rebalancing': cfg.get('use_targeted_rebalancing', False),
                    'targeted_rebalancing_settings': cfg.get('targeted_rebalancing_settings', {}),
                    # Note: Ignoring Backtest Engine specific fields like 'portfolio_drag_pct', 'use_custom_dates', etc.
                }
                processed_configs.append(strategy_comparison_config)
            
            st.session_state.strategy_comparison_portfolio_configs = processed_configs
            
            # Update global date widgets based on imported portfolios (use first portfolio's dates as global)
            if processed_configs:
                first_portfolio = processed_configs[0]
                imported_start_date = first_portfolio.get('start_date_user')
                imported_end_date = first_portfolio.get('end_date_user')
                
                if imported_start_date is not None:
                    st.session_state["strategy_comparison_start_date"] = imported_start_date
                if imported_end_date is not None:
                    st.session_state["strategy_comparison_end_date"] = imported_end_date
                
                # Update custom dates checkbox based on imported dates
                has_imported_dates = imported_start_date is not None or imported_end_date is not None
                st.session_state["strategy_comparison_use_custom_dates"] = has_imported_dates
            
            # Reset active selection and derived mappings so the UI reflects the new configs
            if processed_configs:
                st.session_state.strategy_comparison_active_portfolio_index = 0
                st.session_state.strategy_comparison_portfolio_selector = processed_configs[0].get('name', '')
                # Mirror several active_* widget defaults so the UI selectboxes/inputs update
                st.session_state['strategy_comparison_active_name'] = processed_configs[0].get('name', '')
                st.session_state['strategy_comparison_active_initial'] = int(processed_configs[0].get('initial_value', 0) or 0)
                st.session_state['strategy_comparison_active_added_amount'] = int(processed_configs[0].get('added_amount', 0) or 0)
                st.session_state['strategy_comparison_active_rebal_freq'] = processed_configs[0].get('rebalancing_frequency', 'none')
                st.session_state['strategy_comparison_active_add_freq'] = processed_configs[0].get('added_frequency', 'none')
                st.session_state['strategy_comparison_active_benchmark'] = processed_configs[0].get('benchmark_ticker', '')
                st.session_state['strategy_comparison_active_use_momentum'] = bool(processed_configs[0].get('use_momentum', True))
                st.session_state['strategy_comparison_active_use_targeted_rebalancing'] = bool(processed_configs[0].get('use_targeted_rebalancing', False))
                st.session_state['strategy_comparison_active_collect_dividends_as_cash'] = bool(processed_configs[0].get('collect_dividends_as_cash', False))
                st.session_state['strategy_comparison_active_use_threshold'] = bool(processed_configs[0].get('use_minimal_threshold', False))
                st.session_state['strategy_comparison_active_threshold_percent'] = float(processed_configs[0].get('minimal_threshold_percent', 0.0))
                
                # UPDATE GLOBAL TICKERS FROM FIRST PORTFOLIO
                first_portfolio_stocks = processed_configs[0].get('stocks', [])
                if first_portfolio_stocks:
                    # Clear existing global tickers
                    st.session_state.strategy_comparison_global_tickers = []
                    
                    # Clear all ticker widget keys to prevent UI interference
                    for key in list(st.session_state.keys()):
                        if key.startswith("strategy_comparison_global_ticker_") or key.startswith("strategy_comparison_global_alloc_") or key.startswith("strategy_comparison_global_div_"):
                            del st.session_state[key]
                    
                    # Update global tickers from first portfolio
                    for stock in first_portfolio_stocks:
                        if stock.get('ticker'):
                            st.session_state.strategy_comparison_global_tickers.append({
                                'ticker': stock['ticker'],
                                'allocation': stock.get('allocation', 0.0),
                                'include_dividends': stock.get('include_dividends', True)
                            })
                    
                    # Sync global tickers to all portfolios
                    sync_global_tickers_to_all_portfolios()
                
                # Update global first rebalance strategy from first portfolio
                if 'first_rebalance_strategy' in processed_configs[0]:
                    st.session_state['_import_first_rebalance_strategy'] = processed_configs[0]['first_rebalance_strategy']
                
                # Update global start_with setting from first portfolio
                if 'start_with' in processed_configs[0]:
                    start_with = processed_configs[0]['start_with']
                    if start_with == 'first':
                        start_with = 'oldest'  # Map 'first' to 'oldest' (closest equivalent)
                    elif start_with not in ['all', 'oldest']:
                        start_with = 'all'  # Default fallback
                    st.session_state['_import_start_with'] = start_with
                
            else:
                st.session_state.strategy_comparison_active_portfolio_index = None
                st.session_state.strategy_comparison_portfolio_selector = ''
            st.session_state.strategy_comparison_portfolio_key_map = {}
            st.session_state.strategy_comparison_ran = False
            st.success('All portfolio configurations updated from JSON.')
            if processed_configs:
                st.info(f"Sync exclusions for first portfolio - Cash Flow: {processed_configs[0].get('exclude_from_cashflow_sync', False)}, Rebalancing: {processed_configs[0].get('exclude_from_rebalancing_sync', False)}")
            
            # Sync date widgets with the updated portfolio
            sync_date_widgets_with_portfolio()
            
            # Force a rerun so widgets rebuild with the new configs
            try:
                st.experimental_rerun()
            except Exception:
                # In some environments experimental rerun may raise; setting a rerun flag is a fallback
                st.session_state.strategy_comparison_rerun_flag = True
        else:
            st.error('JSON must be a list of portfolio configurations.')
    except Exception as e:
        st.error(f'Failed to parse JSON: {e}')

def update_active_portfolio_index():
    # Use safe accessors to avoid AttributeError when keys are not yet set
    selected_name = st.session_state.get('strategy_comparison_portfolio_selector', None)
    portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
    portfolio_names = [cfg.get('name', '') for cfg in portfolio_configs]
    
    if selected_name and selected_name in portfolio_names:
        new_index = portfolio_names.index(selected_name)
        st.session_state.strategy_comparison_active_portfolio_index = new_index
    else:
        # default to first portfolio if selector is missing or value not found
        st.session_state.strategy_comparison_active_portfolio_index = 0 if portfolio_names else None
    
    # Additional safety check - ensure index is always valid
    if (st.session_state.strategy_comparison_active_portfolio_index is not None and 
        st.session_state.strategy_comparison_active_portfolio_index >= len(portfolio_names)):
        st.session_state.strategy_comparison_active_portfolio_index = max(0, len(portfolio_names) - 1) if portfolio_names else None
    
    # Sync date widgets with the new portfolio
    sync_date_widgets_with_portfolio()
    
    # NUCLEAR SYNC: FORCE momentum widgets to sync with the new portfolio
    if portfolio_configs and st.session_state.strategy_comparison_active_portfolio_index is not None:
        active_portfolio = portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        
        # NUCLEAR APPROACH: FORCE all momentum session state widgets to sync
        st.session_state['strategy_comparison_active_use_momentum'] = active_portfolio.get('use_momentum', False)
        st.session_state['strategy_comparison_active_momentum_strategy'] = active_portfolio.get('momentum_strategy', 'Classic')
        st.session_state['strategy_comparison_active_negative_momentum_strategy'] = active_portfolio.get('negative_momentum_strategy', 'Cash')
        st.session_state['strategy_comparison_active_calc_beta'] = active_portfolio.get('calc_beta', False)
        st.session_state['strategy_comparison_active_calc_vol'] = active_portfolio.get('calc_volatility', False)
        st.session_state['strategy_comparison_active_beta_window'] = active_portfolio.get('beta_window_days', 365)
        st.session_state['strategy_comparison_active_beta_exclude'] = active_portfolio.get('exclude_days_beta', 30)
        st.session_state['strategy_comparison_active_vol_window'] = active_portfolio.get('vol_window_days', 365)
        st.session_state['strategy_comparison_active_vol_exclude'] = active_portfolio.get('exclude_days_vol', 30)
        
        # Sync expander state (same pattern as other portfolio parameters)
        st.session_state['strategy_comparison_active_variant_expanded'] = active_portfolio.get('variant_expander_expanded', False)
        st.session_state['strategy_comparison_active_use_threshold'] = active_portfolio.get('use_minimal_threshold', False)
        st.session_state['strategy_comparison_active_threshold_percent'] = active_portfolio.get('minimal_threshold_percent', 0.0)
        
        # NUCLEAR: If portfolio has momentum enabled but no windows, FORCE create them
        if active_portfolio.get('use_momentum', False) and not active_portfolio.get('momentum_windows'):
            active_portfolio['momentum_windows'] = [
                {"lookback": 365, "exclude": 30, "weight": 0.5},
                {"lookback": 180, "exclude": 30, "weight": 0.3},
                {"lookback": 120, "exclude": 30, "weight": 0.2},
            ]
            print(f"NUCLEAR: FORCED momentum windows for portfolio {active_portfolio.get('name', 'Unknown')}")
        
        # NUCLEAR: Ensure threshold settings exist
        if 'use_minimal_threshold' not in active_portfolio:
            active_portfolio['use_minimal_threshold'] = False
        if 'minimal_threshold_percent' not in active_portfolio:
            active_portfolio['minimal_threshold_percent'] = 2.0
        
        print(f"NUCLEAR: Synced momentum widgets for portfolio {active_portfolio.get('name', 'Unknown')}, use_momentum={active_portfolio.get('use_momentum', False)}, windows_count={len(active_portfolio.get('momentum_windows', []))}")
        
        # RESET variant generator checkboxes when switching portfolios
        # This prevents stale checkbox states from previous portfolio selections
        variant_generator_keys = [
            "strategy_use_momentum_vary",
            # Rebalance frequency checkboxes
            "strategy_rebalance_never", "strategy_rebalance_buyhold", "strategy_rebalance_buyhold_target",
            "strategy_rebalance_weekly", "strategy_rebalance_biweekly", "strategy_rebalance_monthly",
            "strategy_rebalance_quarterly", "strategy_rebalance_semiannually", "strategy_rebalance_annually",
            # Momentum variant checkboxes
            "strategy_momentum_classic", "strategy_momentum_relative",
            "strategy_negative_cash", "strategy_negative_equal", "strategy_negative_relative", 
            "strategy_beta_yes", "strategy_beta_no", "strategy_vol_yes", "strategy_vol_no"
        ]
        for key in variant_generator_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    st.session_state.strategy_comparison_rerun_flag = True

def update_name():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['name'] = st.session_state.strategy_comparison_active_name

def update_initial():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['initial_value'] = st.session_state.strategy_comparison_active_initial

def update_added_amount():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['added_amount'] = st.session_state.strategy_comparison_active_added_amount

def update_start_date():
    """Update all portfolio configs when start date changes"""
    start_date = st.session_state.strategy_comparison_start_date
    for i, portfolio in enumerate(st.session_state.strategy_comparison_portfolio_configs):
        st.session_state.strategy_comparison_portfolio_configs[i]['start_date_user'] = start_date

def update_end_date():
    """Update all portfolio configs when end date changes"""
    end_date = st.session_state.strategy_comparison_end_date
    for i, portfolio in enumerate(st.session_state.strategy_comparison_portfolio_configs):
        st.session_state.strategy_comparison_portfolio_configs[i]['end_date_user'] = end_date

def update_custom_dates_checkbox():
    """Update checkbox state when custom dates are toggled"""
    # This function ensures the checkbox state is properly maintained
    pass  # The checkbox state is managed by Streamlit automatically

def clear_dates_callback():
    """Clear the date inputs and reset to None for ALL portfolios"""
    st.session_state.strategy_comparison_start_date = None
    st.session_state.strategy_comparison_end_date = date.today()
    st.session_state.strategy_comparison_use_custom_dates = False
    # Clear from ALL portfolio configs (global date range)
    for i, portfolio in enumerate(st.session_state.strategy_comparison_portfolio_configs):
        st.session_state.strategy_comparison_portfolio_configs[i]['start_date_user'] = None
        st.session_state.strategy_comparison_portfolio_configs[i]['end_date_user'] = None

def parse_date_from_json(date_value):
    """Parse date from JSON string format back to date object"""
    if date_value is None:
        return None
    if isinstance(date_value, date):
        return date_value
    if isinstance(date_value, str):
        try:
            return datetime.strptime(date_value, '%Y-%m-%d').date()
        except ValueError:
            try:
                # Try parsing as ISO format
                return datetime.fromisoformat(date_value).date()
            except ValueError:
                return None
    return None

def sync_date_widgets_with_portfolio():
    """Sync date widgets with current portfolio configuration"""
    from datetime import date
    if st.session_state.strategy_comparison_active_portfolio_index is not None:
        portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        
        # Sync start date
        portfolio_start_date = portfolio.get('start_date_user')
        if portfolio_start_date is not None:
            st.session_state["strategy_comparison_start_date"] = portfolio_start_date
        else:
            st.session_state["strategy_comparison_start_date"] = date(2010, 1, 1)
        
        # Sync end date
        portfolio_end_date = portfolio.get('end_date_user')
        if portfolio_end_date is not None:
            st.session_state["strategy_comparison_end_date"] = portfolio_end_date
        else:
            st.session_state["strategy_comparison_end_date"] = date.today()
        
        # Sync custom dates checkbox
        has_custom_dates = portfolio_start_date is not None or portfolio_end_date is not None
        st.session_state["strategy_comparison_use_custom_dates"] = has_custom_dates

def update_add_freq():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['added_frequency'] = st.session_state.strategy_comparison_active_add_freq

def update_rebal_freq():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['rebalancing_frequency'] = st.session_state.strategy_comparison_active_rebal_freq

def update_benchmark():
    # Convert commas to dots for decimal separators (like case conversion)
    converted_benchmark = st.session_state.strategy_comparison_active_benchmark.replace(",", ".")
    
    # Convert benchmark ticker to uppercase and resolve alias
    upper_benchmark = converted_benchmark.upper()
    resolved_benchmark = resolve_ticker_alias(upper_benchmark)
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['benchmark_ticker'] = resolved_benchmark
    # Update the widget to show resolved ticker
    st.session_state.strategy_comparison_active_benchmark = resolved_benchmark

def update_use_momentum():
    current_val = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['use_momentum']
    new_val = st.session_state.strategy_comparison_active_use_momentum
    
    if current_val != new_val:
        portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        
        if new_val:
            # Enabling momentum - restore saved settings or use defaults
            if 'saved_momentum_settings' in portfolio:
                # Restore previously saved momentum settings
                saved_settings = portfolio['saved_momentum_settings']
                portfolio['momentum_windows'] = saved_settings.get('momentum_windows', [
                    {"lookback": 365, "exclude": 30, "weight": 0.5},
                    {"lookback": 180, "exclude": 30, "weight": 0.3},
                    {"lookback": 120, "exclude": 30, "weight": 0.2},
                ])
                portfolio['momentum_strategy'] = saved_settings.get('momentum_strategy', 'Classic')
                portfolio['negative_momentum_strategy'] = saved_settings.get('negative_momentum_strategy', 'Cash')
                portfolio['calc_beta'] = saved_settings.get('calc_beta', True)
                portfolio['calc_volatility'] = saved_settings.get('calc_volatility', True)
                portfolio['beta_window_days'] = saved_settings.get('beta_window_days', 365)
                portfolio['exclude_days_beta'] = saved_settings.get('exclude_days_beta', 30)
                portfolio['vol_window_days'] = saved_settings.get('vol_window_days', 365)
                portfolio['exclude_days_vol'] = saved_settings.get('exclude_days_vol', 30)
                
                # Update UI widgets to reflect restored values
                st.session_state['strategy_comparison_active_momentum_strategy'] = portfolio['momentum_strategy']
                st.session_state['strategy_comparison_active_negative_momentum_strategy'] = portfolio['negative_momentum_strategy']
                st.session_state['strategy_comparison_active_calc_beta'] = portfolio['calc_beta']
                st.session_state['strategy_comparison_active_calc_vol'] = portfolio['calc_volatility']
                st.session_state['strategy_comparison_active_beta_window'] = portfolio['beta_window_days']
                st.session_state['strategy_comparison_active_beta_exclude'] = portfolio['exclude_days_beta']
                st.session_state['strategy_comparison_active_vol_window'] = portfolio['vol_window_days']
                st.session_state['strategy_comparison_active_vol_exclude'] = portfolio['exclude_days_vol']
            else:
                # SMART NUCLEAR: No saved settings, create defaults only if no windows exist
                if not portfolio.get('momentum_windows'):
                    portfolio['momentum_windows'] = [
                        {"lookback": 365, "exclude": 30, "weight": 0.5},
                        {"lookback": 180, "exclude": 30, "weight": 0.3},
                        {"lookback": 120, "exclude": 30, "weight": 0.2},
                    ]
                    print("SMART NUCLEAR: Added default momentum windows (had none)")
                else:
                    print(f"SMART NUCLEAR: Preserved existing momentum windows (had {len(portfolio['momentum_windows'])} windows)")
                # Set default momentum settings only if not already set
                portfolio['momentum_strategy'] = portfolio.get('momentum_strategy', 'Classic')
                portfolio['negative_momentum_strategy'] = portfolio.get('negative_momentum_strategy', 'Cash')
                portfolio['calc_beta'] = portfolio.get('calc_beta', True)
                portfolio['calc_volatility'] = portfolio.get('calc_volatility', True)
        else:
            # Disabling momentum - save current settings before clearing
            saved_settings = {
                'momentum_windows': portfolio.get('momentum_windows', []),
                'momentum_strategy': portfolio.get('momentum_strategy', 'Classic'),
                'negative_momentum_strategy': portfolio.get('negative_momentum_strategy', 'Cash'),
                'calc_beta': portfolio.get('calc_beta', True),
                'calc_volatility': portfolio.get('calc_volatility', True),
                'beta_window_days': portfolio.get('beta_window_days', 365),
                'exclude_days_beta': portfolio.get('exclude_days_beta', 30),
                'vol_window_days': portfolio.get('vol_window_days', 365),
                'exclude_days_vol': portfolio.get('exclude_days_vol', 30),
            }
            portfolio['saved_momentum_settings'] = saved_settings
            # Don't clear momentum_windows - preserve them for variant generation
        
        portfolio['use_momentum'] = new_val
        st.session_state.strategy_comparison_rerun_flag = True

def update_use_targeted_rebalancing():
    """Callback function for targeted rebalancing checkbox"""
    current_val = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index].get('use_targeted_rebalancing', False)
    new_val = st.session_state.strategy_comparison_active_use_targeted_rebalancing
    
    if current_val != new_val:
        portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        portfolio['use_targeted_rebalancing'] = new_val
        
        # If enabling targeted rebalancing, disable momentum
        if new_val:
            portfolio['use_momentum'] = False
            st.session_state['strategy_comparison_active_use_momentum'] = False
        
        st.session_state.strategy_comparison_rerun_flag = True

def update_calc_beta():
    current_val = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['calc_beta']
    new_val = st.session_state.strategy_comparison_active_calc_beta
    
    if current_val != new_val:
        portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        
        if new_val:
            # Enabling beta - restore saved settings or use defaults
            if 'saved_beta_settings' in portfolio:
                # Restore previously saved beta settings
                saved_settings = portfolio['saved_beta_settings']
                portfolio['beta_window_days'] = saved_settings.get('beta_window_days', 365)
                portfolio['exclude_days_beta'] = saved_settings.get('exclude_days_beta', 30)
                
                # Update UI widgets to reflect restored values
                st.session_state['strategy_comparison_active_beta_window'] = portfolio['beta_window_days']
                st.session_state['strategy_comparison_active_beta_exclude'] = portfolio['exclude_days_beta']
            else:
                # No saved settings, use current portfolio values or defaults
                beta_window = portfolio.get('beta_window_days', 365)
                beta_exclude = portfolio.get('exclude_days_beta', 30)
                portfolio['beta_window_days'] = beta_window
                portfolio['exclude_days_beta'] = beta_exclude
                st.session_state['strategy_comparison_active_beta_window'] = beta_window
                st.session_state['strategy_comparison_active_beta_exclude'] = beta_exclude
        else:
            # Disabling beta - save current SESSION STATE values (user's input) to BOTH saved settings AND main portfolio
            beta_window = st.session_state.get('strategy_comparison_active_beta_window', portfolio.get('beta_window_days', 365))
            beta_exclude = st.session_state.get('strategy_comparison_active_beta_exclude', portfolio.get('exclude_days_beta', 30))
            
            # Save to main portfolio keys (so variants inherit them)
            portfolio['beta_window_days'] = beta_window
            portfolio['exclude_days_beta'] = beta_exclude
            
            # Also save to saved_settings (for restore later)
            saved_settings = {
                'beta_window_days': beta_window,
                'exclude_days_beta': beta_exclude,
            }
            portfolio['saved_beta_settings'] = saved_settings
        
        portfolio['calc_beta'] = new_val
        st.session_state.strategy_comparison_rerun_flag = True

def update_sync_exclusion(sync_type):
    """Update sync exclusion settings when checkboxes change"""
    try:
        portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        
        if sync_type == 'cashflow':
            key = f"strategy_comparison_exclude_cashflow_sync_{st.session_state.strategy_comparison_active_portfolio_index}"
            if key in st.session_state:
                portfolio['exclude_from_cashflow_sync'] = st.session_state[key]
        elif sync_type == 'rebalancing':
            key = f"strategy_comparison_exclude_rebalancing_sync_{st.session_state.strategy_comparison_active_portfolio_index}"
            if key in st.session_state:
                portfolio['exclude_from_rebalancing_sync'] = st.session_state[key]
        
        # Force immediate update to session state
        st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] = portfolio
        st.session_state.strategy_comparison_rerun_flag = True
    except Exception:
        pass

def update_beta_window():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['beta_window_days'] = st.session_state.strategy_comparison_active_beta_window

def update_beta_exclude():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['exclude_days_beta'] = st.session_state.strategy_comparison_active_beta_exclude

def update_calc_vol():
    current_val = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['calc_volatility']
    new_val = st.session_state.strategy_comparison_active_calc_vol
    
    if current_val != new_val:
        portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        
        if new_val:
            # Enabling volatility - restore saved settings or use defaults
            if 'saved_vol_settings' in portfolio:
                # Restore previously saved volatility settings
                saved_settings = portfolio['saved_vol_settings']
                portfolio['vol_window_days'] = saved_settings.get('vol_window_days', 365)
                portfolio['exclude_days_vol'] = saved_settings.get('exclude_days_vol', 30)
                
                # Update UI widgets to reflect restored values
                st.session_state['strategy_comparison_active_vol_window'] = portfolio['vol_window_days']
                st.session_state['strategy_comparison_active_vol_exclude'] = portfolio['exclude_days_vol']
            else:
                # No saved settings, use current portfolio values or defaults
                vol_window = portfolio.get('vol_window_days', 365)
                vol_exclude = portfolio.get('exclude_days_vol', 30)
                portfolio['vol_window_days'] = vol_window
                portfolio['exclude_days_vol'] = vol_exclude
                st.session_state['strategy_comparison_active_vol_window'] = vol_window
                st.session_state['strategy_comparison_active_vol_exclude'] = vol_exclude
        else:
            # Disabling volatility - save current SESSION STATE values (user's input) to BOTH saved settings AND main portfolio
            vol_window = st.session_state.get('strategy_comparison_active_vol_window', portfolio.get('vol_window_days', 365))
            vol_exclude = st.session_state.get('strategy_comparison_active_vol_exclude', portfolio.get('exclude_days_vol', 30))
            
            # Save to main portfolio keys (so variants inherit them)
            portfolio['vol_window_days'] = vol_window
            portfolio['exclude_days_vol'] = vol_exclude
            
            # Also save to saved_settings (for restore later)
            saved_settings = {
                'vol_window_days': vol_window,
                'exclude_days_vol': vol_exclude,
            }
            portfolio['saved_vol_settings'] = saved_settings
        
        portfolio['calc_volatility'] = new_val
        st.session_state.strategy_comparison_rerun_flag = True

def update_vol_window():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['vol_window_days'] = st.session_state.strategy_comparison_active_vol_window

def update_vol_exclude():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['exclude_days_vol'] = st.session_state.strategy_comparison_active_vol_exclude

def update_use_threshold():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['use_minimal_threshold'] = st.session_state.strategy_comparison_active_use_threshold

def update_threshold_percent():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['minimal_threshold_percent'] = st.session_state.strategy_comparison_active_threshold_percent

def update_use_max_allocation():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['use_max_allocation'] = st.session_state.strategy_comparison_active_use_max_allocation
    st.session_state.strategy_comparison_rerun_flag = True

def update_max_allocation_percent():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['max_allocation_percent'] = st.session_state.strategy_comparison_active_max_allocation_percent
    st.session_state.strategy_comparison_rerun_flag = True

def update_collect_dividends_as_cash():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['collect_dividends_as_cash'] = st.session_state.strategy_comparison_active_collect_dividends_as_cash

# Sidebar for portfolio selection
st.sidebar.title("Manage Portfolios")
portfolio_names = [cfg['name'] for cfg in st.session_state.strategy_comparison_portfolio_configs]

# Ensure the active portfolio index is valid
if (st.session_state.strategy_comparison_active_portfolio_index is None or 
    st.session_state.strategy_comparison_active_portfolio_index >= len(portfolio_names) or
    st.session_state.strategy_comparison_active_portfolio_index < 0):
    st.session_state.strategy_comparison_active_portfolio_index = 0 if portfolio_names else None

# Use the current portfolio name as the default selection to make it more reliable
current_portfolio_name = None
if (st.session_state.strategy_comparison_active_portfolio_index is not None and 
    st.session_state.strategy_comparison_active_portfolio_index < len(portfolio_names)):
    current_portfolio_name = portfolio_names[st.session_state.strategy_comparison_active_portfolio_index]

selected_portfolio_name = st.sidebar.selectbox(
    "Select Portfolio",
    options=portfolio_names,
    index=st.session_state.strategy_comparison_active_portfolio_index,
    key="strategy_comparison_portfolio_selector",
    on_change=update_active_portfolio_index
)

active_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]

if st.sidebar.button("Add New Portfolio", on_click=add_portfolio_callback):
    pass

# Individual portfolio removal (original functionality)
if len(st.session_state.strategy_comparison_portfolio_configs) > 1:
    if st.sidebar.button("Remove Selected Portfolio", on_click=remove_portfolio_callback):
        pass

# Reset selected portfolio button
if st.sidebar.button("Reset Selected Portfolio", on_click=reset_portfolio_callback):
    pass

# Clear all portfolios button - quick access outside dropdown
if st.sidebar.button("🗑️ Clear All Portfolios", key="strategy_comparison_clear_all_portfolios_immediate", 
                    help="Delete ALL portfolios and create a blank one", use_container_width=True):
    # Clear all portfolios and create a single blank portfolio
    st.session_state.strategy_comparison_portfolio_configs = [{
        'name': 'New Portfolio 1',
        'stocks': [],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 0,
        'added_frequency': 'none',
        'rebalancing_frequency': 'Monthly',
        'start_with': 'all',
        'first_rebalance_strategy': 'rebalancing_date',
        'use_momentum': False,
        'momentum_strategy': 'Classic',
        'negative_momentum_strategy': 'Cash',
        'momentum_windows': [
            {"lookback": 365, "exclude": 30, "weight": 0.5},
            {"lookback": 180, "exclude": 30, "weight": 0.3},
            {"lookback": 120, "exclude": 30, "weight": 0.2}
        ],
        'calc_beta': False,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'calc_volatility': False,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
        'use_minimal_threshold': False,
        'minimal_threshold_percent': 4.0,
        'use_max_allocation': False,
        'max_allocation_percent': 20.0,
        'collect_dividends_as_cash': False,
        'start_date_user': None,
        'end_date_user': None,
        'fusion_portfolio': {'enabled': False, 'selected_portfolios': [], 'allocations': {}}
    }]
    st.session_state.strategy_comparison_active_portfolio_index = 0
    st.session_state.strategy_comparison_portfolio_checkboxes = {}
    
    # Clear all ticker-related session state
    # Reset global tickers to one empty ticker
    st.session_state.strategy_comparison_global_tickers = [
        {'ticker': '', 'allocation': 0.0, 'include_dividends': True}
    ]
    
    # Clear bulk ticker input
    if 'strategy_comparison_bulk_tickers' in st.session_state:
        del st.session_state['strategy_comparison_bulk_tickers']
    
    # Reset benchmark ticker
    st.session_state['strategy_comparison_benchmark_ticker'] = '^GSPC'
    
    # Clear all individual ticker inputs
    keys_to_clear = [key for key in st.session_state.keys() if key.startswith('strategy_comparison_global_ticker_')]
    for key in keys_to_clear:
        del st.session_state[key]
    
    # Clear allocation and dividend inputs
    alloc_keys_to_clear = [key for key in st.session_state.keys() if key.startswith('strategy_comparison_global_alloc_')]
    for key in alloc_keys_to_clear:
        del st.session_state[key]
    
    div_keys_to_clear = [key for key in st.session_state.keys() if key.startswith('strategy_comparison_global_div_')]
    for key in div_keys_to_clear:
        del st.session_state[key]
    
    st.success("✅ All portfolios cleared! Created 'New Portfolio 1'")
    st.rerun()

# NEW: Enhanced bulk portfolio management dropdown
if len(st.session_state.strategy_comparison_portfolio_configs) > 1:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Bulk Portfolio Management")
    
    # Initialize session state for selected portfolios
    if "strategy_comparison_portfolio_checkboxes" not in st.session_state:
        st.session_state.strategy_comparison_portfolio_checkboxes = {}
    
    # Enhanced dropdown with built-in selection controls
    with st.sidebar.expander("📋 Manage Multiple Portfolios", expanded=False):
        st.caption(f"Total portfolios: {len(portfolio_names)}")
        
        # Create checkboxes for each portfolio
        st.markdown("**Select portfolios to delete:**")
        
        # Quick selection buttons at the top
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("✅ Select All", key="strategy_comparison_select_all_portfolios", 
                        help="Select all portfolios for deletion", use_container_width=True):
                for name in portfolio_names:
                    st.session_state.strategy_comparison_portfolio_checkboxes[name] = True
                st.session_state.strategy_comparison_rerun_flag = True
        
        with col2:
            if st.button("❌ Clear All", key="strategy_comparison_clear_all_portfolios", 
                        help="Clear all portfolio selections", use_container_width=True):
                st.session_state.strategy_comparison_portfolio_checkboxes = {}
                st.session_state.strategy_comparison_rerun_flag = True
        
        with col3:
            if st.button("🔄 Refresh", key="strategy_comparison_refresh_selections", 
                        help="Refresh the selection list", use_container_width=True):
                st.session_state.strategy_comparison_rerun_flag = True
        
        # Portfolio checkboxes with scrollable container
        st.markdown("---")
        
        # Create a scrollable container for many portfolios
        with st.container():
            # Limit height and add scrollbar for many portfolios
            st.markdown("""
            <style>
            .portfolio-checkboxes {
                max-height: 300px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Portfolio checkboxes with individual callback functions
            for i, portfolio_name in enumerate(portfolio_names):
                # Initialize checkbox state if not exists
                if portfolio_name not in st.session_state.strategy_comparison_portfolio_checkboxes:
                    st.session_state.strategy_comparison_portfolio_checkboxes[portfolio_name] = False
                
                # Create a unique callback function for each portfolio
                def create_portfolio_callback(portfolio_name):
                    def callback():
                        # Toggle the current state
                        current_state = st.session_state.strategy_comparison_portfolio_checkboxes.get(portfolio_name, False)
                        st.session_state.strategy_comparison_portfolio_checkboxes[portfolio_name] = not current_state
                    return callback
                
                # Create checkbox for each portfolio with callback
                checkbox_key = f"strategy_comparison_portfolio_checkbox_{hash(portfolio_name)}"
                is_checked = st.checkbox(
                    f"🗑️ {portfolio_name}",
                    value=st.session_state.strategy_comparison_portfolio_checkboxes[portfolio_name],
                    key=checkbox_key,
                    help=f"Select {portfolio_name} for deletion",
                    on_change=create_portfolio_callback(portfolio_name)
                )
        
        # Get selected portfolios from checkboxes
        selected_portfolios_for_deletion = [
            name for name, checked in st.session_state.strategy_comparison_portfolio_checkboxes.items() 
            if checked
        ]
        
        # Show success message if portfolios were deleted
        if "strategy_comparison_bulk_delete_success" in st.session_state and st.session_state.strategy_comparison_bulk_delete_success:
            st.success(st.session_state.strategy_comparison_bulk_delete_success)
            # Clear the success message after showing it
            del st.session_state.strategy_comparison_bulk_delete_success
        
        # Show selection summary
        if selected_portfolios_for_deletion:
            st.info(f"📊 Selected: {len(selected_portfolios_for_deletion)} portfolio(s)")
            st.caption(f"Selected: {', '.join(selected_portfolios_for_deletion[:3])}{'...' if len(selected_portfolios_for_deletion) > 3 else ''}")
            
            # Bulk delete button with confirmation
            confirm_deletion = st.checkbox(
                f"🗑️ Confirm deletion of {len(selected_portfolios_for_deletion)} portfolio(s)",
                key="strategy_comparison_confirm_bulk_deletion",
                help="Check this box to enable the delete button"
            )
            
            if confirm_deletion:
                if st.button("🚨 DELETE SELECTED PORTFOLIOS", 
                           type="secondary",
                           help=f"Delete {len(selected_portfolios_for_deletion)} selected portfolio(s)",
                           on_click=bulk_delete_portfolios_callback,
                           args=(selected_portfolios_for_deletion,),
                           use_container_width=True):
                    pass
        else:
            st.caption("No portfolios selected for deletion")

# Global Ticker Management Section (moved to sidebar)
st.sidebar.markdown("---")
st.sidebar.subheader("Global Ticker Management")
st.sidebar.markdown("*All portfolios use the same tickers*")

# Handle seamless ticker management operations - OPTIMIZED NO REFRESH
if 'strategy_comparison_add_stock_flag' in st.session_state and st.session_state.strategy_comparison_add_stock_flag:
    st.session_state.strategy_comparison_global_tickers.append({'ticker': '', 'allocation': 0.0, 'include_dividends': True})
    # Sync to all portfolios but don't trigger refresh
    sync_global_tickers_to_all_portfolios()
    st.session_state.strategy_comparison_add_stock_flag = False



# Handle seamless momentum window operations
if 'strategy_comparison_add_momentum_window_flag' in st.session_state and st.session_state.strategy_comparison_add_momentum_window_flag:
    try:
        idx = st.session_state.strategy_comparison_active_portfolio_index
        if 0 <= idx < len(st.session_state.strategy_comparison_portfolio_configs):
            cfg = st.session_state.strategy_comparison_portfolio_configs[idx]
            if 'momentum_windows' not in cfg:
                cfg['momentum_windows'] = []
            cfg['momentum_windows'].append({"lookback": 90, "exclude": 30, "weight": 0.1})
            st.session_state.strategy_comparison_portfolio_configs[idx] = cfg
    except (IndexError, KeyError) as e:
        # Handle rapid UI changes gracefully
        pass
    st.session_state.strategy_comparison_add_momentum_window_flag = False

if 'strategy_comparison_remove_momentum_window_flag' in st.session_state and st.session_state.strategy_comparison_remove_momentum_window_flag:
    try:
        idx = st.session_state.strategy_comparison_active_portfolio_index
        if (0 <= idx < len(st.session_state.strategy_comparison_portfolio_configs) and
            'momentum_windows' in st.session_state.strategy_comparison_portfolio_configs[idx] and
            st.session_state.strategy_comparison_portfolio_configs[idx]['momentum_windows']):
            cfg = st.session_state.strategy_comparison_portfolio_configs[idx]
            cfg['momentum_windows'].pop()
            st.session_state.strategy_comparison_portfolio_configs[idx] = cfg
    except (IndexError, KeyError) as e:
        # Handle rapid UI changes gracefully
        pass
    st.session_state.strategy_comparison_remove_momentum_window_flag = False

# Stock management buttons
col_stock_buttons = st.sidebar.columns([1, 1])
with col_stock_buttons[0]:
    if st.sidebar.button("Normalize Tickers %", on_click=normalize_stock_allocations_callback, use_container_width=True):
        pass
with col_stock_buttons[1]:
    if st.sidebar.button("Equal Allocation %", on_click=equal_stock_allocation_callback, use_container_width=True):
        pass

col_stock_buttons2 = st.sidebar.columns([1, 1])
with col_stock_buttons2[0]:
    if st.sidebar.button("Reset Tickers", on_click=reset_stock_selection_callback, use_container_width=True):
        pass
with col_stock_buttons2[1]:
    if st.sidebar.button("Add Ticker", on_click=add_stock_callback, use_container_width=True):
        pass


# Calculate live total ticker allocation for global tickers
valid_stocks = [s for s in st.session_state.strategy_comparison_global_tickers if s['ticker']]
total_stock_allocation = sum(s['allocation'] for s in valid_stocks)

# Always show allocation status (not hidden by momentum)
if abs(total_stock_allocation - 1.0) > 0.001:
    st.sidebar.warning(f"Total allocation: {total_stock_allocation*100:.1f}%")
else:
    st.sidebar.success(f"Total allocation: {total_stock_allocation*100:.1f}%")

# Stock inputs in sidebar (using global tickers) - Layout similar to app.py
for i in range(len(st.session_state.strategy_comparison_global_tickers)):
    stock = st.session_state.strategy_comparison_global_tickers[i]
    
    # Use columns to display ticker, allocation, dividends, and remove button on same line
    col1, col2, col3, col4 = st.sidebar.columns([1, 1, 1, 0.2])
    
    with col1:
        # Ticker input - always use current value from global tickers
        ticker_key = f"strategy_comparison_global_ticker_{i}"
        # Always sync the session state with the portfolio config to show resolved ticker
        st.session_state[ticker_key] = stock['ticker']
        ticker_val = st.text_input(f"Ticker {i+1}", key=ticker_key, on_change=update_global_stock_ticker, args=(i,))
    
    with col2:
        # Allocation input - always use current value from global tickers
        alloc_key = f"strategy_comparison_global_alloc_{i}"
        st.session_state[alloc_key] = int(stock['allocation'] * 100)  # Always update to current value
        alloc_val = st.number_input(f"Alloc % {i+1}", min_value=0, step=1, format="%d", key=alloc_key, on_change=update_global_stock_allocation, args=(i,))
    
    with col3:
        # Dividends checkbox - always use current value from global tickers
        div_key = f"strategy_comparison_global_div_{i}"
        st.session_state[div_key] = stock['include_dividends']  # Always update to current value
        div_val = st.checkbox("Dividends", key=div_key, on_change=update_global_stock_dividends, args=(i,))
        
    
    with col4:
        # Remove button
        if st.button("x", key=f"remove_global_stock_{i}_{stock['ticker']}_{id(stock)}", help="Remove this ticker", on_click=remove_global_stock_callback, args=(stock['ticker'],)):
            pass

# Bulk Leverage Controls
with st.sidebar.expander("🔧 Bulk Leverage Controls", expanded=False):
    def apply_bulk_leverage_callback():
        """Apply leverage and expense ratio to selected tickers in the current portfolio"""
        try:
            leverage_value = st.session_state.get('bulk_leverage_value', 1.0)
            expense_ratio_value = st.session_state.get('bulk_expense_ratio_value', 1.0)
            selected_tickers = st.session_state.get('bulk_selected_tickers', [])
            
            # Check if any tickers are selected
            if not selected_tickers:
                st.warning("⚠️ Please select at least one ticker to apply leverage to.")
                return
            
            applied_count = 0
            for i, stock in enumerate(st.session_state.strategy_comparison_global_tickers):
                current_ticker = stock['ticker']
                
                # Check if this ticker should be modified
                base_ticker, _, _ = parse_ticker_parameters(current_ticker)
                if base_ticker in selected_tickers or current_ticker in selected_tickers:
                    # Create new ticker with leverage and expense ratio
                    new_ticker = base_ticker
                    if leverage_value != 1.0:
                        new_ticker += f"?L={leverage_value}"
                    if expense_ratio_value != 0.0:
                        new_ticker += f"?E={expense_ratio_value}"
                    
                    # Update the ticker in the global tickers
                    st.session_state.strategy_comparison_global_tickers[i]['ticker'] = new_ticker
                    
                    # Update the session state for the text input
                    ticker_key = f"strategy_comparison_global_ticker_{i}"
                    st.session_state[ticker_key] = new_ticker
                    
                    # If leverage is negative (short position), uncheck dividends checkbox
                    # User can manually re-check it if desired
                    if leverage_value < 0:
                        st.session_state.strategy_comparison_global_tickers[i]['include_dividends'] = False
                        div_key = f"strategy_comparison_global_div_{i}"
                        st.session_state[div_key] = False
                    
                    applied_count += 1
            
            if applied_count > 0:
                st.toast(f"✅ Applied {leverage_value}x leverage and {expense_ratio_value}% expense ratio to {applied_count} ticker(s)!")
            else:
                st.warning("⚠️ No tickers were selected for modification.")
            
        except Exception as e:
            st.error(f"Error applying bulk leverage: {str(e)}")

    def remove_bulk_leverage_callback():
        """Remove all leverage and expense ratio from selected tickers"""
        try:
            selected_tickers = st.session_state.get('bulk_selected_tickers', [])
            
            # Check if any tickers are selected
            if not selected_tickers:
                st.warning("⚠️ Please select at least one ticker to remove leverage from.")
                return
            
            removed_count = 0
            for i, stock in enumerate(st.session_state.strategy_comparison_global_tickers):
                current_ticker = stock['ticker']
                
                # Check if this ticker should be modified
                base_ticker, _, _ = parse_ticker_parameters(current_ticker)
                if base_ticker in selected_tickers or current_ticker in selected_tickers:
                    # Update the ticker to base ticker (no leverage, no expense ratio)
                    st.session_state.strategy_comparison_global_tickers[i]['ticker'] = base_ticker
                    
                    # Update the session state for the text input
                    ticker_key = f"strategy_comparison_global_ticker_{i}"
                    st.session_state[ticker_key] = base_ticker
                    
                    removed_count += 1
            
            if removed_count > 0:
                st.toast(f"✅ Removed leverage and expense ratio from {removed_count} ticker(s)!")
            else:
                st.warning("⚠️ No tickers were selected for modification.")
            
        except Exception as e:
            st.error(f"Error removing leverage: {str(e)}")

    # Get current global tickers for selection
    available_tickers = [stock['ticker'] for stock in st.session_state.strategy_comparison_global_tickers]
    
    # Initialize selected tickers if not exists
    if 'bulk_selected_tickers' not in st.session_state:
        st.session_state.bulk_selected_tickers = []
    
    # Ticker selection interface
    st.markdown("**Select Tickers:**")
    
    # Quick selection buttons
    col_quick1, col_quick2 = st.columns([1, 1])
    
    with col_quick1:
        if st.button("Select All", key="page6_select_all_tickers", use_container_width=True):
            st.session_state.bulk_selected_tickers = available_tickers.copy()
            st.rerun()
    
    with col_quick2:
        if st.button("Clear", key="page6_clear_all_tickers", use_container_width=True):
            st.session_state.bulk_selected_tickers = []
            st.rerun()
    
    # Individual ticker selection
    if available_tickers:
        # Create checkboxes for each ticker
        for i, ticker in enumerate(available_tickers):
            base_ticker, leverage, expense = parse_ticker_parameters(ticker)
            display_text = f"{base_ticker}"
            if leverage != 1.0 or expense > 0.0:
                display_text += f" (L:{leverage}x, E:{expense}%)"
            
            # Use checkbox state directly
            checkbox_key = f"page6_bulk_ticker_select_{i}"
            is_checked = st.checkbox(
                display_text, 
                value=ticker in st.session_state.bulk_selected_tickers,
                key=checkbox_key
            )
            
            # Update selection based on checkbox state
            if is_checked and ticker not in st.session_state.bulk_selected_tickers:
                st.session_state.bulk_selected_tickers.append(ticker)
            elif not is_checked and ticker in st.session_state.bulk_selected_tickers:
                st.session_state.bulk_selected_tickers.remove(ticker)
    else:
        st.info("No tickers available.")
    
    # Show selected tickers count
    selected_count = len(st.session_state.bulk_selected_tickers)
    if selected_count > 0:
        st.success(f"📊 {selected_count} selected")
    else:
        st.info("💡 No selection = ALL tickers")

    # Bulk leverage controls - Compact layout
    st.markdown("**Leverage & Expense Ratio:**")
    
    # First row: Input fields
    col1, col2 = st.columns([1, 1])
    with col1:
        st.number_input(
            "Leverage",
            value=2.0,
            step=0.1,
            format="%.1f",
            key="bulk_leverage_value",
            help="Leverage multiplier (e.g., 2.0 for 2x leverage, -3.0 for -3x inverse)"
        )
    with col2:
        st.number_input(
            "Expense Ratio (%)",
            value=1.0,
            step=0.01,
            format="%.2f",
            key="bulk_expense_ratio_value",
            help="Annual expense ratio in percentage (e.g., 0.84 for 0.84%, can be negative)"
        )
    
    # Second row: Buttons
    col3, col4 = st.columns([1, 1])
    with col3:
        if st.button("Apply to Selected", on_click=apply_bulk_leverage_callback, type="primary", use_container_width=True):
            pass
    with col4:
        if st.button("Remove from Selected", on_click=remove_bulk_leverage_callback, type="secondary", use_container_width=True):
            pass

# Bulk ticker input section
with st.sidebar.expander("📝 Bulk Ticker Input", expanded=False):
    st.markdown("**Enter multiple tickers separated by spaces or commas:**")
    
    # Initialize bulk ticker input in session state
    if 'strategy_comparison_bulk_tickers' not in st.session_state:
        st.session_state.strategy_comparison_bulk_tickers = ""
    
    # Auto-populate bulk ticker input with current tickers (only if user hasn't entered anything)
    current_tickers = [stock['ticker'] for stock in st.session_state.strategy_comparison_global_tickers if stock['ticker']]
    if current_tickers:
        current_ticker_string = ' '.join(current_tickers)
        # Only auto-populate if the bulk ticker field is empty or matches the current portfolio
        if not st.session_state.strategy_comparison_bulk_tickers or st.session_state.strategy_comparison_bulk_tickers == current_ticker_string:
            st.session_state.strategy_comparison_bulk_tickers = current_ticker_string
    
    # Text area for bulk ticker input
    bulk_tickers = st.text_area(
        "Tickers (e.g., SPY QQQ GLD TLT or SPY,QQQ,GLD,TLT)",
        value=st.session_state.strategy_comparison_bulk_tickers,
        key="strategy_comparison_bulk_ticker_input",
        height=100,
        help="Enter ticker symbols separated by spaces or commas. Choose 'Replace All' to replace all tickers or 'Add to Existing' to add new tickers."
    )
    
    # Action buttons
    col_replace, col_add, col_fetch, col_copy = st.columns([1, 1, 1, 1])
    
    with col_replace:
        if st.button("🔄 Replace All", key="strategy_comparison_fill_tickers_btn", type="secondary", use_container_width=True):
            if bulk_tickers.strip():
                # Parse tickers (split by comma or space)
                ticker_list = []
            for ticker in bulk_tickers.replace(',', ' ').split():
                ticker = ticker.strip().upper()
                if ticker:
                    # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
                    if ticker == 'BRK.B':
                        ticker = 'BRK-B'
                    elif ticker == 'BRK.A':
                        ticker = 'BRK-A'
                    ticker_list.append(ticker)
            
            if ticker_list:
                current_stocks = st.session_state.strategy_comparison_global_tickers.copy()
                
                # Replace tickers - new ones get 0% allocation
                new_stocks = []
                
                for i, ticker in enumerate(ticker_list):
                    if i < len(current_stocks):
                        # Use existing allocation if available
                        new_stocks.append({
                            'ticker': ticker,
                            'allocation': current_stocks[i]['allocation'],
                            'include_dividends': current_stocks[i]['include_dividends']
                        })
                    else:
                        # New tickers get 0% allocation
                        new_stocks.append({
                            'ticker': ticker,
                            'allocation': 0.0,
                            'include_dividends': True
                        })
                
                # Update the global tickers
                st.session_state.strategy_comparison_global_tickers = new_stocks
                
                # Clear any existing session state keys for individual ticker inputs to force refresh
                for key in list(st.session_state.keys()):
                    if key.startswith("strategy_comparison_global_ticker_") or key.startswith("strategy_comparison_global_alloc_"):
                        del st.session_state[key]
                
                    st.success(f"✅ Replaced all tickers with: {', '.join(ticker_list)}")
                st.info("💡 **Note:** Existing allocations preserved. Adjust allocations manually if needed.")
                
                # Force immediate rerun to refresh the UI
                st.rerun()
            else:
                    st.warning("⚠️ No valid tickers found in input.")
        else:
                st.warning("⚠️ No valid tickers found in input.")
    
    with col_add:
        if st.button("➕ Add to Existing", key="strategy_comparison_add_tickers_btn", type="secondary", use_container_width=True):
            if bulk_tickers.strip():
                # Parse tickers (split by comma or space)
                ticker_list = []
                for ticker in bulk_tickers.replace(',', ' ').split():
                    ticker = ticker.strip().upper()
                    if ticker:
                        # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
                        if ticker == 'BRK.B':
                            ticker = 'BRK-B'
                        elif ticker == 'BRK.A':
                            ticker = 'BRK-A'
                        ticker_list.append(ticker)
                
                if ticker_list:
                    current_stocks = st.session_state.strategy_comparison_global_tickers.copy()
                    
                    # Add new tickers to existing ones
                    for ticker in ticker_list:
                        # Check if ticker already exists
                        ticker_exists = any(stock['ticker'] == ticker for stock in current_stocks)
                        if not ticker_exists:
                            current_stocks.append({
                                'ticker': ticker,
                                'allocation': 0.0,
                                'include_dividends': True
                            })
                    
                    # Update the global tickers
                    st.session_state.strategy_comparison_global_tickers = current_stocks
                    
                    # Clear any existing session state keys for individual ticker inputs to force refresh
                    for key in list(st.session_state.keys()):
                        if key.startswith("strategy_comparison_global_ticker_") or key.startswith("strategy_comparison_global_alloc_"):
                            del st.session_state[key]
                    
                    st.success(f"✅ Added new tickers: {', '.join(ticker_list)}")
                    st.info("💡 **Note:** New tickers added with 0% allocation. Adjust allocations manually if needed.")
                    
                    # Force immediate rerun to refresh the UI
                    st.rerun()
                else:
                    st.warning("⚠️ No valid tickers found in input.")
            else:
                st.warning("⚠️ No valid tickers found in input.")
    
    with col_fetch:
        if st.button("🔍 Fetch Tickers", key="strategy_comparison_fetch_tickers_btn", type="secondary", use_container_width=True):
            # Get current tickers from the global tickers
            current_tickers = [stock['ticker'] for stock in st.session_state.strategy_comparison_global_tickers if stock['ticker']]
            
            if current_tickers:
                # Update the bulk ticker input with current tickers
                current_ticker_string = ' '.join(current_tickers)
                st.session_state.strategy_comparison_bulk_tickers = current_ticker_string
                st.success(f"✅ Fetched {len(current_tickers)} tickers: {current_ticker_string}")
                st.rerun()
            else:
                st.warning("⚠️ No tickers found in the current portfolio.")
    
    with col_copy:
        if bulk_tickers.strip():
            # Create a custom button with direct copy functionality
            import streamlit.components.v1 as components
            
            # JavaScript function to copy and show feedback
            copy_js = f"""
            <script>
            function copyTickers() {{
                navigator.clipboard.writeText({json.dumps(bulk_tickers.strip())}).then(function() {{
                    // Show success feedback
                    const button = document.querySelector('#copy-tickers-btn');
                    const originalText = button.innerHTML;
                    button.innerHTML = '✅ Copied!';
                    button.style.backgroundColor = '#28a745';
                    setTimeout(function() {{
                        button.innerHTML = originalText;
                        button.style.backgroundColor = '';
                    }}, 2000);
                }}).catch(function(err) {{
                    alert('Failed to copy: ' + err);
                }});
            }}
            </script>
            <button id="copy-tickers-btn" onclick="copyTickers()" style="
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                width: 100%;
                font-size: 14px;
            ">📋 Copy</button>
            """
            components.html(copy_js, height=50)
        else:
            st.button("📋 Copy", key="strategy_comparison_copy_tickers_btn", type="secondary", use_container_width=True, disabled=True)
            st.warning("⚠️ No tickers to copy. Please enter some tickers first.")

# Validation constants
_TOTAL_TOL = 1.0
_ALLOC_TOL = 1.0

# Clear All Outputs Function
def clear_all_outputs():
    """Clear all backtest results and outputs while preserving portfolio configurations"""
    # Clear all result data
    st.session_state.multi_all_results = None
    st.session_state.multi_all_allocations = None
    st.session_state.multi_all_metrics = None
    st.session_state.multi_backtest_all_drawdowns = None
    st.session_state.multi_backtest_stats_df_display = None
    st.session_state.multi_backtest_all_years = None
    st.session_state.multi_backtest_portfolio_key_map = {}
    st.session_state.multi_backtest_ran = False
    
    # Clear strategy comparison page specific data
    st.session_state.strategy_comparison_all_results = None
    st.session_state.strategy_comparison_all_allocations = None
    st.session_state.strategy_comparison_all_metrics = None
    st.session_state.strategy_comparison_snapshot_data = None
    st.session_state.strategy_comparison_ran = False
    
    # Clear any processing flags
    for key in list(st.session_state.keys()):
        if key.startswith("processing_portfolio_"):
            del st.session_state[key]
    
    # Clear any cached data
    if 'raw_data' in st.session_state:
        del st.session_state['raw_data']
    
    st.success("✅ All outputs cleared! Portfolio configurations preserved.")

# Clear All Outputs Button
if st.sidebar.button("🗑️ Clear All Outputs", type="secondary", use_container_width=True, help="Clear all charts and results while keeping portfolio configurations"):
    clear_all_outputs()
    st.rerun()

# Cancel Run Button
if st.sidebar.button("🛑 Cancel Run", type="secondary", use_container_width=True, help="Stop current backtest execution gracefully"):
    st.session_state.hard_kill_requested = True
    st.toast("🛑 **CANCELLING** - Stopping backtest execution...", icon="⏹️")
    st.rerun()

# Emergency Kill Button
if st.sidebar.button("🚨 EMERGENCY KILL", type="secondary", use_container_width=True, help="Force terminate all processes immediately - Use for crashes, freezes, or unresponsive states"):
    st.toast("🚨 **EMERGENCY KILL** - Force terminating all processes...", icon="💥")
    emergency_kill()

# Run Backtest button
if st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True):
    # Reset kill request when starting new backtest
    st.session_state.hard_kill_requested = False
    # Pre-validation check for all portfolios
    configs_to_run = st.session_state.strategy_comparison_portfolio_configs
    valid_configs = True
    validation_errors = []
    
    for cfg in configs_to_run:
        if cfg['use_momentum']:
            total_momentum_weight = sum(w['weight'] for w in cfg['momentum_windows'])
            if abs(total_momentum_weight - 1.0) > (_TOTAL_TOL / 100.0):
                validation_errors.append(f"Portfolio '{cfg['name']}' has momentum enabled but the total momentum weight is {total_momentum_weight*100:.2f}% (must be 100%)")
                valid_configs = False
        else:
            valid_stocks_for_cfg = [s for s in cfg['stocks'] if s['ticker']]
            total_stock_allocation = sum(s['allocation'] for s in valid_stocks_for_cfg)
            if abs(total_stock_allocation - 1.0) > (_ALLOC_TOL / 100.0):
                validation_errors.append(f"Portfolio '{cfg['name']}' is not using momentum, but the total ticker allocation is {total_stock_allocation*100:.2f}% (must be 100%)")
                valid_configs = False
                
    if not valid_configs:
        for error in validation_errors:
            st.error(error)
        # Don't set the run flag, but continue showing the UI
        pass
    else:
        st.session_state.strategy_comparison_run_backtest = True
        # Show standalone popup notification that code is really running
        st.toast("**Code is running!** Starting backtest...", icon="🚀")
        
        # Check for kill request
        check_kill_request()

# Leverage Summary Section - moved here to appear after ticker input
leveraged_tickers = []
for stock in st.session_state.strategy_comparison_global_tickers:
    if "?L=" in stock['ticker']:
        try:
            base_ticker, leverage = parse_leverage_ticker(stock['ticker'])
            leveraged_tickers.append((base_ticker, leverage))
        except:
            pass

if leveraged_tickers:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Leverage Summary")
    
    # Get risk-free rate for drag calculation
    try:
        risk_free_rates = get_risk_free_rate_robust([pd.Timestamp.now()])
        daily_rf = risk_free_rates.iloc[0] if len(risk_free_rates) > 0 else 0.000105
        annual_rf = daily_rf * 365.25 * 100  # Convert daily to annual percentage
    except:
        daily_rf = 0.000105  # fallback
        annual_rf = 3.86  # fallback annual rate
    
    # Group by leverage level
    leverage_groups = {}
    for base_ticker, leverage in leveraged_tickers:
        if leverage not in leverage_groups:
            leverage_groups[leverage] = []
        leverage_groups[leverage].append(base_ticker)
    
    for leverage in sorted(leverage_groups.keys()):
        base_tickers = leverage_groups[leverage]
        daily_drag = (leverage - 1) * daily_rf * 100
        st.sidebar.markdown(f"🚀 **{leverage}x leverage** on {', '.join(base_tickers)}")
        st.sidebar.markdown(f"📉 **Daily drag:** {daily_drag:.3f}% (RF: {annual_rf:.2f}%)")

# Special tickers and leverage guide sections
with st.sidebar.expander("📈 Broad Long-Term Tickers", expanded=False):
    st.markdown("""
    **Recommended tickers for long-term strategies:**
    
    **Core ETFs:**
    - **SPY** - S&P 500 (0.09% expense ratio)
    - **QQQ** - NASDAQ-100 (0.20% expense ratio)  
    - **VTI** - Total Stock Market (0.03% expense ratio)
    - **VEA** - Developed Markets (0.05% expense ratio)
    - **VWO** - Emerging Markets (0.10% expense ratio)
    
    **Sector ETFs:**
    - **XLK** - Technology (0.10% expense ratio)
    - **XLF** - Financials (0.10% expense ratio)
    - **XLE** - Energy (0.10% expense ratio)
    - **XLV** - Healthcare (0.10% expense ratio)
    - **XLI** - Industrials (0.10% expense ratio)
    
    **Bond ETFs:**
    - **TLT** - 20+ Year Treasury (0.15% expense ratio)
    - **IEF** - 7-10 Year Treasury (0.15% expense ratio)
    - **LQD** - Investment Grade Corporate (0.14% expense ratio)
    - **HYG** - High Yield Corporate (0.49% expense ratio)
    
    **Commodity ETFs:**
    - **GLD** - Gold (0.40% expense ratio)
    - **SLV** - Silver (0.50% expense ratio)
    - **DBA** - Agriculture (0.93% expense ratio)
    - **USO** - Oil (0.60% expense ratio)
    """)

# Special Tickers Section
with st.sidebar.expander("🎯 Special Long-Term Tickers", expanded=False):
    st.markdown("**Quick access to ticker aliases that the system accepts:**")
    
    # Get the actual ticker aliases from the function
    aliases = get_ticker_aliases()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📈 Stock Indices**")
        stock_mapping = {
            'S&P 500 (No Dividend) (1927+)': ('SPYND', '^GSPC'),
            'S&P 500 (Total Return) (1988+)': ('SPYTR', '^SP500TR'), 
            'NASDAQ (No Dividend) (1971+)': ('QQQND', '^IXIC'),
            'NASDAQ 100 (1985+)': ('NDX', '^NDX'),
            'Dow Jones (1992+)': ('DOW', '^DJI')
        }
        
        for name, (alias, ticker) in stock_mapping.items():
            if st.button(f"➕ {name}", key=f"add_stock_{ticker}", help=f"Add {alias} → {ticker}"):
                # Ensure global tickers exist
                if 'strategy_comparison_global_tickers' not in st.session_state:
                    st.session_state.strategy_comparison_global_tickers = [
                        {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                        {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                        {'ticker': 'IEF', 'allocation': 0.25, 'include_dividends': True},
                        {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True}
                    ]
                
                # Resolve the alias to the actual Yahoo ticker before storing
                resolved_ticker = resolve_ticker_alias(alias)
                st.session_state.strategy_comparison_global_tickers.append({
                    'ticker': resolved_ticker,  # Add the resolved Yahoo ticker
                    'allocation': 0.0, 
                    'include_dividends': True
                })
                # Sync to all portfolios
                sync_global_tickers_to_all_portfolios()
                st.rerun()
    
    with col2:
        st.markdown("**🏭 Sector Indices**")
        sector_mapping = {
            'Technology (XLK) (1990+)': ('XLKND', '^SP500-45'),
            'Healthcare (XLV) (1990+)': ('XLVND', '^SP500-35'),
            'Consumer Staples (XLP) (1990+)': ('XLPND', '^SP500-30'),
            'Financials (XLF) (1990+)': ('XLFND', '^SP500-40'),
            'Energy (XLE) (1990+)': ('XLEND', '^SP500-10'),
            'Industrials (XLI) (1990+)': ('XLIND', '^SP500-20'),
            'Consumer Discretionary (XLY) (1990+)': ('XLYND', '^SP500-25'),
            'Materials (XLB) (1990+)': ('XLBND', '^SP500-15'),
            'Utilities (XLU) (1990+)': ('XLUND', '^SP500-55'),
            'Real Estate (XLRE) (1990+)': ('XLREND', '^SP500-60'),
            'Communication Services (XLC) (1990+)': ('XLCND', '^SP500-50')
        }
        
        for name, (alias, ticker) in sector_mapping.items():
            if st.button(f"➕ {name}", key=f"add_sector_{ticker}", help=f"Add {alias} → {ticker}"):
                # Ensure global tickers exist
                if 'strategy_comparison_global_tickers' not in st.session_state:
                    st.session_state.strategy_comparison_global_tickers = [
                        {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                        {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                        {'ticker': 'IEF', 'allocation': 0.25, 'include_dividends': True},
                        {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True}
                    ]
                
                # Resolve the alias to the actual Yahoo ticker before storing
                resolved_ticker = resolve_ticker_alias(alias)
                st.session_state.strategy_comparison_global_tickers.append({
                    'ticker': resolved_ticker,  # Add the resolved Yahoo ticker
                    'allocation': 0.0, 
                    'include_dividends': True
                })
                # Sync to all portfolios
                sync_global_tickers_to_all_portfolios()
                st.rerun()
    
    with col3:
        st.markdown("**🔬 Synthetic Tickers**")
        synthetic_tickers = {
            # Ordered by asset class: Stocks → Bonds → Gold → Managed Futures → Bitcoin
            'Complete S&P 500 Simulation (1885+)': ('SPYSIM', 'SPYSIM_COMPLETE'),
            'Dynamic S&P 500 Top 20 (Historical)': ('SP500TOP20', 'SP500TOP20'),
            'Cash Simulator (ZEROX)': ('ZEROX', 'ZEROX'),
            'Complete TBILL Dataset (1948+)': ('TBILL', 'TBILL_COMPLETE'),
            'Complete IEF Dataset (1962+)': ('IEFTR', 'IEF_COMPLETE'),
            'Complete TLT Dataset (1962+)': ('TLTTR', 'TLT_COMPLETE'),
            'Complete ZROZ Dataset (1962+)': ('ZROZX', 'ZROZ_COMPLETE'),
            'Complete Gold Simulation (1968+)': ('GOLDSIM', 'GOLDSIM_COMPLETE'),
            'Complete Gold Dataset (1975+)': ('GOLDX', 'GOLD_COMPLETE'),
            'Complete KMLM Dataset (1992+)': ('KMLMX', 'KMLM_COMPLETE'),
            'Complete DBMF Dataset (2000+)': ('DBMFX', 'DBMF_COMPLETE'),
            'Complete Bitcoin Dataset (2010+)': ('BITCOINX', 'BTC_COMPLETE'),
            
            # Leveraged & Inverse ETFs (Synthetic) - NASDAQ-100 versions
            'Simulated TQQQ (3x QQQ) (1985+)': ('TQQQND', '^NDX?L=3?E=0.95'),
            'Simulated QLD (2x QQQ) (1985+)': ('QLDND', '^NDX?L=2?E=0.95'),
            'Simulated PSQ (-1x QQQ) (1985+)': ('PSQND', '^NDX?L=-1?E=0.95'),
            'Simulated QID (-2x QQQ) (1985+)': ('QIDND', '^NDX?L=-2?E=0.95'),
            'Simulated SQQQ (-3x QQQ) (1985+)': ('SQQQND', '^NDX?L=-3?E=0.95'),
            
            # Leveraged & Inverse ETFs (Synthetic) - NASDAQ Composite versions (longer history)
            'Simulated TQQQ-IXIC (3x IXIC) (1971+)': ('TQQQIXIC', '^IXIC?L=3?E=0.95'),
            'Simulated QLD-IXIC (2x IXIC) (1971+)': ('QLDIXIC', '^IXIC?L=2?E=0.95'),
            'Simulated PSQ-IXIC (-1x IXIC) (1971+)': ('PSQIXIC', '^IXIC?L=-1?E=0.95'),
            'Simulated QID-IXIC (-2x IXIC) (1971+)': ('QIDIXIC', '^IXIC?L=-2?E=0.95'),
            'Simulated SQQQ-IXIC (-3x IXIC) (1971+)': ('SQQQIXIC', '^IXIC?L=-3?E=0.95'),
            
            # S&P 500 leveraged/inverse (unchanged)
            'Simulated SPXL (3x SPY) (1988+)': ('SPXLTR', '^SP500TR?L=3?E=1.00'),
            'Simulated UPRO (3x SPY) (1988+)': ('UPROTR', '^SP500TR?L=3?E=0.91'),
            'Simulated SSO (2x SPY) (1988+)': ('SSOTR', '^SP500TR?L=2?E=0.91'),
            'Simulated SH (-1x SPY) (1927+)': ('SHND', '^GSPC?L=-1?E=0.89'),
            'Simulated SDS (-2x SPY) (1927+)': ('SDSND', '^GSPC?L=-2?E=0.91'),
            'Simulated SPXU (-3x SPY) (1927+)': ('SPXUND', '^GSPC?L=-3?E=1.00')
        }
        
        for name, (alias, ticker) in synthetic_tickers.items():
            # Custom help text for different ticker types
            if alias == 'SP500TOP20':
                help_text = "Add SP500TOP20 → SP500TOP20 - BETA ticker: Dynamic portfolio of top 20 S&P 500 companies rebalanced annually based on historical market cap data"
            elif alias == 'ZEROX':
                help_text = "Add ZEROX → ZEROX - Cash Simulator: Simulates a cash position that does nothing (no price movement, no dividends)"
            elif 'IXIC' in ticker:
                # Special warning for IXIC versions
                help_text = f"Add {alias} → {ticker} ⚠️ WARNING: This tracks NASDAQ Composite (broader index), NOT NASDAQ-100 like the real ETF!"
            else:
                help_text = f"Add {alias} → {ticker}"
            
            if st.button(f"➕ {name}", key=f"add_synthetic_{ticker}", help=help_text):
                # Ensure global tickers exist
                if 'strategy_comparison_global_tickers' not in st.session_state:
                    st.session_state.strategy_comparison_global_tickers = [
                        {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                        {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                        {'ticker': 'IEF', 'allocation': 0.25, 'include_dividends': True},
                        {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True}
                    ]
                
                # Resolve the alias to the actual ticker before storing
                resolved_ticker = resolve_ticker_alias(alias)
                # Auto-disable dividends for negative leverage (inverse ETFs)
                include_divs = False if '?L=-' in resolved_ticker else True
                st.session_state.strategy_comparison_global_tickers.append({
                    'ticker': resolved_ticker,  # Add the resolved ticker
                    'allocation': 0.0, 
                    'include_dividends': include_divs
                })
                # Sync to all portfolios
                sync_global_tickers_to_all_portfolios()
                st.rerun()
    
    st.markdown("---")
    
    # Ticker Aliases Section INSIDE the expander
    st.markdown("**💡 Ticker Aliases:** You can also use these shortcuts in the text input below:")
    st.markdown("- `SPX` → `^GSPC` (S&P 500, 1957+), `SPXTR` → `^SP500TR` (S&P 500 with dividends, 1957+)")
    st.markdown("- `SPYTR` → `^SP500TR` (S&P 500 Total Return, 1957+), `QQQTR` → `^NDX` (NASDAQ 100, 1985+)")
    st.markdown("- `TLTETF` → `TLT` (20+ Year Treasury ETF, 2002+), `IEFETF` → `IEF` (7-10 Year Treasury ETF, 2002+)")
    st.markdown("- `ZROZX` → `ZROZ` (25+ Year Zero Coupon Treasury, 2009+), `GOVZTR` → `GOVZ` (25+ Year Treasury STRIPS, 2019+)")
    st.markdown("- `TNX` → `^TNX` (10Y Treasury Yield, 1962+), `TYX` → `^TYX` (30Y Treasury Yield, 1977+)")
    st.markdown("- `TBILL3M` → `^IRX` (3M Treasury Yield, 1982+), `SHY` → `SHY` (1-3 Year Treasury ETF, 2002+)")
    st.markdown("- `ZEROX` (Cash doing nothing - zero return), `GOLD50` → `GOLD_COMPLETE` (Complete Gold Dataset, 1975+), `ZROZ50` → `ZROZ_COMPLETE` (Complete ZROZ Dataset, 1962+), `TLT50` → `TLT_COMPLETE` (Complete TLT Dataset, 1962+), `BTC50` → `BTC_COMPLETE` (Complete Bitcoin Dataset, 2010+), `GOLDX` → `GC=F` (Gold Futures, 1975+)")

with st.sidebar.expander("⚡ Leverage & Expense Ratio Guide", expanded=False):
    st.markdown("""
    **Leverage Format:** Use `TICKER?L=N` where N is the leverage multiplier
    **Expense Ratio Format:** Use `TICKER?E=N` where N is the annual expense ratio percentage
    
    **Examples:**
    - **SPY?L=2** - 2x leveraged S&P 500
    - **QQQ?L=3?E=0.84** - 3x leveraged NASDAQ-100 with 0.84% expense ratio (like TQQQ)
    - **QQQ?E=1** - QQQ with 1% expense ratio
    - **TLT?L=2?E=0.5** - 2x leveraged 20+ Year Treasury with 0.5% expense ratio
    - **SPY?E=2?L=3** - Order doesn't matter: 3x leveraged S&P 500 with 2% expense ratio
    - **QQQ?E=5** - QQQ with 5% expense ratio (high fee for testing)
    
    **Parameter Combinations:**
    - **QQQ?L=3?E=0.84** - Simulates TQQQ (3x QQQ with 0.84% expense ratio)
    - **SPY?L=2?E=0.95** - Simulates SSO (2x SPY with 0.95% expense ratio)
    - **QQQ?E=0.2** - Simulates QQQ with 0.2% expense ratio
    
    **Important Notes:**
    - **Daily Reset:** Leverage resets daily (like real leveraged ETFs)
    - **Cost Drag:** Includes daily cost drag = (leverage - 1) × risk-free rate
    - **Expense Drag:** Daily expense drag = annual_expense_ratio / 365.25
    - **Volatility Decay:** High volatility can cause significant decay over time
    - **Risk Warning:** Leveraged products are high-risk and can lose value quickly
    
    **Real Leveraged ETFs for Reference:**
    - **SSO** - 2x S&P 500 (0.95% expense ratio)
    - **UPRO** - 3x S&P 500 (0.95% expense ratio)
    - **TQQQ** - 3x NASDAQ-100 (0.84% expense ratio)
    - **TMF** - 3x 20+ Year Treasury (1.05% expense ratio)
    
    **Best Practices:**
    - Use for short-term strategies or hedging
    - Avoid holding for extended periods due to decay
    - Consider the underlying asset's volatility
    - Monitor risk-free rate changes affecting cost drag
    - Factor in expense ratios for realistic performance expectations
    """)

# Start with option
st.sidebar.markdown("---")
st.sidebar.subheader("Data Options")
if "strategy_comparison_start_with_radio" not in st.session_state:
    st.session_state["strategy_comparison_start_with_radio"] = st.session_state.get("strategy_comparison_start_with", "all")
st.sidebar.radio(
    "How to handle assets with different start dates?",
    ["all", "oldest"],
    format_func=lambda x: "Start when ALL assets are available" if x == "all" else "Start with OLDEST asset",
    help="""
    **All:** Starts the backtest when all selected assets are available.
    **Oldest:** Starts at the oldest date of any asset and adds assets as they become available.
    """,
    key="strategy_comparison_start_with_radio",
    on_change=lambda: setattr(st.session_state, 'strategy_comparison_start_with', st.session_state.strategy_comparison_start_with_radio)
)

# First rebalance strategy option
if "strategy_comparison_first_rebalance_strategy_radio" not in st.session_state:
    st.session_state["strategy_comparison_first_rebalance_strategy_radio"] = st.session_state.get("strategy_comparison_first_rebalance_strategy", "rebalancing_date")
st.sidebar.radio(
    "When should the first rebalancing occur?",
    ["rebalancing_date", "momentum_window_complete"],
    format_func=lambda x: "First rebalance on rebalancing date" if x == "rebalancing_date" else "First rebalance when momentum window complete",
    help="""
    **First rebalance on rebalancing date:** Start rebalancing immediately when possible.
    **First rebalance when momentum window complete:** Wait for the largest momentum window to complete before first rebalance.
    """,
    key="strategy_comparison_first_rebalance_strategy_radio",
    on_change=lambda: setattr(st.session_state, 'strategy_comparison_first_rebalance_strategy', st.session_state.strategy_comparison_first_rebalance_strategy_radio)
)

# Date range options
st.sidebar.markdown("---")
st.sidebar.subheader("Date Range Options")

# Initialize session state for custom dates if not exists
if "strategy_comparison_use_custom_dates" not in st.session_state:
    st.session_state["strategy_comparison_use_custom_dates"] = False

# Initialize date session state if not exists
if "strategy_comparison_start_date" not in st.session_state:
    st.session_state["strategy_comparison_start_date"] = date(2010, 1, 1)
if "strategy_comparison_end_date" not in st.session_state:
    st.session_state["strategy_comparison_end_date"] = date.today()

# Sync checkbox state with portfolio configs
if 'strategy_comparison_portfolio_configs' in st.session_state and st.session_state.strategy_comparison_portfolio_configs:
    active_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
    portfolio_start = active_portfolio.get('start_date_user')
    portfolio_end = active_portfolio.get('end_date_user')
    
    # If portfolio has custom dates, sync them to session state and enable checkbox
    if portfolio_start is not None or portfolio_end is not None:
        if portfolio_start is not None:
            st.session_state["strategy_comparison_start_date"] = portfolio_start
        if portfolio_end is not None:
            st.session_state["strategy_comparison_end_date"] = portfolio_end
        st.session_state["strategy_comparison_use_custom_dates"] = True

use_custom_dates = st.sidebar.checkbox("Use custom date range", key="strategy_comparison_use_custom_dates", help="Enable to set custom start and end dates for ALL portfolios in the backtest", on_change=update_custom_dates_checkbox)

if use_custom_dates:
    col_start_date, col_end_date, col_clear_dates = st.sidebar.columns([1, 1, 1])
    with col_start_date:
        # Initialize widget key with session state value
        if "strategy_comparison_start_date" not in st.session_state:
            st.session_state["strategy_comparison_start_date"] = date(2010, 1, 1)
        # Let Streamlit manage the session state automatically
        start_date = st.date_input("Start Date", min_value=date(1900, 1, 1), key="strategy_comparison_start_date", on_change=update_start_date)
    
    with col_end_date:
        # Initialize widget key with session state value
        if "strategy_comparison_end_date" not in st.session_state:
            st.session_state["strategy_comparison_end_date"] = date.today()
        # Let Streamlit manage the session state automatically
        end_date = st.date_input("End Date", min_value=date(1900, 1, 1), key="strategy_comparison_end_date", on_change=update_end_date)
    
    with col_clear_dates:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer for alignment
        st.button("Clear Dates", on_click=clear_dates_callback)
else:
    st.session_state["strategy_comparison_start_date"] = None
    st.session_state["strategy_comparison_end_date"] = None
    # Clear dates from ALL portfolios when custom dates is disabled
    for i, portfolio in enumerate(st.session_state.strategy_comparison_portfolio_configs):
        st.session_state.strategy_comparison_portfolio_configs[i]['start_date_user'] = None
        st.session_state.strategy_comparison_portfolio_configs[i]['end_date_user'] = None

# JSON section for all portfolios
st.sidebar.markdown("---")
with st.sidebar.expander('All Portfolios JSON (Export / Import)', expanded=False):
    # Clean portfolio configs for export by removing unused settings
    def clean_portfolio_configs_for_export(configs):
        cleaned_configs = []
        for i, config in enumerate(configs):
            cleaned_config = config.copy()
            # Remove unused settings that were cleaned up
            cleaned_config.pop('use_relative_momentum', None)
            cleaned_config.pop('equal_if_all_negative', None)
            # Update global settings from session state
            cleaned_config['start_with'] = st.session_state.get('strategy_comparison_start_with', 'all')
            cleaned_config['first_rebalance_strategy'] = st.session_state.get('strategy_comparison_first_rebalance_strategy', 'rebalancing_date')
            
            # Ensure threshold and maximum allocation settings are included (read from current config)
            cleaned_config['use_minimal_threshold'] = config.get('use_minimal_threshold', False)
            cleaned_config['minimal_threshold_percent'] = config.get('minimal_threshold_percent', 2.0)
            cleaned_config['use_max_allocation'] = config.get('use_max_allocation', False)
            cleaned_config['max_allocation_percent'] = config.get('max_allocation_percent', 10.0)
            
            # Ensure targeted rebalancing settings are included (read from current config)
            cleaned_config['use_targeted_rebalancing'] = config.get('use_targeted_rebalancing', False)
            
            # Get current targeted rebalancing settings from session state for this portfolio
            current_settings = config.get('targeted_rebalancing_settings', {}).copy()
            for ticker in current_settings:
                enabled_key = f"targeted_rebalancing_enabled_{ticker}_{i}"
                # Always read from session state if it exists, otherwise use stored value
                if enabled_key in st.session_state:
                    current_settings[ticker]['enabled'] = st.session_state[enabled_key]
                else:
                    # If session state doesn't exist, initialize it from the stored value
                    current_settings[ticker]['enabled'] = current_settings[ticker].get('enabled', False)
            cleaned_config['targeted_rebalancing_settings'] = current_settings
            
            # Convert date objects to strings for JSON serialization
            if cleaned_config.get('start_date_user') is not None:
                cleaned_config['start_date_user'] = cleaned_config['start_date_user'].isoformat() if hasattr(cleaned_config['start_date_user'], 'isoformat') else str(cleaned_config['start_date_user'])
            if cleaned_config.get('end_date_user') is not None:
                cleaned_config['end_date_user'] = cleaned_config['end_date_user'].isoformat() if hasattr(cleaned_config['end_date_user'], 'isoformat') else str(cleaned_config['end_date_user'])
            
            cleaned_configs.append(cleaned_config)
        return cleaned_configs
    
    cleaned_configs = clean_portfolio_configs_for_export(st.session_state.get('strategy_comparison_portfolio_configs', []))
    all_json = json.dumps(cleaned_configs, indent=2)
    st.code(all_json, language='json')
    import streamlit.components.v1 as components
    copy_html_all = f"""
    <button onclick='navigator.clipboard.writeText({json.dumps(all_json)});' style='margin-bottom:10px;'>Copy All Configs to Clipboard</button>
    """
    components.html(copy_html_all, height=40)
    
    # Add PDF download button for JSON
    def generate_json_pdf(custom_name=""):
        """Generate a PDF with pure JSON content only for easy CTRL+A / CTRL+V copying."""
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Preformatted
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        import io
        from datetime import datetime
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Add proper PDF metadata
        portfolio_count = len(st.session_state.get('strategy_comparison_portfolio_configs', []))
        
        # Use custom name if provided, otherwise use default
        if custom_name.strip():
            title = f"Strategy Comparison - {custom_name.strip()} - JSON Configuration"
            subject = f"JSON Configuration for Strategy Comparison: {custom_name.strip()} ({portfolio_count} portfolios)"
        else:
            title = f"Strategy Comparison - All Portfolios ({portfolio_count}) - JSON Configuration"
            subject = f"JSON Configuration for {portfolio_count} Strategy Comparison Portfolios"
        
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=36, 
            leftMargin=36, 
            topMargin=36, 
            bottomMargin=36,
            title=title,
            author="Portfolio Backtest System",
            subject=subject,
            creator="Strategy Comparison Application"
        )
        story = []
        
        # Pure JSON style - just monospace text
        json_style = ParagraphStyle(
            'PureJSONStyle',
            fontName='Courier',
            fontSize=10,
            leading=12,
            leftIndent=0,
            rightIndent=0,
            spaceAfter=0,
            spaceBefore=0
        )
        
        # Add only the JSON content - no headers, no instructions, just pure JSON
        json_lines = all_json.split('\n')
        for line in json_lines:
            story.append(Preformatted(line, json_style))
        
        # Build PDF
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    # Optional custom PDF name
    custom_pdf_name = st.text_input(
        "📝 Custom PDF Name (optional):", 
        value="",
        placeholder="e.g., Growth vs Value Strategy, Risk Analysis 2024, etc.",
        help="Leave empty to use automatic naming: 'Strategy Comparison - All Portfolios (X) - JSON Configuration'",
        key="strategy_comparison_custom_pdf_name"
    )
    
    if st.button("📄 Download JSON as PDF", help="Download a PDF containing the JSON configuration for easy copying", key="strategy_comparison_multi_json_pdf_btn"):
        try:
            pdf_data = generate_json_pdf(custom_pdf_name)
            
            # Generate filename based on custom name or default
            if custom_pdf_name.strip():
                clean_name = custom_pdf_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
                filename = f"{clean_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            else:
                filename = f"strategy_comparison_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            st.download_button(
                label="💾 Download Strategy Comparison JSON PDF",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf",
                key="strategy_comparison_json_pdf_download"
            )
            st.success("PDF generated successfully! Click the download button above.")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
    
    st.text_area('Paste JSON Here to Replace All Portfolios', key='strategy_comparison_paste_all_json_text', height=240)
    st.button('Update All Portfolios from JSON', on_click=paste_all_json_callback)
    
    # Add PDF drag and drop functionality for all portfolios
    st.markdown("**OR** 📎 **Drag & Drop JSON PDF:**")
    
    def extract_json_from_pdf_all(pdf_file):
        """Extract JSON content from a PDF file."""
        try:
            # Try pdfplumber first (more reliable)
            try:
                import pdfplumber
                import io
                
                # Read PDF content with pdfplumber
                pdf_bytes = io.BytesIO(pdf_file.read())
                text_content = ""
                
                with pdfplumber.open(pdf_bytes) as pdf:
                    for page in pdf.pages:
                        text_content += page.extract_text() or ""
                        
            except ImportError:
                # Fallback to PyPDF2 if pdfplumber not available
                try:
                    import PyPDF2
                    import io
                    
                    # Reset file pointer
                    pdf_file.seek(0)
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
                    
                    # Extract text from all pages
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text()
                        
                except ImportError:
                    return None, "PDF extraction libraries not available. Please install 'pip install PyPDF2' or 'pip install pdfplumber'"
            
            # Clean up the text and try to parse as JSON
            cleaned_text = text_content.strip()
            
            # Try to parse as JSON
            import json
            json_data = json.loads(cleaned_text)
            return json_data, None
            
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON in PDF: {str(e)}"
        except Exception as e:
            return None, str(e)
    
    uploaded_pdf_all = st.file_uploader(
        "Drop your All Portfolios JSON PDF here", 
        type=['pdf'], 
        help="Upload a JSON PDF file containing all portfolio configurations",
        key="strategy_all_pdf_upload"
    )
    
    if uploaded_pdf_all is not None:
        json_data, error = extract_json_from_pdf_all(uploaded_pdf_all)
        if json_data:
            # Store the extracted JSON in a different session state key to avoid widget conflicts
            st.session_state["strategy_comparison_extracted_json_all"] = json.dumps(json_data, indent=2)
            st.success(f"✅ Successfully extracted JSON from {uploaded_pdf_all.name}")
            st.info("👇 Click the button below to load the JSON into the text area.")
            def load_extracted_json_all():
                st.session_state["strategy_comparison_paste_all_json_text"] = st.session_state["strategy_comparison_extracted_json_all"]
            
            st.button("📋 Load Extracted JSON", key="load_extracted_json_all", on_click=load_extracted_json_all)
        else:
            st.error(f"❌ Failed to extract JSON from PDF: {error}")
            st.info("💡 Make sure the PDF contains valid JSON content (generated by this app)")

st.header(f"Editing Portfolio: {active_portfolio['name']}")
# Ensure session-state key exists before creating widgets to avoid duplicate-default warnings
if "strategy_comparison_active_name" not in st.session_state:
    st.session_state["strategy_comparison_active_name"] = active_portfolio['name']
active_portfolio['name'] = st.text_input("Portfolio Name", key="strategy_comparison_active_name", on_change=update_name)

# Portfolio Variant Generator - Multi-Select with Custom Options
st.markdown("---")  # Add separator

# NUCLEAR APPROACH: Portfolio-specific expander with forced refresh
portfolio_index = st.session_state.strategy_comparison_active_portfolio_index

# Store expander state in portfolio config  
if 'variant_expander_expanded' not in active_portfolio:
    active_portfolio['variant_expander_expanded'] = False

# NUCLEAR: Force expander to refresh by clearing its widget state when portfolio changes
last_portfolio_key = "strategy_comparison_last_portfolio_for_variants"
if st.session_state.get(last_portfolio_key) != portfolio_index:
    # Portfolio changed - clear all variant-related widget states
    keys_to_clear = [k for k in st.session_state.keys() if 'variant' in k.lower() and 'strategy' in k]
    for key in keys_to_clear:
        if key != last_portfolio_key:  # Don't clear the tracker itself
            del st.session_state[key]
    st.session_state[last_portfolio_key] = portfolio_index

# Use the beautiful expander with portfolio state
current_state = active_portfolio.get('variant_expander_expanded', False)

# NUCLEAR: Use a unique key that includes portfolio info to force recreation
unique_expander_key = f"variants_exp_p{portfolio_index}_v{hash(str(active_portfolio.get('name', '')))}"

with st.expander("🔧 Generate Portfolio Variants", expanded=current_state):
    st.markdown("**Select parameters to vary and customize their values:**")
    
    # Add checkbox to keep current portfolio
    keep_current_portfolio = st.checkbox(
        "✅ Keep Current Portfolio", 
        value=True, 
        key="strategy_comparison_keep_current_portfolio",
        help="When checked, the current portfolio (including benchmark) will be kept. When unchecked, only the generated variants will be created."
    )
    
    st.markdown("---")
    
    variant_params = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Rebalance Frequency:**")
        rebalance_options = []
        if st.checkbox("Never", key="strategy_rebalance_never"):
            rebalance_options.append("Never")
        if st.checkbox("Buy & Hold", key="strategy_rebalance_buyhold"):
            rebalance_options.append("Buy & Hold")
        if st.checkbox("Buy & Hold (Target)", key="strategy_rebalance_buyhold_target"):
            rebalance_options.append("Buy & Hold (Target)")
        if st.checkbox("Weekly", key="strategy_rebalance_weekly"):
            rebalance_options.append("Weekly")
        if st.checkbox("Biweekly", key="strategy_rebalance_biweekly"):
            rebalance_options.append("Biweekly")
        if st.checkbox("Monthly", key="strategy_rebalance_monthly"):
            rebalance_options.append("Monthly")
        if st.checkbox("Quarterly", value=True, key="strategy_rebalance_quarterly"):
            rebalance_options.append("Quarterly")
        if st.checkbox("Semiannually", key="strategy_rebalance_semiannually"):
            rebalance_options.append("Semiannually")
        if st.checkbox("Annually", key="strategy_rebalance_annually"):
            rebalance_options.append("Annually")
        
        if rebalance_options:
            variant_params["rebalance_frequency"] = rebalance_options
        else:
            st.error("⚠️ **At least one Rebalance Frequency must be selected!**")
    
    with col2:
        use_momentum_vary = st.checkbox("Use Momentum", value=True, key="strategy_use_momentum_vary")
    
    # Show momentum options ONLY if user checked "Use Momentum"
    if use_momentum_vary:
        st.markdown("---")
        col_mom_left, col_mom_right = st.columns(2)
        
        with col_mom_left:
            st.markdown("**Momentum strategy when NOT all negative:**")
            momentum_options = []
            if st.checkbox("Classic momentum", value=True, key="strategy_momentum_classic"):
                momentum_options.append("Classic")
            if st.checkbox("Relative momentum", key="strategy_momentum_relative"):
                momentum_options.append("Relative Momentum")
            
            if momentum_options:
                variant_params["momentum_strategy"] = momentum_options
            else:
                st.error("⚠️ **At least one momentum strategy must be selected!**")
            
            st.markdown("---")
            
            st.markdown("**Strategy when ALL momentum scores are negative:**")
            negative_options = []
            if st.checkbox("Cash", value=True, key="strategy_negative_cash"):
                negative_options.append("Cash")
            if st.checkbox("Equal weight", key="strategy_negative_equal"):
                negative_options.append("Equal weight")
            if st.checkbox("Relative momentum", key="strategy_negative_relative"):
                negative_options.append("Relative momentum")
            
            if negative_options:
                variant_params["negative_strategy"] = negative_options
            else:
                st.error("⚠️ **At least one negative strategy must be selected!**")
        
        with col_mom_right:
            st.markdown("**Include Beta in momentum weighting:**")
            beta_options = []
            if st.checkbox("With Beta", value=True, key="strategy_beta_yes"):
                beta_options.append(True)
            if st.checkbox("Without Beta", key="strategy_beta_no"):
                beta_options.append(False)
            
            if beta_options:
                variant_params["include_beta"] = beta_options
            else:
                st.error("⚠️ **At least one Beta option must be selected!**")
            
            st.markdown("---")
            
            st.markdown("**Include Volatility in momentum weighting:**")
            vol_options = []
            if st.checkbox("With Volatility", value=True, key="strategy_vol_yes"):
                vol_options.append(True)
            if st.checkbox("Without Volatility", key="strategy_vol_no"):
                vol_options.append(False)
            
            if vol_options:
                variant_params["include_volatility"] = vol_options
            else:
                st.error("⚠️ **At least one Volatility option must be selected!**")
        
        # Minimal Threshold Filter Section - COMPLETELY NEW APPROACH
        st.markdown("---")
        st.markdown("**Minimal Threshold Filter:**")
        
        # Initialize threshold values list if not exists
        if f"threshold_values_{portfolio_index}" not in st.session_state:
            st.session_state[f"threshold_values_{portfolio_index}"] = [2.0]
        
        # Checkboxes for both options (can be both selected)
        col1, col2 = st.columns(2)
        
        with col1:
            disabled = st.checkbox(
                "Disable Threshold",
                value=True,
                key=f"thresh_disabled_{portfolio_index}"
            )
        
        with col2:
            enabled = st.checkbox(
                "Enable Threshold",
                key=f"thresh_enabled_{portfolio_index}"
            )
        
        # Build threshold options
        threshold_options = []
        
        if disabled:
            threshold_options.append(None)
        
        if enabled:
            st.markdown("**Threshold Values:**")
            
            # Add button
            if st.button("➕ Add", key=f"add_thresh_{portfolio_index}"):
                st.session_state[f"threshold_values_{portfolio_index}"].append(2.0)
                st.rerun()
            
            # Display values with truly unique keys for each value
            values = st.session_state[f"threshold_values_{portfolio_index}"]
            for i in range(len(values)):
                col1, col2 = st.columns([4, 1])
                
                # Create truly unique key using timestamp and index
                unique_id = f"{portfolio_index}_{i}_{id(values)}"
                
                with col1:
                    val = st.number_input(
                        f"Value {i+1}",
                        min_value=0.0,
                        max_value=100.0,
                        value=values[i],
                        step=0.1,
                        key=f"thresh_input_{unique_id}"
                    )
                    # Update the value in the list
                    values[i] = val
                    threshold_options.append(val)
                
                with col2:
                    if len(values) > 1 and st.button("🗑️", key=f"del_thresh_{unique_id}"):
                        # Remove the specific index
                        st.session_state[f"threshold_values_{portfolio_index}"] = values[:i] + values[i+1:]
                        st.rerun()
        
        # Add to variant params
        if threshold_options:
            variant_params["minimal_threshold"] = threshold_options
    else:
        variant_params["minimal_threshold"] = [None]
        
        # CLEAN SESSION STATE: When momentum is disabled, clean up threshold session state
        if f"threshold_filters_{portfolio_index}" in st.session_state:
            del st.session_state[f"threshold_filters_{portfolio_index}"]
        if f"disable_threshold_{portfolio_index}" in st.session_state:
            del st.session_state[f"disable_threshold_{portfolio_index}"]
        if f"enable_threshold_{portfolio_index}" in st.session_state:
            del st.session_state[f"enable_threshold_{portfolio_index}"]
    
    # Maximum Allocation Filter Section - COMPLETELY NEW APPROACH
    if use_momentum_vary:
        st.markdown("---")
        st.markdown("**Maximum Allocation Filter:**")
        
        # Initialize max allocation values list if not exists
        if f"max_allocation_values_{portfolio_index}" not in st.session_state:
            st.session_state[f"max_allocation_values_{portfolio_index}"] = [10.0]
        
        # Checkboxes for both options (can be both selected)
        col1, col2 = st.columns(2)
        
        with col1:
            disabled = st.checkbox(
                "Disable Max Allocation",
                value=True,
                key=f"max_disabled_{portfolio_index}"
            )
        
        with col2:
            enabled = st.checkbox(
                "Enable Max Allocation",
                key=f"max_enabled_{portfolio_index}"
            )
        
        # Build max allocation options
        max_allocation_options = []
        
        if disabled:
            max_allocation_options.append(None)
        
        if enabled:
            st.markdown("**Max Allocation Values:**")
            
            # Add button
            if st.button("➕ Add", key=f"add_max_{portfolio_index}"):
                st.session_state[f"max_allocation_values_{portfolio_index}"].append(10.0)
                st.rerun()
            
            # Display values with truly unique keys for each value
            values = st.session_state[f"max_allocation_values_{portfolio_index}"]
            for i in range(len(values)):
                col1, col2 = st.columns([4, 1])
                
                # Create truly unique key using timestamp and index
                unique_id = f"{portfolio_index}_{i}_{id(values)}"
                
                with col1:
                    val = st.number_input(
                        f"Value {i+1}",
                        min_value=0.1,
                        max_value=100.0,
                        value=values[i],
                        step=0.1,
                        key=f"max_input_{unique_id}"
                    )
                    # Update the value in the list
                    values[i] = val
                    max_allocation_options.append(val)
                
                with col2:
                    if len(values) > 1 and st.button("🗑️", key=f"del_max_{unique_id}"):
                        # Remove the specific index
                        st.session_state[f"max_allocation_values_{portfolio_index}"] = values[:i] + values[i+1:]
                        st.rerun()
        
        # Add to variant params
        if max_allocation_options:
            variant_params["max_allocation"] = max_allocation_options
    else:
        variant_params["max_allocation"] = [None]
        
        # CLEAN SESSION STATE: When momentum is disabled, clean up max allocation session state
        if f"max_allocation_filters_{portfolio_index}" in st.session_state:
            del st.session_state[f"max_allocation_filters_{portfolio_index}"]
        if f"disable_max_allocation_{portfolio_index}" in st.session_state:
            del st.session_state[f"disable_max_allocation_{portfolio_index}"]
        if f"enable_max_allocation_{portfolio_index}" in st.session_state:
            del st.session_state[f"enable_max_allocation_{portfolio_index}"]
    
    # Calculate total combinations
    total_variants = 1
    for param_values in variant_params.values():
        total_variants *= len(param_values)
    
    if variant_params:
        st.info(f"🎯 **{total_variants} variants** will be generated")
        
        if st.button(f"✨ Generate {total_variants} Portfolio Variants", type="primary", key=f"generate_variants_{portfolio_index}"):
            
            def generate_portfolio_variants(base_portfolio, variant_params, base_name):
                """
                Generate multiple portfolio variants based on the base portfolio and variant parameters.
                """
                import itertools
                import copy
                
                # Get all parameter names and their possible values
                param_names = list(variant_params.keys())
                param_values = list(variant_params.values())
                
                # Generate all combinations using itertools.product
                combinations = list(itertools.product(*param_values))
                
                variants = []
                
                for combination in combinations:
                    # Create a deep copy of the base portfolio
                    variant = copy.deepcopy(base_portfolio)
                    
                    # Apply each parameter value from the combination
                    for param, value in zip(param_names, combination):
                        if param == "rebalance_frequency":
                            variant["rebalancing_frequency"] = value
                        elif param == "momentum_strategy":
                            variant["momentum_strategy"] = value
                        elif param == "negative_strategy":
                            variant["negative_momentum_strategy"] = value
                        elif param == "include_beta":
                            variant["calc_beta"] = value
                        elif param == "include_volatility":
                            variant["calc_volatility"] = value
                        elif param == "minimal_threshold":
                            if value is not None:
                                variant["use_minimal_threshold"] = True
                                variant["minimal_threshold_percent"] = value
                            else:
                                variant["use_minimal_threshold"] = False
                                variant["minimal_threshold_percent"] = 2.0
                        elif param == "max_allocation":
                            if value is not None:
                                variant["use_max_allocation"] = True
                                variant["max_allocation_percent"] = value
                            else:
                                variant["use_max_allocation"] = False
                                variant["max_allocation_percent"] = 10.0
                    
                    # Generate variant name (exact same format as page 1)
                    clear_name_parts = []
                    
                    # Rebalancing frequency
                    freq = variant.get('rebalancing_frequency', 'Monthly')
                    clear_name_parts.append(freq)
                    
                    # Add dash separator
                    clear_name_parts.append("-")
                    
                    # Momentum status and strategy
                    if variant.get('use_momentum', False):
                        clear_name_parts.append("Momentum :")
                        
                        # Momentum strategy
                        if variant.get('momentum_strategy') == 'Classic':
                            clear_name_parts.append("Classic")
                        elif variant.get('momentum_strategy') == 'Relative Momentum':
                            clear_name_parts.append("Relative")
                        
                        # Negative strategy with "and" connector
                        if variant.get('negative_momentum_strategy') == 'Cash':
                            clear_name_parts.append("and Cash")
                        elif variant.get('negative_momentum_strategy') == 'Equal weight':
                            clear_name_parts.append("and Equal Weight")
                        elif variant.get('negative_momentum_strategy') == 'Relative momentum':
                            clear_name_parts.append("and Relative")
                        
                        # Beta and Volatility (only show when True, omit when False)
                        if variant.get('calc_beta', False):
                            clear_name_parts.append("- Beta")
                        if variant.get('calc_volatility', False):
                            clear_name_parts.append("- Volatility")
                    else:
                        clear_name_parts.append("No Momentum")
                        # Stop here - no beta/volatility for non-momentum portfolios
                    
                    # Add threshold information (only show when enabled)
                    if variant.get('use_minimal_threshold', False):
                        threshold_percent = variant.get('minimal_threshold_percent', 2.0)
                        clear_name_parts.append(f"- Min {threshold_percent:.2f}%")
                    
                    # Add max allocation information (only show when enabled)
                    if variant.get('use_max_allocation', False):
                        max_allocation_percent = variant.get('max_allocation_percent', 10.0)
                        clear_name_parts.append(f"- Max {max_allocation_percent:.2f}%")
                    
                    # Create the new clear name
                    clear_name = f"{base_name} ({' '.join(clear_name_parts)})"
                    variant["name"] = clear_name
                    
                    variants.append(variant)
                
                return variants
            
            import copy
            
            base_portfolio = copy.deepcopy(active_portfolio)
            base_name = base_portfolio['name']
            
            # Handle momentum based on "Use Momentum" checkbox
            if use_momentum_vary:
                base_portfolio['use_momentum'] = True
                if not base_portfolio.get('momentum_windows'):
                    base_portfolio['momentum_windows'] = [
                        {"lookback": 365, "exclude": 30, "weight": 0.5},
                        {"lookback": 180, "exclude": 30, "weight": 0.3},
                        {"lookback": 120, "exclude": 30, "weight": 0.2},
                    ]
            else:
                base_portfolio['use_momentum'] = False
                base_portfolio['momentum_windows'] = []
            
            # Generate variants
            variants = generate_portfolio_variants(base_portfolio, variant_params, base_name)
            
            # Handle portfolio removal if requested
            if not keep_current_portfolio:
                if len(st.session_state.strategy_comparison_portfolio_configs) > 1:
                    st.session_state.strategy_comparison_portfolio_configs.pop(portfolio_index)
                    if portfolio_index >= len(st.session_state.strategy_comparison_portfolio_configs):
                        st.session_state.strategy_comparison_active_portfolio_index = len(st.session_state.strategy_comparison_portfolio_configs) - 1
                else:
                    st.warning("⚠️ Cannot remove the only portfolio. Keeping original portfolio.")
                    keep_current_portfolio = True
            
            # Add variants to portfolio list
            for variant in variants:
                # Use central function to ensure unique names
                add_portfolio_to_configs(variant)
            
            # Store success message in session state to persist after rerun
            success_msg = f"🎉 **Generated {len(variants)} variants** of '{base_name}'! Original portfolio kept." if keep_current_portfolio else f"🎉 **Generated {len(variants)} variants** of '{base_name}'! Original portfolio removed."
            info_msg = f"📊 Total portfolios: {len(st.session_state.strategy_comparison_portfolio_configs)}"
            
            st.session_state[f"success_message_{portfolio_index}"] = success_msg
            st.session_state[f"info_message_{portfolio_index}"] = info_msg
            
            # DEBUG: Show what we're storing
            st.write(f"DEBUG: Storing success_msg: {success_msg}")
            st.write(f"DEBUG: Storing info_msg: {info_msg}")
            
            st.session_state.strategy_comparison_rerun_flag = True
            st.rerun()
    else:
        st.warning("⚠️ **No variants will be generated** - please select at least one option to vary.")
    
    # Display success messages AFTER the generate button (so they appear near the button)
    success_key = f"success_message_{portfolio_index}"
    info_key = f"info_message_{portfolio_index}"
    
    if success_key in st.session_state:
        st.success(st.session_state[success_key])
        del st.session_state[success_key]  # Clear after display
    
    if info_key in st.session_state:
        st.info(st.session_state[info_key])
        del st.session_state[info_key]  # Clear after display





col_left, col_right = st.columns([1, 1])
with col_left:
    if "strategy_comparison_active_initial" not in st.session_state:
        st.session_state["strategy_comparison_active_initial"] = int(active_portfolio['initial_value'])
    st.number_input("Initial Value ($)", min_value=0, step=1000, format="%d", key="strategy_comparison_active_initial", on_change=update_initial, help="Starting cash", )
with col_right:
    if "strategy_comparison_active_added_amount" not in st.session_state:
        st.session_state["strategy_comparison_active_added_amount"] = int(active_portfolio['added_amount'])
    st.number_input("Added Amount ($)", min_value=0, step=1000, format="%d", key="strategy_comparison_active_added_amount", on_change=update_added_amount, help="Amount added at each Added Frequency")

# Swap positions: show Rebalancing Frequency first, then Added Frequency.
# Use two equal-width columns and make selectboxes use the container width so they match visually.
col_freq_rebal, col_freq_add = st.columns([1, 1])
freq_options = ["Never", "Buy & Hold", "Buy & Hold (Target)", "Weekly", "Biweekly", "Monthly", "Quarterly", "Semiannually", "Annually"]
with col_freq_rebal:
    if "strategy_comparison_active_rebal_freq" not in st.session_state:
        st.session_state["strategy_comparison_active_rebal_freq"] = active_portfolio['rebalancing_frequency']
    st.selectbox("Rebalancing Frequency", freq_options, key="strategy_comparison_active_rebal_freq", on_change=update_rebal_freq, help="How often the portfolio is rebalanced. 'Buy & Hold' reinvests cash immediately using current proportions. 'Buy & Hold (Target)' reinvests cash immediately using target allocations. Cash from dividends (if 'Collect Dividends as Cash' is enabled) will be available for rebalancing.", )
with col_freq_add:
    if "strategy_comparison_active_add_freq" not in st.session_state:
        st.session_state["strategy_comparison_active_add_freq"] = active_portfolio['added_frequency']
    st.selectbox("Added Frequency", freq_options, key="strategy_comparison_active_add_freq", on_change=update_add_freq, help="How often cash is added to the portfolio. 'Buy & Hold' reinvests cash immediately using current proportions. 'Buy & Hold (Target)' reinvests cash immediately using target allocations.")

# Dividend handling option
st.session_state["strategy_comparison_active_collect_dividends_as_cash"] = active_portfolio.get('collect_dividends_as_cash', False)
st.checkbox(
    "Collect Dividends as Cash", 
    key="strategy_comparison_active_collect_dividends_as_cash",
    help="When enabled, dividends are collected as cash instead of being automatically reinvested in the stock. This cash will be available for rebalancing.",
    on_change=update_collect_dividends_as_cash
)

with st.expander("Rebalancing and Added Frequency Explained", expanded=False):
    st.markdown("""
    **Added Frequency** is the frequency at which cash is added to the portfolio.
    
    **Rebalancing Frequency** is the frequency at which the portfolio is rebalanced to the specified allocations. It is also at this date that any additional cash from the `Added Frequency` is invested into the portfolio.
    
    **Buy & Hold Options:**
    - **Buy & Hold**: When cash is available (from additions or dividends), it's immediately reinvested using the current portfolio proportions
    - **Buy & Hold (Target)**: When cash is available (from additions or dividends), it's immediately reinvested using the target allocations
    
    **Collect Dividends as Cash**: When enabled, dividends are collected as cash instead of being automatically reinvested. This cash becomes available for rebalancing.
    
    *Keeping a Rebalancing Frequency to "Never" will mean no additional cash is invested, even if you have an `Added Frequency` specified.*
    """)

# Sync buttons
if len(st.session_state.strategy_comparison_portfolio_configs) > 1:
    if st.button("Sync ALL Portfolios Cashflow from First Portfolio", on_click=sync_cashflow_from_first_portfolio_callback, use_container_width=True):
        pass
    if st.button("Sync ALL Portfolios Rebalancing Frequency from First Portfolio", on_click=sync_rebalancing_from_first_portfolio_callback, use_container_width=True):
        pass
    
    # Display sync messages locally below the buttons
    if 'strategy_comparison_cashflow_sync_message' in st.session_state and st.session_state['strategy_comparison_cashflow_sync_message']:
        message = st.session_state['strategy_comparison_cashflow_sync_message']
        message_type = st.session_state.get('strategy_comparison_cashflow_sync_message_type', 'info')
        
        if message_type == 'success':
            st.success(message)
        elif message_type == 'error':
            st.error(message)
        else:
            st.info(message)
        
        # Clear the message after displaying it
        del st.session_state['strategy_comparison_cashflow_sync_message']
        del st.session_state['strategy_comparison_cashflow_sync_message_type']
    
    if 'strategy_comparison_rebalancing_sync_message' in st.session_state and st.session_state['strategy_comparison_rebalancing_sync_message']:
        message = st.session_state['strategy_comparison_rebalancing_sync_message']
        message_type = st.session_state.get('strategy_comparison_rebalancing_sync_message_type', 'info')
        
        if message_type == 'success':
            st.success(message)
        elif message_type == 'error':
            st.error(message)
        else:
            st.info(message)
        
        # Clear the message after displaying it
        del st.session_state['strategy_comparison_rebalancing_sync_message']
        del st.session_state['strategy_comparison_rebalancing_sync_message_type']

# Sync exclusion options (only show if there are multiple portfolios and not for the first portfolio)
if len(st.session_state.strategy_comparison_portfolio_configs) > 1 and st.session_state.strategy_comparison_active_portfolio_index > 0:
    st.markdown("**🔄 Sync Exclusion Options:**")
    col_sync1, col_sync2 = st.columns(2)
    
    with col_sync1:
        # Initialize sync exclusion settings if not present
        if 'exclude_from_cashflow_sync' not in active_portfolio:
            active_portfolio['exclude_from_cashflow_sync'] = False
        if 'exclude_from_rebalancing_sync' not in active_portfolio:
            active_portfolio['exclude_from_rebalancing_sync'] = False
        
        # Rebalancing sync exclusion - use direct portfolio value to avoid caching issues
        exclude_rebalancing = st.checkbox(
            "Exclude from Rebalancing Sync", 
            value=active_portfolio['exclude_from_rebalancing_sync'],
            key=f"strategy_comparison_exclude_rebalancing_sync_{st.session_state.strategy_comparison_active_portfolio_index}",
            help="When checked, this portfolio will not be affected by 'Sync ALL Portfolios Rebalancing' button",
            on_change=lambda: update_sync_exclusion('rebalancing')
        )
        
        # Update portfolio config when checkbox changes
        if exclude_rebalancing != active_portfolio['exclude_from_rebalancing_sync']:
            active_portfolio['exclude_from_rebalancing_sync'] = exclude_rebalancing
            # Force immediate update to session state
            st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] = active_portfolio
            st.session_state.strategy_comparison_rerun_flag = True
    
    with col_sync2:
        # Cash flow sync exclusion - use direct portfolio value to avoid caching issues
        exclude_cashflow = st.checkbox(
            "Exclude from Cash Flow Sync", 
            value=active_portfolio['exclude_from_cashflow_sync'],
            key=f"strategy_comparison_exclude_cashflow_sync_{st.session_state.strategy_comparison_active_portfolio_index}",
            help="When checked, this portfolio will not be affected by 'Sync ALL Portfolios Cashflow' button",
            on_change=lambda: update_sync_exclusion('cashflow')
        )
        
        # Update portfolio config when checkbox changes
        if exclude_cashflow != active_portfolio['exclude_from_cashflow_sync']:
            active_portfolio['exclude_from_cashflow_sync'] = exclude_cashflow
            # Force immediate update to session state
            st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] = active_portfolio
            st.session_state.strategy_comparison_rerun_flag = True

if "strategy_comparison_active_benchmark" not in st.session_state:
    st.session_state["strategy_comparison_active_benchmark"] = active_portfolio['benchmark_ticker']
st.text_input("Benchmark Ticker (default: ^GSPC, starts 1927-12-30, used for beta calculation. Use SPYSIM for earlier dates, starts 1885-03-01)", key="strategy_comparison_active_benchmark", on_change=update_benchmark)


st.subheader("Strategy")
if "strategy_comparison_active_use_momentum" not in st.session_state:
    st.session_state["strategy_comparison_active_use_momentum"] = active_portfolio['use_momentum']
if "strategy_comparison_active_use_targeted_rebalancing" not in st.session_state:
    st.session_state["strategy_comparison_active_use_targeted_rebalancing"] = active_portfolio.get('use_targeted_rebalancing', False)
if "strategy_comparison_active_use_threshold" not in st.session_state:
    st.session_state["strategy_comparison_active_use_threshold"] = active_portfolio.get('use_minimal_threshold', False)
if "strategy_comparison_active_threshold_percent" not in st.session_state:
    st.session_state["strategy_comparison_active_threshold_percent"] = active_portfolio.get('minimal_threshold_percent', 0.0)
# Only show momentum strategy if targeted rebalancing is disabled
if not st.session_state.get("strategy_comparison_active_use_targeted_rebalancing", False):
    st.checkbox("Use Momentum Strategy", key="strategy_comparison_active_use_momentum", on_change=update_use_momentum, help="Enables momentum-based weighting of stocks.")
else:
    # Hide momentum strategy when targeted rebalancing is enabled
    st.session_state["strategy_comparison_active_use_momentum"] = False

if st.session_state.get('strategy_comparison_active_use_momentum', active_portfolio.get('use_momentum', True)):
    st.markdown("---")
    col_mom_options, col_beta_vol = st.columns(2)
    with col_mom_options:
        st.markdown("**Momentum Strategy Options**")
        momentum_strategy = st.selectbox(
            "Momentum strategy when NOT all negative:",
            ["Classic", "Relative Momentum"],
            index=["Classic", "Relative Momentum"].index(active_portfolio.get('momentum_strategy', 'Classic')),
            key=f"strategy_comparison_momentum_strategy_{st.session_state.strategy_comparison_active_portfolio_index}"
        )
        negative_momentum_strategy = st.selectbox(
            "Strategy when ALL momentum scores are negative:",
            ["Cash", "Equal weight", "Relative momentum"],
            index=["Cash", "Equal weight", "Relative momentum"].index(active_portfolio.get('negative_momentum_strategy', 'Cash')),
            key=f"strategy_comparison_negative_momentum_strategy_{st.session_state.strategy_comparison_active_portfolio_index}"
        )
        active_portfolio['momentum_strategy'] = momentum_strategy
        active_portfolio['negative_momentum_strategy'] = negative_momentum_strategy
        st.markdown("💡 **Note:** These options control how weights are assigned based on momentum scores.")

    with col_beta_vol:
        if "strategy_comparison_active_calc_beta" not in st.session_state:
            st.session_state["strategy_comparison_active_calc_beta"] = active_portfolio['calc_beta']
        st.checkbox("Include Beta in momentum weighting", key="strategy_comparison_active_calc_beta", on_change=update_calc_beta, help="Incorporates a stock's Beta (volatility relative to the benchmark) into its momentum score.")
        # Reset Beta button
        if st.button("Reset Beta", key=f"strategy_comparison_reset_beta_btn_{st.session_state.strategy_comparison_active_portfolio_index}", on_click=reset_beta_callback):
            pass
        if st.session_state.get('strategy_comparison_active_calc_beta', False):
            # Always ensure widgets show current portfolio values when beta is enabled
            st.session_state["strategy_comparison_active_beta_window"] = active_portfolio.get('beta_window_days', 365)
            st.session_state["strategy_comparison_active_beta_exclude"] = active_portfolio.get('exclude_days_beta', 30)
            st.number_input("Beta Lookback (days)", min_value=1, key="strategy_comparison_active_beta_window", on_change=update_beta_window)
            st.number_input("Beta Exclude (days)", min_value=0, key="strategy_comparison_active_beta_exclude", on_change=update_beta_exclude)
        if "strategy_comparison_active_calc_vol" not in st.session_state:
            st.session_state["strategy_comparison_active_calc_vol"] = active_portfolio['calc_volatility']
        st.checkbox("Include Volatility in momentum weighting", key="strategy_comparison_active_calc_vol", on_change=update_calc_vol, help="Incorporates a stock's volatility (standard deviation of returns) into its momentum score.")
        # Reset Volatility button
        if st.button("Reset Volatility", key=f"strategy_comparison_reset_vol_btn_{st.session_state.strategy_comparison_active_portfolio_index}", on_click=reset_vol_callback):
            pass
        if st.session_state.get('strategy_comparison_active_calc_vol', False):
            # Always ensure widgets show current portfolio values when volatility is enabled
            st.session_state["strategy_comparison_active_vol_window"] = active_portfolio.get('vol_window_days', 365)
            st.session_state["strategy_comparison_active_vol_exclude"] = active_portfolio.get('exclude_days_vol', 30)
            st.number_input("Volatility Lookback (days)", min_value=1, key="strategy_comparison_active_vol_window", on_change=update_vol_window)
            st.number_input("Volatility Exclude (days)", min_value=0, key="strategy_comparison_active_vol_exclude", on_change=update_vol_exclude)
    
    # Minimal Threshold Filter Section
    st.markdown("---")
    st.subheader("Minimal Threshold Filter")
    
    # ALWAYS sync threshold settings from portfolio (not just if not present)
    st.session_state["strategy_comparison_active_use_threshold"] = active_portfolio.get('use_minimal_threshold', False)
    st.session_state["strategy_comparison_active_threshold_percent"] = active_portfolio.get('minimal_threshold_percent', 2.0)
    
    st.checkbox(
        "Enable Minimal Threshold Filter", 
        key="strategy_comparison_active_use_threshold", 
        on_change=update_use_threshold,
        help="Exclude stocks with allocations below the threshold percentage and normalize remaining allocations to 100%"
    )
    
    if st.session_state.get("strategy_comparison_active_use_threshold", False):
        st.number_input(
            "Minimal Threshold (%)", 
            min_value=0.1, 
            max_value=50.0, 
            value=st.session_state.get("strategy_comparison_active_threshold_percent", 0.0), 
            step=0.1,
            key="strategy_comparison_active_threshold_percent", 
            on_change=update_threshold_percent,
            help="Stocks with allocations below this percentage will be excluded and their weight redistributed to remaining stocks"
        )
    
    # Maximum Allocation Filter Section
    st.markdown("---")
    st.subheader("Maximum Allocation Filter")
    
    # Initialize session state for maximum allocation settings - ALWAYS sync with active portfolio
    st.session_state["strategy_comparison_active_use_max_allocation"] = active_portfolio.get('use_max_allocation', False)
    st.session_state["strategy_comparison_active_max_allocation_percent"] = active_portfolio.get('max_allocation_percent', 0.0)
    
    st.checkbox(
        "Enable Maximum Allocation Filter", 
        key="strategy_comparison_active_use_max_allocation", 
        on_change=update_use_max_allocation,
        help="Cap individual stock allocations at the maximum percentage and redistribute excess weight to other stocks"
    )
    
    if st.session_state.get("strategy_comparison_active_use_max_allocation", False):
        st.number_input(
            "Maximum Allocation (%)", 
            min_value=0.1, 
            max_value=100.0, 
            step=0.1,
            key="strategy_comparison_active_max_allocation_percent", 
            on_change=update_max_allocation_percent,
            help="Individual stocks will be capped at this maximum allocation percentage"
        )
    
    st.markdown("---")
    st.subheader("Momentum Windows")
    col_reset, col_norm, col_addrem = st.columns([0.4, 0.4, 0.2])
    with col_reset:
        if st.button("Reset Momentum Windows", on_click=reset_momentum_windows_callback):
            pass
    with col_norm:
        if st.button("Normalize Weights to 100%", on_click=normalize_momentum_weights_callback):
            pass
    with col_addrem:
        if st.button("Add Window", on_click=add_momentum_window_callback):
            pass
        if st.button("Remove Window", on_click=remove_momentum_window_callback):
            pass

    total_weight = sum(w['weight'] for w in active_portfolio['momentum_windows'])
    if abs(total_weight - 1.0) > 0.001:
        st.warning(f"Current total weight is {total_weight*100:.2f}%, not 100%. Click 'Normalize Weights' to fix.")
    else:
        st.success(f"Total weight is {total_weight*100:.2f}%.")

    def update_momentum_lookback(index):
        try:
            active_index = st.session_state.strategy_comparison_active_portfolio_index
            portfolio_configs = st.session_state.strategy_comparison_portfolio_configs
            if (active_index < len(portfolio_configs) and 
                'momentum_windows' in portfolio_configs[active_index] and 
                index < len(portfolio_configs[active_index]['momentum_windows'])):
                portfolio_configs[active_index]['momentum_windows'][index]['lookback'] = st.session_state[f"strategy_comparison_lookback_active_{index}"]
        except Exception:
            pass  # Silently ignore if indices are invalid

    def update_momentum_exclude(index):
        try:
            active_index = st.session_state.strategy_comparison_active_portfolio_index
            portfolio_configs = st.session_state.strategy_comparison_portfolio_configs
            if (active_index < len(portfolio_configs) and 
                'momentum_windows' in portfolio_configs[active_index] and 
                index < len(portfolio_configs[active_index]['momentum_windows'])):
                portfolio_configs[active_index]['momentum_windows'][index]['exclude'] = st.session_state[f"strategy_comparison_exclude_active_{index}"]
        except Exception:
            pass  # Silently ignore if indices are invalid
    
    def update_momentum_weight(index):
        try:
            active_index = st.session_state.strategy_comparison_active_portfolio_index
            portfolio_configs = st.session_state.strategy_comparison_portfolio_configs
            if (active_index < len(portfolio_configs) and 
                'momentum_windows' in portfolio_configs[active_index] and 
                index < len(portfolio_configs[active_index]['momentum_windows'])):
                portfolio_configs[active_index]['momentum_windows'][index]['weight'] = st.session_state[f"strategy_comparison_weight_input_active_{index}"] / 100.0
        except Exception:
            pass  # Silently ignore if indices are invalid

    # Create lambda functions for on_change callbacks
    def create_momentum_lookback_callback(index):
        return lambda: update_momentum_lookback(index)
    
    def create_momentum_exclude_callback(index):
        return lambda: update_momentum_exclude(index)
    
    def create_momentum_weight_callback(index):
        return lambda: update_momentum_weight(index)

    # Allow the user to remove momentum windows down to zero.
    # Previously the UI forced a minimum of 3 windows which prevented removing them.
    # If no windows exist, show an informational message and allow adding via the button.
    if len(active_portfolio.get('momentum_windows', [])) == 0:
        st.info("No momentum windows configured. Click 'Add Window' to create momentum lookback windows.")
    col_headers = st.columns(3)
    with col_headers[0]:
        st.markdown("**Lookback (days)**")
    with col_headers[1]:
        st.markdown("**Exclude (days)**")
    with col_headers[2]:
        st.markdown("**Weight %**")

    for j in range(len(active_portfolio['momentum_windows'])):
        with st.container():
            col_mw1, col_mw2, col_mw3 = st.columns(3)
            lookback_key = f"strategy_comparison_lookback_active_{j}"
            exclude_key = f"strategy_comparison_exclude_active_{j}"
            weight_key = f"strategy_comparison_weight_input_active_{j}"
            # Initialize session state values if not present
            if lookback_key not in st.session_state:
                # Convert lookback to integer to match min_value type
                st.session_state[lookback_key] = int(active_portfolio['momentum_windows'][j]['lookback'])
            if exclude_key not in st.session_state:
                # Convert exclude to integer to match min_value type
                st.session_state[exclude_key] = int(active_portfolio['momentum_windows'][j]['exclude'])
            if weight_key not in st.session_state:
                # Sanitize weight to prevent StreamlitValueAboveMaxError
                weight = active_portfolio['momentum_windows'][j]['weight']
                if isinstance(weight, (int, float)):
                    # If weight is already a percentage (e.g., 50 for 50%), use it directly
                    if weight > 1.0:
                        # Cap at 100% and use as percentage
                        weight_percentage = min(weight, 100.0)
                    else:
                        # Convert decimal to percentage
                        weight_percentage = weight * 100.0
                else:
                    # Invalid weight, set to default
                    weight_percentage = 10.0
                st.session_state[weight_key] = int(weight_percentage)
            
            # Get current values from session state
            current_lookback = st.session_state[lookback_key]
            current_exclude = st.session_state[exclude_key]
            current_weight = st.session_state[weight_key]
            
            with col_mw1:
                st.number_input(f"Lookback {j+1}", min_value=1, key=lookback_key, on_change=create_momentum_lookback_callback(j), label_visibility="collapsed")
            with col_mw2:
                st.number_input(f"Exclude {j+1}", min_value=0, key=exclude_key, on_change=create_momentum_exclude_callback(j), label_visibility="collapsed")
            with col_mw3:
                st.number_input(f"Weight {j+1}", min_value=0, max_value=100, step=1, format="%d", key=weight_key, on_change=create_momentum_weight_callback(j), label_visibility="collapsed")
else:
    # Don't clear momentum_windows - they should persist when momentum is disabled
    # so they're available when momentum is re-enabled or for variant generation
    active_portfolio['momentum_windows'] = []

# Targeted Rebalancing Section (only when momentum is disabled)
st.markdown("---")
st.subheader("Targeted Rebalancing")

# Only show targeted rebalancing if momentum strategy is disabled
if not st.session_state.get('strategy_comparison_active_use_momentum', False):
    st.checkbox(
        "Enable Targeted Rebalancing", 
        key="strategy_comparison_active_use_targeted_rebalancing", 
        on_change=update_use_targeted_rebalancing,
        help="Automatically rebalance when ticker allocations exceed min/max thresholds"
    )
else:
    # Hide targeted rebalancing when momentum strategy is enabled
    st.session_state["strategy_comparison_active_use_targeted_rebalancing"] = False

# Update active portfolio with current targeted rebalancing state
active_portfolio['use_targeted_rebalancing'] = st.session_state.get("strategy_comparison_active_use_targeted_rebalancing", False)

if st.session_state.get("strategy_comparison_active_use_targeted_rebalancing", False):
    st.markdown("**Configure per-ticker allocation limits:**")
    st.markdown("💡 *Example: TQQQ 70-40% means if TQQQ goes above 70%, sell to buy others; if below 40%, buy TQQQ with others*")
    
    # Get current tickers
    stocks_list = active_portfolio.get('stocks', [])
    current_tickers = [s['ticker'] for s in stocks_list if s.get('ticker')]
    
    if current_tickers:
        # Initialize targeted rebalancing settings for each ticker
        if 'targeted_rebalancing_settings' not in active_portfolio:
            active_portfolio['targeted_rebalancing_settings'] = {}
        
        for ticker in current_tickers:
            if ticker not in active_portfolio.get('targeted_rebalancing_settings', {}):
                if 'targeted_rebalancing_settings' not in active_portfolio:
                    active_portfolio['targeted_rebalancing_settings'] = {}
                active_portfolio['targeted_rebalancing_settings'][ticker] = {
                    'enabled': False,
                    'min_allocation': 0.0,
                    'max_allocation': 100.0
                }
        
        # Create columns for ticker settings
        cols = st.columns(min(len(current_tickers), 3))
        
        for i, ticker in enumerate(current_tickers):
            with cols[i % 3]:
                st.markdown(f"**{ticker}**")
                
                # Enable/disable for this ticker
                enabled_key = f"targeted_rebalancing_enabled_{ticker}_{st.session_state.strategy_comparison_active_portfolio_index}"
                if enabled_key not in st.session_state:
                    st.session_state[enabled_key] = active_portfolio['targeted_rebalancing_settings'][ticker]['enabled']
                
                # Create callback function for this specific ticker
                def create_ticker_callback(t):
                    def ticker_callback():
                        # Update portfolio settings immediately when checkbox changes
                        active_portfolio['targeted_rebalancing_settings'][t]['enabled'] = st.session_state[enabled_key]
                    return ticker_callback
                
                enabled = st.checkbox(
                    "Enable", 
                    key=enabled_key,
                    on_change=create_ticker_callback(ticker),
                    help=f"Enable targeted rebalancing for {ticker}"
                )
                # ALWAYS update portfolio settings to match session state (even if checkbox wasn't clicked)
                active_portfolio['targeted_rebalancing_settings'][ticker]['enabled'] = st.session_state[enabled_key]
                
                if st.session_state[enabled_key]:
                    # Max allocation (on top)
                    max_key = f"targeted_rebalancing_max_{ticker}_{st.session_state.strategy_comparison_active_portfolio_index}"
                    if max_key not in st.session_state:
                        st.session_state[max_key] = active_portfolio['targeted_rebalancing_settings'][ticker]['max_allocation']
                    
                    max_allocation = st.number_input(
                        "Max %", 
                        min_value=0.0, 
                        max_value=100.0, 
                        step=0.1,
                        key=max_key,
                        help=f"Maximum allocation percentage for {ticker}"
                    )
                    active_portfolio['targeted_rebalancing_settings'][ticker]['max_allocation'] = max_allocation
                    
                    # Min allocation (below)
                    min_key = f"targeted_rebalancing_min_{ticker}_{st.session_state.strategy_comparison_active_portfolio_index}"
                    if min_key not in st.session_state:
                        st.session_state[min_key] = active_portfolio['targeted_rebalancing_settings'][ticker]['min_allocation']
                    
                    min_allocation = st.number_input(
                        "Min %", 
                        min_value=0.0, 
                        max_value=100.0, 
                        step=0.1,
                        key=min_key,
                        help=f"Minimum allocation percentage for {ticker}"
                    )
                    active_portfolio['targeted_rebalancing_settings'][ticker]['min_allocation'] = min_allocation
                    
                    # Validation
                    if min_allocation >= max_allocation:
                        st.error(f"Min % must be less than Max % for {ticker}")
    else:
        st.info("Add tickers to configure targeted rebalancing settings.")

with st.expander("JSON Configuration (Copy & Paste)", expanded=False):
    # Clean portfolio config for export by removing unused settings
    cleaned_config = active_portfolio.copy()
    cleaned_config.pop('use_relative_momentum', None)
    cleaned_config.pop('equal_if_all_negative', None)
    # Update global settings from session state
    cleaned_config['start_with'] = st.session_state.get('strategy_comparison_start_with', 'all')
    cleaned_config['first_rebalance_strategy'] = st.session_state.get('strategy_comparison_first_rebalance_strategy', 'rebalancing_date')
    
    # Update targeted rebalancing settings from session state
    cleaned_config['use_targeted_rebalancing'] = st.session_state.get('strategy_comparison_active_use_targeted_rebalancing', False)
    cleaned_config['targeted_rebalancing_settings'] = active_portfolio.get('targeted_rebalancing_settings', {})
    
    # Also update the active portfolio to keep it in sync
    active_portfolio['use_targeted_rebalancing'] = st.session_state.get('strategy_comparison_active_use_targeted_rebalancing', False)
    
    # Convert date objects to strings for JSON serialization
    if cleaned_config.get('start_date_user') is not None:
        cleaned_config['start_date_user'] = cleaned_config['start_date_user'].isoformat() if hasattr(cleaned_config['start_date_user'], 'isoformat') else str(cleaned_config['start_date_user'])
    if cleaned_config.get('end_date_user') is not None:
        cleaned_config['end_date_user'] = cleaned_config['end_date_user'].isoformat() if hasattr(cleaned_config['end_date_user'], 'isoformat') else str(cleaned_config['end_date_user'])
    
    config_json = json.dumps(cleaned_config, indent=4)
    st.code(config_json, language='json')
    # Fixed JSON copy button
    import streamlit.components.v1 as components
    copy_html = f"""
    <button onclick='navigator.clipboard.writeText({json.dumps(config_json)});' style='margin-bottom:10px;'>Copy to Clipboard</button>
    """
    components.html(copy_html, height=40)
    
    # Add PDF download button for individual portfolio JSON
    def generate_individual_json_pdf(custom_name=""):
        """Generate a PDF with pure JSON content only for easy CTRL+A / CTRL+V copying."""
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Preformatted
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        import io
        from datetime import datetime
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Add proper PDF metadata
        portfolio_name = active_portfolio.get('name', 'Portfolio')
        
        # Use custom name if provided, otherwise use portfolio name
        if custom_name.strip():
            title = f"Strategy Comparison - {custom_name.strip()} - JSON Configuration"
            subject = f"JSON Configuration: {custom_name.strip()}"
        else:
            title = f"Strategy Comparison - {portfolio_name} - JSON Configuration"
            subject = f"JSON Configuration for {portfolio_name}"
        
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=36, 
            leftMargin=36, 
            topMargin=36, 
            bottomMargin=36,
            title=title,
            author="Portfolio Backtest System",
            subject=subject,
            creator="Strategy Comparison Application"
        )
        story = []
        
        # Pure JSON style - just monospace text
        json_style = ParagraphStyle(
            'PureJSONStyle',
            fontName='Courier',
            fontSize=10,
            leading=12,
            leftIndent=0,
            rightIndent=0,
            spaceAfter=0,
            spaceBefore=0
        )
        
        # Add only the JSON content - no headers, no instructions, just pure JSON
        json_lines = config_json.split('\n')
        for line in json_lines:
            story.append(Preformatted(line, json_style))
        
        # Build PDF
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    # Optional custom PDF name for individual portfolio
    custom_individual_pdf_name = st.text_input(
        "📝 Custom Portfolio JSON PDF Name (optional):", 
        value="",
        placeholder=f"e.g., {active_portfolio.get('name', 'Portfolio')} Strategy Config, Custom Analysis Setup",
        help="Leave empty to use automatic naming based on portfolio name",
        key="strategy_individual_custom_pdf_name"
    )
    
    if st.button("📄 Download JSON as PDF", help="Download a PDF containing the JSON configuration for easy copying", key="strategy_individual_json_pdf_btn"):
        try:
            pdf_data = generate_individual_json_pdf(custom_individual_pdf_name)
            
            # Generate filename based on custom name or default
            if custom_individual_pdf_name.strip():
                clean_name = custom_individual_pdf_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
                filename = f"{clean_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            else:
                filename = f"strategy_portfolio_{active_portfolio.get('name', 'portfolio').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            st.download_button(
                label="💾 Download Portfolio JSON PDF",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf",
                key="strategy_individual_json_pdf_download"
            )
            st.success("PDF generated successfully! Click the download button above.")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
    
    st.text_area("Paste JSON Here to Update Portfolio", key="strategy_comparison_paste_json_text", height=200)
    st.button("Update with Pasted JSON", on_click=paste_json_callback)
    
    # Add PDF drag and drop functionality
    st.markdown("**OR** 📎 **Drag & Drop JSON PDF:**")
    
    def extract_json_from_pdf(pdf_file):
        """Extract JSON content from a PDF file."""
        try:
            # Try pdfplumber first (more reliable)
            try:
                import pdfplumber
                import io
                
                # Read PDF content with pdfplumber
                pdf_bytes = io.BytesIO(pdf_file.read())
                text_content = ""
                
                with pdfplumber.open(pdf_bytes) as pdf:
                    for page in pdf.pages:
                        text_content += page.extract_text() or ""
                        
            except ImportError:
                # Fallback to PyPDF2 if pdfplumber not available
                try:
                    import PyPDF2
                    import io
                    
                    # Reset file pointer
                    pdf_file.seek(0)
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
                    
                    # Extract text from all pages
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text()
                        
                except ImportError:
                    return None, "PDF extraction libraries not available. Please install 'pip install PyPDF2' or 'pip install pdfplumber'"
            
            # Clean up the text and try to parse as JSON
            cleaned_text = text_content.strip()
            
            # Try to parse as JSON
            import json
            json_data = json.loads(cleaned_text)
            return json_data, None
            
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON in PDF: {str(e)}"
        except Exception as e:
            return None, str(e)
    
    uploaded_pdf = st.file_uploader(
        "Drop your JSON PDF here", 
        type=['pdf'], 
        help="Upload a JSON PDF file generated by this app to automatically load the configuration",
        key="strategy_individual_pdf_upload"
    )
    
    if uploaded_pdf is not None:
        json_data, error = extract_json_from_pdf(uploaded_pdf)
        if json_data:
            # Store the extracted JSON in a different session state key to avoid widget conflicts
            st.session_state["strategy_comparison_extracted_json"] = json.dumps(json_data, indent=4)
            st.success(f"✅ Successfully extracted JSON from {uploaded_pdf.name}")
            st.info("👇 Click the button below to load the JSON into the text area.")
            def load_extracted_json():
                st.session_state["strategy_comparison_paste_json_text"] = st.session_state["strategy_comparison_extracted_json"]
            
            st.button("📋 Load Extracted JSON", key="load_extracted_json", on_click=load_extracted_json)
        else:
            st.error(f"❌ Failed to extract JSON from PDF: {error}")
            st.info("💡 Make sure the PDF contains valid JSON content (generated by this app)")

# Run backtests when triggered from sidebar
if st.session_state.get('strategy_comparison_run_backtest', False):
    st.session_state.strategy_comparison_run_backtest = False
    
    # Pre-backtest validation check for all portfolios
    configs_to_run = st.session_state.strategy_comparison_portfolio_configs
    valid_configs = True
    for cfg in configs_to_run:
        if cfg['use_momentum']:
            total_momentum_weight = sum(w['weight'] for w in cfg['momentum_windows'])
            if abs(total_momentum_weight - 1.0) > 0.001:
                st.error(f"Portfolio '{cfg['name']}' has momentum enabled but the total momentum weight is not 100%. Please fix and try again.")
                valid_configs = False
        else:
            valid_stocks_for_cfg = [s for s in cfg['stocks'] if s['ticker']]
            total_stock_allocation = sum(s['allocation'] for s in valid_stocks_for_cfg)
            if abs(total_stock_allocation - 1.0) > 0.001:
                st.warning(f"Portfolio '{cfg['name']}' is not using momentum, but the total ticker allocation is not 100%. Click 'Normalize Tickers %' to fix.")
                
    if not valid_configs:
        st.stop()

    progress_bar = st.empty()
    progress_bar.progress(0, text="Initializing multi-strategy backtest...")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        all_tickers = sorted(list(set(s['ticker'] for cfg in st.session_state.strategy_comparison_portfolio_configs for s in cfg['stocks'] if s['ticker']) | set(cfg['benchmark_ticker'] for cfg in st.session_state.strategy_comparison_portfolio_configs if 'benchmark_ticker' in cfg)))
        all_tickers = [t for t in all_tickers if t]
        
        # CRITICAL FIX: Add base tickers for leveraged tickers to ensure dividend data is available
        base_tickers_to_add = set()
        for ticker in all_tickers:
            if "?L=" in ticker:
                base_ticker, leverage = parse_leverage_ticker(ticker)
                base_tickers_to_add.add(base_ticker)

        # Add base tickers to the list if they're not already there
        for base_ticker in base_tickers_to_add:
            if base_ticker not in all_tickers:
                all_tickers.append(base_ticker)
        # OPTIMIZED: Batch download with smart fallback
        data = {}
        invalid_tickers = []
        
        progress_text = f"Downloading data for {len(all_tickers)} tickers (batch mode)..."
        progress_bar.progress(0.1, text=progress_text)
        
        # Use batch download for all tickers (much faster!)
        batch_results = get_multiple_tickers_batch(list(all_tickers), period="max", auto_adjust=False)
        
        # Process batch results
        for i, t in enumerate(all_tickers):
            progress_text = f"Processing {t} ({i+1}/{len(all_tickers)})..."
            progress_bar.progress((i + 1) / (len(all_tickers) + 1), text=progress_text)
            
            hist = batch_results.get(t, pd.DataFrame())
            
            if hist.empty:
                # No data available for ticker
                invalid_tickers.append(t)
                continue
            
            try:
                # Force tz-naive for hist (like Backtest_Engine.py)
                hist = hist.copy()
                hist.index = hist.index.tz_localize(None)
                
                hist["Price_change"] = hist["Close"].pct_change(fill_method=None).fillna(0)
                data[t] = hist
                # Data loaded successfully
            except Exception as e:
                # Error loading ticker data
                invalid_tickers.append(t)
        # Display invalid ticker warnings in Streamlit UI
        if invalid_tickers:
            # Separate portfolio tickers from benchmark tickers
            portfolio_tickers = set(s['ticker'] for cfg in configs_to_run for s in cfg['stocks'] if s['ticker'])
            benchmark_tickers = set(cfg.get('benchmark_ticker') for cfg in configs_to_run if 'benchmark_ticker' in cfg)
            
            portfolio_invalid = [t for t in invalid_tickers if t in portfolio_tickers]
            benchmark_invalid = [t for t in invalid_tickers if t in benchmark_tickers]
            
            if portfolio_invalid:
                st.warning(f"The following portfolio tickers are invalid and will be skipped: {', '.join(portfolio_invalid)}")
            if benchmark_invalid:
                st.warning(f"The following benchmark tickers are invalid and will be skipped: {', '.join(benchmark_invalid)}")
        
        # BULLETPROOF VALIDATION: Check for valid tickers and stop gracefully if none
        if not data:
            if invalid_tickers and len(invalid_tickers) == len(all_tickers):
                st.error(f"❌ **No valid tickers found!** All tickers are invalid: {', '.join(invalid_tickers)}. Please check your ticker symbols and try again.")
            else:
                st.error("❌ **No valid tickers found!** No data downloaded; aborting.")
            progress_bar.empty()
            st.session_state.strategy_comparison_all_results = None
            st.session_state.strategy_comparison_all_allocations = None
            st.session_state.strategy_comparison_all_metrics = None
            st.stop()
        else:
            # Persist raw downloaded price data so later recomputations can access benchmark series
            st.session_state.strategy_comparison_raw_data = data
            common_start = max(df.first_valid_index() for df in data.values())
            common_end = min(df.last_valid_index() for df in data.values())
            # Determine common date range for all portfolios
            common_start = max(df.first_valid_index() for df in data.values())
            common_end = min(df.last_valid_index() for df in data.values())
            all_results = {}
            all_drawdowns = {}
            
            # Override with global start_with selection
            global_start_with = st.session_state.get('strategy_comparison_start_with', 'all')
            
            # Get all portfolio tickers (excluding benchmarks)
            all_portfolio_tickers = set()
            for cfg in configs_to_run:
                portfolio_tickers = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                all_portfolio_tickers.update(portfolio_tickers)
            
            # Check for non-USD tickers and display currency warning
            check_currency_warning(list(all_portfolio_tickers))
            
            # Determine final start date based on global setting
            # Filter to only valid tickers that exist in data
            valid_portfolio_tickers = [t for t in all_portfolio_tickers if t in data]
            
            if not valid_portfolio_tickers:
                st.error("❌ **No valid tickers found!** None of your portfolio tickers have data available. Please check your ticker symbols and try again.")
                progress_bar.empty()
                st.session_state.strategy_comparison_all_results = None
                st.session_state.strategy_comparison_all_allocations = None
                st.session_state.strategy_comparison_all_metrics = None
                st.stop()
            
            if global_start_with == 'all':
                final_start = max(data[t].first_valid_index() for t in valid_portfolio_tickers)
                # All portfolio assets start date determined
            else:  # global_start_with == 'oldest'
                final_start = min(data[t].first_valid_index() for t in valid_portfolio_tickers)
                # Oldest portfolio asset start date determined
            
            # Initialize final_end with the common end date
            final_end = common_end
            
            # Apply user date constraints if any
            for cfg in configs_to_run:
                if cfg.get('start_date_user'):
                    user_start = pd.to_datetime(cfg['start_date_user'])
                    final_start = max(final_start, user_start)
                if cfg.get('end_date_user'):
                    user_end = pd.to_datetime(cfg['end_date_user'])
                    final_end = min(final_end, user_end)
            
            if final_start > final_end:
                st.error(f"Start date {final_start.date()} is after end date {final_end.date()}. Cannot proceed.")
                st.stop()
            
            # Create simulation index for the entire period
            simulation_index = pd.date_range(start=final_start, end=final_end, freq='D')
            # Simulation period determined
            
            # Reindex all data to the simulation period (only valid tickers)
            data_reindexed = {}
            for t in all_tickers:
                if t in data:  # Only process tickers that have data
                    df = data[t].reindex(simulation_index)
                    df["Close"] = df["Close"].ffill()
                    df["Dividends"] = df["Dividends"].fillna(0)
                    df["Price_change"] = df["Close"].pct_change(fill_method=None).fillna(0)
                    data_reindexed[t] = df
            
            progress_bar.progress(1.0, text="Executing multi-strategy backtest analysis...")
            
            # =============================================================================
            # SIMPLE, FAST, AND RELIABLE STRATEGY PROCESSING
            # =============================================================================
            
            # Initialize results storage
            all_stats = {}
            all_allocations = {}
            all_metrics = {}
            portfolio_key_map = {}
            successful_strategies = 0
            failed_strategies = []
            
            st.info(f"🚀 **Processing {len(st.session_state.strategy_comparison_portfolio_configs)} strategies with enhanced reliability (NO_CACHE)...**")
            
            # Process strategies one by one with robust error handling
            for i, cfg in enumerate(st.session_state.strategy_comparison_portfolio_configs, start=1):
                try:
                    # Update progress
                    progress_percent = i / len(st.session_state.strategy_comparison_portfolio_configs)
                    progress_bar.progress(progress_percent, text=f"Processing strategy {i}/{len(st.session_state.strategy_comparison_portfolio_configs)}: {cfg.get('name', f'Strategy {i}')}")
                    
                    name = cfg.get('name', f'Strategy {i}')
                    
                    # Ensure unique key for storage
                    base_name = name
                    unique_name = base_name
                    suffix = 1
                    while unique_name in all_results or unique_name in all_allocations:
                        unique_name = f"{base_name} ({suffix})"
                        suffix += 1
                    
                    # Run single backtest for this strategy
                    total_series, total_series_no_additions, historical_allocations, historical_metrics = single_backtest(cfg, simulation_index, data_reindexed)
                    
                    if total_series is not None and len(total_series) > 0:
                        # Compute today_weights_map
                        today_weights_map = {}
                        try:
                            alloc_dates = sorted(list(historical_allocations.keys()))
                            if alloc_dates:
                                final_d = alloc_dates[-1]
                                metrics_local = historical_metrics
                                
                                # Check if momentum is used for this portfolio
                                use_momentum = cfg.get('use_momentum', True)
                                
                                if final_d in metrics_local:
                                    if use_momentum:
                                        # Extract Calculated_Weight if present (momentum-based)
                                        weights = {t: v.get('Calculated_Weight', 0) for t, v in metrics_local[final_d].items()}
                                        # Normalize (ensure sums to 1 excluding CASH)
                                        sumw = sum(w for k, w in weights.items() if k != 'CASH')
                                        if sumw > 0:
                                            norm = {k: (w / sumw) if k != 'CASH' else weights.get('CASH', 0) for k, w in weights.items()}
                                        else:
                                            norm = weights
                                        today_weights_map = norm
                                    else:
                                        # Use user-defined allocations from portfolio config
                                        today_weights_map = {}
                                        for stock in cfg.get('stocks', []):
                                            ticker = stock.get('ticker', '').strip()
                                            if ticker:
                                                today_weights_map[ticker] = stock.get('allocation', 0)
                                        # Add CASH if needed
                                        total_alloc = sum(today_weights_map.values())
                                        if total_alloc < 1.0:
                                            today_weights_map['CASH'] = 1.0 - total_alloc
                                        else:
                                            today_weights_map['CASH'] = 0
                                else:
                                    # Fallback: use allocation snapshot at final date
                                    final_alloc = historical_allocations.get(final_d, {})
                                    noncash = {k: v for k, v in final_alloc.items() if k != 'CASH'}
                                    s = sum(noncash.values())
                                    if s > 0:
                                        norm = {k: (v / s) for k, v in noncash.items()}
                                        norm['CASH'] = final_alloc.get('CASH', 0)
                                    else:
                                        norm = final_alloc
                                    today_weights_map = norm
                        except Exception as e:
                            # If computation fails, use user-defined allocations as fallback
                            today_weights_map = {}
                            for stock in cfg.get('stocks', []):
                                ticker = stock.get('ticker', '').strip()
                                if ticker:
                                    today_weights_map[ticker] = stock.get('allocation', 0)
                            # Add CASH if needed
                            total_alloc = sum(today_weights_map.values())
                            if total_alloc < 1.0:
                                today_weights_map['CASH'] = 1.0 - total_alloc
                            else:
                                today_weights_map['CASH'] = 0

                        # Calculate total money added for this portfolio
                        total_money_added = calculate_total_money_added(cfg, total_series.index[0] if len(total_series.index) > 0 else None, total_series.index[-1] if len(total_series.index) > 0 else None)
                        
                        # Store results in simplified format
                        all_results[unique_name] = {
                            'no_additions': total_series_no_additions,
                            'with_additions': total_series,
                            'today_weights_map': today_weights_map,
                            'total_money_added': total_money_added
                        }
                        all_allocations[unique_name] = historical_allocations
                        all_metrics[unique_name] = historical_metrics
                        
                        # Remember mapping from portfolio index (0-based) to unique key
                        portfolio_key_map[i-1] = unique_name
                        
                        successful_strategies += 1
                        
                        # Memory cleanup every 20 strategies
                        if successful_strategies % 20 == 0:
                            import gc
                            gc.collect()
                            
                    else:
                        failed_strategies.append((name, "Empty results from backtest"))
                        st.warning(f"⚠️ Strategy {name} failed: Empty results from backtest")
                        
                except Exception as e:
                    failed_strategies.append((cfg.get('name', f'Strategy {i}'), str(e)))
                    st.warning(f"⚠️ Strategy {cfg.get('name', f'Strategy {i}')} failed: {str(e)}")
                    continue
            
            # Final progress update
            progress_bar.progress(1.0, text="Strategy processing completed!")
            
            # Show results summary
            if successful_strategies > 0:
                st.success(f"🎉 **Successfully processed {successful_strategies}/{len(st.session_state.strategy_comparison_portfolio_configs)} strategies!**")
                if failed_strategies:
                    st.warning(f"⚠️ **{len(failed_strategies)} strategies failed** - check warnings above for details")
            else:
                st.error("❌ **No strategies were processed successfully!** Please check your configuration.")
                st.stop()
            
            # Memory cleanup
            import gc
            gc.collect()
            progress_bar.empty()
            
            # --- PATCHED CASH FLOW LOGIC ---
            # Track cash flows as pandas Series indexed by date
            cash_flows = pd.Series(0.0, index=total_series.index)
            # Initial investment: negative cash flow on first date
            if len(total_series.index) > 0:
                cash_flows.iloc[0] = -cfg.get('initial_value', 0)
            # Periodic additions: negative cash flow on their respective dates
            dates_added = get_dates_by_freq(cfg.get('added_frequency'), total_series.index[0], total_series.index[-1], total_series.index)
            for d in dates_added:
                if d in cash_flows.index and d != cash_flows.index[0]:
                    cash_flows.loc[d] -= cfg.get('added_amount', 0)
            # Final value: positive cash flow on last date for MWRR
            if len(total_series.index) > 0:
                cash_flows.iloc[-1] += total_series.iloc[-1]
            # Get benchmark returns for stats calculation
            benchmark_returns = None
            if cfg['benchmark_ticker'] and cfg['benchmark_ticker'] in data_reindexed:
                benchmark_returns = data_reindexed[cfg['benchmark_ticker']]['Price_change']
            # Ensure benchmark_returns is a pandas Series aligned to total_series
            if benchmark_returns is not None:
                benchmark_returns = pd.Series(benchmark_returns, index=total_series.index).dropna()
            # Ensure cash_flows is a pandas Series indexed by date, with initial investment and additions
            cash_flows = pd.Series(cash_flows, index=total_series.index)
            # Align for stats calculation
            # Track cash flows for MWRR exactly as in app.py
            # Initial investment: negative cash flow on first date
            mwrr_cash_flows = pd.Series(0.0, index=total_series.index)
            if len(total_series.index) > 0:
                mwrr_cash_flows.iloc[0] = -cfg.get('initial_value', 0)
            # Periodic additions: negative cash flow on their respective dates
            dates_added = get_dates_by_freq(cfg.get('added_frequency'), total_series.index[0], total_series.index[-1], total_series.index)
            for d in dates_added:
                if d in mwrr_cash_flows.index and d != mwrr_cash_flows.index[0]:
                    mwrr_cash_flows.loc[d] -= cfg.get('added_amount', 0)
            # Final value: positive cash flow on last date for MWRR
            if len(total_series.index) > 0:
                mwrr_cash_flows.iloc[-1] += total_series.iloc[-1]

            # Use the no-additions series returned by single_backtest (do NOT reconstruct it here)
            # total_series_no_additions is returned by single_backtest and already represents the portfolio value without added cash.

            # Calculate statistics
            # Use total_series_no_additions for all stats except MWRR
            stats_values = total_series_no_additions.values
            stats_dates = total_series_no_additions.index
            stats_returns = pd.Series(stats_values, index=stats_dates).pct_change().fillna(0)
            cagr = calculate_cagr(stats_values, stats_dates)
            max_dd, drawdowns = calculate_max_drawdown(stats_values)
            vol = calculate_volatility(stats_returns)
            
            # Use 2% annual risk-free rate (same as Backtest_Engine.py default)
            risk_free_rate = 0.02
            sharpe = calculate_sharpe(stats_returns, risk_free_rate)
            sortino = calculate_sortino(stats_returns, risk_free_rate)
            ulcer = calculate_ulcer_index(pd.Series(stats_values, index=stats_dates))
            upi = calculate_upi(cagr, ulcer)
            # --- Beta calculation (copied from app.py) ---
            beta = np.nan
            if benchmark_returns is not None:
                portfolio_returns = stats_returns.copy()
                benchmark_returns_series = pd.Series(benchmark_returns, index=stats_dates).dropna()
                common_idx = portfolio_returns.index.intersection(benchmark_returns_series.index)
                if len(common_idx) >= 2:
                    pr = portfolio_returns.reindex(common_idx).dropna()
                    br = benchmark_returns_series.reindex(common_idx).dropna()
                    common_idx2 = pr.index.intersection(br.index)
                    if len(common_idx2) >= 2 and br.loc[common_idx2].var() != 0:
                        cov = pr.loc[common_idx2].cov(br.loc[common_idx2])
                        var = br.loc[common_idx2].var()
                        beta = cov / var
            # MWRR uses the full backtest with additions
            mwrr = calculate_mwrr(total_series, mwrr_cash_flows, total_series.index)
            def scale_pct(val):
                if val is None or np.isnan(val):
                    return np.nan
                # Only scale if value is between -1 and 1 (decimal)
                if -1.5 < val < 1.5:
                    return val * 100
                return val

            def clamp_stat(val, stat_type):
                if val is None or np.isnan(val):
                    return np.nan
                v = scale_pct(val)
                # Clamp ranges for each stat type
                if stat_type in ["CAGR", "Volatility", "Total Return"]:
                    if v > 100:
                        return np.nan
                elif stat_type == "MWRR":
                    # MWRR can be negative or exceed 100%, so don't clamp it
                    pass
                elif stat_type == "MaxDrawdown":
                    if v < -100 or v > 0:
                        return np.nan
                return f"{v:.2f}%" if stat_type in ["CAGR", "MaxDrawdown", "Volatility", "MWRR", "Total Return"] else f"{v:.3f}" if isinstance(v, float) else v

            # Calculate total return (no additions)
                total_return = None
                if len(stats_values) > 0:
                    initial_val = stats_values[0]
                    final_val = stats_values[-1]
                    if initial_val > 0:
                        # Calculate CAGR first to determine which formula to use
                        cagr_temp = calculate_cagr(stats_values, stats_dates)
                        if cagr_temp < 0:
                            # If CAGR is negative: use DIFFERENT formula
                            total_return = (final_val / initial_val - 1) * 100  # Return as percentage
                        else:
                            # If CAGR is positive: use NORMAL calculation with * 100
                            total_return = (final_val / initial_val - 1) * 100  # Return as percentage

                stats = {
                    "Total Return": clamp_stat(total_return, "Total Return"),
                    "CAGR": clamp_stat(cagr, "CAGR"),
                    "MaxDrawdown": clamp_stat(max_dd, "MaxDrawdown"),
                    "Volatility": clamp_stat(vol, "Volatility"),
                    "Sharpe": clamp_stat(sharpe / 100 if isinstance(sharpe, (int, float)) and pd.notna(sharpe) else sharpe, "Sharpe"),
                    "Sortino": clamp_stat(sortino / 100 if isinstance(sortino, (int, float)) and pd.notna(sortino) else sortino, "Sortino"),
                    "UlcerIndex": clamp_stat(ulcer, "UlcerIndex"),
                    "UPI": clamp_stat(upi / 100 if isinstance(upi, (int, float)) and pd.notna(upi) else upi, "UPI"),
                    "Beta": clamp_stat(beta / 100 if isinstance(beta, (int, float)) and pd.notna(beta) else beta, "Beta"),
                    "MWRR": clamp_stat(mwrr, "MWRR"),
                }
                all_stats[unique_name] = stats
                all_drawdowns[unique_name] = pd.Series(drawdowns, index=stats_dates)
            progress_bar.progress(100, text="Multi-strategy backtest analysis complete!")
            progress_bar.empty()
            
            # Final Performance Statistics
            stats_df = pd.DataFrame(all_stats).T
            def fmt_pct(x):
                if isinstance(x, (int, float)) and pd.notna(x):
                    return f"{x*100:.2f}%"
                if isinstance(x, str):
                    return x
                return np.nan
            def fmt_num(x, prec=2):
                if isinstance(x, (int, float)) and pd.notna(x):
                    return f"{x:.2f}"
                if isinstance(x, str):
                    return x
                return np.nan
            if not stats_df.empty:
                stats_df_display = stats_df.copy()
                stats_df_display.rename(columns={'MaxDrawdown': 'Max Drawdown', 'UlcerIndex': 'Ulcer Index'}, inplace=True)
                stats_df_display['Total Return'] = stats_df_display['Total Return'].apply(lambda x: fmt_pct(x))
                stats_df_display['CAGR'] = stats_df_display['CAGR'].apply(lambda x: fmt_pct(x))
                stats_df_display['Max Drawdown'] = stats_df_display['Max Drawdown'].apply(lambda x: fmt_pct(x))
                stats_df_display['Volatility'] = stats_df_display['Volatility'].apply(lambda x: fmt_pct(x))
                # Ensure MWRR is the last column, Beta immediately before it, Total Return at the very end
                if 'Beta' in stats_df_display.columns and 'MWRR' in stats_df_display.columns and 'Total Return' in stats_df_display.columns:
                    cols = list(stats_df_display.columns)
                    # Remove Beta, MWRR, and Total Return
                    beta_col = cols.pop(cols.index('Beta'))
                    mwrr_col = cols.pop(cols.index('MWRR'))
                    total_return_col = cols.pop(cols.index('Total Return'))
                    # Insert Beta before MWRR, then Total Return at the very end
                    cols.append(beta_col)
                    cols.append(mwrr_col)
                    cols.append(total_return_col)
                    stats_df_display = stats_df_display[cols]
                # MWRR is already a percentage from calculate_mwrr, format it properly
                if 'MWRR' in stats_df_display.columns:
                    stats_df_display['MWRR'] = stats_df_display['MWRR'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) and pd.notna(x) else np.nan)
                stats_df_display['Sharpe'] = stats_df_display['Sharpe'].apply(lambda x: fmt_num(x))
                stats_df_display['Sortino'] = stats_df_display['Sortino'].apply(lambda x: fmt_num(x))
                stats_df_display['Ulcer Index'] = stats_df_display['Ulcer Index'].apply(lambda x: fmt_num(x))
                stats_df_display['UPI'] = stats_df_display['UPI'].apply(lambda x: fmt_num(x))
                if 'Beta' in stats_df_display.columns:
                    stats_df_display['Beta'] = stats_df_display['Beta'].apply(lambda x: fmt_num(x))
                
                # REMOVED - Extra formatting logic not in Multi-Backtest.py
                # Statistics displayed
            else:
                # No stats to display
                pass
            # Yearly performance section (interactive table below)
            all_years = {}
            for name, ser in all_results.items():
                # Use the with-additions series for yearly performance (user requested)
                yearly = ser['with_additions'].resample('YE').last()
                all_years[name] = yearly
            years = sorted(list(set(y.year for ser in all_years.values() for y in ser.index)))
            names = list(all_years.keys())
            
            # Print console log yearly table correctly
            col_width = 22
            header_format = "{:<6} |" + "".join([" {:^" + str(col_width*2+1) + "} |" for _ in names])
            row_format = "{:<6} |" + "".join([" {:>" + str(col_width) + "} {:>" + str(col_width) + "} |" for _ in names])
            
            # Yearly performance table header
            
            for y in years:
                row_items = [f"{y}"]
                for nm in names:
                    ser = all_years[nm]
                    ser_year = ser[ser.index.year == y]
                    
                    # Corrected logic for yearly performance calculation
                    start_val_for_year = None
                    if y == min(years):
                        config_for_name = next((c for c in st.session_state.strategy_comparison_portfolio_configs if c['name'] == nm), None)
                        if config_for_name:
                            initial_val_of_config = config_for_name['initial_value']
                            if initial_val_of_config > 0:
                                start_val_for_year = initial_val_of_config
                    else:
                        prev_year = y - 1
                        prev_ser_year = all_years[nm][all_years[nm].index.year == prev_year]
                        if not prev_ser_year.empty:
                            start_val_for_year = prev_ser_year.iloc[-1]
                        
                    if not ser_year.empty and start_val_for_year is not None:
                        end_val = ser_year.iloc[-1]
                        if start_val_for_year > 0:
                            pct = f"{(end_val - start_val_for_year) / start_val_for_year * 100:.2f}%"
                            final_val = f"${end_val:,.2f}"
                        else:
                            pct = "N/A"
                            final_val = "N/A"
                    else:
                        pct = "N/A"
                        final_val = "N/A"
                        
                    row_items.extend([pct, final_val])
                # Yearly performance row displayed
    
            # console output captured previously is no longer shown on the page
            st.session_state.strategy_comparison_all_results = all_results
            st.session_state.strategy_comparison_all_drawdowns = all_drawdowns
            if 'stats_df_display' in locals():
                st.session_state.strategy_comparison_stats_df_display = stats_df_display
            st.session_state.strategy_comparison_all_years = all_years
            # Clear any previous sorting when new results are calculated
            st.session_state.strategy_comparison_final_stats_sorted_df = None
            
            # Statistics table creation moved to main display section - EXACT same as Multi-Backtest
            # Save a snapshot used by the allocations UI so charts/tables remain static until rerun
            try:
                # Create today_weights_map for all portfolios
                today_weights_map = {}
                total_money_added_map = {}
                for unique_name, results in all_results.items():
                    if isinstance(results, dict):
                        if 'today_weights_map' in results:
                            today_weights_map[unique_name] = results['today_weights_map']
                        if 'total_money_added' in results:
                            total_money_added_map[unique_name] = results['total_money_added']
                
                st.session_state.strategy_comparison_snapshot_data = {
                    'raw_data': data,
                    'portfolio_configs': st.session_state.strategy_comparison_portfolio_configs,
                    'all_allocations': all_allocations,
                    'all_metrics': all_metrics,
                    'today_weights_map': today_weights_map,
                    'total_money_added_map': total_money_added_map
                }
            except Exception:
                pass
            
            st.session_state.strategy_comparison_all_allocations = all_allocations
            st.session_state.strategy_comparison_all_metrics = all_metrics
            # Save portfolio index -> unique key mapping so UI selectors can reference results reliably
            st.session_state.strategy_comparison_portfolio_key_map = portfolio_key_map
            st.session_state.strategy_comparison_ran = True
            
            # Statistics table creation moved to main display section - EXACT same as Multi-Backtest

if 'strategy_comparison_ran' in st.session_state and st.session_state.strategy_comparison_ran:
    if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
        try:
            first_date = min(series['no_additions'].index.min() for series in st.session_state.strategy_comparison_all_results.values())
            last_date = max(series['no_additions'].index.max() for series in st.session_state.strategy_comparison_all_results.values())
            st.subheader(f"Results for Backtest Period: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            st.error(f"Error calculating date range: {e}")
            # Remove debug code that was causing issues
            pass

        # Create performance chart
        try:
            fig1 = go.Figure()
            for name, series_dict in st.session_state.strategy_comparison_all_results.items():
                # Use the with-additions series for performance comparison (includes all money)
                series_to_plot = series_dict['with_additions'] if isinstance(series_dict, dict) and 'with_additions' in series_dict else series_dict
                
                # Convert timestamp index to proper datetime for plotting - EXACT same as drawdown chart
                if hasattr(series_to_plot.index, 'to_pydatetime'):
                    x_dates = series_to_plot.index.to_pydatetime()
                else:
                    x_dates = pd.to_datetime(series_to_plot.index)
                fig1.add_trace(go.Scatter(x=x_dates, y=series_to_plot.values, mode='lines', name=name))
            
            fig1.update_layout(
                title="Backtest Comparison — Portfolio Value (with cash additions)",
                xaxis_title="Date",
                legend_title="Portfolios",
                hovermode='closest',
                hoverdistance=100,
                spikedistance=1000,
                template="plotly_dark",
                yaxis_tickprefix="$",
                yaxis_tickformat=",.0f",
                # No width/height restrictions - let them be responsive like other plots
                xaxis=dict(
                    type='date',  # Explicitly set as date type
                    tickformat="%Y-%m-%d",  # Proper date format
                    tickmode="auto",
                    nticks=10,  # Reasonable number of ticks
                    tickangle=45,  # Angle labels for better readability
                    automargin=True,  # Ensure labels fit
                    range=None  # Let Plotly auto-range to ensure perfect alignment
                ),
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="top",
                    y=1.15,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=80, r=80, t=120, b=80),  # EXACT same margins as drawdown chart
                height=600,  # Taller height to prevent crushing
                yaxis=dict(
                    title="Portfolio Value ($)", 
                    title_standoff=20,
                    side="left",
                    position=0.0  # Force left alignment
                )
            )
            # Store fig1 for PDF generation
            st.session_state.strategy_comparison_fig1 = fig1
            
            st.plotly_chart(fig1, use_container_width=True, key="multi_performance_chart")
        except Exception as e:
            st.error(f"Error creating performance chart: {e}")
            st.write("Chart data not available")

        # Create drawdown chart
        try:
            fig2 = go.Figure()
            for name, series_dict in st.session_state.strategy_comparison_all_results.items():
                # Use the no-additions series for drawdown calculation (pure portfolio performance) - same as Multi-Backtest
                series_to_plot = series_dict['no_additions'] if isinstance(series_dict, dict) and 'no_additions' in series_dict else series_dict
                
                # Calculate drawdown for this series - same as Multi-Backtest
                values = series_to_plot.values
                peak = np.maximum.accumulate(values)
                drawdowns = (values - peak) / np.where(peak == 0, 1, peak) * 100  # Convert to percentage
                
                # Convert timestamp index to proper datetime for plotting - same as Multi-Backtest
                if hasattr(series_to_plot.index, 'to_pydatetime'):
                    x_dates = series_to_plot.index.to_pydatetime()
                else:
                    x_dates = pd.to_datetime(series_to_plot.index)
                fig2.add_trace(go.Scatter(x=x_dates, y=drawdowns, mode='lines', name=name))
            
            fig2.update_layout(
                title="Backtest Comparison (Max Drawdown)",
                xaxis_title="Date",
                legend_title="Portfolios",
                hovermode='closest',
                hoverdistance=100,
                spikedistance=1000,
                template="plotly_dark",
                # No width/height restrictions - let them be responsive like other plots
                xaxis=dict(
                    type='date',  # Explicitly set as date type
                    tickformat="%Y-%m-%d",  # Proper date format
                    tickmode="auto",
                    nticks=10,  # Reasonable number of ticks
                    tickangle=45,  # Angle labels for better readability
                    automargin=True,  # Ensure labels fit
                    range=None  # Let Plotly auto-range to ensure perfect alignment
                ),
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="top",
                    y=1.15,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=80, r=80, t=120, b=80),  # EXACT same margins as performance chart
                height=600,  # Taller height to prevent crushing
                yaxis=dict(
                    title="Drawdown (%)", 
                    title_standoff=20,
                    side="left",
                    position=0.0  # Force left alignment
                )
            )
            # Store fig2 for PDF generation
            st.session_state.strategy_comparison_fig2 = fig2
            
            st.plotly_chart(fig2, use_container_width=True, key="multi_drawdown_chart")
        except Exception as e:
            st.error(f"Error creating drawdown chart: {e}")
            st.write("Chart data not available")

        # VIX Index Plot (for correlation with drawdowns)
        fig_vix = go.Figure()
        try:
            # Get VIX data for the same date range
            vix_data = yf.download('^VIX', start=first_date, end=last_date, progress=False)
            
            # VIX data has multi-level columns, need to access it properly
            # The structure is ('Close', '^VIX') instead of just 'Close'
            if ('Close', '^VIX') in vix_data.columns:
                vix_close = vix_data[('Close', '^VIX')].dropna()
            else:
                # Fallback to regular Close column
                vix_close = vix_data['Close'].dropna()
            
            # Add flat line before VIX data starts (like interest rates do)
            if len(vix_close) > 0:
                first_vix_date = vix_close.index[0]
                first_vix_value = vix_close.iloc[0]
                
                # If VIX data starts after our backtest period, add flat line
                if first_vix_date > pd.Timestamp(first_date):
                    # Create flat line from start to first VIX date
                    flat_dates = pd.date_range(start=first_date, end=first_vix_date, freq='D')
                    fig_vix.add_trace(go.Scatter(
                        x=flat_dates, 
                        y=[first_vix_value] * len(flat_dates), 
                        mode='lines', 
                        name='VIX Index (Pre-Data)', 
                        line=dict(color='red', dash='dash'),
                        hovertemplate='<b>VIX Index (Pre-Data)</b><br>' +
                                     'Date: %{x}<br>' +
                                     'VIX: %{y:.2f}<br>' +
                                     '<extra></extra>'
                    ))
            
            # Add actual VIX data
            fig_vix.add_trace(go.Scatter(
                x=vix_close.index, 
                y=vix_close.values, 
                mode='lines', 
                name='VIX Index', 
                line=dict(color='red'),
                hovertemplate='<b>VIX Index</b><br>' +
                             'Date: %{x}<br>' +
                             'VIX: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
            
        except Exception as e:
            # Fallback: create a simple line at default VIX if fetching fails
            x_dates = pd.date_range(start=first_date, end=last_date, freq='D')
            fig_vix.add_trace(go.Scatter(
                x=x_dates, 
                y=[20.0] * len(x_dates), 
                mode='lines', 
                name='VIX Index (Default)', 
                line=dict(color='red')
            ))
        
        fig_vix.update_layout(
            title="VIX Index (Fear Gauge)",
            xaxis_title="Date",
            legend_title="Index",
            hovermode='closest',
            hoverdistance=100,
            spikedistance=1000,
            template="plotly_dark",
            # EXACT same formatting as the other plots
            xaxis=dict(
                type='date',  # Explicitly set as date type
                tickformat="%Y-%m-%d",  # Proper date format
                tickmode="auto",
                nticks=10,  # Reasonable number of ticks
                tickangle=45,  # Angle labels for better readability
                automargin=True,  # Ensure labels fit
                range=None  # Let Plotly auto-range to ensure perfect alignment
            ),
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=80, r=80, t=120, b=80),  # EXACT same margins as the other plots
            height=600,  # Same height as the other plots
            yaxis=dict(
                title="VIX Level", 
                title_standoff=20,
                side="left",
                position=0.0,  # Force left alignment for perfect positioning
                range=[0, 80] if 'vix_close' in locals() and len(vix_close) > 0 else [0, 80]
            )
        )
        st.plotly_chart(fig_vix, use_container_width=True, key="vix_chart")
        # Store in session state for PDF export
        st.session_state.fig_vix = fig_vix

        # Multi-Portfolio PE Ratio Comparison
        if 'strategy_comparison_all_allocations' in st.session_state and st.session_state.strategy_comparison_all_allocations:
            st.markdown("---")
            st.markdown("**📊 Multi-Portfolio PE Ratio Comparison**")
            st.warning("⚠️ **Work in Progress:** PE ratio calculations are currently using current PE ratios only. Historical PE evolution is not yet implemented and may not be fully accurate.")
            
            try:
                # Create multi-portfolio PE ratio chart
                fig_pe_multi = go.Figure()
                
                # Get all available portfolio names
                available_portfolio_names = [cfg.get('name', 'Portfolio') for cfg in st.session_state.get('strategy_comparison_portfolio_configs', [])]
                extra_names = [n for n in st.session_state.get('strategy_comparison_all_results', {}).keys() if n not in available_portfolio_names]
                all_portfolio_names = available_portfolio_names + extra_names
                
                # Color palette for different portfolios
                colors = [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
                ]
                
                # Collect PE data for all portfolios
                all_pe_data = {}
                portfolio_pe_series = {}
                
                for i, portfolio_name in enumerate(all_portfolio_names):
                    allocs_data = st.session_state.strategy_comparison_all_allocations.get(portfolio_name, {})
                    
                    if allocs_data:
                        # Get all unique tickers for this portfolio (exclude CASH)
                        all_tickers = set()
                        for date, allocs in allocs_data.items():
                            for ticker in allocs.keys():
                                if ticker is not None and ticker != 'CASH':
                                    all_tickers.add(ticker)
                        all_tickers = sorted(list(all_tickers))
                        
                        if all_tickers:
                            # Fetch PE data for all tickers (cached per portfolio)
                            pe_data = {}
                            for ticker in all_tickers:
                                try:
                                    stock = yf.Ticker(ticker)
                                    info = stock.info
                                    pe_ratio = info.get('trailingPE', None)
                                    if pe_ratio is not None and pe_ratio > 0:
                                        pe_data[ticker] = pe_ratio
                                except:
                                    continue
                            
                            if pe_data:
                                # Calculate weighted PE ratio over time for this portfolio
                                dates = sorted(allocs_data.keys())
                                portfolio_pe_ratios = []
                                
                                for date in dates:
                                    allocs = allocs_data[date]
                                    weighted_pe = 0
                                    total_weight = 0
                                    
                                    # Check if portfolio is in cash
                                    stock_allocation = sum(weight for ticker, weight in allocs.items() if ticker != 'CASH' and weight > 0)
                                    
                                    if stock_allocation == 0:
                                        portfolio_pe_ratios.append(None)
                                    else:
                                        # Calculate weighted PE only for stock allocations
                                        for ticker, weight in allocs.items():
                                            if ticker != 'CASH' and ticker in pe_data and weight > 0:
                                                weighted_pe += pe_data[ticker] * weight
                                                total_weight += weight
                                        
                                        if total_weight > 0:
                                            portfolio_pe_ratios.append(weighted_pe / total_weight)
                                        else:
                                            portfolio_pe_ratios.append(None)
                                
                                # Filter out None values and create clean data
                                clean_dates = []
                                clean_pe_ratios = []
                                for j, pe in enumerate(portfolio_pe_ratios):
                                    if pe is not None:
                                        clean_dates.append(dates[j])
                                        clean_pe_ratios.append(pe)
                                
                                if clean_pe_ratios:
                                    portfolio_pe_series[portfolio_name] = {
                                        'dates': clean_dates,
                                        'pe_ratios': clean_pe_ratios,
                                        'color': colors[i % len(colors)]
                                    }
                
                # Add traces for each portfolio
                for portfolio_name, data in portfolio_pe_series.items():
                    fig_pe_multi.add_trace(go.Scatter(
                        x=data['dates'],
                        y=data['pe_ratios'],
                        mode='lines',
                        name=portfolio_name,
                        line=dict(color=data['color'], width=2),
                        hovertemplate=(
                            f'<b>{portfolio_name}</b><br>' +
                            'Date: %{x|%Y-%m-%d}<br>' +
                            'PE Ratio: %{y:.2f}<br>' +
                            '<extra></extra>'
                        ),
                        connectgaps=False
                    ))
                
                if portfolio_pe_series:
                    # Update layout
                    fig_pe_multi.update_layout(
                        title="Multi-Portfolio PE Ratio Comparison",
                        xaxis_title="Date",
                        yaxis_title="PE Ratio",
                        template='plotly_dark',
                        height=500,
                        hovermode='closest',
                        hoverdistance=100,
                        spikedistance=1000,
                        showlegend=True,
                        xaxis=dict(
                            type='date',
                            automargin=True
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Store in session state
                    st.session_state.fig_pe_multi = fig_pe_multi
                    
                    # Display chart
                    st.plotly_chart(st.session_state.fig_pe_multi, use_container_width=True)
                else:
                    st.warning("No PE ratio data available for any portfolios.")
                    
            except Exception as e:
                st.error(f"Error creating multi-portfolio PE ratio chart: {str(e)}")

        # Third plot: Daily Risk-Free Rate (13-Week Treasury)
        fig3 = go.Figure()
        try:
            # Get risk-free rate data for the same date range
            risk_free_rates = get_risk_free_rate_robust(pd.date_range(start=first_date, end=last_date, freq='D'))
            
            # Convert timestamp index to proper datetime for plotting
            if hasattr(risk_free_rates.index, 'to_pydatetime'):
                x_dates = risk_free_rates.index.to_pydatetime()
            else:
                x_dates = pd.to_datetime(risk_free_rates.index)
            
            # Convert to daily basis points for display (multiply by 10000)
            daily_rates_bp = risk_free_rates * 10000
            
            fig3.add_trace(go.Scatter(
                x=x_dates, 
                y=daily_rates_bp.values, 
                mode='lines', 
                name='Daily Risk-Free Rate', 
                line=dict(color='#00ff88'),
                hovertemplate='<b>Daily Risk-Free Rate</b><br>' +
                             'Date: %{x}<br>' +
                             'Rate: %{y:.2f} bps<br>' +
                             '<extra></extra>'
            ))
            
        except Exception as e:
            # Fallback: create a simple line at default rate if risk-free rate fetching fails
            x_dates = pd.date_range(start=first_date, end=last_date, freq='D')
            default_daily_bp = 0.02 / 365.25 * 10000  # Convert 2% annual to daily basis points
            fig3.add_trace(go.Scatter(
                x=x_dates, 
                y=[default_daily_bp] * len(x_dates), 
                mode='lines', 
                name='Daily Risk-Free Rate (Default)', 
                line=dict(color='#00ff88')
            ))
        
        fig3.update_layout(
            title="Daily Risk-Free Rate (13-Week Treasury)",
            xaxis_title="Date",
            legend_title="Rate",
            hovermode='closest',
            hoverdistance=100,
            spikedistance=1000,
            template="plotly_dark",
            # EXACT same formatting as the other two plots
            xaxis=dict(
                type='date',  # Explicitly set as date type
                tickformat="%Y-%m-%d",  # Proper date format
                tickmode="auto",
                nticks=10,  # Reasonable number of ticks
                tickangle=45,  # Angle labels for better readability
                automargin=True,  # Ensure labels fit
                range=None  # Let Plotly auto-range to ensure perfect alignment
            ),
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=80, r=80, t=120, b=80),  # EXACT same margins as the other plots
            height=600,  # Same height as the other plots
            yaxis=dict(
                title="Daily Risk-Free Rate (basis points)", 
                title_standoff=20,
                side="left",
                position=0.0,  # Force left alignment for perfect positioning
                range=[0, max(daily_rates_bp) * 1.1] if 'daily_rates_bp' in locals() and len(daily_rates_bp) > 0 else [0, 2]
            )
        )
        st.plotly_chart(fig3, use_container_width=True, key="strategy_comparison_daily_risk_free_chart")
        # Store in session state for PDF export
        st.session_state.strategy_comparison_fig3 = fig3

        # Fourth plot: Annualized Risk-Free Rate (13-Week Treasury)
        fig4 = go.Figure()
        try:
            # Get risk-free rate data for the same date range
            risk_free_rates = get_risk_free_rate_robust(pd.date_range(start=first_date, end=last_date, freq='D'))
            
            # Convert timestamp index to proper datetime for plotting
            if hasattr(risk_free_rates.index, 'to_pydatetime'):
                x_dates = risk_free_rates.index.to_pydatetime()
            else:
                x_dates = pd.to_datetime(risk_free_rates.index)
            
            # Convert to annualized percentage for display
            # Daily rate to annual: (1 + daily_rate)^365.25 - 1
            annual_rates = ((1 + risk_free_rates) ** 365.25 - 1) * 100
            
            fig4.add_trace(go.Scatter(
                x=x_dates, 
                y=annual_rates.values, 
                mode='lines', 
                name='Annualized Risk-Free Rate', 
                line=dict(color='#ff8800'),
                hovertemplate='<b>Annualized Risk-Free Rate</b><br>' +
                             'Date: %{x}<br>' +
                             'Rate: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ))
            
        except Exception as e:
            # Fallback: create a simple line at 2% if risk-free rate fetching fails
            x_dates = pd.date_range(start=first_date, end=last_date, freq='D')
            fig4.add_trace(go.Scatter(
                x=x_dates, 
                y=[2.0] * len(x_dates), 
                mode='lines', 
                name='Annualized Risk-Free Rate (Default)', 
                line=dict(color='#ff8800')
            ))
        
        fig4.update_layout(
            title="Annualized Risk-Free Rate (13-Week Treasury)",
            xaxis_title="Date",
            legend_title="Rate",
            hovermode='closest',
            hoverdistance=100,
            spikedistance=1000,
            template="plotly_dark",
            # EXACT same formatting as the other plots
            xaxis=dict(
                type='date',  # Explicitly set as date type
                tickformat="%Y-%m-%d",  # Proper date format
                tickmode="auto",
                nticks=10,  # Reasonable number of ticks
                tickangle=45,  # Angle labels for better readability
                automargin=True,  # Ensure labels fit
                range=None  # Let Plotly auto-range to ensure perfect alignment
            ),
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=80, r=80, t=120, b=80),  # EXACT same margins as the other plots
            height=600,  # Same height as the other plots
            yaxis=dict(
                title="Annualized Risk-Free Rate (%)", 
                title_standoff=20,
                side="left",
                position=0.0,  # Force left alignment for perfect positioning
                range=[0, max(annual_rates) * 1.1] if 'annual_rates' in locals() and len(annual_rates) > 0 else [0, 6]
            )
        )
        st.plotly_chart(fig4, use_container_width=True, key="strategy_comparison_annual_risk_free_chart")
        # Store in session state for PDF export
        st.session_state.strategy_comparison_fig4 = fig4

        # --- Variation summary chart: compares total return, CAGR, volatility and max drawdown across portfolios ---
        try:
            def get_no_additions_series(obj):
                return obj['no_additions'] if isinstance(obj, dict) and 'no_additions' in obj else obj if isinstance(obj, pd.Series) else None

            metrics_summary = {}
            for name, series_obj in st.session_state.strategy_comparison_all_results.items():
                ser_no = get_no_additions_series(series_obj)
                if ser_no is None or len(ser_no) < 2:
                    continue
                vals = ser_no.values
                dates = ser_no.index
                # Total return over the period
                try:
                    total_return = (vals[-1] / vals[0] - 1) * 100 if vals[0] and not np.isnan(vals[0]) else np.nan
                except Exception:
                    total_return = np.nan

                # CAGR, volatility, max drawdown (convert to percent for display)
                try:
                    cagr = calculate_cagr(vals, dates)
                except Exception:
                    cagr = np.nan
                try:
                    returns = pd.Series(vals, index=dates).pct_change().fillna(0)
                    vol = calculate_volatility(returns)
                except Exception:
                    vol = np.nan
                try:
                    max_dd, _ = calculate_max_drawdown(vals)
                except Exception:
                    max_dd = np.nan

                metrics_summary[name] = {
                    'Total Return': total_return,
                    'CAGR': (cagr * 100) if isinstance(cagr, (int, float)) and not np.isnan(cagr) else np.nan,
                    'Volatility': (vol * 100) if isinstance(vol, (int, float)) and not np.isnan(vol) else np.nan,
                    'Max Drawdown': (max_dd * 100) if isinstance(max_dd, (int, float)) and not np.isnan(max_dd) else np.nan,
                }

            if metrics_summary:
                df_metrics = pd.DataFrame(metrics_summary).T
                # Ensure numeric columns
                for c in df_metrics.columns:
                    df_metrics[c] = pd.to_numeric(df_metrics[c], errors='coerce')

                # Create grouped bar chart
                fig_metrics = go.Figure()
                metric_order = ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown']
                colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
                for i, metric in enumerate(metric_order):
                    if metric in df_metrics.columns:
                        y_values = df_metrics[metric].values
                        
                        # Transform values for symmetric log-like display
                        # Positive values: log scale, Negative values: -log(abs(value))
                        transformed_values = []
                        for val in y_values:
                            if pd.isna(val):
                                transformed_values.append(np.nan)
                            elif val > 0:
                                transformed_values.append(np.log10(val + 1))  # +1 to handle 0
                            elif val < 0:
                                transformed_values.append(-np.log10(abs(val) + 1))  # Negative log for negative values
                            else:  # val == 0
                                transformed_values.append(0)
                        
                        fig_metrics.add_trace(go.Bar(
                            x=df_metrics.index,
                            y=transformed_values,
                            name=metric,
                            marker_color=colors[i % len(colors)],
                            text=[f"{v:.2f}%" if not pd.isna(v) else 'N/A' for v in y_values],
                            textposition='auto',
                            showlegend=True
                        ))

                fig_metrics.update_layout(
                    title='Portfolio Variation Summary (percent)',
                    barmode='group',
                    template='plotly_dark',
                    yaxis=dict(
                        title='Percent (Log Scale)', 
                        ticksuffix='%',
                        showticklabels=False,  # Remove Y-axis tick labels
                        # Linear scale since we transformed the values manually
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=10)
                    ),
                    height=520,
                    margin=dict(l=60, r=40, t=80, b=120),
                )

                st.plotly_chart(fig_metrics, use_container_width=True, key="multi_metrics_chart")
        except Exception as e:
            # Failed to build metrics summary chart
            pass

        # --- Monthly returns heatmap: rows = portfolios, columns = Year-Month, values = monthly % change ---
        try:
            # Build a DataFrame of monthly returns for each portfolio
            monthly_returns = {}
            for name, series_obj in st.session_state.strategy_comparison_all_results.items():
                ser_no = series_obj['no_additions'] if isinstance(series_obj, dict) and 'no_additions' in series_obj else series_obj if isinstance(series_obj, pd.Series) else None
                if ser_no is None or len(ser_no) < 2:
                    continue
                # Resample to month-end and compute percent change
                try:
                    # Use month-end resample with 'ME' alias to avoid FutureWarning; keep as DatetimeIndex
                    ser_month = ser_no.resample('ME').last()
                    pct_month = ser_month.pct_change().dropna() * 100
                    # label months as 'YYYY-MM' using DatetimeIndex to avoid PeriodArray conversion
                    pct_month.index = pct_month.index.strftime('%Y-%m')
                    monthly_returns[name] = pct_month
                except Exception:
                    continue

            if monthly_returns:
                # Align indexes (months) across portfolios
                all_months = sorted(list({m for ser in monthly_returns.values() for m in ser.index}))
                heat_data = pd.DataFrame(index=list(monthly_returns.keys()), columns=all_months)
                for name, ser in monthly_returns.items():
                    for m, v in ser.items():
                        heat_data.at[name, m] = v
                heat_data = heat_data.astype(float)

                # Create heatmap with Plotly
                fig_heat = go.Figure(data=go.Heatmap(
                    z=heat_data.values,
                    x=heat_data.columns.astype(str),
                    y=heat_data.index.astype(str),
                    colorscale='RdYlGn',
                    colorbar=dict(title='Monthly %'),
                    hovertemplate='Portfolio: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
                ))
                fig_heat.update_layout(
                    title='Monthly Returns Heatmap (rows=portfolios, columns=year-month)',
                    xaxis_nticks=20,
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_heat, use_container_width=True, key="multi_heatmap_chart")
        except Exception as e:
            # Failed to build monthly heatmap
            pass

        # Recompute Final Performance Statistics from stored results to ensure they use the no-additions series - EXACT same as Multi-Backtest
        if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
            # Helper to extract no-additions series whether stored as dict or Series
            def get_no_additions(series_or_dict):
                return series_or_dict['no_additions'] if isinstance(series_or_dict, dict) and 'no_additions' in series_or_dict else series_or_dict

            # MWRR will be calculated fresh for each portfolio in the recomputation loop

            recomputed_stats = {}

            def scale_pct(val):
                if val is None or (isinstance(val, (int, float)) and np.isnan(val)):
                    return np.nan
                if isinstance(val, str):
                    return val  # Return strings as-is (like "N/A")
                if isinstance(val, (int, float)) and -1.5 < val < 1.5:
                    return val * 100
                return val

            def clamp_stat(val, stat_type):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return np.nan
                v = scale_pct(val)
                
                # If scale_pct returned a string (like "N/A"), return it as-is
                if isinstance(v, str):
                    return v
                
                # Apply specific scaling for Total Return before clamping
                if stat_type == "Total Return":
                    # Total Return is now always in percentage format for both positive and negative CAGR
                    # No conversion needed - already in percentage format
                    pass
                
                # Clamping logic - separate Total Return from other percentage stats
                if stat_type in ["CAGR", "Volatility", "MWRR"]:
                    if isinstance(v, (int, float)) and v > 100:
                        return np.nan
                elif stat_type == "Total Return":
                    # Allow negative total returns - no clamping needed
                    pass
                elif stat_type == "MaxDrawdown":
                    if isinstance(v, (int, float)) and (v < -100 or v > 0):
                        return np.nan
                
                return f"{v:.2f}%" if stat_type in ["CAGR", "MaxDrawdown", "Volatility", "MWRR", "Total Return"] else f"{v:.3f}" if isinstance(v, float) else v

            for name, series_obj in st.session_state.strategy_comparison_all_results.items():
                ser_noadd = get_no_additions(series_obj)
                if ser_noadd is None or len(ser_noadd) < 2:
                    recomputed_stats[name] = {
                        "Total Return": np.nan,
                        "CAGR": np.nan,
                        "MaxDrawdown": np.nan,
                        "Volatility": np.nan,
                        "Sharpe": np.nan,
                        "Sortino": np.nan,
                        "UlcerIndex": np.nan,
                        "UPI": np.nan,
                        "Beta": np.nan,
                        "MWRR": np.nan,
                        # Final values with and without additions (if available)
                        "Final Value (with)": (series_obj['with_additions'].iloc[-1] if isinstance(series_obj, dict) and 'with_additions' in series_obj and len(series_obj['with_additions'])>0 else np.nan),
                        "Final Value (no_additions)": (ser_noadd.iloc[-1] if isinstance(ser_noadd, pd.Series) and len(ser_noadd)>0 else np.nan)
                    }
                    continue

                stats_values = ser_noadd.values
                stats_dates = ser_noadd.index
                stats_returns = pd.Series(stats_values, index=stats_dates).pct_change().fillna(0)
                
                # Calculate total return (no additions)
                total_return = None
                if len(stats_values) > 0:
                    initial_val = stats_values[0]
                    final_val = stats_values[-1]
                    if initial_val > 0:
                        # Calculate CAGR first to determine which formula to use
                        cagr_temp = calculate_cagr(stats_values, stats_dates)
                        if cagr_temp < 0:
                            # If CAGR is negative: use DIFFERENT formula
                            total_return = (final_val / initial_val - 1) * 100  # Return as percentage
                        else:
                            # If CAGR is positive: use NORMAL calculation with * 100
                            total_return = (final_val / initial_val - 1) * 100  # Return as percentage
                
                cagr = calculate_cagr(stats_values, stats_dates)
                max_dd, drawdowns = calculate_max_drawdown(stats_values)
                vol = calculate_volatility(stats_returns)
                
                # Use 2% annual risk-free rate (same as Backtest_Engine.py default)
                risk_free_rate = 0.02
                sharpe = calculate_sharpe(stats_returns, risk_free_rate)
                sortino = calculate_sortino(stats_returns, risk_free_rate)
                ulcer = calculate_ulcer_index(pd.Series(stats_values, index=stats_dates))
                upi = calculate_upi(cagr, ulcer)
                # Compute Beta based on the no-additions portfolio returns and the portfolio's benchmark (if available)
                beta = np.nan
                # Find the portfolio config to get benchmark ticker
                cfg_for_name = next((c for c in st.session_state.strategy_comparison_portfolio_configs if c['name'] == name), None)
                if cfg_for_name:
                    bench_ticker = cfg_for_name.get('benchmark_ticker')
                    raw_data = st.session_state.get('strategy_comparison_raw_data')
                    if bench_ticker and raw_data and bench_ticker in raw_data:
                        # get benchmark price_change series aligned to ser_noadd index
                        try:
                            bench_df = raw_data[bench_ticker].reindex(ser_noadd.index)
                            if 'Price_change' in bench_df.columns:
                                bench_returns = bench_df['Price_change'].fillna(0)
                            else:
                                bench_returns = bench_df['Close'].pct_change().fillna(0)

                            portfolio_returns = pd.Series(stats_values, index=stats_dates).pct_change().fillna(0)
                            common_idx = portfolio_returns.index.intersection(bench_returns.index)
                            if len(common_idx) >= 2:
                                pr = portfolio_returns.reindex(common_idx).dropna()
                                br = bench_returns.reindex(common_idx).dropna()
                                common_idx2 = pr.index.intersection(br.index)
                                if len(common_idx2) >= 2 and br.loc[common_idx2].var() != 0:
                                    cov = pr.loc[common_idx2].cov(br.loc[common_idx2])
                                    var = br.loc[common_idx2].var()
                                    beta = cov / var
                        except Exception as e:
                            # Failed to compute beta
                            pass
                
                # Calculate MWRR for this portfolio using the complete cash flow series
                mwrr_val = np.nan  # Use NaN instead of "N/A" string
                if isinstance(series_obj, dict) and 'with_additions' in series_obj:
                    portfolio_values = series_obj['with_additions']
                    # Reconstruct cash flows for this portfolio
                    cfg_for_name = next((c for c in st.session_state.strategy_comparison_portfolio_configs if c['name'] == name), None)
                    if cfg_for_name and len(portfolio_values) > 0:
                        cash_flows = pd.Series(0.0, index=portfolio_values.index)
                        # Initial investment: negative cash flow on first date
                        cash_flows.iloc[0] = -cfg_for_name.get('initial_value', 0)
                        # Periodic additions: negative cash flow on their respective dates
                        dates_added = get_dates_by_freq(cfg_for_name.get('added_frequency'), portfolio_values.index[0], portfolio_values.index[-1], portfolio_values.index)
                        for d in dates_added:
                            if d in cash_flows.index and d != cash_flows.index[0]:
                                cash_flows.loc[d] -= cfg_for_name.get('added_amount', 0)
                        # Final value: positive cash flow on last date for MWRR
                        cash_flows.iloc[-1] += portfolio_values.iloc[-1]
                        # Calculate MWRR
                        mwrr = calculate_mwrr(portfolio_values, cash_flows, portfolio_values.index)
                        mwrr_val = mwrr

                # Calculate total money added for this portfolio
                total_money_added = np.nan  # Use NaN instead of "N/A" string
                cfg_for_name = next((c for c in st.session_state.strategy_comparison_portfolio_configs if c['name'] == name), None)
                if cfg_for_name and isinstance(ser_noadd, pd.Series) and len(ser_noadd) > 0:
                    total_money_added = calculate_total_money_added(cfg_for_name, ser_noadd.index[0], ser_noadd.index[-1])
                
                # Calculate total return based on total money contributed
                total_return_contributed = np.nan  # Use NaN instead of "N/A" string
                if isinstance(series_obj, dict) and 'with_additions' in series_obj and len(series_obj['with_additions']) > 0:
                    final_value_with_additions = series_obj['with_additions'].iloc[-1]
                    if isinstance(total_money_added, (int, float)) and total_money_added > 0:
                        total_return_contributed = (final_value_with_additions / total_money_added - 1) * 100  # Return as percentage

                recomputed_stats[name] = {
                    "Total Return": clamp_stat(total_return, "Total Return"),
                    "Total Return (Contributed)": clamp_stat(total_return_contributed, "Total Return"),
                    "CAGR": clamp_stat(cagr, "CAGR"),
                    "MaxDrawdown": clamp_stat(max_dd, "MaxDrawdown"),
                    "Volatility": clamp_stat(vol, "Volatility"),
                    "Sharpe": clamp_stat(sharpe / 100 if isinstance(sharpe, (int, float)) and pd.notna(sharpe) else sharpe, "Sharpe"),
                    "Sortino": clamp_stat(sortino / 100 if isinstance(sortino, (int, float)) and pd.notna(sortino) else sortino, "Sortino"),
                    "UlcerIndex": clamp_stat(ulcer, "UlcerIndex"),
                    "UPI": clamp_stat(upi / 100 if isinstance(upi, (int, float)) and pd.notna(upi) else upi, "UPI"),
                    "Beta": clamp_stat(beta / 100 if isinstance(beta, (int, float)) and pd.notna(beta) else beta, "Beta"),
                    "MWRR": mwrr_val,
                    # Final values with and without additions
                    "Final Value (with)": (series_obj['with_additions'].iloc[-1] if isinstance(series_obj, dict) and 'with_additions' in series_obj and len(series_obj['with_additions'])>0 else np.nan),
                    "Final Value (no_additions)": (ser_noadd.iloc[-1] if isinstance(ser_noadd, pd.Series) and len(ser_noadd)>0 else np.nan),
                    "Total Money Added": total_money_added
                }

            stats_df_display = pd.DataFrame(recomputed_stats).T
            # Move final value columns to the front and format them as currency
            cols = list(stats_df_display.columns)
            fv_with = 'Final Value (with)'
            fv_no = 'Final Value (no_additions)'
            front = [c for c in [fv_with, fv_no] if c in cols]
            for c in front:
                cols.remove(c)
            cols = front + cols
            stats_df_display = stats_df_display[cols]
            # Rename and format columns to be more descriptive
            stats_df_display.rename(columns={
                'MaxDrawdown': 'Max Drawdown', 
                'UlcerIndex': 'Ulcer Index',
                'Final Value (with)': 'Final Portfolio Value',
                'Final Value (no_additions)': 'Final Value (No Contributions)',
                'Total Return (Contributed)': 'Total Return (All Money)'
            }, inplace=True)
            # Ensure ordering: MWRR after CAGR, Total Return columns after MWRR, then risk metrics, then Beta and Total Money Added at the end
            cols = list(stats_df_display.columns)
            if 'MWRR' in cols and 'Total Return' in cols and 'Total Return (All Money)' in cols and 'Beta' in cols and 'Total Money Added' in cols:
                # Remove the columns we want to reorder
                cols.remove('MWRR'); cols.remove('Total Return'); cols.remove('Total Return (All Money)'); cols.remove('Beta'); cols.remove('Total Money Added')
                
                # Find the position after CAGR to insert MWRR and Total Return columns
                cagr_index = cols.index('CAGR') if 'CAGR' in cols else 0
                insert_position = cagr_index + 1
                
                # Insert MWRR and Total Return columns after CAGR
                cols.insert(insert_position, 'MWRR')
                cols.insert(insert_position + 1, 'Total Return')
                cols.insert(insert_position + 2, 'Total Return (All Money)')
                
                # Add Beta and Total Money Added at the end
                cols.extend(['Beta', 'Total Money Added'])
                stats_df_display = stats_df_display[cols]

            # Display start and end dates next to the title
            col_title, col_dates = st.columns([2, 1])
            with col_title:
                st.subheader("Final Performance Statistics")
            with col_dates:
                if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
                    # Get the first portfolio's dates (they should all be the same)
                    first_portfolio = next(iter(st.session_state.strategy_comparison_all_results.values()))
                    if isinstance(first_portfolio, dict) and 'no_additions' in first_portfolio:
                        series = first_portfolio['no_additions']
                        if hasattr(series, 'index') and len(series.index) > 0:
                            start_date = series.index[0].strftime('%Y-%m-%d')
                            end_date = series.index[-1].strftime('%Y-%m-%d')
                            st.markdown(f"**📅 Period:** {start_date} to {end_date}")
                        else:
                            st.markdown("**📅 Period:** N/A")
                    else:
                        st.markdown("**📅 Period:** N/A")
                else:
                    st.markdown("**📅 Period:** N/A")
            # Format currency for final value columns if present
            fmt_map_display = {}
            if 'Final Portfolio Value' in stats_df_display.columns:
                fmt_map_display['Final Portfolio Value'] = '${:,.2f}'
            if 'Final Value (No Contributions)' in stats_df_display.columns:
                fmt_map_display['Final Value (No Contributions)'] = '${:,.2f}'
            if 'Total Money Added' in stats_df_display.columns:
                fmt_map_display['Total Money Added'] = '${:,.2f}'
            # Format MWRR as percentage - but only if it contains numeric data
            if 'MWRR' in stats_df_display.columns:
                # Check if MWRR column has any non-N/A values
                mwrr_has_numeric = False
                for value in stats_df_display['MWRR']:
                    if pd.notna(value) and value != 'N/A' and value != '':
                        try:
                            float(value)
                            mwrr_has_numeric = True
                            break
                        except (ValueError, TypeError):
                            pass
                if mwrr_has_numeric:
                    fmt_map_display['MWRR'] = '{:.2f}%'
            
            # Create tooltips for each column
            tooltip_data = {
                'Total Return': 'Return based on initial investment only. Formula: (Final Value / Initial Investment) - 1',
                'Total Return (All Money)': 'Return based on all money contributed. Formula: (Final Portfolio Value / Total Money Added) - 1',
                'CAGR': 'Compound Annual Growth Rate. Average annual return over the entire period.',
                'Max Drawdown': 'Largest peak-to-trough decline. Shows the worst loss from a peak.',
                'Volatility': 'Standard deviation of returns. Measures price variability.',
                'Sharpe': 'Excess return per unit of total volatility. >1 good, >2 very good, >3 excellent.',
                'Sortino': 'Excess return per unit of downside volatility. >1 good, >2 very good, >3 excellent.',
                'Ulcer Index': 'Average depth of drawdowns. <5 excellent, 5-10 moderate, >10 high.',
                'UPI': 'Ulcer Performance Index. Excess return relative to Ulcer Index. >1 good, >2 very good, >3 excellent.',
                'Beta': 'Portfolio volatility relative to benchmark. <1 less volatile, >1 more volatile than market.',
                'MWRR': 'Money-Weighted Rate of Return. Accounts for timing and size of cash flows.',
                'Final Portfolio Value': 'Final value including all contributions and investment returns.',
                'Final Value (No Contributions)': 'Final Value (No Contributions) - What $10,000 would grow to over the selected period using CAGR',
                'Total Money Added': 'Total amount of money contributed (initial + periodic additions).'
            }
            
            # Clean the dataframe to handle problematic values before styling
            stats_df_clean = stats_df_display.copy()
            
            # Replace problematic values that can't be formatted
            for col in stats_df_clean.columns:
                for idx in stats_df_clean.index:
                    value = stats_df_clean.loc[idx, col]
                    if pd.isna(value) or value is None or value == 'N/A' or value == '':
                        stats_df_clean.loc[idx, col] = 'N/A'
                    elif isinstance(value, str) and value.strip() == '':
                        stats_df_clean.loc[idx, col] = 'N/A'
            
            # Create a safe formatting map that only includes numeric columns
            safe_fmt_map = {}
            has_problematic_data = False
            
            # Check if any column has problematic data
            for col in stats_df_clean.columns:
                for value in stats_df_clean[col]:
                    if pd.isna(value) or value == 'N/A' or value == '':
                        has_problematic_data = True
                        break
                if has_problematic_data:
                    break
            
            # Only apply formatting if no problematic data
            if not has_problematic_data:
                for col, fmt in fmt_map_display.items():
                    if col in stats_df_clean.columns:
                        # Check if the column contains numeric data
                        numeric_count = 0
                        total_count = 0
                        for value in stats_df_clean[col]:
                            if pd.notna(value) and value != 'N/A' and value != '':
                                total_count += 1
                                try:
                                    float(value)
                                    numeric_count += 1
                                except (ValueError, TypeError):
                                    pass
                        
                        # Only apply formatting if most values are numeric
                        if total_count > 0 and numeric_count / total_count > 0.5:
                            safe_fmt_map[col] = fmt
            
            # Use sorted dataframe if available, otherwise use original
            sorted_df = st.session_state.get('strategy_comparison_final_stats_sorted_df', None)
            display_df = sorted_df if sorted_df is not None else stats_df_clean
            
            # Add tooltips to the dataframe
            if safe_fmt_map and not has_problematic_data:
                try:
                    styled_df = display_df.style.format(safe_fmt_map)
                except Exception as e:
                    styled_df = display_df
            else:
                # Skip styling entirely if there's problematic data
                styled_df = display_df
            
            # Add tooltips using HTML
            tooltip_html = "<div style='background-color: #1e1e1e; color: white; padding: 10px; border-radius: 5px; font-size: 12px;'>"
            tooltip_html += "<b>Column Definitions:</b><br><br>"
            for col, tooltip in tooltip_data.items():
                if col in display_df.columns:
                    tooltip_html += f"<b>{col}:</b> {tooltip}<br><br>"
            tooltip_html += "</div>"
            
            # Display tooltip info
            with st.expander("ℹ️ Column Definitions", expanded=False):
                st.markdown(tooltip_html, unsafe_allow_html=True)
            
            # Add sorting controls
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                sort_column = st.selectbox(
                    "Sort by:",
                    options=stats_df_clean.columns.tolist(),
                    index=0,
                    key="strategy_comparison_final_stats_sort_column",
                    help="Select a column to sort the table numerically"
                )
            with col2:
                if st.button("⬇️ Sort ↓", key="strategy_comparison_final_stats_sort_desc_button", help="Sort table in descending order (highest to lowest values)"):
                    sorted_df = sort_dataframe_numerically(stats_df_clean, sort_column, ascending=False)
                    st.session_state.strategy_comparison_final_stats_sorted_df = sorted_df
                    st.session_state.strategy_comparison_rerun_flag = True
            with col3:
                if st.button("⬆️ Sort ↑", key="strategy_comparison_final_stats_sort_asc_button", help="Sort table in ascending order (lowest to highest values)"):
                    sorted_df = sort_dataframe_numerically(stats_df_clean, sort_column, ascending=True)
                    st.session_state.strategy_comparison_final_stats_sorted_df = sorted_df
                    st.session_state.strategy_comparison_rerun_flag = True
            
            # Display the dataframe with multiple fallback options
            try:
                st.dataframe(styled_df, use_container_width=True)
            except Exception as e1:
                try:
                    st.dataframe(stats_df_clean, use_container_width=True)
                except Exception as e2:
                    # Last resort: convert all values to strings
                    stats_df_strings = stats_df_clean.astype(str)
                    st.dataframe(stats_df_strings, use_container_width=True)

            
            # Store the statistics table as a Plotly figure for PDF export - EXACT same as Multi-Backtest
            try:
                import plotly.graph_objects as go
                # Create a Plotly table from the stats DataFrame with MUCH better formatting
                # Format the values to remove excessive decimals with proper error handling
                formatted_values = []
                for col in stats_df_display.columns:
                    col_values = []
                    for name in stats_df_display.index:
                        value = stats_df_display.loc[name, col]
                        
                        # Handle NaN, None, and string values
                        if pd.isna(value) or value is None or value == 'N/A' or value == '':
                            col_values.append('N/A')
                            continue
                        
                        try:
                            if 'Portfolio Value' in col or 'Final Value' in col:
                                # Portfolio values: full numbers only (no decimals)
                                col_values.append(f"${float(value):,.0f}")
                            elif 'MWRR' in col:
                                # MWRR: max 2 decimal places
                                col_values.append(f"{float(value):.2f}%")
                            elif 'Total Money Added' in col:
                                # Money added: full numbers only
                                col_values.append(f"${float(value):,.0f}")
                            else:
                                # For other numeric columns, try to format as float
                                try:
                                    float_val = float(value)
                                    if 'Ratio' in col or 'Index' in col:
                                        col_values.append(f"{float_val:.3f}")
                                    elif 'Drawdown' in col or 'Volatility' in col or 'CAGR' in col:
                                        col_values.append(f"{float_val:.2f}%")
                                    else:
                                        col_values.append(f"{float_val:.2f}")
                                except (ValueError, TypeError):
                                    # If conversion fails, keep original value
                                    col_values.append(str(value))
                        except (ValueError, TypeError):
                            # If any conversion fails, use original value
                            col_values.append(str(value))
                    
                    formatted_values.append(col_values)
                
                fig_stats = go.Figure(data=[go.Table(
                    header=dict(
                        values=['Portfolio'] + list(stats_df_display.columns),
                        fill_color='rgb(51, 102, 153)',
                        align='center',
                        font=dict(color='white', size=14, family='Arial Black')  # Bigger, bolder font
                    ),
                    cells=dict(
                        values=[stats_df_display.index] + formatted_values,
                        fill_color='rgb(242, 242, 242)',
                        align='center',
                        font=dict(color='black', size=12, family='Arial'),  # Bigger font
                        height=35  # Taller cells for better readability
                    ),
                    columnwidth=[0.15] + [0.85/len(stats_df_display.columns)] * len(stats_df_display.columns)  # Portfolio column wider, others equal
                )])
                fig_stats.update_layout(
                    title="Final Performance Statistics",
                    title_x=0.5,
                    width=2000,  # Much wider to fit all columns
                    height=600,   # Taller for better spacing
                    margin=dict(l=20, r=20, t=50, b=20)  # Better margins
                )
                st.session_state.strategy_comparison_fig_stats = fig_stats
            except Exception as e:
                pass


        # Focused Performance Analysis with Date Range
        # Initialize session state variables first (using page-specific keys)
        if 'strategy_comparison_focused_analysis_results' not in st.session_state:
            st.session_state.strategy_comparison_focused_analysis_results = None
        if 'strategy_comparison_focused_analysis_show_essential' not in st.session_state:
            st.session_state.strategy_comparison_focused_analysis_show_essential = False
        if 'strategy_comparison_focused_analysis_period' not in st.session_state:
            st.session_state.strategy_comparison_focused_analysis_period = None
        if 'strategy_comparison_focused_analysis_start_date' not in st.session_state:
            st.session_state.strategy_comparison_focused_analysis_start_date = None
        if 'strategy_comparison_focused_analysis_end_date' not in st.session_state:
            st.session_state.strategy_comparison_focused_analysis_end_date = None
        if 'strategy_comparison_focused_analysis_sorted_df' not in st.session_state:
            st.session_state.strategy_comparison_focused_analysis_sorted_df = None
        if 'strategy_comparison_final_stats_sorted_df' not in st.session_state:
            st.session_state.strategy_comparison_final_stats_sorted_df = None
        
        # Date range controls
        col_date1, col_date2, col_metrics = st.columns([1, 1, 1])
        
        # Get the earliest and latest available dates from backtest data
        min_date = None
        max_date = None
        if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
            # Use the same reliable method as used elsewhere in the code
            try:
                first_date = min(series['no_additions'].index.min() for series in st.session_state.strategy_comparison_all_results.values() if 'no_additions' in series)
                last_date = max(series['no_additions'].index.max() for series in st.session_state.strategy_comparison_all_results.values() if 'no_additions' in series)
                min_date = first_date.date()
                max_date = last_date.date()
            except (ValueError, KeyError, AttributeError):
                # Fallback to the previous method if the above fails
                all_dates = []
                for portfolio_name, results in st.session_state.strategy_comparison_all_results.items():
                    if isinstance(results, dict) and 'no_additions' in results:
                        series = results['no_additions']
                        if hasattr(series, 'index') and len(series.index) > 0:
                            all_dates.extend(series.index.tolist())
                
                if all_dates:
                    min_date = min(all_dates).date()
                    max_date = max(all_dates).date()
        
        with col_date1:
            # Check if we need to update the date range due to new portfolio data
            current_min_date = min_date if min_date else datetime.date(1900, 1, 1)
            current_max_date = max_date if max_date else datetime.date.today()
            
            # Update session state if the available date range has changed
            if 'strategy_comparison_focused_analysis_available_min_date' not in st.session_state:
                st.session_state.strategy_comparison_focused_analysis_available_min_date = current_min_date
                st.session_state.strategy_comparison_focused_analysis_available_max_date = current_max_date
            elif (st.session_state.strategy_comparison_focused_analysis_available_min_date != current_min_date or 
                  st.session_state.strategy_comparison_focused_analysis_available_max_date != current_max_date):
                # Date range has changed, update session state and reset selected dates
                st.session_state.strategy_comparison_focused_analysis_available_min_date = current_min_date
                st.session_state.strategy_comparison_focused_analysis_available_max_date = current_max_date
                st.session_state.strategy_comparison_focused_analysis_start_date = current_min_date
                st.session_state.strategy_comparison_focused_analysis_end_date = current_max_date
            
            # Initialize start date in session state with fallback and validation
            if st.session_state.strategy_comparison_focused_analysis_start_date is None:
                st.session_state.strategy_comparison_focused_analysis_start_date = current_min_date
            
            # Ensure the start date is valid and within bounds
            start_date_value = st.session_state.strategy_comparison_focused_analysis_start_date
            if start_date_value is None:
                start_date_value = current_min_date
            else:
                # Convert to datetime.date if it's a pandas Timestamp or other datetime-like object
                try:
                    if hasattr(start_date_value, 'date'):
                        start_date_value = start_date_value.date()
                    elif not isinstance(start_date_value, datetime.date):
                        start_date_value = pd.to_datetime(start_date_value).date()
                except:
                    start_date_value = current_min_date
                
                # Check bounds after conversion
                if start_date_value < current_min_date:
                    start_date_value = current_min_date
                elif start_date_value > current_max_date:
                    start_date_value = current_max_date
            
            start_date = st.date_input(
                "Start Date", 
                value=start_date_value,
                min_value=current_min_date,
                max_value=current_max_date,
                key="focused_analysis_start_date_input",
                help="Start date for focused performance analysis"
            )
            
            # Store the selected start date in session state
            if start_date != st.session_state.strategy_comparison_focused_analysis_start_date:
                st.session_state.strategy_comparison_focused_analysis_start_date = start_date
        
        with col_date2:
            # Initialize end date in session state with fallback and validation
            if st.session_state.strategy_comparison_focused_analysis_end_date is None:
                st.session_state.strategy_comparison_focused_analysis_end_date = current_max_date
            
            # Ensure the end date is valid and within bounds
            end_date_value = st.session_state.strategy_comparison_focused_analysis_end_date
            if end_date_value is None:
                end_date_value = current_max_date
            else:
                # Convert to datetime.date if it's a pandas Timestamp or other datetime-like object
                try:
                    if hasattr(end_date_value, 'date'):
                        end_date_value = end_date_value.date()
                    elif not isinstance(end_date_value, datetime.date):
                        end_date_value = pd.to_datetime(end_date_value).date()
                except:
                    end_date_value = current_max_date
                
                # Check bounds after conversion
                if end_date_value < current_min_date:
                    end_date_value = current_min_date
                elif end_date_value > current_max_date:
                    end_date_value = current_max_date
            
            end_date = st.date_input(
                "End Date", 
                value=end_date_value,
                min_value=current_min_date,
                max_value=current_max_date,
                key="focused_analysis_end_date_input",
                help="End date for focused performance analysis"
            )
            
            # Store the selected end date in session state
            if end_date != st.session_state.strategy_comparison_focused_analysis_end_date:
                st.session_state.strategy_comparison_focused_analysis_end_date = end_date
        
        with col_metrics:
            show_essential_only = st.checkbox(
                "Essential Metrics Only", 
                value=False,
                help="Show only CAGR, Max Drawdown, Volatility, and Total Return for quick analysis"
            )
        
        # Add title after date inputs are defined
        st.subheader("📊 Focused Performance Analysis")
        
        # Calculate Analysis Button
        calculate_analysis = st.button(
            "📊 Calculate Analysis", 
            type="primary",
            use_container_width=True,
            help="Click to generate the focused performance analysis with your selected date range"
        )
        
        # Session state variables are already initialized above
        
        # Update session state when essential metrics checkbox changes
        if show_essential_only != st.session_state.strategy_comparison_focused_analysis_show_essential:
            st.session_state.strategy_comparison_focused_analysis_show_essential = show_essential_only
            # Recalculate if we have existing results
            if st.session_state.strategy_comparison_focused_analysis_results is not None:
                calculate_analysis = True
        
        # Calculate focused performance metrics only when button is clicked
        if calculate_analysis and start_date and end_date and 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
            focused_stats = {}
            
            for portfolio_name, results in st.session_state.strategy_comparison_all_results.items():
                if isinstance(results, dict) and 'no_additions' in results:
                    series = results['no_additions']
                    if hasattr(series, 'index') and len(series.index) > 0:
                        # Filter series by date range
                        # Convert dates to datetime for proper comparison
                        start_datetime = pd.to_datetime(start_date)
                        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # Include the end date
                        mask = (series.index >= start_datetime) & (series.index < end_datetime)
                        filtered_series = series[mask]
                        
                        if len(filtered_series) > 1:
                            # Calculate returns with ffill compatibility
                            returns = filtered_series.pct_change().fillna(0)
                            
                            if len(returns) > 0:
                                # Smart weekend filter for win/loss rates
                                zero_rate = (abs(returns) < 1e-5).mean()
                                if zero_rate > 0.25:  # If more than 25% are zero (likely weekends)
                                    # Filter to include only non-zero returns or returns on weekdays
                                    weekday_mask = returns.index.weekday < 5  # Monday=0, Friday=4
                                    non_zero_mask = abs(returns) > 1e-5
                                    smart_mask = weekday_mask | non_zero_mask
                                    returns = returns[smart_mask]
                                
                                # Calculate metrics for the date range
                                cagr = calculate_cagr(filtered_series, filtered_series.index)
                                
                                # Calculate volatility using the SAME method as Final Performance Statistics
                                # Use the original returns (before smart filtering) like Final Performance Statistics does
                                original_returns = filtered_series.pct_change().fillna(0)
                                volatility = calculate_volatility(original_returns)
                                
                                # Use original returns for Sharpe and Sortino (same as Final Performance Statistics)
                                sharpe = calculate_sharpe(original_returns, 0.02)  # 2% risk-free rate
                                sortino = calculate_sortino(original_returns, 0.02)
                                
                                # Calculate max drawdown using original returns (like Final Performance Statistics)
                                cumulative = (1 + original_returns).cumprod()
                                running_max = cumulative.expanding().max()
                                drawdown = (cumulative - running_max) / running_max
                                max_drawdown = drawdown.min()
                                
                                # Calculate Ulcer Index using original returns (like Final Performance Statistics)
                                ulcer_index = np.sqrt((drawdown ** 2).mean()) * 100
                                
                                # Calculate UPI
                                upi = calculate_upi(cagr, ulcer_index) if ulcer_index > 0 else np.nan
                                
                                # Calculate Beta (same as Final Performance Statistics)
                                beta = np.nan
                                # Find the portfolio config to get benchmark ticker
                                cfg_for_name = next((c for c in st.session_state.strategy_comparison_portfolio_configs if c['name'] == portfolio_name), None)
                                if cfg_for_name:
                                    bench_ticker = cfg_for_name.get('benchmark_ticker')
                                    raw_data = st.session_state.get('strategy_comparison_raw_data')
                                    if bench_ticker and raw_data and bench_ticker in raw_data:
                                        # get benchmark price_change series aligned to filtered_series index
                                        try:
                                            bench_df = raw_data[bench_ticker].reindex(filtered_series.index)
                                            if 'Price_change' in bench_df.columns:
                                                bench_returns = bench_df['Price_change'].fillna(0)
                                            else:
                                                bench_returns = bench_df['Close'].pct_change().fillna(0)

                                            portfolio_returns = original_returns  # Use original returns like Final Performance Statistics
                                            common_idx = portfolio_returns.index.intersection(bench_returns.index)
                                            if len(common_idx) >= 2:
                                                pr = portfolio_returns.reindex(common_idx).dropna()
                                                br = bench_returns.reindex(common_idx).dropna()
                                                common_idx2 = pr.index.intersection(br.index)
                                                if len(common_idx2) >= 2 and br.loc[common_idx2].var() != 0:
                                                    cov = pr.loc[common_idx2].cov(br.loc[common_idx2])
                                                    var = br.loc[common_idx2].var()
                                                    beta = cov / var
                                        except Exception as e:
                                            pass
                                
                                # Calculate additional instantaneous metrics
                                total_return = (filtered_series.iloc[-1] / filtered_series.iloc[0] - 1)
                                final_value = filtered_series.iloc[-1]
                                # For no contributions, start with $10,000 and apply CAGR
                                years = (filtered_series.index[-1] - filtered_series.index[0]).days / 365.25
                                cagr = calculate_cagr(filtered_series, filtered_series.index)
                                final_value_no_contrib = 10000 * ((1 + cagr) ** years)
                                total_money_added = 0  # Not available in no_additions series
                                
                                # Calculate median drawdown
                                median_drawdown = drawdown.median()
                                
                                # Calculate win/loss rates
                                positive_returns = returns[returns > 1e-5]
                                negative_returns = returns[returns < -1e-5]
                                win_rate = (len(positive_returns) / len(returns)) * 100 if len(returns) > 0 else 0
                                loss_rate = (len(negative_returns) / len(returns)) * 100 if len(returns) > 0 else 0
                                
                                # Calculate median win/loss
                                median_win = positive_returns.median() * 100 if len(positive_returns) > 0 else 0
                                median_loss = negative_returns.median() * 100 if len(negative_returns) > 0 else 0
                                
                                # Calculate profit factor
                                gross_profit = positive_returns.sum() if len(positive_returns) > 0 else 0
                                gross_loss = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0
                                profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
                                
                                # Calculate monthly returns for monthly metrics
                                monthly_returns = filtered_series.resample('M').last().pct_change().fillna(0) * 100
                                best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
                                worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
                                median_monthly = monthly_returns.median() if len(monthly_returns) > 0 else 0
                                
                                # Calculate risk-adjusted ratios
                                calmar_ratio = (cagr * 100) / abs(max_drawdown * 100) if max_drawdown != 0 else np.nan
                                sterling_ratio = (cagr * 100) / abs(median_drawdown * 100) if median_drawdown != 0 else np.nan
                                recovery_factor = abs(total_return) / abs(max_drawdown) if max_drawdown != 0 else np.nan
                                
                                # Calculate tail ratio (95th percentile / 5th percentile)
                                tail_ratio = returns.quantile(0.95) / abs(returns.quantile(0.05)) if returns.quantile(0.05) != 0 else np.nan
                                
                                if show_essential_only:
                                    focused_stats[portfolio_name] = {
                                        'Total Return': total_return * 100,
                                        'CAGR': cagr * 100,
                                        'Max Drawdown': max_drawdown * 100,
                                        'Volatility': volatility * 100
                                    }
                                else:
                                    focused_stats[portfolio_name] = {
                                        'Total Return': total_return * 100,
                                        'CAGR': cagr * 100,
                                        'Max Drawdown': max_drawdown * 100,
                                        'Volatility': volatility * 100,
                                        'Sharpe': sharpe,
                                        'Sortino': sortino,
                                        'Ulcer Index': ulcer_index,
                                        'UPI': upi,
                                        'Beta': beta,
                                        'Final Value (No Contributions)': final_value_no_contrib,
                                        'Median Drawdown': median_drawdown * 100,
                                        'Win Rate': win_rate,
                                        'Loss Rate': loss_rate,
                                        'Median Win': median_win,
                                        'Median Loss': median_loss,
                                        'Profit Factor': profit_factor,
                                        'Best Month': best_month,
                                        'Worst Month': worst_month,
                                        'Median Monthly': median_monthly,
                                        'Calmar Ratio': calmar_ratio,
                                        'Sterling Ratio': sterling_ratio,
                                        'Recovery Factor': recovery_factor,
                                        'Tail Ratio': tail_ratio
                                    }
            
            if focused_stats:
                # Store results in session state
                st.session_state.strategy_comparison_focused_analysis_results = focused_stats
                st.session_state.strategy_comparison_focused_analysis_show_essential = show_essential_only
                st.session_state.strategy_comparison_focused_analysis_period = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                # Clear any previous sorting when new results are calculated
                st.session_state.strategy_comparison_focused_analysis_sorted_df = None
                
                # Create focused stats DataFrame
                focused_df = pd.DataFrame.from_dict(focused_stats, orient='index')
                focused_df.index.name = 'Portfolio'
                
                # Format the DataFrame
                if show_essential_only:
                    focused_df['CAGR'] = focused_df['CAGR'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Max Drawdown'] = focused_df['Max Drawdown'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Volatility'] = focused_df['Volatility'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Total Return'] = focused_df['Total Return'].apply(lambda x: f"{x:.2f}%")
                else:
                    # Format all columns
                    focused_df['CAGR'] = focused_df['CAGR'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Max Drawdown'] = focused_df['Max Drawdown'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Volatility'] = focused_df['Volatility'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Sharpe'] = focused_df['Sharpe'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                    focused_df['Sortino'] = focused_df['Sortino'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                    focused_df['Ulcer Index'] = focused_df['Ulcer Index'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                    focused_df['UPI'] = focused_df['UPI'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                    focused_df['Beta'] = focused_df['Beta'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                    focused_df['Total Return'] = focused_df['Total Return'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Final Value (No Contributions)'] = focused_df['Final Value (No Contributions)'].apply(lambda x: f"${x:,.2f}")
                    focused_df['Median Drawdown'] = focused_df['Median Drawdown'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Win Rate'] = focused_df['Win Rate'].apply(lambda x: f"{x:.1f}%")
                    focused_df['Loss Rate'] = focused_df['Loss Rate'].apply(lambda x: f"{x:.1f}%")
                    focused_df['Median Win'] = focused_df['Median Win'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Median Loss'] = focused_df['Median Loss'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Profit Factor'] = focused_df['Profit Factor'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) and x != np.inf else "N/A")
                    focused_df['Best Month'] = focused_df['Best Month'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Worst Month'] = focused_df['Worst Month'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Median Monthly'] = focused_df['Median Monthly'].apply(lambda x: f"{x:.2f}%")
                    focused_df['Calmar Ratio'] = focused_df['Calmar Ratio'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                    focused_df['Sterling Ratio'] = focused_df['Sterling Ratio'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                    focused_df['Recovery Factor'] = focused_df['Recovery Factor'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                    focused_df['Tail Ratio'] = focused_df['Tail Ratio'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                
                # Add column definitions expander
                focused_tooltip_data = {
                    'Final Value (No Contributions)': 'Final Value (No Contributions) - What $10,000 would grow to over the selected period using CAGR',
                    'CAGR': 'Compound Annual Growth Rate - The annualized rate of return over the investment period',
                    'Max Drawdown': 'Maximum Drawdown - The largest peak-to-trough decline during the period',
                    'Volatility': 'Volatility - Standard deviation of returns, measuring price dispersion',
                    'Sharpe Ratio': 'Sharpe Ratio - Risk-adjusted return (excess return per unit of volatility)',
                    'Sortino Ratio': 'Sortino Ratio - Risk-adjusted return considering only downside volatility',
                    'Calmar Ratio': 'Calmar Ratio - Annual return divided by maximum drawdown',
                    'Sterling Ratio': 'Sterling Ratio - Average annual return divided by average drawdown',
                    'Recovery Factor': 'Recovery Factor - Net profit divided by maximum drawdown',
                    'Tail Ratio': 'Tail Ratio - 95th percentile return divided by 5th percentile return',
                    'Win Rate': 'Win Rate - Percentage of positive return periods',
                    'Loss Rate': 'Loss Rate - Percentage of negative return periods',
                    'Median Win': 'Median Win - Median return of positive periods',
                    'Median Loss': 'Median Loss - Median return of negative periods',
                    'Profit Factor': 'Profit Factor - Gross profit divided by gross loss',
                    'Best Month': 'Best Month - Highest single month return',
                    'Worst Month': 'Worst Month - Lowest single month return',
                    'Median Monthly': 'Median Monthly - Median monthly return',
                    'Median Drawdown': 'Median Drawdown - Median drawdown value'
                }
                
                # Create tooltip HTML
                focused_tooltip_html = "<div style='background-color: #1e1e1e; color: white; padding: 10px; border-radius: 5px; font-size: 12px;'>"
                focused_tooltip_html += "<b>Column Definitions:</b><br><br>"
                for col in focused_df.columns:
                    if col in focused_tooltip_data:
                        focused_tooltip_html += f"<b>{col}:</b> {focused_tooltip_data[col]}<br><br>"
                focused_tooltip_html += "</div>"
                
                # Display tooltip info
                with st.expander("ℹ️ Column Definitions", expanded=False):
                    st.markdown(focused_tooltip_html, unsafe_allow_html=True)
                
                # Add sorting controls for focused analysis
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    focused_sort_column = st.selectbox(
                        "Sort by:",
                        options=focused_df.columns.tolist(),
                        index=0,
                        key="strategy_comparison_focused_analysis_sort_column",
                        help="Select a column to sort the table numerically"
                    )
                with col2:
                    if st.button("⬇️ Sort ↓", key="strategy_comparison_focused_analysis_sort_desc_button", help="Sort table in descending order (highest to lowest values)"):
                        sorted_df = sort_dataframe_numerically(focused_df, focused_sort_column, ascending=False)
                        st.session_state.strategy_comparison_focused_analysis_sorted_df = sorted_df
                        st.session_state.strategy_comparison_rerun_flag = True
                with col3:
                    if st.button("⬆️ Sort ↑", key="strategy_comparison_focused_analysis_sort_asc_button", help="Sort table in ascending order (lowest to highest values)"):
                        sorted_df = sort_dataframe_numerically(focused_df, focused_sort_column, ascending=True)
                        st.session_state.strategy_comparison_focused_analysis_sorted_df = sorted_df
                        st.session_state.strategy_comparison_rerun_flag = True
                
                # Display the focused table (use sorted version if available)
                sorted_df = st.session_state.get('strategy_comparison_focused_analysis_sorted_df', None)
                display_df = sorted_df if sorted_df is not None else focused_df
                st.dataframe(display_df, use_container_width=True)
                
                # Show date range info
                st.info(f"📅 **Analysis Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            else:
                st.warning("⚠️ No data available for the selected date range. Please check your date selection.")
        elif st.session_state.strategy_comparison_focused_analysis_results is not None:
            # Display stored results even when not recalculating
            focused_stats = st.session_state.strategy_comparison_focused_analysis_results
            show_essential = st.session_state.strategy_comparison_focused_analysis_show_essential
            
            # Create focused stats DataFrame
            focused_df = pd.DataFrame.from_dict(focused_stats, orient='index')
            focused_df.index.name = 'Portfolio'
            
            # Format the DataFrame based on current essential setting
            if show_essential_only:
                # Filter to only show essential columns
                essential_cols = ['Total Return', 'CAGR', 'Max Drawdown', 'Volatility']
                if all(col in focused_df.columns for col in essential_cols):
                    focused_df = focused_df[essential_cols]
                focused_df['Total Return'] = focused_df['Total Return'].apply(lambda x: f"{x:.2f}%")
                focused_df['CAGR'] = focused_df['CAGR'].apply(lambda x: f"{x:.2f}%")
                focused_df['Max Drawdown'] = focused_df['Max Drawdown'].apply(lambda x: f"{x:.2f}%")
                focused_df['Volatility'] = focused_df['Volatility'].apply(lambda x: f"{x:.2f}%")
            else:
                # Format all columns
                focused_df['CAGR'] = focused_df['CAGR'].apply(lambda x: f"{x:.2f}%")
                focused_df['Max Drawdown'] = focused_df['Max Drawdown'].apply(lambda x: f"{x:.2f}%")
                focused_df['Volatility'] = focused_df['Volatility'].apply(lambda x: f"{x:.2f}%")
                if 'Sharpe' in focused_df.columns:
                    focused_df['Sharpe'] = focused_df['Sharpe'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                if 'Sortino' in focused_df.columns:
                    focused_df['Sortino'] = focused_df['Sortino'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                if 'Ulcer Index' in focused_df.columns:
                    focused_df['Ulcer Index'] = focused_df['Ulcer Index'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                if 'UPI' in focused_df.columns:
                    focused_df['UPI'] = focused_df['UPI'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                if 'Beta' in focused_df.columns:
                    focused_df['Beta'] = focused_df['Beta'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                if 'Total Return' in focused_df.columns:
                    focused_df['Total Return'] = focused_df['Total Return'].apply(lambda x: f"{x:.2f}%")
                if 'Final Value (No Contributions)' in focused_df.columns:
                    focused_df['Final Value (No Contributions)'] = focused_df['Final Value (No Contributions)'].apply(lambda x: f"${x:,.2f}")
                if 'Median Drawdown' in focused_df.columns:
                    focused_df['Median Drawdown'] = focused_df['Median Drawdown'].apply(lambda x: f"{x:.2f}%")
                if 'Win Rate' in focused_df.columns:
                    focused_df['Win Rate'] = focused_df['Win Rate'].apply(lambda x: f"{x:.1f}%")
                if 'Loss Rate' in focused_df.columns:
                    focused_df['Loss Rate'] = focused_df['Loss Rate'].apply(lambda x: f"{x:.1f}%")
                if 'Median Win' in focused_df.columns:
                    focused_df['Median Win'] = focused_df['Median Win'].apply(lambda x: f"{x:.2f}%")
                if 'Median Loss' in focused_df.columns:
                    focused_df['Median Loss'] = focused_df['Median Loss'].apply(lambda x: f"{x:.2f}%")
                if 'Profit Factor' in focused_df.columns:
                    focused_df['Profit Factor'] = focused_df['Profit Factor'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) and x != np.inf else "N/A")
                if 'Best Month' in focused_df.columns:
                    focused_df['Best Month'] = focused_df['Best Month'].apply(lambda x: f"{x:.2f}%")
                if 'Worst Month' in focused_df.columns:
                    focused_df['Worst Month'] = focused_df['Worst Month'].apply(lambda x: f"{x:.2f}%")
                if 'Median Monthly' in focused_df.columns:
                    focused_df['Median Monthly'] = focused_df['Median Monthly'].apply(lambda x: f"{x:.2f}%")
                if 'Calmar Ratio' in focused_df.columns:
                    focused_df['Calmar Ratio'] = focused_df['Calmar Ratio'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                if 'Sterling Ratio' in focused_df.columns:
                    focused_df['Sterling Ratio'] = focused_df['Sterling Ratio'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                if 'Recovery Factor' in focused_df.columns:
                    focused_df['Recovery Factor'] = focused_df['Recovery Factor'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                if 'Tail Ratio' in focused_df.columns:
                    focused_df['Tail Ratio'] = focused_df['Tail Ratio'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            
            # Add column definitions expander
            focused_tooltip_data = {
                'Final Value (No Contributions)': 'Final Value (No Contributions) - What $10,000 would grow to over the selected period using CAGR',
                'CAGR': 'Compound Annual Growth Rate - The annualized rate of return over the investment period',
                'Max Drawdown': 'Maximum Drawdown - The largest peak-to-trough decline during the period',
                'Volatility': 'Volatility - Standard deviation of returns, measuring price dispersion',
                'Sharpe Ratio': 'Sharpe Ratio - Risk-adjusted return (excess return per unit of volatility)',
                'Sortino Ratio': 'Sortino Ratio - Risk-adjusted return considering only downside volatility',
                'Calmar Ratio': 'Calmar Ratio - Annual return divided by maximum drawdown',
                'Sterling Ratio': 'Sterling Ratio - Average annual return divided by average drawdown',
                'Recovery Factor': 'Recovery Factor - Net profit divided by maximum drawdown',
                'Tail Ratio': 'Tail Ratio - 95th percentile return divided by 5th percentile return',
                'Win Rate': 'Win Rate - Percentage of positive return periods',
                'Loss Rate': 'Loss Rate - Percentage of negative return periods',
                'Median Win': 'Median Win - Median return of positive periods',
                'Median Loss': 'Median Loss - Median return of negative periods',
                'Profit Factor': 'Profit Factor - Gross profit divided by gross loss',
                'Best Month': 'Best Month - Highest single month return',
                'Worst Month': 'Worst Month - Lowest single month return',
                'Median Monthly': 'Median Monthly - Median monthly return',
                'Median Drawdown': 'Median Drawdown - Median drawdown value'
            }
            
            # Create tooltip HTML
            focused_tooltip_html = "<div style='background-color: #1e1e1e; color: white; padding: 10px; border-radius: 5px; font-size: 12px;'>"
            focused_tooltip_html += "<b>Column Definitions:</b><br><br>"
            for col in focused_df.columns:
                if col in focused_tooltip_data:
                    focused_tooltip_html += f"<b>{col}:</b> {focused_tooltip_data[col]}<br><br>"
            focused_tooltip_html += "</div>"
            
            # Display tooltip info
            with st.expander("ℹ️ Column Definitions", expanded=False):
                st.markdown(focused_tooltip_html, unsafe_allow_html=True)
            
            # Add sorting controls for focused analysis
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                focused_sort_column = st.selectbox(
                    "Sort by:",
                    options=focused_df.columns.tolist(),
                    index=0,
                    key="strategy_comparison_focused_analysis_sort_column_2",
                    help="Select a column to sort the table numerically"
                )
            with col2:
                if st.button("⬇️ Sort ↓", key="strategy_comparison_focused_analysis_sort_desc_button_2", help="Sort table in descending order (highest to lowest values)"):
                    sorted_df = sort_dataframe_numerically(focused_df, focused_sort_column, ascending=False)
                    st.session_state.strategy_comparison_focused_analysis_sorted_df = sorted_df
                    st.session_state.strategy_comparison_rerun_flag = True
            with col3:
                if st.button("⬆️ Sort ↑", key="strategy_comparison_focused_analysis_sort_asc_button_2", help="Sort table in ascending order (lowest to highest values)"):
                    sorted_df = sort_dataframe_numerically(focused_df, focused_sort_column, ascending=True)
                    st.session_state.strategy_comparison_focused_analysis_sorted_df = sorted_df
                    st.session_state.strategy_comparison_rerun_flag = True
            
            # Display the focused analysis table (use sorted version if available)
            sorted_df = st.session_state.get('strategy_comparison_focused_analysis_sorted_df', None)
            display_df = sorted_df if sorted_df is not None else focused_df
            st.dataframe(display_df, use_container_width=True)
            
            # Show date range info
            if st.session_state.strategy_comparison_focused_analysis_period is not None:
                st.info(f"📅 **Analysis Period:** {st.session_state.strategy_comparison_focused_analysis_period}")
        elif not calculate_analysis:
            st.info("👆 **Select your date range above and click 'Calculate Analysis' to generate the focused performance metrics.**")
        else:
            st.warning("⚠️ Please ensure you have portfolio data loaded and valid date range selected.")

        # Portfolio Configuration Comparison Table
        st.subheader("Portfolio Configuration Comparison")
        
        # Create configuration comparison dataframe
        config_data = {}
        for cfg in st.session_state.strategy_comparison_portfolio_configs:
            portfolio_name = cfg.get('name', 'Unknown')
            
            # Extract configuration details
            config_data[portfolio_name] = {
                'Initial Investment': f"${cfg.get('initial_value', 0):,.2f}",
                'Added Amount': f"${cfg.get('added_amount', 0):,.2f}",
                'Added Frequency': cfg.get('added_frequency', 'None'),
                'Rebalancing Frequency': cfg.get('rebalancing_frequency', 'None'),
                'Use Momentum': 'Yes' if cfg.get('use_momentum', False) else 'No',
                'Momentum Strategy': cfg.get('momentum_strategy', 'N/A'),
                'Negative Momentum Strategy': cfg.get('negative_momentum_strategy', 'N/A'),
                'Number of Stocks': len(cfg.get('stocks', [])),
                'Stocks': ', '.join([s.get('ticker', '') for s in cfg.get('stocks', [])]),
                'Benchmark': cfg.get('benchmark_ticker', 'N/A'),
                'Momentum Windows': str(cfg.get('momentum_windows', [])),
                'Beta Enabled': 'Yes' if cfg.get('calc_beta', False) else 'No',
                'Volatility Enabled': 'Yes' if cfg.get('calc_volatility', False) else 'No',
                'Beta Window': f"{cfg.get('beta_window_days', 0)} days" if cfg.get('calc_beta', False) else 'N/A',
                'Volatility Window': f"{cfg.get('vol_window_days', 0)} days" if cfg.get('calc_volatility', False) else 'N/A',
                'Beta Exclude Days': f"{cfg.get('exclude_days_beta', 0)} days" if cfg.get('calc_beta', False) else 'N/A',
                'Volatility Exclude Days': f"{cfg.get('exclude_days_vol', 0)} days" if cfg.get('calc_volatility', False) else 'N/A',
                'Minimal Threshold': f"{cfg.get('minimal_threshold_percent', 2.0):.1f}%" if cfg.get('use_minimal_threshold', False) else 'Disabled',
                'Maximum Allocation': f"{cfg.get('max_allocation_percent', 10.0):.1f}%" if cfg.get('use_max_allocation', False) else 'Disabled'
            }
        
        config_df = pd.DataFrame(config_data).T
        
        # Format the configuration table
        st.dataframe(config_df, use_container_width=True)

        st.subheader("Yearly Performance (Interactive Table)")
        all_years = st.session_state.strategy_comparison_all_years
        years = sorted(list(set(y.year for ser in all_years.values() for y in ser.index)))
        # Order portfolio columns according to the portfolio_configs order so new portfolios are added to the right
        names = [cfg['name'] for cfg in st.session_state.strategy_comparison_portfolio_configs if cfg.get('name') in all_years]

        # Corrected yearly table creation
        df_yearly_pct_data = {}
        df_yearly_final_data = {}
        for name in names:
            pct_list = []
            final_list = []
            # with-additions yearly series (used for final values)
            ser_with = all_years.get(name) if isinstance(all_years, dict) else None
            # no-additions yearly series (used for percent-change to avoid skew)
            ser_noadd = None
            try:
                series_obj = st.session_state.strategy_comparison_all_results.get(name)
                if isinstance(series_obj, dict) and 'no_additions' in series_obj:
                    ser_noadd = series_obj['no_additions'].resample('YE').last()
                elif isinstance(series_obj, pd.Series):
                    ser_noadd = series_obj.resample('YE').last()
            except Exception:
                ser_noadd = None

            for y in years:
                # get year slices
                ser_year_with = ser_with[ser_with.index.year == y] if ser_with is not None else pd.Series()
                ser_year_no = ser_noadd[ser_noadd.index.year == y] if ser_noadd is not None else pd.Series()

                start_val_for_year = None
                if y == min(years):
                    config_for_name = next((c for c in st.session_state.strategy_comparison_portfolio_configs if c['name'] == name), None)
                    if config_for_name:
                        initial_val_of_config = config_for_name['initial_value']
                        if initial_val_of_config > 0:
                            start_val_for_year = initial_val_of_config
                else:
                    prev_year = y - 1
                    # Use no-additions previous year end as the start value for pct change
                    prev_ser_year_no = ser_noadd[ser_noadd.index.year == prev_year] if ser_noadd is not None else pd.Series()
                    if not prev_ser_year_no.empty:
                        start_val_for_year = prev_ser_year_no.iloc[-1]

                # Percent change computed from no-additions series
                if not ser_year_no.empty and start_val_for_year is not None:
                    end_val_no = ser_year_no.iloc[-1]
                    if start_val_for_year > 0:
                        pct_change = (end_val_no - start_val_for_year) / start_val_for_year * 100
                    else:
                        pct_change = np.nan
                else:
                    pct_change = np.nan

                # Final value displayed from with-additions series (if available)
                if not ser_year_with.empty:
                    final_value = ser_year_with.iloc[-1]
                else:
                    final_value = np.nan

                pct_list.append(pct_change)
                final_list.append(final_value)

            df_yearly_pct_data[f'{name} % Change'] = pct_list
            df_yearly_final_data[f'{name} Final Value'] = final_list

        df_yearly_pct = pd.DataFrame(df_yearly_pct_data, index=years)
        df_yearly_final = pd.DataFrame(df_yearly_final_data, index=years)
        # Build combined dataframe but preserve the desired column order (selected portfolio first)
        temp_combined = pd.concat([df_yearly_pct, df_yearly_final], axis=1)
        ordered_cols = []
        for nm in names:
            pct_col = f'{nm} % Change'
            val_col = f'{nm} Final Value'
            if pct_col in temp_combined.columns:
                ordered_cols.append(pct_col)
            if val_col in temp_combined.columns:
                ordered_cols.append(val_col)
        # Fallback: if nothing matched, use whatever columns exist
        if not ordered_cols:
            combined_df = temp_combined
        else:
            combined_df = temp_combined[ordered_cols]

        def color_gradient_stock(val):
            if isinstance(val, (int, float)):
                if val > 50:
                    return 'background-color: #004d00'
                elif val > 20:
                    return 'background-color: #1e8449'
                elif val > 5:
                    return 'background-color: #388e3c'
                elif val > 0:
                    return 'background-color: #66bb6a'
                elif val < -50:
                    return 'background-color: #7b0000'
                elif val < -20:
                    return 'background-color: #b22222'
                elif val < -5:
                    return 'background-color: #d32f2f'
                elif val < 0:
                    return 'background-color: #ef5350'
            return ''
        
        # Ensure columns and index are unique (pandas Styler requires unique labels)
        if combined_df.columns.duplicated().any():
            cols = list(combined_df.columns)
            seen = {}
            new_cols = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c} ({seen[c]})")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            combined_df.columns = new_cols

        if combined_df.index.duplicated().any():
            idx = list(map(str, combined_df.index))
            seen_idx = {}
            new_idx = []
            for v in idx:
                if v in seen_idx:
                    seen_idx[v] += 1
                    new_idx.append(f"{v} ({seen_idx[v]})")
                else:
                    seen_idx[v] = 0
                    new_idx.append(v)
            combined_df.index = new_idx

        # Recompute percent and final value column lists after any renaming
        pct_cols = [col for col in combined_df.columns if '% Change' in col]
        final_val_cols = [col for col in combined_df.columns if 'Final Value' in col]

        # Coerce percent columns to numeric so formatting applies correctly
        for col in pct_cols:
            if col in combined_df.columns:
                try:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                except TypeError:
                    # Unexpected column type (not Series/array). Try to coerce via pd.Series or fall back to NaN.
                    try:
                        combined_df[col] = pd.to_numeric(pd.Series(combined_df[col]), errors='coerce')
                    except Exception:
                        combined_df[col] = np.nan

        # Create combined format mapping: percent columns get '%' suffix, final value columns get currency
        fmt_map = {col: '{:,.2f}%' for col in pct_cols if col in combined_df.columns}
        fmt_map.update({col: '${:,.2f}' for col in final_val_cols if col in combined_df.columns})

        styler = combined_df.style
        # Color percent cells with a gradient and then apply formatting in one call
        if pct_cols:
            try:
                # Styler.map is the supported replacement for applymap
                styler = styler.map(color_gradient_stock, subset=pct_cols)
            except Exception:
                # If map still fails (edge cases), skip coloring to avoid breaking the page
                pass
        if fmt_map:
            styler = styler.format(fmt_map, na_rep='N/A')

        st.dataframe(styler, use_container_width=True, hide_index=False)

        # Monthly Performance Table
        st.subheader("Monthly Performance (Interactive Table)")
        # Use the original results data for monthly calculation, not the yearly resampled data
        all_results = st.session_state.strategy_comparison_all_results
        # Get all available months from the original data
        all_months_data = {}
        for name, results in all_results.items():
            if isinstance(results, dict) and 'with_additions' in results:
                all_months_data[name] = results['with_additions']
            elif isinstance(results, pd.Series):
                all_months_data[name] = results
        
        # Extract all unique year-month combinations from the original data
        months = set()
        for ser in all_months_data.values():
            if not ser.empty:
                for date in ser.index:
                    months.add((date.year, date.month))
        months = sorted(list(months))
        
        # Order portfolio columns according to the portfolio_configs order so new portfolios are added to the right
        names = [cfg['name'] for cfg in st.session_state.strategy_comparison_portfolio_configs if cfg.get('name') in all_months_data]

        # Monthly table creation
        df_monthly_pct_data = {}
        df_monthly_final_data = {}
        for name in names:
            pct_list = []
            final_list = []
            # with-additions monthly series (used for final values)
            ser_with = all_months_data.get(name) if isinstance(all_months_data, dict) else None
            # no-additions monthly series (used for percent-change to avoid skew)
            ser_noadd = None
            try:
                series_obj = st.session_state.strategy_comparison_all_results.get(name)
                if isinstance(series_obj, dict) and 'no_additions' in series_obj:
                    ser_noadd = series_obj['no_additions'].resample('ME').last()
                elif isinstance(series_obj, pd.Series):
                    ser_noadd = series_obj.resample('ME').last()
            except Exception:
                ser_noadd = None

            for y, m in months:
                # get month slices
                ser_month_with = ser_with[(ser_with.index.year == y) & (ser_with.index.month == m)] if ser_with is not None else pd.Series()
                ser_month_no = ser_noadd[(ser_noadd.index.year == y) & (ser_noadd.index.month == m)] if ser_noadd is not None else pd.Series()

                start_val_for_month = None
                if (y, m) == min(months):
                    config_for_name = next((c for c in st.session_state.strategy_comparison_portfolio_configs if c['name'] == name), None)
                    if config_for_name:
                        initial_val_of_config = config_for_name['initial_value']
                        if initial_val_of_config > 0:
                            start_val_for_month = initial_val_of_config
                else:
                    # Find previous month
                    prev_month_idx = months.index((y, m)) - 1
                    if prev_month_idx >= 0:
                        prev_y, prev_m = months[prev_month_idx]
                        # Use no-additions previous month end as the start value for pct change
                        prev_ser_month_no = ser_noadd[(ser_noadd.index.year == prev_y) & (ser_noadd.index.month == prev_m)] if ser_noadd is not None else pd.Series()
                        if not prev_ser_month_no.empty:
                            start_val_for_month = prev_ser_month_no.iloc[-1]

                # Percent change computed from no-additions series
                if not ser_month_no.empty and start_val_for_month is not None:
                    end_val_no = ser_month_no.iloc[-1]
                    if start_val_for_month > 0:
                        pct_change = (end_val_no - start_val_for_month) / start_val_for_month * 100
                    else:
                        pct_change = np.nan
                else:
                    pct_change = np.nan

                # Final value displayed from with-additions series (if available)
                if not ser_month_with.empty:
                    final_value = ser_month_with.iloc[-1]
                else:
                    final_value = np.nan

                pct_list.append(pct_change)
                final_list.append(final_value)

            df_monthly_pct_data[f'{name} % Change'] = pct_list
            df_monthly_final_data[f'{name} Final Value'] = final_list

        df_monthly_pct = pd.DataFrame(df_monthly_pct_data, index=[f"{y}-{m:02d}" for y, m in months])
        df_monthly_final = pd.DataFrame(df_monthly_final_data, index=[f"{y}-{m:02d}" for y, m in months])
        # Build combined dataframe but preserve the desired column order (selected portfolio first)
        temp_combined_monthly = pd.concat([df_monthly_pct, df_monthly_final], axis=1)
        ordered_cols_monthly = []
        for nm in names:
            pct_col = f'{nm} % Change'
            val_col = f'{nm} Final Value'
            if pct_col in temp_combined_monthly.columns:
                ordered_cols_monthly.append(pct_col)
            if val_col in temp_combined_monthly.columns:
                ordered_cols_monthly.append(val_col)
        # Fallback: if nothing matched, use whatever columns exist
        if not ordered_cols_monthly:
            combined_df_monthly = temp_combined_monthly
        else:
            combined_df_monthly = temp_combined_monthly[ordered_cols_monthly]

        # Ensure columns and index are unique (pandas Styler requires unique labels)
        if combined_df_monthly.columns.duplicated().any():
            cols = list(combined_df_monthly.columns)
            seen = {}
            new_cols = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c} ({seen[c]})")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            combined_df_monthly.columns = new_cols

        if combined_df_monthly.index.duplicated().any():
            idx = list(map(str, combined_df_monthly.index))
            seen_idx = {}
            new_idx = []
            for v in idx:
                if v in seen_idx:
                    seen_idx[v] += 1
                    new_idx.append(f"{v} ({seen_idx[v]})")
                else:
                    seen_idx[v] = 0
                    new_idx.append(v)
            combined_df_monthly.index = new_idx

        # Recompute percent and final value column lists after any renaming
        pct_cols_monthly = [col for col in combined_df_monthly.columns if '% Change' in col]
        final_val_cols_monthly = [col for col in combined_df_monthly.columns if 'Final Value' in col]

        # Coerce percent columns to numeric so formatting applies correctly
        for col in pct_cols_monthly:
            if col in combined_df_monthly.columns:
                try:
                    combined_df_monthly[col] = pd.to_numeric(combined_df_monthly[col], errors='coerce')
                except TypeError:
                    # Unexpected column type (not Series/array). Try to coerce via pd.Series or fall back to NaN.
                    try:
                        combined_df_monthly[col] = pd.to_numeric(pd.Series(combined_df_monthly[col]), errors='coerce')
                    except Exception:
                        combined_df_monthly[col] = np.nan

        # Create combined format mapping: percent columns get '%' suffix, final value columns get currency
        fmt_map_monthly = {col: '{:,.2f}%' for col in pct_cols_monthly if col in combined_df_monthly.columns}
        fmt_map_monthly.update({col: '${:,.2f}' for col in final_val_cols_monthly if col in combined_df_monthly.columns})

        styler_monthly = combined_df_monthly.style
        # Color percent cells with a gradient and then apply formatting in one call
        if pct_cols_monthly:
            try:
                # Styler.map is the supported replacement for applymap
                styler_monthly = styler_monthly.map(color_gradient_stock, subset=pct_cols_monthly)
            except Exception:
                # If map still fails (edge cases), skip coloring to avoid breaking the page
                pass
        if fmt_map_monthly:
            styler_monthly = styler_monthly.format(fmt_map_monthly, na_rep='N/A')

        st.dataframe(styler_monthly, use_container_width=True, hide_index=False)

        st.markdown("---")
        st.markdown("**Detailed Portfolio Information**")
        # Make the selector visually prominent
        st.markdown(
            "<div style='background:#0b1221;padding:12px;border-radius:8px;margin-bottom:8px;'>"
            "<div style='font-size:16px;font-weight:700;color:#ffffff;margin-bottom:6px;'>Select a portfolio for detailed view</div>"
            "</div>", unsafe_allow_html=True)

        # HYBRID APPROACH: Simple selectbox with state persistence
        portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
        
        # Get all available portfolio names
        available_portfolio_names = [cfg.get('name', 'Portfolio') for cfg in portfolio_configs]
        extra_names = [n for n in st.session_state.get('strategy_comparison_all_results', {}).keys() if n not in available_portfolio_names]
        all_portfolio_names = available_portfolio_names + extra_names
        
        if not all_portfolio_names:
            st.warning("No portfolios available for detailed view.")
            selected_portfolio_detail = None
        else:
            # Initialize and maintain selection persistence
            if "strategy_comparison_selected_portfolio_name" not in st.session_state:
                st.session_state["strategy_comparison_selected_portfolio_name"] = all_portfolio_names[0]
            
            # Ensure the selected name is still valid (in case portfolios changed)
            if st.session_state["strategy_comparison_selected_portfolio_name"] not in all_portfolio_names:
                st.session_state["strategy_comparison_selected_portfolio_name"] = all_portfolio_names[0]
            
            # Find the current selection index for the selectbox
            current_index = 0
            try:
                current_index = all_portfolio_names.index(st.session_state["strategy_comparison_selected_portfolio_name"])
            except ValueError:
                current_index = 0
                st.session_state["strategy_comparison_selected_portfolio_name"] = all_portfolio_names[0]
            
            # Place the selectbox in its own column to make it larger/centered
            left_col, mid_col, right_col = st.columns([1, 3, 1])
            with mid_col:
                st.markdown("<div style='display:flex; gap:8px; align-items:center;'>", unsafe_allow_html=True)
                
                # Callback to update state immediately when selection changes
                def update_portfolio_selection():
                    selected = st.session_state["strategy_comparison_simple_portfolio_selector"]
                    st.session_state["strategy_comparison_selected_portfolio_name"] = selected
                
                # Simple selectbox with state persistence
                selected_portfolio_detail = st.selectbox(
                    "Select portfolio for details", 
                    options=all_portfolio_names,
                    index=current_index,
                    key="strategy_comparison_simple_portfolio_selector", 
                    help='Choose which portfolio to inspect in detail', 
                    label_visibility='collapsed',
                    on_change=update_portfolio_selection
                )
                
                # Use the persisted state value (updated by callback)
                selected_portfolio_detail = st.session_state["strategy_comparison_selected_portfolio_name"]
                
                # Add a prominent view button with a professional color
                view_clicked = st.button("View Details", key='strategy_comparison_view_details_btn')
                st.markdown("</div>", unsafe_allow_html=True)

        if selected_portfolio_detail:
            # Highlight the selected portfolio and optionally expand details when the View button is used
            st.markdown(f"<div style='padding:8px 12px;background:#04293a;border-radius:6px;margin-top:8px;'><strong style='color:#bde0fe;'>Showing details for:</strong> <span style='font-size:16px;color:#ffffff;margin-left:8px;'>{selected_portfolio_detail}</span></div>", unsafe_allow_html=True)
            if view_clicked:
                # No-op here; the detail panels below will render based on selected_portfolio_detail. Keep a small indicator
                st.success(f"Loaded details for {selected_portfolio_detail}")
            # Table 1: Historical Allocations
            if selected_portfolio_detail in st.session_state.strategy_comparison_all_allocations:
                st.markdown("---")
                st.markdown(f"**Historical Allocations for {selected_portfolio_detail}**")
                # Ensure proper DataFrame structure with explicit column names
                # Ensure all tickers (including CASH) are present in all dates for proper DataFrame creation
                allocation_data = st.session_state.strategy_comparison_all_allocations[selected_portfolio_detail]
                all_tickers = set()
                for date, alloc_dict in allocation_data.items():
                    all_tickers.update(alloc_dict.keys())
                
                # Create a complete allocation data structure with all tickers for all dates
                complete_allocation_data = {}
                for date, alloc_dict in allocation_data.items():
                    complete_allocation_data[date] = {}
                    for ticker in all_tickers:
                        if ticker in alloc_dict:
                            complete_allocation_data[date][ticker] = alloc_dict[ticker]
                        else:
                            # Fill missing tickers with 0
                            complete_allocation_data[date][ticker] = 0.0
                
                allocations_df_raw = pd.DataFrame(complete_allocation_data).T
                
                allocations_df_raw.index.name = "Date"
                
                # Corrected styling logic for alternating row colors (no green background for Historical Allocations)
                def highlight_rows_by_index(s):
                    is_even_row = allocations_df_raw.index.get_loc(s.name) % 2 == 0
                    bg_color = 'background-color: #0e1117' if is_even_row else 'background-color: #262626'
                    return [f'{bg_color}; color: white;'] * len(s)

                styler = allocations_df_raw.mul(100).style.apply(highlight_rows_by_index, axis=1)
                styler.format('{:,.0f}%', na_rep='N/A')
                st.dataframe(styler, use_container_width=True)


            # Table 2: Momentum Metrics and Calculated Weights
            if selected_portfolio_detail in st.session_state.strategy_comparison_all_metrics:
                st.markdown("---")
                st.markdown(f"**Momentum Metrics and Calculated Weights for {selected_portfolio_detail}**")

                # Process metrics data directly - EXACT SAME AS PAGE 7
                metrics_records = []
                for date, tickers_data in st.session_state.strategy_comparison_all_metrics[selected_portfolio_detail].items():
                        # Add all asset lines
                        asset_weights = []
                        for ticker, data in tickers_data.items():
                            # Handle None ticker as CASH
                            display_ticker = 'CASH' if ticker is None else ticker
                            if display_ticker != 'CASH':
                                asset_weights.append(data.get('Calculated_Weight', 0))
                            # Filter out any internal-only keys (e.g., 'Composite') so they don't show in the UI
                            filtered_data = {k: v for k, v in (data or {}).items() if k != 'Composite'}
                            
                            # Check if momentum is used for this portfolio
                            portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                            portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                            use_momentum = portfolio_cfg.get('use_momentum', True) if portfolio_cfg else True
                            
                            # If momentum is not used, replace Calculated_Weight with target_allocation
                            if not use_momentum:
                                if 'target_allocation' in filtered_data:
                                    filtered_data['Calculated_Weight'] = filtered_data['target_allocation']
                                else:
                                    # If target_allocation is not available, use the entered allocations from portfolio_cfg
                                    ticker_name = display_ticker if display_ticker != 'CASH' else None
                                if ticker_name and portfolio_cfg:
                                    # Find the stock in portfolio_cfg and use its allocation
                                    for stock in portfolio_cfg.get('stocks', []):
                                        if stock.get('ticker', '').strip() == ticker_name:
                                            filtered_data['Calculated_Weight'] = stock.get('allocation', 0)
                                            break
                                elif display_ticker == 'CASH' and portfolio_cfg:
                                    # For CASH, calculate the remaining allocation
                                    total_alloc = sum(stock.get('allocation', 0) for stock in portfolio_cfg.get('stocks', []))
                                    filtered_data['Calculated_Weight'] = max(0, 1.0 - total_alloc)
                        
                            record = {'Date': date, 'Ticker': display_ticker, **filtered_data}
                            metrics_records.append(record)
                        
                        # Add CASH row only if it's significant (more than 5% allocation)
                        # This prevents showing CASH for small cash balances
                        total_calculated_weight = sum(asset_weights)
                        cash_allocation = 1.0 - total_calculated_weight
                        
                        if cash_allocation > 0.05:  # Only show CASH if it's more than 5%
                            cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_allocation}
                            metrics_records.append(cash_record)
                    
                        # Ensure CASH line is added if there's non-zero cash in allocations
                        allocs_for_portfolio = st.session_state.strategy_comparison_all_allocations.get(selected_portfolio_detail) if 'strategy_comparison_all_allocations' in st.session_state else None
                        if allocs_for_portfolio and date in allocs_for_portfolio:
                            cash_alloc = allocs_for_portfolio[date].get('CASH', 0)
                            if cash_alloc > 0:
                                # Check if CASH is already in metrics_records for this date
                                cash_exists = any(record['Date'] == date and record['Ticker'] == 'CASH' for record in metrics_records)
                                if not cash_exists:
                                    # Add CASH line to metrics
                                    # Check if momentum is used to determine which weight to show
                                    portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                    portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                    use_momentum = portfolio_cfg.get('use_momentum', True) if portfolio_cfg else True
                                    
                                    if not use_momentum:
                                        # When momentum is not used, calculate CASH allocation from entered allocations
                                        total_alloc = sum(stock.get('allocation', 0) for stock in portfolio_cfg.get('stocks', []))
                                        cash_weight = max(0, 1.0 - total_alloc)
                                        cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_weight}
                                    else:
                                        cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_alloc}
                                    metrics_records.append(cash_record)
                    
                        # Add CASH line if fully allocated to cash (100%) or all asset weights are 0% (fallback logic)
                        cash_line_needed = False
                        if 'CASH' in tickers_data or None in tickers_data:
                            cash_data = tickers_data.get('CASH', tickers_data.get(None, {}))
                            cash_weight = cash_data.get('Calculated_Weight', 0)
                            if abs(cash_weight - 1.0) < 1e-6:  # 100% in decimal
                                cash_line_needed = True
                        if all(w == 0 for w in asset_weights) and asset_weights:
                            cash_line_needed = True
                        if cash_line_needed and 'CASH' not in [r['Ticker'] for r in metrics_records if r['Date'] == date]:
                            # If no explicit CASH data, create a default line
                            cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': 1.0}
                            metrics_records.append(cash_record)

                if metrics_records:
                    metrics_df = pd.DataFrame(metrics_records)
                    
                    # Filter out CASH lines where Calculated_Weight is 0 for the last date
                    if 'Calculated_Weight' in metrics_df.columns:
                        # Get the last date
                        last_date = metrics_df['Date'].max()
                        # Remove CASH records where Calculated_Weight is 0 for the last date
                        mask = ~((metrics_df['Ticker'] == 'CASH') & (metrics_df['Date'] == last_date) & (metrics_df['Calculated_Weight'] == 0))
                        metrics_df = metrics_df[mask].reset_index(drop=True)
                    
                    if not metrics_df.empty:
                        # Ensure unique index by adding a counter if needed
                        if metrics_df.duplicated(subset=['Date', 'Ticker']).any():
                            # Add a counter to make indices unique
                            metrics_df['Counter'] = metrics_df.groupby(['Date', 'Ticker']).cumcount()
                            metrics_df['Ticker_Unique'] = metrics_df['Ticker'] + metrics_df['Counter'].astype(str)
                            metrics_df.set_index(['Date', 'Ticker_Unique'], inplace=True)
                        else:
                            metrics_df.set_index(['Date', 'Ticker'], inplace=True)
                        
                    metrics_df_display = metrics_df.copy()

                    # Ensure Momentum column exists and normalize to percent when present
                    if 'Momentum' in metrics_df_display.columns:
                        metrics_df_display['Momentum'] = metrics_df_display['Momentum'].fillna(0) * 100
                    else:
                        metrics_df_display['Momentum'] = np.nan

                    def color_momentum(val):
                        if isinstance(val, (int, float)):
                            color = 'green' if val > 0 else 'red'
                            return f'color: {color}'
                        # Force white color for None, NA, and other non-numeric values
                        return 'color: #FFFFFF; font-weight: bold;'
                    
                    def color_all_columns(val):
                        # Force white color for None, NA, and other non-numeric values in ALL columns
                        if pd.isna(val) or val == 'None' or val == 'NA' or val == '':
                            return 'color: #FFFFFF; font-weight: bold;'
                        if isinstance(val, (int, float)):
                            return ''  # Let default styling handle numeric values
                        return 'color: #FFFFFF; font-weight: bold;'  # Force white for any other text
                    
                    def highlight_metrics_rows(s):
                        date_str = s.name[0]
                        ticker_str = s.name[1]
                        # If this is the CASH row, use dark green background
                        if 'CASH' in ticker_str:
                            return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                        # Otherwise, alternate row colors by date with WHITE TEXT
                        unique_dates = list(metrics_df_display.index.get_level_values(0).unique())
                        is_even = unique_dates.index(date_str) % 2 == 0
                        bg_color = 'background-color: #0e1117' if is_even else 'background-color: #262626'
                        return [f'{bg_color}; color: white;'] * len(s)

                    # Format Calculated_Weight as a percentage if present
                    if 'Calculated_Weight' in metrics_df_display.columns:
                        metrics_df_display['Calculated_Weight'] = metrics_df_display['Calculated_Weight'].fillna(0) * 100
                    # Convert Volatility from decimal (e.g., 0.20) to percent (20.0)
                    if 'Volatility' in metrics_df_display.columns:
                        metrics_df_display['Volatility'] = metrics_df_display['Volatility'].fillna(np.nan) * 100

                    # Corrected styling logic for alternating row colors and momentum color
                    styler_metrics = metrics_df_display.style.apply(highlight_metrics_rows, axis=1)
                    if 'Momentum' in metrics_df_display.columns:
                        styler_metrics = styler_metrics.map(color_momentum, subset=['Momentum'])
                    # Force white color for None/NA values in ALL columns
                    styler_metrics = styler_metrics.map(color_all_columns)

                    fmt_dict = {}
                    if 'Momentum' in metrics_df_display.columns:
                        fmt_dict['Momentum'] = '{:,.0f}%'
                    if 'Beta' in metrics_df_display.columns:
                        fmt_dict['Beta'] = '{:,.2f}'
                    if 'Volatility' in metrics_df_display.columns:
                        fmt_dict['Volatility'] = '{:,.2f}%'
                    if 'Calculated_Weight' in metrics_df_display.columns:
                        fmt_dict['Calculated_Weight'] = '{:,.0f}%'

                    if fmt_dict:
                        styler_metrics = styler_metrics.format(fmt_dict)
                    
                    # NUCLEAR OPTION: Inject custom CSS to override Streamlit's stubborn styling
                    st.markdown("""
                    <style>
                    /* NUCLEAR CSS OVERRIDE - BEAT STREAMLIT INTO SUBMISSION */
                    .stDataFrame [data-testid="stDataFrame"] div[data-testid="stDataFrame"] table td,
                    .stDataFrame [data-testid="stDataFrame"] div[data-testid="stDataFrame"] table th,
                    .stDataFrame table td,
                    .stDataFrame table th {
                        color: #FFFFFF !important;
                        font-weight: bold !important;
                        text-shadow: 1px 1px 2px black !important;
                    }
                    
                    /* Force ALL text in dataframes to be white */
                    .stDataFrame * {
                        color: #FFFFFF !important;
                    }
                    
                    /* Override any Streamlit bullshit */
                    .stDataFrame [data-testid="stDataFrame"] *,
                    .stDataFrame div[data-testid="stDataFrame"] * {
                        color: #FFFFFF !important;
                        font-weight: bold !important;
                    }
                    
                    /* Target EVERYTHING in the dataframe */
                    .stDataFrame table *,
                    .stDataFrame div table *,
                    .stDataFrame [data-testid="stDataFrame"] table *,
                    .stDataFrame [data-testid="stDataFrame"] div table * {
                        color: #FFFFFF !important;
                        font-weight: bold !important;
                    }
                    
                    /* Target all cells specifically */
                    .stDataFrame td,
                    .stDataFrame th,
                    .stDataFrame table td,
                    .stDataFrame table th {
                        color: #FFFFFF !important;
                        font-weight: bold !important;
                    }
                    
                    /* Target Streamlit's specific elements */
                    div[data-testid="stDataFrame"] table td,
                    div[data-testid="stDataFrame"] table th,
                    div[data-testid="stDataFrame"] div table td,
                    div[data-testid="stDataFrame"] div table th {
                        color: #FFFFFF !important;
                        font-weight: bold !important;
                    }
                    
                    /* Target everything with maximum specificity */
                    div[data-testid="stDataFrame"] *,
                    div[data-testid="stDataFrame"] div *,
                    div[data-testid="stDataFrame"] table *,
                    div[data-testid="stDataFrame"] div table * {
                        color: #FFFFFF !important;
                        font-weight: bold !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    st.dataframe(styler_metrics, use_container_width=True)
                    

                    # --- Allocation plots: Final allocation and last rebalance allocation ---
                    allocs_for_portfolio = st.session_state.strategy_comparison_all_allocations.get(selected_portfolio_detail) if 'strategy_comparison_all_allocations' in st.session_state else None
                    if allocs_for_portfolio:
                        try:
                                # Sort allocation dates
                                alloc_dates = sorted(list(allocs_for_portfolio.keys()))
                                if len(alloc_dates) == 0:
                                    st.info("No allocation history available to plot.")
                                else:
                                    final_date = alloc_dates[-1]
                                    
                                    # Find the actual last rebalance date by looking at the backtest results
                                    # The last rebalance date should be from the actual rebalancing dates, not just any date
                                    last_rebal_date = None
                                    if 'strategy_comparison_all_results' in st.session_state and selected_portfolio_detail in st.session_state.strategy_comparison_all_results:
                                        portfolio_results = st.session_state.strategy_comparison_all_results[selected_portfolio_detail]
                                        if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                            # Get the simulation index (actual trading days)
                                            sim_index = portfolio_results['with_additions'].index
                                            # Get the rebalancing frequency to calculate actual rebalance dates
                                            portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                            portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                            if portfolio_cfg:
                                                rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none')
                                                # Get actual rebalancing dates
                                                actual_rebal_dates = get_dates_by_freq(rebalancing_frequency, sim_index[0], sim_index[-1], sim_index)
                                                if actual_rebal_dates:
                                                    # Find the most recent actual rebalance date that's in our allocation data
                                                    actual_rebal_dates_sorted = sorted(list(actual_rebal_dates))
                                                    for rebal_date in reversed(actual_rebal_dates_sorted):
                                                        if rebal_date in allocs_for_portfolio:
                                                            last_rebal_date = rebal_date
                                                            break
                                    
                                    # Fallback to second-to-last date if we couldn't find an actual rebalance date
                                    if last_rebal_date is None:
                                        last_rebal_date = alloc_dates[-2] if len(alloc_dates) > 1 else alloc_dates[-1]

                                # Keep original dates for data retrieval
                                final_date_original = final_date
                                last_rebal_date_original = last_rebal_date
                                
                                # Convert dates to datetime objects only for display purposes
                                if isinstance(final_date, str):
                                    final_date_display = pd.to_datetime(final_date)
                                else:
                                    final_date_display = final_date
                                if isinstance(last_rebal_date, str):
                                    last_rebal_date_display = pd.to_datetime(last_rebal_date)
                                else:
                                    last_rebal_date_display = last_rebal_date
                                
                                # Remove timezone info if present (same as Multi-Backtest)
                                if hasattr(final_date_display, 'tzinfo') and final_date_display.tzinfo is not None:
                                    final_date_display = final_date_display.replace(tzinfo=None)
                                if hasattr(last_rebal_date_display, 'tzinfo') and last_rebal_date_display.tzinfo is not None:
                                    last_rebal_date_display = last_rebal_date_display.replace(tzinfo=None)

                                final_alloc = allocs_for_portfolio.get(final_date_original, {})
                                rebal_alloc = allocs_for_portfolio.get(last_rebal_date_original, {})

                                # Helper to prepare bar data
                                def prepare_bar_data(d):
                                    labels = []
                                    values = []
                                    for k, v in sorted(d.items(), key=lambda x: (-x[1], x[0])):
                                        labels.append(k)
                                        try:
                                            values.append(float(v) * 100)
                                        except Exception:
                                            values.append(0.0)
                                    return labels, values

                                labels_final, vals_final = prepare_bar_data(final_alloc)
                                labels_rebal, vals_rebal = prepare_bar_data(rebal_alloc)

                                # Add timer for next rebalance date
                                try:
                                    # Get the actual last rebalance date from the backtest results
                                    last_rebal_date_for_timer = None
                                    if 'strategy_comparison_all_results' in st.session_state and selected_portfolio_detail in st.session_state.strategy_comparison_all_results:
                                        portfolio_results = st.session_state.strategy_comparison_all_results[selected_portfolio_detail]
                                        if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                            # Get the simulation index (actual trading days)
                                            sim_index = portfolio_results['with_additions'].index
                                            # Get the rebalancing frequency to calculate actual rebalance dates
                                            portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                            portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                            if portfolio_cfg:
                                                rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none')
                                                # Get actual rebalancing dates
                                                actual_rebal_dates = get_dates_by_freq(rebalancing_frequency, sim_index[0], sim_index[-1], sim_index)
                                                if actual_rebal_dates:
                                                    # Find the most recent actual rebalance date that's in our allocation data
                                                    actual_rebal_dates_sorted = sorted(list(actual_rebal_dates))
                                                    for rebal_date in reversed(actual_rebal_dates_sorted):
                                                        if rebal_date in allocs_for_portfolio:
                                                            last_rebal_date_for_timer = rebal_date
                                                            break
                                    
                                    # Fallback to last allocation date if we couldn't find an actual rebalance date
                                    if last_rebal_date_for_timer is None and len(alloc_dates) >= 1:
                                        last_rebal_date_for_timer = alloc_dates[-1]
                                    
                                    # Get rebalancing frequency from portfolio config
                                    portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                    portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                    rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none') if portfolio_cfg else 'none'
                                    # Convert to lowercase and map to function expectations
                                    rebalancing_frequency = rebalancing_frequency.lower()
                                    # Map frequency names to what the function expects
                                    frequency_mapping = {
                                        'monthly': 'Monthly',
                                        'weekly': 'Weekly',
                                        'bi-weekly': 'Biweekly',
                                        'biweekly': 'Biweekly',
                                        'quarterly': 'Quarterly',
                                        'semi-annually': 'Semiannually',
                                        'semiannually': 'Semiannually',
                                        'annually': 'Annually',
                                        'yearly': 'Annually',
                                        'market_day': 'market_day',
                                        'calendar_day': 'calendar_day',
                                        'never': 'Never',
                                        'none': 'Never'
                                    }
                                    rebalancing_frequency = frequency_mapping.get(rebalancing_frequency, rebalancing_frequency)
                                    
                                    if last_rebal_date_for_timer and rebalancing_frequency != 'Never':
                                        # Ensure last_rebal_date_for_timer is a naive datetime object
                                        if isinstance(last_rebal_date_for_timer, str):
                                            last_rebal_date_for_timer = pd.to_datetime(last_rebal_date_for_timer)
                                        if hasattr(last_rebal_date_for_timer, 'tzinfo') and last_rebal_date_for_timer.tzinfo is not None:
                                            last_rebal_date_for_timer = last_rebal_date_for_timer.replace(tzinfo=None)
                                        next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                                            rebalancing_frequency, last_rebal_date_for_timer
                                        )
                                        
                                        if next_date and time_until:
                                            st.markdown("---")
                                            st.markdown("**⏰ Next Rebalance Timer**")
                                            
                                            # Create columns for timer display
                                            col1, col2, col3 = st.columns(3)
                                            
                                            with col1:
                                                st.metric(
                                                    label="Time Until Next Rebalance",
                                                    value=format_time_until(time_until),
                                                    delta=None
                                                )
                                            
                                            with col2:
                                                st.metric(
                                                    label="Target Rebalance Date",
                                                    value=next_date.strftime("%B %d, %Y"),
                                                    delta=None
                                                )
                                            
                                            with col3:
                                                st.metric(
                                                    label="Rebalancing Frequency",
                                                    value=rebalancing_frequency.replace('_', ' ').title(),
                                                    delta=None
                                                )
                                            
                                            # Add a progress bar showing progress to next rebalance
                                            if rebalancing_frequency in ['week', '2weeks', 'month', '3months', '6months', 'year']:
                                                # Calculate progress percentage
                                                if hasattr(last_rebal_date_for_timer, 'to_pydatetime'):
                                                    last_rebal_datetime = last_rebal_date_for_timer.to_pydatetime()
                                                else:
                                                    last_rebal_datetime = last_rebal_date_for_timer
                                                
                                                total_period = (next_rebalance_datetime - last_rebal_datetime).total_seconds()
                                                elapsed_period = (datetime.now() - last_rebal_datetime).total_seconds()
                                                progress = min(max(elapsed_period / total_period, 0), 1)
                                                
                                                st.progress(progress, text=f"Progress to next rebalance: {progress:.1%}")
                                            
                                            # Create and store timer table figure for PDF export
                                            try:
                                                timer_data = [
                                                    ['Time Until Next Rebalance', format_time_until(time_until)],
                                                    ['Target Rebalance Date', next_date.strftime("%B %d, %Y")],
                                                    ['Rebalancing Frequency', rebalancing_frequency.replace('_', ' ').title()]
                                                ]
                                                
                                                fig_timer = go.Figure(data=[go.Table(
                                                    header=dict(
                                                        values=['Parameter', 'Value'],
                                                        fill_color='#2E86AB',
                                                        align='center',
                                                        font=dict(color='white', size=16, family='Arial Black')
                                                    ),
                                                    cells=dict(
                                                        values=[[row[0] for row in timer_data], [row[1] for row in timer_data]],
                                                        fill_color=[['#F8F9FA', '#FFFFFF'] * 2, ['#F8F9FA', '#FFFFFF'] * 2],
                                                        align='center',
                                                        font=dict(color='black', size=14, family='Arial'),
                                                        height=40
                                                    )
                                                )])
                                                
                                                fig_timer.update_layout(
                                                    title=dict(
                                                        text="⏰ Next Rebalance Timer",
                                                        x=0.5,
                                                        font=dict(size=18, color='#2E86AB', family='Arial Black')
                                                    ),
                                                    width=700,
                                                    height=250,
                                                    margin=dict(l=20, r=20, t=60, b=20)
                                                )
                                                
                                                # Store in session state for PDF export
                                                st.session_state[f'strategy_comparison_timer_table_{selected_portfolio_detail}'] = fig_timer
                                            except Exception as e:
                                                pass  # Silently ignore timer table creation errors
                                            
                                            # Also create timer tables for ALL portfolios for PDF export
                                            try:
                                                # Get all portfolio configs
                                                all_portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                                
                                                for portfolio_cfg in all_portfolio_configs:
                                                    portfolio_name = portfolio_cfg.get('name', 'Unknown')
                                                    
                                                    # Get rebalancing frequency for this portfolio (use EXACT same logic as main window)
                                                    rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none')
                                                    rebalancing_frequency = rebalancing_frequency.lower()
                                                    
                                                    # Map frequency names to what the function expects (EXACT same as main window)
                                                    frequency_mapping = {
                                                        'monthly': 'Monthly',
                                                        'weekly': 'Weekly',
                                                        'bi-weekly': 'Biweekly',
                                                        'biweekly': 'Biweekly',
                                                        'quarterly': 'Quarterly',
                                                        'semi-annually': 'Semiannually',
                                                        'semiannually': 'Semiannually',
                                                        'annually': 'Annually',
                                                        'yearly': 'Annually',
                                                        'market_day': 'market_day',
                                                        'calendar_day': 'calendar_day',
                                                        'never': 'Never',
                                                        'none': 'Never'
                                                    }
                                                    rebalancing_frequency = frequency_mapping.get(rebalancing_frequency, rebalancing_frequency)
                                                    
                                                    # Get last rebalance date from allocation history (EXACT same as main window)
                                                    allocs_for_portfolio = st.session_state.strategy_comparison_all_allocations.get(portfolio_name) if 'strategy_comparison_all_allocations' in st.session_state else None
                                                    if allocs_for_portfolio:
                                                        alloc_dates = sorted(list(allocs_for_portfolio.keys()))
                                                        # Get the actual last rebalance date from the backtest results
                                                        last_rebal_date_for_timer = None
                                                        if 'strategy_comparison_all_results' in st.session_state and portfolio_name in st.session_state.strategy_comparison_all_results:
                                                            portfolio_results = st.session_state.strategy_comparison_all_results[portfolio_name]
                                                            if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                                                # Get the simulation index (actual trading days)
                                                                sim_index = portfolio_results['with_additions'].index
                                                                # Get the rebalancing frequency to calculate actual rebalance dates
                                                                portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                                                portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == portfolio_name), None)
                                                                if portfolio_cfg:
                                                                    rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none')
                                                                    # Get actual rebalancing dates
                                                                    actual_rebal_dates = get_dates_by_freq(rebalancing_frequency, sim_index[0], sim_index[-1], sim_index)
                                                                    if actual_rebal_dates:
                                                                        # Find the most recent actual rebalance date that's in our allocation data
                                                                        actual_rebal_dates_sorted = sorted(list(actual_rebal_dates))
                                                                        for rebal_date in reversed(actual_rebal_dates_sorted):
                                                                            if rebal_date in allocs_for_portfolio:
                                                                                last_rebal_date_for_timer = rebal_date
                                                                                break
                                                        
                                                        # Fallback to last allocation date if we couldn't find an actual rebalance date
                                                        if last_rebal_date_for_timer is None and len(alloc_dates) >= 1:
                                                            last_rebal_date_for_timer = alloc_dates[-1]
                                                    else:
                                                        last_rebal_date_for_timer = None
                                                    
                                                    if last_rebal_date_for_timer and rebalancing_frequency != 'Never':
                                                        # Ensure last_rebal_date_for_timer is a naive datetime object (EXACT same as main window)
                                                        if isinstance(last_rebal_date_for_timer, str):
                                                            last_rebal_date_for_timer = pd.to_datetime(last_rebal_date_for_timer)
                                                        if hasattr(last_rebal_date_for_timer, 'tzinfo') and last_rebal_date_for_timer.tzinfo is not None:
                                                            last_rebal_date_for_timer = last_rebal_date_for_timer.replace(tzinfo=None)
                                                        
                                                        # Use EXACT same function as main window
                                                        next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                                                            rebalancing_frequency, last_rebal_date_for_timer
                                                        )
                                                        
                                                        if next_date and time_until:
                                                            # Create timer data for this portfolio (EXACT same as main window)
                                                            timer_data_port = [
                                                                ['Time Until Next Rebalance', format_time_until(time_until)],
                                                                ['Target Rebalance Date', next_date.strftime("%B %d, %Y")],
                                                                ['Rebalancing Frequency', rebalancing_frequency.replace('_', ' ').title()]
                                                            ]
                                                        else:
                                                            # Fallback if calculation fails
                                                            timer_data_port = [
                                                                ['Time Until Next Rebalance', 'Calculation failed'],
                                                                ['Target Rebalance Date', 'N/A'],
                                                                ['Rebalancing Frequency', rebalancing_frequency.replace('_', ' ').title()]
                                                            ]
                                                    else:
                                                        # For portfolios with no rebalancing or no last rebalance date
                                                        if rebalancing_frequency == 'none':
                                                            timer_data_port = [
                                                                ['Time Until Next Rebalance', 'No rebalancing scheduled'],
                                                                ['Target Rebalance Date', 'N/A'],
                                                                ['Rebalancing Frequency', 'No rebalancing']
                                                            ]
                                                        else:
                                                            timer_data_port = [
                                                                ['Time Until Next Rebalance', 'No rebalance history'],
                                                                ['Target Rebalance Date', 'N/A'],
                                                                ['Rebalancing Frequency', rebalancing_frequency.replace('_', ' ').title()]
                                                            ]
                                                    
                                                    # Create timer table figure for this portfolio
                                                    fig_timer_port = go.Figure(data=[go.Table(
                                                        header=dict(
                                                            values=['Parameter', 'Value'],
                                                            fill_color='#2E86AB',
                                                            align='center',
                                                            font=dict(color='white', size=16, family='Arial Black')
                                                        ),
                                                        cells=dict(
                                                            values=[[row[0] for row in timer_data_port], [row[1] for row in timer_data_port]],
                                                            fill_color=[['#F8F9FA', '#FFFFFF'] * 2, ['#F8F9FA', '#FFFFFF'] * 2],
                                                            align='center',
                                                            font=dict(color='black', size=14, family='Arial'),
                                                            height=40
                                                        )
                                                    )])
                                                    
                                                    fig_timer_port.update_layout(
                                                        title=dict(
                                                            text=f"⏰ Next Rebalance Timer - {portfolio_name}",
                                                            x=0.5,
                                                            font=dict(size=18, color='#2E86AB', family='Arial Black')
                                                        ),
                                                        width=700,
                                                        height=250,
                                                        margin=dict(l=20, r=20, t=60, b=20)
                                                    )
                                                    
                                                    # Store in session state for PDF export
                                                    st.session_state[f'strategy_comparison_timer_table_{portfolio_name}'] = fig_timer_port
                                                    # Timer table created successfully
                                            
                                            except Exception as e:
                                                # Error creating timer tables
                                                pass  # Silently ignore timer table creation errors
                                            
                                            # Also create allocation charts and tables for ALL portfolios for PDF export
                                            try:
                                                # Get all portfolio configs
                                                all_portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                                snapshot = st.session_state.get('strategy_comparison_snapshot_data', {})
                                                today_weights_map = snapshot.get('today_weights_map', {})
                                                last_rebalance_dates = snapshot.get('last_rebalance_dates', {})
                                                
                                                for portfolio_cfg in all_portfolio_configs:
                                                    portfolio_name = portfolio_cfg.get('name', 'Unknown')
                                                    
                                                    # Get today's weights for this portfolio
                                                    portfolio_today_weights = today_weights_map.get(portfolio_name, {})
                                                    
                                                    if portfolio_today_weights:
                                                        # Create allocation pie chart for this portfolio
                                                        labels_portfolio = [k for k, v in sorted(portfolio_today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
                                                        vals_portfolio = [float(portfolio_today_weights[k]) * 100 for k in labels_portfolio]
                                                        
                                                        if labels_portfolio and vals_portfolio:
                                                            fig_today_portfolio = go.Figure()
                                                            fig_today_portfolio.add_trace(go.Pie(labels=labels_portfolio, values=vals_portfolio, hole=0.3))
                                                            fig_today_portfolio.update_traces(textinfo='percent+label')
                                                            fig_today_portfolio.update_layout(
                                                                template='plotly_dark', 
                                                                margin=dict(t=30),
                                                                width=600,
                                                                height=600,
                                                                showlegend=True
                                                            )
                                                            
                                                            # Store for PDF export
                                                            st.session_state[f'strategy_comparison_fig_today_{portfolio_name}'] = fig_today_portfolio
                                                            
                                                            # Create allocation table for this portfolio
                                                            try:
                                                                # Get portfolio value for calculations
                                                                portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)
                                                                
                                                                # Get current portfolio value from backtest results
                                                                if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
                                                                    portfolio_results = st.session_state.strategy_comparison_all_results.get(portfolio_name)
                                                                    if portfolio_results:
                                                                        if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                                                            final_value = portfolio_results['with_additions'].iloc[-1]
                                                                            if not pd.isna(final_value) and final_value > 0:
                                                                                portfolio_value = float(final_value)
                                                                        elif isinstance(portfolio_results, dict) and 'no_additions' in portfolio_results:
                                                                            final_value = portfolio_results['no_additions'].iloc[-1]
                                                                            if not pd.isna(final_value) and final_value > 0:
                                                                                portfolio_value = float(final_value)
                                                                        elif isinstance(portfolio_results, pd.Series):
                                                                            latest_value = portfolio_results.iloc[-1]
                                                                            if not pd.isna(latest_value) and latest_value > 0:
                                                                                portfolio_value = float(latest_value)
                                                                
                                                                # Get raw data for price calculations
                                                                raw_data = st.session_state.get('strategy_comparison_raw_data', {})
                                                                
                                                                def _price_on_or_before(df, target_date):
                                                                    try:
                                                                        idx = df.index[df.index <= pd.to_datetime(target_date)]
                                                                        if len(idx) == 0:
                                                                            return None
                                                                        return float(df.loc[idx[-1], 'Close'])
                                                                    except Exception:
                                                                        return None
                                                                
                                                                # Build allocation table data
                                                                rows = []
                                                                for tk in sorted(portfolio_today_weights.keys()):
                                                                    alloc_pct = float(portfolio_today_weights.get(tk, 0))
                                                                    if tk == 'CASH':
                                                                        price = None
                                                                        shares = 0
                                                                        total_val = portfolio_value * alloc_pct
                                                                    else:
                                                                        df = raw_data.get(tk)
                                                                        price = None
                                                                        if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                                            try:
                                                                                price = float(df['Close'].iloc[-1])
                                                                            except Exception:
                                                                                price = None
                                                                        try:
                                                                            if price and price > 0:
                                                                                allocation_value = portfolio_value * alloc_pct
                                                                                shares = round(allocation_value / price, 1)
                                                                                total_val = shares * price
                                                                            else:
                                                                                shares = 0.0
                                                                                total_val = portfolio_value * alloc_pct
                                                                        except Exception:
                                                                            shares = 0
                                                                            total_val = portfolio_value * alloc_pct
                                                                    
                                                                    pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                                                                    rows.append({
                                                                        'Ticker': tk,
                                                                        'Allocation %': round(alloc_pct * 100, 2),
                                                                        'Price ($)': round(price, 2) if price is not None else float('nan'),
                                                                        'Shares': round(shares, 2),
                                                                        'Total Value ($)': round(total_val, 2),
                                                                        '% of Portfolio': round(pct_of_port, 2),
                                                                    })
                                                                
                                                                if rows:
                                                                    df_display = pd.DataFrame(rows)
                                                                    df_display = df_display.sort_values('Total Value ($)', ascending=False)
                                                                    
                                                                    # Format the data to ensure 2 decimal places for display - EXACT same as Multi-Backtest
                                                                    formatted_values = []
                                                                    for col in df_display.columns:
                                                                        if col in ['Price ($)', 'Total Value ($)', '% of Portfolio']:
                                                                            # Format monetary and percentage values to 2 decimal places
                                                                            formatted_values.append([f"{df_display[col][i]:.2f}" if pd.notna(df_display[col][i]) else "" for i in range(len(df_display))])
                                                                        elif col == 'Shares':
                                                                            # Format shares to 1 decimal place
                                                                            formatted_values.append([f"{df_display[col][i]:.1f}" if pd.notna(df_display[col][i]) else "" for i in range(len(df_display))])
                                                                        elif col == 'Allocation %':
                                                                            # Format allocation to 2 decimal places
                                                                            formatted_values.append([f"{df_display[col][i]:.2f}" if pd.notna(df_display[col][i]) else "" for i in range(len(df_display))])
                                                                        else:
                                                                            # Keep other columns as is
                                                                            formatted_values.append([str(df_display[col][i]) if pd.notna(df_display[col][i]) else "" for i in range(len(df_display))])
                                                                    
                                                                    # Create Plotly table figure for PDF export
                                                                    fig_alloc_table_portfolio = go.Figure(data=[go.Table(
                                                                        header=dict(
                                                                            values=list(df_display.columns),
                                                                            fill_color='#2E86AB',
                                                                            align='center',
                                                                            font=dict(color='white', size=14, family='Arial Black')
                                                                        ),
                                                                        cells=dict(
                                                                            values=formatted_values,
                                                                            fill_color=[['#F8F9FA', '#FFFFFF'] * 2, ['#F8F9FA', '#FFFFFF'] * 2],
                                                                            align='center',
                                                                            font=dict(color='black', size=12, family='Arial'),
                                                                            height=35
                                                                        )
                                                                    )])
                                                                    
                                                                    fig_alloc_table_portfolio.update_layout(
                                                                        title=dict(
                                                                            text=f"Allocation Table - {portfolio_name}",
                                                                            x=0.5,
                                                                            font=dict(size=16, color='#2E86AB', family='Arial Black')
                                                                        ),
                                                                        width=800,
                                                                        height=400,
                                                                        margin=dict(l=20, r=20, t=60, b=20)
                                                                    )
                                                                    
                                                                    # Store for PDF export
                                                                    st.session_state[f'strategy_comparison_fig_alloc_table_{portfolio_name}'] = fig_alloc_table_portfolio
                                                                
                                                            except Exception as e:
                                                                st.session_state[f'strategy_comparison_fig_alloc_table_{portfolio_name}'] = None
                                            
                                            except Exception as e:
                                                pass
                                except Exception as e:
                                    pass  # Silently ignore timer calculation errors

                                # Main "Rebalance as of today" plot and table - this should be the main rebalancing representation
                                st.markdown("---")
                                st.markdown(f"**🔄 Rebalance as of Today ({pd.Timestamp.now().strftime('%Y-%m-%d')})**")
                                
                                # Get momentum-based calculated weights for today's rebalancing from stored snapshot
                                today_weights = {}
                                
                                # Get the stored today_weights_map from snapshot data
                                snapshot = st.session_state.get('strategy_comparison_snapshot_data', {})
                                today_weights_map = snapshot.get('today_weights_map', {}) if snapshot else {}
                                
                                if selected_portfolio_detail in today_weights_map:
                                    today_weights = today_weights_map.get(selected_portfolio_detail, {})
                                else:
                                    # Fallback to current allocation if no stored weights found
                                    today_weights = final_alloc
                                
                                # Create labels and values for the plot
                                labels_today = [k for k, v in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
                                vals_today = [float(today_weights[k]) * 100 for k in labels_today]
                                
                                # Create a larger plot for the main rebalancing representation
                                st.markdown(f"**Target Allocation if Rebalanced Today**")
                                fig_today = go.Figure()
                                fig_today.add_trace(go.Pie(labels=labels_today, values=vals_today, hole=0.3))
                                fig_today.update_traces(textinfo='percent+label')
                                fig_today.update_layout(
                                    template='plotly_dark', 
                                    margin=dict(t=30),
                                    width=600,
                                    height=600,  # Make it even bigger
                                    showlegend=True
                                )
                                # Store fig_today for PDF generation
                                st.session_state[f'strategy_comparison_fig_today_{selected_portfolio_detail}'] = fig_today
                                
                                # Add timer for next rebalance date (above the pie chart)
                                try:
                                    # Get the actual last rebalance date from the backtest results
                                    last_rebal_date_for_timer = None
                                    if 'strategy_comparison_all_results' in st.session_state and selected_portfolio_detail in st.session_state.strategy_comparison_all_results:
                                        portfolio_results = st.session_state.strategy_comparison_all_results[selected_portfolio_detail]
                                        if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                            # Get the simulation index (actual trading days)
                                            sim_index = portfolio_results['with_additions'].index
                                            # Get the rebalancing frequency to calculate actual rebalance dates
                                            portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                            portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                            if portfolio_cfg:
                                                rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none')
                                                # Get actual rebalancing dates
                                                actual_rebal_dates = get_dates_by_freq(rebalancing_frequency, sim_index[0], sim_index[-1], sim_index)
                                                if actual_rebal_dates:
                                                    # Find the most recent actual rebalance date that's in our allocation data
                                                    actual_rebal_dates_sorted = sorted(list(actual_rebal_dates))
                                                    for rebal_date in reversed(actual_rebal_dates_sorted):
                                                        if rebal_date in allocs_for_portfolio:
                                                            last_rebal_date_for_timer = rebal_date
                                                            break
                                    
                                    # Fallback to last allocation date if we couldn't find an actual rebalance date
                                    if last_rebal_date_for_timer is None and len(alloc_dates) >= 1:
                                        last_rebal_date_for_timer = alloc_dates[-1]
                                    
                                    # Get rebalancing frequency from portfolio config
                                    portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                    portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                    rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none') if portfolio_cfg else 'none'
                                    # Convert to lowercase and map to function expectations
                                    rebalancing_frequency = rebalancing_frequency.lower()
                                    # Map frequency names to what the function expects
                                    frequency_mapping = {
                                        'monthly': 'month',
                                        'weekly': 'week',
                                        'bi-weekly': '2weeks',
                                        'biweekly': '2weeks',
                                        'quarterly': '3months',
                                        'semi-annually': '6months',
                                        'semiannually': '6months',
                                        'annually': 'year',
                                        'yearly': 'year',
                                        'market_day': 'market_day',
                                        'calendar_day': 'calendar_day',
                                        'never': 'none',
                                        'none': 'none'
                                    }
                                    rebalancing_frequency = frequency_mapping.get(rebalancing_frequency, rebalancing_frequency)
                                    
                                    if last_rebal_date_for_timer and rebalancing_frequency != 'none':
                                        # Ensure last_rebal_date_for_timer is a naive datetime object
                                        if isinstance(last_rebal_date_for_timer, str):
                                            last_rebal_date_for_timer = pd.to_datetime(last_rebal_date_for_timer)
                                        if hasattr(last_rebal_date_for_timer, 'tzinfo') and last_rebal_date_for_timer.tzinfo is not None:
                                            last_rebal_date_for_timer = last_rebal_date_for_timer.replace(tzinfo=None)
                                        
                                        # Calculate next rebalance date and time until
                                        next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                                            rebalancing_frequency, last_rebal_date_for_timer
                                        )
                                        
                                        if next_date and time_until:
                                            st.markdown("---")
                                            st.markdown("**⏰ Next Rebalance Timer**")
                                            
                                            # Create columns for timer display (same format as page 1)
                                            col1, col2, col3 = st.columns(3)
                                            
                                            with col1:
                                                st.metric(
                                                    label="Time Until Next Rebalance",
                                                    value=format_time_until(time_until),
                                                    delta=None
                                                )
                                            
                                            with col2:
                                                st.metric(
                                                    label="Target Rebalance Date",
                                                    value=next_date.strftime("%B %d, %Y"),
                                                    delta=None
                                                )
                                            
                                            with col3:
                                                st.metric(
                                                    label="Rebalancing Frequency",
                                                    value=rebalancing_frequency.replace('_', ' ').title(),
                                                    delta=None
                                                )
                                            
                                            # Add a progress bar showing progress to next rebalance (same format as page 1)
                                            if rebalancing_frequency in ['week', '2weeks', 'month', '3months', '6months', 'year']:
                                                # Calculate progress percentage
                                                if hasattr(last_rebal_date_for_timer, 'to_pydatetime'):
                                                    last_rebal_datetime = last_rebal_date_for_timer.to_pydatetime()
                                                else:
                                                    last_rebal_datetime = last_rebal_date_for_timer
                                                
                                                total_period = (next_rebalance_datetime - last_rebal_datetime).total_seconds()
                                                elapsed_period = (datetime.now() - last_rebal_datetime).total_seconds()
                                                progress = min(max(elapsed_period / total_period, 0), 1)
                                                
                                                st.progress(progress, text=f"Progress to next rebalance: {progress:.1%}")
                                            
                                            # Create and store timer table figure for PDF export
                                            try:
                                                timer_data = [
                                                    ['Time Until Next Rebalance', format_time_until(time_until)],
                                                    ['Target Rebalance Date', next_date.strftime("%B %d, %Y")],
                                                    ['Rebalancing Frequency', rebalancing_frequency.replace('_', ' ').title()]
                                                ]
                                                
                                                fig_timer = go.Figure(data=[go.Table(
                                                    header=dict(
                                                        values=['Parameter', 'Value'],
                                                        fill_color='#2E86AB',
                                                        align='center',
                                                        font=dict(color='white', size=16, family='Arial Black')
                                                    ),
                                                    cells=dict(
                                                        values=list(zip(*timer_data)),
                                                        fill_color='#1f2937',
                                                        align=['left', 'center'],
                                                        font=dict(color='white', size=14)
                                                    )
                                                )])
                                                
                                                fig_timer.update_layout(
                                                    title=dict(
                                                        text="⏰ Next Rebalance Timer",
                                                        x=0.5,
                                                        font=dict(size=18, color='#2E86AB', family='Arial Black')
                                                    ),
                                                    width=700,
                                                    height=250,
                                                    margin=dict(l=20, r=20, t=60, b=20)
                                                )
                                                
                                                # Store in session state for PDF export
                                                st.session_state[f'strategy_comparison_timer_table_{selected_portfolio_detail}'] = fig_timer
                                            except Exception as e:
                                                pass  # Silently ignore timer table creation errors
                                    else:
                                        st.info("⏰ No rebalancing schedule configured for this portfolio.")
                                except Exception as e:
                                    pass  # Silently ignore timer calculation errors
                                
                                # Display the pie chart
                                st.plotly_chart(fig_today, use_container_width=True, key=f"multi_today_{selected_portfolio_detail}")
                                
                                # Table moved under the plot
                                # Add the "Rebalance as of today" table
                                try:
                                        # Get portfolio configuration for calculations
                                        portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                        portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                        
                                        if portfolio_cfg:
                                            # Use current portfolio value from backtest results instead of initial value
                                            portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)  # fallback to initial value
                                            
                                            # Get current portfolio value from backtest results
                                            if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
                                                portfolio_results = st.session_state.strategy_comparison_all_results.get(selected_portfolio_detail)
                                                if portfolio_results:
                                                    # Use the Final Value (with additions) for Strategy Comparison - total portfolio value including all cash additions and compounding
                                                    if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                                        # Get the final value from the with_additions series (includes all cash additions and compounding)
                                                        final_value = portfolio_results['with_additions'].iloc[-1]
                                                        if not pd.isna(final_value) and final_value > 0:
                                                            portfolio_value = float(final_value)
                                                    elif isinstance(portfolio_results, dict) and 'no_additions' in portfolio_results:
                                                        # Fallback to no_additions if with_additions not available
                                                        final_value = portfolio_results['no_additions'].iloc[-1]
                                                        if not pd.isna(final_value) and final_value > 0:
                                                            portfolio_value = float(final_value)
                                                    elif isinstance(portfolio_results, pd.Series):
                                                        # Get the latest value from the series
                                                        latest_value = portfolio_results.iloc[-1]
                                                        if not pd.isna(latest_value) and latest_value > 0:
                                                            portfolio_value = float(latest_value)
                                            
                                            # Get raw data for price calculations
                                            raw_data = st.session_state.get('strategy_comparison_raw_data', {})
                                            
                                            def _price_on_or_before(df, target_date):
                                                try:
                                                    idx = df.index[df.index <= pd.to_datetime(target_date)]
                                                    if len(idx) == 0:
                                                        return None
                                                    return float(df.loc[idx[-1], 'Close'])
                                                except Exception:
                                                    return None

                                            def build_table_from_alloc(alloc_dict, price_date, label):
                                                rows = []
                                                for tk in sorted(alloc_dict.keys()):
                                                    alloc_pct = float(alloc_dict.get(tk, 0))
                                                    if tk == 'CASH':
                                                        price = None
                                                        shares = 0
                                                        total_val = portfolio_value * alloc_pct
                                                    else:
                                                        df = raw_data.get(tk)
                                                        price = None
                                                        if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                            if price_date is None:
                                                                # use latest price
                                                                try:
                                                                    price = float(df['Close'].iloc[-1])
                                                                except Exception:
                                                                    price = None
                                                            else:
                                                                price = _price_on_or_before(df, price_date)
                                                        try:
                                                            if price and price > 0:
                                                                allocation_value = portfolio_value * alloc_pct
                                                                # allow fractional shares shown to 1 decimal place
                                                                shares = round(allocation_value / price, 1)
                                                                total_val = shares * price
                                                            else:
                                                                shares = 0.0
                                                                total_val = portfolio_value * alloc_pct
                                                        except Exception:
                                                            shares = 0
                                                            total_val = portfolio_value * alloc_pct

                                                    pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                                                    rows.append({
                                                        'Ticker': tk,
                                                        'Allocation %': alloc_pct * 100,
                                                        'Price ($)': price if price is not None else float('nan'),
                                                        'Shares': shares,
                                                        'Total Value ($)': total_val,
                                                        '% of Portfolio': pct_of_port,
                                                    })

                                                df_table = pd.DataFrame(rows).set_index('Ticker')
                                                # Decide whether to show CASH row: hide if Total Value is zero or Shares zero/NaN
                                                df_display = df_table.copy()
                                                show_cash = False
                                                if 'CASH' in df_display.index:
                                                    cash_val = None
                                                    if 'Total Value ($)' in df_display.columns:
                                                        cash_val = df_display.at['CASH', 'Total Value ($)']
                                                    elif 'Shares' in df_display.columns:
                                                        cash_val = df_display.at['CASH', 'Shares']
                                                    try:
                                                        show_cash = bool(cash_val and not pd.isna(cash_val) and cash_val != 0)
                                                    except Exception:
                                                        show_cash = False
                                                    if not show_cash:
                                                        df_display = df_display.drop('CASH')

                                                # formatting for display
                                                fmt = {
                                                    'Allocation %': '{:,.1f}%',
                                                    'Price ($)': '${:,.2f}',
                                                    'Shares': '{:,.1f}',
                                                    'Total Value ($)': '${:,.2f}',
                                                    '% of Portfolio': '{:,.2f}%'
                                                }
                                                try:
                                                    st.markdown(f"**{label}**")
                                                    
                                                    # Add total row
                                                    total_alloc_pct = df_display['Allocation %'].sum()
                                                    total_value = df_display['Total Value ($)'].sum()
                                                    total_port_pct = df_display['% of Portfolio'].sum()

                                                    total_row = pd.DataFrame({
                                                        'Allocation %': [total_alloc_pct],
                                                        'Price ($)': [float('nan')],
                                                        'Shares': [float('nan')],
                                                        'Total Value ($)': [total_value],
                                                        '% of Portfolio': [total_port_pct]
                                                    }, index=['TOTAL'])

                                                    df_display = pd.concat([df_display, total_row])
                                                    
                                                    sty = df_display.style.format(fmt)
                                                    if 'CASH' in df_table.index and show_cash:
                                                        def _highlight_cash_row(s):
                                                            if s.name == 'CASH':
                                                                return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                                                            return [''] * len(s)
                                                        sty = sty.apply(_highlight_cash_row, axis=1)

                                                    # Highlight TOTAL row
                                                    def _highlight_total_row(s):
                                                        if s.name == 'TOTAL':
                                                            return ['background-color: #1f4e79; color: white; font-weight: bold;' for _ in s]
                                                        return [''] * len(s)
                                                    sty = sty.apply(_highlight_total_row, axis=1)
                                                    
                                                    st.dataframe(sty, use_container_width=True)
                                                except Exception:
                                                    st.dataframe(df_display, use_container_width=True)
                                                
                                                # Create Plotly table figure for PDF generation
                                                try:
                                                    # Reset index to include ticker names
                                                    df_for_table = df_display.reset_index()
                                                    
                                                    # Format the data to ensure 2 decimal places for display - EXACT same as Multi-Backtest
                                                    formatted_values = []
                                                    for col in df_for_table.columns:
                                                        if col in ['Price ($)', 'Total Value ($)', '% of Portfolio']:
                                                            # Format monetary and percentage values to 2 decimal places
                                                            formatted_values.append([f"{df_for_table[col][i]:.2f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                        elif col == 'Shares':
                                                            # Format shares to 1 decimal place
                                                            formatted_values.append([f"{df_for_table[col][i]:.1f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                        elif col == 'Allocation %':
                                                            # Format allocation to 2 decimal places
                                                            formatted_values.append([f"{df_for_table[col][i]:.2f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                        else:
                                                            # Keep other columns as is
                                                            formatted_values.append([str(df_for_table[col][i]) if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                    
                                                    # Create Plotly table
                                                    fig_alloc_table = go.Figure(data=[go.Table(
                                                        header=dict(
                                                            values=list(df_for_table.columns),
                                                            fill_color='#1f77b4',
                                                            align='center',
                                                            font=dict(color='white', size=12)
                                                        ),
                                                        cells=dict(
                                                            values=formatted_values,
                                                            fill_color='#f9f9f9',
                                                            align='center',
                                                            font=dict(size=10),
                                                            height=30
                                                        )
                                                    )])
                                                    
                                                    fig_alloc_table.update_layout(
                                                        title=label,
                                                        height=300,
                                                        margin=dict(t=50, b=20, l=20, r=20)
                                                    )
                                                    
                                                    # Store fig_alloc_table for PDF generation
                                                    st.session_state[f'strategy_comparison_fig_alloc_table_{selected_portfolio_detail}'] = fig_alloc_table
                                                    
                                                except Exception as e:
                                                    st.session_state[f'strategy_comparison_fig_alloc_table_{selected_portfolio_detail}'] = None
                                            
                                            # "Rebalance as of today" table (use momentum-based calculated weights)
                                            build_table_from_alloc(today_weights, None, f"Target Allocation if Rebalanced Today")
                                            
                                except Exception as e:
                                    pass

                                # Other rebalancing plots (smaller, placed after the main one)
                                st.markdown("---")
                                st.markdown("**📊 Historical Rebalancing Comparison**")
                                
                                col_plot1, col_plot2 = st.columns(2)
                                with col_plot1:
                                    st.markdown(f"**Last Rebalance Allocation (as of {last_rebal_date_display.date()})**")
                                    fig_rebal = go.Figure()
                                    fig_rebal.add_trace(go.Pie(labels=labels_rebal, values=vals_rebal, hole=0.3))
                                    fig_rebal.update_traces(textinfo='percent+label')
                                    fig_rebal.update_layout(
                                        template='plotly_dark', 
                                        margin=dict(t=30), 
                                        height=400
                                    )
                                    st.plotly_chart(fig_rebal, use_container_width=True, key=f"multi_rebal_{selected_portfolio_detail}")
                                with col_plot2:
                                    st.markdown(f"**Current Allocation (as of {final_date_display.date()})**")
                                    fig_final = go.Figure()
                                    fig_final.add_trace(go.Pie(labels=labels_final, values=vals_final, hole=0.3))
                                    fig_final.update_traces(textinfo='percent+label')
                                    fig_final.update_layout(
                                        template='plotly_dark', 
                                        margin=dict(t=30), 
                                        height=400
                                    )
                                    st.plotly_chart(fig_final, use_container_width=True, key=f"multi_final_{selected_portfolio_detail}")
                                
                                # Add the three allocation tables from Allocations page
                                try:
                                    # Get portfolio configuration for calculations
                                    portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                    portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                    
                                    if portfolio_cfg:
                                        # Use current portfolio value from backtest results instead of initial value
                                        portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)  # fallback to initial value
                                        
                                        # Get current portfolio value from backtest results
                                        if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
                                            portfolio_results = st.session_state.strategy_comparison_all_results.get(selected_portfolio_detail)
                                            if portfolio_results:
                                                # Use the Final Value (with additions) for Strategy Comparison - total portfolio value including all cash additions and compounding
                                                if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                                    # Get the final value from the with_additions series (includes all cash additions and compounding)
                                                    final_value = portfolio_results['with_additions'].iloc[-1]
                                                    if not pd.isna(final_value) and final_value > 0:
                                                        portfolio_value = float(final_value)
                                                elif isinstance(portfolio_results, dict) and 'no_additions' in portfolio_results:
                                                    # Fallback to no_additions if with_additions not available
                                                    final_value = portfolio_results['no_additions'].iloc[-1]
                                                    if not pd.isna(final_value) and final_value > 0:
                                                        portfolio_value = float(final_value)
                                                elif isinstance(portfolio_results, pd.Series):
                                                    # Get the latest value from the series
                                                    latest_value = portfolio_results.iloc[-1]
                                                    if not pd.isna(latest_value) and latest_value > 0:
                                                        portfolio_value = float(latest_value)
                                        
                                        # Get raw data for price calculations
                                        raw_data = st.session_state.get('strategy_comparison_raw_data', {})
                                        
                                        def _price_on_or_before(df, target_date):
                                            try:
                                                idx = df.index[df.index <= pd.to_datetime(target_date)]
                                                if len(idx) == 0:
                                                    return None
                                                return float(df.loc[idx[-1], 'Close'])
                                            except Exception:
                                                return None

                                        def build_table_from_alloc(alloc_dict, price_date, label):
                                            rows = []
                                            for tk in sorted(alloc_dict.keys()):
                                                alloc_pct = float(alloc_dict.get(tk, 0))
                                                if tk == 'CASH':
                                                    price = None
                                                    shares = 0
                                                    total_val = portfolio_value * alloc_pct
                                                else:
                                                    df = raw_data.get(tk)
                                                    price = None
                                                    if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                        if price_date is None:
                                                            # use latest price
                                                            try:
                                                                price = float(df['Close'].iloc[-1])
                                                            except Exception:
                                                                price = None
                                                        else:
                                                            price = _price_on_or_before(df, price_date)
                                                    try:
                                                        if price and price > 0:
                                                            allocation_value = portfolio_value * alloc_pct
                                                            # allow fractional shares shown to 1 decimal place
                                                            shares = round(allocation_value / price, 1)
                                                            total_val = shares * price
                                                        else:
                                                            shares = 0.0
                                                            total_val = portfolio_value * alloc_pct
                                                    except Exception:
                                                        shares = 0
                                                        total_val = portfolio_value * alloc_pct

                                                pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                                                rows.append({
                                                    'Ticker': tk,
                                                    'Allocation %': alloc_pct * 100,
                                                    'Price ($)': price if price is not None else float('nan'),
                                                    'Shares': shares,
                                                    'Total Value ($)': total_val,
                                                    '% of Portfolio': pct_of_port,
                                                })

                                            df_table = pd.DataFrame(rows).set_index('Ticker')
                                            # Decide whether to show CASH row: hide if Total Value is zero or Shares zero/NaN
                                            df_display = df_table.copy()
                                            show_cash = False
                                            if 'CASH' in df_display.index:
                                                cash_val = None
                                                if 'Total Value ($)' in df_display.columns:
                                                    cash_val = df_display.at['CASH', 'Total Value ($)']
                                                elif 'Shares' in df_display.columns:
                                                    cash_val = df_display.at['CASH', 'Shares']
                                                try:
                                                    show_cash = bool(cash_val and not pd.isna(cash_val) and cash_val != 0)
                                                except Exception:
                                                    show_cash = False
                                                if not show_cash:
                                                    df_display = df_display.drop('CASH')

                                            # formatting for display
                                            fmt = {
                                                'Allocation %': '{:,.1f}%',
                                                'Price ($)': '${:,.2f}',
                                                'Shares': '{:,.1f}',
                                                'Total Value ($)': '${:,.2f}',
                                                '% of Portfolio': '{:,.2f}%'
                                            }
                                            try:
                                                st.markdown(f"**{label}**")
                                                
                                                # Add total row
                                                total_alloc_pct = df_display['Allocation %'].sum()
                                                total_value = df_display['Total Value ($)'].sum()
                                                total_port_pct = df_display['% of Portfolio'].sum()

                                                total_row = pd.DataFrame({
                                                    'Allocation %': [total_alloc_pct],
                                                    'Price ($)': [float('nan')],
                                                    'Shares': [float('nan')],
                                                    'Total Value ($)': [total_value],
                                                    '% of Portfolio': [total_port_pct]
                                                }, index=['TOTAL'])

                                                df_display = pd.concat([df_display, total_row])
                                                
                                                sty = df_display.style.format(fmt)
                                                if 'CASH' in df_table.index and show_cash:
                                                    def _highlight_cash_row(s):
                                                        if s.name == 'CASH':
                                                            return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                                                        return [''] * len(s)
                                                    sty = sty.apply(_highlight_cash_row, axis=1)

                                                # Highlight TOTAL row
                                                def _highlight_total_row(s):
                                                    if s.name == 'TOTAL':
                                                        return ['background-color: #1f4e79; color: white; font-weight: bold;' for _ in s]
                                                    return [''] * len(s)
                                                sty = sty.apply(_highlight_total_row, axis=1)
                                                
                                                st.dataframe(sty, use_container_width=True)
                                            except Exception:
                                                st.dataframe(df_display, use_container_width=True)
                                            
                                            # Create Plotly table figure for PDF generation
                                            try:
                                                # Reset index to include ticker names
                                                df_for_table = df_display.reset_index()
                                                
                                                # Format the data to ensure 2 decimal places for display - EXACT same as Multi-Backtest
                                                formatted_values = []
                                                for col in df_for_table.columns:
                                                    if col in ['Price ($)', 'Total Value ($)', '% of Portfolio']:
                                                        # Format monetary and percentage values to 2 decimal places
                                                        formatted_values.append([f"{df_for_table[col][i]:.2f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                    elif col == 'Shares':
                                                        # Format shares to 1 decimal place
                                                        formatted_values.append([f"{df_for_table[col][i]:.1f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                    elif col == 'Allocation %':
                                                        # Format allocation to 2 decimal places
                                                        formatted_values.append([f"{df_for_table[col][i]:.2f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                    else:
                                                        # Keep other columns as is
                                                        formatted_values.append([str(df_for_table[col][i]) if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                
                                                # Create Plotly table
                                                fig_alloc_table = go.Figure(data=[go.Table(
                                                    header=dict(
                                                        values=list(df_for_table.columns),
                                                        fill_color='#1f77b4',
                                                        align='center',
                                                        font=dict(color='white', size=12)
                                                    ),
                                                    cells=dict(
                                                        values=formatted_values,
                                                        fill_color='#f9f9f9',
                                                        align='center',
                                                        font=dict(size=10),
                                                        height=30
                                                    )
                                                )])
                                                
                                                fig_alloc_table.update_layout(
                                                    title=label,
                                                    height=300,
                                                    margin=dict(t=50, b=20, l=20, r=20)
                                                )
                                                
                                                # Store fig_alloc_table for PDF generation
                                                st.session_state[f'strategy_comparison_fig_alloc_table_{selected_portfolio_detail}'] = fig_alloc_table
                                                
                                            except Exception as e:
                                                st.session_state[f'strategy_comparison_fig_alloc_table_{selected_portfolio_detail}'] = None
                                        
                                        # Last rebalance table (use last_rebal_date)
                                        build_table_from_alloc(rebal_alloc, last_rebal_date, f"Target Allocation at Last Rebalance ({last_rebal_date_display.date()})")
                                        # Current / Today table (use final_date's latest available prices as of now)
                                        build_table_from_alloc(final_alloc, None, f"Portfolio Evolution (Current Allocation)")
                                        
                                except Exception as e:
                                    pass
                                    
                        except Exception as e:
                            pass
                    else:
                        st.info("No allocation history available for this portfolio to show allocation plots.")
                else:
                    # Fallback: show table and plots based on last known allocations so UI stays visible
                    allocs_for_portfolio = st.session_state.strategy_comparison_all_allocations.get(selected_portfolio_detail) if 'strategy_comparison_all_allocations' in st.session_state else None
                    if not allocs_for_portfolio:
                        st.info("No allocation or momentum metrics available for this portfolio.")
                    else:
                        alloc_dates = sorted(list(allocs_for_portfolio.keys()))
                        last_date = alloc_dates[-1]
                        last_alloc = allocs_for_portfolio.get(last_date, {})
                        metrics_records_fb = []
                        for ticker, alloc in last_alloc.items():
                            record = {'Date': last_date, 'Ticker': ticker, 'Momentum': np.nan, 'Beta': np.nan, 'Volatility': np.nan, 'Calculated_Weight': alloc}
                            metrics_records_fb.append(record)

                        metrics_df = pd.DataFrame(metrics_records_fb)
                        metrics_df.set_index(['Date', 'Ticker'], inplace=True)
                        metrics_df_display = metrics_df.copy()
                        if 'Momentum' in metrics_df_display.columns:
                            metrics_df_display['Momentum'] = metrics_df_display['Momentum'].fillna(0) * 100
                        if 'Calculated_Weight' in metrics_df_display.columns:
                            metrics_df_display['Calculated_Weight'] = metrics_df_display['Calculated_Weight'].fillna(0) * 100
                            metrics_df_display['Volatility'] = metrics_df_display['Volatility'].fillna(np.nan) * 100

                        def color_momentum(val):
                            if isinstance(val, (int, float)):
                                color = 'green' if val > 0 else 'red'
                                return f'color: {color}'
                            return ''

                        def highlight_metrics_rows(s):
                            date_str = s.name[0]
                            if s.name[1] == 'CASH':
                                return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                            unique_dates = list(metrics_df_display.index.get_level_values('Date').unique())
                            is_even = unique_dates.index(date_str) % 2 == 0
                            bg_color = 'background-color: #0e1117' if is_even else 'background-color: #262626'
                            return [bg_color] * len(s)

                        fmt_map = {}
                        if 'Momentum' in metrics_df_display.columns:
                            fmt_map['Momentum'] = '{:,.0f}%'
                        if 'Beta' in metrics_df_display.columns:
                            fmt_map['Beta'] = '{:,.2f}'
                        if 'Volatility' in metrics_df_display.columns:
                            fmt_map['Volatility'] = '{:,.2f}%'
                        if 'Calculated_Weight' in metrics_df_display.columns:
                            fmt_map['Calculated_Weight'] = '{:,.0f}%'

                        styler_metrics = metrics_df_display.style.apply(highlight_metrics_rows, axis=1)
                        if 'Momentum' in metrics_df_display.columns:
                            styler_metrics = styler_metrics.map(color_momentum, subset=['Momentum'])
                        if fmt_map:
                            styler_metrics = styler_metrics.format(fmt_map)
                        st.dataframe(styler_metrics, use_container_width=True)

                        # Add timer for next rebalance date (fallback scenario)
                        try:
                            # Get rebalancing frequency from portfolio config
                            portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                            portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                            rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none') if portfolio_cfg else 'none'
                            # Convert to lowercase and map to function expectations
                            rebalancing_frequency = rebalancing_frequency.lower()
                            # Map frequency names to what the function expects
                            frequency_mapping = {
                                'monthly': 'Monthly',
                                'weekly': 'Weekly',
                                'bi-weekly': 'Biweekly',
                                'biweekly': 'Biweekly',
                                'quarterly': 'Quarterly',
                                'semi-annually': 'Semiannually',
                                'semiannually': 'Semiannually',
                                'annually': 'Annually',
                                'yearly': 'Annually',
                                'market_day': 'market_day',
                                'calendar_day': 'calendar_day',
                                'never': 'Never',
                                'none': 'Never'
                            }
                            rebalancing_frequency = frequency_mapping.get(rebalancing_frequency, rebalancing_frequency)
                            
                            if last_date and rebalancing_frequency != 'Never':
                                # Ensure last_date is a naive datetime object
                                if isinstance(last_date, str):
                                    last_date_for_timer = pd.to_datetime(last_date)
                                else:
                                    last_date_for_timer = last_date
                                if hasattr(last_date_for_timer, 'tzinfo') and last_date_for_timer.tzinfo is not None:
                                    last_date_for_timer = last_date_for_timer.replace(tzinfo=None)
                                next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                                    rebalancing_frequency, last_date_for_timer
                                )
                                
                                if next_date and time_until:
                                    st.markdown("---")
                                    st.markdown("**⏰ Next Rebalance Timer**")
                                    
                                    # Create columns for timer display
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            label="Time Until Next Rebalance",
                                            value=format_time_until(time_until),
                                            delta=None
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            label="Target Rebalance Date",
                                            value=next_date.strftime("%B %d, %Y"),
                                            delta=None
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            label="Rebalancing Frequency",
                                            value=rebalancing_frequency.replace('_', ' ').title(),
                                            delta=None
                                        )
                                    
                                    # Add a progress bar showing progress to next rebalance
                                    if rebalancing_frequency in ['week', '2weeks', 'month', '3months', '6months', 'year']:
                                        # Calculate progress percentage
                                        if hasattr(last_date_for_timer, 'to_pydatetime'):
                                            last_rebal_datetime = last_date_for_timer.to_pydatetime()
                                        else:
                                            last_rebal_datetime = last_date_for_timer
                                        
                                        total_period = (next_rebalance_datetime - last_rebal_datetime).total_seconds()
                                        elapsed_period = (datetime.now() - last_rebal_datetime).total_seconds()
                                        progress = min(max(elapsed_period / total_period, 0), 1)
                                        
                                        st.progress(progress, text=f"Progress to next rebalance: {progress:.1%}")
                                    
                                    # Create and store timer table figure for PDF export
                                    try:
                                        timer_data = [
                                            ['Time Until Next Rebalance', format_time_until(time_until)],
                                            ['Target Rebalance Date', next_date.strftime("%B %d, %Y")],
                                            ['Rebalancing Frequency', rebalancing_frequency.replace('_', ' ').title()]
                                        ]
                                        
                                        fig_timer = go.Figure(data=[go.Table(
                                            header=dict(
                                                values=['Parameter', 'Value'],
                                                fill_color='#2E86AB',
                                                align='center',
                                                font=dict(color='white', size=16, family='Arial Black')
                                            ),
                                            cells=dict(
                                                values=[[row[0] for row in timer_data], [row[1] for row in timer_data]],
                                                fill_color=[['#F8F9FA', '#FFFFFF'] * 2, ['#F8F9FA', '#FFFFFF'] * 2],
                                                align='center',
                                                font=dict(color='black', size=14, family='Arial'),
                                                height=40
                                            )
                                        )])
                                        
                                        fig_timer.update_layout(
                                            title=dict(
                                                text="⏰ Next Rebalance Timer",
                                                x=0.5,
                                                font=dict(size=18, color='#2E86AB', family='Arial Black')
                                            ),
                                            width=700,
                                            height=250,
                                            margin=dict(l=20, r=20, t=60, b=20)
                                        )
                                        
                                        # Store in session state for PDF export
                                        st.session_state[f'strategy_comparison_timer_table_{selected_portfolio_detail}'] = fig_timer
                                    except Exception as e:
                                        pass  # Silently ignore timer table creation errors
                                    
                                    # Also create timer tables for ALL portfolios for PDF export
                                    try:
                                        # Get all portfolio configs
                                        all_portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                        snapshot = st.session_state.get('strategy_comparison_snapshot_data', {})
                                        last_rebalance_dates = snapshot.get('last_rebalance_dates', {})
                                        
                                        for portfolio_cfg in all_portfolio_configs:
                                            portfolio_name = portfolio_cfg.get('name', 'Unknown')
                                            
                                            # Get rebalancing frequency for this portfolio
                                            rebal_freq = portfolio_cfg.get('rebalancing_frequency', 'none')
                                            rebal_freq = rebal_freq.lower()
                                            
                                            # Frequency mapping
                                            frequency_mapping = {
                                                'week': '1 week',
                                                '2weeks': '2 weeks', 
                                                'month': '1 month',
                                                '3months': '3 months',
                                                '6months': '6 months',
                                                'year': '1 year'
                                            }
                                            
                                            rebal_freq = frequency_mapping.get(rebal_freq, rebal_freq)
                                            
                                            # Get last rebalance date for this portfolio
                                            last_rebal_date = last_rebalance_dates.get(portfolio_name)
                                            
                                            if rebal_freq != 'none':
                                                # Calculate next rebalance date for this portfolio
                                                if last_rebal_date:
                                                    # Use last rebalance date if available
                                                    if rebal_freq == '1 week':
                                                        next_rebalance_datetime_port = last_rebal_date + timedelta(weeks=1)
                                                    elif rebal_freq == '2 weeks':
                                                        next_rebalance_datetime_port = last_rebal_date + timedelta(weeks=2)
                                                    elif rebal_freq == '1 month':
                                                        next_rebalance_datetime_port = last_rebal_date + timedelta(days=30)
                                                    elif rebal_freq == '3 months':
                                                        next_rebalance_datetime_port = last_rebal_date + timedelta(days=90)
                                                    elif rebal_freq == '6 months':
                                                        next_rebalance_datetime_port = last_rebal_date + timedelta(days=180)
                                                    elif rebal_freq == '1 year':
                                                        next_rebalance_datetime_port = last_rebal_date + timedelta(days=365)
                                                    else:
                                                        next_rebalance_datetime_port = None
                                                else:
                                                    # No last rebalance date, calculate from today
                                                    today = datetime.now()
                                                    if rebal_freq == '1 week':
                                                        next_rebalance_datetime_port = today + timedelta(weeks=1)
                                                    elif rebal_freq == '2 weeks':
                                                        next_rebalance_datetime_port = today + timedelta(weeks=2)
                                                    elif rebal_freq == '1 month':
                                                        next_rebalance_datetime_port = today + timedelta(days=30)
                                                    elif rebal_freq == '3 months':
                                                        next_rebalance_datetime_port = today + timedelta(days=90)
                                                    elif rebal_freq == '6 months':
                                                        next_rebalance_datetime_port = today + timedelta(days=180)
                                                    elif rebal_freq == '1 year':
                                                        next_rebalance_datetime_port = today + timedelta(days=365)
                                                    else:
                                                        next_rebalance_datetime_port = None
                                                
                                                if next_rebalance_datetime_port:
                                                    time_until_port = next_rebalance_datetime_port - datetime.now()
                                                    
                                                    if time_until_port.total_seconds() > 0:
                                                        # Create timer data for this portfolio
                                                        timer_data_port = [
                                                            ['Time Until Next Rebalance', format_time_until(time_until_port)],
                                                            ['Target Rebalance Date', next_rebalance_datetime_port.strftime("%B %d, %Y")],
                                                            ['Rebalancing Frequency', rebal_freq.replace('_', ' ').title()]
                                                        ]
                                                        
                                                        # Create timer table figure for this portfolio
                                                        fig_timer_port = go.Figure(data=[go.Table(
                                                            header=dict(
                                                                values=['Parameter', 'Value'],
                                                                fill_color='#2E86AB',
                                                                align='center',
                                                                font=dict(color='white', size=16, family='Arial Black')
                                                            ),
                                                            cells=dict(
                                                                values=[[row[0] for row in timer_data_port], [row[1] for row in timer_data_port]],
                                                                fill_color=[['#F8F9FA', '#FFFFFF'] * 2, ['#F8F9FA', '#FFFFFF'] * 2],
                                                                align='center',
                                                                font=dict(color='black', size=14, family='Arial'),
                                                                height=40
                                                            )
                                                        )])
                                                        
                                                        fig_timer_port.update_layout(
                                                            title=dict(
                                                                text=f"⏰ Next Rebalance Timer - {portfolio_name}",
                                                                x=0.5,
                                                                font=dict(size=18, color='#2E86AB', family='Arial Black')
                                                            ),
                                                            width=700,
                                                            height=250,
                                                            margin=dict(l=20, r=20, t=60, b=20)
                                                        )
                                                        
                                                        # Store in session state for PDF export
                                                        st.session_state[f'strategy_comparison_timer_table_{portfolio_name}'] = fig_timer_port
                                    except Exception as e:
                                        pass  # Silently ignore timer table creation errors
                        except Exception as e:
                            pass  # Silently ignore timer calculation errors

                        # Main "Rebalance as of today" plot and table for fallback scenario
                        st.markdown("---")
                        st.markdown(f"**🔄 Rebalance as of Today ({pd.Timestamp.now().strftime('%Y-%m-%d')})**")
                        
                        # Get momentum-based calculated weights for today's rebalancing from stored snapshot (fallback scenario)
                        today_weights = {}
                        
                        # Get the stored today_weights_map from snapshot data
                        snapshot = st.session_state.get('strategy_comparison_snapshot_data', {})
                        today_weights_map = snapshot.get('today_weights_map', {}) if snapshot else {}
                        
                        if selected_portfolio_detail in today_weights_map:
                            today_weights = today_weights_map.get(selected_portfolio_detail, {})
                        else:
                            # Fallback to current allocation if no stored weights found
                            final_date = last_date
                            final_alloc = last_alloc
                            today_weights = final_alloc
                        
                        # Create labels and values for the plot
                        labels_today = [k for k, v in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
                        vals_today = [float(today_weights[k]) * 100 for k in labels_today]
                        
                        # Create a larger plot for the main rebalancing representation
                        col_main_plot, col_main_table = st.columns([2, 1])
                        
                        with col_main_plot:
                            st.markdown(f"**Target Allocation if Rebalanced Today**")
                            fig_today = go.Figure()
                            fig_today.add_trace(go.Pie(labels=labels_today, values=vals_today, hole=0.3))
                            fig_today.update_traces(textinfo='percent+label')
                            fig_today.update_layout(
                                template='plotly_dark', 
                                margin=dict(t=30),
                                height=500,  # Make it bigger
                                showlegend=True
                            )
                            # Store fig_today for PDF generation
                            st.session_state[f'strategy_comparison_fig_today_{selected_portfolio_detail}'] = fig_today
                        
                        # Add timer for next rebalance date (fallback scenario) - above the pie chart
                        try:
                            # Get rebalancing frequency from portfolio config
                            portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                            portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                            rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none') if portfolio_cfg else 'none'
                            # Convert to lowercase and map to function expectations
                            rebalancing_frequency = rebalancing_frequency.lower()
                            # Map frequency names to what the function expects
                            frequency_mapping = {
                                'monthly': 'month',
                                'weekly': 'week',
                                'bi-weekly': '2weeks',
                                'biweekly': '2weeks',
                                'quarterly': '3months',
                                'semi-annually': '6months',
                                'semiannually': '6months',
                                'annually': 'year',
                                'yearly': 'year',
                                'market_day': 'market_day',
                                'calendar_day': 'calendar_day',
                                'never': 'none',
                                'none': 'none'
                            }
                            rebalancing_frequency = frequency_mapping.get(rebalancing_frequency, rebalancing_frequency)
                            
                            # Use last_date as the last rebalance date for fallback scenario
                            last_rebal_date_for_timer = last_date
                            
                            if last_rebal_date_for_timer and rebalancing_frequency != 'none':
                                # Calculate next rebalance date and time until
                                next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                                    rebalancing_frequency, last_rebal_date_for_timer
                                )
                                
                                if next_date and time_until:
                                    st.markdown("---")
                                    st.markdown("**⏰ Next Rebalance Timer**")
                                    
                                    # Create columns for timer display (same format as page 1)
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            label="Time Until Next Rebalance",
                                            value=format_time_until(time_until),
                                            delta=None
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            label="Target Rebalance Date",
                                            value=next_date.strftime("%B %d, %Y"),
                                            delta=None
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            label="Rebalancing Frequency",
                                            value=rebalancing_frequency.replace('_', ' ').title(),
                                            delta=None
                                        )
                                    
                                    # Add a progress bar showing progress to next rebalance (same format as page 1)
                                    if rebalancing_frequency in ['week', '2weeks', 'month', '3months', '6months', 'year']:
                                        # Calculate progress percentage
                                        if hasattr(last_rebal_date_for_timer, 'to_pydatetime'):
                                            last_rebal_datetime = last_rebal_date_for_timer.to_pydatetime()
                                        else:
                                            last_rebal_datetime = last_rebal_date_for_timer
                                        
                                        total_period = (next_rebalance_datetime - last_rebal_datetime).total_seconds()
                                        elapsed_period = (datetime.now() - last_rebal_datetime).total_seconds()
                                        progress = min(max(elapsed_period / total_period, 0), 1)
                                        
                                        st.progress(progress, text=f"Progress to next rebalance: {progress:.1%}")
                            else:
                                st.info("⏰ No rebalancing schedule configured for this portfolio.")
                        except Exception as e:
                            pass  # Silently ignore timer calculation errors
                        
                        # Display the pie chart
                        st.plotly_chart(fig_today, use_container_width=True, key=f"multi_today_fallback_{selected_portfolio_detail}")
                        
                        with col_main_table:
                            # Add the "Rebalance as of today" table for fallback
                            try:
                                # Get portfolio configuration for calculations
                                portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                
                                if portfolio_cfg:
                                    # Use current portfolio value from backtest results instead of initial value
                                    portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)  # fallback to initial value
                                    
                                    # Get current portfolio value from backtest results
                                    if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
                                        portfolio_results = st.session_state.strategy_comparison_all_results.get(selected_portfolio_detail)
                                        if portfolio_results:
                                            # Use the Final Value (with additions) for Strategy Comparison - total portfolio value including all cash additions and compounding
                                            if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                                # Get the final value from the with_additions series (includes all cash additions and compounding)
                                                final_value = portfolio_results['with_additions'].iloc[-1]
                                                if not pd.isna(final_value) and final_value > 0:
                                                    portfolio_value = float(final_value)
                                            elif isinstance(portfolio_results, dict) and 'no_additions' in portfolio_results:
                                                # Fallback to no_additions if with_additions not available
                                                final_value = portfolio_results['no_additions'].iloc[-1]
                                                if not pd.isna(final_value) and final_value > 0:
                                                    portfolio_value = float(final_value)
                                            elif isinstance(portfolio_results, pd.Series):
                                                # Get the latest value from the series
                                                latest_value = portfolio_results.iloc[-1]
                                                if not pd.isna(latest_value) and latest_value > 0:
                                                    portfolio_value = float(latest_value)
                                    
                                    # Get raw data for price calculations
                                    raw_data = st.session_state.get('strategy_comparison_raw_data', {})
                                    
                                    def _price_on_or_before(df, target_date):
                                        try:
                                            idx = df.index[df.index <= pd.to_datetime(target_date)]
                                            if len(idx) == 0:
                                                return None
                                            return float(df.loc[idx[-1], 'Close'])
                                        except Exception:
                                            return None

                                    def build_table_from_alloc(alloc_dict, price_date, label):
                                        rows = []
                                        for tk in sorted(alloc_dict.keys()):
                                            alloc_pct = float(alloc_dict.get(tk, 0))
                                            if tk == 'CASH':
                                                price = None
                                                shares = 0
                                                total_val = portfolio_value * alloc_pct
                                            else:
                                                df = raw_data.get(tk)
                                                price = None
                                                if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                    if price_date is None:
                                                        # use latest price
                                                        try:
                                                            price = float(df['Close'].iloc[-1])
                                                        except Exception:
                                                            price = None
                                                    else:
                                                        price = _price_on_or_before(df, price_date)
                                                try:
                                                    if price and price > 0:
                                                        allocation_value = portfolio_value * alloc_pct
                                                        # allow fractional shares shown to 1 decimal place
                                                        shares = round(allocation_value / price, 1)
                                                        total_val = shares * price
                                                    else:
                                                        shares = 0.0
                                                        total_val = portfolio_value * alloc_pct
                                                except Exception:
                                                    shares = 0
                                                    total_val = portfolio_value * alloc_pct

                                            pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                                            rows.append({
                                                'Ticker': tk,
                                                'Allocation %': alloc_pct * 100,
                                                'Price ($)': price if price is not None else float('nan'),
                                                'Shares': shares,
                                                'Total Value ($)': total_val,
                                                '% of Portfolio': pct_of_port,
                                            })

                                        df_table = pd.DataFrame(rows).set_index('Ticker')
                                        # Decide whether to show CASH row: hide if Total Value is zero or Shares zero/NaN
                                        df_display = df_table.copy()
                                        show_cash = False
                                        if 'CASH' in df_display.index:
                                            cash_val = None
                                            if 'Total Value ($)' in df_display.columns:
                                                cash_val = df_display.at['CASH', 'Total Value ($)']
                                            elif 'Shares' in df_display.columns:
                                                cash_val = df_display.at['CASH', 'Shares']
                                            try:
                                                show_cash = bool(cash_val and not pd.isna(cash_val) and cash_val != 0)
                                            except Exception:
                                                show_cash = False
                                            if not show_cash:
                                                df_display = df_display.drop('CASH')

                                        # formatting for display
                                        fmt = {
                                            'Allocation %': '{:,.1f}%',
                                            'Price ($)': '${:,.2f}',
                                            'Shares': '{:,.1f}',
                                            'Total Value ($)': '${:,.2f}',
                                            '% of Portfolio': '{:,.2f}%'
                                        }
                                        try:
                                            st.markdown(f"**{label}**")
                                            sty = df_display.style.format(fmt)
                                            if 'CASH' in df_table.index and show_cash:
                                                def _highlight_cash_row(s):
                                                    if s.name == 'CASH':
                                                        return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                                                sty = sty.apply(_highlight_cash_row, axis=1)
                                            st.dataframe(sty, use_container_width=True)
                                        except Exception:
                                            st.dataframe(df_display, use_container_width=True)
                                        
                                        # Create Plotly table figure for PDF generation
                                        try:
                                            # Reset index to include ticker names
                                            df_for_table = df_display.reset_index()
                                            
                                            # Format the data to ensure 2 decimal places for display - EXACT same as Multi-Backtest
                                            formatted_values = []
                                            for col in df_for_table.columns:
                                                if col in ['Price ($)', 'Total Value ($)', '% of Portfolio']:
                                                    # Format monetary and percentage values to 2 decimal places
                                                    formatted_values.append([f"{df_for_table[col][i]:.2f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                elif col == 'Shares':
                                                    # Format shares to 1 decimal place
                                                    formatted_values.append([f"{df_for_table[col][i]:.1f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                elif col == 'Allocation %':
                                                    # Format allocation to 2 decimal places
                                                    formatted_values.append([f"{df_for_table[col][i]:.2f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                else:
                                                    # Keep other columns as is
                                                    formatted_values.append([str(df_for_table[col][i]) if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                            
                                            # Create Plotly table
                                            fig_alloc_table = go.Figure(data=[go.Table(
                                                header=dict(
                                                    values=list(df_for_table.columns),
                                                    fill_color='#1f77b4',
                                                    align='center',
                                                    font=dict(color='white', size=12)
                                                ),
                                                cells=dict(
                                                    values=formatted_values,
                                                    fill_color='#f9f9f9',
                                                    align='center',
                                                    font=dict(size=10),
                                                    height=30
                                                )
                                            )])
                                            
                                            fig_alloc_table.update_layout(
                                                title=label,
                                                height=300,
                                                margin=dict(t=50, b=20, l=20, r=20)
                                            )
                                            
                                            # Store fig_alloc_table for PDF generation
                                            st.session_state[f'strategy_comparison_fig_alloc_table_{selected_portfolio_detail}'] = fig_alloc_table
                                            
                                        except Exception as e:
                                            st.session_state[f'strategy_comparison_fig_alloc_table_{selected_portfolio_detail}'] = None
                                    
                                # "Rebalance as of today" table for fallback (use momentum-based calculated weights)
                                build_table_from_alloc(today_weights, None, f"Target Allocation if Rebalanced Today")
                                
                            except Exception as e:
                                pass

                        # Other rebalancing plots (smaller, placed after the main one) for fallback
                        st.markdown("---")
                        st.markdown("**📊 Historical Rebalancing Comparison**")
                        
                        col_plot1, col_plot2 = st.columns(2)
                        with col_plot1:
                            st.markdown(f"**Last Rebalance Allocation (as of {last_date.date()})**")
                            fig_rebal = go.Figure()
                            fig_rebal.add_trace(go.Pie(labels=labels_today, values=vals_today, hole=0.3))
                            fig_rebal.update_traces(textinfo='percent+label')
                            fig_rebal.update_layout(template='plotly_dark', margin=dict(t=30), height=400)
                            st.plotly_chart(fig_rebal, use_container_width=True, key=f"multi_rebal_fallback_{selected_portfolio_detail}")
                        with col_plot2:
                            st.markdown(f"**Current Allocation (as of {last_date.date()})**")
                            fig_final = go.Figure()
                            fig_final.add_trace(go.Pie(labels=labels_today, values=vals_today, hole=0.3))
                            fig_final.update_traces(textinfo='percent+label')
                            fig_final.update_layout(template='plotly_dark', margin=dict(t=30), height=400)
                            st.plotly_chart(fig_final, use_container_width=True, key=f"multi_final_fallback_{selected_portfolio_detail}")
                            
                            # Add allocation tables for fallback case as well
                            try:
                                # Get portfolio configuration for calculations
                                portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                
                                if portfolio_cfg:
                                    # Use current portfolio value from backtest results instead of initial value
                                    portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)  # fallback to initial value
                                    
                                    # Get current portfolio value from backtest results
                                    if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
                                        portfolio_results = st.session_state.strategy_comparison_all_results.get(selected_portfolio_detail)
                                        if portfolio_results:
                                            # Use the Final Value (with additions) for Strategy Comparison - total portfolio value including all cash additions and compounding
                                            if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                                # Get the final value from the with_additions series (includes all cash additions and compounding)
                                                final_value = portfolio_results['with_additions'].iloc[-1]
                                                if not pd.isna(final_value) and final_value > 0:
                                                    portfolio_value = float(final_value)
                                            elif isinstance(portfolio_results, dict) and 'no_additions' in portfolio_results:
                                                # Fallback to no_additions if with_additions not available
                                                final_value = portfolio_results['no_additions'].iloc[-1]
                                                if not pd.isna(final_value) and final_value > 0:
                                                    portfolio_value = float(final_value)
                                            elif isinstance(portfolio_results, pd.Series):
                                                # Get the latest value from the series
                                                latest_value = portfolio_results.iloc[-1]
                                                if not pd.isna(latest_value) and latest_value > 0:
                                                    portfolio_value = float(latest_value)
                                    
                                    # Get raw data for price calculations
                                    raw_data = st.session_state.get('strategy_comparison_raw_data', {})
                                    
                                    def _price_on_or_before(df, target_date):
                                        try:
                                            idx = df.index[df.index <= pd.to_datetime(target_date)]
                                            if len(idx) == 0:
                                                return None
                                            return float(df.loc[idx[-1], 'Close'])
                                        except Exception:
                                            return None

                                    def build_table_from_alloc(alloc_dict, price_date, label):
                                        rows = []
                                        for tk in sorted(alloc_dict.keys()):
                                            alloc_pct = float(alloc_dict.get(tk, 0))
                                            if tk == 'CASH':
                                                price = None
                                                shares = 0
                                                total_val = portfolio_value * alloc_pct
                                            else:
                                                df = raw_data.get(tk)
                                                price = None
                                                if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                    if price_date is None:
                                                        # use latest price
                                                        try:
                                                            price = float(df['Close'].iloc[-1])
                                                        except Exception:
                                                            price = None
                                                    else:
                                                        price = _price_on_or_before(df, price_date)
                                                try:
                                                    if price and price > 0:
                                                        allocation_value = portfolio_value * alloc_pct
                                                        # allow fractional shares shown to 1 decimal place
                                                        shares = round(allocation_value / price, 1)
                                                        total_val = shares * price
                                                    else:
                                                        shares = 0.0
                                                        total_val = portfolio_value * alloc_pct
                                                except Exception:
                                                    shares = 0
                                                    total_val = portfolio_value * alloc_pct

                                            pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                                            rows.append({
                                                'Ticker': tk,
                                                'Allocation %': alloc_pct * 100,
                                                'Price ($)': price if price is not None else float('nan'),
                                                'Shares': shares,
                                                'Total Value ($)': total_val,
                                                '% of Portfolio': pct_of_port,
                                            })

                                        df_table = pd.DataFrame(rows).set_index('Ticker')
                                        # Decide whether to show CASH row: hide if Total Value is zero or Shares zero/NaN
                                        df_display = df_table.copy()
                                        show_cash = False
                                        if 'CASH' in df_display.index:
                                            cash_val = None
                                            if 'Total Value ($)' in df_display.columns:
                                                cash_val = df_display.at['CASH', 'Total Value ($)']
                                            elif 'Shares' in df_display.columns:
                                                cash_val = df_display.at['CASH', 'Shares']
                                            try:
                                                show_cash = bool(cash_val and not pd.isna(cash_val) and cash_val != 0)
                                            except Exception:
                                                show_cash = False
                                            if not show_cash:
                                                df_display = df_display.drop('CASH')

                                        # formatting for display
                                        fmt = {
                                            'Allocation %': '{:,.1f}%',
                                            'Price ($)': '${:,.2f}',
                                            'Shares': '{:,.1f}',
                                            'Total Value ($)': '${:,.2f}',
                                            '% of Portfolio': '{:,.2f}%'
                                        }
                                        try:
                                            st.markdown(f"**{label}**")
                                            sty = df_display.style.format(fmt)
                                            if 'CASH' in df_table.index and show_cash:
                                                def _highlight_cash_row(s):
                                                    if s.name == 'CASH':
                                                        return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                                                sty = sty.apply(_highlight_cash_row, axis=1)
                                            st.dataframe(sty, use_container_width=True)
                                        except Exception:
                                            st.dataframe(df_display, use_container_width=True)
                                        
                                        # Create Plotly table figure for PDF generation
                                        try:
                                            # Reset index to include ticker names
                                            df_for_table = df_display.reset_index()
                                            
                                            # Format the data to ensure 2 decimal places for display - EXACT same as Multi-Backtest
                                            formatted_values = []
                                            for col in df_for_table.columns:
                                                if col in ['Price ($)', 'Total Value ($)', '% of Portfolio']:
                                                    # Format monetary and percentage values to 2 decimal places
                                                    formatted_values.append([f"{df_for_table[col][i]:.2f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                elif col == 'Shares':
                                                    # Format shares to 1 decimal place
                                                    formatted_values.append([f"{df_for_table[col][i]:.1f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                elif col == 'Allocation %':
                                                    # Format allocation to 2 decimal places
                                                    formatted_values.append([f"{df_for_table[col][i]:.2f}" if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                                else:
                                                    # Keep other columns as is
                                                    formatted_values.append([str(df_for_table[col][i]) if pd.notna(df_for_table[col][i]) else "" for i in range(len(df_for_table))])
                                            
                                            # Create Plotly table
                                            fig_alloc_table = go.Figure(data=[go.Table(
                                                header=dict(
                                                    values=list(df_for_table.columns),
                                                    fill_color='#1f77b4',
                                                    align='center',
                                                    font=dict(color='white', size=12)
                                                ),
                                                cells=dict(
                                                    values=formatted_values,
                                                    fill_color='#f9f9f9',
                                                    align='center',
                                                    font=dict(size=10),
                                                    height=30
                                                )
                                            )])
                                            
                                            fig_alloc_table.update_layout(
                                                title=label,
                                                height=300,
                                                margin=dict(t=50, b=20, l=20, r=20)
                                            )
                                            
                                            # Store fig_alloc_table for PDF generation
                                            st.session_state[f'strategy_comparison_fig_alloc_table_{selected_portfolio_detail}'] = fig_alloc_table
                                            
                                        except Exception as e:
                                            st.session_state[f'strategy_comparison_fig_alloc_table_{selected_portfolio_detail}'] = None
                                    
                                    # Current allocation table (use final_date's latest available prices as of now)
                                    build_table_from_alloc(final_alloc, None, f"Portfolio Evolution (Current Allocation)")
                                    
                            except Exception as e:
                                pass

        else:
            st.info("Configuration is ready. Press 'Run Backtests' to see results.")
    

    # Allocation Evolution Chart Section
    if 'strategy_comparison_all_allocations' in st.session_state and st.session_state.strategy_comparison_all_allocations:
        st.markdown("---")
        st.markdown("**📈 Portfolio Allocation Evolution**")
        
        # Get all available portfolio names
        portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
        available_portfolio_names = [cfg.get('name', 'Portfolio') for cfg in portfolio_configs]
        extra_names = [n for n in st.session_state.get('strategy_comparison_all_results', {}).keys() if n not in available_portfolio_names]
        all_portfolio_names = available_portfolio_names + extra_names
        
        # Portfolio selector for allocation evolution chart
        if all_portfolio_names:
            selected_portfolio_evolution = st.selectbox(
                "Select portfolio for allocation evolution chart",
                all_portfolio_names,
                key="strategy_comparison_allocation_evolution_portfolio_selector",
                help="Choose which portfolio to show allocation evolution over time"
            )
            
            if selected_portfolio_evolution in st.session_state.strategy_comparison_all_allocations:
                try:
                    # Get allocation data for the selected portfolio
                    allocs_data = st.session_state.strategy_comparison_all_allocations[selected_portfolio_evolution]
                    
                    if allocs_data:
                        # Convert to DataFrame for easier processing
                        alloc_df = pd.DataFrame(allocs_data).T
                        alloc_df.index = pd.to_datetime(alloc_df.index)
                        alloc_df = alloc_df.sort_index()
                        
                        # Get all unique tickers (excluding None)
                        all_tickers = set()
                        for date, allocs in allocs_data.items():
                            for ticker in allocs.keys():
                                if ticker is not None:
                                    all_tickers.add(ticker)
                        all_tickers = sorted(list(all_tickers))
                        
                        # Fill missing values with 0 for unavailable assets (instead of forward fill)
                        alloc_df = alloc_df.fillna(0)
                        
                        # Convert to percentages - same as page 1
                        alloc_df = alloc_df * 100
                        
                        # Create the evolution chart
                        fig_evolution = go.Figure()
                        
                        # Color palette for different tickers
                        colors = [
                            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
                        ]
                        
                        # Add a trace for each ticker - same as page 1
                        for i, ticker in enumerate(all_tickers):
                            if ticker in alloc_df.columns:
                                # Get the allocation data for this ticker
                                ticker_data = alloc_df[ticker].dropna()
                                
                                if not ticker_data.empty:  # Only add if we have data
                                    fig_evolution.add_trace(go.Scatter(
                                        x=ticker_data.index,
                                        y=ticker_data.values,
                                        mode='lines',
                                        name=ticker,
                                        line=dict(color=colors[i % len(colors)], width=2),
                                        hovertemplate=f'<b>{ticker}</b><br>' +
                                                    'Date: %{x}<br>' +
                                                    'Allocation: %{y:.1f}%<br>' +
                                                    '<extra></extra>'
                                    ))
                        
                        # Update layout
                        fig_evolution.update_layout(
                            title=f"Portfolio Allocation Evolution - {selected_portfolio_evolution}",
                            xaxis_title="Date",
                            yaxis_title="Allocation (%)",
                            template='plotly_dark',
                            height=600,
                            hovermode='closest',
                        hoverdistance=100,
                        spikedistance=1000,
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.01
                            )
                        )
                        
                        # Add range selector
                        fig_evolution.update_layout(
                            xaxis=dict(
                                rangeselector=dict(
                                    buttons=list([
                                        dict(count=1, label="1M", step="month", stepmode="backward"),
                                        dict(count=3, label="3M", step="month", stepmode="backward"),
                                        dict(count=6, label="6M", step="month", stepmode="backward"),
                                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                                        dict(step="all")
                                    ])
                                ),
                                rangeslider=dict(visible=False),
                                type="date"
                            )
                        )
                        
                        # Display the chart
                        st.plotly_chart(fig_evolution, use_container_width=True)
                        
                        # Store for PDF export
                        st.session_state[f'strategy_comparison_allocation_evolution_chart_{selected_portfolio_evolution}'] = fig_evolution
                        
                except Exception as e:
                    st.error(f"Error creating allocation evolution chart: {str(e)}")
            else:
                st.info("No allocation data available for the selected portfolio.")
        else:
            st.info("No portfolios available for allocation evolution chart.")
    
    # PE Ratio Evolution Chart Section
    if 'strategy_comparison_all_allocations' in st.session_state and st.session_state.strategy_comparison_all_allocations:
        st.markdown("---")
        st.markdown("**📊 Portfolio PE Ratio Evolution**")
        
        # Get all available portfolio names
        portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
        available_portfolio_names = [cfg.get('name', 'Portfolio') for cfg in portfolio_configs]
        extra_names = [n for n in st.session_state.get('strategy_comparison_all_results', {}).keys() if n not in available_portfolio_names]
        all_portfolio_names = available_portfolio_names + extra_names
        
        if all_portfolio_names:
            selected_portfolio_pe = st.selectbox(
                "Select Portfolio for PE Ratio Analysis:",
                all_portfolio_names,
                key="strategy_comparison_pe_portfolio_selector"
            )
            
            if selected_portfolio_pe:
                # Get allocation data for the selected portfolio
                allocs_data = st.session_state.strategy_comparison_all_allocations.get(selected_portfolio_pe, {})
                
                if allocs_data:
                    try:
                        # Create PE ratio evolution chart
                        fig_pe = go.Figure()
                        
                        # Get all unique tickers and their allocations over time (exclude CASH)
                        all_tickers = set()
                        for date, allocs in allocs_data.items():
                            for ticker in allocs.keys():
                                if ticker is not None and ticker != 'CASH':
                                    all_tickers.add(ticker)
                        all_tickers = sorted(list(all_tickers))
                        
                        if all_tickers:
                            # Fetch PE data for all tickers sequentially to avoid threading issues
                            pe_data = {}
                            with st.spinner("Fetching PE ratio data..."):
                                for ticker in all_tickers:
                                    try:
                                        # Get stock info for PE ratio
                                        stock = yf.Ticker(ticker)
                                        info = stock.info
                                        pe_ratio = info.get('trailingPE', None)
                                        if pe_ratio is not None and pe_ratio > 0:
                                            pe_data[ticker] = pe_ratio
                                    except:
                                        continue
                            
                            if pe_data:
                                # Calculate daily weighted PE ratio using the same approach as portfolio allocation evolution
                                dates = sorted(allocs_data.keys())
                                
                                # Show data range info
                                if dates:
                                    st.info(f"📅 **PE Data Range:** {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
                                    
                                    # Check if data is recent
                                    last_date = dates[-1]
                                    current_date = pd.Timestamp.now()
                                    days_behind = (current_date - last_date).days
                                    
                                    if days_behind > 7:  # More than a week behind
                                        st.warning(f"⚠️ **Data is {days_behind} days behind current date.** To get the latest PE ratios, re-run your backtest to refresh the allocation data.")
                                
                                portfolio_pe_ratios = []
                                
                                for date in dates:
                                    allocs = allocs_data[date]
                                    weighted_pe = 0
                                    total_weight = 0
                                    
                                    # Check if portfolio is in cash (100% cash or no stock allocations)
                                    stock_allocation = sum(weight for ticker, weight in allocs.items() if ticker != 'CASH' and weight > 0)
                                    
                                    if stock_allocation == 0:
                                        # Portfolio is in cash - no PE ratio applicable
                                        portfolio_pe_ratios.append(None)
                                    else:
                                        # Calculate weighted PE only for stock allocations (daily precision)
                                        for ticker, weight in allocs.items():
                                            if ticker != 'CASH' and ticker in pe_data and weight > 0:
                                                weighted_pe += pe_data[ticker] * weight
                                                total_weight += weight
                                        
                                        if total_weight > 0:
                                            portfolio_pe_ratios.append(weighted_pe / total_weight)
                                        else:
                                            portfolio_pe_ratios.append(None)
                                
                                # Use all data including None values to show proper gaps for cash periods
                                if portfolio_pe_ratios:
                                    # Add PE ratio line with gaps for cash periods (smooth line, no markers)
                                    fig_pe.add_trace(go.Scatter(
                                        x=dates,
                                        y=portfolio_pe_ratios,  # Include None values to show gaps
                                        mode='lines',  # Smooth line only, no markers
                                        name=f'Portfolio PE Ratio',
                                        line=dict(color='#00ff88', width=3),  # Bright green
                                        hovertemplate=(
                                            '<b>%{fullData.name}</b><br>' +
                                            'Date: %{x|%Y-%m-%d}<br>' +
                                            'PE Ratio: %{y:.2f}<br>' +
                                            '<extra></extra>'
                                        ),
                                        connectgaps=False  # Show gaps when in cash
                                    ))
                                    
                                    # Calculate statistical metrics (filter out None values for calculations)
                                    clean_pe_ratios = [pe for pe in portfolio_pe_ratios if pe is not None]
                                    if clean_pe_ratios:
                                        median_pe = np.median(clean_pe_ratios)
                                        std_pe = np.std(clean_pe_ratios)
                                        mean_pe = np.mean(clean_pe_ratios)
                                    else:
                                        median_pe = std_pe = mean_pe = 0
                                    
                                    # Add statistical reference lines
                                    fig_pe.add_hline(y=median_pe, line_dash="dash", line_color="blue", 
                                                   annotation_text=f"Median PE: {median_pe:.2f}", annotation_position="top right")
                                    fig_pe.add_hline(y=mean_pe, line_dash="dot", line_color="purple", 
                                                   annotation_text=f"Mean PE: {mean_pe:.2f}", annotation_position="top right")
                                    
                                    # Add multiple standard deviation lines
                                    fig_pe.add_hline(y=mean_pe + std_pe, line_dash="dash", line_color="cyan", 
                                                   annotation_text=f"+1σ: {mean_pe + std_pe:.2f}", annotation_position="top right")
                                    fig_pe.add_hline(y=mean_pe - std_pe, line_dash="dash", line_color="green", 
                                                   annotation_text=f"-1σ: {mean_pe - std_pe:.2f}", annotation_position="top right")
                                    
                                    # Add 2 standard deviation lines
                                    fig_pe.add_hline(y=mean_pe + 2*std_pe, line_dash="dot", line_color="red", 
                                                   annotation_text=f"+2σ: {mean_pe + 2*std_pe:.2f}", annotation_position="top right")
                                    fig_pe.add_hline(y=mean_pe - 2*std_pe, line_dash="dot", line_color="lightgreen", 
                                                   annotation_text=f"-2σ: {mean_pe - 2*std_pe:.2f}", annotation_position="top right")
                                    
                                    # Add 3 standard deviation lines
                                    fig_pe.add_hline(y=mean_pe + 3*std_pe, line_dash="dashdot", line_color="darkred", 
                                                   annotation_text=f"+3σ: {mean_pe + 3*std_pe:.2f}", annotation_position="top right")
                                    fig_pe.add_hline(y=mean_pe - 3*std_pe, line_dash="dashdot", line_color="darkgreen", 
                                                   annotation_text=f"-3σ: {mean_pe - 3*std_pe:.2f}", annotation_position="top right")
                                    
                                    # Update layout with proper date range
                                    fig_pe.update_layout(
                                        title=f"Portfolio PE Ratio Evolution - {selected_portfolio_pe}",
                                        xaxis_title="Date",
                                        yaxis_title="PE Ratio",
                                        template='plotly_dark',
                                        height=500,
                                        hovermode='closest',
                        hoverdistance=100,
                        spikedistance=1000,
                                        showlegend=True,
                                        # Let Plotly handle margins automatically to prevent clipping
                                        xaxis=dict(
                                            type='date',
                                            automargin=True
                                        ),
                                        # Move legend to top
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1
                                        )
                                    )
                                    
                                    # Store in session state
                                    pe_chart_key = f"strategy_comparison_pe_chart_{selected_portfolio_pe}"
                                    st.session_state[pe_chart_key] = fig_pe
                                    
                                    # Display chart
                                    st.plotly_chart(st.session_state[pe_chart_key], use_container_width=True)
                                    
                                    # Show PE data info
                                    st.warning("⚠️ **Work in Progress:** PE ratio calculations are currently using current PE ratios only. Historical PE evolution is not yet implemented and may not be fully accurate.")
                                    
                                    # Show cash periods warning outside the chart
                                    if len(clean_pe_ratios) < len(dates):
                                        st.info("💡 **Note:** Gaps in the chart indicate periods when the portfolio was in cash (no PE ratio applicable)")
                                    
                                    # Show PE data summary with statistical metrics
                                    if clean_pe_ratios:
                                        last_pe = clean_pe_ratios[-1]  # Today's PE (most recent)
                                        median_pe = np.median(clean_pe_ratios)
                                        std_pe = np.std(clean_pe_ratios)
                                        mean_pe = np.mean(clean_pe_ratios)
                                    else:
                                        last_pe = median_pe = std_pe = mean_pe = 0
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Last PE (Today)", f"{last_pe:.2f}")
                                    with col2:
                                        st.metric("Median PE", f"{median_pe:.2f}")
                                    with col3:
                                        st.metric("Mean PE", f"{mean_pe:.2f}")
                                    with col4:
                                        st.metric("Std Deviation", f"{std_pe:.2f}")
                                    
                                    # Show individual ticker PE ratios
                                    st.info(f"📊 **Individual PE Ratios**: {len(pe_data)} tickers with PE data")
                                    pe_cols = st.columns(4)
                                    for i, (ticker, pe) in enumerate(pe_data.items()):
                                        with pe_cols[i % 4]:
                                            st.markdown(f"**{ticker}**: {pe:.2f}")
                                else:
                                    st.warning("No valid PE ratio data available for the selected period.")
                            else:
                                st.warning("No PE ratio data available for any tickers in this portfolio.")
                        else:
                            st.warning("No tickers found in allocation data.")
                    except Exception as e:
                        st.error(f"Error creating PE ratio chart: {str(e)}")
                else:
                    st.warning(f"No allocation data available for {selected_portfolio_pe}")
        else:
            st.info("No portfolios available for PE ratio analysis.")
    
    # Removed duplicate section - using strategy_comparison_all_allocations above
    
    # PDF Export Section
    st.markdown("---")
    st.subheader("📄 PDF Export")
    
    # Optional custom PDF report name
    custom_report_name = st.text_input(
        "📝 Custom Report Name (optional):", 
        value="",
        placeholder="e.g., Growth vs Value Strategy, Risk Analysis 2024, Sector Comparison Study",
        help="Leave empty to use automatic naming: 'Strategy_Comparison_Report_[timestamp].pdf'",
        key="strategy_comparison_custom_report_name"
    )
    
    if st.button("Generate PDF Report", type="primary", use_container_width=True):
        try:
            pdf_buffer = generate_strategy_comparison_pdf_report(custom_report_name)
            if pdf_buffer:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Generate filename based on custom name or default
                if custom_report_name.strip():
                    clean_name = custom_report_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
                    filename = f"{clean_name}_{timestamp}.pdf"
                else:
                    filename = f"Strategy_Comparison_Report_{timestamp}.pdf"
                
                st.success("✅ PDF Report Generated Successfully!")
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("❌ Failed to generate PDF report")
        except Exception as e:
            st.error(f"❌ Error generating PDF: {str(e)}")
            st.exception(e)

