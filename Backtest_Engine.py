import numpy as np
# Backtest_Engine.py
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Quantitative Portfolio Momentum Backtest & Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
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
try:
    import mplcursors
except ImportError:
    mplcursors = None
from scipy.optimize import newton, brentq, root_scalar
import pandas_market_calendars as mcal
from warnings import warn
import plotly.graph_objects as go
import logging
import os
import signal
import sys
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

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
    # Reset running flags to bring back the Run button
    st.session_state.running = False
    st.session_state._run_requested = False
    if "_pending_backtest_params" in st.session_state:
        del st.session_state["_pending_backtest_params"]
    st.rerun()

# =============================================================================
# LEVERAGE ETF SIMULATION FUNCTIONS
# =============================================================================

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

    # Parse leverage parameter first
    if "?L=" in base_ticker:
        try:
            parts = base_ticker.split("?L=", 1)
            base_ticker = parts[0]
            leverage_part = parts[1]
            
            # Check if there are more parameters after leverage
            if "?" in leverage_part:
                leverage_str, remaining = leverage_part.split("?", 1)
                leverage = float(leverage_str)
                base_ticker += "?" + remaining
            else:
                leverage = float(leverage_part)
                
            # Leverage validation removed - allow any leverage value for testing
        except (ValueError, IndexError) as e:
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
                base_ticker += "?" + remaining
            else:
                expense_ratio = float(expense_part)
                
            # Expense ratio validation removed - allow any expense ratio value for testing
        except (ValueError, IndexError) as e:
            expense_ratio = 0.0
            
    return base_ticker.strip(), leverage, expense_ratio

def parse_leverage_ticker(ticker_symbol: str) -> tuple[str, float]:
    """
    Parse ticker symbol to extract base ticker and leverage multiplier.
    Backward compatibility wrapper for parse_ticker_parameters.
    
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

def get_ticker_aliases():
    """Get dictionary of ticker aliases for synthetic tickers"""
    return {
        # Synthetic ticker aliases
        'SPYSIM': 'SPYSIM_COMPLETE',
        'GOLDSIM': 'GOLDSIM_COMPLETE', 
        'GOLDX': 'GOLD_COMPLETE',
        'ZROZX': 'ZROZ_COMPLETE',
        'TLTTR': 'TLT_COMPLETE',
        'BITCOINX': 'BTC_COMPLETE',
        'KMLMX': 'KMLM_COMPLETE',
        'IEFTR': 'IEF_COMPLETE',
        'DBMFX': 'DBMF_COMPLETE',
        'TBILL': 'TBILL_COMPLETE',
        
        # Modern ETF aliases (to avoid conflicts)
        'TLTETF': 'TLT',
        'IEFETF': 'IEF', 
        'TBILL3M': '^IRX',
        
        # Standard aliases
        'SPYTR': '^SP500TR',
        'SP500': '^GSPC',
        'VIX': '^VIX',
        'DXY': 'DX-Y.NYB',
        'GOLD': 'GLD',
        'SILVER': 'SLV',
        'BITCOIN': 'BTC-USD',
        'ETHEREUM': 'ETH-USD',
        'CRYPTO': 'BTC-USD',
        'CRYPTO_BTC': 'BTC-USD',
        'CRYPTO_ETH': 'ETH-USD',
        'TREASURY_10Y': '^TNX',
        'TREASURY_2Y': '^IRX',
        'TREASURY_30Y': '^TYX',
        'REAL_ESTATE': 'VNQ',
        'COMMODITIES': 'DJP',
        'ENERGY': 'XLE',
        'TECH': 'XLK',
        'HEALTHCARE': 'XLV',
        'FINANCIALS': 'XLF',
        'UTILITIES': 'XLU',
        'CONSUMER_DISCRETIONARY': 'XLY',
        'CONSUMER_STAPLES': 'XLP',
        'INDUSTRIALS': 'XLI',
        'MATERIALS': 'XLB',
        'COMMUNICATION': 'XLC',
        'EMERGING_MARKETS': 'EEM',
        'DEVELOPED_MARKETS': 'EFA',
        'SMALL_CAP': 'IWM',
        'MID_CAP': 'IJH',
        'LARGE_CAP': 'SPY',
        'GROWTH': 'VUG',
        'VALUE': 'VTV',
        'QUALITY': 'QUAL',
        'MOMENTUM': 'MTUM',
        'LOW_VOLATILITY': 'USMV',
        'DIVIDEND': 'VYM',
        'HIGH_DIVIDEND': 'HDV',
        'REIT': 'VNQ',
        'INFRASTRUCTURE': 'IGF',
        'CLEAN_ENERGY': 'ICLN',
        'SEMICONDUCTORS': 'SMH',
        'BIOTECH': 'IBB',
        'AEROSPACE': 'ITA',
        'DEFENSE': 'ITA',
        'GOLD_MINERS': 'GDX',
        'SILVER_MINERS': 'SIL',
        'COPPER': 'JJC',
        'OIL': 'USO',
        'NATURAL_GAS': 'UNG',
        'AGRICULTURE': 'DBA',
        'LIVESTOCK': 'COW',
        'CORN': 'CORN',
        'SOYBEANS': 'SOYB',
        'WHEAT': 'WEAT',
        'SUGAR': 'SGG',
        'COFFEE': 'JO',
        'COTTON': 'BAL',
        'LUMBER': 'WOOD',
        'PLATINUM': 'PPLT',
        'PALLADIUM': 'PALL',
        'URANIUM': 'URA',
        'LITHIUM': 'LIT',
        'RARE_EARTH': 'REMX',
        'STEEL': 'SLX',
        'ALUMINUM': 'JJU',
        'NICKEL': 'JJN',
        'ZINC': 'ZINC',
        'LEAD': 'LEDD',
        'TIN': 'JJT',
        'COBALT': 'COBC',
        'MOLYBDENUM': 'MOLY',
        'TUNGSTEN': 'TUNG',
        'VANADIUM': 'VAN',
        'CHROMIUM': 'CHRO',
        'MANGANESE': 'MANG',
        'SILICON': 'SILC',
        'GRAPHITE': 'GRAPH',
        'PHOSPHATE': 'PHOS',
        'POTASH': 'POT',
        'NITROGEN': 'NITR',
        'SULFUR': 'SULF',
        'BORON': 'BOR',
        'FLUORINE': 'FLUO',
        'CHLORINE': 'CHLO',
        'BROMINE': 'BROM',
        'IODINE': 'IODE',
        'SELENIUM': 'SELE',
        'TELLURIUM': 'TELL',
        'GERMANIUM': 'GERM',
        'GALLIUM': 'GALL',
        'INDIUM': 'INDI',
        'THALLIUM': 'THAL',
        'BISMUTH': 'BISM',
        'POLONIUM': 'POLO',
        'ASTATINE': 'ASTA',
        'RADON': 'RADO',
        'FRANCIUM': 'FRAN',
        'RADIUM': 'RADI',
        'ACTINIUM': 'ACTI',
        'THORIUM': 'THOR',
        'PROTACTINIUM': 'PROT',
        'URANIUM_235': 'U235',
        'URANIUM_238': 'U238',
        'PLUTONIUM': 'PLUT',
        'AMERICIUM': 'AMER',
        'CURIUM': 'CURI',
        'BERKELIUM': 'BERK',
        'CALIFORNIUM': 'CALI',
        'EINSTEINIUM': 'EINS',
        'FERMIUM': 'FERM',
        'MENDELEVIUM': 'MEND',
        'NOBELIUM': 'NOBE',
        'LAWRENCIUM': 'LAWR',
        'RUTHERFORDIUM': 'RUTH',
        'DUBNIUM': 'DUBN',
        'SEABORGIUM': 'SEAB',
        'BOHRIUM': 'BOHR',
        'HASSIUM': 'HASS',
        'MEITNERIUM': 'MEIT',
        'DARMSTADTIUM': 'DARM',
        'ROENTGENIUM': 'ROEN',
        'COPERNICIUM': 'COPE',
        'NIHONIUM': 'NIHO',
        'FLEROVIUM': 'FLER',
        'MOSCOVIUM': 'MOSC',
        'LIVERMORIUM': 'LIVE',
        'TENNESSINE': 'TENN',
        'OGANESSON': 'OGAN'
    }

def resolve_ticker_alias(ticker_symbol: str) -> str:
    """
    Resolve ticker alias to actual ticker symbol.
    
    Args:
        ticker_symbol: Ticker symbol that might be an alias
        
    Returns:
        str: Resolved ticker symbol
    """
    aliases = get_ticker_aliases()
    upper_ticker = ticker_symbol.upper()
    
    # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
    if upper_ticker == 'BRK.B':
        upper_ticker = 'BRK-B'
    elif upper_ticker == 'BRK.A':
        upper_ticker = 'BRK-A'
    
    return aliases.get(upper_ticker, upper_ticker)

def get_synthetic_ticker_data(ticker_symbol: str, start_date=None, end_date=None, period=None):
    """
    Get data for synthetic tickers (GOLDX, GOLDSIM, SPYSIM, etc.)
    
    Args:
        ticker_symbol: Synthetic ticker symbol
        start_date: Start date for data
        end_date: End date for data  
        period: Period for data (e.g., 'max', '1y', '5y')
        
    Returns:
        pd.DataFrame: Historical data with Close and Dividends columns
    """
    try:
        if ticker_symbol == "GOLD_COMPLETE":
            from Complete_Tickers.GOLD_COMPLETE_TICKER import create_gold_complete_ticker
            gold_data = create_gold_complete_ticker()
            if gold_data is not None and not gold_data.empty:
                result = pd.DataFrame({
                    'Close': gold_data,
                    'Dividends': [0.0] * len(gold_data)
                }, index=gold_data.index)
                return result
                
        elif ticker_symbol == "GOLDSIM_COMPLETE":
            from Complete_Tickers.GOLDSIM_COMPLETE_TICKER import create_goldsim_complete_ticker
            goldsim_data = create_goldsim_complete_ticker()
            if goldsim_data is not None and not goldsim_data.empty:
                result = pd.DataFrame({
                    'Close': goldsim_data,
                    'Dividends': [0.0] * len(goldsim_data)
                }, index=goldsim_data.index)
                return result
                
        elif ticker_symbol == "SPYSIM_COMPLETE":
            from Complete_Tickers.SPYSIM_COMPLETE_TICKER import create_spysim_complete_ticker
            spysim_data = create_spysim_complete_ticker()
            if spysim_data is not None and not spysim_data.empty:
                result = pd.DataFrame({
                    'Close': spysim_data,
                    'Dividends': [0.0] * len(spysim_data)
                }, index=spysim_data.index)
                return result
                
        elif ticker_symbol == "ZROZ_COMPLETE":
            from Complete_Tickers.ZROZ_COMPLETE_TICKER import create_safe_zroz_ticker
            zroz_data = create_safe_zroz_ticker()
            if zroz_data is not None and not zroz_data.empty:
                result = pd.DataFrame({
                    'Close': zroz_data,
                    'Dividends': [0.0] * len(zroz_data)
                }, index=zroz_data.index)
                return result
                
        elif ticker_symbol == "TLT_COMPLETE":
            from Complete_Tickers.TLT_COMPLETE_TICKER import create_safe_tlt_ticker
            tlt_data = create_safe_tlt_ticker()
            if tlt_data is not None and not tlt_data.empty:
                result = pd.DataFrame({
                    'Close': tlt_data,
                    'Dividends': [0.0] * len(tlt_data)
                }, index=tlt_data.index)
                return result
                
        elif ticker_symbol == "BTC_COMPLETE":
            from Complete_Tickers.BITCOIN_COMPLETE_TICKER import create_bitcoin_complete_ticker
            btc_data = create_bitcoin_complete_ticker()
            if btc_data is not None and not btc_data.empty:
                result = pd.DataFrame({
                    'Close': btc_data,
                    'Dividends': [0.0] * len(btc_data)
                }, index=btc_data.index)
                return result
                
        elif ticker_symbol == "KMLM_COMPLETE":
            from Complete_Tickers.KMLM_COMPLETE_TICKER import create_kmlm_complete_ticker
            kmlm_data = create_kmlm_complete_ticker()
            if kmlm_data is not None and not kmlm_data.empty:
                result = pd.DataFrame({
                    'Close': kmlm_data,
                    'Dividends': [0.0] * len(kmlm_data)
                }, index=kmlm_data.index)
                return result
                
        elif ticker_symbol == "IEF_COMPLETE":
            from Complete_Tickers.IEF_COMPLETE_TICKER import create_ief_complete_ticker
            ief_data = create_ief_complete_ticker()
            if ief_data is not None and not ief_data.empty:
                result = pd.DataFrame({
                    'Close': ief_data,
                    'Dividends': [0.0] * len(ief_data)
                }, index=ief_data.index)
                return result
                
        elif ticker_symbol == "DBMF_COMPLETE":
            from Complete_Tickers.DBMF_COMPLETE_TICKER import create_dbmf_complete_ticker
            dbmf_data = create_dbmf_complete_ticker()
            if dbmf_data is not None and not dbmf_data.empty:
                result = pd.DataFrame({
                    'Close': dbmf_data,
                    'Dividends': [0.0] * len(dbmf_data)
                }, index=dbmf_data.index)
                return result
                
        elif ticker_symbol == "TBILL_COMPLETE":
            from Complete_Tickers.TBILL_COMPLETE_TICKER import create_tbill_complete_ticker
            tbill_data = create_tbill_complete_ticker()
            if tbill_data is not None and not tbill_data.empty:
                result = pd.DataFrame({
                    'Close': tbill_data,
                    'Dividends': [0.0] * len(tbill_data)
                }, index=tbill_data.index)
                return result
                
    except ImportError as e:
        print(f"⚠️ WARNING: Import error for {ticker_symbol}: {e}")
    except Exception as e:
        print(f"⚠️ WARNING: Error creating {ticker_symbol}: {e}")
    
    return pd.DataFrame()

def get_risk_free_rate() -> float:
    """
    Get the current risk-free rate from Yahoo Finance (10-Year Treasury Rate).
    
    Returns:
        float: Annual risk-free rate as a decimal (e.g., 0.045 for 4.5%)
    """
    try:
        # Get optimal risk-free rate using hierarchy: ^IRX → ^FVX → ^TNX → ^TYX
        symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    # Get the most recent rate and convert from percentage to decimal
                    rate = hist['Close'].iloc[-1] / 100.0
                    return rate
            except Exception:
                continue
        
        # Fallback to a reasonable default if all symbols fail
        return 0.045  # 4.5% as default
    except Exception:
        # Fallback to a reasonable default if any error occurs
        return 0.045  # 4.5% as default

def apply_daily_leverage(price_data: pd.DataFrame, leverage: float, expense_ratio: float = 0.0) -> pd.DataFrame:
    """
    Apply daily leverage multiplier to price data, simulating leveraged ETF behavior.
    
    Leveraged ETFs reset daily, so we apply the leverage to daily returns and then
    compound the results to get the leveraged price series. Includes daily cost drag
    equivalent to (leverage - 1) × risk_free_rate.
    
    Args:
        price_data: DataFrame with 'Close' column containing price data
        leverage: Leverage multiplier (e.g., 3.0 for 3x leverage)
        
    Returns:
        DataFrame with leveraged price data including cost drag
    """
    if leverage == 1.0:
        return price_data.copy()
    
    # Create a copy to avoid modifying original data
    leveraged_data = price_data.copy()
    
    # Get time-varying risk-free rates for the entire period
    try:
        logger.debug(f"price_data.index timezone: {getattr(price_data.index, 'tz', None)}")
        risk_free_rates = get_risk_free_rate(price_data.index)
        logger.debug(f"risk_free_rates.index timezone after get_risk_free_rate: {getattr(risk_free_rates.index, 'tz', None)}")
        # Ensure risk-free rates are timezone-naive to match price_data
        if getattr(risk_free_rates.index, "tz", None) is not None:
            risk_free_rates.index = risk_free_rates.index.tz_localize(None)
        logger.debug(f"risk_free_rates.index timezone after tz_localize: {getattr(risk_free_rates.index, 'tz', None)}")
        logger.debug(f"Got risk-free rates: {len(risk_free_rates)} points, range: {risk_free_rates.min():.6f} to {risk_free_rates.max():.6f}")
    except Exception as e:
        logger.error(f"Error getting risk-free rates: {e}")
        raise
    
    # Calculate daily cost drag: (leverage - 1) × risk_free_rate / 252 trading days
    # risk_free_rates is already in daily format, so we don't need to divide by 252
    logger.debug(f"About to calculate daily_cost_drag. leverage: {leverage}, risk_free_rates type: {type(risk_free_rates)}, risk_free_rates.index timezone: {getattr(risk_free_rates.index, 'tz', None)}")
    try:
        daily_cost_drag = (leverage - 1) * risk_free_rates
        logger.debug(f"Successfully calculated daily_cost_drag. Shape: {daily_cost_drag.shape}, index timezone: {getattr(daily_cost_drag.index, 'tz', None)}")
    except Exception as e:
        logger.error(f"Error calculating daily_cost_drag: {e}")
        raise
    
    # Calculate daily expense ratio drag: expense_ratio / 100 / 252 (annual to trading day)
    daily_expense_drag = expense_ratio / 100.0 / 252
    
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

# =============================================================================
# PERFORMANCE OPTIMIZATION: CACHING FUNCTIONS FOR MAIN ENGINE
# =============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_ticker_data(ticker_symbol, start_date=None, end_date=None, period=None, auto_adjust=False):
    """Cache ticker data to dramatically improve performance - MAIN ENGINE VERSION
    
    Args:
        ticker_symbol: Stock ticker symbol (supports leverage format like SPY?L=3)
        start_date: Start date for data (None for period-based)
        end_date: End date for data (None for period-based)
        period: Period string like "max", "1y" (None for date-based)
        auto_adjust: Auto-adjust setting
    """
    try:
        # Resolve ticker alias first
        resolved_ticker = resolve_ticker_alias(ticker_symbol)
        
        # Check if this is a synthetic ticker
        if resolved_ticker.endswith('_COMPLETE'):
            synthetic_data = get_synthetic_ticker_data(resolved_ticker, start_date, end_date, period)
            if not synthetic_data.empty:
                # Parse leverage and expense ratio from original ticker symbol
                base_ticker, leverage, expense_ratio = parse_ticker_parameters(ticker_symbol)
                
                # Apply leverage and expense ratio to synthetic data
                if leverage != 1.0:
                    synthetic_data['Close'] = synthetic_data['Close'] * leverage
                if expense_ratio != 0:
                    # Apply expense ratio as a daily drag
                    daily_expense = (expense_ratio / 100) / 365
                    synthetic_data['Close'] = synthetic_data['Close'] * (1 - daily_expense) ** (synthetic_data.index - synthetic_data.index[0]).days
                
                return synthetic_data[['Close', 'Dividends']]
        
        # Parse leverage and expense ratio from ticker symbol
        base_ticker, leverage, expense_ratio = parse_ticker_parameters(ticker_symbol)
        
        ticker = yf.Ticker(base_ticker)
        
        if period:
            # Period-based download
            hist = ticker.history(period=period, auto_adjust=auto_adjust)
        else:
            # Date-based download
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=auto_adjust, raise_errors=False)
        
        # Ensure we return a valid DataFrame
        if hist is None:
            return pd.DataFrame()
        if not isinstance(hist, pd.DataFrame):
            return pd.DataFrame()
        if hist.empty:
            return pd.DataFrame()
        
        # Process dividends into Dividend_per_share column
        divs = ticker.dividends.copy()
        
        # Ensure dividend dates match the timezone of hist.index
        if getattr(hist.index, "tz", None) is not None:
            # hist.index is timezone-aware, make divs.index timezone-aware too
            if getattr(divs.index, "tz", None) is None:
                divs.index = divs.index.tz_localize(hist.index.tz)
            else:
                divs.index = divs.index.tz_convert(hist.index.tz)
        else:
            # hist.index is timezone-naive, make divs.index timezone-naive too
            if getattr(divs.index, "tz", None) is not None:
                divs.index = divs.index.tz_convert(None)
        
        # Map dividend payments to the next available trading day if not present
        divs_mapped = pd.Series(0.0, index=hist.index)
        for dt, val in divs.items():
            # Find the next available trading day in hist.index
            next_idx = hist.index[hist.index >= dt]
            if len(next_idx) > 0:
                pay_date = next_idx[0]
                divs_mapped[pay_date] += val
        hist["Dividend_per_share"] = divs_mapped
        
        # Apply leverage if specified
        if leverage != 1.0:
            try:
                logger.debug(f"About to apply leverage {leverage} to {base_ticker}. hist.index timezone: {getattr(hist.index, 'tz', None)}")
                hist = apply_daily_leverage(hist, leverage, expense_ratio)
                logger.debug(f"Successfully applied leverage {leverage} to {base_ticker}")
            except Exception as e:
                logger.error(f"Error applying leverage {leverage} to {base_ticker}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return pd.DataFrame()
        else:
            # Add price change calculation for non-leveraged data
            hist["Price_change"] = hist["Close"].pct_change(fill_method=None)
            
        return hist
    except Exception as e:
        logger.error(f"Error fetching data for {ticker_symbol}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_ticker_download(ticker_symbol, start_date=None, end_date=None, progress=False):
    """Cache yf.download calls for fallback scenarios"""
    try:
        result = yf.download(ticker_symbol, start=start_date, end=end_date, progress=progress)
        
        # Ensure we return a valid DataFrame
        if result is None:
            return pd.DataFrame()
        if not isinstance(result, pd.DataFrame):
            return pd.DataFrame()
        if result.empty:
            return pd.DataFrame()
            
        return result
    except Exception as e:
        logger.debug(f"Error downloading data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def get_multiple_tickers_batch(ticker_list, period="max", auto_adjust=False):
    """
    Smart batch download with fallback to individual downloads.
    
    Strategy:
    1. Try batch download (fast - 1 API call for all tickers)
    2. If batch fails → fallback to individual downloads (reliable)
    3. Invalid tickers are skipped, others continue
    
    Args:
        ticker_list: List of ticker symbols (can include leverage format)
        period: Data period
        auto_adjust: Auto-adjust setting
    
    Returns:
        Dict[ticker_symbol, DataFrame]: Data for each ticker
    """
    if not ticker_list:
        return {}
    
    results = {}
    
    # Separate tickers into Yahoo-fetchable vs custom
    yahoo_tickers = []
    custom_tickers = {}
    
    for ticker_symbol in ticker_list:
        # Parse parameters
        base_ticker, leverage, expense_ratio = parse_ticker_parameters(ticker_symbol)
        resolved = resolve_ticker_alias(base_ticker)
        
        # Check if it's a custom ticker (local data, no Yahoo call)
        custom_list = ["ZEROX", "GOLD_COMPLETE", "ZROZ_COMPLETE", "TLT_COMPLETE", 
                      "BTC_COMPLETE", "IEF_COMPLETE", "KMLM_COMPLETE", "DBMF_COMPLETE",
                      "TBILL_COMPLETE", "SPYSIM_COMPLETE", "GOLDSIM_COMPLETE"]
        
        if resolved in custom_list:
            # Handle custom tickers individually (they don't use Yahoo)
            custom_tickers[ticker_symbol] = (resolved, leverage, expense_ratio)
        else:
            yahoo_tickers.append((ticker_symbol, resolved, leverage, expense_ratio))
    
    # Process custom tickers first (no Yahoo calls)
    for ticker_symbol, (resolved, leverage, expense_ratio) in custom_tickers.items():
        try:
            results[ticker_symbol] = get_cached_ticker_data(ticker_symbol, None, None, period, auto_adjust)
        except:
            results[ticker_symbol] = pd.DataFrame()
    
    # If no Yahoo tickers, return early
    if not yahoo_tickers:
        return results
    
    # Extract just the resolved tickers for batch download
    resolved_list = list(set([resolved for _, resolved, _, _ in yahoo_tickers]))
    
    try:
        # BATCH DOWNLOAD - Fast path (1 API call for all)
        if len(resolved_list) > 1:
            batch_data = yf.download(
                resolved_list,
                period=period,
                auto_adjust=auto_adjust,
                progress=False,
                group_by='ticker'
            )
            
            # Process batch data
            if not batch_data.empty:
                for ticker_symbol, resolved, leverage, expense_ratio in yahoo_tickers:
                    try:
                        if len(resolved_list) > 1:
                            # Multi-ticker batch
                            ticker_data = batch_data[resolved][['Close', 'Dividends']] if resolved in batch_data else pd.DataFrame()
                        else:
                            # Single ticker batch
                            ticker_data = batch_data[['Close', 'Dividends']]
                        
                        if not ticker_data.empty:
                            # Apply leverage/expense if needed
                            if leverage != 1.0 or expense_ratio != 0.0:
                                ticker_data = apply_daily_leverage(ticker_data, leverage, expense_ratio)
                            results[ticker_symbol] = ticker_data
                        else:
                            results[ticker_symbol] = pd.DataFrame()
                    except:
                        # Individual ticker failed in batch, will retry below
                        pass
            else:
                raise Exception("Batch download returned empty")
                
    except Exception:
        # FALLBACK - Batch failed, download individually (reliable but slower)
        pass
    
    # Download any missing tickers individually (fallback or single ticker)
    for ticker_symbol, resolved, leverage, expense_ratio in yahoo_tickers:
        if ticker_symbol not in results or results[ticker_symbol].empty:
            try:
                ticker = yf.Ticker(resolved)
                hist = ticker.history(period=period, auto_adjust=auto_adjust)[["Close", "Dividends"]]
                
                if not hist.empty:
                    # Apply leverage/expense if needed
                    if leverage != 1.0 or expense_ratio != 0.0:
                        hist = apply_daily_leverage(hist, leverage, expense_ratio)
                    results[ticker_symbol] = hist
                else:
                    results[ticker_symbol] = pd.DataFrame()
            except:
                results[ticker_symbol] = pd.DataFrame()
    
    return results

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
    return obj
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

def get_risk_free_rate_robust(dates):
    """Working risk-free rate fetcher that bypasses Streamlit caching issues."""
    try:
        dates = pd.to_datetime(dates)
        if isinstance(dates, pd.DatetimeIndex):
            if getattr(dates, "tz", None) is not None:
                dates = dates.tz_convert(None)
        
        logger.info("🚀 Starting risk-free rate fetch...")
        
        # Try multiple treasury symbols in order of preference
        symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]
        
        for symbol in symbols:
            try:
                logger.debug(f"Trying {symbol}...")
                ticker = yf.Ticker(symbol)
                
                # Calculate the date range we need
                start_date = dates.min() - pd.Timedelta(days=30)  # Get some extra data before
                end_date = dates.max() + pd.Timedelta(days=1)     # Get some extra data after
                
                # Try to get data for the specific date range
                try:
                    hist = ticker.history(start=start_date, end=end_date, auto_adjust=False)
                    
                    if hist is not None and not hist.empty and 'Close' in hist.columns:
                        # Check if we have valid data
                        valid_data = hist[hist['Close'].notnull() & (hist['Close'] > 0)]
                        if not valid_data.empty:
                            logger.info(f"✅ SUCCESS with {symbol}: {len(valid_data)} rows, latest rate: {valid_data['Close'].iloc[-1]:.2f}%")
                            
                            # Process the data
                            return _process_treasury_data(valid_data, dates)
                            
                except Exception as e:
                    logger.debug(f"Date range failed for {symbol}: {e}")
                    continue
                
            except Exception as e:
                logger.debug(f"Symbol {symbol} failed: {e}")
                continue
        
        # If all symbols fail, use default
        logger.warning("⚠️  All treasury symbols failed, using default 2% rate")
        return _get_default_risk_free_rate(dates)
        
    except Exception as e:
        logger.error(f"Critical error in risk-free rate fetching: {e}")
        return _get_default_risk_free_rate(dates)

# Removed old functions - now using the working approach directly in get_risk_free_rate_robust

# Removed old unused functions - now using the working approach directly

# Removed unused validation functions - now using simple validation inline

def _process_treasury_data(hist, dates):
    """Process treasury data into daily risk-free rates."""
    try:
        hist = _ensure_naive_index(hist)
        hist = hist[hist['Close'].notnull() & (hist['Close'] > 0)]
        
        if hist.empty:
            logger.debug("No valid treasury data after filtering")
            return None
        
        # Convert percentage to decimal
        annual_rate = hist['Close'] / 100.0
        
        # Convert to daily rate using 365.25 calendar days
        # Leveraged ETFs operate daily (including weekends/holidays), so financing costs accrue continuously
        with np.errstate(over='ignore', invalid='ignore'):
            daily_rate = (1 + annual_rate) ** (1 / 365.25) - 1.0
        
        # Ensure daily_rate is timezone-naive
        daily_rate = _ensure_naive_index(daily_rate)
        
        # Align to requested dates
        target_index = pd.to_datetime(dates)
        if getattr(target_index, 'tz', None) is not None:
            target_index = target_index.tz_convert(None)
        
        # Create a series with the daily rates (ensure timezone-naive index)
        # Make sure the index is timezone-naive before creating the series
        naive_index = daily_rate.index
        if getattr(naive_index, "tz", None) is not None:
            naive_index = naive_index.tz_localize(None)
        daily_rate_series = pd.Series(daily_rate.values, index=naive_index)
        
        # If we only have one data point, use it for all dates
        if len(daily_rate_series) == 1:
            logger.debug(f"Single data point: using rate {daily_rate_series.iloc[0]:.6f} for all dates")
            result = pd.Series([daily_rate_series.iloc[0]] * len(target_index), index=target_index)
        else:
            # Reindex to include all target dates and forward fill
            daily_rate_series = daily_rate_series.reindex(daily_rate_series.index.union(target_index)).ffill()
            result = daily_rate_series.reindex(target_index).ffill().fillna(0)
        
        # CRITICAL: Ensure the final result has a timezone-naive index
        result = _ensure_naive_index(result)
        
        logger.debug(f"Processed treasury data: {len(result)} points, rate range: {result.min():.6f} to {result.max():.6f}")
        return result
        
    except Exception as e:
        logger.debug(f"Error processing treasury data: {e}")
        return None

def _get_default_risk_free_rate(dates):
    """Get default risk-free rate when all other methods fail."""
    logger.warning("Using default 2% annual risk-free rate - Yahoo Finance treasury data unavailable")
    default_daily = (1 + 0.02) ** (1 / 365.25) - 1
    return pd.Series(default_daily, index=pd.to_datetime(dates))

def test_risk_free_rate_system():
    """Test function to verify risk-free rate fetching works."""
    logger.info("🧪 Testing risk-free rate system...")
    
    # Create test date range (last 30 days)
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=30)
    test_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    try:
        result = get_risk_free_rate_robust(test_dates)
        
        if result is not None and not result.empty:
            logger.info(f"✅ Test successful!")
            logger.info(f"   Data points: {len(result)}")
            logger.info(f"   Date range: {result.index.min().date()} to {result.index.max().date()}")
            logger.info(f"   Rate range: {result.min():.6f} to {result.max():.6f} (daily)")
            logger.info(f"   Annual equivalent: {result.mean() * 365.25 * 100:.2f}%")
            
            # Check if we got real data or default
            default_daily = (1 + 0.02) ** (1 / 365.25) - 1
            if abs(result.mean() - default_daily) < 1e-6:
                logger.warning("⚠️  Using default rate - no real treasury data available")
            else:
                logger.info("✅ Real treasury data successfully fetched!")
            
            return True
        else:
            logger.error("❌ Test failed - no data returned")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def get_risk_free_rate(dates):
    """Downloads the risk-free rate (IRX) and aligns it to a given date range.
    
    This function now uses the robust approach with multiple fallback strategies.
    """
    logger.debug(f"get_risk_free_rate called with dates type: {type(dates)}, timezone: {getattr(dates, 'tz', None) if hasattr(dates, 'tz') else 'No tz attr'}")
    result = get_risk_free_rate_robust(dates)
    logger.debug(f"get_risk_free_rate returning result type: {type(result)}, timezone: {getattr(result.index, 'tz', None) if hasattr(result, 'index') else 'No index attr'}")
    # CRITICAL: Ensure the result is timezone-naive
    if hasattr(result, 'index') and getattr(result.index, 'tz', None) is not None:
        logger.debug("Converting result to timezone-naive")
        result = _ensure_naive_index(result)
        logger.debug(f"After conversion: timezone: {getattr(result.index, 'tz', None)}")
    return result

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
            logger.debug(f"Loading data for ticker: {t}")
            # Get data with leverage processing handled in get_cached_ticker_data
            hist = get_cached_ticker_data(t, start_date=start_date, end_date=yf_end_date, auto_adjust=False)

            if hist.empty:
                logger.warning(f"No data available for {t}")
                invalid_tickers.append(t)
                continue

            # Force tz-naive for hist
            hist = hist.copy()
            hist.index = hist.index.tz_localize(None)

            # Clip to end_date in case yf_end_date brought extra rows
            hist = hist[hist.index <= end_date]

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

    # Base start: either oldest or latest depending on choice (same logic as Multi Backtest)
    if start_with == "oldest":
        # For "oldest", use the earliest start date among portfolio assets
        base_start = oldest_data_start
    else:
        # For "all", use the latest start date when all assets are available
        base_start = latest_data_start
    
    # Apply user date constraints if any
    if start_date_user_dt:
        if start_with == "oldest":
            # For "oldest", only allow custom start date if it's earlier than the oldest
            if start_date_user_dt < base_start:
                base_start = start_date_user_dt
        else:
            # For "all", only allow custom start date if it's later than the latest
            if start_date_user_dt > base_start:
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
    # Respect the start_with parameter: use intersection for "all", union for "oldest"
    all_trading_days = None
    for t, d in data.items():
        if t in portfolio_tickers and not d.empty:
            # Get trading days from this stock's data within our range
            stock_dates = set(d.index[(d.index >= backtest_start) & (d.index <= backtest_end)])
            if all_trading_days is None:
                all_trading_days = stock_dates
            else:
                if start_with == "all":
                    # Use intersection to ensure ALL stocks have data on each date
                    all_trading_days = all_trading_days.intersection(stock_dates)
                else:  # start_with == "oldest"
                    # Use union to include all dates when any stock has data
                    all_trading_days = all_trading_days.union(stock_dates)
    
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
        # First trading day on or after the 1st of each month
        monthly_dates = []
        for year in sorted(set(all_dates.year)):
            for month in range(1, 13):
                # Create 1st of this month
                first_of_month = pd.Timestamp(year=year, month=month, day=1)
                # Find the first trading day on or after the 1st
                mask = all_dates >= first_of_month
                if mask.any():
                    monthly_dates.append(all_dates[mask][0])
        return pd.DatetimeIndex(monthly_dates)
    elif rebalancing_frequency == "Quarterly":
        # First trading day on or after Jan 1, Apr 1, Jul 1, Oct 1
        quarterly_dates = []
        for year in sorted(set(all_dates.year)):
            for month in [1, 4, 7, 10]:  # January, April, July, October
                # Create 1st of this quarter month
                quarter_start = pd.Timestamp(year=year, month=month, day=1)
                # Find the first trading day on or after the quarter start
                mask = all_dates >= quarter_start
                if mask.any():
                    quarterly_dates.append(all_dates[mask][0])
        return pd.DatetimeIndex(quarterly_dates)
    elif rebalancing_frequency == "Semiannually":
        # First trading day on or after Jan 1, Jul 1
        semiannual_dates = []
        for year in sorted(set(all_dates.year)):
            for month in [1, 7]:  # January and July
                # Create 1st of this semi-annual month
                semi_start = pd.Timestamp(year=year, month=month, day=1)
                # Find the first trading day on or after the semi-annual start
                mask = all_dates >= semi_start
                if mask.any():
                    semiannual_dates.append(all_dates[mask][0])
        return pd.DatetimeIndex(semiannual_dates)
    elif rebalancing_frequency == "Annually":
        # First trading day on or after January 1st each year
        annual_dates = []
        for year in sorted(set(all_dates.year)):
            # Create January 1st of this year
            jan_1st = pd.Timestamp(year=year, month=1, day=1)
            # Find the first trading day on or after January 1st
            mask = all_dates >= jan_1st
            if mask.any():
                annual_dates.append(all_dates[mask][0])
        return pd.DatetimeIndex(annual_dates)
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
        # First available trading day of each month
        months = all_dates.to_series().groupby([all_dates.year, all_dates.month]).first()
        return pd.DatetimeIndex(months.values)
    elif added_frequency == "Quarterly":
        # First available trading day of each quarter
        quarters = all_dates.to_series().groupby([all_dates.year, all_dates.quarter]).first()
        return pd.DatetimeIndex(quarters.values)
    elif added_frequency == "Annually":
        return all_dates[all_dates.is_year_end]
    else:
        return pd.DatetimeIndex([])

# =============================
# NEW MOMENTUM LOGIC
# =============================

def calculate_momentum(date, current_assets, momentum_windows, data_dict, include_dividends=None):
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
                
                # ACADEMIC FIX: Include dividends in momentum calculation if configured (Jegadeesh & Titman 1993)
                if include_dividends and include_dividends.get(t, False):
                    # Calculate cumulative dividends in the momentum window
                    # Backtest_Engine uses "Dividend_per_share" column, not "Dividends"
                    if "Dividend_per_share" in df_t.columns:
                        divs_in_period = df_t.loc[price_start_index:price_end_index, "Dividend_per_share"].fillna(0).sum()
                    else:
                        divs_in_period = 0.0
                    ret = ((price_end + divs_in_period) - price_start) / price_start
                else:
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
    rebalancing_frequency=None,
    current_asset_values=None,
    include_dividends=None,
):
    """Rebalancing now uses the new, more robust momentum logic."""
    global data, calc_beta, calc_volatility, beta_window_days, exclude_days_beta, benchmark_ticker, vol_window_days, exclude_days_vol, use_relative_momentum
    data = data_dict
    calc_beta = use_beta_flag
    calc_volatility = use_volatility_flag
    beta_window_days = beta_window_days_val
    exclude_days_beta = exclude_days_beta_val
    # Resolve benchmark ticker alias
    benchmark_ticker = resolve_ticker_alias(benchmark_ticker_val)
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

    # Handle Buy & Hold strategies - they don't rebalance existing positions, only add new cash
    if rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"]:
        # For buy and hold, total_portfolio_value is actually just the cash to distribute
        
        if rebalancing_frequency == "Buy & Hold":
            # Use current proportions from existing holdings
            if current_asset_values and sum(current_asset_values.values()) > 0:
                # Calculate current proportions based on existing holdings
                total_current_value = sum(current_asset_values.values())
                current_proportions = {t: current_asset_values.get(t, 0) / total_current_value for t in tradable_tickers_today}
            else:
                # If no current holdings, use equal weights
                current_proportions = {t: 1.0 / len(tradable_tickers_today) for t in tradable_tickers_today}
            
            for t in tradable_tickers_today:
                target_allocation[t] = total_portfolio_value * current_proportions.get(t, 0)
                rebalance_metrics["target_allocation"][t] = current_proportions.get(t, 0)
        else:  # "Buy & Hold (Target)"
            # Use initial target allocations
            alloc_sum_available = sum(allocations.get(t, 0.0) for t in tradable_tickers_today)
            if alloc_sum_available <= 0:
                alloc_sum_available = sum(allocations.values()) if sum(allocations.values()) > 0 else 1.0
            
            for t in tradable_tickers_today:
                normalized_alloc = allocations.get(t, 0.0) / alloc_sum_available
                target_allocation[t] = total_portfolio_value * normalized_alloc
                rebalance_metrics["target_allocation"][t] = normalized_alloc
        
        return target_allocation, rebalance_metrics

    if use_momentum:
        returns, valid_assets = calculate_momentum(current_date, set(tradable_tickers_today), momentum_windows, data, include_dividends)
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
    
    # CRITICAL FIX: Add base tickers for leveraged tickers to ensure dividend data is available
    base_tickers_to_add = set()
    for ticker in all_tickers_to_fetch:
        if "?L=" in ticker or "?E=" in ticker:
            base_ticker, leverage = parse_leverage_ticker(ticker)
            base_tickers_to_add.add(base_ticker)
    
    # Add base tickers to the list if they're not already there
    for base_ticker in base_tickers_to_add:
        if base_ticker not in all_tickers_to_fetch:
            all_tickers_to_fetch.append(base_ticker)
    
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
                    # CRITICAL FIX: For leveraged tickers, dividends should be handled differently
                    # The dividend RATE should be the same as the base asset
                    if "?L=" in t or "?E=" in t:
                        # For leveraged tickers, get the base ticker's dividend rate (not amount)
                        base_ticker, leverage = parse_leverage_ticker(t)
                        if base_ticker in data:
                            base_ticker_data = data[base_ticker]
                            if not base_ticker_data.empty and current_date in base_ticker_data.index:
                                base_price = float(base_ticker_data.loc[current_date, "Close"])
                                if base_price > 0:
                                    # Use the base ticker's dividend rate, not the leveraged amount
                                    dividend_rate = dividend / base_price
                                    dividend_cash = shares_with * price * dividend_rate
                                    dividend_cash_wo = shares_without * price * dividend_rate
                                else:
                                    # Fallback: use original logic
                                    dividend_cash = shares_with * dividend
                                    dividend_cash_wo = shares_without * dividend
                            else:
                                # Fallback: use original logic
                                dividend_cash = shares_with * dividend
                                dividend_cash_wo = shares_without * dividend
                        else:
                            # Fallback: use original logic
                            dividend_cash = shares_with * dividend
                            dividend_cash_wo = shares_without * dividend
                    else:
                        # For regular tickers, use normal dividend calculation
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
            
            # For Buy & Hold strategies, only distribute the new cash, not rebalance entire portfolio
            if rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"]:
                # Only distribute the new cash amount
                cash_to_distribute = cash_with_additions
                target_alloc_with, rebalance_metrics = _rebalance_portfolio(
                    current_date, 
                    cash_to_distribute,  # Only the new cash, not total portfolio
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
                    exclude_days_vol,
                    rebalancing_frequency,
                    asset_values_with_additions,
                    include_dividends
                )
            else:
                # Normal rebalancing for other strategies
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
                    exclude_days_vol,
                    rebalancing_frequency,
                    asset_values_with_additions,
                    include_dividends
                )
            # Store rebalance metrics ONLY on true rebalancing dates
            rebalance_metrics_list.append(rebalance_metrics)
            
            # Record last rebalance allocation
            last_rebalance_allocations = rebalance_metrics["target_allocation"]

            # For Buy & Hold strategies, only add new shares without touching existing holdings
            if rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"]:
                # Keep existing shares and only add new shares from cash distribution
                new_shares = asset_shares_with_additions.copy()
                new_cash = cash_with_additions
                
                for t, target_val in target_alloc_with.items():
                    # Use last available price on or before current_date
                    df_t = data.get(t, pd.DataFrame())
                    if not df_t.empty:
                        # Find the last available price before or on current_date
                        price_idx = df_t.index.asof(current_date)
                        if pd.notna(price_idx):
                            price = float(df_t.loc[price_idx, "Close"])
                            if price > 0:
                                # Add new shares to existing shares
                                new_shares.loc[t] += target_val / price
                                new_cash -= target_val
                
                asset_shares_with_additions = new_shares
                cash_with_additions = new_cash
            else:
                # Normal rebalancing: Reset shares to zero and calculate new shares
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
                exclude_days_vol,
                rebalancing_frequency,
                None,
                include_dividends
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
_ss_default("added_amount", 10000)
_ss_default("rebalancing_frequency", "Monthly")
_ss_default("added_frequency", "Annually")
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
            
            # Clean up widget states for the removed ticker
            ticker_key = f"ticker_{idx}"
            div_key = f"divs_checkbox_{idx}"
            alloc_key = f"alloc_input_{idx}"
            remove_key = f"remove_ticker_{idx}"
            
            # Remove widget states to prevent IndexError
            for key in [ticker_key, div_key, alloc_key, remove_key]:
                if key in st.session_state:
                    del st.session_state[key]
            
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
    """Callback for ticker input to convert commas to dots and to uppercase"""
    try:
        key = f"ticker_{index}"
        val = st.session_state.get(key, None)
        if val is not None:
            # Convert commas to dots for decimal separators (like case conversion)
            converted_val = val.replace(",", ".")
            
            # Convert the input value to uppercase
            upper_val = converted_val.upper()
            
            # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
            if upper_val == 'BRK.B':
                upper_val = 'BRK-B'
            elif upper_val == 'BRK.A':
                upper_val = 'BRK-A'
            
            # CRITICAL: Resolve ticker alias BEFORE storing
            resolved_ticker = resolve_ticker_alias(upper_val)
            
            # Update the portfolio configuration with the resolved ticker
            st.session_state.tickers[index] = resolved_ticker
            
            # Update the text box's state to show the resolved ticker
            st.session_state[key] = resolved_ticker
            
            # Auto-disable dividends for negative leverage (inverse ETFs)
            if '?L=-' in resolved_ticker:
                # Ensure divs list exists and has enough elements
                if 'divs' not in st.session_state:
                    st.session_state.divs = [True] * len(st.session_state.tickers)
                elif len(st.session_state.divs) <= index:
                    st.session_state.divs.extend([True] * (index + 1 - len(st.session_state.divs)))
                
                # Set dividends to False for inverse ETFs
                st.session_state.divs[index] = False
                
                # Also update the checkbox UI state
                div_key = f"divs_checkbox_{index}"
                st.session_state[div_key] = False
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
    # The radio button automatically updates the session state via the key
    # This callback ensures the value is properly synchronized
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
    st.session_state.added_amount = 10000
    st.session_state.rebalancing_frequency = "Monthly"
    st.session_state.added_frequency = "Annually"
    # Reset widget keys to match
    st.session_state["initial_value_input"] = 10000
    st.session_state["initial_value_input_int"] = 10000
    st.session_state["added_amount_input"] = 10000
    st.session_state["added_amount_input_int"] = 10000
    st.session_state["rebalancing_frequency_widget"] = "Monthly"
    st.session_state["added_frequency_widget"] = "Annually"
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
            # EXACT SAME SYSTEM AS PAGE 1 MULTI BACKTEST
            div_key = f"backtest_engine_div_{i}"
            # Ensure include_dividends key exists with default value
            if i < len(st.session_state.divs):
                include_dividends_default = st.session_state.divs[i]
            else:
                include_dividends_default = True
            
            # Auto-disable dividends for negative leverage (inverse ETFs) ONLY on first display
            # Don't override if user has explicitly set a value
            if '?L=-' in st.session_state.tickers[i] and div_key not in st.session_state:
                include_dividends_default = False
            
            if div_key not in st.session_state:
                st.session_state[div_key] = include_dividends_default
            st.checkbox("Reinvest Dividends", key=div_key)
            if st.session_state[div_key] != include_dividends_default and i < len(st.session_state.divs):
                st.session_state.divs[i] = st.session_state[div_key]
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

    # Bulk ticker input section
    with st.expander("📝 Bulk Ticker Input", expanded=False):
        st.markdown("**Enter multiple tickers separated by spaces or commas:**")
        
        # Initialize bulk ticker input in session state
        if 'backtest_engine_bulk_tickers' not in st.session_state:
            st.session_state.backtest_engine_bulk_tickers = ""
        
        # Auto-populate bulk ticker input with current tickers
        current_tickers = [ticker for ticker in st.session_state.tickers if ticker]
        if current_tickers:
            current_ticker_string = ' '.join(current_tickers)
            if st.session_state.backtest_engine_bulk_tickers != current_ticker_string:
                st.session_state.backtest_engine_bulk_tickers = current_ticker_string
        
        # Text area for bulk ticker input
        bulk_tickers = st.text_area(
            "Tickers (e.g., SPY QQQ GLD TLT or SPY,QQQ,GLD,TLT)",
            value=st.session_state.backtest_engine_bulk_tickers,
            key="backtest_engine_bulk_ticker_input",
            height=100,
            help="Enter ticker symbols separated by spaces or commas. Click 'Fill Tickers' to replace tickers (keeps existing allocations)."
        )
        
        if st.button("Fill Tickers", key="backtest_engine_fill_tickers_btn"):
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
                    current_tickers = st.session_state.tickers.copy()
                    current_allocs = st.session_state.allocs.copy()
                    current_divs = st.session_state.divs.copy()
                    
                    # Replace tickers - new ones get 0% allocation
                    new_tickers = []
                    new_allocs = []
                    new_divs = []
                    
                    for i, ticker in enumerate(ticker_list):
                        if i < len(current_tickers):
                            # Use existing allocation if available
                            new_tickers.append(ticker)
                            new_allocs.append(current_allocs[i])
                            new_divs.append(current_divs[i])
                        else:
                            # New tickers get 0% allocation
                            new_tickers.append(ticker)
                            new_allocs.append(0.0)
                            new_divs.append(True)
                    
                    # Update session state
                    st.session_state.tickers = new_tickers
                    st.session_state.allocs = new_allocs
                    st.session_state.divs = new_divs
                    
                    # Clear any existing session state keys for individual ticker inputs to force refresh
                    for key in list(st.session_state.keys()):
                        if key.startswith("ticker_") or key.startswith("alloc_input_") or key.startswith("divs_checkbox_"):
                            del st.session_state[key]
                    
                    st.success(f"✅ Replaced tickers with: {', '.join(ticker_list)}")
                    st.info("💡 **Note:** Existing allocations preserved. Adjust allocations manually if needed.")
                    
                    # Force immediate rerun
                    st.rerun()
                else:
                    st.error("❌ No valid tickers found. Please enter ticker symbols separated by spaces or commas.")
            else:
                st.error("❌ Please enter ticker symbols.")

    # Clear all tickers button - exact same format as page 1
    if st.button("🗑️ Clear All Tickers", key="backtest_engine_clear_all_tickers_immediate", 
                help="Delete ALL tickers and create a blank one", use_container_width=True):
        # Clear all tickers and create a single blank ticker
        st.session_state.tickers = ['']
        st.session_state.allocs = [0.0]
        st.session_state.divs = [True]
        
        # Clear bulk ticker input
        st.session_state.backtest_engine_bulk_tickers = ""
        
        # Clear any existing session state keys for individual ticker inputs to force refresh
        for key in list(st.session_state.keys()):
            if key.startswith("ticker_") or key.startswith("alloc_input_") or key.startswith("divs_checkbox_"):
                del st.session_state[key]
        
        # Clear any backtest results
        for key in list(st.session_state.keys()):
            if key.startswith("backtest_") or key in ["results", "fig", "fig_stats", "fig_drawdown", "fig_allocations"]:
                del st.session_state[key]
        
        st.success("✅ All tickers cleared! Ready for fresh start.")
        st.rerun()

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
    
    # Use the radio button without specifying index - let Streamlit handle it via the key
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
                        "Benchmark Ticker (default: ^GSPC, starts 1927-12-30, used for beta calculation. Use SPYSIM for earlier dates, starts 1885-03-01)", value=default_benchmark
    ).replace(",", ".").upper()



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
            hist_yf = get_cached_ticker_data(benchmark_ticker, period="max")
            if hist_yf.empty:
                benchmark_error = f"Benchmark ticker '{benchmark_ticker}' not found on Yahoo Finance. Please choose another ticker."
            else:
                bench_start_date = hist_yf.index.min().date()
                # Get start dates for all asset tickers
                for t in st.session_state.tickers:
                    try:
                        hist_asset = get_cached_ticker_data(t, period="max")
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
            # Reset kill request when starting new backtest
            st.session_state.hard_kill_requested = False
            # --- PRE-CHECK BLOCK ---
            print("DEBUG: Run Backtest button pressed. Checking pre-conditions...")
            # Print start date of each selected ticker and benchmark for debugging
            for t in st.session_state.tickers:
                try:
                    hist = get_cached_ticker_data(t, period="max")
                    if not hist.empty:
                        print(f"DEBUG: Asset Ticker {t} starts on {hist.index.min().date()}")
                    else:
                        print(f"DEBUG: Asset Ticker {t}: No data found.")
                except Exception as e:
                    print(f"DEBUG: Asset Ticker {t}: Error - {e}")
            if benchmark_ticker:
                try:
                    hist = get_cached_ticker_data(benchmark_ticker, period="max")
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

# Stop buttons - always visible, same size as run/clear buttons
col_cancel, col_emergency = st.columns([1, 1])
with col_cancel:
    if st.button("🛑 Cancel Run", type="secondary", use_container_width=True, help="Stop current backtest execution gracefully"):
        st.session_state.hard_kill_requested = True
        # Reset running flags to bring back the Run button
        st.session_state.running = False
        st.session_state._run_requested = False
        if "_pending_backtest_params" in st.session_state:
            del st.session_state["_pending_backtest_params"]
        st.toast("🛑 **CANCELLING** - Stopping backtest execution...", icon="⏹️")
        st.rerun()

with col_emergency:
    if st.button("🚨 EMERGENCY KILL", type="secondary", use_container_width=True, help="Force stop backtest immediately - Use for unresponsive states"):
        st.toast("🚨 **EMERGENCY KILL** - Force stopping backtest...", icon="💥")
        emergency_kill()

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
        # Check for kill request before starting
        check_kill_request()
        
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
                xaxis=dict(rangeslider=dict(visible=False), type='date')
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
            
            # Add Risk-Free Rate Chart
            try:
                # Get risk-free rate data for the same period as the backtest
                # Use the portfolio data from the current backtest
                if hasattr(st.session_state, 'portfolio_value_without_additions') and st.session_state.portfolio_value_without_additions is not None:
                    portfolio_dates = st.session_state.portfolio_value_without_additions.index
                elif hasattr(st.session_state, 'portfolio_value_with_additions') and st.session_state.portfolio_value_with_additions is not None:
                    portfolio_dates = st.session_state.portfolio_value_with_additions.index
                else:
                    # Fallback to last 30 days if no portfolio data available
                    end_date = pd.Timestamp.now()
                    start_date = end_date - pd.Timedelta(days=30)
                    portfolio_dates = pd.date_range(start_date, end_date, freq='D')
                
                dates = portfolio_dates
                
                risk_free_rates = get_risk_free_rate(dates)
                
                if risk_free_rates is not None and not risk_free_rates.empty:
                    # Convert daily rates to annual rates for display
                    annual_rates = (1 + risk_free_rates) ** 365.25 - 1
                    annual_rates_pct = annual_rates * 100
                    
                    # Create the chart
                    fig_rf = go.Figure()
                    
                    # Add the risk-free rate line
                    fig_rf.add_trace(go.Scatter(
                        x=annual_rates_pct.index,
                        y=annual_rates_pct.values,
                        mode='lines',
                        name='Risk-Free Rate',
                        line=dict(color='#FF6B6B', width=2),
                        hovertemplate='<b>%{x}</b><br>Rate: %{y:.2f}%<extra></extra>'
                    ))
                    
                    # Add average line
                    avg_rate = annual_rates_pct.mean()
                    fig_rf.add_hline(
                        y=avg_rate,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text=f"Average: {avg_rate:.2f}%",
                        annotation_position="bottom right"
                    )
                    
                    # Update layout
                    fig_rf.update_layout(
                        title="Risk-Free Rate Over Time (3-Month Treasury)",
                        xaxis_title="Date",
                        yaxis_title="Annual Rate (%)",
                        height=400,
                        showlegend=True,
                        hovermode='x unified'
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig_rf, use_container_width=True, config=config)
                    
                    # Show current rate
                    current_rate = annual_rates_pct.iloc[-1]
                    st.info(f"📈 **Current Risk-Free Rate: {current_rate:.2f}%** (3-Month Treasury)")
                else:
                    st.info("📈 **Current Risk-Free Rate: ~4.0%** (Estimated)")
                    
            except Exception as e:
                st.info("📈 **Current Risk-Free Rate: ~4.0%** (Estimated)")
                pass
                
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
    
    # Calculate total money added
    try:
        cash_flows = st.session_state.get('cash_flows_with_additions', None)
        if cash_flows is not None and not cash_flows.empty:
            # Sum all negative values (cash invested) and make positive for display
            total_added = abs(cash_flows[cash_flows < 0].sum())
        else:
            total_added = np.nan
    except Exception:
        total_added = np.nan
    
    # Calculate total return (all money)
    total_return_all_money = ((final_with - total_added) / total_added) if total_added > 0 and not pd.isna(final_with) and not pd.isna(total_added) else np.nan
    
    # Use internal keys that map to friendly display names below
    stats["FinalValueWithAdditions"] = final_with
    stats["FinalValueWithoutAdditions"] = final_without
    stats["ReturnNoAdditions"] = return_no_additions
    stats["TotalReturnAllMoney"] = total_return_all_money
    stats["TotalMoneyAdded"] = total_added
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
        "ReturnNoAdditions": "Return (no additions)",
        "TotalReturnAllMoney": "Total Return (All Money)",
        "TotalMoneyAdded": "Total Money Added"
    }
    # Create DataFrame and reorder to put Final Values at the top
    stats_df = pd.DataFrame({
        "Metric": [display_names.get(k, k) for k in stats.keys()],
        "Value": list(stats.values())
    })
    
    # Custom ordering as requested:
    # 1. Final Values at the top
    # 2. Total Return (All Money) under Return (no additions)
    # 3. MWRR right after CAGR
    # 4. Total Money Added at the end
    final_value_metrics = ["Final Value (with additions)", "Final Value (no additions)", "Return (no additions)", "Total Return (All Money)"]
    performance_metrics = ["CAGR", "MWRR", "Volatility", "Sharpe", "Sortino", "UPI"]
    risk_metrics = ["Max Drawdown", "Ulcer Index", "Beta"]
    other_metrics = [metric for metric in stats_df["Metric"] if metric not in final_value_metrics + performance_metrics + risk_metrics + ["Total Money Added"]]
    
    # Reorder the DataFrame
    reordered_metrics = final_value_metrics + performance_metrics + risk_metrics + other_metrics + ["Total Money Added"]
    stats_df = stats_df.set_index("Metric").reindex(reordered_metrics).reset_index()
    def format_value(row):
        val = row["Value"]
        if pd.isna(val):
            return "n/a"
        # Percentage metrics
        if row["Metric"] in ["CAGR", "MWRR", "Max Drawdown", "Volatility", "Return (no additions)", "Total Return (All Money)"]:
            return f"{val:.2%}"
        # Simple float metrics
        elif row["Metric"] in ["Sharpe", "Sortino", "UPI", "Ulcer Index", "Beta"]:
            return f"{val:.2f}"
        # Currency metrics (final portfolio values)
        elif row["Metric"] in ["Final Value (with additions)", "Final Value (no additions)"]:
            try:
                return f"${val:,.2f}"
            except Exception:
                return "n/a"
        # Ensure all other values are converted to strings
        return str(val) if val is not None else "n/a"
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

