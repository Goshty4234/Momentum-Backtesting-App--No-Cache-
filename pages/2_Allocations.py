import streamlit as st
import datetime
from datetime import timedelta, time
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
import contextlib
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors as reportlab_colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import base64
warnings.filterwarnings('ignore')

# =============================================================================
# PERFORMANCE OPTIMIZATION: NO CACHING VERSION
# =============================================================================

def get_ticker_data(ticker_symbol, period="max", auto_adjust=False):
    """Get ticker data without caching (NO_CACHE version)
    
    Args:
        ticker_symbol: Stock ticker symbol
        period: Data period
        auto_adjust: Auto-adjust setting
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=period, auto_adjust=auto_adjust)[["Close", "Dividends"]]
        return hist
    except Exception:
        return pd.DataFrame()

def get_ticker_info(ticker_symbol):
    """Get ticker info without caching (NO_CACHE version)"""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        return info
    except Exception:
        return {}

def calculate_portfolio_metrics(portfolio_config, allocation_data):
    """Calculate portfolio metrics without caching (NO_CACHE version)"""
    # This will calculate the results without caching
    # Note: The actual calculation logic remains unchanged
    return portfolio_config, allocation_data  # Placeholder - will be filled in by calling functions

def optimize_data_loading():
    """Session state optimization to prevent redundant operations - PAGE-SPECIFIC"""
    # Use page-specific keys to prevent conflicts between pages
    page_prefix = "alloc_page_"
    
    # Initialize performance flags if not present
    if f'{page_prefix}data_loaded' not in st.session_state:
        st.session_state[f'{page_prefix}data_loaded'] = False
    if f'{page_prefix}last_refresh' not in st.session_state:
        st.session_state[f'{page_prefix}last_refresh'] = None
    
    # Check if data needs refresh (5 minutes)
    current_time = datetime.datetime.now()
    if (st.session_state[f'{page_prefix}last_refresh'] is None or 
        (current_time - st.session_state[f'{page_prefix}last_refresh']).seconds > 300):
        st.session_state[f'{page_prefix}data_loaded'] = False
        st.session_state[f'{page_prefix}last_refresh'] = current_time
    
    return st.session_state[f'{page_prefix}data_loaded']

def create_safe_cache_key(data):
    """Create a safe, consistent cache key from complex data structures"""
    import hashlib
    import json
    try:
        # Convert to JSON string and hash for consistent cache keys
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    except Exception:
        # Fallback to string representation
        return hashlib.md5(str(data).encode()).hexdigest()

def run_cached_backtest(portfolios_config_hash, start_date_str, end_date_str, benchmark_str, page_id="allocations"):
    """Run backtest without caching (NO_CACHE version)
    
    Args:
        portfolios_config_hash: Hash of portfolio configurations to detect changes
        start_date_str: Start date as string for consistent cache key
        end_date_str: End date as string for consistent cache key  
        benchmark_str: Benchmark ticker as string
        page_id: Page identifier to prevent cross-page conflicts
    """
    # This will be called by the actual backtest functions when needed
    # No caching is applied in this version
    return portfolios_config_hash, start_date_str, end_date_str, benchmark_str, page_id  # Placeholder

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
st.set_page_config(layout="wide", page_title="Portfolio Allocation Analysis", page_icon="📈")
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



# ...existing code...

# ==============================================================================
# PAGE-SCOPED SESSION STATE INITIALIZATION - ALLOCATIONS PAGE
# ==============================================================================
# Ensure complete independence from other pages by using page-specific session keys
if 'allocations_page_initialized' not in st.session_state:
    st.session_state.allocations_page_initialized = True

# Initialize page-specific session state with default configurations
if 'alloc_portfolio_configs' not in st.session_state:
    # Default configuration for allocations page
    st.session_state.alloc_portfolio_configs = [
        {
            'name': 'Allocation Portfolio',
            'stocks': [
                {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
            ],
            'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
                          'added_amount': 0,
              'added_frequency': 'none',
              'rebalancing_frequency': 'Monthly',
              'start_date_user': None,
              'end_date_user': None,
              'start_with': 'all',
              'use_momentum': True,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [
                {"lookback": 365, "exclude": 30, "weight": 0.5},
                {"lookback": 180, "exclude": 30, "weight": 0.3},
                {"lookback": 120, "exclude": 30, "weight": 0.2},
            ],
            'calc_beta': True,
            'calc_volatility': True,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
        }
    ]
if 'alloc_active_portfolio_index' not in st.session_state:
    st.session_state.alloc_active_portfolio_index = 0
if 'alloc_rerun_flag' not in st.session_state:
    st.session_state.alloc_rerun_flag = False

# Clean up any existing portfolio configs to remove unused settings
if 'alloc_portfolio_configs' in st.session_state:
    for config in st.session_state.alloc_portfolio_configs:
        config.pop('use_relative_momentum', None)
        config.pop('equal_if_all_negative', None)
if 'alloc_paste_json_text' not in st.session_state:
    st.session_state.alloc_paste_json_text = ""

# ==============================================================================
# END PAGE-SCOPED SESSION STATE INITIALIZATION
# ==============================================================================

# Use page-scoped active portfolio for the allocations page
active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index] if 'alloc_portfolio_configs' in st.session_state and 'alloc_active_portfolio_index' in st.session_state else None
if active_portfolio:
    # Removed duplicate Portfolio Name input field
    if st.session_state.get('alloc_rerun_flag', False):
        st.session_state.alloc_rerun_flag = False
        st.rerun()

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

st.set_page_config(layout="wide", page_title="Portfolio Allocation Analysis")

st.title("Portfolio Allocations")
st.markdown("Use the forms below to configure and run backtests to obtain allocation insights.")

# Portfolio Name
if 'alloc_portfolio_name' not in st.session_state:
    st.session_state.alloc_portfolio_name = "Allocation Portfolio"
alloc_portfolio_name = st.text_input("Portfolio Name", value=st.session_state.alloc_portfolio_name, key="alloc_portfolio_name_input")
st.session_state.alloc_portfolio_name = alloc_portfolio_name

# Sync portfolio name with active portfolio configuration
if 'alloc_active_portfolio_index' in st.session_state:
    active_idx = st.session_state.alloc_active_portfolio_index
    if 'alloc_portfolio_configs' in st.session_state:
        if active_idx < len(st.session_state.alloc_portfolio_configs):
            st.session_state.alloc_portfolio_configs[active_idx]['name'] = alloc_portfolio_name

# -----------------------
# Default JSON configs (for initialization)
# -----------------------
default_configs = [
    # 1) Benchmark only (SPY) - yearly rebalancing and yearly additions
    {
        'name': 'Benchmark Only (SPY)',
        'stocks': [
            {'ticker': 'SPY', 'allocation': 1.0, 'include_dividends': True},
        ],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 10000,
        'added_frequency': 'Annually',
        'rebalancing_frequency': 'Annually',
    'start_date_user': None,
    'end_date_user': None,
    'start_with': 'oldest',
        'use_momentum': False,
        'momentum_windows': [],
    'calc_beta': True,
    'calc_volatility': True,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
                    'vol_window_days': 365,
            'exclude_days_vol': 30,
            'use_minimal_threshold': False,
            'minimal_threshold_percent': 2.0,
        },
]

# -----------------------
# Helper functions
# -----------------------
def get_trading_days(start_date, end_date):
    return pd.bdate_range(start=start_date, end=end_date)

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
        base = pd.date_range(start=start, end=end, freq='MS')
    elif freq == "Quarterly":
        base = pd.date_range(start=start, end=end, freq='3MS')
    elif freq == "Semiannually":
        # First day of Jan and Jul each year
        semi = []
        for y in range(start.year, end.year + 1):
            for m in [1, 7]:
                semi.append(pd.Timestamp(year=y, month=m, day=1))
        base = pd.DatetimeIndex(semi)
    elif freq == "Annually":
        base = pd.date_range(start=start, end=end, freq='YS')
    elif freq == "Never" or freq == "none" or freq is None:
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

def calculate_cagr(values, dates):
    if len(values) < 2:
        return np.nan
    start_val = values[0]
    end_val = values[-1]
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0 or start_val <= 0:
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

# FIXED: Correct Sortino Ratio calculation
def calculate_sortino(returns, risk_free_rate=0):
    # Annualized Sortino ratio
    target_return = risk_free_rate / 252  # Daily target
    downside_returns = returns[returns < target_return]
    if len(downside_returns) < 2:
        return np.nan
    downside_std = np.std(downside_returns) * np.sqrt(365)
    if downside_std == 0:
        return np.nan
    expected_return = returns.mean() * 252
    return (expected_return - risk_free_rate) / downside_std

# FIXED: Correct Ulcer Index calculation
def calculate_ulcer_index(values):
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    peak[peak == 0] = 1 # Avoid division by zero
    drawdown_sq = ((values - peak) / peak)**2
    return np.sqrt(np.mean(drawdown_sq)) if len(drawdown_sq) > 0 else np.nan

# FIXED: Correct UPI calculation
def calculate_upi(cagr, ulcer_index, risk_free_rate=0):
    if pd.isna(cagr) or pd.isna(ulcer_index) or ulcer_index == 0:
        return np.nan
    return (cagr - risk_free_rate) / ulcer_index

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

# -----------------------
# PDF Generation Functions
# -----------------------
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
        if ax.get_legend_handles_labels()[0]:  # Only add legend if there are labels
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
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
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.text(0.5, 0.5, f'Error converting plot: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig

def create_matplotlib_table(data, headers=None):
    """Create a matplotlib table for PDF generation."""
    try:
        if data is None or len(data) == 0:
            return None
        
        # Convert data to proper format
        if isinstance(data, pd.DataFrame):
            table_data = data.values.tolist()
            if headers is None:
                headers = data.columns.tolist()
        else:
            table_data = data
            if headers is None:
                headers = [f'Col {i+1}' for i in range(len(data[0]))]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        return fig
    except Exception as e:
        print(f"Error creating matplotlib table: {e}")
        return None

def generate_allocations_pdf(custom_name=""):
    """Generate PDF report for allocations page."""
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("📄 Initializing PDF document...")
        
        # Get active portfolio configuration
        active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
        
        # Create PDF document
        buffer = io.BytesIO()
        
        # Add proper PDF metadata
        if custom_name.strip():
            title = f"Allocations Report - {custom_name.strip()}"
            subject = f"Portfolio Allocation Analysis: {custom_name.strip()}"
        else:
            title = "Allocations Report"
            subject = "Portfolio Allocation and Asset Distribution Analysis"
        
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=72, 
            leftMargin=72, 
            topMargin=72, 
            bottomMargin=72,
            title=title,
            author="Portfolio Backtest System",
            subject=subject,
            creator="Allocations Application"
        )
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER,
            leftIndent=0,
            rightIndent=0,
            firstLineIndent=0,
            wordWrap=False
        )
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15
        )
        
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
            main_title = f"Allocations Report - {custom_name.strip()}"
            subtitle = f"Portfolio Allocation Analysis: {custom_name.strip()}"
        else:
            main_title = "Portfolio Allocations Report"
            subtitle = "Comprehensive Investment Portfolio Analysis"
        
        story.append(Paragraph(main_title, title_style))
        story.append(Paragraph(subtitle, subtitle_style))
        
        # Document metadata is set in SimpleDocTemplate creation above
        
        # Report metadata
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated on: {current_time}", styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Table of contents
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
            "Target Allocation if Rebalanced Today",
            "Portfolio-Weighted Summary Statistics",
            "Portfolio Composition Analysis"
        ]
        
        for i, point in enumerate(toc_points, 1):
            story.append(Paragraph(f"{i}. {point}", styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Report overview
        overview_style = ParagraphStyle(
            'Overview',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=reportlab_colors.Color(0.3, 0.5, 0.7)
        )
        
        story.append(Paragraph("Report Overview", overview_style))
        story.append(Paragraph("This report provides comprehensive analysis of investment portfolios, including:", styles['Normal']))
        
        # Overview bullet points
        overview_points = [
            "Detailed portfolio configurations with all parameters and strategies",
            "Current allocations and rebalancing countdown timers"
        ]
        
        for point in overview_points:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        
        story.append(PageBreak())
        
        # Update progress
        progress_bar.progress(20)
        status_text.text("📊 Adding portfolio configurations...")
        
        # SECTION 1: Portfolio Configurations & Parameters
        story.append(Paragraph("1. Portfolio Configurations & Parameters", heading_style))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph(f"Portfolio: {active_portfolio.get('name', 'Unknown')}", subheading_style))
        story.append(Spacer(1, 10))
        
        # Create configuration table with all parameters
        config_data = [
            ['Parameter', 'Value', 'Description'],
            ['Initial Value', f"${active_portfolio.get('initial_value', 0):,.2f}", 'Starting portfolio value'],
            ['Added Amount', f"${active_portfolio.get('added_amount', 0):,.2f}", 'Regular contribution amount'],
            ['Added Frequency', active_portfolio.get('added_frequency', 'N/A'), 'How often contributions are made'],
            ['Rebalancing Frequency', active_portfolio.get('rebalancing_frequency', 'N/A'), 'How often portfolio is rebalanced'],
            ['Benchmark', active_portfolio.get('benchmark_ticker', 'N/A'), 'Performance comparison index'],
            ['Use Momentum', 'Yes' if active_portfolio.get('use_momentum', False) else 'No', 'Whether momentum strategy is enabled'],
            ['Momentum Strategy', active_portfolio.get('momentum_strategy', 'N/A'), 'Type of momentum calculation'],
            ['Negative Momentum Strategy', active_portfolio.get('negative_momentum_strategy', 'N/A'), 'How to handle negative momentum'],
            ['Calculate Beta', 'Yes' if active_portfolio.get('calc_beta', False) else 'No', 'Include beta in momentum weighting'],
            ['Calculate Volatility', 'Yes' if active_portfolio.get('calc_volatility', False) else 'No', 'Include volatility in momentum weighting'],
            ['Start Strategy', active_portfolio.get('start_with', 'N/A'), 'Initial allocation strategy'],
            ['Beta Lookback', f"{active_portfolio.get('beta_window_days', 0)} days", 'Days for beta calculation'],
            ['Beta Exclude', f"{active_portfolio.get('exclude_days_beta', 0)} days", 'Days excluded from beta calculation'],
            ['Volatility Lookback', f"{active_portfolio.get('vol_window_days', 0)} days", 'Days for volatility calculation'],
            ['Volatility Exclude', f"{active_portfolio.get('exclude_days_vol', 0)} days", 'Days excluded from volatility calculation'],
            ['Minimal Threshold', f"{active_portfolio.get('minimal_threshold_percent', 2.0):.1f}%" if active_portfolio.get('use_minimal_threshold', False) else 'Disabled', 'Minimum allocation percentage threshold']
        ]
        
        # Add momentum windows if they exist
        momentum_windows = active_portfolio.get('momentum_windows', [])
        if momentum_windows:
            for i, window in enumerate(momentum_windows, 1):
                lookback = window.get('lookback', 0)
                weight = window.get('weight', 0)
                config_data.append([
                    f'Momentum Window {i}',
                    f"{lookback} days, {weight:.2f}",
                    f"Lookback: {lookback} days, Weight: {weight:.2f}"
                ])
        
        # Add tickers with enhanced information
        tickers_data = [['Ticker', 'Allocation %', 'Include Dividends']]
        for ticker_config in active_portfolio.get('stocks', []):
            tickers_data.append([
                ticker_config['ticker'],
                f"{ticker_config['allocation']*100:.1f}%",
                "✓" if ticker_config['include_dividends'] else "✗"
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
        
        tickers_table = Table(tickers_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
        tickers_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
        ]))
        
        story.append(config_table)
        story.append(PageBreak())
        # Show ticker allocations table, but hide Allocation % column if momentum is enabled
        if not active_portfolio.get('use_momentum', True):
            story.append(Paragraph("Initial Ticker Allocations (Entered by User):", styles['Heading3']))
            story.append(Paragraph("Note: These are the initial allocations entered by the user, not rebalanced allocations.", styles['Normal']))
            story.append(tickers_table)
            story.append(Spacer(1, 15))
        else:
            story.append(Paragraph("Initial Ticker Allocations:", styles['Heading3']))
            story.append(Paragraph("Note: Momentum strategy is enabled - ticker allocations are calculated dynamically based on momentum scores.", styles['Normal']))
            
            # Create modified table without Allocation % column for momentum strategies
            tickers_data_momentum = [['Ticker', 'Include Dividends']]
            for ticker_config in active_portfolio.get('stocks', []):
                tickers_data_momentum.append([
                    ticker_config['ticker'],
                    "✓" if ticker_config['include_dividends'] else "✗"
                ])
            
            tickers_table_momentum = Table(tickers_data_momentum, colWidths=[2.25*inch, 2.25*inch])
            tickers_table_momentum.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
            ]))
            
            story.append(tickers_table_momentum)
            story.append(Spacer(1, 15))
        
        # Update progress
        progress_bar.progress(40)
        status_text.text("🎯 Adding allocation charts and timers...")
        
        # SECTION 2: Target Allocation if Rebalanced Today
        story.append(PageBreak())
        current_date_str = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"2. Target Allocation if Rebalanced Today ({current_date_str})", heading_style))
        story.append(Spacer(1, 10))
        
        # Get the allocation data from your existing UI - fetch the existing allocation data
        if 'alloc_snapshot_data' in st.session_state:
            snapshot = st.session_state.alloc_snapshot_data
            today_weights_map = snapshot.get('today_weights_map', {})
            
            # Process ALL portfolios, not just the active one
            portfolio_count = 0
            for portfolio_name, today_weights in today_weights_map.items():
                if today_weights:
                    # Add page break for all portfolios except the first one
                    if portfolio_count > 0:
                        story.append(PageBreak())
                    
                    portfolio_count += 1
                    
                    # Add portfolio header
                    story.append(Paragraph(f"Portfolio: {portfolio_name}", subheading_style))
                    story.append(Spacer(1, 10))
                    
                    # Create pie chart for this portfolio
                    try:
                        # Create labels and values for the plot
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
                            
                            # Add to PDF - reduce pie chart size to fit everything on one page
                            story.append(Image(target_img_buffer, width=5.5*inch, height=5.5*inch))
                            
                            # Add Next Rebalance Timer information - calculate from portfolio data
                            story.append(Paragraph(f"Next Rebalance Timer - {portfolio_name}", subheading_style))
                            story.append(Spacer(1, 1))
                            
                            # Calculate timer information from portfolio configuration and allocation data
                            try:
                                # Get portfolio configuration
                                portfolio_cfg = None
                                if 'alloc_snapshot_data' in st.session_state:
                                    snapshot = st.session_state['alloc_snapshot_data']
                                    portfolio_configs = snapshot.get('portfolio_configs', [])
                                    portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == portfolio_name), None)
                                
                                if portfolio_cfg:
                                    rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'Monthly')
                                    initial_value = portfolio_cfg.get('initial_value', 10000)
                                    
                                    # Get last rebalance date from allocation data
                                    all_allocations = snapshot.get('all_allocations', {})
                                    portfolio_allocations = all_allocations.get(portfolio_name, {})
                                    
                                    if portfolio_allocations:
                                        alloc_dates = sorted(list(portfolio_allocations.keys()))
                                        if len(alloc_dates) > 1:
                                            last_rebal_date = alloc_dates[-2]  # Second to last date
                                        else:
                                            last_rebal_date = alloc_dates[-1] if alloc_dates else None
                                        
                                        if last_rebal_date:
                                            # Map frequency to function expectations
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
                                            mapped_frequency = frequency_mapping.get(rebalancing_frequency.lower(), rebalancing_frequency.lower())
                                            
                                            # Calculate next rebalance date
                                            next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                                                mapped_frequency, last_rebal_date
                                            )
                                            
                                            if next_date and time_until:
                                                story.append(Paragraph(f"Time Until Next Rebalance: {format_time_until(time_until)}", styles['Normal']))
                                                story.append(Paragraph(f"Target Rebalance Date: {next_date.strftime('%B %d, %Y')}", styles['Normal']))
                                                story.append(Paragraph(f"Rebalancing Frequency: {rebalancing_frequency}", styles['Normal']))
                                                story.append(Paragraph(f"Portfolio Value: ${initial_value:,.2f}", styles['Normal']))
                                            else:
                                                story.append(Paragraph("Next rebalance date calculation not available", styles['Normal']))
                                        else:
                                            story.append(Paragraph("No rebalancing history available", styles['Normal']))
                                    else:
                                        story.append(Paragraph("No allocation data available for timer calculation", styles['Normal']))
                                else:
                                    story.append(Paragraph("Portfolio configuration not found for timer calculation", styles['Normal']))
                            except Exception as e:
                                story.append(Paragraph(f"Error calculating timer information: {str(e)}", styles['Normal']))
                            
                            # Add page break after pie plot + timer to separate from allocation table
                            story.append(PageBreak())
                            
                            # Now add the allocation table on the next page
                            story.append(Paragraph(f"Allocation Details for {portfolio_name}", subheading_style))
                            story.append(Spacer(1, 10))
                            
                            # Create comprehensive allocation table with all columns
                            try:
                                if today_weights:
                                    # Get portfolio value and raw data for price calculations
                                    portfolio_value = 10000  # Default value
                                    raw_data = {}
                                    
                                    if 'alloc_snapshot_data' in st.session_state:
                                        snapshot = st.session_state['alloc_snapshot_data']
                                        portfolio_configs = snapshot.get('portfolio_configs', [])
                                        portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == portfolio_name), None)
                                        if portfolio_cfg:
                                            portfolio_value = portfolio_cfg.get('initial_value', 10000)
                                        raw_data = snapshot.get('raw_data', {})
                                    
                                    # Create comprehensive table with all columns
                                    headers = ['Asset', 'Allocation %', 'Price ($)', 'Shares', 'Total Value ($)', '% of Portfolio']
                                    table_rows = []
                                    
                                    for asset, weight in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])):
                                        if float(weight) > 0:
                                            alloc_pct = float(weight) * 100
                                            allocation_value = portfolio_value * float(weight)
                                            
                                            # Get current price
                                            current_price = None
                                            shares = 0.0
                                            if asset != 'CASH' and asset in raw_data:
                                                try:
                                                    df = raw_data[asset]
                                                    if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                        current_price = float(df['Close'].iloc[-1])
                                                        if current_price and current_price > 0:
                                                            shares = round(allocation_value / current_price, 1)
                                                except Exception:
                                                    pass
                                            
                                            # Calculate total value
                                            if current_price and current_price > 0:
                                                total_val = shares * current_price
                                            else:
                                                total_val = allocation_value
                                            
                                            # Calculate percentage of portfolio
                                            pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                                            
                                            # Format values for table
                                            price_str = f"${current_price:,.2f}" if current_price else "N/A"
                                            shares_str = f"{shares:,.1f}" if shares > 0 else "0.0"
                                            total_val_str = f"${total_val:,.2f}"
                                            
                                            table_rows.append([
                                                asset,
                                                f"{alloc_pct:.2f}%",
                                                price_str,
                                                shares_str,
                                                total_val_str,
                                                f"{pct_of_port:.2f}%"
                                            ])
                                    
                                    if table_rows:
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

                                        # Create table with proper column widths
                                        page_width = 7.5*inch
                                        col_widths = [1.2*inch, 1.0*inch, 1.2*inch, 1.0*inch, 1.5*inch, 1.0*inch]
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
                                    else:
                                        story.append(Paragraph("No allocation data available", styles['Normal']))
                                else:
                                    story.append(Paragraph("No allocation data available", styles['Normal']))
                            except Exception as e:
                                story.append(Paragraph(f"Error creating allocation table: {str(e)}", styles['Normal']))
                            
                            story.append(Spacer(1, 5))
                        else:
                            story.append(Paragraph(f"No allocation data available for {portfolio_name}", styles['Normal']))
                    except Exception as e:
                        story.append(Paragraph(f"Error creating pie chart for {portfolio_name}: {str(e)}", styles['Normal']))
        else:
            story.append(Paragraph("Allocation data not available. Please run the allocation analysis first.", styles['Normal']))
            story.append(Spacer(1, 5))
        
        # Update progress
        progress_bar.progress(70)
        status_text.text("📊 Adding portfolio-weighted summary statistics...")
        
        # SECTION 3: Portfolio-Weighted Summary Statistics
        story.append(PageBreak())
        story.append(Paragraph("3. Portfolio-Weighted Summary Statistics", heading_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Metrics weighted by portfolio allocation - represents the total portfolio characteristics", styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Add data accuracy warning
        warning_style = ParagraphStyle(
            'WarningStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=reportlab_colors.Color(0.8, 0.4, 0.2),
            leftIndent=20,
            rightIndent=20,
            spaceAfter=15
        )
        story.append(Paragraph("⚠️ <b>Data Accuracy Notice:</b> Portfolio metrics (PE, Beta, etc.) are calculated from available data and may not accurately represent the portfolio if some ticker data is missing, outdated, or incorrect. These metrics should be used as indicative values for portfolio analysis.", warning_style))
        story.append(Spacer(1, 15))
        
        # Get portfolio-weighted metrics using the same approach as the main UI
        active_name = active_portfolio.get('name', 'Unknown')
        
        # Create summary statistics table
        summary_data = []
        
        # Get portfolio metrics from session state (these are calculated and stored in the main UI)
        portfolio_pe = getattr(st.session_state, 'portfolio_pe', None)
        portfolio_forward_pe = getattr(st.session_state, 'portfolio_forward_pe', None)
        portfolio_pb = getattr(st.session_state, 'portfolio_pb', None)
        portfolio_peg = getattr(st.session_state, 'portfolio_peg', None)
        portfolio_ps = getattr(st.session_state, 'portfolio_ps', None)
        portfolio_ev_ebitda = getattr(st.session_state, 'portfolio_ev_ebitda', None)
        portfolio_beta = getattr(st.session_state, 'portfolio_beta', None)
        portfolio_roe = getattr(st.session_state, 'portfolio_roe', None)
        portfolio_roa = getattr(st.session_state, 'portfolio_roa', None)
        portfolio_profit_margin = getattr(st.session_state, 'portfolio_profit_margin', None)
        portfolio_operating_margin = getattr(st.session_state, 'portfolio_operating_margin', None)
        portfolio_gross_margin = getattr(st.session_state, 'portfolio_gross_margin', None)
        portfolio_revenue_growth = getattr(st.session_state, 'portfolio_revenue_growth', None)
        portfolio_earnings_growth = getattr(st.session_state, 'portfolio_earnings_growth', None)
        portfolio_eps_growth = getattr(st.session_state, 'portfolio_eps_growth', None)
        portfolio_dividend_yield = getattr(st.session_state, 'portfolio_dividend_yield', None)
        portfolio_payout_ratio = getattr(st.session_state, 'portfolio_payout_ratio', None)
        portfolio_market_cap = getattr(st.session_state, 'portfolio_market_cap', None)
        portfolio_enterprise_value = getattr(st.session_state, 'portfolio_enterprise_value', None)
        
        # Valuation metrics
        if portfolio_pe is not None and not pd.isna(portfolio_pe):
            summary_data.append(["Valuation", "P/E Ratio", f"{portfolio_pe:.2f}", "Price-to-Earnings ratio weighted by portfolio allocation"])
        if portfolio_forward_pe is not None and not pd.isna(portfolio_forward_pe):
            summary_data.append(["Valuation", "Forward P/E", f"{portfolio_forward_pe:.2f}", "Forward Price-to-Earnings ratio weighted by portfolio allocation"])
        if portfolio_pb is not None and not pd.isna(portfolio_pb):
            summary_data.append(["Valuation", "Price/Book", f"{portfolio_pb:.2f}", "Price-to-Book ratio weighted by portfolio allocation"])
        if portfolio_peg is not None and not pd.isna(portfolio_peg):
            summary_data.append(["Valuation", "PEG Ratio", f"{portfolio_peg:.2f}", "P/E to Growth ratio weighted by portfolio allocation"])
        if portfolio_ps is not None and not pd.isna(portfolio_ps):
            summary_data.append(["Valuation", "Price/Sales", f"{portfolio_ps:.2f}", "Price-to-Sales ratio weighted by portfolio allocation"])
        if portfolio_ev_ebitda is not None and not pd.isna(portfolio_ev_ebitda):
            summary_data.append(["Valuation", "EV/EBITDA", f"{portfolio_ev_ebitda:.2f}", "EV/EBITDA ratio weighted by portfolio allocation"])
        
        # Risk metrics
        if portfolio_beta is not None and not pd.isna(portfolio_beta):
            summary_data.append(["Risk", "Beta", f"{portfolio_beta:.2f}", "Portfolio volatility relative to market (1.0 = market average)"])
        
        # Profitability metrics
        if portfolio_roe is not None and not pd.isna(portfolio_roe):
            summary_data.append(["Profitability", "ROE (%)", f"{portfolio_roe:.2f}%", "Return on Equity weighted by portfolio allocation"])
        if portfolio_roa is not None and not pd.isna(portfolio_roa):
            summary_data.append(["Profitability", "ROA (%)", f"{portfolio_roa:.2f}%", "Return on Assets weighted by portfolio allocation"])
        if portfolio_profit_margin is not None and not pd.isna(portfolio_profit_margin):
            summary_data.append(["Profitability", "Profit Margin (%)", f"{portfolio_profit_margin:.2f}%", "Net profit margin weighted by portfolio allocation"])
        if portfolio_operating_margin is not None and not pd.isna(portfolio_operating_margin):
            summary_data.append(["Profitability", "Operating Margin (%)", f"{portfolio_operating_margin:.2f}%", "Operating profit margin weighted by portfolio allocation"])
        if portfolio_gross_margin is not None and not pd.isna(portfolio_gross_margin):
            summary_data.append(["Profitability", "Gross Margin (%)", f"{portfolio_gross_margin:.2f}%", "Gross profit margin weighted by portfolio allocation"])
        
        # Growth metrics
        if portfolio_revenue_growth is not None and not pd.isna(portfolio_revenue_growth):
            summary_data.append(["Growth", "Revenue Growth (%)", f"{portfolio_revenue_growth:.2f}%", "Revenue growth rate weighted by portfolio allocation"])
        if portfolio_earnings_growth is not None and not pd.isna(portfolio_earnings_growth):
            summary_data.append(["Growth", "Earnings Growth (%)", f"{portfolio_earnings_growth:.2f}%", "Earnings growth rate weighted by portfolio allocation"])
        if portfolio_eps_growth is not None and not pd.isna(portfolio_eps_growth):
            summary_data.append(["Growth", "EPS Growth (%)", f"{portfolio_eps_growth:.2f}%", "Earnings per share growth rate weighted by portfolio allocation"])
        
        # Dividend metrics
        if portfolio_dividend_yield is not None and not pd.isna(portfolio_dividend_yield):
            summary_data.append(["Dividends", "Dividend Yield (%)", f"{portfolio_dividend_yield:.2f}%", "Dividend yield weighted by portfolio allocation"])
        if portfolio_payout_ratio is not None and not pd.isna(portfolio_payout_ratio):
            summary_data.append(["Dividends", "Payout Ratio (%)", f"{portfolio_payout_ratio:.2f}%", "Dividend payout ratio weighted by portfolio allocation"])
        
        # Size metrics
        if portfolio_market_cap is not None and not pd.isna(portfolio_market_cap):
            summary_data.append(["Size", "Market Cap ($B)", f"${portfolio_market_cap:.2f}B", "Market capitalization weighted by portfolio allocation"])
        if portfolio_enterprise_value is not None and not pd.isna(portfolio_enterprise_value):
            summary_data.append(["Size", "Enterprise Value ($B)", f"${portfolio_enterprise_value:.2f}B", "Enterprise value weighted by portfolio allocation"])
        
        if summary_data:
            # Create summary table
            summary_headers = ['Category', 'Metric', 'Value', 'Description']
            summary_table = Table([summary_headers] + summary_data, colWidths=[0.8*inch, 1.2*inch, 0.8*inch, 3.2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('WORDWRAP', (0, 0), (-1, -1), True),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 15))
            
            # Add interpretation
            story.append(Paragraph("Portfolio Interpretation:", subheading_style))
            story.append(Spacer(1, 10))
            
            if portfolio_beta is not None and not pd.isna(portfolio_beta):
                if portfolio_beta < 0.8:
                    story.append(Paragraph(f"• Low Risk Portfolio - Beta {portfolio_beta:.2f} indicates lower volatility than market", styles['Normal']))
                elif portfolio_beta < 1.2:
                    story.append(Paragraph(f"• Moderate Risk Portfolio - Beta {portfolio_beta:.2f} indicates market-average volatility", styles['Normal']))
                else:
                    story.append(Paragraph(f"• High Risk Portfolio - Beta {portfolio_beta:.2f} indicates higher volatility than market", styles['Normal']))
            
            if portfolio_pe is not None and not pd.isna(portfolio_pe):
                if portfolio_pe < 15:
                    story.append(Paragraph(f"• Undervalued Portfolio - P/E {portfolio_pe:.2f} suggests attractive valuations", styles['Normal']))
                elif portfolio_pe < 25:
                    story.append(Paragraph(f"• Fairly Valued Portfolio - P/E {portfolio_pe:.2f} suggests reasonable valuations", styles['Normal']))
                else:
                    story.append(Paragraph(f"• Potentially Overvalued Portfolio - P/E {portfolio_pe:.2f} suggests high valuations", styles['Normal']))
        else:
            story.append(Paragraph("No portfolio-weighted metrics available for display.", styles['Normal']))
        
        # Update progress
        progress_bar.progress(80)
        status_text.text("🏢 Adding portfolio composition analysis...")
        
        # SECTION 4: Portfolio Composition Analysis
        story.append(PageBreak())
        story.append(Paragraph("4. Portfolio Composition Analysis", heading_style))
        story.append(Spacer(1, 15))
        
        # Get sector and industry data from session state (these are calculated and stored in the main UI)
        sector_data = getattr(st.session_state, 'sector_data', pd.Series(dtype=float))
        industry_data = getattr(st.session_state, 'industry_data', pd.Series(dtype=float))
        
        # Sector Allocation
        if not sector_data.empty:
            story.append(Paragraph("Sector Allocation", subheading_style))
            story.append(Spacer(1, 10))
            
            # Create sector table
            sector_table_data = [['Sector', 'Allocation (%)']]
            for sector, allocation in sector_data.items():
                sector_table_data.append([sector, f"{allocation:.2f}%"])
            
            sector_table = Table(sector_table_data, colWidths=[3.0*inch, 1.5*inch])
            sector_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
            ]))
            story.append(sector_table)
            story.append(Spacer(1, 15))
        
        # Industry Allocation
        if not industry_data.empty:
            story.append(Paragraph("Industry Allocation", subheading_style))
            story.append(Spacer(1, 10))
            
            # Create industry table
            industry_table_data = [['Industry', 'Allocation (%)']]
            for industry, allocation in industry_data.items():
                industry_table_data.append([industry, f"{allocation:.2f}%"])
            
            industry_table = Table(industry_table_data, colWidths=[3.0*inch, 1.5*inch])
            industry_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
            ]))
            story.append(industry_table)
            story.append(Spacer(1, 15))
        
        # Add page break and create charts page
        if not sector_data.empty or not industry_data.empty:
            story.append(PageBreak())
            story.append(Paragraph("Portfolio Distribution Charts", heading_style))
            story.append(Spacer(1, 15))
            
            # Check if we actually have any allocation data to display
            has_sector_data = not sector_data.empty and len(sector_data) > 0 and sector_data.sum() > 0
            has_industry_data = not industry_data.empty and len(industry_data) > 0 and industry_data.sum() > 0
            
            if not has_sector_data and not has_industry_data:
                # Portfolio is all cash - show message instead of charts
                story.append(Paragraph("This portfolio is currently allocated 100% to cash. No sector or industry distribution charts are available.", styles['Normal']))
                story.append(Spacer(1, 15))
            else:
                # Create combined figure with both pie charts one above the other
                try:
                    # Create figure with two subplots - square subplots to ensure circular pie charts
                    fig, (ax_sector, ax_industry) = plt.subplots(2, 1, figsize=(12, 12))
                
                    # Sector pie chart
                    if has_sector_data:
                        sectors = sector_data.index.tolist()
                        allocations = sector_data.values.tolist()
                        
                        # Create pie chart with percentage labels but only for larger slices to avoid overlap
                        def make_autopct(values):
                            def my_autopct(pct):
                                if pd.isna(pct) or pct is None:
                                    return ''
                                total = sum(values)
                                if pd.isna(total) or total == 0:
                                    return ''
                                val = int(round(pct*total/100.0))
                                # Only show percentage if slice is large enough (>5%)
                                return f'{pct:.1f}%' if pct > 5 else ''
                            return my_autopct
                        
                        wedges_sector, texts_sector, autotexts_sector = ax_sector.pie(allocations, autopct=make_autopct(allocations), 
                                                                                     startangle=90, textprops={'fontsize': 10})
                        
                        # Create legend with percentages - positioned further to the right
                        legend_labels = [f"{sector} ({alloc:.1f}%)" for sector, alloc in zip(sectors, allocations)]
                        ax_sector.legend(wedges_sector, legend_labels, title="Sectors", loc="center left", bbox_to_anchor=(1.05, 0, 0.5, 1), fontsize=10)
                        
                        # Wrap long titles to prevent them from going out of bounds
                        title_text = f'Sector Allocation - {portfolio_name}'
                        # Use textwrap for proper word-based wrapping
                        import textwrap
                        wrapped_title = textwrap.fill(title_text, width=40, break_long_words=True, break_on_hyphens=False)
                        ax_sector.set_title(wrapped_title, fontsize=14, fontweight='bold')
                        # Force perfectly circular shape
                        ax_sector.set_aspect('equal')
                        ax_sector.set_xlim(-1.2, 1.2)
                        ax_sector.set_ylim(-1.2, 1.2)
                    else:
                        # No sector data - show placeholder
                        ax_sector.text(0.5, 0.5, 'No sector data available', 
                                     horizontalalignment='center', verticalalignment='center', 
                                     transform=ax_sector.transAxes, fontsize=12)
                        # Wrap long titles to prevent them from going out of bounds
                        title_text = f'Sector Allocation - {portfolio_name}'
                        # Use textwrap for proper word-based wrapping
                        import textwrap
                        wrapped_title = textwrap.fill(title_text, width=40, break_long_words=True, break_on_hyphens=False)
                        ax_sector.set_title(wrapped_title, fontsize=14, fontweight='bold')
                    
                    # Industry pie chart
                    if has_industry_data:
                        industries = industry_data.index.tolist()
                        allocations = industry_data.values.tolist()
                        
                        # Create pie chart with percentage labels but only for larger slices to avoid overlap
                        def make_autopct(values):
                            def my_autopct(pct):
                                if pd.isna(pct) or pct is None:
                                    return ''
                                total = sum(values)
                                if pd.isna(total) or total == 0:
                                    return ''
                                val = int(round(pct*total/100.0))
                                # Only show percentage if slice is large enough (>5%)
                                return f'{pct:.1f}%' if pct > 5 else ''
                            return my_autopct
                        
                        wedges_industry, texts_industry, autotexts_industry = ax_industry.pie(allocations, autopct=make_autopct(allocations), 
                                                                                             startangle=90, textprops={'fontsize': 10})
                        
                        # Create legend with percentages - positioned further to the right
                        legend_labels = [f"{industry} ({alloc:.1f}%)" for industry, alloc in zip(industries, allocations)]
                        ax_industry.legend(wedges_industry, legend_labels, title="Industries", loc="center left", bbox_to_anchor=(1.05, 0, 0.5, 1), fontsize=10)
                        
                        # Wrap long titles to prevent them from going out of bounds
                        title_text = f'Industry Allocation - {portfolio_name}'
                        # Use textwrap for proper word-based wrapping
                        import textwrap
                        wrapped_title = textwrap.fill(title_text, width=40, break_long_words=True, break_on_hyphens=False)
                        ax_industry.set_title(wrapped_title, fontsize=14, fontweight='bold')
                        # Force perfectly circular shape
                        ax_industry.set_aspect('equal')
                        ax_industry.set_xlim(-1.2, 1.2)
                        ax_industry.set_ylim(-1.2, 1.2)
                    else:
                        # No industry data - show placeholder
                        ax_industry.text(0.5, 0.5, 'No industry data available', 
                                       horizontalalignment='center', verticalalignment='center', 
                                       transform=ax_industry.transAxes, fontsize=12)
                        # Wrap long titles to prevent them from going out of bounds
                        title_text = f'Industry Allocation - {portfolio_name}'
                        # Use textwrap for proper word-based wrapping
                        import textwrap
                        wrapped_title = textwrap.fill(title_text, width=40, break_long_words=True, break_on_hyphens=False)
                        ax_industry.set_title(wrapped_title, fontsize=14, fontweight='bold')
                    
                    # Adjust layout to maintain circular shapes and accommodate legends
                    plt.subplots_adjust(hspace=0.4, left=0.1, right=0.7, top=0.95, bottom=0.05)
                    
                    # Save to buffer - don't use bbox_inches='tight' to preserve aspect ratio
                    combined_img_buffer = io.BytesIO()
                    fig.savefig(combined_img_buffer, format='png', dpi=300, facecolor='white')
                    combined_img_buffer.seek(0)
                    plt.close(fig)
                    
                    # Add to PDF - maintain square aspect ratio for circular charts
                    story.append(Image(combined_img_buffer, width=8*inch, height=8*inch))
                    story.append(Spacer(1, 15))
                    
                except Exception as e:
                    story.append(Paragraph(f"Error creating charts: {str(e)}", styles['Normal']))
        
        # Add page break and move Portfolio Risk Metrics Summary to next page
        story.append(PageBreak())
        story.append(Paragraph("Portfolio Risk Metrics Summary", subheading_style))
        story.append(Spacer(1, 10))
        
        risk_metrics_data = []
        
        # Beta
        if portfolio_beta is not None and not pd.isna(portfolio_beta):
            if portfolio_beta < 0.8:
                beta_risk = "Low Risk"
            elif portfolio_beta < 1.2:
                beta_risk = "Balanced Risk"
            elif portfolio_beta < 1.5:
                beta_risk = "Moderate Risk"
            else:
                beta_risk = "High Risk"
            risk_metrics_data.append(["Portfolio Risk Level", beta_risk, f"Beta: {portfolio_beta:.2f}"])
        else:
            risk_metrics_data.append(["Portfolio Risk Level", "NA", "Beta: NA"])
        
        # P/E
        if portfolio_pe is not None and not pd.isna(portfolio_pe):
            if portfolio_pe < 15:
                pe_rating = "Undervalued"
            elif portfolio_pe < 25:
                pe_rating = "Fair Value"
            elif portfolio_pe < 35:
                pe_rating = "Expensive"
            else:
                pe_rating = "Overvalued"
            risk_metrics_data.append(["Current P/E Rating", pe_rating, f"P/E: {portfolio_pe:.2f}"])
        else:
            risk_metrics_data.append(["Current P/E Rating", "NA", "P/E: NA"])
        
        # Forward P/E
        if portfolio_forward_pe is not None and not pd.isna(portfolio_forward_pe):
            if portfolio_forward_pe < 15:
                fpe_rating = "Undervalued"
            elif portfolio_forward_pe < 25:
                fpe_rating = "Fair Value"
            elif portfolio_forward_pe < 35:
                fpe_rating = "Expensive"
            else:
                fpe_rating = "Overvalued"
            risk_metrics_data.append(["Forward P/E Rating", fpe_rating, f"Forward P/E: {portfolio_forward_pe:.2f}"])
        else:
            risk_metrics_data.append(["Forward P/E Rating", "NA", "Forward P/E: NA"])
        
        # Dividend
        if portfolio_dividend_yield is not None and not pd.isna(portfolio_dividend_yield):
            if portfolio_dividend_yield > 5:
                div_rating = "Very High Yield"
            elif portfolio_dividend_yield > 3:
                div_rating = "Good Yield"
            elif portfolio_dividend_yield > 1.5:
                div_rating = "Moderate Yield"
            else:
                div_rating = "Low Yield"
            risk_metrics_data.append(["Dividend Rating", div_rating, f"Yield: {portfolio_dividend_yield:.2f}%"])
        else:
            risk_metrics_data.append(["Dividend Rating", "NA", "Yield: NA"])
        
        if risk_metrics_data:
            risk_headers = ['Metric', 'Rating', 'Value']
            risk_table = Table([risk_headers] + risk_metrics_data, colWidths=[2.0*inch, 1.5*inch, 1.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
            ]))
            story.append(risk_table)
        
        # Add detailed financial indicators table
        try:
            # Get comprehensive dataframe from session state
            df_comprehensive = getattr(st.session_state, 'df_comprehensive', None)
            
            if df_comprehensive is not None and not df_comprehensive.empty:
                story.append(PageBreak())
                story.append(Paragraph("Detailed Financial Indicators for Each Position", heading_style))
                story.append(Spacer(1, 10))
                story.append(Paragraph("This section provides comprehensive financial metrics for each position in your portfolio, organized into focused categories for better analysis.", styles['Normal']))
                story.append(Spacer(1, 10))
                
                # Helper function to create wrapped text for PDF with enhanced company name, sector, and industry handling
                def wrap_text_for_pdf(text, max_length=25):
                    """Wrap long text to fit in PDF cells with intelligent line breaks for company names, sectors, and industries"""
                    if pd.isna(text) or text is None:
                        return 'N/A'
                    text_str = str(text)
                    if len(text_str) <= max_length:
                        return text_str
                    
                    # For company names, sector, and industry, prioritize showing the full name with line breaks
                    if 'Name' in str(text) or 'Sector' in str(text) or 'Industry' in str(text) or len(text_str) > max_length * 1.2:
                        # Try to break at spaces first for better readability
                        words = text_str.split()
                        if len(words) > 1:
                            # Find the best break point that maximizes text visibility
                            best_break = 0
                            for i, word in enumerate(words):
                                if len(' '.join(words[:i+1])) <= max_length:
                                    best_break = i + 1
                                else:
                                    break
                            
                            if best_break > 0:
                                # Use line break for better PDF formatting
                                first_line = ' '.join(words[:best_break])
                                second_line = ' '.join(words[best_break:])
                                
                                # For company names, sector, and industry, allow longer second lines to show more text
                                if len(second_line) > max_length:
                                    # Instead of truncating, try to break again
                                    second_words = second_line.split()
                                    if len(second_words) > 1:
                                        # Find another break point in the second line
                                        second_break = 0
                                        for j, word in enumerate(second_words):
                                            if len(' '.join(second_words[:j+1])) <= max_length:
                                                second_break = j + 1
                                            else:
                                                break
                                        if second_break > 0:
                                            second_line = ' '.join(second_words[:second_break])
                                            third_line = ' '.join(second_words[second_break:])
                                            if len(third_line) > max_length:
                                                third_line = third_line[:max_length-3] + '...'
                                            return first_line + '\n' + second_line + '\n' + third_line
                                        else:
                                            # If we can't break further, truncate the second line
                                            second_line = second_line[:max_length-3] + '...'
                                    else:
                                        # Single long word in second line, truncate
                                        second_line = second_line[:max_length-3] + '...'
                                
                                return first_line + '\n' + second_line
                            else:
                                # If we can't break at words, truncate with ellipsis
                                return text_str[:max_length-3] + '...'
                    
                    # For other columns, try to break at spaces or special characters
                    words = text_str.split()
                    if len(words) == 1:
                        # Single long word, break at max_length
                        return text_str[:max_length-3] + '...'
                    # Try to fit as many words as possible
                    result = ''
                    for word in words:
                        if len(result + ' ' + word) <= max_length:
                            result += (' ' + word) if result else word
                        else:
                            break
                    if not result:
                        result = text_str[:max_length-3] + '...'
                    return result
                
                # Helper function to create focused tables with specific columns and styling
                def create_focused_table(title, columns, data_subset, col_widths):
                    """Create a focused table with specific columns and enhanced styling"""
                    story.append(Paragraph(title, subheading_style))
                    story.append(Spacer(1, 5))
                    
                    # Filter data to only include the specified columns
                    if all(col in data_subset.columns for col in columns):
                        table_data = [columns]  # Header row
                        
                        for _, row in data_subset.iterrows():
                            pdf_row = []
                            for col in columns:
                                value = row[col]
                                # Apply text wrapping based on column type
                                if 'Name' in col:
                                    pdf_row.append(wrap_text_for_pdf(value, 22))  # Increased for better company name display
                                elif 'Sector' in col or 'Industry' in col:
                                    pdf_row.append(wrap_text_for_pdf(value, 25))  # Increased for better sector/industry display
                                elif 'Ticker' in col:
                                    pdf_row.append(wrap_text_for_pdf(value, 8))
                                else:
                                    pdf_row.append(wrap_text_for_pdf(value, 20))
                            
                            table_data.append(pdf_row)
                        
                        # Create table with custom column widths
                        table = Table(table_data, colWidths=col_widths)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, -1), 6),  # Smaller font to fit more data
                            ('GRID', (0, 0), (-1, -1), 0.5, reportlab_colors.grey),  # Thinner grid lines
                            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [reportlab_colors.Color(0.98, 0.98, 0.98), reportlab_colors.Color(0.95, 0.95, 0.95)]),
                            ('WORDWRAP', (0, 0), (-1, -1), True),
                            ('LEFTPADDING', (0, 0), (-1, -1), 3),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                            ('MINIMUMHEIGHT', (0, 0), (-1, -1), 15)
                        ]))
                        story.append(table)
                        story.append(Spacer(1, 10))
                    else:
                        story.append(Paragraph(f"Note: {title} data not available for this portfolio.", styles['Normal']))
                
                # Section 1: Overview & Basic Info - Company Name column optimized to 1.5" for balance
                # Industry column optimized to 1.1" to prevent truncation while ensuring table fits within page
                # Enhanced text wrapping for Company Name, Sector, and Industry columns to ensure full text visibility
                # Total table width: 7.1 inches (ensures small margin on each side of 8.5" page)
                overview_cols = ['Ticker', 'Company Name', 'Sector', 'Industry', 'Current Price ($)', 'Allocation %', 'Shares', 'Total Value ($)', '% of Portfolio']
                overview_widths = [0.6*inch, 1.5*inch, 1.0*inch, 1.1*inch, 0.8*inch, 0.7*inch, 0.6*inch, 1.0*inch, 0.8*inch]
                create_focused_table("📊 Overview & Basic Information", overview_cols, df_comprehensive, overview_widths)
                
                # Section 2: Valuation Metrics
                valuation_cols = ['Ticker', 'Market Cap ($B)', 'Enterprise Value ($B)', 'P/E Ratio', 'Forward P/E', 'PEG Ratio', 'Price/Book', 'Price/Sales', 'EV/EBITDA']
                valuation_widths = [0.6*inch, 0.8*inch, 1.0*inch, 0.6*inch, 0.7*inch, 0.6*inch, 0.7*inch, 0.7*inch, 0.7*inch]
                create_focused_table("💰 Valuation Metrics", valuation_cols, df_comprehensive, valuation_widths)
                
                # Section 3: Financial Health - Increased column widths to prevent title overlap
                health_cols = ['Ticker', 'Debt/Equity', 'Current Ratio', 'Quick Ratio', 'ROE (%)', 'ROA (%)', 'ROIC (%)', 'Profit Margin (%)', 'Operating Margin (%)']
                health_widths = [0.6*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.0*inch, 1.0*inch]
                create_focused_table("🏥 Financial Health", health_cols, df_comprehensive, health_widths)
                
                # Section 4: Growth & Dividends - Increased column widths to prevent title overlap
                growth_cols = ['Ticker', 'Revenue Growth (%)', 'Earnings Growth (%)', 'EPS Growth (%)', 'Dividend Yield (%)', 'Dividend Rate ($)', 'Payout Ratio (%)', '5Y Dividend Growth (%)']
                growth_widths = [0.6*inch, 1.0*inch, 1.0*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.8*inch, 1.0*inch]
                create_focused_table("📈 Growth & Dividends", growth_cols, df_comprehensive, growth_widths)
                
                # Section 5: Technical & Trading
                technical_cols = ['Ticker', '52W High ($)', '52W Low ($)', '50D MA ($)', '200D MA ($)', 'Beta', 'Volume', 'Avg Volume', 'Analyst Rating']
                technical_widths = [0.6*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.5*inch, 0.7*inch, 0.7*inch, 0.7*inch]
                create_focused_table("📊 Technical & Trading", technical_cols, df_comprehensive, technical_widths)
                
                story.append(Spacer(1, 10))
                story.append(Paragraph("Note: This comprehensive analysis covers all 5 sections (Overview, Valuation, Financial Health, Growth & Dividends, Technical) with the most important metrics for each position. For complete data and interactive analysis, run the allocation analysis in the Streamlit interface.", styles['Normal']))
                
            else:
                # Fallback: show message when comprehensive data is not available
                story.append(Spacer(1, 10))
                story.append(Paragraph("Note: Detailed financial indicators are not available for this portfolio. To view comprehensive financial metrics for each position, please run the allocation analysis in the Streamlit interface first.", styles['Normal']))
        
        except Exception as e:
            story.append(Paragraph(f"Note: Detailed financial indicators table could not be generated: {str(e)}", styles['Normal']))
        
        # Update progress
        progress_bar.progress(90)
        status_text.text("💾 Finalizing PDF...")
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Update progress
        progress_bar.progress(100)
        status_text.text("✅ PDF generated successfully!")
        
        # Store PDF data in session state for download button
        st.session_state['pdf_buffer'] = pdf_data
        
        return True
        
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return False

# -----------------------
# Single-backtest core (adapted from your code, robust)
# -----------------------
def single_backtest(config, sim_index, reindexed_data):
    stocks_list = config.get('stocks', [])
    raw_tickers = [s.get('ticker') for s in stocks_list if s.get('ticker')]
    # Filter out tickers not present in reindexed_data to avoid crashes for invalid tickers
    if reindexed_data:
        tickers = [t for t in raw_tickers if t in reindexed_data]
    else:
        tickers = raw_tickers[:]
    missing_tickers = [t for t in raw_tickers if t not in tickers]
    if missing_tickers:
        # Log a warning and ignore unknown tickers
        print(f"[ALLOC WARN] Ignoring unknown or missing tickers: {missing_tickers}")
    # Handle duplicate tickers by summing their allocations
    allocations = {}
    include_dividends = {}
    for s in stocks_list:
        if s.get('ticker') and s.get('ticker') in tickers:
            ticker = s.get('ticker')
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
    benchmark_ticker = config.get('benchmark_ticker')
    initial_value = config.get('initial_value', 0)
    # Allocation tracker: ignore added cash for this mode. Use initial_value as current portfolio value.
    added_amount = 0
    added_frequency = 'none'
    # Map frequency to ensure compatibility with get_dates_by_freq
    raw_rebalancing_frequency = config.get('rebalancing_frequency', 'none')
    def map_frequency_for_backtest(freq):
        if freq is None:
            return 'Never'
        freq_map = {
            'Never': 'Never',
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
    
    rebalancing_frequency = map_frequency_for_backtest(raw_rebalancing_frequency)
    use_momentum = config.get('use_momentum', True)
    momentum_windows = config.get('momentum_windows', [])
    calc_beta = config.get('calc_beta', False)
    calc_volatility = config.get('calc_volatility', False)
    beta_window_days = config.get('beta_window_days', 365)
    exclude_days_beta = config.get('exclude_days_beta', 30)
    vol_window_days = config.get('vol_window_days', 365)
    exclude_days_vol = config.get('exclude_days_vol', 30)
    current_data = {t: reindexed_data[t] for t in tickers + [benchmark_ticker] if t in reindexed_data}
    # Respect start_with setting: 'all' (default) or 'oldest' (add assets over time)
    start_with = config.get('start_with', 'all')
    # Precompute first-valid dates for each ticker to decide availability
    start_dates_config = {}
    for t in tickers:
        if t in reindexed_data and isinstance(reindexed_data.get(t), pd.DataFrame):
            fd = reindexed_data[t].first_valid_index()
            start_dates_config[t] = fd if fd is not None else pd.NaT
        else:
            start_dates_config[t] = pd.NaT
    dates_added = set()
    dates_rebal = sorted(get_dates_by_freq(rebalancing_frequency, sim_index[0], sim_index[-1], sim_index))

    # Dictionaries to store historical data for new tables
    historical_allocations = {}
    historical_metrics = {}

    def calculate_momentum(date, current_assets, momentum_windows):
        cumulative_returns, valid_assets = {}, []
        filtered_windows = [w for w in momentum_windows if w.get("weight", 0) > 0]
        # Normalize weights so they sum to 1 (same as app.py)
        total_weight = sum(w.get("weight", 0) for w in filtered_windows)
        if total_weight == 0:
            normalized_weights = [0 for _ in filtered_windows]
        else:
            normalized_weights = [w.get("weight", 0) / total_weight for w in filtered_windows]
        # Only consider assets that exist in current_data (filtered earlier)
        candidate_assets = [t for t in current_assets if t in current_data]
        for t in candidate_assets:
            is_valid, asset_returns = True, 0.0
            df_t = current_data.get(t)
            if not (isinstance(df_t, pd.DataFrame) and 'Close' in df_t.columns and not df_t['Close'].dropna().empty):
                # no usable data for this ticker
                continue
            for idx, window in enumerate(filtered_windows):
                lookback, exclude = window.get("lookback", 0), window.get("exclude", 0)
                weight = normalized_weights[idx]
                start_mom = date - pd.Timedelta(days=lookback)
                end_mom = date - pd.Timedelta(days=exclude)
                sd = start_dates_config.get(t, pd.NaT)
                # If no start date or asset starts after required lookback, mark invalid
                if pd.isna(sd) or sd > start_mom:
                    is_valid = False
                    break
                try:
                    price_start_index = df_t.index.asof(start_mom)
                    price_end_index = df_t.index.asof(end_mom)
                except Exception:
                    is_valid = False
                    break
                if pd.isna(price_start_index) or pd.isna(price_end_index):
                    is_valid = False
                    break
                price_start = df_t.loc[price_start_index, "Close"]
                price_end = df_t.loc[price_end_index, "Close"]
                if pd.isna(price_start) or pd.isna(price_end) or price_start == 0:
                    is_valid = False
                    break
                ret = (price_end - price_start) / price_start
                asset_returns += ret * weight
            if is_valid:
                cumulative_returns[t] = asset_returns
                valid_assets.append(t)
        return cumulative_returns, valid_assets

    def calculate_momentum_weights(returns, valid_assets, date, momentum_strategy='Classic', negative_momentum_strategy='Cash'):
        if not valid_assets: return {}, {}
        rets = {t: returns[t] for t in valid_assets if not pd.isna(returns[t])}
        if not rets: return {}, {}
        beta_vals, vol_vals = {}, {}
        metrics = {t: {} for t in tickers}
        if calc_beta or calc_volatility:
            df_bench = current_data.get(benchmark_ticker)
            if calc_beta:
                start_beta = date - pd.Timedelta(days=beta_window_days)
                end_beta = date - pd.Timedelta(days=exclude_days_beta)
            if calc_volatility:
                start_vol = date - pd.Timedelta(days=vol_window_days)
                end_vol = date - pd.Timedelta(days=exclude_days_vol)
            for t in valid_assets:
                df_t = current_data[t]
                if calc_beta and df_bench is not None:
                    mask_beta = (df_t.index >= start_beta) & (df_t.index <= end_beta)
                    returns_t_beta = df_t.loc[mask_beta, "Price_change"]
                    mask_bench_beta = (df_bench.index >= start_beta) & (df_bench.index <= end_beta)
                    returns_bench_beta = df_bench.loc[mask_bench_beta, "Price_change"]
                    if len(returns_t_beta) < 2 or len(returns_bench_beta) < 2:
                        beta_vals[t] = np.nan
                    else:
                        covariance = np.cov(returns_t_beta, returns_bench_beta)[0,1]
                        variance = np.var(returns_bench_beta)
                        beta_vals[t] = covariance/variance if variance>0 else np.nan
                    metrics[t]['Beta'] = beta_vals[t]
                if calc_volatility:
                    mask_vol = (df_t.index >= start_vol) & (df_t.index <= end_vol)
                    returns_t_vol = df_t.loc[mask_vol, "Price_change"]
                    if len(returns_t_vol) < 2:
                        vol_vals[t] = np.nan
                    else:
                        vol_vals[t] = returns_t_vol.std() * np.sqrt(365)
                    metrics[t]['Volatility'] = vol_vals[t]
        
        for t in rets:
            metrics[t]['Momentum'] = rets[t]

        # Compute initial weights from raw momentum scores (relative/classic) then apply
        # post-filtering by inverse volatility and inverse absolute beta (app.py approach).
        weights = {}
        # raw momentum values
        rets_keys = list(rets.keys())
        all_negative = all(r <= 0 for r in rets.values())

        # Helper: detect relative mode from momentum_strategy string
        relative_mode = isinstance(momentum_strategy, str) and momentum_strategy.lower().startswith('relat')

        if all_negative:
            if negative_momentum_strategy == 'Cash':
                weights = {t: 0 for t in rets_keys}
            elif negative_momentum_strategy == 'Equal weight':
                weights = {t: 1 / len(rets_keys) for t in rets_keys}
            elif negative_momentum_strategy == 'Relative momentum':
                min_score = min(rets.values())
                offset = -min_score + 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
        else:
            if relative_mode:
                min_score = min(rets.values())
                offset = -min_score + 0.01 if min_score < 0 else 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
            else:
                positive_scores = {t: s for t, s in rets.items() if s > 0}
                if positive_scores:
                    sum_positive = sum(positive_scores.values())
                    weights = {t: positive_scores[t] / sum_positive for t in positive_scores}
                    for t in [t for t in rets_keys if rets.get(t, 0) <= 0]:
                        weights[t] = 0
                else:
                    weights = {t: 0 for t in rets_keys}

        # Apply post-filtering using inverse volatility and inverse absolute beta (like app.py)
        fallback_mode = all_negative and negative_momentum_strategy == 'Equal weight'
        if weights and (calc_volatility or calc_beta) and not fallback_mode:
            filtered_weights = {}
            for t, w in weights.items():
                if w > 0:
                    score = 1.0
                    if calc_volatility:
                        v = vol_vals.get(t, np.nan)
                        if not pd.isna(v) and v > 0:
                            score *= (1.0 / v)
                        else:
                            score *= 0
                    if calc_beta:
                        b = beta_vals.get(t, np.nan)
                        if not pd.isna(b):
                            abs_beta = abs(b)
                            if abs_beta > 0:
                                score *= (1.0 / abs_beta)
                            else:
                                score *= 1.0
                    filtered_weights[t] = w * score
            total_filtered = sum(filtered_weights.values())
            if total_filtered > 0:
                weights = {t: v / total_filtered for t, v in filtered_weights.items()}
            else:
                weights = {}

        # Apply minimal threshold filter if enabled
        use_threshold = config.get('use_minimal_threshold', False)
        threshold_percent = config.get('minimal_threshold_percent', 2.0)
        
        if use_threshold and weights:
            threshold_decimal = threshold_percent / 100.0
            
            # STEP 1: Complete rebalancing simulation is already done above
            # (momentum calculation, beta/volatility filtering, etc. - weights are the "simulation" result)
            
            # STEP 2: Check which stocks are below threshold in the simulation
            filtered_weights = {}
            for ticker, weight in weights.items():
                if weight >= threshold_decimal:
                    # Keep stocks above or equal to threshold (remove stocks below threshold)
                    filtered_weights[ticker] = weight
            
            # STEP 3: Do the actual rebalancing with only the remaining stocks
            if filtered_weights:
                total_weight = sum(filtered_weights.values())
                if total_weight > 0:
                    # Normalize remaining stocks to sum to 1.0
                    weights = {ticker: weight / total_weight for ticker, weight in filtered_weights.items()}
                else:
                    weights = {}
            else:
                # If no stocks meet threshold, keep original weights
                weights = weights

        for t in weights:
            metrics[t]['Calculated_Weight'] = weights.get(t, 0)

        # Debug: print metrics summary for this rebal date when beta/vol modifiers are active
        if calc_beta or calc_volatility:
            try:
                debug_lines = [
                    f"[MOM DEBUG] Date: {date} | Ticker: {t} | Momentum: {metrics[t].get('Momentum')} | Beta: {metrics[t].get('Beta')} | Vol: {metrics[t].get('Volatility')} | Weight: {weights.get(t, metrics[t].get('Calculated_Weight'))}"
                    for t in rets_keys
                ]
                for ln in debug_lines:
                    print(ln)
            except Exception as e:
                print(f"[MOM DEBUG] Error printing debug metrics: {e}")

        return weights, metrics
        # --- MODIFIED LOGIC END ---

    values = {t: [0.0] for t in tickers}
    unallocated_cash = [0.0]
    unreinvested_cash = [0.0]
    portfolio_no_additions = [initial_value]
    
    # Initial allocation and metric storage
    if not use_momentum:
        # If start_with is 'oldest', only allocate to tickers that are available at the simulation start
        if start_with == 'oldest':
            available_at_start = [t for t in tickers if start_dates_config.get(t, pd.Timestamp.max) <= sim_index[0]]
            current_allocations = {t: allocations.get(t, 0) if t in available_at_start else 0 for t in tickers}
        else:
            current_allocations = {t: allocations.get(t,0) for t in tickers}
        
        # Apply minimal threshold filter for non-momentum strategies
        use_threshold = config.get('use_minimal_threshold', False)
        threshold_percent = config.get('minimal_threshold_percent', 2.0)
        
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
                if "Dividends" in df.columns:
                    # If dividend is not on a trading day, roll forward to next available trading day
                    if date in df.index:
                        div = df.loc[date, "Dividends"]
                    else:
                        # Find next trading day in index after 'date'
                        future_dates = df.index[df.index > date]
                        if len(future_dates) > 0:
                            div = df.loc[future_dates[0], "Dividends"]
                var = df.loc[date, "Price_change"] if date in df.index else 0.0
                if include_dividends.get(t, False):
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
            if "Dividends" in df.columns:
                if date in df.index:
                    div = df.loc[date, "Dividends"]
                else:
                    future_dates = df.index[df.index > date]
                    if len(future_dates) > 0:
                        div = df.loc[future_dates[0], "Dividends"]
            var = df.loc[date, "Price_change"] if date in df.index else 0.0
            if include_dividends.get(t, False):
                rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                val_new = val_prev * (1 + rate_of_return)
            else:
                val_new = val_prev * (1 + var)
            values[t].append(val_new)
        unallocated_cash.append(unallocated_cash[-1])
        unreinvested_cash.append(unreinvested_cash[-1] + total_unreinvested_dividends)
        portfolio_no_additions.append(portfolio_no_additions[-1] * daily_growth_factor)
        
        current_total = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
        if date in dates_rebal and set(tickers):
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
                        for t in tickers:
                            values[t][-1] = current_total * weights.get(t, 0)
                        unreinvested_cash[-1] = 0
                        unallocated_cash[-1] = 0
            else:
                # Non-momentum rebalancing: respect 'start_with' option
                if start_with == 'oldest':
                    # Only consider tickers that have data by this rebalancing date
                    available = [t for t in tickers if start_dates_config.get(t, pd.Timestamp.max) <= date]
                    sum_alloc_avail = sum(allocations.get(t,0) for t in available)
                    if sum_alloc_avail > 0:
                        for t in tickers:
                            if t in available:
                                weight = allocations.get(t,0)/sum_alloc_avail
                                values[t][-1] = current_total * weight
                            else:
                                values[t][-1] = 0
                        unreinvested_cash[-1] = 0
                        unallocated_cash[-1] = 0
                    else:
                        # No assets available yet — keep everything as cash
                        for t in tickers:
                            values[t][-1] = 0
                        unreinvested_cash[-1] = 0
                        unallocated_cash[-1] = current_total
                else:
                    # Apply threshold filter for non-momentum strategies during rebalancing
                    use_threshold = config.get('use_minimal_threshold', False)
                    threshold_percent = config.get('minimal_threshold_percent', 2.0)
                    
                    if use_threshold:
                        threshold_decimal = threshold_percent / 100.0
                        
                        # First: Filter out stocks below threshold
                        filtered_allocations = {}
                        for t in tickers:
                            allocation = allocations.get(t, 0)
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
                    else:
                        # Use original allocations if threshold filter is disabled
                        rebalance_allocations = {t: allocations.get(t, 0) for t in tickers}
                    
                    sum_alloc = sum(rebalance_allocations.values())
                    if sum_alloc > 0:
                        for t in tickers:
                            weight = rebalance_allocations.get(t, 0) / sum_alloc
                            values[t][-1] = current_total * weight
                        unreinvested_cash[-1] = 0
                        unallocated_cash[-1] = 0
            
            # Store allocations at rebalancing date
            current_total_after_rebal = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
            if current_total_after_rebal > 0:
                allocs = {t: values[t][-1] / current_total_after_rebal for t in tickers}
                allocs['CASH'] = unallocated_cash[-1] / current_total_after_rebal if current_total_after_rebal > 0 else 0
                historical_allocations[date] = allocs
            else:
                allocs = {t: 0 for t in tickers}
                allocs['CASH'] = 0
                historical_allocations[date] = allocs

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
# Main App Logic
# -----------------------

from copy import deepcopy
# Use page-scoped session keys so this page does not share state with other pages
if 'alloc_portfolio_configs' not in st.session_state:
    # initialize from existing global configs if present, but deep-copy to avoid shared references
            st.session_state.alloc_portfolio_configs = deepcopy(st.session_state.get('alloc_portfolio_configs', default_configs))
if 'alloc_active_portfolio_index' not in st.session_state:
    st.session_state.alloc_active_portfolio_index = 0
if 'alloc_paste_json_text' not in st.session_state:
    st.session_state.alloc_paste_json_text = ""
if 'alloc_rerun_flag' not in st.session_state:
    st.session_state.alloc_rerun_flag = False

def add_portfolio_callback():
    new_portfolio = default_configs[1].copy()
    new_portfolio['name'] = f"New Portfolio {len(st.session_state.alloc_portfolio_configs) + 1}"
    st.session_state.alloc_portfolio_configs.append(new_portfolio)
    st.session_state.alloc_active_portfolio_index = len(st.session_state.alloc_portfolio_configs) - 1
    st.session_state.alloc_rerun_flag = True

def remove_portfolio_callback():
    if len(st.session_state.alloc_portfolio_configs) > 1:
        st.session_state.alloc_portfolio_configs.pop(st.session_state.alloc_active_portfolio_index)
        st.session_state.alloc_active_portfolio_index = max(0, st.session_state.alloc_active_portfolio_index - 1)
        st.session_state.alloc_rerun_flag = True

def add_stock_callback():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'].append({'ticker': '', 'allocation': 0.0, 'include_dividends': True})
    st.session_state.alloc_rerun_flag = True

def remove_stock_callback(ticker):
    """Immediate stock removal callback"""
    try:
        active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
        stocks = active_portfolio['stocks']
        
        # Find and remove the stock with matching ticker
        for i, stock in enumerate(stocks):
            if stock['ticker'] == ticker:
                stocks.pop(i)
                # If this was the last stock, add an empty one
                if len(stocks) == 0:
                    stocks.append({'ticker': '', 'allocation': 0.0, 'include_dividends': True})
                st.session_state.alloc_rerun_flag = True
                break
    except (IndexError, KeyError):
        pass

def normalize_stock_allocations_callback():
    if 'alloc_portfolio_configs' not in st.session_state or 'alloc_active_portfolio_index' not in st.session_state:
        return
    stocks = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks']
    valid_stocks = [s for s in stocks if s['ticker']]
    total_alloc = sum(s['allocation'] for s in valid_stocks)
    if total_alloc > 0:
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] /= total_alloc
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = int(s['allocation'] * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] = stocks
    st.session_state.alloc_rerun_flag = True

def equal_stock_allocation_callback():
    if 'alloc_portfolio_configs' not in st.session_state or 'alloc_active_portfolio_index' not in st.session_state:
        return
    stocks = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks']
    valid_stocks = [s for s in stocks if s['ticker']]
    if valid_stocks:
        equal_weight = 1.0 / len(valid_stocks)
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] = equal_weight
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = int(equal_weight * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] = stocks
    st.session_state.alloc_rerun_flag = True
    
def reset_portfolio_callback():
    current_name = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['name']
    default_cfg_found = next((cfg for cfg in default_configs if cfg['name'] == current_name), None)
    if default_cfg_found is None:
        default_cfg_found = default_configs[1].copy()
        default_cfg_found['name'] = current_name
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index] = default_cfg_found
    st.session_state.alloc_rerun_flag = True

def reset_stock_selection_callback():
    current_name = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['name']
    default_cfg_found = next((cfg for cfg in default_configs if cfg['name'] == current_name), None)
    if default_cfg_found is None:
        default_cfg_found = default_configs[1].copy()
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] = default_cfg_found['stocks']
    st.session_state.alloc_rerun_flag = True

def reset_momentum_windows_callback():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = [
        {"lookback": 365, "exclude": 30, "weight": 0.5},
        {"lookback": 180, "exclude": 30, "weight": 0.3},
        {"lookback": 120, "exclude": 30, "weight": 0.2},
    ]
    st.session_state.alloc_rerun_flag = True

def reset_beta_callback():
    # Reset beta lookback/exclude to defaults and enable beta calculation for alloc page
    idx = st.session_state.alloc_active_portfolio_index
    st.session_state.alloc_portfolio_configs[idx]['beta_window_days'] = 365
    st.session_state.alloc_portfolio_configs[idx]['exclude_days_beta'] = 30
    # Ensure checkbox state reflects enabled
    st.session_state.alloc_portfolio_configs[idx]['calc_beta'] = True
    st.session_state['alloc_active_calc_beta'] = True
    st.session_state.alloc_rerun_flag = True

def reset_vol_callback():
    # Reset volatility lookback/exclude to defaults and enable volatility calculation
    idx = st.session_state.alloc_active_portfolio_index
    st.session_state.alloc_portfolio_configs[idx]['vol_window_days'] = 365
    st.session_state.alloc_portfolio_configs[idx]['exclude_days_vol'] = 30
    st.session_state.alloc_portfolio_configs[idx]['calc_volatility'] = True
    st.session_state['alloc_active_calc_vol'] = True
    st.session_state.alloc_rerun_flag = True

def add_momentum_window_callback():
    # Append a new momentum window with modest defaults (alloc page)
    idx = st.session_state.alloc_active_portfolio_index
    cfg = st.session_state.alloc_portfolio_configs[idx]
    if 'momentum_windows' not in cfg:
        cfg['momentum_windows'] = []
    # default new window
    cfg['momentum_windows'].append({"lookback": 90, "exclude": 30, "weight": 0.1})
    st.session_state.alloc_portfolio_configs[idx] = cfg
    st.session_state.alloc_rerun_flag = True

def remove_momentum_window_callback():
    idx = st.session_state.alloc_active_portfolio_index
    cfg = st.session_state.alloc_portfolio_configs[idx]
    if 'momentum_windows' in cfg and cfg['momentum_windows']:
        cfg['momentum_windows'].pop()
        st.session_state.alloc_portfolio_configs[idx] = cfg
        st.session_state.alloc_rerun_flag = True

def normalize_momentum_weights_callback():
    # Use page-scoped configs for allocations page
    if 'alloc_portfolio_configs' not in st.session_state or 'alloc_active_portfolio_index' not in st.session_state:
        return
    active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
    total_weight = sum(w['weight'] for w in active_portfolio.get('momentum_windows', []))
    if total_weight > 0:
        for idx, w in enumerate(active_portfolio.get('momentum_windows', [])):
            w['weight'] /= total_weight
            weight_key = f"alloc_weight_input_active_{st.session_state.alloc_active_portfolio_index}_{idx}"
            # Sanitize weight to prevent StreamlitValueAboveMaxError
            weight = w['weight']
            if isinstance(weight, (int, float)):
                # Convert decimal to percentage, ensuring it's within bounds
                weight_percentage = max(0.0, min(weight * 100.0, 100.0))
            else:
                # Invalid weight, set to default
                weight_percentage = 10.0
            st.session_state[weight_key] = int(weight_percentage)
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = active_portfolio.get('momentum_windows', [])
    st.session_state.alloc_rerun_flag = True

def paste_json_callback():
    try:
        # Use the SAME parsing logic as successful PDF extraction
        raw_text = st.session_state.get('alloc_paste_json_text', '{}')
        
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
        
        # Debug: Show what we received
        st.info(f"Received JSON keys: {list(json_data.keys())}")
        if 'tickers' in json_data:
            st.info(f"Tickers in JSON: {json_data['tickers']}")
        if 'stocks' in json_data:
            st.info(f"Stocks in JSON: {json_data['stocks']}")
        if 'momentum_windows' in json_data:
            st.info(f"Momentum windows in JSON: {json_data['momentum_windows']}")
        if 'use_momentum' in json_data:
            st.info(f"Use momentum in JSON: {json_data['use_momentum']}")
        
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
            
            # Debug output
            st.info(f"Converted {len(stocks)} stocks from legacy format: {[s['ticker'] for s in stocks]}")
        
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
        
        # Map frequency values from app.py format to Allocations format
        def map_frequency(freq):
            if freq is None:
                return 'Never'
            freq_map = {
                'Never': 'Never',
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
        
        # Allocations page specific: ensure all required fields are present
        # and ignore fields that are specific to other pages
        allocations_config = {
            'name': json_data.get('name', 'Allocation Portfolio'),
            'stocks': stocks,
            'benchmark_ticker': json_data.get('benchmark_ticker', '^GSPC'),
            'initial_value': json_data.get('initial_value', 10000),
            'added_amount': json_data.get('added_amount', 0),  # Allocations page typically doesn't use additions
            'added_frequency': map_frequency(json_data.get('added_frequency', 'Never')),  # Allocations page typically doesn't use additions
            'rebalancing_frequency': map_frequency(json_data.get('rebalancing_frequency', 'Monthly')),
            'start_date_user': json_data.get('start_date_user'),
            'end_date_user': json_data.get('end_date_user'),
            'start_with': json_data.get('start_with', 'all'),
            'use_momentum': json_data.get('use_momentum', True),
            'momentum_strategy': momentum_strategy,
            'negative_momentum_strategy': negative_momentum_strategy,
            'momentum_windows': momentum_windows,
            'use_minimal_threshold': json_data.get('use_minimal_threshold', False),
            'minimal_threshold_percent': json_data.get('minimal_threshold_percent', 2.0),
            'calc_beta': json_data.get('calc_beta', True),
            'calc_volatility': json_data.get('calc_volatility', True),
            'beta_window_days': json_data.get('beta_window_days', 365),
            'exclude_days_beta': json_data.get('exclude_days_beta', 30),
            'vol_window_days': json_data.get('vol_window_days', 365),
            'exclude_days_vol': json_data.get('exclude_days_vol', 30),
        }
        
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index] = allocations_config
        
        # Update session state for threshold settings
        st.session_state['alloc_active_use_threshold'] = allocations_config.get('use_minimal_threshold', False)
        st.session_state['alloc_active_threshold_percent'] = allocations_config.get('minimal_threshold_percent', 2.0)
        
        st.success("Portfolio configuration updated from JSON (Allocations page).")
        st.info(f"Final stocks list: {[s['ticker'] for s in allocations_config['stocks']]}")
        st.info(f"Final momentum windows: {allocations_config['momentum_windows']}")
        st.info(f"Final use_momentum: {allocations_config['use_momentum']}")
        st.info(f"Final threshold settings: use={allocations_config.get('use_minimal_threshold', False)}, percent={allocations_config.get('minimal_threshold_percent', 2.0)}")
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check the text and try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    st.session_state.alloc_rerun_flag = True

def update_active_portfolio_index():
    # Allocation page: keep a page-scoped index. If a selector exists, respect it; otherwise default to 0
    selected_name = st.session_state.get('alloc_portfolio_selector', None)
    portfolio_configs = st.session_state.get('alloc_portfolio_configs', [])
    portfolio_names = [cfg.get('name', '') for cfg in portfolio_configs]
    if selected_name and selected_name in portfolio_names:
        st.session_state.alloc_active_portfolio_index = portfolio_names.index(selected_name)
    else:
        st.session_state.alloc_active_portfolio_index = 0 if portfolio_names else None
    st.session_state.alloc_rerun_flag = True

def update_name():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['name'] = st.session_state.get('alloc_active_name', '')

def update_initial():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['initial_value'] = st.session_state.get('alloc_active_initial', 0)

def update_added_amount():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['added_amount'] = st.session_state.get('alloc_active_added_amount', 0)

def update_add_freq():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['added_frequency'] = st.session_state.get('alloc_active_add_freq', 'none')

def update_rebal_freq():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['rebalancing_frequency'] = st.session_state.get('alloc_active_rebal_freq', 'none')

def update_benchmark():
    # Convert benchmark ticker to uppercase
    benchmark_val = st.session_state.get('alloc_active_benchmark', '')
    upper_benchmark = benchmark_val.upper()
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['benchmark_ticker'] = upper_benchmark
    # Update the widget to show uppercase value
    st.session_state['alloc_active_benchmark'] = upper_benchmark

def update_use_momentum():
    current_val = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('use_momentum', True)
    new_val = st.session_state.get('alloc_active_use_momentum', True)
    if current_val != new_val:
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['use_momentum'] = new_val
        if new_val:
            st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = [
                {"lookback": 365, "exclude": 30, "weight": 0.5},
                {"lookback": 180, "exclude": 30, "weight": 0.3},
                {"lookback": 120, "exclude": 30, "weight": 0.2},
            ]
        else:
            st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = []
        st.session_state.alloc_rerun_flag = True



def update_calc_beta():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['calc_beta'] = st.session_state.get('alloc_active_calc_beta', True)

def update_beta_window():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['beta_window_days'] = st.session_state.get('alloc_active_beta_window', 365)

def update_beta_exclude():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['exclude_days_beta'] = st.session_state.get('alloc_active_beta_exclude', 30)

def update_calc_vol():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['calc_volatility'] = st.session_state.get('alloc_active_calc_vol', True)

def update_vol_window():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['vol_window_days'] = st.session_state.get('alloc_active_vol_window', 365)

def update_vol_exclude():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['exclude_days_vol'] = st.session_state.get('alloc_active_vol_exclude', 30)

def update_use_threshold():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['use_minimal_threshold'] = st.session_state.alloc_active_use_threshold

def update_threshold_percent():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['minimal_threshold_percent'] = st.session_state.alloc_active_threshold_percent

# Sidebar simplified for single-portfolio allocation tracker
st.sidebar.title("Allocation Tracker")


# Work with the first portfolio as active (single-portfolio mode). Keep inputs accessible.
active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
# Do not show portfolio name in allocation tracker. Keep a page-scoped session key for compatibility.
if "alloc_active_name" not in st.session_state:
    st.session_state["alloc_active_name"] = active_portfolio['name']

col_left, col_right = st.columns([1, 1])
with col_left:
    if "alloc_active_initial" not in st.session_state:
        # Treat this as the current portfolio value (not a backtest initial cash)
        st.session_state["alloc_active_initial"] = int(active_portfolio.get('initial_value', 0))
    st.number_input("Portfolio Value ($)", min_value=0, step=1000, format="%d", key="alloc_active_initial", on_change=update_initial, help="Current total portfolio value used to compute required shares.")
# Removed Added Amount / Added Frequency UI - allocation tracker is not running periodic additions

# Swap positions: show Rebalancing Frequency first, then Added Frequency.
# Use two equal-width columns and make selectboxes use the container width so they match visually.
col_freq_rebal, col_freq_add = st.columns([1, 1])
freq_options = ["Never", "Weekly", "Biweekly", "Monthly", "Quarterly", "Semiannually", "Annually"]
with col_freq_rebal:
    if "alloc_active_rebal_freq" not in st.session_state:
        st.session_state["alloc_active_rebal_freq"] = active_portfolio['rebalancing_frequency']
    st.selectbox("Rebalancing Frequency", freq_options, key="alloc_active_rebal_freq", on_change=update_rebal_freq, help="How often the portfolio is rebalanced.", )
# Note: Added Frequency removed for allocation tracker

# Rebalancing and Added Frequency explanation removed for allocation tracker UI

if "alloc_active_benchmark" not in st.session_state:
    st.session_state["alloc_active_benchmark"] = active_portfolio['benchmark_ticker']
st.text_input("Benchmark Ticker (default: ^GSPC, used for beta calculation)", key="alloc_active_benchmark", on_change=update_benchmark)

st.subheader("Tickers")
col_ticker_buttons = st.columns([0.3, 0.3, 0.3, 0.1])
with col_ticker_buttons[0]:
    if st.button("Normalize Tickers %", on_click=normalize_stock_allocations_callback, use_container_width=True):
        pass
with col_ticker_buttons[1]:
    if st.button("Equal Allocation %", on_click=equal_stock_allocation_callback, use_container_width=True):
        pass
with col_ticker_buttons[2]:
    if st.button("Reset Tickers", on_click=reset_stock_selection_callback, use_container_width=True):
        pass

# Calculate live total ticker allocation
valid_tickers = [s for s in st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] if s['ticker']]
total_ticker_allocation = sum(s['allocation'] for s in valid_tickers)

if active_portfolio['use_momentum']:
    st.info("Ticker allocations are not used directly for Momentum strategies.")
else:
    if abs(total_ticker_allocation - 1.0) > 0.001:
        st.warning(f"Total ticker allocation is {total_ticker_allocation*100:.2f}%, not 100%. Click 'Normalize' to fix.")
    else:
        st.success(f"Total ticker allocation is {total_ticker_allocation*100:.2f}%.")

def update_stock_allocation(index):
    try:
        key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][index]['allocation'] = float(val) / 100.0
    except Exception:
        # Ignore transient errors (e.g., active_portfolio_index changed); UI will reflect state on next render
        return


def update_stock_ticker(index):
    try:
        key = f"alloc_ticker_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        
        # Convert the input value to uppercase
        upper_val = val.upper()

        # Update the portfolio configuration with the uppercase value
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][index]['ticker'] = upper_val
        
        # Update the text box's state to show the uppercase value
        st.session_state[key] = upper_val

    except Exception:
        # Defensive: if portfolio index or structure changed, skip silently
        return


def update_stock_dividends(index):
    try:
        key = f"alloc_div_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][index]['include_dividends'] = bool(val)
    except Exception:
        return

# Update active_portfolio
active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
 
for i in range(len(active_portfolio['stocks'])):
    stock = active_portfolio['stocks'][i]
    col_t, col_a, col_d, col_b = st.columns([0.2, 0.2, 0.3, 0.15])
    with col_t:
        ticker_key = f"alloc_ticker_{st.session_state.alloc_active_portfolio_index}_{i}"
        if ticker_key not in st.session_state:
            st.session_state[ticker_key] = stock['ticker']
        st.text_input("Ticker", key=ticker_key, label_visibility="visible", on_change=update_stock_ticker, args=(i,))
    with col_a:
        use_mom = st.session_state.get('alloc_active_use_momentum', active_portfolio.get('use_momentum', True))
        if not use_mom:
            alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{i}"
            if alloc_key not in st.session_state:
                st.session_state[alloc_key] = int(stock['allocation'] * 100)
            st.number_input("Allocation %", min_value=0, step=1, format="%d", key=alloc_key, label_visibility="visible", on_change=update_stock_allocation, args=(i,))
            if st.session_state[alloc_key] != int(stock['allocation'] * 100):
                st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][i]['allocation'] = st.session_state[alloc_key] / 100.0
        else:
            st.write("")
    with col_d:
        div_key = f"alloc_div_{st.session_state.alloc_active_portfolio_index}_{i}"
        if div_key not in st.session_state:
            st.session_state[div_key] = stock['include_dividends']
        st.checkbox("Include Dividends", key=div_key)
        if st.session_state[div_key] != stock['include_dividends']:
            st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][i]['include_dividends'] = st.session_state[div_key]
    with col_b:
        st.write("")
        if st.button("Remove", key=f"alloc_rem_stock_{st.session_state.alloc_active_portfolio_index}_{i}_{stock['ticker']}_{id(stock)}", on_click=remove_stock_callback, args=(stock['ticker'],)):
            pass

if st.button("Add Ticker", on_click=add_stock_callback):
    pass

# Bulk ticker input section
with st.expander("📝 Bulk Ticker Input", expanded=False):
    st.markdown("**Enter multiple tickers separated by spaces or commas:**")
    
    # Initialize bulk ticker input in session state
    if 'alloc_bulk_tickers' not in st.session_state:
        st.session_state.alloc_bulk_tickers = ""
    
    # Auto-populate bulk ticker input with current tickers
    portfolio_index = st.session_state.alloc_active_portfolio_index
    current_tickers = [stock['ticker'] for stock in st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'] if stock['ticker']]
    if current_tickers:
        current_ticker_string = ' '.join(current_tickers)
        if st.session_state.alloc_bulk_tickers != current_ticker_string:
            st.session_state.alloc_bulk_tickers = current_ticker_string
    
    # Text area for bulk ticker input
    bulk_tickers = st.text_area(
        "Tickers (e.g., SPY QQQ GLD TLT or SPY,QQQ,GLD,TLT)",
        value=st.session_state.alloc_bulk_tickers,
        key="alloc_bulk_ticker_input",
        height=100,
        help="Enter ticker symbols separated by spaces or commas. Click 'Fill Tickers' to replace tickers (keeps existing allocations)."
    )
    
    if st.button("Fill Tickers", key="alloc_fill_tickers_btn"):
        if bulk_tickers.strip():
            # Parse tickers (split by comma or space)
            ticker_list = []
            for ticker in bulk_tickers.replace(',', ' ').split():
                ticker = ticker.strip().upper()
                if ticker:
                    ticker_list.append(ticker)
            
            if ticker_list:
                portfolio_index = st.session_state.alloc_active_portfolio_index
                current_stocks = st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'].copy()
                
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
                
                # Update the portfolio with new stocks
                st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'] = new_stocks
                
                # Update the active_portfolio reference to match session state
                active_portfolio['stocks'] = new_stocks
                
                # Clear any existing session state keys for individual ticker inputs to force refresh
                for key in list(st.session_state.keys()):
                    if key.startswith(f"alloc_ticker_{portfolio_index}_") or key.startswith(f"alloc_input_alloc_{portfolio_index}_"):
                        del st.session_state[key]
                
                st.success(f"✅ Replaced tickers with: {', '.join(ticker_list)}")
                st.info("💡 **Note:** Existing allocations preserved. Adjust allocations manually if needed.")
                
                # Force immediate rerun
                st.rerun()
            else:
                st.error("❌ No valid tickers found. Please enter ticker symbols separated by spaces or commas.")
        else:
            st.error("❌ Please enter ticker symbols.")


st.subheader("Strategy")
if "alloc_active_use_momentum" not in st.session_state:
    st.session_state["alloc_active_use_momentum"] = active_portfolio['use_momentum']
if "alloc_active_use_threshold" not in st.session_state:
    st.session_state["alloc_active_use_threshold"] = active_portfolio.get('use_minimal_threshold', False)
if "alloc_active_threshold_percent" not in st.session_state:
    st.session_state["alloc_active_threshold_percent"] = active_portfolio.get('minimal_threshold_percent', 2.0)
st.checkbox("Use Momentum Strategy", key="alloc_active_use_momentum", on_change=update_use_momentum, help="Enables momentum-based weighting of stocks.")

if active_portfolio['use_momentum']:
    st.markdown("---")
    col_mom_options, col_beta_vol = st.columns(2)
    with col_mom_options:
        st.markdown("**Momentum Strategy Options**")
        momentum_strategy = st.selectbox(
            "Momentum strategy when NOT all negative:",
            ["Classic", "Relative Momentum"],
            index=["Classic", "Relative Momentum"].index(active_portfolio.get('momentum_strategy', 'Classic')),
            key=f"momentum_strategy_{st.session_state.alloc_active_portfolio_index}"
        )
        negative_momentum_strategy = st.selectbox(
            "Strategy when ALL momentum scores are negative:",
            ["Cash", "Equal weight", "Relative momentum"],
            index=["Cash", "Equal weight", "Relative momentum"].index(active_portfolio.get('negative_momentum_strategy', 'Cash')),
            key=f"negative_momentum_strategy_{st.session_state.alloc_active_portfolio_index}"
        )
        active_portfolio['momentum_strategy'] = momentum_strategy
        active_portfolio['negative_momentum_strategy'] = negative_momentum_strategy
        st.markdown("💡 **Note:** These options control how weights are assigned based on momentum scores.")

    with col_beta_vol:
        if "alloc_active_calc_beta" not in st.session_state:
            st.session_state["alloc_active_calc_beta"] = active_portfolio.get('calc_beta', True)
        st.checkbox("Include Beta in momentum weighting", key="alloc_active_calc_beta", on_change=update_calc_beta, help="Incorporates a stock's Beta (volatility relative to the benchmark) into its momentum score.")
        if st.session_state.get('alloc_active_calc_beta', False):
            if "alloc_active_beta_window" not in st.session_state:
                st.session_state["alloc_active_beta_window"] = active_portfolio['beta_window_days']
            if "alloc_active_beta_exclude" not in st.session_state:
                st.session_state["alloc_active_beta_exclude"] = active_portfolio['exclude_days_beta']
            st.number_input("Beta Lookback (days)", min_value=1, key="alloc_active_beta_window", on_change=update_beta_window)
            st.number_input("Beta Exclude (days)", min_value=0, key="alloc_active_beta_exclude", on_change=update_beta_exclude)
            if st.button("Reset Beta", on_click=reset_beta_callback):
                pass
        if "alloc_active_calc_vol" not in st.session_state:
            st.session_state["alloc_active_calc_vol"] = active_portfolio.get('calc_volatility', True)
        st.checkbox("Include Volatility in momentum weighting", key="alloc_active_calc_vol", on_change=update_calc_vol, help="Incorporates a stock's volatility (standard deviation of returns) into its momentum score.")
        if st.session_state.get('alloc_active_calc_vol', False):
            if "alloc_active_vol_window" not in st.session_state:
                st.session_state["alloc_active_vol_window"] = active_portfolio['vol_window_days']
            if "alloc_active_vol_exclude" not in st.session_state:
                st.session_state["alloc_active_vol_exclude"] = active_portfolio['exclude_days_vol']
            st.number_input("Volatility Lookback (days)", min_value=1, key="alloc_active_vol_window", on_change=update_vol_window)
            st.number_input("Volatility Exclude (days)", min_value=0, key="alloc_active_vol_exclude", on_change=update_vol_exclude)
            if st.button("Reset Volatility", on_click=reset_vol_callback):
                pass
    
    # Minimal Threshold Filter Section
    st.markdown("---")
    st.subheader("Minimal Threshold Filter")
    
    # Initialize threshold settings if not present
    if "alloc_active_use_threshold" not in st.session_state:
        st.session_state["alloc_active_use_threshold"] = active_portfolio.get('use_minimal_threshold', False)
    if "alloc_active_threshold_percent" not in st.session_state:
        st.session_state["alloc_active_threshold_percent"] = active_portfolio.get('minimal_threshold_percent', 2.0)
    
    st.checkbox(
        "Enable Minimal Threshold Filter", 
        key="alloc_active_use_threshold", 
        on_change=update_use_threshold,
        help="Exclude stocks with allocations below the threshold percentage and normalize remaining allocations to 100%"
    )
    
    if st.session_state.get("alloc_active_use_threshold", False):
        st.number_input(
            "Minimal Threshold (%)", 
            min_value=0.1, 
            max_value=50.0, 
            value=st.session_state.get("alloc_active_threshold_percent", 2.0), 
            step=0.1,
            key="alloc_active_threshold_percent", 
            on_change=update_threshold_percent,
            help="Stocks with allocations below this percentage will be excluded and their weight redistributed to remaining stocks"
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
        key = f"alloc_lookback_active_{st.session_state.alloc_active_portfolio_index}_{index}"
        momentum_windows = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('momentum_windows', [])
        if index < len(momentum_windows):
            momentum_windows[index]['lookback'] = st.session_state.get(key, None)

    def update_momentum_exclude(index):
        key = f"alloc_exclude_active_{st.session_state.alloc_active_portfolio_index}_{index}"
        momentum_windows = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('momentum_windows', [])
        if index < len(momentum_windows):
            momentum_windows[index]['exclude'] = st.session_state.get(key, None)
    
    def update_momentum_weight(index):
        key = f"alloc_weight_input_active_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        momentum_windows = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('momentum_windows', [])
        if index < len(momentum_windows):
            momentum_windows[index]['weight'] = val / 100.0

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

    for j in range(len(active_portfolio.get('momentum_windows', []))):
        with st.container():
            col_mw1, col_mw2, col_mw3 = st.columns(3)
            lookback_key = f"alloc_lookback_active_{st.session_state.alloc_active_portfolio_index}_{j}"
            exclude_key = f"alloc_exclude_active_{st.session_state.alloc_active_portfolio_index}_{j}"
            weight_key = f"alloc_weight_input_active_{st.session_state.alloc_active_portfolio_index}_{j}"
            
            # Initialize session state values if not present
            if lookback_key not in st.session_state:
                st.session_state[lookback_key] = int(active_portfolio['momentum_windows'][j]['lookback'])
            if exclude_key not in st.session_state:
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
            
            with col_mw1:
                st.number_input(f"Lookback {j+1}", min_value=1, key=lookback_key, label_visibility="collapsed", on_change=update_momentum_lookback, args=(j,))
            with col_mw2:
                st.number_input(f"Exclude {j+1}", min_value=0, key=exclude_key, label_visibility="collapsed", on_change=update_momentum_exclude, args=(j,))
            with col_mw3:
                st.number_input(f"Weight {j+1}", min_value=0, max_value=100, step=1, format="%d", key=weight_key, label_visibility="collapsed", on_change=update_momentum_weight, args=(j,))
else:
    
    active_portfolio['momentum_windows'] = []

with st.expander("JSON Configuration (Copy & Paste)", expanded=False):
    # Clean portfolio config for export by removing unused settings
    cleaned_config = active_portfolio.copy()
    cleaned_config.pop('use_relative_momentum', None)
    cleaned_config.pop('equal_if_all_negative', None)
    config_json = json.dumps(cleaned_config, indent=4)
    st.code(config_json, language='json')
    # Fixed JSON copy button
    import streamlit.components.v1 as components
    copy_html = f"""
    <button onclick='navigator.clipboard.writeText({json.dumps(config_json)});' style='margin-bottom:10px;'>Copy to Clipboard</button>
    """
    components.html(copy_html, height=40)
    
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
        portfolio_name = active_portfolio.get('name', 'Portfolio')
        
        # Use custom name if provided, otherwise use portfolio name
        if custom_name.strip():
            title = f"Allocations - {custom_name.strip()} - JSON Configuration"
            subject = f"JSON Configuration: {custom_name.strip()}"
        else:
            title = f"Allocations - {portfolio_name} - JSON Configuration"
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
            creator="Allocations Application"
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
        placeholder=f"e.g., {active_portfolio.get('name', 'Portfolio')} Allocation Config, Asset Setup Analysis",
        help="Leave empty to use automatic naming based on portfolio name",
        key="alloc_individual_custom_pdf_name"
    )
    
    if st.button("📄 Download JSON as PDF", help="Download a PDF containing the JSON configuration for easy copying", key="alloc_json_pdf_btn"):
        try:
            pdf_data = generate_json_pdf(custom_individual_pdf_name)
            
            # Generate filename based on custom name or default
            if custom_individual_pdf_name.strip():
                clean_name = custom_individual_pdf_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
                filename = f"{clean_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            else:
                filename = f"allocations_config_{active_portfolio.get('name', 'portfolio').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            st.download_button(
                label="💾 Download Allocations JSON PDF",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf",
                key="alloc_json_pdf_download"
            )
            st.success("PDF generated successfully! Click the download button above.")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
    

    st.text_area("Paste JSON Here to Update Portfolio", key="alloc_paste_json_text", height=200)
    st.button("Update with Pasted JSON", on_click=paste_json_callback)
    
    # Add PDF drag and drop functionality
    st.markdown("**OR** 📎 **Drag & Drop JSON PDF:**")
    
    def extract_json_from_pdf_alloc(pdf_file):
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
        key="alloc_individual_pdf_upload"
    )
    
    if uploaded_pdf is not None:
        json_data, error = extract_json_from_pdf_alloc(uploaded_pdf)
        if json_data:
            # Store the extracted JSON in a different session state key to avoid widget conflicts
            st.session_state["alloc_extracted_json"] = json.dumps(json_data, indent=4)
            st.success(f"✅ Successfully extracted JSON from {uploaded_pdf.name}")
            st.info("👇 Click the button below to load the JSON into the text area.")
            def load_extracted_json():
                st.session_state["alloc_paste_json_text"] = st.session_state["alloc_extracted_json"]
            
            st.button("📋 Load Extracted JSON", key="load_extracted_json", on_click=load_extracted_json)
        else:
            st.error(f"❌ Failed to extract JSON from PDF: {error}")
            st.info("💡 Make sure the PDF contains valid JSON content (generated by this app)")

# Validation constants
_TOTAL_TOL = 1.0
_ALLOC_TOL = 1.0

# Move Run Backtest to the first sidebar to make it conspicuous and separate from config
if st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True):
    
    # Pre-backtest validation check for all portfolios
    # Prefer the allocations page configs when present so this page's edits are included
    configs_to_run = st.session_state.get('alloc_portfolio_configs', [])
    # Local alias used throughout the run block
    portfolio_list = configs_to_run
    # Set flag to show metrics after running backtest
    st.session_state['alloc_backtest_run'] = True
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
                validation_errors.append(f"Portfolio '{cfg['name']}' is not using momentum, but the total ticker allocation is {total_ticker_allocation*100:.2f}% (must be 100%)")
                valid_configs = False
                
    # Initialize progress bar
    progress_bar = st.empty()
    
    if not valid_configs:
        for error in validation_errors:
            st.error(error)
        # Don't run the backtest, but continue showing the UI
        progress_bar.empty()
        st.stop()
    else:
        # Show standalone popup notification that code is really running
        st.toast("**Code is running!** Starting backtest...", icon="🚀")
        
        progress_bar.progress(0, text="Starting backtest...")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        all_tickers = sorted(list(set(s['ticker'] for cfg in portfolio_list for s in cfg['stocks'] if s['ticker']) | set(cfg.get('benchmark_ticker') for cfg in portfolio_list if 'benchmark_ticker' in cfg)))
        all_tickers = [t for t in all_tickers if t]
        print("Downloading data for all tickers...")
        data = {}
        invalid_tickers = []
        for i, t in enumerate(all_tickers):
            try:
                progress_text = f"Downloading data for {t} ({i+1}/{len(all_tickers)})..."
                progress_bar.progress((i + 1) / (len(all_tickers) + len(portfolio_list)), text=progress_text)
                hist = get_ticker_data(t, period="max", auto_adjust=False)
                if hist.empty:
                    print(f"No data available for {t}")
                    invalid_tickers.append(t)
                    continue
                # Force tz-naive for hist (like Backtest_Engine.py)
                hist = hist.copy()
                hist.index = hist.index.tz_localize(None)
                
                hist["Price_change"] = hist["Close"].pct_change(fill_method=None).fillna(0)
                data[t] = hist
                print(f"Data loaded for {t} from {data[t].index[0].date()}")
            except Exception as e:
                print(f"Error loading {t}: {e}")
                invalid_tickers.append(t)
        
        # Display invalid ticker warnings in Streamlit UI
        if invalid_tickers:
            # Separate portfolio tickers from benchmark tickers
            portfolio_tickers = set(s['ticker'] for cfg in portfolio_list for s in cfg['stocks'] if s['ticker'])
            benchmark_tickers = set(cfg.get('benchmark_ticker') for cfg in portfolio_list if 'benchmark_ticker' in cfg)
            
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
            st.session_state.alloc_all_results = None
            st.session_state.alloc_all_allocations = None
            st.session_state.alloc_all_metrics = None
            st.stop()
        else:
            # Check if any portfolio has valid tickers
            all_portfolio_tickers = set()
            for cfg in portfolio_list:
                portfolio_tickers = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                all_portfolio_tickers.update(portfolio_tickers)
            
            # Check for non-USD tickers and display currency warning
            check_currency_warning(list(all_portfolio_tickers))
            
            valid_portfolio_tickers = [t for t in all_portfolio_tickers if t in data]
            if not valid_portfolio_tickers:
                st.error(f"❌ **No valid tickers found!** No valid portfolio tickers found. Invalid tickers: {', '.join(all_portfolio_tickers)}. Please check your ticker symbols and try again.")
                progress_bar.empty()
                st.session_state.alloc_all_results = None
                st.session_state.alloc_all_allocations = None
                st.session_state.alloc_all_metrics = None
                st.stop()
            else:
                # Persist raw downloaded price data so later recomputations can access benchmark series
                st.session_state.alloc_raw_data = data
                common_start = max(df.first_valid_index() for df in data.values())
                common_end = min(df.last_valid_index() for df in data.values())
                print()
                all_results = {}
                all_drawdowns = {}
                all_stats = {}
                all_allocations = {}
                all_metrics = {}
                # Map portfolio index (0-based) to the unique key used in the result dicts
                portfolio_key_map = {}
                
                for i, cfg in enumerate(portfolio_list, start=1):
                    progress_text = f"Running backtest for {cfg.get('name', f'Backtest {i}')} ({i}/{len(portfolio_list)})..."
                    progress_bar.progress((len(all_tickers) + i) / (len(all_tickers) + len(portfolio_list)), text=progress_text)
                    name = cfg.get('name', f'Backtest {i}')
                    # Ensure unique key for storage to avoid overwriting when duplicate names exist
                    base_name = name
                    unique_name = base_name
                    suffix = 1
                    while unique_name in all_results or unique_name in all_allocations:
                        unique_name = f"{base_name} ({suffix})"
                        suffix += 1
                    print(f"\nRunning backtest {i}/{len(portfolio_list)}: {name}")
                    # Separate asset tickers from benchmark. Do NOT use benchmark when
                    # computing start/end/simulation dates or available-rebalance logic.
                    asset_tickers = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                    asset_tickers = [t for t in asset_tickers if t in data and t is not None]
                    benchmark_local = cfg.get('benchmark_ticker')
                    benchmark_in_data = benchmark_local if benchmark_local in data else None
                    tickers_for_config = asset_tickers
                    # Build the list of tickers whose data we will reindex (include benchmark if present)
                    data_tickers = list(asset_tickers)
                    if benchmark_in_data:
                        data_tickers.append(benchmark_in_data)
                    if not tickers_for_config:
                        # Check if this is because all tickers are invalid
                        original_asset_tickers = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                        missing_tickers = [t for t in original_asset_tickers if t not in data]
                        if missing_tickers:
                            print(f"  No available asset tickers for {name}; invalid tickers: {missing_tickers}. Skipping.")
                        else:
                            print(f"  No available asset tickers for {name}; skipping.")
                        continue
                    if cfg.get('start_with') == 'all':
                        # Start only when all asset tickers have data
                        final_start = max(data[t].first_valid_index() for t in tickers_for_config)
                    else:
                        # 'oldest' -> start at the earliest asset ticker date so assets can be added over time
                        final_start = min(data[t].first_valid_index() for t in tickers_for_config)
                    if cfg.get('start_date_user'):
                        user_start = pd.to_datetime(cfg['start_date_user'])
                        final_start = max(final_start, user_start)
                    # Preserve previous global alignment only for 'all' mode; do NOT force 'oldest' back to global latest
                    if cfg.get('start_with') == 'all':
                        final_start = max(final_start, common_start)
                    if cfg.get('end_date_user'):
                        final_end = min(pd.to_datetime(cfg['end_date_user']), min(data[t].last_valid_index() for t in tickers_for_config))
                    else:
                        final_end = min(data[t].last_valid_index() for t in tickers_for_config)
                    if final_start > final_end:
                        print(f"  Start date {final_start.date()} is after end date {final_end.date()}. Skipping {name}.")
                        continue
                    
                    simulation_index = pd.date_range(start=final_start, end=final_end, freq='D')
                    print(f"  Simulation period for {name}: {final_start.date()} to {final_end.date()}\n")
                    data_reindexed_for_config = {}
                    invalid_tickers = []
                    for t in data_tickers:
                        if t in data:  # Only process tickers that have data
                            df = data[t].reindex(simulation_index)
                            df["Close"] = df["Close"].ffill()
                            df["Dividends"] = df["Dividends"].fillna(0)
                            df["Price_change"] = df["Close"].pct_change(fill_method=None).fillna(0)
                            data_reindexed_for_config[t] = df
                        else:
                            invalid_tickers.append(t)
                            print(f"Warning: Invalid ticker '{t}' - no data available, skipping reindexing")
                    
                    # Display invalid ticker warnings in Streamlit UI
                    if invalid_tickers:
                        st.warning(f"The following tickers are invalid and will be skipped: {', '.join(invalid_tickers)}")
                    total_series, total_series_no_additions, historical_allocations, historical_metrics = single_backtest(cfg, simulation_index, data_reindexed_for_config)
                    # Store both series under the unique key for later use
                    all_results[unique_name] = {
                        'no_additions': total_series_no_additions,
                        'with_additions': total_series
                    }
                    all_allocations[unique_name] = historical_allocations
                    all_metrics[unique_name] = historical_metrics
                    # Remember mapping from portfolio index (0-based) to unique key
                    portfolio_key_map[i-1] = unique_name
                # --- PATCHED CASH FLOW LOGIC ---
                # Track cash flows as pandas Series indexed by date
                cash_flows = pd.Series(0.0, index=total_series.index)
                # Initial investment: negative cash flow on first date
                if len(total_series.index) > 0:
                    cash_flows.iloc[0] = -cfg.get('initial_value', 0)
                # No periodic additions for allocation tracker
                # Final value: positive cash flow on last date for MWRR
                if len(total_series.index) > 0:
                    cash_flows.iloc[-1] += total_series.iloc[-1]
                # Get benchmark returns for stats calculation
                benchmark_returns = None
                if cfg['benchmark_ticker'] and cfg['benchmark_ticker'] in data_reindexed_for_config:
                    benchmark_returns = data_reindexed_for_config[cfg['benchmark_ticker']]['Price_change']
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
                sharpe = np.nan if stats_returns.std() == 0 else stats_returns.mean() * 365 / (stats_returns.std() * np.sqrt(365))
                sortino = calculate_sortino(stats_returns)
                ulcer = calculate_ulcer_index(stats_values)
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
                        common_idx = pr.index.intersection(br.index)
                        if len(common_idx) >= 2 and br.loc[common_idx].var() != 0:
                            cov = pr.loc[common_idx].cov(br.loc[common_idx])
                            var = br.loc[common_idx].var()
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
                        return "N/A"
                    v = scale_pct(val)
                    # Clamp ranges for each stat type
                    if stat_type in ["CAGR", "Volatility", "MWRR"]:
                        if v < 0 or v > 100:
                            return "N/A"
                    if stat_type == "MaxDrawdown":
                        if v < -100 or v > 0:
                            return "N/A"
                    return f"{v:.2f}%" if stat_type in ["CAGR", "MaxDrawdown", "Volatility", "MWRR"] else f"{v:.3f}" if isinstance(v, float) else v

                stats = {
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
                all_stats[name] = stats
                all_drawdowns[name] = pd.Series(drawdowns, index=stats_dates)
            progress_bar.progress(100, text="Backtests complete!")
            progress_bar.empty()
            print("\n" + "="*80)
            print(" " * 25 + "FINAL PERFORMANCE STATISTICS")
            print("="*80 + "\n")
            stats_df = pd.DataFrame(all_stats).T
            def fmt_pct(x):
                if isinstance(x, (int, float)) and pd.notna(x):
                    return f"{x*100:.2f}%"
                if isinstance(x, str):
                    return x
                return "N/A"
            def fmt_num(x, prec=3):
                if isinstance(x, (int, float)) and pd.notna(x):
                    return f"{x:.3f}"
                if isinstance(x, str):
                    return x
                return "N/A"
            if not stats_df.empty:
                stats_df_display = stats_df.copy()
                stats_df_display.rename(columns={'MaxDrawdown': 'Max Drawdown', 'UlcerIndex': 'Ulcer Index'}, inplace=True)
                stats_df_display['CAGR'] = stats_df_display['CAGR'].apply(lambda x: fmt_pct(x))
                stats_df_display['Max Drawdown'] = stats_df_display['Max Drawdown'].apply(lambda x: fmt_pct(x))
                stats_df_display['Volatility'] = stats_df_display['Volatility'].apply(lambda x: fmt_pct(x))
                # Ensure MWRR is the last column, Beta immediately before it
                if 'Beta' in stats_df_display.columns and 'MWRR' in stats_df_display.columns:
                    cols = list(stats_df_display.columns)
                    # Remove Beta and MWRR
                    beta_col = cols.pop(cols.index('Beta'))
                    mwrr_col = cols.pop(cols.index('MWRR'))
                    # Insert Beta before MWRR at the end
                    cols.append(beta_col)
                    cols.append(mwrr_col)
                    stats_df_display = stats_df_display[cols]
                stats_df_display['MWRR'] = stats_df_display['MWRR'].apply(lambda x: fmt_pct(x))
                stats_df_display['Sharpe'] = stats_df_display['Sharpe'].apply(lambda x: fmt_num(x))
                stats_df_display['Sortino'] = stats_df_display['Sortino'].apply(lambda x: fmt_num(x))
                stats_df_display['Ulcer Index'] = stats_df_display['Ulcer Index'].apply(lambda x: fmt_num(x))
                stats_df_display['UPI'] = stats_df_display['UPI'].apply(lambda x: fmt_num(x))
                if 'Beta' in stats_df_display.columns:
                    stats_df_display['Beta'] = stats_df_display['Beta'].apply(lambda x: fmt_num(x))
                print(stats_df_display.to_string())
            else:
                print("No stats to display.")
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
            
            print(header_format.format("Year", *names))
            print("-" * (6 + 3 + (col_width*2+1 + 3)*len(names)))
            print(row_format.format(" ", *[item for pair in [('% Change', 'Final Value')] * len(names) for item in pair]))
            print("=" * (6 + 3 + (col_width*2+1 + 3)*len(names)))
            
            for y in years:
                row_items = [f"{y}"]
                for nm in names:
                    ser = all_years[nm]
                    ser_year = ser[ser.index.year == y]
                    
                    # Corrected logic for yearly performance calculation
                    start_val_for_year = None
                    if y == min(years):
                        config_for_name = next((c for c in portfolio_list if c['name'] == nm), None)
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
                print(row_format.format(*row_items))
            print("\n" + "="*80)
    
            # console output captured previously is no longer shown on the page
            st.session_state.alloc_all_results = all_results
            st.session_state.alloc_all_drawdowns = all_drawdowns
            if 'stats_df_display' in locals():
                st.session_state.alloc_stats_df_display = stats_df_display
            st.session_state.alloc_all_years = all_years
            st.session_state.alloc_all_allocations = all_allocations
            # Save a snapshot used by the allocations UI so charts/tables remain static until rerun
            try:
                # compute today_weights_map (target weights as-if rebalanced at final snapshot date)
                today_weights_map = {}
                for pname, allocs in all_allocations.items():
                    try:
                        alloc_dates = sorted(list(allocs.keys()))
                        final_d = alloc_dates[-1]
                        metrics_local = all_metrics.get(pname, {})
                        
                        # Check if momentum is used for this portfolio
                        portfolio_cfg = next((cfg for cfg in portfolio_list if cfg.get('name') == pname), None)
                        use_momentum = portfolio_cfg.get('use_momentum', True) if portfolio_cfg else True
                        
                        if final_d in metrics_local:
                            if use_momentum:
                                # extract Calculated_Weight if present (momentum-based)
                                weights = {t: v.get('Calculated_Weight', 0) for t, v in metrics_local[final_d].items()}
                                # normalize (ensure sums to 1 excluding CASH)
                                sumw = sum(w for k, w in weights.items() if k != 'CASH')
                                if sumw > 0:
                                    norm = {k: (w / sumw) if k != 'CASH' else weights.get('CASH', 0) for k, w in weights.items()}
                                else:
                                    norm = weights
                                today_weights_map[pname] = norm
                            else:
                                # When momentum is not used, use target_allocation (user-defined allocations)
                                weights = {t: v.get('target_allocation', 0) for t, v in metrics_local[final_d].items()}
                                # normalize (ensure sums to 1 excluding CASH)
                                sumw = sum(w for k, w in weights.items() if k != 'CASH')
                                if sumw > 0:
                                    norm = {k: (w / sumw) if k != 'CASH' else weights.get('CASH', 0) for k, w in weights.items()}
                                else:
                                    norm = weights
                                today_weights_map[pname] = norm
                        else:
                            # fallback: use allocation snapshot at final date but convert market-value alloc to target weights (exclude CASH then renormalize)
                            final_alloc = allocs.get(final_d, {})
                            noncash = {k: v for k, v in final_alloc.items() if k != 'CASH'}
                            s = sum(noncash.values())
                            if s > 0:
                                norm = {k: (v / s) for k, v in noncash.items()}
                                norm['CASH'] = final_alloc.get('CASH', 0)
                            else:
                                norm = final_alloc
                            today_weights_map[pname] = norm
                    except Exception:
                        today_weights_map[pname] = {}

                st.session_state.alloc_snapshot_data = {
                    'raw_data': data,
                    'portfolio_configs': portfolio_list,
                    'all_allocations': all_allocations,
                    'all_metrics': all_metrics,
                    'today_weights_map': today_weights_map
                }
            except Exception:
                pass
            st.session_state.alloc_all_metrics = all_metrics
            # Save portfolio index -> unique key mapping so UI selectors can reference results reliably
            st.session_state.alloc_portfolio_key_map = portfolio_key_map
            st.session_state.alloc_backtest_run = True

# Sidebar JSON export/import for ALL portfolios
def paste_all_json_callback():
    txt = st.session_state.get('alloc_paste_all_json_text', '')
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
        if isinstance(obj, list):
            # Process each portfolio configuration for Allocations page
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
            
            # Process each portfolio configuration for Allocations page (existing logic)
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
                
                # Debug: Show what we received for this portfolio
                if 'momentum_windows' in cfg:
                    st.info(f"Momentum windows for {cfg.get('name', 'Unknown')}: {cfg['momentum_windows']}")
                if 'use_momentum' in cfg:
                    st.info(f"Use momentum for {cfg.get('name', 'Unknown')}: {cfg['use_momentum']}")
                
                # Map frequency values from app.py format to Allocations format
                def map_frequency(freq):
                    if freq is None:
                        return 'Never'
                    freq_map = {
                        'Never': 'Never',
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
                
                # Allocations page specific: ensure all required fields are present
                # and ignore fields that are specific to other pages
                allocations_config = {
                    'name': cfg.get('name', 'Allocation Portfolio'),
                    'stocks': stocks,
                    'benchmark_ticker': cfg.get('benchmark_ticker', '^GSPC'),
                    'initial_value': cfg.get('initial_value', 10000),
                    'added_amount': cfg.get('added_amount', 0),  # Allocations page typically doesn't use additions
                    'added_frequency': map_frequency(cfg.get('added_frequency', 'Never')),  # Allocations page typically doesn't use additions
                                          'rebalancing_frequency': map_frequency(cfg.get('rebalancing_frequency', 'Monthly')),
                      'start_date_user': cfg.get('start_date_user'),
                      'end_date_user': cfg.get('end_date_user'),
                      'start_with': cfg.get('start_with', 'all'),
                      'use_momentum': cfg.get('use_momentum', True),
                    'momentum_strategy': momentum_strategy,
                    'negative_momentum_strategy': negative_momentum_strategy,
                    'momentum_windows': momentum_windows,
                    'calc_beta': cfg.get('calc_beta', True),
                    'calc_volatility': cfg.get('calc_volatility', True),
                    'beta_window_days': cfg.get('beta_window_days', 365),
                    'exclude_days_beta': cfg.get('exclude_days_beta', 30),
                    'vol_window_days': cfg.get('vol_window_days', 365),
                    'exclude_days_vol': cfg.get('exclude_days_vol', 30),
                }
                processed_configs.append(allocations_config)
            
            st.session_state.alloc_portfolio_configs = processed_configs
            # Reset active selection and derived mappings so the UI reflects the new configs
            if processed_configs:
                st.session_state.alloc_active_portfolio_index = 0
                st.session_state.alloc_portfolio_selector = processed_configs[0].get('name', '')
                # Update portfolio name input field to match the first imported portfolio
                st.session_state.alloc_portfolio_name = processed_configs[0].get('name', 'Allocation Portfolio')
                # Mirror several active_* widget defaults so the UI selectboxes/inputs update
                st.session_state['alloc_active_name'] = processed_configs[0].get('name', '')
                st.session_state['alloc_active_initial'] = int(processed_configs[0].get('initial_value', 0) or 0)
                st.session_state['alloc_active_added_amount'] = int(processed_configs[0].get('added_amount', 0) or 0)
                st.session_state['alloc_active_rebal_freq'] = processed_configs[0].get('rebalancing_frequency', 'month')
                st.session_state['alloc_active_add_freq'] = processed_configs[0].get('added_frequency', 'none')
                st.session_state['alloc_active_benchmark'] = processed_configs[0].get('benchmark_ticker', '')
                st.session_state['alloc_active_use_momentum'] = bool(processed_configs[0].get('use_momentum', True))
            else:
                st.session_state.alloc_active_portfolio_index = None
                st.session_state.alloc_portfolio_selector = ''
            st.session_state.alloc_portfolio_key_map = {}
            st.session_state.alloc_backtest_run = False
            st.success('All portfolio configurations updated from JSON (Allocations page).')
            # Debug: Show final momentum windows for first portfolio
            if processed_configs:
                st.info(f"Final momentum windows for first portfolio: {processed_configs[0]['momentum_windows']}")
                st.info(f"Final use_momentum for first portfolio: {processed_configs[0]['use_momentum']}")
            # Force a rerun so widgets rebuild with the new configs
            try:
                st.experimental_rerun()
            except Exception:
                # In some environments experimental rerun may raise; setting a rerun flag is a fallback
                st.session_state.alloc_rerun_flag = True
        else:
            st.error('JSON must be a list of portfolio configurations.')
    except Exception as e:
        st.error(f'Failed to parse JSON: {e}')




# Simplified display for allocation tracker: only allocation pies and rebalancing metrics are shown
active_name = active_portfolio.get('name')
if st.session_state.get('alloc_backtest_run', False):
    st.subheader("Allocation & Rebalancing Metrics")
    

    allocs_for_portfolio = st.session_state.get('alloc_all_allocations', {}).get(active_name) if st.session_state.get('alloc_all_allocations') else None
    metrics_for_portfolio = st.session_state.get('alloc_all_metrics', {}).get(active_name) if st.session_state.get('alloc_all_metrics') else None

    if not allocs_for_portfolio and not metrics_for_portfolio:
        st.info("No allocation or rebalancing history available. If you have precomputed allocation snapshots, store them in session state keys `alloc_all_allocations` and `alloc_all_metrics` under this portfolio name.")
    else:
        # --- Rebalance as of Today (static snapshot from last Run Backtests) ---
        snapshot = st.session_state.get('alloc_snapshot_data', {})
        today_weights_map = snapshot.get('today_weights_map', {}) if snapshot else {}
        
        # Check if momentum is used for this portfolio
        use_momentum = active_portfolio.get('use_momentum', True) if active_portfolio else True
        
        if snapshot and active_name in today_weights_map:
            today_weights = today_weights_map.get(active_name, {})
            
            # If momentum is not used, use the initial target allocation from portfolio configuration
            if not use_momentum and (not today_weights or all(v == 0 for v in today_weights.values())):
                # Use the target allocation from portfolio configuration (same as initial allocation)
                if active_portfolio and active_portfolio.get('stocks'):
                    today_weights = {}
                    total_allocation = sum(stock.get('allocation', 0) for stock in active_portfolio['stocks'] if stock.get('ticker'))
                    
                    if total_allocation > 0:
                        for stock in active_portfolio['stocks']:
                            ticker = stock.get('ticker', '').strip()
                            allocation = stock.get('allocation', 0)
                            if ticker and allocation > 0:
                                # Convert to proportion (0-1 range)
                                today_weights[ticker] = allocation / total_allocation
                    
                    # If no valid stocks or allocations, leave today_weights empty (will show info message)
            
            labels_today = [k for k, v in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
            vals_today = [float(today_weights[k]) * 100 for k in labels_today]
            
            if labels_today and vals_today:
                st.markdown(f"## Rebalance as of Today ({pd.Timestamp.now().strftime('%Y-%m-%d')})")
                fig_today = go.Figure()
                fig_today.add_trace(go.Pie(labels=labels_today, values=vals_today, hole=0.35))
                fig_today.update_traces(textinfo='percent+label')
                fig_today.update_layout(template='plotly_dark', margin=dict(t=10), height=600)
                st.plotly_chart(fig_today, use_container_width=True, key=f"alloc_today_chart_{active_name}")
            
            # static shares table

            # Define build_table_from_alloc before usage
            def build_table_from_alloc(alloc_dict, price_date, label):
                rows = []
                # Use portfolio_value from session state or active_portfolio (current portfolio value)
                try:
                    portfolio_value = float(st.session_state.get('alloc_active_initial', active_portfolio.get('initial_value', 0) or 0))
                except Exception:
                    portfolio_value = active_portfolio.get('initial_value', 0) or 0
                # Use raw_data from snapshot or session state
                snapshot = st.session_state.get('alloc_snapshot_data', {})
                raw_data = snapshot.get('raw_data') if snapshot and snapshot.get('raw_data') is not None else st.session_state.get('alloc_raw_data', {})
                def _price_on_or_before(df, target_date):
                    try:
                        idx = df.index[df.index <= pd.to_datetime(target_date)]
                        if len(idx) == 0:
                            return None
                        return float(df.loc[idx[-1], 'Close'])
                    except Exception:
                        return None
                for tk in sorted(alloc_dict.keys()):
                    alloc_pct = float(alloc_dict.get(tk, 0))
                    if tk == 'CASH':
                        price = None
                        shares = 0.0
                        total_val = portfolio_value * alloc_pct
                    else:
                        df = raw_data.get(tk)
                        price = None
                        # Ensure df is a valid DataFrame with Close prices before accessing
                        if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                            if price_date is None:
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
                            shares = 0.0
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
                
                # Add comprehensive portfolio data table right after the main allocation table
                build_comprehensive_portfolio_table(alloc_dict, portfolio_value)
            
            # Add comprehensive portfolio data table
            def build_comprehensive_portfolio_table(alloc_dict, portfolio_value):
                """
                Build a comprehensive table with all available financial indicators from Yahoo Finance
                """
                st.markdown("### 📊 Comprehensive Portfolio Data")
                st.markdown("#### Detailed financial indicators for each position")
                
                # Get current date for data freshness
                current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
                
                # Create progress bar for data fetching
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                rows = []
                tickers = [tk for tk in alloc_dict.keys() if tk != 'CASH']
                total_tickers = len(tickers)
                
                for i, ticker in enumerate(tickers):
                    status_text.text(f"Fetching data for {ticker}... ({i+1}/{total_tickers})")
                    progress_bar.progress((i + 1) / total_tickers)
                    
                    try:
                        # Fetch comprehensive data from Yahoo Finance
                        info = get_ticker_info(ticker)
                        
                        # Get current price
                        current_price = info.get('currentPrice', info.get('regularMarketPrice', None))
                        if current_price is None:
                            # Try to get from historical data
                            hist = stock.history(period='1d')
                            if not hist.empty:
                                current_price = hist['Close'].iloc[-1]
                        
                        # Calculate allocation values
                        alloc_pct = float(alloc_dict.get(ticker, 0))
                        allocation_value = portfolio_value * alloc_pct
                        shares = round(allocation_value / current_price, 1) if current_price and current_price > 0 else 0
                        total_val = shares * current_price if current_price else allocation_value
                        
                        # Determine if this is an ETF or stock for intelligent data handling
                        quote_type = info.get('quoteType', '').lower()
                        is_etf = quote_type in ['etf', 'fund']
                        is_commodity = ticker in ['GLD', 'SLV', 'USO', 'UNG'] or 'gold' in info.get('longName', '').lower()
                        
                        # Extract all available financial indicators with intelligent handling
                        row = {
                            'Ticker': ticker,
                            'Company Name': info.get('longName', info.get('shortName', 'N/A')),
                            'Sector': info.get('sector', 'N/A'),
                            'Industry': info.get('industry', 'N/A'),
                            'Current Price ($)': current_price,
                            'Allocation %': alloc_pct * 100,
                            'Shares': shares,
                            'Total Value ($)': total_val,
                            '% of Portfolio': (total_val / portfolio_value * 100) if portfolio_value > 0 else 0,
                            
                            # Valuation Metrics (not applicable for commodities/ETFs)
                            'Market Cap ($B)': info.get('marketCap', 0) / 1e9 if info.get('marketCap') and not is_commodity else None,
                            'Enterprise Value ($B)': info.get('enterpriseValue', 0) / 1e9 if info.get('enterpriseValue') and not is_commodity else None,
                            'P/E Ratio': info.get('trailingPE', None) if not is_commodity else None,
                            'Forward P/E': info.get('forwardPE', None) if not is_commodity else None,
                            'PEG Ratio': None,  # Will be calculated below with fallback strategies
                            'Price/Book': None,  # Will be calculated manually below
                            'Price/Sales': info.get('priceToSalesTrailing12Months', None) if not is_commodity else None,
                            'Price/Cash Flow': info.get('priceToCashflow', None) if not is_commodity else None,
                            'EV/EBITDA': info.get('enterpriseToEbitda', None) if not is_commodity else None,
                            
                            # Financial Health (not applicable for commodities/ETFs)
                            'Debt/Equity': info.get('debtToEquity', None) if not is_commodity else None,
                            'Current Ratio': info.get('currentRatio', None) if not is_commodity else None,
                            'Quick Ratio': info.get('quickRatio', None) if not is_commodity else None,
                            'ROE (%)': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') and not is_commodity else None,
                            'ROA (%)': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') and not is_commodity else None,
                            'ROIC (%)': info.get('returnOnInvestedCapital', 0) * 100 if info.get('returnOnInvestedCapital') and not is_commodity else None,
                            
                            # Growth Metrics (not applicable for commodities/ETFs)
                            'Revenue Growth (%)': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') and not is_commodity else None,
                            'Earnings Growth (%)': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') and not is_commodity else None,
                            'EPS Growth (%)': info.get('earningsQuarterlyGrowth', 0) * 100 if info.get('earningsQuarterlyGrowth') and not is_commodity else None,
                            
                            # Dividend Information (available for ETFs and some stocks)
                            # Yahoo Finance returns dividend yield as decimal (0.0002 for 0.02%)
                            'Dividend Yield (%)': info.get('dividendYield', 0) if info.get('dividendYield') else None,
                            'Dividend Rate ($)': info.get('dividendRate', None),
                            'Payout Ratio (%)': info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') and not is_commodity else None,
                            '5Y Dividend Growth (%)': info.get('fiveYearAvgDividendYield', 0) if info.get('fiveYearAvgDividendYield') else None,
                            
                            # Trading Metrics (available for all securities)
                            '52W High ($)': info.get('fiftyTwoWeekHigh', None),
                            '52W Low ($)': info.get('fiftyTwoWeekLow', None),
                            '50D MA ($)': info.get('fiftyDayAverage', None),
                            '200D MA ($)': info.get('twoHundredDayAverage', None),
                            'Beta': info.get('beta', None),
                            'Volume': info.get('volume', None),
                            'Avg Volume': info.get('averageVolume', None),
                            
                            # Analyst Ratings (not available for commodities/ETFs)
                            'Analyst Rating': info.get('recommendationKey', 'N/A').title() if info.get('recommendationKey') and not is_commodity else 'N/A',
                            'Target Price ($)': info.get('targetMeanPrice', None) if not is_commodity else None,
                            'Target High ($)': info.get('targetHighPrice', None) if not is_commodity else None,
                            'Target Low ($)': info.get('targetLowPrice', None) if not is_commodity else None,
                            
                            # Additional Metrics (not applicable for commodities/ETFs)
                            'Book Value ($)': info.get('bookValue', None) if not is_commodity else None,
                            'Cash per Share ($)': info.get('totalCashPerShare', None) if not is_commodity else None,
                            'Revenue per Share ($)': info.get('revenuePerShare', None) if not is_commodity else None,
                            'Profit Margin (%)': info.get('profitMargins', 0) * 100 if info.get('profitMargins') and not is_commodity else None,
                            'Operating Margin (%)': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') and not is_commodity else None,
                            'Gross Margin (%)': info.get('grossMargins', 0) * 100 if info.get('grossMargins') and not is_commodity else None,
                        }
                        
                        # Simple PEG Ratio calculation: P/E ÷ Earnings Growth
                        pe_ratio = info.get('trailingPE')
                        earnings_growth = info.get('earningsGrowth')
                        peg_ratio = None
                        peg_source = "N/A"
                        
                        if not is_commodity and pe_ratio and pe_ratio > 0 and earnings_growth and earnings_growth > 0:
                            # Standard PEG Ratio calculation: P/E ÷ Earnings Growth Rate
                            # Yahoo Finance returns growth as decimal (0.15 = 15% growth)
                            # We need to convert to percentage for PEG calculation
                            growth_percentage = earnings_growth * 100
                            
                            peg_ratio = pe_ratio / growth_percentage
                            peg_source = "P/E ÷ Earnings Growth"
                        
                        # Price/Book will be calculated AFTER dual-class adjustment
                        
                        # Calculate EV/EBITDA manually to ensure consistency
                        enterprise_value = row.get('Enterprise Value ($B)')
                        ebitda = info.get('ebitda')
                        if enterprise_value and ebitda and ebitda > 0:
                            # Convert Enterprise Value from billions to actual value for calculation
                            ev_actual = enterprise_value * 1e9
                            row['EV/EBITDA'] = ev_actual / ebitda
                        
                        # Update the row with calculated PEG ratio and source
                        row['PEG Ratio'] = peg_ratio
                        row['PEG Source'] = peg_source
                        
                        # Define per-share fields for dual-class share adjustments
                        per_share_fields = ['Book Value ($)', 'Cash per Share ($)', 'Revenue per Share ($)']
                        
                        # Fix for dual-class shares - detect and adjust for share class differences
                        # This handles cases where Yahoo Finance returns same per-share data for different share classes
                        if current_price > 0:
                            # Store the current row data for comparison with other shares of same company
                            company_base = ticker.split('-')[0] if '-' in ticker else ticker
                            if company_base not in st.session_state:
                                st.session_state[company_base] = {}
                            
                            # Store reference data for this share class
                            st.session_state[company_base][ticker] = {
                                'price': current_price,
                                'per_share_data': {field: row[field] for field in per_share_fields if row[field]},
                                'enterprise_value': row.get('Enterprise Value ($B)')
                            }
                            
                            # Check if we have data for another share class of the same company
                            for other_ticker, other_data in st.session_state[company_base].items():
                                if other_ticker != ticker and other_data['price'] > 0:
                                    price_ratio = current_price / other_data['price']
                                    
                                    # If price ratio is significantly different (>10x), adjust per-share metrics
                                    # Use the higher-priced share class as reference to avoid scaling issues
                                    if price_ratio > 10 or price_ratio < 0.1:
                                        # Only adjust if current share class has lower price (use higher price as reference)
                                        if current_price < other_data['price']:
                                            for field in per_share_fields:
                                                if row[field] and other_data['per_share_data'].get(field):
                                                    # Use the price ratio to adjust the per-share values
                                                    row[field] = other_data['per_share_data'][field] * price_ratio
                                            
                                            # Also ensure Enterprise Value is consistent between share classes
                                            # Enterprise Value should be the same for both share classes of the same company
                                            if other_data.get('enterprise_value') is not None:
                                                row['Enterprise Value ($B)'] = other_data['enterprise_value']
                                                # Also recalculate EV/EBITDA with the corrected Enterprise Value
                                                ebitda = info.get('ebitda')
                                                if ebitda and ebitda > 0:
                                                    ev_actual = row['Enterprise Value ($B)'] * 1e9
                                                    row['EV/EBITDA'] = ev_actual / ebitda
                                        break
                        
                        # Calculate Price/Book ratio AFTER dual-class adjustment
                        book_value = row.get('Book Value ($)')
                        if current_price and book_value and book_value > 0:
                            row['Price/Book'] = current_price / book_value
                        elif current_price and book_value == 0:
                            row['Price/Book'] = None  # Explicitly set to None if book value is 0
                        
                        # Use Yahoo Finance data directly - no manual scaling fixes
                        # This ensures we get the most accurate data from Yahoo Finance
                        
                        # No manual calculations needed - use Yahoo Finance data directly
                        rows.append(row)
                        
                    except Exception as e:
                        st.warning(f"Error fetching data for {ticker}: {str(e)}")
                        # Add basic row with available data
                        alloc_pct = float(alloc_dict.get(ticker, 0))
                        allocation_value = portfolio_value * alloc_pct
                        rows.append({
                            'Ticker': ticker,
                            'Company Name': 'Error fetching data',
                            'Current Price ($)': None,
                            'Allocation %': alloc_pct * 100,
                            'Shares': 0,
                            'Total Value ($)': allocation_value,
                            '% of Portfolio': (allocation_value / portfolio_value * 100) if portfolio_value > 0 else 0,
                        })
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                if rows:
                    df_comprehensive = pd.DataFrame(rows)
                    
                    # Store comprehensive dataframe in session state for PDF access
                    st.session_state.df_comprehensive = df_comprehensive
                    

                    
                    # Calculate portfolio-weighted metrics BEFORE formatting
                    def weighted_average(df, column, weight_column='% of Portfolio'):
                        """Calculate weighted average, handling NaN values and filtering invalid values"""
                        # Check if columns exist in DataFrame
                        if column not in df.columns or weight_column not in df.columns:
                            return None
                        
                        # Base mask for valid (non-NaN) values
                        valid_mask = df[column].notna() & df[weight_column].notna()
                        
                        # Special handling for PE ratios and similar valuation metrics
                        if 'P/E' in column or 'PE' in column or 'PEG' in column:
                            # Filter out negative or extremely high PE values that are likely errors
                            # Negative PE means negative earnings (company losing money)
                            # Very high PE (>1000) is usually a data error or near-zero earnings
                            valid_mask = valid_mask & (df[column] > 0) & (df[column] <= 1000)
                        elif column == 'Beta':
                            # Filter out extreme beta values (likely data errors)
                            valid_mask = valid_mask & (df[column] >= -5) & (df[column] <= 5)
                        elif 'Ratio' in column and column != 'PEG Ratio':
                            # For other ratios, filter out negative values (usually data errors)
                            valid_mask = valid_mask & (df[column] >= 0)
                        
                        if valid_mask.sum() == 0:
                            return None
                            
                        valid_df = df[valid_mask]
                        # Since weight_column is already in percentage, we divide by 100 to get decimal weights
                        result = (valid_df[column] * valid_df[weight_column] / 100).sum() / (valid_df[weight_column].sum() / 100)
                        
                        return result
                    
                    # Calculate all portfolio-weighted metrics first
                    portfolio_pe = weighted_average(df_comprehensive, 'P/E Ratio')
                    portfolio_pb = weighted_average(df_comprehensive, 'Price/Book')
                    portfolio_beta = weighted_average(df_comprehensive, 'Beta')
                    portfolio_peg = weighted_average(df_comprehensive, 'PEG Ratio')
                    portfolio_ps = weighted_average(df_comprehensive, 'Price/Sales')
                    portfolio_ev_ebitda = weighted_average(df_comprehensive, 'EV/EBITDA')
                    portfolio_roe = weighted_average(df_comprehensive, 'ROE (%)')
                    portfolio_roa = weighted_average(df_comprehensive, 'ROA (%)')
                    portfolio_roic = weighted_average(df_comprehensive, 'ROIC (%)')
                    portfolio_debt_equity = weighted_average(df_comprehensive, 'Debt/Equity')
                    portfolio_current_ratio = weighted_average(df_comprehensive, 'Current Ratio')
                    portfolio_quick_ratio = weighted_average(df_comprehensive, 'Quick Ratio')
                    portfolio_profit_margin = weighted_average(df_comprehensive, 'Profit Margin (%)')
                    portfolio_operating_margin = weighted_average(df_comprehensive, 'Operating Margin (%)')
                    portfolio_gross_margin = weighted_average(df_comprehensive, 'Gross Margin (%)')
                    portfolio_revenue_growth = weighted_average(df_comprehensive, 'Revenue Growth (%)')
                    portfolio_earnings_growth = weighted_average(df_comprehensive, 'Earnings Growth (%)')
                    portfolio_eps_growth = weighted_average(df_comprehensive, 'EPS Growth (%)')
                    portfolio_dividend_yield = weighted_average(df_comprehensive, 'Dividend Yield (%)')
                    
                    portfolio_payout_ratio = weighted_average(df_comprehensive, 'Payout Ratio (%)')
                    portfolio_dividend_growth = weighted_average(df_comprehensive, '5Y Dividend Growth (%)')
                    portfolio_market_cap = weighted_average(df_comprehensive, 'Market Cap ($B)')
                    portfolio_enterprise_value = weighted_average(df_comprehensive, 'Enterprise Value ($B)')
                    portfolio_forward_pe = weighted_average(df_comprehensive, 'Forward P/E')
                    
                    # Store portfolio metrics in session state for PDF generation
                    st.session_state.portfolio_pe = portfolio_pe
                    st.session_state.portfolio_pb = portfolio_pb
                    st.session_state.portfolio_beta = portfolio_beta
                    st.session_state.portfolio_peg = portfolio_peg
                    st.session_state.portfolio_ps = portfolio_ps
                    st.session_state.portfolio_ev_ebitda = portfolio_ev_ebitda
                    st.session_state.portfolio_roe = portfolio_roe
                    st.session_state.portfolio_roa = portfolio_roa
                    st.session_state.portfolio_roic = portfolio_roic
                    st.session_state.portfolio_debt_equity = portfolio_debt_equity
                    st.session_state.portfolio_current_ratio = portfolio_current_ratio
                    st.session_state.portfolio_quick_ratio = portfolio_quick_ratio
                    st.session_state.portfolio_profit_margin = portfolio_profit_margin
                    st.session_state.portfolio_operating_margin = portfolio_operating_margin
                    st.session_state.portfolio_gross_margin = portfolio_gross_margin
                    st.session_state.portfolio_revenue_growth = portfolio_revenue_growth
                    st.session_state.portfolio_earnings_growth = portfolio_earnings_growth
                    st.session_state.portfolio_eps_growth = portfolio_eps_growth
                    st.session_state.portfolio_dividend_yield = portfolio_dividend_yield
                    st.session_state.portfolio_payout_ratio = portfolio_payout_ratio
                    st.session_state.portfolio_dividend_growth = portfolio_dividend_growth
                    st.session_state.portfolio_market_cap = portfolio_market_cap
                    st.session_state.portfolio_enterprise_value = portfolio_enterprise_value
                    st.session_state.portfolio_forward_pe = portfolio_forward_pe
                    
                    # Calculate sector and industry breakdowns BEFORE formatting
                    sector_data = pd.Series(dtype=float)
                    industry_data = pd.Series(dtype=float)
                    
                    if 'Sector' in df_comprehensive.columns and '% of Portfolio' in df_comprehensive.columns:
                        sector_data = df_comprehensive.groupby('Sector')['% of Portfolio'].sum().sort_values(ascending=False)
                    
                    if 'Industry' in df_comprehensive.columns and '% of Portfolio' in df_comprehensive.columns:
                        industry_data = df_comprehensive.groupby('Industry')['% of Portfolio'].sum().sort_values(ascending=False)
                    
                    # Store sector and industry data in session state for PDF generation
                    st.session_state.sector_data = sector_data
                    st.session_state.industry_data = industry_data
                    
                    # Format the dataframe with safe formatting
                    def safe_format(value, format_str):
                        """Safely format values, handling NaN and None"""
                        if pd.isna(value) or value is None:
                            return 'N/A'
                        try:
                            if format_str.startswith('${:,.2f}'):
                                return f"${value:,.2f}"
                            elif format_str.startswith('{:,.2f}%'):
                                return f"{value:,.2f}%"
                            elif format_str.startswith('{:,.0f}'):
                                return f"{value:,.0f}"
                            elif format_str.startswith('{:,.2f}'):
                                return f"{value:,.2f}"
                            else:
                                return str(value)
                        except (ValueError, TypeError):
                            return 'N/A'
                    
                    # Apply safe formatting to all numeric columns
                    for col in df_comprehensive.columns:
                        if col in ['Ticker', 'Company Name', 'Sector', 'Industry', 'Analyst Rating']:
                            continue
                        elif col in ['Allocation %', 'ROE (%)', 'ROA (%)', 'ROIC (%)', 'Revenue Growth (%)', 
                                   'Earnings Growth (%)', 'EPS Growth (%)', 'Dividend Yield (%)', 
                                   'Payout Ratio (%)', '5Y Dividend Growth (%)', 'Profit Margin (%)', 
                                   'Operating Margin (%)', 'Gross Margin (%)', '% of Portfolio']:
                            df_comprehensive[col] = df_comprehensive[col].apply(lambda x: safe_format(x, '{:,.2f}%'))
                        elif col in ['Current Price ($)', 'Total Value ($)', '52W High ($)', '52W Low ($)', 
                                   '50D MA ($)', '200D MA ($)', 'Target Price ($)', 'Target High ($)', 
                                   'Target Low ($)', 'Book Value ($)', 'Cash per Share ($)', 
                                   'Revenue per Share ($)', 'Dividend Rate ($)']:
                            df_comprehensive[col] = df_comprehensive[col].apply(lambda x: safe_format(x, '${:,.2f}'))
                        elif col in ['Market Cap ($B)', 'Enterprise Value ($B)']:
                            df_comprehensive[col] = df_comprehensive[col].apply(lambda x: safe_format(x, '${:,.2f}B'))
                        elif col in ['Volume', 'Avg Volume']:
                            df_comprehensive[col] = df_comprehensive[col].apply(lambda x: safe_format(x, '{:,.0f}'))
                        else:
                            df_comprehensive[col] = df_comprehensive[col].apply(lambda x: safe_format(x, '{:,.2f}'))
                    
                    # Display the comprehensive table
                    st.markdown(f"**Data as of {current_date}**")
                    
                    # Create tabs for different metric categories
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "💰 Valuation", "🏥 Financial Health", "📊 Growth & Dividends", "📈 Technical"])
                    
                    with tab1:
                        # Overview tab - basic info and key metrics
                        overview_cols = ['Ticker', 'Company Name', 'Sector', 'Industry', 'Current Price ($)', 
                                       'Allocation %', 'Shares', 'Total Value ($)', '% of Portfolio', 
                                       'Market Cap ($B)', 'P/E Ratio', 'PEG Ratio', 'PEG Source', 'Beta', 'Analyst Rating']
                        df_overview = df_comprehensive[overview_cols].copy()
                        st.dataframe(df_overview, use_container_width=True)
                    
                    with tab2:
                        # Valuation tab - all valuation metrics
                        valuation_cols = ['Ticker', 'Current Price ($)', 'Market Cap ($B)', 'Enterprise Value ($B)',
                                        'P/E Ratio', 'Forward P/E', 'PEG Ratio', 'PEG Source', 'Price/Book', 'Price/Sales',
                                        'Price/Cash Flow', 'EV/EBITDA', 'Book Value ($)', 'Cash per Share ($)',
                                        'Revenue per Share ($)', 'Target Price ($)', 'Target High ($)', 'Target Low ($)']
                        df_valuation = df_comprehensive[valuation_cols].copy()
                        st.dataframe(df_valuation, use_container_width=True)
                    
                    with tab3:
                        # Financial Health tab - ratios and margins
                        health_cols = ['Ticker', 'Debt/Equity', 'Current Ratio', 'Quick Ratio', 'ROE (%)', 
                                     'ROA (%)', 'ROIC (%)', 'Profit Margin (%)', 'Operating Margin (%)', 
                                     'Gross Margin (%)']
                        df_health = df_comprehensive[health_cols].copy()
                        st.dataframe(df_health, use_container_width=True)
                    
                    with tab4:
                        # Growth & Dividends tab
                        growth_cols = ['Ticker', 'Revenue Growth (%)', 'Earnings Growth (%)', 'EPS Growth (%)',
                                     'Dividend Yield (%)', 'Dividend Rate ($)', 'Payout Ratio (%)', 
                                     '5Y Dividend Growth (%)']
                        df_growth = df_comprehensive[growth_cols].copy()
                        st.dataframe(df_growth, use_container_width=True)
                    
                    with tab5:
                        # Technical tab - price levels and volume
                        technical_cols = ['Ticker', 'Current Price ($)', '52W High ($)', '52W Low ($)', 
                                        '50D MA ($)', '200D MA ($)', 'Beta', 'Volume', 'Avg Volume']
                        df_technical = df_comprehensive[technical_cols].copy()
                        st.dataframe(df_technical, use_container_width=True)
                    
                    # Add portfolio-weighted summary statistics in collapsible section
                    with st.expander("📊 Portfolio-Weighted Summary Statistics", expanded=True):
                        st.markdown("*Metrics weighted by portfolio allocation - represents the total portfolio characteristics*")
                        
                        # Add data accuracy warning
                        st.warning("⚠️ **Data Accuracy Notice:** Portfolio metrics (PE, Beta, etc.) are calculated from available data and may not accurately represent the portfolio if some ticker data is missing, outdated, or incorrect. These metrics should be used as indicative values for portfolio analysis.")
                        
                        # Create a comprehensive summary table
                        summary_data = []
                        
                        # Valuation metrics
                        if portfolio_pe is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "P/E Ratio", "Value": f"{portfolio_pe:.2f}", "Description": "Price-to-Earnings ratio weighted by portfolio allocation"})
                        if portfolio_forward_pe is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "Forward P/E", "Value": f"{portfolio_forward_pe:.2f}", "Description": "Forward Price-to-Earnings ratio weighted by portfolio allocation"})
                        if portfolio_pb is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "Price/Book", "Value": f"{portfolio_pb:.2f}", "Description": "Price-to-Book ratio weighted by portfolio allocation"})
                        if portfolio_peg is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "PEG Ratio", "Value": f"{portfolio_peg:.2f}", "Description": "P/E to Growth ratio weighted by portfolio allocation (calculated from best available source)"})
                        if portfolio_ps is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "Price/Sales", "Value": f"{portfolio_ps:.2f}", "Description": "Price-to-Sales ratio weighted by portfolio allocation"})
                        if portfolio_ev_ebitda is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "EV/EBITDA", "Value": f"{portfolio_ev_ebitda:.2f}", "Description": "Enterprise Value to EBITDA ratio weighted by portfolio allocation"})
                        
                        # Risk metrics
                        if portfolio_beta is not None:
                            summary_data.append({"Category": "Risk", "Metric": "Beta", "Value": f"{portfolio_beta:.2f}", "Description": "Portfolio volatility relative to market (1.0 = market average)"})
                        
                        # Profitability metrics
                        if portfolio_roe is not None:
                            summary_data.append({"Category": "Profitability", "Metric": "ROE (%)", "Value": f"{portfolio_roe:.2f}%", "Description": "Return on Equity weighted by portfolio allocation"})
                        if portfolio_roa is not None:
                            summary_data.append({"Category": "Profitability", "Metric": "ROA (%)", "Value": f"{portfolio_roa:.2f}%", "Description": "Return on Assets weighted by portfolio allocation"})
                        if portfolio_profit_margin is not None:
                            summary_data.append({"Category": "Profitability", "Metric": "Profit Margin (%)", "Value": f"{portfolio_profit_margin:.2f}%", "Description": "Net profit margin weighted by portfolio allocation"})
                        if portfolio_operating_margin is not None:
                            summary_data.append({"Category": "Profitability", "Metric": "Operating Margin (%)", "Value": f"{portfolio_operating_margin:.2f}%", "Description": "Operating profit margin weighted by portfolio allocation"})
                        if portfolio_gross_margin is not None:
                            summary_data.append({"Category": "Profitability", "Metric": "Gross Margin (%)", "Value": f"{portfolio_gross_margin:.2f}%", "Description": "Gross profit margin weighted by portfolio allocation"})
                        
                        # Growth metrics
                        if portfolio_revenue_growth is not None:
                            summary_data.append({"Category": "Growth", "Metric": "Revenue Growth (%)", "Value": f"{portfolio_revenue_growth:.2f}%", "Description": "Revenue growth rate weighted by portfolio allocation"})
                        if portfolio_earnings_growth is not None:
                            summary_data.append({"Category": "Growth", "Metric": "Earnings Growth (%)", "Value": f"{portfolio_earnings_growth:.2f}%", "Description": "Earnings growth rate weighted by portfolio allocation"})
                        if portfolio_eps_growth is not None:
                            summary_data.append({"Category": "Growth", "Metric": "EPS Growth (%)", "Value": f"{portfolio_eps_growth:.2f}%", "Description": "Earnings per share growth rate weighted by portfolio allocation"})
                        
                        # Dividend metrics
                        if portfolio_dividend_yield is not None:
                            summary_data.append({"Category": "Dividends", "Metric": "Dividend Yield (%)", "Value": f"{portfolio_dividend_yield:.2f}%", "Description": "Dividend yield weighted by portfolio allocation"})
                        if portfolio_payout_ratio is not None:
                            summary_data.append({"Category": "Dividends", "Metric": "Payout Ratio (%)", "Value": f"{portfolio_payout_ratio:.2f}%", "Description": "Dividend payout ratio weighted by portfolio allocation"})
                        
                        # Size metrics
                        if portfolio_market_cap is not None:
                            summary_data.append({"Category": "Size", "Metric": "Market Cap ($B)", "Value": f"${portfolio_market_cap:.2f}B", "Description": "Market capitalization weighted by portfolio allocation"})
                        if portfolio_enterprise_value is not None:
                            summary_data.append({"Category": "Size", "Metric": "Enterprise Value ($B)", "Value": f"${portfolio_enterprise_value:.2f}B", "Description": "Enterprise value weighted by portfolio allocation"})
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)
                            
                            # Add interpretation
                            st.markdown("**📈 Portfolio Interpretation:**")
                            if portfolio_beta is not None:
                                if portfolio_beta < 0.8:
                                    st.success(f"**Low Risk Portfolio** - Beta {portfolio_beta:.2f} indicates lower volatility than market")
                                elif portfolio_beta < 1.2:
                                    st.info(f"**Moderate Risk Portfolio** - Beta {portfolio_beta:.2f} indicates market-average volatility")
                                else:
                                    st.warning(f"**High Risk Portfolio** - Beta {portfolio_beta:.2f} indicates higher volatility than market")
                            
                            if portfolio_pe is not None:
                                if portfolio_pe < 15:
                                    st.success(f"**Undervalued Portfolio** - P/E {portfolio_pe:.2f} suggests attractive valuations")
                                elif portfolio_pe < 25:
                                    st.info(f"**Fairly Valued Portfolio** - P/E {portfolio_pe:.2f} suggests reasonable valuations")
                                else:
                                    st.warning(f"**Potentially Overvalued Portfolio** - P/E {portfolio_pe:.2f} suggests high valuations")
                        else:
                            st.warning("No portfolio-weighted metrics available for display.")
                    
                    # Add portfolio composition analysis
                    st.markdown("### 🏢 Portfolio Composition Analysis")
                    
                    # Create a nice table for sector and industry breakdown
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sector breakdown with table and pie chart
                        if not sector_data.empty:
                            st.markdown("**📊 Sector Allocation**")
                            
                            # Create sector table
                            sector_df = pd.DataFrame({
                                'Sector': sector_data.index,
                                'Allocation (%)': sector_data.values
                            }).round(2)
                            
                            # Display table
                            st.dataframe(sector_df, use_container_width=True, hide_index=True)
                            
                            # Create pie chart for sectors
                            if len(sector_data) > 0:
                                fig_sector = px.pie(
                                    values=sector_data.values,
                                    names=sector_data.index,
                                    title="Sector Distribution",
                                    color_discrete_sequence=px.colors.qualitative.Set3
                                )
                                fig_sector.update_traces(textposition='inside', textinfo='percent+label')
                                fig_sector.update_layout(height=400, showlegend=True)
                                st.plotly_chart(fig_sector, use_container_width=True)
                    
                    with col2:
                        # Industry breakdown with table and pie chart
                        if not industry_data.empty:
                            st.markdown("**🏭 Industry Allocation**")
                            
                            # Create industry table
                            industry_df = pd.DataFrame({
                                'Industry': industry_data.index,
                                'Allocation (%)': industry_data.values
                            }).round(2)
                            
                            # Display table
                            st.dataframe(industry_df, use_container_width=True, hide_index=True)
                            
                            # Create pie chart for industries
                            if len(industry_data) > 0:
                                fig_industry = px.pie(
                                    values=industry_data.values,
                                    names=industry_data.index,
                                    title="Industry Distribution",
                                    color_discrete_sequence=px.colors.qualitative.Pastel
                                )
                                fig_industry.update_traces(textposition='inside', textinfo='percent+label')
                                fig_industry.update_layout(height=400, showlegend=True)
                                st.plotly_chart(fig_industry, use_container_width=True)
                    
                    # Portfolio risk metrics
                    st.markdown("### ⚠️ Portfolio Risk Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # Beta analysis
                        if portfolio_beta is not None and not pd.isna(portfolio_beta):
                            if portfolio_beta < 0.8:
                                beta_risk = "Low Risk"
                                beta_color = "green"
                            elif portfolio_beta < 1.2:
                                beta_risk = "Balanced Risk"
                                beta_color = "green"
                            elif portfolio_beta < 1.5:
                                beta_risk = "Moderate Risk"
                                beta_color = "orange"
                            else:
                                beta_risk = "High Risk"
                                beta_color = "red"
                            st.metric("Portfolio Risk Level", beta_risk)
                            st.markdown(f"<span style='color: {beta_color}'>Beta: {portfolio_beta:.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.metric("Portfolio Risk Level", "NA")
                            st.markdown("<span style='color: gray'>Beta: NA</span>", unsafe_allow_html=True)
                    
                    with col2:
                        # Current P/E analysis
                        if portfolio_pe is not None and not pd.isna(portfolio_pe):
                            if portfolio_pe < 15:
                                pe_rating = "Undervalued"
                                pe_color = "green"
                            elif portfolio_pe < 25:
                                pe_rating = "Fair Value"
                                pe_color = "lime"
                            elif portfolio_pe < 35:
                                pe_rating = "Expensive"
                                pe_color = "orange"
                            else:
                                pe_rating = "Overvalued"
                                pe_color = "red"
                            st.metric("Current P/E Rating", pe_rating)
                            st.markdown(f"<span style='color: {pe_color}'>P/E: {portfolio_pe:.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.metric("Current P/E Rating", "NA")
                            st.markdown("<span style='color: gray'>P/E: NA</span>", unsafe_allow_html=True)
                    
                    with col3:
                        # Forward P/E analysis
                        if portfolio_forward_pe is not None and not pd.isna(portfolio_forward_pe):
                            if portfolio_forward_pe < 15:
                                fpe_rating = "Undervalued"
                                fpe_color = "green"
                            elif portfolio_forward_pe < 25:
                                fpe_rating = "Fair Value"
                                fpe_color = "lime"
                            elif portfolio_forward_pe < 35:
                                fpe_rating = "Expensive"
                                fpe_color = "orange"
                            else:
                                fpe_rating = "Overvalued"
                                fpe_color = "red"
                            st.metric("Forward P/E Rating", fpe_rating)
                            st.markdown(f"<span style='color: {fpe_color}'>Forward P/E: {portfolio_forward_pe:.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.metric("Forward P/E Rating", "NA")
                            st.markdown("<span style='color: gray'>Forward P/E: NA</span>", unsafe_allow_html=True)
                    
                    with col4:
                        # Dividend analysis
                        if portfolio_dividend_yield is not None and not pd.isna(portfolio_dividend_yield):
                            if portfolio_dividend_yield > 5:
                                div_rating = "Very High Yield"
                            elif portfolio_dividend_yield > 3:
                                div_rating = "Good Yield"
                            elif portfolio_dividend_yield > 1.5:
                                div_rating = "Moderate Yield"
                            else:
                                div_rating = "Low Yield"
                            st.metric("Dividend Rating", div_rating)
                            st.write(f"Yield: {portfolio_dividend_yield:.2f}%")
                        else:
                            st.metric("Dividend Rating", "NA")
                            st.write("Yield: NA")
                
                else:
                    st.warning("No portfolio data available to display.")
                
                # Add professional data explanation
                with st.expander("🔍 Data Sources & Methodology", expanded=False):
                    st.markdown("### Financial Data Information")
                    
                    st.markdown("**📊 Data Source**: All financial metrics are sourced directly from Yahoo Finance API")
                    st.markdown("**📈 Portfolio Metrics**: Weighted averages calculated based on portfolio allocation percentages")
                    st.markdown("**📊 Data Availability**: Some metrics may show N/A for securities where data is unavailable")
                    
                    st.markdown("**📊 Valuation Guidelines:**")
                    st.markdown("- **P/E Ratio**: <15 = undervalued, 15-25 = fair, >25 = potentially overvalued")
                    st.markdown("- **PEG Ratio**: <1 = undervalued, 1-2 = fair, >2 = potentially overvalued")
                    st.markdown("- **Price/Book**: <1 = potentially undervalued, 1-3 = fair, >3 = potentially overvalued")
                    st.markdown("- **Dividend Yield**: Low yield is not necessarily bad (growth stocks often have low yields)")
                    
                    st.markdown("**🔍 PEG Ratio Calculation:**")
                    st.markdown("- **Formula**: P/E Ratio ÷ Earnings Growth Rate")
                    st.markdown("- **What it measures**: Price relative to earnings growth (lower = better value)")
                    st.markdown("- **Source**: Direct calculation using Yahoo Finance data")
                    st.markdown("- **Realistic ranges**: <1.0 (undervalued), 1.0-1.5 (fair), >2.0 (overvalued)")
            


        # Add timer for next rebalance date
        if allocs_for_portfolio and active_portfolio:
            try:
                # Get the last rebalance date from allocation history
                alloc_dates = sorted(list(allocs_for_portfolio.keys()))
                if len(alloc_dates) > 1:
                    last_rebal_date = alloc_dates[-2]  # Second to last date (excluding today/yesterday)
                else:
                    last_rebal_date = alloc_dates[-1] if alloc_dates else None
                
                # Get rebalancing frequency from active portfolio
                rebalancing_frequency = active_portfolio.get('rebalancing_frequency', 'none')
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
                
                if last_rebal_date and rebalancing_frequency != 'none':
                    # Ensure last_rebal_date is a naive datetime object
                    import pandas as pd
                    if isinstance(last_rebal_date, str):
                        last_rebal_date = pd.to_datetime(last_rebal_date)
                    if hasattr(last_rebal_date, 'tzinfo') and last_rebal_date.tzinfo is not None:
                        last_rebal_date = last_rebal_date.replace(tzinfo=None)
                    
                    next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                        rebalancing_frequency, last_rebal_date
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
                            if hasattr(last_rebal_date, 'to_pydatetime'):
                                last_rebal_datetime = last_rebal_date.to_pydatetime()
                            else:
                                last_rebal_datetime = last_rebal_date
                            
                            total_period = (next_rebalance_datetime - last_rebal_datetime).total_seconds()
                            elapsed_period = (datetime.now() - last_rebal_datetime).total_seconds()
                            progress = min(max(elapsed_period / total_period, 0), 1)
                            
                            st.progress(progress, text=f"Progress to next rebalance: {progress:.1%}")
            except Exception as e:
                pass  # Silently ignore timer calculation errors
        
        build_table_from_alloc({**today_weights, 'CASH': today_weights.get('CASH', 0)}, None, f"Shares if Rebalanced Today (snapshot)")

    if allocs_for_portfolio:
        st.markdown("**Historical Allocations**")
        # Ensure proper DataFrame structure with explicit column names
        allocations_df_raw = pd.DataFrame(allocs_for_portfolio).T
        
        # Handle case where only CASH exists - ensure column name is preserved
        if allocations_df_raw.empty or (len(allocations_df_raw.columns) == 1 and allocations_df_raw.columns[0] is None):
            # Reconstruct DataFrame with proper column names
            processed_data = {}
            for date, alloc_dict in allocs_for_portfolio.items():
                processed_data[date] = {}
                for ticker, value in alloc_dict.items():
                    if ticker is None:
                        processed_data[date]['CASH'] = value
                    else:
                        processed_data[date][ticker] = value
            allocations_df_raw = pd.DataFrame(processed_data).T
        
        allocations_df_raw.index.name = "Date"

        def highlight_rows_by_index(s):
            is_even_row = allocations_df_raw.index.get_loc(s.name) % 2 == 0
            bg_color = 'background-color: #0e1117' if is_even_row else 'background-color: #262626'
            return [f'{bg_color}; color: white;'] * len(s)

        styler = allocations_df_raw.mul(100).style.apply(highlight_rows_by_index, axis=1)
        styler.format('{:,.0f}%', na_rep='N/A')
        
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
        
        st.dataframe(styler, use_container_width=True)

    if metrics_for_portfolio:
        st.markdown("---")
        st.markdown("**Rebalancing Metrics & Calculated Weights**")
        metrics_records = []
        for date, tickers_data in metrics_for_portfolio.items():
            for ticker, data in tickers_data.items():
                # Handle None ticker as CASH
                display_ticker = 'CASH' if ticker is None else ticker
                filtered_data = {k: v for k, v in (data or {}).items() if k != 'Composite'}
                
                # Check if momentum is used for this portfolio
                use_momentum = active_portfolio.get('use_momentum', True) if active_portfolio else True
                
                # If momentum is not used, replace Calculated_Weight with target_allocation
                if not use_momentum:
                    if 'target_allocation' in filtered_data:
                        filtered_data['Calculated_Weight'] = filtered_data['target_allocation']
                    else:
                        # If target_allocation is not available, use the entered allocations from active_portfolio
                        ticker_name = display_ticker if display_ticker != 'CASH' else None
                        if ticker_name:
                            # Find the stock in active_portfolio and use its allocation
                            for stock in active_portfolio.get('stocks', []):
                                if stock.get('ticker', '').strip() == ticker_name:
                                    filtered_data['Calculated_Weight'] = stock.get('allocation', 0)
                                    break
                        else:
                            # For CASH, calculate the remaining allocation
                            total_alloc = sum(stock.get('allocation', 0) for stock in active_portfolio.get('stocks', []))
                            filtered_data['Calculated_Weight'] = max(0, 1.0 - total_alloc)
                
                record = {'Date': date, 'Ticker': display_ticker, **filtered_data}
                metrics_records.append(record)
            
            # Ensure CASH line is added if there's non-zero cash in allocations
            if allocs_for_portfolio and date in allocs_for_portfolio:
                cash_alloc = allocs_for_portfolio[date].get('CASH', 0)
                if cash_alloc > 0:
                    # Check if CASH is already in metrics_records for this date
                    cash_exists = any(record['Date'] == date and record['Ticker'] == 'CASH' for record in metrics_records)
                    if not cash_exists:
                        # Add CASH line to metrics
                        # Check if momentum is used to determine which weight to show
                        use_momentum = active_portfolio.get('use_momentum', True) if active_portfolio else True
                        if not use_momentum:
                            # When momentum is not used, calculate CASH allocation from entered allocations
                            total_alloc = sum(stock.get('allocation', 0) for stock in active_portfolio.get('stocks', []))
                            cash_weight = max(0, 1.0 - total_alloc)
                            cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_weight}
                        else:
                            cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_alloc}
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
                metrics_df.set_index(['Date', 'Ticker'], inplace=True)
                metrics_df_display = metrics_df.copy()
            if 'Momentum' in metrics_df_display.columns:
                metrics_df_display['Momentum'] = metrics_df_display['Momentum'].fillna(0) * 100
            if 'Calculated_Weight' in metrics_df_display.columns:
                metrics_df_display['Calculated_Weight'] = metrics_df_display['Calculated_Weight'].fillna(0) * 100
            if 'Volatility' in metrics_df_display.columns:
                metrics_df_display['Volatility'] = metrics_df_display['Volatility'].fillna(np.nan) * 100

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
                if s.name[1] == 'CASH':
                    return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                unique_dates = list(metrics_df_display.index.get_level_values('Date').unique())
                is_even = unique_dates.index(s.name[0]) % 2 == 0
                bg_color = 'background-color: #0e1117' if is_even else 'background-color: #262626'
                return [f'{bg_color}; color: white;'] * len(s)

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

    # Allocation pie charts (last rebalance vs current)
    if allocs_for_portfolio:
        try:
            alloc_dates = sorted(list(allocs_for_portfolio.keys()))
            final_date = alloc_dates[-1]
            last_rebal_date = alloc_dates[-2] if len(alloc_dates) > 1 else alloc_dates[-1]
            final_alloc = allocs_for_portfolio.get(final_date, {})
            rebal_alloc = allocs_for_portfolio.get(last_rebal_date, {})

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
            # prepare helpers used by the 'Rebalance Today' UI
            # Prefer the snapshot saved when backtests were run so this UI is static until rerun
            snapshot = st.session_state.get('alloc_snapshot_data', {})
            snapshot_raw = snapshot.get('raw_data')
            snapshot_portfolios = snapshot.get('portfolio_configs')

            # select raw_data and portfolio config from snapshot if available, otherwise fall back to live state
            raw_data = snapshot_raw if snapshot_raw is not None else st.session_state.get('alloc_raw_data', {})
            # find snapshot portfolio config by name if present
            snapshot_cfg = None
            if snapshot_portfolios:
                try:
                    snapshot_cfg = next((c for c in snapshot_portfolios if c.get('name') == active_name), None)
                except Exception:
                    snapshot_cfg = None
            portfolio_cfg_for_today = snapshot_cfg if snapshot_cfg is not None else active_portfolio

            try:
                portfolio_value = float(portfolio_cfg_for_today.get('initial_value', 0) or 0)
            except Exception:
                portfolio_value = portfolio_cfg_for_today.get('initial_value', 0) or 0
            


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
            
            # Render small pies for Last Rebalance and Current Allocation
            try:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Target Allocation at Last Rebalance ({last_rebal_date.date()})**")
                    fig_rebal_small = go.Figure()
                    fig_rebal_small.add_trace(go.Pie(labels=labels_rebal, values=vals_rebal, hole=0.35))
                    fig_rebal_small.update_traces(textinfo='percent+label')
                    fig_rebal_small.update_layout(template='plotly_dark', margin=dict(t=10))
                    st.plotly_chart(fig_rebal_small, use_container_width=True, key=f"alloc_rebal_small_{active_name}")
                with col2:
                    st.markdown(f"**Portfolio Evolution (Current Allocation)**")
                    fig_today_small = go.Figure()
                    fig_today_small.add_trace(go.Pie(labels=labels_final, values=vals_final, hole=0.35))
                    fig_today_small.update_traces(textinfo='percent+label')
                    fig_today_small.update_layout(template='plotly_dark', margin=dict(t=10))
                    st.plotly_chart(fig_today_small, use_container_width=True, key=f"alloc_today_small_{active_name}")
            except Exception:
                # If plotting fails, continue and still render the tables below
                pass

            # Last rebalance table (use last_rebal_date)
            build_table_from_alloc(rebal_alloc, last_rebal_date, f"Target Allocation at Last Rebalance ({last_rebal_date.date()})")
            # Current / Today table (use final_date's latest available prices as of now)
            build_table_from_alloc(final_alloc, None, f"Portfolio Evolution (Current Allocation)")
        except Exception as e:
            print(f"[ALLOC PLOT DEBUG] Failed to render allocation plots for {active_name}: {e}")

    # Add PDF generation button at the very end
    st.markdown("---")
    st.markdown("### 📄 Generate PDF Report")
    
    # Optional custom PDF report name
    custom_report_name = st.text_input(
        "📝 Custom Report Name (optional):", 
        value="",
        placeholder="e.g., Portfolio Allocation Analysis, Asset Distribution Q4, Sector Breakdown Study",
        help="Leave empty to use automatic naming: 'Allocations_Report_[timestamp].pdf'",
        key="allocations_custom_report_name"
    )
    
    if st.button("Generate PDF Report", type="primary", use_container_width=True, key="alloc_pdf_btn_2"):
        try:
            success = generate_allocations_pdf(custom_report_name)
            if success:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Generate filename based on custom name or default
                if custom_report_name.strip():
                    clean_name = custom_report_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
                    filename = f"{clean_name}_{timestamp}.pdf"
                else:
                    filename = f"Allocations_Report_{timestamp}.pdf"
                
                st.success("✅ PDF Report Generated Successfully!")
                st.download_button(
                    label="📥 Download PDF Report",
                    data=st.session_state.get('pdf_buffer', b''),
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("❌ Failed to generate PDF report")
        except Exception as e:
            st.error(f"❌ Error generating PDF: {str(e)}")
            st.exception(e)
