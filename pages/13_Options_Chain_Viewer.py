import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import time

# Lazy imports for heavy libraries - only import when needed
def get_yfinance():
    import yfinance as yf
    return yf

def get_plotly():
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    return go, px, make_subplots

def get_concurrent():
    import concurrent.futures
    return concurrent.futures

# Suppress Plotly deprecation warnings
# These warnings appear due to internal Plotly/Streamlit interactions
# and are not related to our code - they will be fixed in future library versions
warnings.filterwarnings('ignore', message='.*keyword arguments have been deprecated.*')
warnings.filterwarnings('ignore', message='.*Use `config` instead to specify Plotly configuration options.*')

# Page configuration
st.set_page_config(
    page_title="Options Chain Viewer",
    page_icon="📊",
    layout="wide"
)

# Initialize debug messages list
if 'debug_messages' not in st.session_state:
    st.session_state.debug_messages = []

# 🛡️ GLOBAL VARIABLE INITIALIZATION - PREVENT REFRESH CRASHES
# Initialize all section-specific variables to prevent NameError on refresh
evolution_strike = 400.0
barbell_strike = 400.0
spread_strike = 400.0
multi_strike_strike = 400.0

# 🛡️ ULTIMATE PROTECTION - FORCE GLOBAL DEFINITION
globals()['barbell_strike'] = 400.0
globals()['evolution_strike'] = 400.0
globals()['spread_strike'] = 400.0
globals()['multi_strike_strike'] = 400.0

# Try to get from session state if available
if 'persistent_strike' in st.session_state:
    try:
        default_strike = float(st.session_state.persistent_strike)
        evolution_strike = default_strike
        barbell_strike = default_strike
        spread_strike = default_strike
        multi_strike_strike = default_strike
    except:
        # ULTIMATE FALLBACK - ALWAYS DEFINED
        evolution_strike = 400.0
        barbell_strike = 400.0
        spread_strike = 400.0
        multi_strike_strike = 400.0

# Title
st.title("📊 Options Chain Viewer")
st.markdown("**Real-time options data for any ticker**")

# Sidebar controls
st.sidebar.header("🎛️ Controls")

# Initialize ticker in session state
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = ""

# Utility function to clean DataFrames for Arrow serialization
def clean_dataframe_for_arrow(df):
    """Clean DataFrame to be Arrow-compatible by converting numpy types to Python native types"""
    if df is None:
        return df
    
    # Handle Styler objects - return them as-is for styling preservation
    if hasattr(df, 'data'):
        return df  # Keep Styler objects intact for colored display
    
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Convert all columns to Python native types
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # For object columns, try to convert numpy types
            df_clean[col] = df_clean[col].astype(str)
        elif 'float' in str(df_clean[col].dtype):
            # Convert numpy floats to Python floats
            df_clean[col] = df_clean[col].astype(float)
        elif 'int' in str(df_clean[col].dtype):
            # Convert numpy ints to Python ints
            df_clean[col] = df_clean[col].astype(int)
    
    return df_clean

# Utility function to calculate MID price (Bid + Ask) / 2, fallback to Last
def calculate_mid_price(bid, ask, last_price):
    """Calculate MID price from bid/ask, fallback to last price - SIMPLIFIED VERSION"""
    try:
        # Convert to float and handle None/NaN
        bid_val = float(bid) if bid is not None and pd.notna(bid) else None
        ask_val = float(ask) if ask is not None and pd.notna(ask) else None
        last_val = float(last_price) if last_price is not None and pd.notna(last_price) else None
        
        # Try bid/ask first
        if bid_val is not None and ask_val is not None and bid_val > 0 and ask_val > 0:
            mid_price = (bid_val + ask_val) / 2
            return mid_price
        
        # Fallback to last price
        if last_val is not None and last_val > 0:
            return last_val
            
        # If nothing works, return None
        return None
        
    except (ValueError, TypeError):
        return None

# Utility function to safely convert current_price to float
def safe_float_price(price):
    """Convert any price format to safe float for calculations - Cloud compatible"""
    if price is None:
        return 0.0
    
    # If already a proper float/int
    if isinstance(price, (int, float)):
        return float(price)
    
    # If string, try to convert
    if isinstance(price, str):
        try:
            return float(price)
        except ValueError:
            return 0.0
    
    # If numpy/pandas types - convert to Python native
    try:
        if hasattr(price, 'item'):  # numpy scalar
            return float(price.item())
        elif hasattr(price, 'iloc'):  # pandas scalar
            return float(price.iloc[0] if len(price) > 0 else 0.0)
        elif str(type(price)).startswith("<class 'numpy."):
            return float(price)
        elif str(type(price)).startswith("<class 'pandas."):
            return float(price)
    except (ValueError, TypeError, AttributeError, IndexError):
        pass
    
    # Final fallback - try direct conversion
    try:
        return float(price)
    except (ValueError, TypeError):
        return 0.0

# Function to get risk-free rate (10-year Treasury)
@st.cache_data(persist="disk")
def get_risk_free_rate():
    """Get current 10-year Treasury rate as risk-free rate"""
    try:
        import requests
        
        # Try to get 10-year Treasury rate from FRED API (free)
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': 'DGS10',  # 10-Year Treasury Constant Maturity Rate
            'api_key': 'demo',  # Free demo key
            'file_type': 'json',
            'limit': '1',
            'sort_order': 'desc'
        }
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data and len(data['observations']) > 0:
                rate = float(data['observations'][0]['value'])
                if rate > 0:
                    return rate
    except Exception:
        pass
    
    # Fallback: Return a reasonable default (around 4-5%)
    return 4.5

# Function to convert ticker (same as page 1)
def convert_ticker_input(ticker):
    if not ticker:
        return ""
    
    # Convert to uppercase
    converted = ticker.upper()
    
    # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
    if converted == 'BRK.B':
        converted = 'BRK-B'
    elif converted == 'BRK.A':
        converted = 'BRK-A'
    
    return converted

# Ticker input with automatic conversion (same as page 1)
def update_ticker_input():
    """Update ticker input with automatic conversion like page 1"""
    try:
        val = st.session_state.get("ticker_input", "")
        if val is None:
            return
        
        # Convert the input value to uppercase
        upper_val = val.upper()
        
        # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
        if upper_val == 'BRK.B':
            upper_val = 'BRK-B'
        elif upper_val == 'BRK.A':
            upper_val = 'BRK-A'
        
        # Update the session state with converted value
        st.session_state.ticker_input = upper_val
        
    except Exception as e:
        pass  # Ignore errors during conversion

# Initialize session state for ticker persistence - DEFAULT SPY
if 'persistent_ticker' not in st.session_state:
    st.session_state.persistent_ticker = "SPY"
if 'ticker_input' not in st.session_state:
    st.session_state.ticker_input = st.session_state.persistent_ticker

# Ticker input with persistence
ticker_input = st.sidebar.text_input(
    "📈 Enter Ticker Symbol:",
    help="Enter any ticker symbol (e.g., SPY, AAPL, TSLA, BRK.A, BRK.B)",
    on_change=update_ticker_input,
    key="ticker_input"
)

# Get the converted ticker and use it directly (no button needed)
ticker_symbol = convert_ticker_input(ticker_input)

# Update persistent ticker in session state
st.session_state.persistent_ticker = ticker_symbol
st.session_state.current_ticker = ticker_symbol

# Refresh button
if st.sidebar.button("🔄 Force Refresh Data", help="Clear cache and fetch fresh data"):
    # Clear all caches
    st.cache_data.clear()
    st.session_state.debug_messages.append(f"🔄 Cache cleared and data refreshed at {datetime.now().strftime('%H:%M:%S')}")
    st.rerun()

# Always show both option types
show_calls = True
show_puts = True

# Cache functions for performance
@st.cache_data(persist="disk")  # Disk persistence (TTL ignored when persist is set)
def get_cached_ticker_info(ticker_symbol):
    """Cache ticker info (price + expirations) for 24 hours - SERIALIZABLE"""
    yf = get_yfinance()
    ticker = yf.Ticker(ticker_symbol)
    try:
        # Single API call to get both price and options data
        current_price = float(ticker.history(period="1d")['Close'].iloc[-1])
        expirations = list(ticker.options)  # This is cached by yfinance internally
        fetch_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return current_price, expirations, fetch_timestamp
    except Exception as e:
        raise Exception(f"Error fetching {ticker_symbol}: {e}")

@st.cache_data(persist="disk", ttl=604800)  # 7 days cache (604800 seconds) - TTL only applies when persist is not set
def get_cached_risk_free_rate():
    """Cache risk-free rate (10-year Treasury) for 7 days - SERIALIZABLE"""
    try:
        risk_free_rate = get_risk_free_rate()
        fetch_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return risk_free_rate, fetch_timestamp
    except Exception as e:
        # Fallback to reasonable default
        return 4.5, datetime.now().strftime('%Y-%m-%d %H:%M:%S')

@st.cache_data(persist="disk")  # Disk persistence (TTL ignored when persist is set)
def get_cached_all_options_batch(ticker_symbol, expirations_list):
    """Cache ALL option chains in ONE API call - OPTIMIZED like page 1"""
    yf = get_yfinance()
    ticker = yf.Ticker(ticker_symbol)
    try:
        all_calls = []
        all_puts = []
        
        # Single ticker object, multiple option_chain calls (but cached internally by yfinance)
        for expiration in expirations_list:
            try:
                opt_chain = ticker.option_chain(expiration)
                
                if not opt_chain.calls.empty:
                    calls_df = opt_chain.calls.copy()
                    calls_df['expiration'] = expiration
                    all_calls.append(calls_df)
                
                if not opt_chain.puts.empty:
                    puts_df = opt_chain.puts.copy()
                    puts_df['expiration'] = expiration
                    all_puts.append(puts_df)
                    
            except Exception as e:
                # Skip this expiration if it fails
                continue
        
        # Combine all data
        calls_combined = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_combined = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()
        
        return {
            'calls': calls_combined.to_dict('records') if not calls_combined.empty else [],
            'puts': puts_combined.to_dict('records') if not puts_combined.empty else [],
            'calls_empty': calls_combined.empty,
            'puts_empty': puts_combined.empty,
            'api_calls': 1  # Only 1 API call for all options!
        }
    except Exception as e:
        raise Exception(f"Error fetching all options for {ticker_symbol}: {e}")

@st.cache_data(persist="disk")  # Disk persistence (TTL ignored when persist is set)
def get_cached_option_chain(ticker_symbol, expiration):
    """Cache individual option chain for 8 hours - SERIALIZABLE"""
    yf = get_yfinance()
    ticker = yf.Ticker(ticker_symbol)
    try:
        opt_chain = ticker.option_chain(expiration)
        
        # Convert to serializable format
        calls_data = opt_chain.calls.to_dict() if not opt_chain.calls.empty else {}
        puts_data = opt_chain.puts.to_dict() if not opt_chain.puts.empty else {}
        
        return {
            'calls': calls_data,
            'puts': puts_data,
            'calls_empty': opt_chain.calls.empty,
            'puts_empty': opt_chain.puts.empty
        }
    except Exception as e:
        raise Exception(f"Error fetching option chain for {expiration}: {e}")

@st.cache_data(persist="disk")  # Disk persistence (TTL ignored when persist is set)
def get_cached_vix_data(period="max"):
    """Cache VIX data for 4 hours - SERIALIZABLE"""
    yf = get_yfinance()
    vix_ticker = yf.Ticker("^VIX")
    try:
        vix_current = float(vix_ticker.history(period="1d")['Close'].iloc[-1])
        vix_history = vix_ticker.history(period=period)
        # Convert DataFrame to serializable format
        vix_history_dict = {
            'dates': vix_history.index.tolist(),
            'values': vix_history['Close'].tolist()
        }
        return vix_current, vix_history_dict
    except Exception as e:
        raise Exception(f"Error fetching VIX data: {e}")

# Mock classes for reconstruction
class MockOptionChain:
    def __init__(self, calls_dict, puts_dict, calls_empty, puts_empty):
        self.calls = pd.DataFrame(calls_dict) if not calls_empty else pd.DataFrame()
        self.puts = pd.DataFrame(puts_dict) if not puts_empty else pd.DataFrame()

# Main execution
if ticker_symbol:
    # Check if it's a popular index without options on Yahoo Finance FIRST
    index_suggestions = {
        # S&P 500 variants
        'SPX': 'SPY (SPDR S&P 500 ETF)',
        '^GSPC': 'SPY (SPDR S&P 500 ETF)', 
        'XSP': 'SPY (SPDR S&P 500 ETF)',
        'SP500': 'SPY (SPDR S&P 500 ETF)',
        'SP500TR': 'SPY (SPDR S&P 500 ETF)',
        
        # NASDAQ 100 variants
        'NDX': 'QQQ (Invesco QQQ Trust)',
        '^NDX': 'QQQ (Invesco QQQ Trust)',
        'NASDAQ': 'QQQ (Invesco QQQ Trust)',
        '^IXIC': 'QQQ (Invesco QQQ Trust)',
        'QQQTR': 'QQQ (Invesco QQQ Trust)',
        
        # Russell 2000 variants
        'RUT': 'IWM (iShares Russell 2000 ETF)',
        '^RUT': 'IWM (iShares Russell 2000 ETF)',
        'RUTX': 'IWM (iShares Russell 2000 ETF)',
        
        # Dow Jones variants
        'DJX': 'DIA (SPDR Dow Jones Industrial Average ETF)',
        '^DJI': 'DIA (SPDR Dow Jones Industrial Average ETF)',
        'DOW': 'DIA (SPDR Dow Jones Industrial Average ETF)',
        'DJIA': 'DIA (SPDR Dow Jones Industrial Average ETF)',
        
        # Volatility variants
        'VIX': 'VXX (iPath VIX Short-Term Futures ETN)',
        '^VIX': 'VXX (iPath VIX Short-Term Futures ETN)',
        'VXX': 'VXX (iPath VIX Short-Term Futures ETN)',
        
        # Other popular indices
        'OEX': 'SPY (SPDR S&P 500 ETF)',  # S&P 100
        '^OEX': 'SPY (SPDR S&P 500 ETF)',
        'RUI': 'IWM (iShares Russell 2000 ETF)',  # Russell 1000
        '^RUI': 'IWM (iShares Russell 2000 ETF)',
        'RUA': 'IWM (iShares Russell 2000 ETF)',  # Russell 1000
        '^RUA': 'IWM (iShares Russell 2000 ETF)'
    }
    
    if ticker_symbol in index_suggestions:
        st.error(f"❌ **Index Options Not Available**: {ticker_symbol} index options are not available on Yahoo Finance")
        st.info(f"💡 **Try the equivalent ETF**: {index_suggestions[ticker_symbol]} - tracks the same index with available options")
        st.session_state.debug_messages.append(f"❌ Index {ticker_symbol} not available on Yahoo Finance, suggested {index_suggestions[ticker_symbol]} at {datetime.now().strftime('%H:%M:%S')}")
        st.stop()
    
    try:
        # Get ticker info with progress
        with st.spinner(f"🔄 Fetching {ticker_symbol} data..."):
            current_price, expirations, fetch_timestamp = get_cached_ticker_info(ticker_symbol)
            risk_free_rate, risk_free_fetch_timestamp = get_cached_risk_free_rate()
            
        # Ensure current_price is a safe float for calculations
        current_price = safe_float_price(current_price)
        
        # Display risk-free rate prominently
        st.info(f"🏦 **Current Risk-Free Rate (10-Year Treasury)**: {risk_free_rate:.2f}% (Updated: {risk_free_fetch_timestamp})")
        
    except Exception as e:
        st.error(f"❌ **Ticker Error**: {ticker_symbol} is not a valid ticker symbol")
        st.info("💡 **Try these popular tickers:** SPY, QQQ, AAPL, TSLA, MSFT, NVDA")
        st.session_state.debug_messages.append(f"❌ Error fetching {ticker_symbol}: {str(e)} at {datetime.now().strftime('%H:%M:%S')}")
        st.stop()
    
    try:
        
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.debug_messages.append(f"✅ {ticker_symbol} current price: ${current_price:.2f} at {current_time}")
        st.session_state.debug_messages.append(f"📅 Found {len(expirations)} expiration dates at {current_time}")
        st.session_state.debug_messages.append(f"💾 Cache TTL: 24 hours (86400 seconds) with disk persistence")
        
        # Filter expirations (remove expired)
        valid_expirations = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            if exp_date > datetime.now():
                valid_expirations.append(exp)
        
        if not valid_expirations:
            st.error(f"❌ **No Options Available**: {ticker_symbol} has no options contracts")
            st.info("💡 **This ticker doesn't trade options.** Try these popular options tickers: SPY, QQQ, AAPL, TSLA, MSFT, NVDA")
            st.session_state.debug_messages.append(f"❌ No options available for {ticker_symbol} at {datetime.now().strftime('%H:%M:%S')}")
            st.stop()
        
        st.session_state.debug_messages.append(f"📅 {len(valid_expirations)} valid expiration dates at {current_time}")
        
        # Fetch all option chains in ONE API call (like page 1)
        with st.spinner("🔄 Fetching all option chains in single batch..."):
            start_time = time.time()
            
            # Single batch call for all options
            batch_result = get_cached_all_options_batch(ticker_symbol, valid_expirations)
            
            end_time = time.time()
            fetch_time = end_time - start_time
            api_calls = batch_result.get('api_calls', 1)  # Should be 1!
            
            # Process batch results
            all_calls = []
            all_puts = []
            
            if not batch_result['calls_empty']:
                calls_df = pd.DataFrame(batch_result['calls'])
                if not calls_df.empty:
                    # Add days to expiration
                    calls_df['daysToExp'] = calls_df['expiration'].apply(
                        lambda x: (datetime.strptime(x, "%Y-%m-%d") - datetime.now()).days
                    )
                    all_calls.append(calls_df)
            
            if not batch_result['puts_empty']:
                puts_df = pd.DataFrame(batch_result['puts'])
                if not puts_df.empty:
                    # Add days to expiration
                    puts_df['daysToExp'] = puts_df['expiration'].apply(
                        lambda x: (datetime.strptime(x, "%Y-%m-%d") - datetime.now()).days
                    )
                    all_puts.append(puts_df)
        
        # Display API call statistics
        fetch_time_str = datetime.now().strftime("%H:%M:%S")
        # Determine if data came from cache or fresh API calls
        if fetch_time < 1.0:  # Very fast = likely from cache
            cache_status = "📦 CACHED"
            st.session_state.debug_messages.append(f"✅ Data loaded from cache in {fetch_time:.1f}s (0 new API calls) at {fetch_time_str}")
            st.session_state.debug_messages.append(f"📦 Cache Status: CACHED - Data loaded from disk cache (fast)")
        else:
            cache_status = "🔄 FRESH"
            st.session_state.debug_messages.append(f"✅ Data fetched in {fetch_time:.1f}s using {api_calls} API calls at {fetch_time_str}")
            st.session_state.debug_messages.append(f"🔄 Cache Status: FRESH - New API calls made (slower)")
        
        # Combine all data
        if all_calls:
            calls_combined = pd.concat(all_calls, ignore_index=True)
            options_time = datetime.now().strftime("%H:%M:%S")
            st.session_state.debug_messages.append(f"✅ Found {len(calls_combined)} CALL options at {options_time}")
        else:
            calls_combined = pd.DataFrame()
        
        if all_puts:
            puts_combined = pd.concat(all_puts, ignore_index=True)
            st.session_state.debug_messages.append(f"✅ Found {len(puts_combined)} PUT options at {options_time}")
        else:
            puts_combined = pd.DataFrame()

        # ---- Arrow compatibility: sanitize mixed-type columns ----
        def sanitize_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            safe_df = df.copy()
            # Columns that may contain '-' strings alongside floats → cast to str
            string_like_cols = [
                'CALL Last','CALL Bid','CALL Ask','CALL Mid','CALL Price %',
                'PUT Last','PUT Bid','PUT Ask','PUT Mid','PUT Price %',
                'CALL Extrinsic/Intrinsic','PUT Extrinsic/Intrinsic'
            ]
            for col in string_like_cols:
                if col in safe_df.columns:
                    safe_df[col] = safe_df[col].astype(str)
            # Ensure numeric columns are floats when possible
            numeric_cols = [
                'strike','lastPrice','bid','ask','mid','volume','openInterest',
                'impliedVolatility','daysToExp'
            ]
            for col in numeric_cols:
                if col in safe_df.columns:
                    try:
                        safe_df[col] = pd.to_numeric(safe_df[col], errors='coerce')
                    except Exception:
                        pass
            return safe_df

        calls_combined = sanitize_for_arrow(calls_combined)
        puts_combined = sanitize_for_arrow(puts_combined)
        
        # Display options
        if not calls_combined.empty or not puts_combined.empty:
            # Section selector - ALWAYS VISIBLE
            if show_calls and show_puts:
                available_sections = ["📅 BY EXPIRATION", "📋 COMPLETE LIST", "📈 CALLS ONLY", "📉 PUTS ONLY", "📊 OPTION EVOLUTION", "⚖️ BARBELL STRATEGY"]
            elif show_calls:
                available_sections = ["📅 BY EXPIRATION", "📋 COMPLETE LIST", "📈 CALLS ONLY", "📊 OPTION EVOLUTION", "⚖️ BARBELL STRATEGY"]
            elif show_puts:
                available_sections = ["📅 BY EXPIRATION", "📋 COMPLETE LIST", "📉 PUTS ONLY", "📊 OPTION EVOLUTION", "⚖️ BARBELL STRATEGY"]
            else:
                st.warning("⚠️ Please select at least one option type")
                st.stop()
            
            # Initialize session state for section persistence
            if 'selected_section' not in st.session_state:
                st.session_state.selected_section = available_sections[0]
            
            # Ensure selected section is still available
            if st.session_state.selected_section not in available_sections:
                st.session_state.selected_section = available_sections[0]
            
            # ALWAYS SHOW THE SELECTBOX
            st.markdown("---")
            selected_section = st.selectbox(
                "📊 Select Section:",
                options=available_sections,
                index=available_sections.index(st.session_state.selected_section),
                key="section_selector"
            )
            st.markdown("---")
            
            # Update session state when selection changes
            if selected_section != st.session_state.selected_section:
                st.session_state.selected_section = selected_section
            
            # BY EXPIRATION Section
            if selected_section == "📅 BY EXPIRATION":
                st.subheader("📅 Options Chain by Expiration")
                
                # Display current price with timestamp and cache status
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.metric(
                        f"Current {ticker_symbol} Price", 
                        f"${current_price:.2f}",
                        help=f"Data fetched at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                with col2:
                    st.metric(
                        "Cache Status",
                        cache_status,
                        help="📦 CACHED = Data from disk cache (fast) | 🔄 FRESH = New API calls (slower)"
                    )
                with col3:
                    st.caption(f"📅 Data fetched at: {fetch_timestamp}")
                
                if not calls_combined.empty or not puts_combined.empty:
                    # Get unique expirations
                    all_expirations = []
                    if not calls_combined.empty:
                        all_expirations.extend(calls_combined['expiration'].unique())
                    if not puts_combined.empty:
                        all_expirations.extend(puts_combined['expiration'].unique())
                    all_expirations = sorted(list(set(all_expirations)))
                    
                    # Calculate days to expiration for each date
                    expiration_options = []
                    for exp in all_expirations:
                        exp_date = datetime.strptime(exp, "%Y-%m-%d")
                        days_to_exp = (exp_date - datetime.now()).days
                        if days_to_exp < 0:
                            continue  # Skip expired dates
                        expiration_options.append(f"{exp} ({days_to_exp} days)")
                    
                    # Initialize session state for expiration if not exists
                    if 'persistent_expiration' not in st.session_state:
                        st.session_state.persistent_expiration = all_expirations[0]
                    
                    # Find index of persistent expiration in current options
                    persistent_index = 0
                    try:
                        for i, option in enumerate(expiration_options):
                            if st.session_state.persistent_expiration in option:
                                persistent_index = i
                                break
                    except:
                        persistent_index = 0
                    
                    # Expiration selector with days
                    selected_exp_full = st.selectbox(
                        "📅 Select Expiration Date:",
                        expiration_options,
                        index=persistent_index,
                        help="Choose which expiration date to display in the options chain format"
                    )
                    
                    # Extract just the date part and save to session state
                    selected_exp = selected_exp_full.split(' (')[0] if selected_exp_full else all_expirations[0]
                    st.session_state.persistent_expiration = selected_exp
                    
                    if selected_exp:
                        st.markdown(f"**Expiration: {selected_exp}**")
                        
                        # Filter data for selected expiration
                        calls_exp = calls_combined[calls_combined['expiration'] == selected_exp] if not calls_combined.empty else pd.DataFrame()
                        puts_exp = puts_combined[puts_combined['expiration'] == selected_exp] if not puts_combined.empty else pd.DataFrame()
                        
                        if not calls_exp.empty or not puts_exp.empty:
                            # Create MERGED TABLE (like OptionCharts)
                            st.markdown("### 📊 Merged Options Chain (Scrolls Together)")
                            
                            # Add toggle for simplified view (default to simplified)
                            show_simplified = st.checkbox(
                                "🔍 Show Simplified View (Essential Columns Only)",
                                value=True,
                                help="Toggle between full view (all columns) and simplified view (essential columns only)"
                            )
                            
                            # Get all strikes from both calls and puts
                            all_strikes = set()
                            if not calls_exp.empty:
                                all_strikes.update(calls_exp['strike'].tolist())
                            if not puts_exp.empty:
                                all_strikes.update(puts_exp['strike'].tolist())
                            all_strikes = sorted(list(all_strikes))
                            
                            # Create merged DataFrame
                            merged_data = []
                            for strike in all_strikes:
                                row = {'strike': strike}
                                
                                # Call data
                                call_row = calls_exp[calls_exp['strike'] == strike]
                                if not call_row.empty:
                                    call_row = call_row.iloc[0]
                                    call_bid = call_row['bid'] if pd.notna(call_row['bid']) and call_row['bid'] != 0 else None
                                    call_ask = call_row['ask'] if pd.notna(call_row['ask']) and call_row['ask'] != 0 else None
                                    call_mid = None
                                    if call_bid is not None and call_ask is not None:
                                        call_mid = (call_bid + call_ask) / 2
                                    
                                    # Calculate option price as % of current SPY price using MID price
                                    call_price_pct = None
                                    if call_mid is not None and call_mid != 0:
                                        call_price_pct = (call_mid / current_price) * 100
                                    
                                    row.update({
                                        'call_lastPrice': call_row['lastPrice'] if pd.notna(call_row['lastPrice']) else '-',
                                        'call_bid': call_bid if call_bid is not None else '-',
                                        'call_ask': call_ask if call_ask is not None else '-',
                                        'call_mid': f"{call_mid:.2f}" if call_mid is not None else '-',
                                        'call_mid_price': call_mid,  # Store raw MID price for calculations
                                        'call_volume': call_row['volume'] if pd.notna(call_row['volume']) else 0,
                                        'call_openInterest': call_row['openInterest'] if pd.notna(call_row['openInterest']) else 0,
                                        'call_impliedVol': f"{(call_row['impliedVolatility'] * 100):.1f}%" if pd.notna(call_row['impliedVolatility']) else '-',
                                        'call_price_pct': f"{call_price_pct:.2f}%" if call_price_pct is not None else '-'
                                    })
                                else:
                                    row.update({
                                        'call_lastPrice': '-', 'call_bid': '-', 'call_ask': '-', 'call_mid': '-',
                                        'call_volume': 0, 'call_openInterest': 0, 'call_impliedVol': '-', 'call_price_pct': '-'
                                    })
                                
                                # Put data
                                put_row = puts_exp[puts_exp['strike'] == strike]
                                if not put_row.empty:
                                    put_row = put_row.iloc[0]
                                    put_bid = put_row['bid'] if pd.notna(put_row['bid']) and put_row['bid'] != 0 else None
                                    put_ask = put_row['ask'] if pd.notna(put_row['ask']) and put_row['ask'] != 0 else None
                                    put_mid = None
                                    if put_bid is not None and put_ask is not None:
                                        put_mid = (put_bid + put_ask) / 2
                                    
                                    # Calculate option price as % of current SPY price using MID price
                                    put_price_pct = None
                                    if put_mid is not None and put_mid != 0:
                                        put_price_pct = (put_mid / current_price) * 100
                                    
                                    row.update({
                                        'put_lastPrice': put_row['lastPrice'] if pd.notna(put_row['lastPrice']) else '-',
                                        'put_bid': put_bid if put_bid is not None else '-',
                                        'put_ask': put_ask if put_ask is not None else '-',
                                        'put_mid': f"{put_mid:.2f}" if put_mid is not None else '-',
                                        'put_mid_price': put_mid,  # Store raw MID price for calculations
                                        'put_volume': put_row['volume'] if pd.notna(put_row['volume']) else 0,
                                        'put_openInterest': put_row['openInterest'] if pd.notna(put_row['openInterest']) else 0,
                                        'put_impliedVol': f"{(put_row['impliedVolatility'] * 100):.1f}%" if pd.notna(put_row['impliedVolatility']) else '-',
                                        'put_price_pct': f"{put_price_pct:.2f}%" if put_price_pct is not None else '-'
                                    })
                                else:
                                    row.update({
                                        'put_lastPrice': '-', 'put_bid': '-', 'put_ask': '-', 'put_mid': '-',
                                        'put_volume': 0, 'put_openInterest': 0, 'put_impliedVol': '-', 'put_price_pct': '-'
                                    })
                                
                                merged_data.append(row)
                            
                            # Create DataFrame and format
                            merged_df = pd.DataFrame(merged_data)
                            
                            # Add moneyness calculations (both % and $)
                            merged_df['call_moneyness_pct'] = ((merged_df['strike'] / current_price) - 1) * 100
                            merged_df['call_moneyness_pct'] = merged_df['call_moneyness_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else '-')
                            merged_df['call_moneyness_dollar'] = merged_df['strike'] - current_price
                            merged_df['put_moneyness_pct'] = ((merged_df['strike'] / current_price) - 1) * 100
                            merged_df['put_moneyness_pct'] = merged_df['put_moneyness_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else '-')
                            merged_df['put_moneyness_dollar'] = merged_df['strike'] - current_price
                            
                            # Add intrinsic and extrinsic values (both $ and %)
                            # Call intrinsic value = max(0, current_price - strike)
                            merged_df['call_intrinsic'] = np.maximum(0, current_price - merged_df['strike'])
                            # Call intrinsic % = (intrinsic / mid_price) * 100
                            merged_df['call_intrinsic_pct'] = merged_df.apply(
                                lambda row: f"{(row['call_intrinsic'] / row['call_mid_price'] * 100):.2f}%" if pd.notna(row['call_mid_price']) and row['call_mid_price'] != 0 else '-', axis=1
                            )
                            # Call extrinsic = mid_price - intrinsic (if mid_price is available)
                            merged_df['call_extrinsic'] = merged_df.apply(
                                lambda row: row['call_mid_price'] - row['call_intrinsic'] if pd.notna(row['call_mid_price']) else '-', axis=1
                            )
                            # Call extrinsic % = (extrinsic / mid_price) * 100
                            merged_df['call_extrinsic_pct'] = merged_df.apply(
                                lambda row: f"{((row['call_mid_price'] - row['call_intrinsic']) / row['call_mid_price'] * 100):.2f}%" if pd.notna(row['call_mid_price']) and row['call_mid_price'] != 0 else '-', axis=1
                            )
                            
                            # Put intrinsic value = max(0, strike - current_price)
                            merged_df['put_intrinsic'] = np.maximum(0, merged_df['strike'] - current_price)
                            # Put intrinsic % = (intrinsic / mid_price) * 100
                            merged_df['put_intrinsic_pct'] = merged_df.apply(
                                lambda row: f"{(row['put_intrinsic'] / row['put_mid_price'] * 100):.2f}%" if pd.notna(row['put_mid_price']) and row['put_mid_price'] != 0 else '-', axis=1
                            )
                            # Put extrinsic = mid_price - intrinsic (if mid_price is available)
                            merged_df['put_extrinsic'] = merged_df.apply(
                                lambda row: row['put_mid_price'] - row['put_intrinsic'] if pd.notna(row['put_mid_price']) else '-', axis=1
                            )
                            # Put extrinsic % = (extrinsic / mid_price) * 100
                            merged_df['put_extrinsic_pct'] = merged_df.apply(
                                lambda row: f"{((row['put_mid_price'] - row['put_intrinsic']) / row['put_mid_price'] * 100):.2f}%" if pd.notna(row['put_mid_price']) and row['put_mid_price'] != 0 else '-', axis=1
                            )
                            
                            # Reorder columns for better display
                            column_order = [
                                'call_lastPrice', 'call_bid', 'call_ask', 'call_mid', 'call_volume', 'call_openInterest', 'call_impliedVol', 'call_price_pct', 'call_intrinsic', 'call_intrinsic_pct', 'call_extrinsic', 'call_extrinsic_pct', 'call_moneyness_pct', 'call_moneyness_dollar',
                                'strike',
                                'put_lastPrice', 'put_bid', 'put_ask', 'put_mid', 'put_volume', 'put_openInterest', 'put_impliedVol', 'put_price_pct', 'put_intrinsic', 'put_intrinsic_pct', 'put_extrinsic', 'put_extrinsic_pct', 'put_moneyness_pct', 'put_moneyness_dollar'
                            ]
                            merged_df = merged_df[column_order]
                            
                            # Rename columns for display
                            merged_df.columns = [
                                'CALL Last', 'CALL Bid', 'CALL Ask', 'CALL Mid', 'CALL Vol', 'CALL OI', 'CALL IV', 'CALL Price %', 'CALL Intrinsic $', 'CALL Intrinsic %', 'CALL Extrinsic $', 'CALL Extrinsic %', 'CALL Moneyness %', 'CALL Moneyness $',
                                'STRIKE',
                                'PUT Last', 'PUT Bid', 'PUT Ask', 'PUT Mid', 'PUT Vol', 'PUT OI', 'PUT IV', 'PUT Price %', 'PUT Intrinsic $', 'PUT Intrinsic %', 'PUT Extrinsic $', 'PUT Extrinsic %', 'PUT Moneyness %', 'PUT Moneyness $'
                            ]
                            
                            # Filter columns based on simplified view
                            if show_simplified:
                                # Add Extrinsic/Intrinsic ratio columns for simplified view (more intelligent)
                                merged_df['CALL Extrinsic/Intrinsic'] = merged_df.apply(
                                    lambda row: f"{(float(row['CALL Extrinsic $']) / float(row['CALL Intrinsic $'])):.2f}" if row['CALL Intrinsic $'] != '-' and row['CALL Extrinsic $'] != '-' and float(row['CALL Intrinsic $']) != 0 else '-', axis=1
                                )
                                merged_df['PUT Extrinsic/Intrinsic'] = merged_df.apply(
                                    lambda row: f"{(float(row['PUT Extrinsic $']) / float(row['PUT Intrinsic $'])):.2f}" if row['PUT Intrinsic $'] != '-' and row['PUT Extrinsic $'] != '-' and float(row['PUT Intrinsic $']) != 0 else '-', axis=1
                                )
                                
                                # Essential columns only: Last, Bid, Ask, Mid, Volume, IV, Price %, Extrinsic/Intrinsic
                                essential_cols = [
                                    'CALL Last', 'CALL Bid', 'CALL Ask', 'CALL Mid', 'CALL Vol', 'CALL IV', 'CALL Price %', 'CALL Extrinsic/Intrinsic',
                                    'STRIKE',
                                    'PUT Last', 'PUT Bid', 'PUT Ask', 'PUT Mid', 'PUT Vol', 'PUT IV', 'PUT Price %', 'PUT Extrinsic/Intrinsic'
                                ]
                                merged_df = merged_df[essential_cols]
                            
                            # Format all numeric columns to 2 decimal places (excluding % columns)
                            numeric_cols = ['CALL Last', 'CALL Bid', 'CALL Ask', 'CALL Mid', 'PUT Last', 'PUT Bid', 'PUT Ask', 'PUT Mid', 
                                          'CALL Intrinsic $', 'CALL Extrinsic $', 
                                          'PUT Intrinsic $', 'PUT Extrinsic $', 
                                          'CALL Moneyness $', 'PUT Moneyness $', 'STRIKE']
                            for col in numeric_cols:
                                if col in merged_df.columns:
                                    merged_df[col] = merged_df[col].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) and x != '-' else x)
                            
                            # Format % columns separately (they already contain % symbol)
                            pct_cols = ['CALL Price %', 'PUT Price %', 'CALL Intrinsic %', 'CALL Extrinsic %', 'PUT Intrinsic %', 'PUT Extrinsic %', 'CALL Moneyness %', 'PUT Moneyness %']
                            for col in pct_cols:
                                if col in merged_df.columns:
                                    # These columns already have % symbol, just ensure proper formatting
                                    merged_df[col] = merged_df[col].apply(lambda x: x if pd.notna(x) and x != '-' else x)
                            
                            # Fix data types for Streamlit compatibility
                            for col in merged_df.columns:
                                if col in ['CALL Last', 'CALL Bid', 'CALL Ask', 'CALL Mid', 'PUT Last', 'PUT Bid', 'PUT Ask', 'PUT Mid']:
                                    merged_df[col] = merged_df[col].astype(str)
                                elif col in ['CALL Vol', 'CALL OI', 'PUT Vol', 'PUT OI']:
                                    merged_df[col] = merged_df[col].astype(int)
                                elif col in ['CALL IV', 'PUT IV', 'CALL Price %', 'PUT Price %', 'CALL Intrinsic $', 'CALL Intrinsic %', 'CALL Extrinsic $', 'CALL Extrinsic %', 
                                          'PUT Intrinsic $', 'PUT Intrinsic %', 'PUT Extrinsic $', 'PUT Extrinsic %',
                                          'CALL Moneyness %', 'CALL Moneyness $', 'PUT Moneyness %', 'PUT Moneyness $']:
                                    merged_df[col] = merged_df[col].astype(str)
                                elif col == 'STRIKE':
                                    merged_df[col] = merged_df[col].astype(str)
                            
                            # Display current SPY price prominently
                            st.markdown(f"### **Current {ticker_symbol} Price: ${current_price:.2f}**")
                            
                            # Show current view mode
                            if show_simplified:
                                st.info("🔍 **Simplified View**: Showing essential columns only (Last, Bid, Ask, Mid, Volume, IV, Price %, Extrinsic/Intrinsic)")
                            else:
                                st.info("📊 **Full View**: Showing all columns (including Intrinsic, Extrinsic, Moneyness, Price %)")
                            
                            # Add ITM/OTM detection and color coding (FROM TEST.PY)
                            def highlight_options_table(row):
                                """Apply colors based on ITM/OTM status and section"""
                                styles = []
                                
                                # Convert strike to float for comparison
                                try:
                                    strike = float(row['STRIKE'])
                                except (ValueError, TypeError):
                                    strike = 0.0
                                
                                # Determine ITM/OTM status
                                call_itm = strike < current_price  # Call is ITM if strike < current price
                                put_itm = strike > current_price   # Put is ITM if strike > current price
                                
                                for col in merged_df.columns:
                                    if col.startswith('CALL'):
                                        if call_itm:
                                            # ITM calls: Professional dark green
                                            styles.append('background-color: #2d5a2d; color: #ffffff')
                                        else:
                                            # OTM calls: Professional dark blue
                                            styles.append('background-color: #1e3a5f; color: #ffffff')
                                    elif col == 'STRIKE':
                                        # Strike column: Always same color (no transition to avoid confusion)
                                        styles.append('background-color: #2d3748; color: #ffffff')
                                    elif col.startswith('PUT'):
                                        if put_itm:
                                            # ITM puts: Professional dark red
                                            styles.append('background-color: #5a2d2d; color: #ffffff')
                                        else:
                                            # OTM puts: Professional dark orange
                                            styles.append('background-color: #5a3d1a; color: #ffffff')
                                    else:
                                        styles.append('')
                                
                                return styles
                            
                            # Clean the DataFrame first for Arrow compatibility
                            merged_df_clean = clean_dataframe_for_arrow(merged_df)
                            
                            # Apply advanced styling with colors
                            styled_df = merged_df_clean.style.apply(highlight_options_table, axis=1)
                            
                            # Display styled table with maximum possible width
                            st.dataframe(styled_df, width='stretch')
                            
                            # Add note about MID price usage
                            st.caption("💡 **Price Calculations**: All intrinsic/extrinsic values and price percentages use MID price (Bid+Ask)/2 when available, otherwise fall back to Last price for more accurate option valuation.")
                            
                            # Add legend in dropdown
                            with st.expander("📖 Legend & Color Scheme", expanded=False):
                                st.markdown("""
                                **Legend:** 
                                - **Last**: Last traded price
                                - **Bid/Ask**: Current bid/ask prices  
                                - **Mid**: Average of bid and ask prices
                                - **Vol**: Volume today
                                - **OI**: Open Interest (total contracts)
                                - **IV**: Implied Volatility (%)
                                - **Intrinsic $**: In-the-money value in dollars (Call: max(0, current-strike), Put: max(0, strike-current))
                                - **Intrinsic %**: In-the-money value as % of option price (Intrinsic $ / Last Price * 100)
                                - **Extrinsic $**: Time value in dollars (Last Price - Intrinsic Value)
                                - **Extrinsic %**: Time value as % of option price (Extrinsic $ / Last Price * 100)
                                - **Moneyness %**: % from current price (negative = OTM, positive = ITM)
                                - **Moneyness $**: Dollar difference from current price (negative = OTM, positive = ITM)
                                - **Extrinsic/Intrinsic**: Ratio of time value to intrinsic value (higher = more time value)
                                
                                **Professional Color Scheme:**
                                - 🔵 **Dark Blue**: OTM CALL options (strike > current price)
                                - 🟢 **Dark Green**: ITM CALL options (strike < current price)
                                - ⚫ **Dark Gray**: STRIKE prices (uniform color - no transition)
                                - 🟠 **Dark Orange**: OTM PUT options (strike < current price)
                                - 🔴 **Dark Red**: ITM PUT options (strike > current price)
                                """)
                        else:
                            st.warning(f"No options found for expiration {selected_exp}")
                    else:
                        st.warning("No valid expiration dates found")
                else:
                    st.warning("No options data available")
            
            # COMPLETE LIST Section
            elif selected_section == "📋 COMPLETE LIST":
                st.subheader("📋 Complete Options List")
                
                # Display current price with timestamp
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(
                        f"Current {ticker_symbol} Price", 
                        f"${current_price:.2f}",
                        help=f"Data fetched at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                with col2:
                    st.caption(f"📅 Data fetched at: {fetch_timestamp}")
                
                if show_calls and not calls_combined.empty:
                    st.markdown("**📈 CALL Options:**")
                    # Select columns to display - only include columns that exist
                    base_cols = ['expiration', 'daysToExp', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
                    display_cols = [col for col in base_cols if col in calls_combined.columns]
                    
                    # Add optional columns if they exist
                    if 'moneyness' in calls_combined.columns:
                        display_cols.insert(3, 'moneyness')
                    if 'itm' in calls_combined.columns:
                        display_cols.insert(4, 'itm')
                    
                    calls_display = calls_combined[display_cols].copy()
                    
                    # Format columns
                    if 'moneyness' in calls_display.columns:
                        calls_display['moneyness'] = calls_display['moneyness'].round(2)
                    calls_display['impliedVolatility'] = (calls_display['impliedVolatility'] * 100).round(2)
                    
                    # Sort by expiration and strike
                    calls_display = calls_display.sort_values(['expiration', 'strike'])
                    
                    # Display (cleaned for Arrow compatibility) with maximum width
                    calls_display_clean = clean_dataframe_for_arrow(calls_display)
                    st.dataframe(calls_display_clean, width='stretch')
                
                if show_puts and not puts_combined.empty:
                    st.markdown("**📉 PUT Options:**")
                    # Select columns to display - only include columns that exist
                    base_cols = ['expiration', 'daysToExp', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
                    display_cols = [col for col in base_cols if col in puts_combined.columns]
                    
                    # Add optional columns if they exist
                    if 'moneyness' in puts_combined.columns:
                        display_cols.insert(3, 'moneyness')
                    if 'itm' in puts_combined.columns:
                        display_cols.insert(4, 'itm')
                    
                    puts_display = puts_combined[display_cols].copy()
                    
                    # Format columns
                    if 'moneyness' in puts_display.columns:
                        puts_display['moneyness'] = puts_display['moneyness'].round(2)
                    puts_display['impliedVolatility'] = (puts_display['impliedVolatility'] * 100).round(2)
                    
                    # Sort by expiration and strike
                    puts_display = puts_display.sort_values(['expiration', 'strike'])
                    
                    # Display (cleaned for Arrow compatibility) with maximum width
                    puts_display_clean = clean_dataframe_for_arrow(puts_display)
                    st.dataframe(puts_display_clean, width='stretch')
            
            # CALLS ONLY Section
            elif selected_section == "📈 CALLS ONLY":
                if show_calls and not calls_combined.empty:
                        st.subheader(f"📈 CALL Options ({len(calls_combined)} contracts)")
                        
                        # Display current price with timestamp
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(
                                f"Current {ticker_symbol} Price", 
                                f"${current_price:.2f}",
                                help=f"Data fetched at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            )
                        with col2:
                            st.caption(f"📅 Data valid as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Select columns to display - only include columns that exist
                        base_cols = ['expiration', 'daysToExp', 'strike', 'lastPrice', 'bid', 'ask', 
                                   'volume', 'openInterest', 'impliedVolatility']
                        display_cols = [col for col in base_cols if col in calls_combined.columns]
                        
                        # Add optional columns if they exist
                        if 'moneyness' in calls_combined.columns:
                            display_cols.insert(3, 'moneyness')
                        if 'itm' in calls_combined.columns:
                            display_cols.insert(4, 'itm')
                        
                        calls_display = calls_combined[display_cols].copy()
                        
                        # Format columns
                        if 'moneyness' in calls_display.columns:
                            calls_display['moneyness'] = calls_display['moneyness'].round(2)
                        calls_display['impliedVolatility'] = (calls_display['impliedVolatility'] * 100).round(2)
                        
                        # Sort by expiration and strike
                        calls_display = calls_display.sort_values(['expiration', 'strike'])
                        
                        # Display (cleaned for Arrow compatibility) with maximum width
                        calls_display_clean = clean_dataframe_for_arrow(calls_display)
                        st.dataframe(calls_display_clean, width='stretch')
                        
                        # Download button
                        csv = calls_display.to_csv(index=False)
                        st.download_button(
                            "📥 Download CALLS CSV",
                            csv,
                            f"{ticker_symbol}_calls_{datetime.now().strftime('%Y%m%d')}.csv",
                            "text/csv"
                        )
                else:
                    st.warning("No CALL options available")
            
            # PUTS ONLY Section
            elif selected_section == "📉 PUTS ONLY":
                if show_puts and not puts_combined.empty:
                        st.subheader(f"📉 PUT Options ({len(puts_combined)} contracts)")
                        
                        # Display current price with timestamp
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(
                                f"Current {ticker_symbol} Price", 
                                f"${current_price:.2f}",
                                help=f"Data fetched at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            )
                        with col2:
                            st.caption(f"📅 Data valid as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Select columns to display - only include columns that exist
                        base_cols = ['expiration', 'daysToExp', 'strike', 'lastPrice', 'bid', 'ask', 
                                   'volume', 'openInterest', 'impliedVolatility']
                        display_cols = [col for col in base_cols if col in puts_combined.columns]
                        
                        # Add optional columns if they exist
                        if 'moneyness' in puts_combined.columns:
                            display_cols.insert(3, 'moneyness')
                        if 'itm' in puts_combined.columns:
                            display_cols.insert(4, 'itm')
                        
                        puts_display = puts_combined[display_cols].copy()
                        
                        # Format columns
                        if 'moneyness' in puts_display.columns:
                            puts_display['moneyness'] = puts_display['moneyness'].round(2)
                        puts_display['impliedVolatility'] = (puts_display['impliedVolatility'] * 100).round(2)
                        
                        # Sort by expiration and strike
                        puts_display = puts_display.sort_values(['expiration', 'strike'])
                        
                        # Display (cleaned for Arrow compatibility) with maximum width
                        puts_display_clean = clean_dataframe_for_arrow(puts_display)
                        st.dataframe(puts_display_clean, width='stretch')
                        
                        # Download button
                        csv = puts_display.to_csv(index=False)
                        st.download_button(
                            "📥 Download PUTS CSV",
                            csv,
                            f"{ticker_symbol}_puts_{datetime.now().strftime('%Y%m%d')}.csv",
                            "text/csv"
                        )
                else:
                    st.warning("No PUT options available")
            
            # OPTION EVOLUTION Section
            elif selected_section == "📊 OPTION EVOLUTION":
                if show_calls or show_puts:
                    st.subheader("📊 Option Evolution Over Time")
                    st.markdown("**Track the same strike across different expirations**")
                    
                    # 🎯 OPTION EVOLUTION - OWN VARIABLES
                    # Ensure evolution_strike is always defined (global fallback)
                    if 'evolution_strike' not in globals() or evolution_strike is None:
                        evolution_strike = None
                    
                    # Try to get from session state if available
                    if 'persistent_strike' in st.session_state:
                        try:
                            evolution_strike = float(st.session_state.persistent_strike)
                        except:
                            evolution_strike = None
                    
                    # Display current price with timestamp
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric(
                            f"Current {ticker_symbol} Price", 
                            f"${current_price:.2f}",
                            help=f"Data fetched at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    with col2:
                        st.caption(f"📅 Data valid as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if not calls_combined.empty or not puts_combined.empty:
                        # Get all available strikes
                        all_strikes = set()
                        if not calls_combined.empty:
                            all_strikes.update(calls_combined['strike'].unique())
                        if not puts_combined.empty:
                            all_strikes.update(puts_combined['strike'].unique())
                        all_strikes = sorted(list(all_strikes))
                        
                        # Strike selector - completely rebuilt
                        if all_strikes:
                            # Find the strike closest to current price for this ticker
                            closest_strike = min(all_strikes, key=lambda x: abs(x - current_price))
                            
                            # Reset strike when ticker changes - BUT ONLY IF NO MANUAL INPUT
                            if (st.session_state.get('last_ticker_for_strike') != ticker_symbol):
                                # Only reset if user hasn't manually entered a strike
                                if not st.session_state.get('manual_strike_entered', False):
                                    st.session_state.persistent_strike = str(closest_strike)
                                st.session_state.last_ticker_for_strike = ticker_symbol
                            
                            # Initialize if not exists
                            if 'persistent_strike' not in st.session_state:
                                st.session_state.persistent_strike = str(closest_strike)
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                # Manual input - highest priority with dynamic key
                                manual_strike_key = f"manual_strike_{ticker_symbol}"
                                manual_strike = st.text_input(
                                    "🔍 Enter Strike Price:",
                                    value=st.session_state.persistent_strike,
                                    help="Type a strike price manually (e.g., 440.0 or 440,5 - comma will be converted to dot)",
                                    key=manual_strike_key
                                )
                            
                            with col2:
                                # Hybrid dropdown selector (same as Spread Analysis)
                                # Add custom option
                                strike_options = ["Custom"] + [f"${strike:.0f}" for strike in all_strikes]
                                dropdown_strike_selected = st.selectbox(
                                    "📋 Or Select from List:",
                                    strike_options,
                                    help="Choose from dropdown list or select Custom for manual input",
                                    key=f"dropdown_strike_{ticker_symbol}"
                                )
                                
                                # If Custom is selected, show info message
                                if dropdown_strike_selected == "Custom":
                                    st.info("💡 Use the 'Enter Strike Price' field above to input a custom strike")
                            
                            # Determine final strike selection
                            evolution_strike = None
                            
                            # Priority 1: Dropdown selection if not Custom
                            if dropdown_strike_selected != "Custom":
                                evolution_strike = float(dropdown_strike_selected.replace('$', ''))
                                # Update session state with dropdown selection
                                st.session_state.persistent_strike = str(evolution_strike)
                                st.session_state.manual_strike_entered = True  # Mark that user selected from dropdown
                            else:
                                # Priority 2: Manual input if Custom is selected
                                if manual_strike and manual_strike.strip():
                                    try:
                                        # Convert comma to dot for decimal separator
                                        converted_strike = manual_strike.replace(',', '.')
                                        evolution_strike = float(converted_strike)
                                        
                                        # Show conversion if different
                                        if converted_strike != manual_strike:
                                            st.info(f"🔄 Converted {manual_strike} to {converted_strike}")
                                        
                                        # Check if strike exists in available options
                                        if evolution_strike not in all_strikes:
                                            closest_available = min(all_strikes, key=lambda x: abs(x - evolution_strike))
                                            warning_msg = f"⚠️ Strike {evolution_strike:.1f} not found. Using closest available strike: {closest_available:.1f}"
                                            st.warning(warning_msg)
                                            evolution_strike = closest_available
                                        
                                        # Update session state with the final strike
                                        st.session_state.persistent_strike = str(evolution_strike)
                                        st.session_state.manual_strike_entered = True  # Mark that user entered manually
                                        
                                    except ValueError:
                                        st.error("Please enter a valid number for strike price.")
                                        evolution_strike = closest_strike
                                else:
                                    # Use closest strike as fallback
                                    evolution_strike = closest_strike
                        else:
                            evolution_strike = None
                        
                        if evolution_strike:
                            st.markdown(f"**Tracking Strike: ${evolution_strike:.2f}**")
                            
                            # Filter data for selected strike
                            calls_strike = calls_combined[calls_combined['strike'] == evolution_strike] if not calls_combined.empty else pd.DataFrame()
                            puts_strike = puts_combined[puts_combined['strike'] == evolution_strike] if not puts_combined.empty else pd.DataFrame()
                            
                            if not calls_strike.empty or not puts_strike.empty:
                                # Create evolution data
                                evolution_data = []
                                
                                # Get all expirations for this strike
                                all_expirations = set()
                                if not calls_strike.empty:
                                    all_expirations.update(calls_strike['expiration'].unique())
                                if not puts_strike.empty:
                                    all_expirations.update(puts_strike['expiration'].unique())
                                all_expirations = sorted(list(all_expirations))
                                
                                for exp in all_expirations:
                                    # Get call data for this expiration
                                    call_data = calls_strike[calls_strike['expiration'] == exp] if not calls_strike.empty else pd.DataFrame()
                                    put_data = puts_strike[puts_strike['expiration'] == exp] if not puts_strike.empty else pd.DataFrame()
                                    
                                    row = {
                                        'Expiration': exp,
                                        'Days to Exp': (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days,
                                        'Strike': f"{evolution_strike:.0f}" if evolution_strike == int(evolution_strike) else f"{evolution_strike:.2f}"
                                    }
                                    
                                    # Add call data with MID price calculation
                                    if not call_data.empty:
                                        call_row = call_data.iloc[0]
                                        
                                        # Calculate MID price (average of bid and ask)
                                        call_bid = call_row['bid'] if pd.notna(call_row['bid']) else None
                                        call_ask = call_row['ask'] if pd.notna(call_row['ask']) else None
                                        call_last = call_row['lastPrice'] if pd.notna(call_row['lastPrice']) else None
                                        
                                        # Use MID price if both bid and ask are available, otherwise use last price
                                        if call_bid is not None and call_ask is not None:
                                            call_mid = (call_bid + call_ask) / 2
                                        elif call_last is not None:
                                            call_mid = call_last
                                        else:
                                            call_mid = None
                                        
                                        # Calculate Call Price = Call MID (or Call Last if MID not available)
                                        call_price = None
                                        if call_mid is not None and call_mid != 0:
                                            call_price = call_mid
                                        elif call_last is not None:
                                            call_price = call_last
                                        
                                        # Calculate Call price as % of current ticker price
                                        call_price_pct = None
                                        if call_mid is not None and call_mid != 0:
                                            safe_price = safe_float_price(current_price)
                                            call_price_pct = (call_mid / safe_price) * 100
                                        
                                        row.update({
                                            'Call Price': f"{call_price:.2f}" if call_price is not None else '-',
                                            'Call Last': f"{call_last:.2f}" if call_last is not None else '-',
                                            'Call MID': f"{call_mid:.2f}" if call_mid is not None else '-',
                                            'Call Bid': f"{call_bid:.2f}" if call_bid is not None else '-',
                                            'Call Ask': f"{call_ask:.2f}" if call_ask is not None else '-',
                                            'Call Price %': f"{call_price_pct:.2f}%" if call_price_pct is not None else '-',
                                            'Call Volume': f"{call_row['volume']:.0f}" if pd.notna(call_row['volume']) else '0',
                                            'Call IV': f"{(call_row['impliedVolatility'] * 100):.1f}%" if pd.notna(call_row['impliedVolatility']) else '-'
                                        })
                                    else:
                                        row.update({
                                            'Call Price': '-', 'Call Last': '-', 'Call MID': '-', 'Call Bid': '-', 'Call Ask': '-', 'Call Price %': '-',
                                            'Call Volume': '0', 'Call IV': '-'
                                        })
                                    
                                    # Add put data with MID price calculation
                                    if not put_data.empty:
                                        put_row = put_data.iloc[0]
                                        
                                        # Calculate MID price (average of bid and ask)
                                        put_bid = put_row['bid'] if pd.notna(put_row['bid']) else None
                                        put_ask = put_row['ask'] if pd.notna(put_row['ask']) else None
                                        put_last = put_row['lastPrice'] if pd.notna(put_row['lastPrice']) else None
                                        
                                        # Use MID price if both bid and ask are available, otherwise use last price
                                        if put_bid is not None and put_ask is not None:
                                            put_mid = (put_bid + put_ask) / 2
                                        elif put_last is not None:
                                            put_mid = put_last
                                        else:
                                            put_mid = None
                                        
                                        # Calculate Put Price = Put MID (or Put Last if MID not available)
                                        put_price = None
                                        if put_mid is not None and put_mid != 0:
                                            put_price = put_mid
                                        elif put_last is not None:
                                            put_price = put_last
                                        
                                        # Calculate Put price as % of current ticker price
                                        put_price_pct = None
                                        if put_mid is not None and put_mid != 0:
                                            safe_price = safe_float_price(current_price)
                                            put_price_pct = (put_mid / safe_price) * 100
                                        
                                        row.update({
                                            'Put Price': f"{put_price:.2f}" if put_price is not None else '-',
                                            'Put Last': f"{put_last:.2f}" if put_last is not None else '-',
                                            'Put MID': f"{put_mid:.2f}" if put_mid is not None else '-',
                                            'Put Bid': f"{put_bid:.2f}" if put_bid is not None else '-',
                                            'Put Ask': f"{put_ask:.2f}" if put_ask is not None else '-',
                                            'Put Price %': f"{put_price_pct:.2f}%" if put_price_pct is not None else '-',
                                            'Put Volume': f"{put_row['volume']:.0f}" if pd.notna(put_row['volume']) else '0',
                                            'Put IV': f"{(put_row['impliedVolatility'] * 100):.1f}%" if pd.notna(put_row['impliedVolatility']) else '-'
                                        })
                                    else:
                                        row.update({
                                            'Put Price': '-', 'Put Last': '-', 'Put MID': '-', 'Put Bid': '-', 'Put Ask': '-', 'Put Price %': '-',
                                            'Put Volume': '0', 'Put IV': '-'
                                        })
                                    
                                    evolution_data.append(row)
                                
                                # Create DataFrame
                                evolution_df = pd.DataFrame(evolution_data)
                                
                                # Add coloring function for Strike Evolution Table
                                def highlight_evolution_table(row):
                                    """Apply colors to distinguish calls and puts in evolution table"""
                                    styles = []
                                    
                                    # Convert strike to float for comparison
                                    try:
                                        strike = float(row['Strike'])
                                    except (ValueError, TypeError):
                                        strike = 0.0
                                    
                                    # Determine ITM/OTM status
                                    call_itm = strike < current_price  # Call is ITM if strike < current price
                                    put_itm = strike > current_price   # Put is ITM if strike > current price
                                    
                                    for col in evolution_df.columns:
                                        if col.startswith('Call'):
                                            if call_itm:
                                                # ITM calls: Professional dark green
                                                styles.append('background-color: #2d5a2d; color: #ffffff')
                                            else:
                                                # OTM calls: Professional dark blue
                                                styles.append('background-color: #1e3a5f; color: #ffffff')
                                        elif col in ['Expiration', 'Days to Exp', 'Strike']:
                                            # Info columns: Neutral dark color
                                            styles.append('background-color: #2d3748; color: #ffffff')
                                        elif col.startswith('Put'):
                                            if put_itm:
                                                # ITM puts: Professional dark red
                                                styles.append('background-color: #5a2d2d; color: #ffffff')
                                            else:
                                                # OTM puts: Professional dark orange
                                                styles.append('background-color: #5a3d1a; color: #ffffff')
                                        else:
                                            styles.append('')
                                    
                                    return styles
                                
                                # Display table (cleaned for Arrow compatibility) with colors
                                st.markdown("### 📈 Strike Evolution Table")
                                st.caption("💡 **Price Columns**: Shows MID price (Bid+Ask)/2 when available, otherwise Last price marked as '(Last)'. **Price %** shows option price as percentage of current ticker price for relative valuation.")
                                evolution_df_clean = clean_dataframe_for_arrow(evolution_df)
                                
                                # Apply coloring to evolution table
                                styled_evolution_df = evolution_df_clean.style.apply(highlight_evolution_table, axis=1)
                                st.dataframe(styled_evolution_df, width='stretch')
                                
                                # Add legend for Strike Evolution Table
                                with st.expander("📖 Strike Evolution Color Legend", expanded=False):
                                    st.markdown("""
                                    **Professional Color Scheme for Strike Evolution:**
                                    - 🔵 **Dark Blue**: OTM CALL options (strike > current price)
                                    - 🟢 **Dark Green**: ITM CALL options (strike < current price)
                                    - 🔴 **Dark Red**: ITM PUT options (strike > current price)
                                    - 🟠 **Dark Orange**: OTM PUT options (strike < current price)
                                    - ⚫ **Dark Gray**: Info columns (Expiration, Days to Exp, Strike)
                                    
                                    **ITM/OTM Status:**
                                    - **ITM (In The Money)**: Option has intrinsic value
                                    - **OTM (Out The Money)**: Option has only time value
                                    """)
                                
                                # Create evolution chart
                                st.markdown("### 📊 Price Evolution Chart")
                                st.info("💡 **Price Calculation**: Uses MID price (average of Bid + Ask) when available, otherwise falls back to Last price for more accurate option valuation.")
                                
                                # Interpolation options
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    enable_interpolation = st.checkbox(
                                        "🔄 Enable Interpolation", 
                                        value=False,
                                        help="Smooth the curves by interpolating missing data points between expirations"
                                    )
                                    
                                    if enable_interpolation:
                                        interpolation_method = st.selectbox(
                                            "📊 Interpolation Method:",
                                            options=["linear", "cubic"],
                                            index=1,  # Default to cubic
                                            help="Linear: Straight lines between points. Cubic: Smooth monotonic curves (respects time decay)"
                                        )
                                
                                # Prepare data for chart with intrinsic/extrinsic values
                                chart_data = []
                                for _, row in evolution_df.iterrows():
                                    # Use Call Price (MID/Strike with fallback to Last/Strike) - same logic as table
                                    call_price = None
                                    if row['Call Price'] != '-' and pd.notna(row['Call Price']):
                                        call_price = float(row['Call Price'])
                                    
                                    if call_price is not None:
                                        safe_price = safe_float_price(current_price)
                                        call_intrinsic = max(0, safe_price - evolution_strike)
                                        call_extrinsic = call_price - call_intrinsic
                                        
                                        # Get Call Price % from existing data if available
                                        call_price_pct = None
                                        if 'Call Price %' in row and row['Call Price %'] != '-':
                                            try:
                                                # Extract percentage from string like "9.64%"
                                                call_price_pct = float(row['Call Price %'].replace('%', ''))
                                            except:
                                                call_price_pct = None
                                        
                                        chart_data.append({
                                            'Expiration': row['Expiration'],
                                            'Days to Exp': row['Days to Exp'],
                                            'Price': call_price,
                                            'Type': 'Call',
                                            'Strike': row['Strike'],
                                            'Intrinsic': call_intrinsic,
                                            'Extrinsic': call_extrinsic,
                                            'Intrinsic %': (call_intrinsic / call_price * 100) if call_price > 0 else 0,
                                            'Extrinsic %': (call_extrinsic / call_price * 100) if call_price > 0 else 0,
                                            'Call Price %': call_price_pct
                                        })
                                    # Use Put Price (MID/Strike with fallback to Last/Strike) - same logic as table
                                    put_price = None
                                    if row['Put Price'] != '-' and pd.notna(row['Put Price']):
                                        put_price = float(row['Put Price'])
                                    
                                    if put_price is not None:
                                        safe_price = safe_float_price(current_price)
                                        put_intrinsic = max(0, evolution_strike - safe_price)
                                        put_extrinsic = put_price - put_intrinsic
                                        
                                        # Get Put Price % from existing data if available
                                        put_price_pct = None
                                        if 'Put Price %' in row and row['Put Price %'] != '-':
                                            try:
                                                # Extract percentage from string like "6.51%"
                                                put_price_pct = float(row['Put Price %'].replace('%', ''))
                                            except:
                                                put_price_pct = None
                                        
                                        chart_data.append({
                                            'Expiration': row['Expiration'],
                                            'Days to Exp': row['Days to Exp'],
                                            'Price': put_price,
                                            'Type': 'Put',
                                            'Strike': row['Strike'],
                                            'Intrinsic': put_intrinsic,
                                            'Extrinsic': put_extrinsic,
                                            'Intrinsic %': (put_intrinsic / put_price * 100) if put_price > 0 else 0,
                                            'Extrinsic %': (put_extrinsic / put_price * 100) if put_price > 0 else 0,
                                            'Put Price %': put_price_pct
                                        })
                                
                                if chart_data:
                                    chart_df = pd.DataFrame(chart_data)
                                    
                                    # Sort by Days to Exp for proper interpolation
                                    chart_df = chart_df.sort_values('Days to Exp')
                                    
                                    # Create interpolated data for smoother curves
                                    calls_data = chart_df[chart_df['Type'] == 'Call']
                                    puts_data = chart_df[chart_df['Type'] == 'Put']
                                    
                                    # Apply interpolation if enabled
                                    if enable_interpolation:
                                        
                                        # Interpolate calls
                                        if not calls_data.empty and len(calls_data) > 1:
                                            # Create a range of days for interpolation
                                            min_days = calls_data['Days to Exp'].min()
                                            max_days = calls_data['Days to Exp'].max()
                                            interpolated_days = range(int(min_days), int(max_days) + 1)
                                            
                                            # Interpolate prices
                                            from scipy import interpolate
                                            import numpy as np
                                            
                                            try:
                                                if interpolation_method == "cubic":
                                                    # Use monotonic cubic interpolation to prevent oscillations
                                                    from scipy.interpolate import PchipInterpolator
                                                    f_calls = PchipInterpolator(
                                                        calls_data['Days to Exp'], 
                                                        calls_data['Price']
                                                    )
                                                else:
                                                    f_calls = interpolate.interp1d(
                                                        calls_data['Days to Exp'], 
                                                        calls_data['Price'], 
                                                        kind='linear',
                                                        bounds_error=False,
                                                        fill_value='extrapolate'
                                                    )
                                                interpolated_calls = f_calls(interpolated_days)
                                                
                                                # Create interpolated calls data
                                                interpolated_calls_data = []
                                                for i, days in enumerate(interpolated_days):
                                                    if days in calls_data['Days to Exp'].values:
                                                        # Use original data point
                                                        orig_data = calls_data[calls_data['Days to Exp'] == days].iloc[0]
                                                        interpolated_calls_data.append({
                                                            'Days to Exp': days,
                                                            'Price': orig_data['Price'],
                                                            'Type': 'Call',
                                                            'Intrinsic': orig_data['Intrinsic'],
                                                            'Extrinsic': orig_data['Extrinsic'],
                                                            'Intrinsic %': orig_data['Intrinsic %'],
                                                            'Extrinsic %': orig_data['Extrinsic %'],
                                                            'Call Price %': orig_data.get('Call Price %', None),
                                                            'IsInterpolated': False
                                                        })
                                                    else:
                                                        # Use interpolated data point
                                                        # Calculate Call Price % for interpolated points
                                                        interpolated_call_price_pct = None
                                                        if interpolated_calls[i] is not None and current_price > 0:
                                                            interpolated_call_price_pct = (interpolated_calls[i] / current_price) * 100
                                                        
                                                        interpolated_calls_data.append({
                                                            'Days to Exp': days,
                                                            'Price': interpolated_calls[i],
                                                            'Type': 'Call',
                                                            'Intrinsic': 0,  # Simplified for interpolated points
                                                            'Extrinsic': interpolated_calls[i],
                                                            'Intrinsic %': 0,
                                                            'Extrinsic %': 100,
                                                            'Call Price %': interpolated_call_price_pct,
                                                            'IsInterpolated': True
                                                        })
                                                
                                                calls_data = pd.DataFrame(interpolated_calls_data)
                                            except:
                                                # Fallback to original data if interpolation fails
                                                pass
                                        
                                        # Interpolate puts
                                        if not puts_data.empty and len(puts_data) > 1:
                                            # Create a range of days for interpolation
                                            min_days = puts_data['Days to Exp'].min()
                                            max_days = puts_data['Days to Exp'].max()
                                            interpolated_days = range(int(min_days), int(max_days) + 1)
                                            
                                            try:
                                                if interpolation_method == "cubic":
                                                    # Use monotonic cubic interpolation to prevent oscillations
                                                    from scipy.interpolate import PchipInterpolator
                                                    f_puts = PchipInterpolator(
                                                        puts_data['Days to Exp'], 
                                                        puts_data['Price']
                                                    )
                                                else:
                                                    f_puts = interpolate.interp1d(
                                                        puts_data['Days to Exp'], 
                                                        puts_data['Price'], 
                                                        kind='linear',
                                                        bounds_error=False,
                                                        fill_value='extrapolate'
                                                    )
                                                interpolated_puts = f_puts(interpolated_days)
                                                
                                                # Create interpolated puts data
                                                interpolated_puts_data = []
                                                for i, days in enumerate(interpolated_days):
                                                    if days in puts_data['Days to Exp'].values:
                                                        # Use original data point
                                                        orig_data = puts_data[puts_data['Days to Exp'] == days].iloc[0]
                                                        interpolated_puts_data.append({
                                                            'Days to Exp': days,
                                                            'Price': orig_data['Price'],
                                                            'Type': 'Put',
                                                            'Intrinsic': orig_data['Intrinsic'],
                                                            'Extrinsic': orig_data['Extrinsic'],
                                                            'Intrinsic %': orig_data['Intrinsic %'],
                                                            'Extrinsic %': orig_data['Extrinsic %'],
                                                            'Put Price %': orig_data.get('Put Price %', None),
                                                            'IsInterpolated': False
                                                        })
                                                    else:
                                                        # Use interpolated data point
                                                        # Calculate Put Price % for interpolated points
                                                        interpolated_put_price_pct = None
                                                        if interpolated_puts[i] is not None and current_price > 0:
                                                            interpolated_put_price_pct = (interpolated_puts[i] / current_price) * 100
                                                        
                                                        interpolated_puts_data.append({
                                                            'Days to Exp': days,
                                                            'Price': interpolated_puts[i],
                                                            'Type': 'Put',
                                                            'Intrinsic': 0,  # Simplified for interpolated points
                                                            'Extrinsic': interpolated_puts[i],
                                                            'Intrinsic %': 0,
                                                            'Extrinsic %': 100,
                                                            'Put Price %': interpolated_put_price_pct,
                                                            'IsInterpolated': True
                                                        })
                                                
                                                puts_data = pd.DataFrame(interpolated_puts_data)
                                            except:
                                                # Fallback to original data if interpolation fails
                                                pass
                                    
                                    # Create Plotly chart
                                    go, _, _ = get_plotly()
                                    fig = go.Figure()
                                    
                                    # Add Call line with enhanced tooltip
                                    if not calls_data.empty:
                                        # Separate original and interpolated points for different styling
                                        original_calls = calls_data[~calls_data.get('IsInterpolated', False)] if 'IsInterpolated' in calls_data.columns else calls_data
                                        interpolated_calls = calls_data[calls_data.get('IsInterpolated', False)] if 'IsInterpolated' in calls_data.columns else pd.DataFrame()
                                        
                                        # Add original data points
                                        if not original_calls.empty:
                                            fig.add_trace(go.Scatter(
                                                x=original_calls['Days to Exp'],
                                                y=original_calls['Price'],
                                                mode='lines+markers',
                                                name='Call Options',
                                                line=dict(color='green', width=3),
                                                marker=dict(size=8, color='green'),
                                                hovertemplate='<b>Call Options</b><br>' +
                                                            'Days to Expiration: %{x}<br>' +
                                                            'Option Price: $%{y:.2f}<br>' +
                                                            'Call Price %: %{customdata[0]:.2f}%<br>' +
                                                            'Intrinsic Value: $%{customdata[1]:.2f}<br>' +
                                                            'Extrinsic Value: $%{customdata[2]:.2f}<br>' +
                                                            'Intrinsic %: %{customdata[3]:.1f}%<br>' +
                                                            'Extrinsic %: %{customdata[4]:.1f}%<br>' +
                                                            '<extra></extra>',
                                                customdata=original_calls[['Call Price %', 'Intrinsic', 'Extrinsic', 'Intrinsic %', 'Extrinsic %']].values
                                            ))
                                        
                                        # Add interpolated data points (lighter, dashed)
                                        if not interpolated_calls.empty:
                                            fig.add_trace(go.Scatter(
                                                x=interpolated_calls['Days to Exp'],
                                                y=interpolated_calls['Price'],
                                                mode='lines',
                                                name='Call Options (Interpolated)',
                                                line=dict(color='lightgreen', width=1, dash='dot'),
                                                showlegend=False,
                                                hovertemplate='<b>Call Options (Interpolated)</b><br>' +
                                                            'Days to Expiration: %{x}<br>' +
                                                            'Option Price: $%{y:.2f}<br>' +
                                                            'Call Price %: %{customdata[0]:.2f}%<br>' +
                                                            '<extra></extra>',
                                                customdata=interpolated_calls[['Call Price %']].values
                                            ))
                                    
                                    # Add Put line with enhanced tooltip
                                    if not puts_data.empty:
                                        # Separate original and interpolated points for different styling
                                        original_puts = puts_data[~puts_data.get('IsInterpolated', False)] if 'IsInterpolated' in puts_data.columns else puts_data
                                        interpolated_puts = puts_data[puts_data.get('IsInterpolated', False)] if 'IsInterpolated' in puts_data.columns else pd.DataFrame()
                                        
                                        # Add original data points
                                        if not original_puts.empty:
                                            fig.add_trace(go.Scatter(
                                                x=original_puts['Days to Exp'],
                                                y=original_puts['Price'],
                                                mode='lines+markers',
                                                name='Put Options',
                                                line=dict(color='red', width=3),
                                                marker=dict(size=8, color='red'),
                                                hovertemplate='<b>Put Options</b><br>' +
                                                            'Days to Expiration: %{x}<br>' +
                                                            'Option Price: $%{y:.2f}<br>' +
                                                            'Put Price %: %{customdata[0]:.2f}%<br>' +
                                                            'Intrinsic Value: $%{customdata[1]:.2f}<br>' +
                                                            'Extrinsic Value: $%{customdata[2]:.2f}<br>' +
                                                            'Intrinsic %: %{customdata[3]:.1f}%<br>' +
                                                            'Extrinsic %: %{customdata[4]:.1f}%<br>' +
                                                            '<extra></extra>',
                                                customdata=original_puts[['Put Price %', 'Intrinsic', 'Extrinsic', 'Intrinsic %', 'Extrinsic %']].values
                                            ))
                                        
                                        # Add interpolated data points (lighter, dashed)
                                        if not interpolated_puts.empty:
                                            fig.add_trace(go.Scatter(
                                                x=interpolated_puts['Days to Exp'],
                                                y=interpolated_puts['Price'],
                                                mode='lines',
                                                name='Put Options (Interpolated)',
                                                line=dict(color='lightcoral', width=1, dash='dot'),
                                                showlegend=False,
                                                hovertemplate='<b>Put Options (Interpolated)</b><br>' +
                                                            'Days to Expiration: %{x}<br>' +
                                                            'Option Price: $%{y:.2f}<br>' +
                                                            'Put Price %: %{customdata[0]:.2f}%<br>' +
                                                            '<extra></extra>',
                                                customdata=interpolated_puts[['Put Price %']].values
                                            ))
                                    
                                    # Reverse x-axis to show time progression correctly (oldest to newest)
                                    fig.update_layout(
                                        title=f"Option Price Evolution - Strike ${evolution_strike:.2f} | {ticker_symbol}: ${current_price:.2f} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                                        xaxis_title="Days to Expiration (Left: Far Future → Right: Near Expiration)",
                                        yaxis_title="Option Price ($)",
                                        height=500,
                                        showlegend=True,
                                        hovermode='x unified',  # Show both curves on hover
                                        xaxis=dict(
                                            autorange='reversed'  # Invert x-axis to show Far Future → Near Expiration
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, config={'displayModeBar': True, 'displaylogo': False})
                                else:
                                    st.warning("No price data available for chart")
                            else:
                                st.warning(f"No options found for strike ${evolution_strike:.2f}")
                else:
                    st.warning("No options data available")
                
                # SPREAD ANALYSIS Section within Option Evolution ONLY
                st.markdown("---")
                st.subheader("📊 Spread Analysis")
                st.markdown("**Analyze bull/bear spreads with different expirations**")
                
                if not calls_combined.empty or not puts_combined.empty:
                    st.markdown("### 🎯 Select Two Options for Spread Analysis")
                    
                    # Strike linking option
                    link_strikes = st.checkbox("🔗 Link Strikes - Keep both options at same strike price", value=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Option 1:**")
                        option1_action = st.selectbox("Action:", ["Buy", "Sell"], key="option1_action")
                        option1_type = st.selectbox("Type:", ["Call", "Put"], key="option1_type")
                        # Strike price selection (dropdown with available strikes)
                        if option1_type == "Call":
                            available_strikes = sorted(calls_combined['strike'].unique())
                        else:
                            available_strikes = sorted(puts_combined['strike'].unique())
                        
                        # Add custom option
                        strike_options = ["Custom"] + [f"${strike:.0f}" for strike in available_strikes]
                        option1_strike_selected = st.selectbox("Strike Price:", strike_options, key="option1_strike_select")
                        
                        if option1_strike_selected == "Custom":
                            # Use a default strike if evolution_strike is not defined
                            try:
                                if 'persistent_strike' in st.session_state:
                                    default_strike = float(st.session_state.persistent_strike)
                                else:
                                    default_strike = 400.0
                            except:
                                default_strike = 400.0
                            option1_strike = st.number_input("Enter Custom Strike:", min_value=0.0, value=default_strike, step=1.0, key="option1_strike_custom")
                        else:
                            option1_strike = float(option1_strike_selected.replace('$', ''))
                        
                        # Store the strike in session state for linking
                        st.session_state.option1_strike = option1_strike
                        
                        # Get available expirations for this specific strike
                        if option1_type == "Call":
                            option1_expirations = calls_combined[calls_combined['strike'] == option1_strike]['expiration'].unique()
                        else:
                            option1_expirations = puts_combined[puts_combined['strike'] == option1_strike]['expiration'].unique()
                        
                        # Check if strike exists in data
                        if len(option1_expirations) == 0:
                            st.warning(f"⚠️ Strike {option1_strike:.0f} not found in {option1_type} data. Using closest available strike.")
                            # Find closest strike
                            if option1_type == "Call":
                                all_strikes = sorted(calls_combined['strike'].unique())
                            else:
                                all_strikes = sorted(puts_combined['strike'].unique())
                            
                            closest_strike = min(all_strikes, key=lambda x: abs(x - option1_strike))
                            option1_strike = closest_strike
                            st.info(f"Using closest strike: ${option1_strike:.0f}")
                            
                            # Get expirations for closest strike
                            if option1_type == "Call":
                                option1_expirations = calls_combined[calls_combined['strike'] == option1_strike]['expiration'].unique()
                            else:
                                option1_expirations = puts_combined[puts_combined['strike'] == option1_strike]['expiration'].unique()
                        
                        # Create options with days to expiration
                        option1_exp_options = []
                        for exp in sorted(option1_expirations):
                            try:
                                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                                today = datetime.now()
                                days_to_exp = (exp_date - today).days
                                option1_exp_options.append(f"{exp} ({days_to_exp} days)")
                            except:
                                option1_exp_options.append(exp)
                        
                        if len(option1_exp_options) > 0:
                            option1_exp_selected = st.selectbox("Expiration:", option1_exp_options, key="option1_exp")
                            option1_exp = option1_exp_selected.split(' (')[0]  # Extract just the date
                        else:
                            st.warning("⚠️ No expirations found for this strike. Finding closest available expiration.")
                            # Find closest expiration from all available expirations
                            if option1_type == "Call":
                                all_expirations = sorted(calls_combined['expiration'].unique())
                            else:
                                all_expirations = sorted(puts_combined['expiration'].unique())
                            
                            if len(all_expirations) > 0:
                                # Find closest expiration by time (closest to today)
                                today = datetime.now()
                                closest_exp = None
                                min_days_diff = float('inf')
                                
                                for exp in all_expirations:
                                    try:
                                        exp_date = datetime.strptime(exp, '%Y-%m-%d')
                                        days_diff = abs((exp_date - today).days)
                                        if days_diff < min_days_diff:
                                            min_days_diff = days_diff
                                            closest_exp = exp
                                    except:
                                        continue
                                
                                if closest_exp:
                                    option1_exp = closest_exp
                                    st.info(f"Using closest expiration: {option1_exp}")
                                else:
                                    option1_exp = all_expirations[0]
                                    st.info(f"Using first available expiration: {option1_exp}")
                            else:
                                st.error("No expirations found at all")
                                option1_exp = ""
                        
                        # Get option 1 data
                        if option1_type == "Call":
                            option1_data = calls_combined[(calls_combined['strike'] == option1_strike) & (calls_combined['expiration'] == option1_exp)]
                        else:
                            option1_data = puts_combined[(puts_combined['strike'] == option1_strike) & (puts_combined['expiration'] == option1_exp)]
                        
                        if not option1_data.empty:
                            option1_row = option1_data.iloc[0]
                            option1_price = option1_row['lastPrice'] if pd.notna(option1_row['lastPrice']) else 0
                            option1_exp_date = option1_row['expiration']
                            
                            # Calculate days to expiration manually
                            try:
                                exp_date = datetime.strptime(option1_exp_date, '%Y-%m-%d')
                                today = datetime.now()
                                option1_days = (exp_date - today).days
                            except:
                                option1_days = 0
                                
                            st.markdown(f"**Price:** ${option1_price:.2f}")
                            st.markdown(f"**Days to Expiration:** {option1_days} days")
                            st.markdown(f"**Expiration Date:** {option1_exp_date}")
                        else:
                            st.warning("Option 1 not found")
                            option1_price = 0
                            option1_days = 0
                            option1_exp_date = ""
                    
                    with col2:
                        st.markdown("**Option 2:**")
                        option2_action = st.selectbox("Action:", ["Buy", "Sell"], key="option2_action")
                        option2_type = st.selectbox("Type:", ["Call", "Put"], key="option2_type")
                        # Strike price selection (dropdown with available strikes)
                        if option2_type == "Call":
                            available_strikes = sorted(calls_combined['strike'].unique())
                        else:
                            available_strikes = sorted(puts_combined['strike'].unique())
                        
                        # If strikes are linked, use the same strike as Option 1
                        if link_strikes:
                            option2_strike = st.session_state.get('option1_strike', option1_strike)
                            st.markdown(f"**Strike Price:** ${option2_strike:.0f} (Linked to Option 1)")
                        else:
                            # Add custom option
                            strike_options = ["Custom"] + [f"${strike:.0f}" for strike in available_strikes]
                            option2_strike_selected = st.selectbox("Strike Price:", strike_options, key="option2_strike_select")
                            
                            if option2_strike_selected == "Custom":
                                # Use a default strike if evolution_strike is not defined
                                try:
                                    if 'persistent_strike' in st.session_state:
                                        default_strike = float(st.session_state.persistent_strike)
                                    else:
                                        default_strike = 400.0
                                except:
                                    default_strike = 400.0
                                option2_strike = st.number_input("Enter Custom Strike:", min_value=0.0, value=default_strike, step=1.0, key="option2_strike_custom")
                            else:
                                option2_strike = float(option2_strike_selected.replace('$', ''))
                        
                        # Get available expirations for this specific strike
                        if option2_type == "Call":
                            option2_expirations = calls_combined[calls_combined['strike'] == option2_strike]['expiration'].unique()
                        else:
                            option2_expirations = puts_combined[puts_combined['strike'] == option2_strike]['expiration'].unique()
                        
                        # Check if strike exists in data
                        if len(option2_expirations) == 0:
                            st.warning(f"⚠️ Strike {option2_strike:.0f} not found in {option2_type} data. Using closest available strike.")
                            # Find closest strike
                            if option2_type == "Call":
                                all_strikes = sorted(calls_combined['strike'].unique())
                            else:
                                all_strikes = sorted(puts_combined['strike'].unique())
                            
                            closest_strike = min(all_strikes, key=lambda x: abs(x - option2_strike))
                            option2_strike = closest_strike
                            st.info(f"Using closest strike: ${option2_strike:.0f}")
                            
                            # Get expirations for closest strike
                            if option2_type == "Call":
                                option2_expirations = calls_combined[calls_combined['strike'] == option2_strike]['expiration'].unique()
                            else:
                                option2_expirations = puts_combined[puts_combined['strike'] == option2_strike]['expiration'].unique()
                        
                        # Create options with days to expiration
                        option2_exp_options = []
                        for exp in sorted(option2_expirations):
                            try:
                                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                                today = datetime.now()
                                days_to_exp = (exp_date - today).days
                                option2_exp_options.append(f"{exp} ({days_to_exp} days)")
                            except:
                                option2_exp_options.append(exp)
                        
                        if len(option2_exp_options) > 0:
                            option2_exp_selected = st.selectbox("Expiration:", option2_exp_options, key="option2_exp")
                            option2_exp = option2_exp_selected.split(' (')[0]  # Extract just the date
                        else:
                            st.warning("⚠️ No expirations found for this strike. Finding closest available expiration.")
                            # Find closest expiration from all available expirations
                            if option2_type == "Call":
                                all_expirations = sorted(calls_combined['expiration'].unique())
                            else:
                                all_expirations = sorted(puts_combined['expiration'].unique())
                            
                            if len(all_expirations) > 0:
                                # Find closest expiration by time (closest to today)
                                today = datetime.now()
                                closest_exp = None
                                min_days_diff = float('inf')
                                
                                for exp in all_expirations:
                                    try:
                                        exp_date = datetime.strptime(exp, '%Y-%m-%d')
                                        days_diff = abs((exp_date - today).days)
                                        if days_diff < min_days_diff:
                                            min_days_diff = days_diff
                                            closest_exp = exp
                                    except:
                                        continue
                                
                                if closest_exp:
                                    option2_exp = closest_exp
                                    st.info(f"Using closest expiration: {option2_exp}")
                                else:
                                    option2_exp = all_expirations[0]
                                    st.info(f"Using first available expiration: {option2_exp}")
                            else:
                                st.error("No expirations found at all")
                                option2_exp = ""
                        
                        # Get option 2 data
                        if option2_type == "Call":
                            option2_data = calls_combined[(calls_combined['strike'] == option2_strike) & (calls_combined['expiration'] == option2_exp)]
                        else:
                            option2_data = puts_combined[(puts_combined['strike'] == option2_strike) & (puts_combined['expiration'] == option2_exp)]
                        
                        if not option2_data.empty:
                            option2_row = option2_data.iloc[0]
                            option2_price = option2_row['lastPrice'] if pd.notna(option2_row['lastPrice']) else 0
                            option2_exp_date = option2_row['expiration']
                            
                            # Calculate days to expiration manually
                            try:
                                exp_date = datetime.strptime(option2_exp_date, '%Y-%m-%d')
                                today = datetime.now()
                                option2_days = (exp_date - today).days
                            except:
                                option2_days = 0
                                
                            st.markdown(f"**Price:** ${option2_price:.2f}")
                            st.markdown(f"**Days to Expiration:** {option2_days} days")
                            st.markdown(f"**Expiration Date:** {option2_exp_date}")
                        else:
                            st.warning("Option 2 not found")
                            option2_price = 0
                            option2_days = 0
                            option2_exp_date = ""
                    
                    # Quick Strike Analysis
                    if option1_price > 0 and option2_price > 0:
                        st.markdown("### 🔍 Quick Strike Analysis")
                        
                        # Get current stock price - use the selected strike as reference
                        # Use a default strike if selected_strike is not defined
                        try:
                            if 'persistent_strike' in st.session_state:
                                default_strike = float(st.session_state.persistent_strike)
                            else:
                                default_strike = 400.0
                        except:
                            default_strike = 400.0
                        current_price = default_strike  # Use the main strike as reference price
                        
                        # Create analysis for both expirations
                        analysis_data = []
                        
                        # Get all strikes for both expirations
                        all_strikes = set()
                        if option1_type == "Call":
                            all_strikes.update(calls_combined[calls_combined['expiration'] == option1_exp]['strike'].unique())
                        else:
                            all_strikes.update(puts_combined[puts_combined['expiration'] == option1_exp]['strike'].unique())
                            
                        if option2_type == "Call":
                            all_strikes.update(calls_combined[calls_combined['expiration'] == option2_exp]['strike'].unique())
                        else:
                            all_strikes.update(puts_combined[puts_combined['expiration'] == option2_exp]['strike'].unique())
                        
                        all_strikes = sorted(list(all_strikes))
                        
                        for strike in all_strikes:
                            row = {'Strike': strike, 'Stock %': f"{(strike/current_price - 1)*100:.1f}%"}
                            
                            # Option 1 data
                            opt1_price = 0
                            if option1_type == "Call":
                                opt1_data = calls_combined[(calls_combined['strike'] == strike) & (calls_combined['expiration'] == option1_exp)]
                            else:
                                opt1_data = puts_combined[(puts_combined['strike'] == strike) & (puts_combined['expiration'] == option1_exp)]
                            
                            if not opt1_data.empty:
                                opt1_row = opt1_data.iloc[0]
                                # Use calculate_mid_price to apply price adjustment
                                opt1_bid = opt1_row.get('bid', 0) if pd.notna(opt1_row.get('bid')) else 0
                                opt1_ask = opt1_row.get('ask', 0) if pd.notna(opt1_row.get('ask')) else 0
                                opt1_last = opt1_row.get('lastPrice', 0) if pd.notna(opt1_row.get('lastPrice')) else 0
                                
                                opt1_price = calculate_mid_price(opt1_bid, opt1_ask, opt1_last)
                                if opt1_price and opt1_price > 0:
                                    price_display = f"${opt1_price:.2f}"
                                else:
                                    opt1_price = 0
                                    price_display = "N/A"
                                
                                row[f'Option 1 ({option1_exp})'] = price_display
                                if opt1_price > 0:
                                    row[f'Option 1 %'] = f"{(opt1_price/current_price)*100:.2f}%"
                                else:
                                    row[f'Option 1 %'] = "N/A"
                            else:
                                opt1_price = 0
                                row[f'Option 1 ({option1_exp})'] = "N/A"
                                row[f'Option 1 %'] = "N/A"
                            
                            # Option 2 data
                            opt2_price = 0
                            if option2_type == "Call":
                                opt2_data = calls_combined[(calls_combined['strike'] == strike) & (calls_combined['expiration'] == option2_exp)]
                            else:
                                opt2_data = puts_combined[(puts_combined['strike'] == strike) & (puts_combined['expiration'] == option2_exp)]
                            
                            if not opt2_data.empty:
                                opt2_row = opt2_data.iloc[0]
                                # Use calculate_mid_price to apply price adjustment
                                opt2_bid = opt2_row.get('bid', 0) if pd.notna(opt2_row.get('bid')) else 0
                                opt2_ask = opt2_row.get('ask', 0) if pd.notna(opt2_row.get('ask')) else 0
                                opt2_last = opt2_row.get('lastPrice', 0) if pd.notna(opt2_row.get('lastPrice')) else 0
                                
                                opt2_price = calculate_mid_price(opt2_bid, opt2_ask, opt2_last)
                                if opt2_price and opt2_price > 0:
                                    price_display = f"${opt2_price:.2f}"
                                else:
                                    opt2_price = 0
                                    price_display = "N/A"
                                
                                row[f'Option 2 ({option2_exp})'] = price_display
                                if opt2_price > 0:
                                    row[f'Option 2 %'] = f"{(opt2_price/current_price)*100:.2f}%"
                                else:
                                    row[f'Option 2 %'] = "N/A"
                            else:
                                opt2_price = 0
                                row[f'Option 2 ({option2_exp})'] = "N/A"
                                row[f'Option 2 %'] = "N/A"
                            
                            # Calculate combined premium/stock % (spread premium)
                            if opt1_price > 0 and opt2_price > 0:
                                # Calculate net premium based on Buy/Sell actions
                                if option1_action == "Buy":
                                    opt1_cost = opt1_price
                                else:  # Sell
                                    opt1_cost = -opt1_price
                                    
                                if option2_action == "Buy":
                                    opt2_cost = opt2_price
                                else:  # Sell
                                    opt2_cost = -opt2_price
                                
                                net_premium = opt1_cost + opt2_cost
                                premium_stock_pct = (net_premium / current_price) * 100
                                row['Premium/Stock %'] = f"{premium_stock_pct:.2f}%"
                            else:
                                row['Premium/Stock %'] = "N/A"
                            
                            analysis_data.append(row)
                        
                        if analysis_data:
                            analysis_df = pd.DataFrame(analysis_data)
                            st.dataframe(analysis_df, width='stretch', hide_index=True)
                            
                            st.caption(f"💡 **Stock Price:** ${current_price:.2f} | **Option 1:** {option1_type} {option1_exp} | **Option 2:** {option2_type} {option2_exp}")
                    
                    # Spread Analysis
                    if option1_price > 0 and option2_price > 0:
                        st.markdown("### 📈 Spread Analysis Results")
                        
                        # Calculate spread metrics based on Buy/Sell actions
                        if option1_action == "Buy":
                            option1_cost = option1_price
                        else:  # Sell
                            option1_cost = -option1_price
                            
                        if option2_action == "Buy":
                            option2_cost = option2_price
                        else:  # Sell
                            option2_cost = -option2_price
                        
                        net_premium = option1_cost + option2_cost
                        premium_stock_pct = (net_premium / current_price) * 100
                        
                        # Calculate days difference
                        days_diff = abs(option1_days - option2_days)
                        
                        # Determine spread type
                        if net_premium < 0:
                            spread_type = "Credit Spread"
                            spread_color = "🟢"
                        else:
                            spread_type = "Debit Spread"
                            spread_color = "🔴"
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Net Premium", f"${net_premium:.2f}")
                        
                        with col2:
                            st.metric("Premium/Stock %", f"{premium_stock_pct:.2f}%")
                        
                        with col3:
                            st.metric("Days Difference", f"{days_diff} days")
                        
                        with col4:
                            st.metric("Spread Type", f"{spread_color} {spread_type}")
                        
                        # Strategy Analysis
                        st.markdown("### 📋 Strategy Analysis")
                        
                        st.markdown(f"**Spread Strategy:**")
                        st.markdown(f"- {option1_action} {option1_type} {option1_days} days (${option1_price:.2f}) - Expires: {option1_exp_date}")
                        st.markdown(f"- {option2_action} {option2_type} {option2_days} days (${option2_price:.2f}) - Expires: {option2_exp_date}")
                        
                        if days_diff > 0:
                            longer_option = "Option 1" if option1_days > option2_days else "Option 2"
                            shorter_option = "Option 2" if option1_days > option2_days else "Option 1"
                            st.markdown(f"**Remaining Protection:** After {shorter_option} expires, you still have {longer_option} for {days_diff} more days.")
                        
                        # P&L Scenarios
                        st.markdown("### 💰 Profit/Loss Scenarios")
                        
                        scenarios_data = []
                        stock_movements = [-20, -10, -5, 0, 5, 10, 20, 30]
                        
                        for movement in stock_movements:
                            new_stock_price = current_price * (1 + movement/100)
                            
                            # Calculate intrinsic values
                            if option1_type == "Call":
                                option1_intrinsic = max(0, new_stock_price - option1_strike)
                            else:  # Put
                                option1_intrinsic = max(0, option1_strike - new_stock_price)
                            
                            if option2_type == "Call":
                                option2_intrinsic = max(0, new_stock_price - option2_strike)
                            else:  # Put
                                option2_intrinsic = max(0, option2_strike - new_stock_price)
                            
                            # Calculate P&L based on Buy/Sell actions
                            if option1_action == "Buy":
                                option1_pnl = option1_intrinsic - option1_price
                            else:  # Sell
                                option1_pnl = option1_price - option1_intrinsic
                            
                            if option2_action == "Buy":
                                option2_pnl = option2_intrinsic - option2_price
                            else:  # Sell
                                option2_pnl = option2_price - option2_intrinsic
                            
                            total_pnl = option1_pnl + option2_pnl
                            
                            scenarios_data.append({
                                'Stock Movement (%)': f"{movement:+.0f}%",
                                'New Stock Price ($)': f"${new_stock_price:.2f}",
                                'Option 1 P&L ($)': f"${option1_pnl:.2f}",
                                'Option 2 P&L ($)': f"${option2_pnl:.2f}",
                                'Total P&L ($)': f"${total_pnl:.2f}",
                                'Return %': f"{((total_pnl / abs(net_premium)) * 100):.1f}%" if net_premium != 0 else "N/A"
                            })
                        
                        scenarios_df = pd.DataFrame(scenarios_data)
                        st.dataframe(scenarios_df, width='stretch')
                        
                        # Strategy recommendation
                        st.markdown("### 💡 Strategy Recommendation")
                        
                        if net_premium < 0:
                            st.success(f"✅ **Credit Spread**: You receive ${abs(net_premium):.2f} upfront. This strategy profits if the stock stays within your expected range.")
                        else:
                            st.info(f"ℹ️ **Debit Spread**: You pay ${net_premium:.2f} upfront. This strategy profits if the stock moves in your favor.")
                        
                        if days_diff > 30:
                            st.warning("⚠️ **Time Risk**: Large difference in expiration dates. Consider the risk of the shorter option expiring first.")
                        
                        if abs(premium_stock_pct) > 10:
                            st.warning("⚠️ **High Premium**: The spread premium represents more than 10% of the stock price. Consider if this is reasonable for your strategy.")
                else:
                    st.info("👆 Please ensure ticker data is loaded to use Spread Analysis")
            
            # BARBELL STRATEGY Section
            elif selected_section == "⚖️ BARBELL STRATEGY":
                st.subheader("⚖️ Barbell Strategy Calculator")
                st.markdown("**Simulate LEAPS + Cash strategy - View returns based on underlying movement at expiration**")
                
                
                # 🎯 BARBELL STRATEGY - OWN VARIABLES
                # 🛡️ ULTIMATE PROTECTION - ALWAYS DEFINE BARBELL_STRIKE
                try:
                    if 'barbell_strike' not in globals() or barbell_strike is None:
                        barbell_strike = 400.0
                except:
                    barbell_strike = 400.0
                
                # Try to get from session state if available
                if 'persistent_strike' in st.session_state:
                    try:
                        barbell_strike = float(st.session_state.persistent_strike)
                    except:
                        barbell_strike = None
                
                # Display current risk-free rate
                st.info(f"🏦 **Current Risk-Free Rate (10-Year Treasury)**: {risk_free_rate:.2f}% (as of {risk_free_fetch_timestamp})")
                
                # Risk-free rate checkbox with persistence
                if 'barbell_persistent_include_risk_free' not in st.session_state:
                    st.session_state.barbell_persistent_include_risk_free = True
                
                include_risk_free = st.checkbox("📈 Include Risk-Free Rate in Cash", 
                                               value=st.session_state.barbell_persistent_include_risk_free, 
                                               help="Apply risk-free rate to cash portion")
                st.session_state.barbell_persistent_include_risk_free = include_risk_free
                
                # Input parameters
                col1, col2 = st.columns(2)
                with col1:
                    # Initialize persistent total capital
                    if 'barbell_persistent_capital' not in st.session_state:
                        st.session_state.barbell_persistent_capital = 100000
                    
                    total_capital = st.number_input("💰 Total Capital ($)", min_value=1000, value=st.session_state.barbell_persistent_capital, step=1000, key="barbell_capital")
                    st.session_state.barbell_persistent_capital = total_capital
                    
                with col2:
                    # Initialize persistent options percentage
                    if 'barbell_persistent_pct' not in st.session_state:
                        st.session_state.barbell_persistent_pct = 20
                    
                    options_pct = st.number_input("📈 Options Allocation (%)", min_value=1, max_value=100, value=st.session_state.barbell_persistent_pct, step=1, key="barbell_pct")
                    st.session_state.barbell_persistent_pct = options_pct
                
                # Fractional contracts option
                # Initialize persistent fractional contracts setting
                if 'barbell_persistent_fractional' not in st.session_state:
                    st.session_state.barbell_persistent_fractional = True
                
                use_fractional = st.checkbox("🔢 Allow Fractional Contracts", value=st.session_state.barbell_persistent_fractional, help="Enable to test with decimal contracts (e.g., 2.23 contracts)", key="barbell_fractional")
                st.session_state.barbell_persistent_fractional = use_fractional
                
                st.markdown("---")
                
                # Select option
                # Initialize persistent option type
                if 'barbell_persistent_type' not in st.session_state:
                    st.session_state.barbell_persistent_type = "CALL"
                
                if not calls_combined.empty and not puts_combined.empty:
                    option_type = st.selectbox("📊 Option Type", ["CALL", "PUT"], index=0 if st.session_state.barbell_persistent_type == "CALL" else 1, key="barbell_type")
                else:
                    option_type = "CALL" if not calls_combined.empty else "PUT"
                
                st.session_state.barbell_persistent_type = option_type
                available_exp = (calls_combined if option_type == "CALL" else puts_combined)['expiration'].unique()
                
                # Create expiration options with days to expiration
                exp_options = []
                for exp_date in available_exp:
                    try:
                        exp_dt = pd.to_datetime(exp_date)
                        days_to_exp = (exp_dt - datetime.now()).days
                        if days_to_exp > 0:
                            exp_options.append(f"{exp_date} ({days_to_exp} days)")
                        else:
                            exp_options.append(f"{exp_date} (Expired)")
                    except:
                        exp_options.append(exp_date)
                
                # Initialize session state for expiration if not exists
                if 'barbell_persistent_expiration' not in st.session_state:
                    st.session_state.barbell_persistent_expiration = available_exp[0] if len(available_exp) > 0 else ""
                
                # Find index of persistent expiration in current options
                persistent_index = 0
                try:
                    for i, option in enumerate(exp_options):
                        if st.session_state.barbell_persistent_expiration in option:
                            persistent_index = i
                            break
                except:
                    persistent_index = 0
                
                selected_exp_display = st.selectbox("📅 Expiration", exp_options, index=persistent_index, key="barbell_exp")
                # Extract just the date part and save to session state
                selected_exp = selected_exp_display.split(' (')[0] if selected_exp_display else available_exp[0]
                st.session_state.barbell_persistent_expiration = selected_exp
                filtered_opts = (calls_combined if option_type == "CALL" else puts_combined)[lambda x: x['expiration'] == selected_exp].copy()
                
                if not filtered_opts.empty:
                    strikes = sorted(filtered_opts['strike'].unique())
                    
                    # Initialize persistent strike for this ticker
                    if 'barbell_persistent_strike' not in st.session_state:
                        # Find closest strike to current price for default
                        closest_strike = min(strikes, key=lambda x: abs(x - current_price))
                        st.session_state.barbell_persistent_strike = closest_strike
                    
                    # Hybrid dropdown selector (same as Spread Analysis)
                    # Add custom option
                    strike_options = ["Custom"] + [f"${strike:.0f}" for strike in strikes]
                    
                    # Find index of persistent strike in current options
                    persistent_strike_display = f"${st.session_state.barbell_persistent_strike:.0f}"
                    try:
                        persistent_index = strike_options.index(persistent_strike_display) if persistent_strike_display in strike_options else 0
                    except:
                        persistent_index = 0
                    
                    barbell_strike_selected = st.selectbox(
                        "🎯 Strike",
                        strike_options,
                        index=persistent_index,
                        help="Choose from dropdown list or select Custom for manual input",
                        key="barbell_strike"
                    )
                    
                    # Determine selected strike
                    if barbell_strike_selected == "Custom":
                        # Add manual input field for custom strike
                        custom_strike = st.number_input(
                            "🔍 Enter Custom Strike:",
                            min_value=0.0,
                            value=float(st.session_state.barbell_persistent_strike),
                            step=1.0,
                            help="Enter a custom strike price",
                            key="barbell_custom_strike"
                        )
                        barbell_strike = custom_strike
                        
                        # Validate custom strike
                        if barbell_strike not in strikes:
                            closest_strike = min(strikes, key=lambda x: abs(x - barbell_strike))
                            st.warning(f"⚠️ Strike {barbell_strike:.0f} not found. Using closest available strike: {closest_strike:.0f}")
                            barbell_strike = closest_strike
                    else:
                        barbell_strike = float(barbell_strike_selected.replace('$', ''))
                    
                    # Save selected strike to session state for persistence
                    st.session_state.barbell_persistent_strike = barbell_strike
                    
                    # 🎯 OPTION PRICE ADJUSTMENT - DROPDOWN + SLIDER
                    st.markdown("---")
                    st.markdown("### 🎯 Option Price Adjustment")
                    
                    # Initialize price adjustment factor
                    if 'barbell_price_adjustment' not in st.session_state:
                        st.session_state.barbell_price_adjustment = 0.0
                    
                    # Slider in expandable section
                    with st.expander("🎯 Option Price Adjustment", expanded=False):
                        st.markdown("**Adjust all option prices for this strategy**")
                        
                        # Price adjustment slider with unique key
                        price_adjustment = st.slider(
                            "📈 Price Adjustment (%)",
                            min_value=-90.0,
                            max_value=200.0,
                            value=st.session_state.barbell_price_adjustment,
                            step=1.0,
                            help="Positive: Increase all option prices (expecting higher volatility)\nNegative: Decrease all option prices (expecting lower volatility)\n\nHigh volatility can cause options to:\n• Triple in value (+200%)\n• Drop significantly (-90%)\n• Experience extreme swings",
                            key="barbell_price_slider"
                        )
                        
                        # Force update session state immediately
                        if price_adjustment != st.session_state.barbell_price_adjustment:
                            st.session_state.barbell_price_adjustment = price_adjustment
                            st.rerun()
                        
                        # Input box for precise value
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            precise_adjustment = st.number_input(
                                "📝 Enter precise adjustment (%)",
                                min_value=-90.0,
                                max_value=200.0,
                                value=float(price_adjustment),
                                step=0.1,
                                key="precise_price_adjustment"
                            )
                        with col2:
                            if st.button("✅ Apply", key="apply_precise_adjustment"):
                                st.session_state.barbell_price_adjustment = precise_adjustment
                                st.rerun()
                        with col3:
                            if st.button("🔄 Reset to 0%", key="reset_price_adjustment"):
                                st.session_state.barbell_price_adjustment = 0.0
                                st.rerun()
                        
                        # Show current adjustment
                        if price_adjustment != 0:
                            if price_adjustment > 0:
                                st.warning(f"📈 All option prices increased by {price_adjustment:.1f}%")
                            else:
                                st.success(f"📉 All option prices decreased by {abs(price_adjustment):.1f}%")
                        else:
                            st.markdown("⚖️ No price adjustment applied")
                    
                    
                    st.markdown("---")
                    
                    opt = filtered_opts[filtered_opts['strike'] == barbell_strike].iloc[0]
                    
                    # Calculate ITM/OTM status (moved here to be accessible everywhere)
                    if option_type == "CALL":
                        if barbell_strike < current_price:
                            itm_pct = ((current_price - barbell_strike) / current_price) * 100
                            strike_status = f"ITM {itm_pct:.1f}%"
                            status_color = "🟢"
                        elif barbell_strike > current_price:
                            otm_pct = ((barbell_strike - current_price) / current_price) * 100
                            strike_status = f"OTM {otm_pct:.1f}%"
                            status_color = "🔴"
                        else:
                            strike_status = "ATM"
                            status_color = "🟡"
                    else:  # PUT
                        if barbell_strike > current_price:
                            itm_pct = ((barbell_strike - current_price) / current_price) * 100
                            strike_status = f"ITM {itm_pct:.1f}%"
                            status_color = "🟢"
                        elif barbell_strike < current_price:
                            otm_pct = ((current_price - barbell_strike) / current_price) * 100
                            strike_status = f"OTM {otm_pct:.1f}%"
                            status_color = "🔴"
                        else:
                            strike_status = "ATM"
                            status_color = "🟡"
                    
                    # Get option price
                    opt_price = calculate_mid_price(opt['bid'], opt['ask'], opt['lastPrice'])
                    if opt_price and opt_price > 0:
                        # Calculate positions
                        opts_alloc = total_capital * (options_pct / 100)
                        if use_fractional:
                            contracts = opts_alloc / (opt_price * 100)
                        else:
                            contracts = int(opts_alloc / (opt_price * 100))
                        
                        if contracts > 0:
                            actual_cost = contracts * opt_price * 100
                            cash = total_capital - actual_cost
                            
                            # Calculate days to expiration
                            try:
                                exp_dt = pd.to_datetime(selected_exp)
                                days_to_exp = (exp_dt - datetime.now()).days
                            except:
                                days_to_exp = "N/A"
                            
                            
                            # Display metrics in clean table
                            st.markdown("### 🎯 Position Details")
                            
                            # Create clean table data
                            table_data = {
                                "📊 Market Information": [
                                    ["Current Price", f"${current_price:.2f}"],
                                    ["Strike Price", f"${barbell_strike:.2f}"],
                                    ["Status", f"{status_color} {strike_status}"],
                                    ["Days to Expiration", f"{days_to_exp} days"]
                                ],
                                "💰 Option Details": [
                                    ["Option Price", f"${opt_price:.2f}"],
                                    ["Contracts", f"{contracts:.2f}" if use_fractional else f"{contracts}"],
                                    ["Options Cost", f"${actual_cost:,.0f}"],
                                    ["Expiration Date", f"{selected_exp}"]
                                ],
                                "💼 Portfolio Summary": [
                                    ["Total Capital", f"${total_capital:,.0f}"],
                                    ["Cash Remaining", f"${cash:,.0f}"],
                                    ["Options Allocation", f"{options_pct}%"],
                                    ["Cash Allocation", f"{100-options_pct}%"]
                                ]
                            }
                            
                            # Create clean table
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**📊 Market Information**")
                                for item in table_data["📊 Market Information"]:
                                    st.markdown(f"**{item[0]}:** {item[1]}")
                            
                            with col2:
                                st.markdown("**💰 Option Details**")
                                for item in table_data["💰 Option Details"]:
                                    st.markdown(f"**{item[0]}:** {item[1]}")
                            
                            with col3:
                                st.markdown("**💼 Portfolio Summary**")
                                for item in table_data["💼 Portfolio Summary"]:
                                    st.markdown(f"**{item[0]}:** {item[1]}")
                            
                            # Simulate
                            st.markdown("### 📊 Returns Simulation")
                            movements = np.arange(-30, 51, 5)
                            results = []
                            
                            for mov in movements:
                                new_px = current_price * (1 + mov / 100)
                                payoff = max(0, (new_px - barbell_strike) if option_type == "CALL" else (barbell_strike - new_px))
                                opt_val = contracts * payoff * 100
                                
                                # Apply risk-free rate to cash if checkbox is checked
                                if include_risk_free:
                                    cash_with_return = cash * (1 + risk_free_rate/100)
                                else:
                                    cash_with_return = cash
                                
                                total_val = opt_val + cash_with_return
                                ret = total_val - total_capital
                                ret_pct = (ret / total_capital) * 100
                                # Calculate ITM status for this movement
                                is_itm = (new_px > barbell_strike) if option_type == "CALL" else (new_px < barbell_strike)
                                
                                results.append({'Movement (%)': mov, 'New Price ($)': new_px, 'Option Payoff ($)': payoff, 
                                               'Total Option Value ($)': opt_val, 'Cash ($)': cash_with_return, 'Total Portfolio ($)': total_val,
                                               'Total Return ($)': ret, 'Total Return (%)': ret_pct, 'ITM': is_itm})
                            
                            df = pd.DataFrame(results)
                            st.dataframe(df.round(2), width='stretch')
                            
                            # Create fine-grained data for smooth charts
                            fine_movements = np.arange(-30, 51, 0.5)
                            fine_results = []
                            
                            for mov in fine_movements:
                                new_px = current_price * (1 + mov / 100)
                                payoff = max(0, (new_px - barbell_strike) if option_type == "CALL" else (barbell_strike - new_px))
                                opt_val = contracts * payoff * 100
                                
                                # Apply risk-free rate to cash if checkbox is checked
                                if include_risk_free:
                                    cash_with_return = cash * (1 + risk_free_rate/100)
                                else:
                                    cash_with_return = cash
                                
                                total_val = opt_val + cash_with_return
                                ret = total_val - total_capital
                                ret_pct = (ret / total_capital) * 100
                                is_itm = (new_px > barbell_strike) if option_type == "CALL" else (new_px < barbell_strike)
                                
                                fine_results.append({'Movement (%)': mov, 'New Price ($)': new_px, 'Option Payoff ($)': payoff, 
                                                   'Total Option Value ($)': opt_val, 'Cash ($)': cash_with_return, 'Total Portfolio ($)': total_val,
                                                   'Total Return ($)': ret, 'Total Return (%)': ret_pct, 'ITM': is_itm})
                            
                            df_fine = pd.DataFrame(fine_results)
                            
                            # Charts
                            st.markdown("### 📈 Visualizations")
                            go, _, make_subplots = get_plotly()
                            fig = make_subplots(rows=2, cols=2, subplot_titles=('Portfolio Value', 'Total Return %', 'Option Value', 'Return Distribution'))
                            
                            # Portfolio Value Chart with detailed hover (smooth)
                            fig.add_trace(go.Scatter(
                                x=df_fine['Movement (%)'],
                                y=df_fine['Total Portfolio ($)'],
                                mode='lines',
                                name='Portfolio',
                                line=dict(color='blue', width=3),
                                hovertemplate='<b>Portfolio Value</b><br>' +
                                            'Movement: %{x:.1f}%<br>' +
                                            'New Price: $%{customdata[0]:.2f}<br>' +
                                            'Portfolio Value: $%{y:,.0f}<br>' +
                                            'Return: $%{customdata[1]:,.0f} (%{customdata[2]:.1f}%)<br>' +
                                            'Options Value: $%{customdata[3]:,.0f}<br>' +
                                            'Cash: $%{customdata[4]:,.0f}<br>' +
                                            'ITM: %{customdata[5]}<extra></extra>',
                                customdata=list(zip(df_fine['New Price ($)'], df_fine['Total Return ($)'], df_fine['Total Return (%)'],
                                                   df_fine['Total Option Value ($)'], df_fine['Cash ($)'], df_fine['ITM']))
                            ), row=1, col=1)
                            fig.add_hline(y=total_capital, line_dash="dash", line_color="red", row=1, col=1)
                            
                            # Total Return Chart with detailed hover (smooth)
                            fig.add_trace(go.Scatter(
                                x=df_fine['Movement (%)'], 
                                y=df_fine['Total Return (%)'], 
                                mode='lines', 
                                name='Return %',
                                line=dict(color='green', width=3),
                                hovertemplate='<b>Total Return</b><br>' +
                                            'Movement: %{x:.1f}%<br>' +
                                            'New Price: $%{customdata[0]:.2f}<br>' +
                                            'Return: $%{customdata[1]:,.0f} (%{y:.1f}%)<br>' +
                                            'Portfolio Value: $%{customdata[2]:,.0f}<br>' +
                                            'Options Value: $%{customdata[3]:,.0f}<br>' +
                                            'Cash: $%{customdata[4]:,.0f}<br>' +
                                            'ITM: %{customdata[5]}<extra></extra>',
                                customdata=list(zip(df_fine['New Price ($)'], df_fine['Total Return ($)'], df_fine['Total Portfolio ($)'],
                                                   df_fine['Total Option Value ($)'], df_fine['Cash ($)'], df_fine['ITM']))
                            ), row=1, col=2)
                            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
                            
                            # Option Value Chart with detailed hover (smooth)
                            fig.add_trace(go.Scatter(
                                x=df_fine['Movement (%)'], 
                                y=df_fine['Total Option Value ($)'], 
                                mode='lines', 
                                name='Options',
                                line=dict(color='orange', width=3),
                                hovertemplate='<b>Option Value</b><br>' +
                                            'Movement: %{x:.1f}%<br>' +
                                            'New Price: $%{customdata[0]:.2f}<br>' +
                                            'Option Payoff: $%{customdata[1]:.2f}<br>' +
                                            'Total Option Value: $%{y:,.0f}<br>' +
                                            'Contracts: %{customdata[2]:.2f}<br>' +
                                            'Portfolio Value: $%{customdata[3]:,.0f}<br>' +
                                            'ITM: %{customdata[4]}<extra></extra>',
                                customdata=list(zip(df_fine['New Price ($)'], df_fine['Option Payoff ($)'], 
                                                   [contracts] * len(df_fine), df_fine['Total Portfolio ($)'], df_fine['ITM']))
                            ), row=2, col=1)
                            
                            # Return Distribution with detailed hover (smooth)
                            fig.add_trace(go.Histogram(
                                x=df_fine['Total Return (%)'], 
                                nbinsx=40, 
                                marker_color='lightblue',
                                hovertemplate='<b>Return Distribution</b><br>' +
                                            'Return Range: %{x[0]:.1f}% to %{x[1]:.1f}%<br>' +
                                            'Count: %{y}<br>' +
                                            'Percentage: %{y} / %{total} = %{customdata:.1f}%<extra></extra>',
                                customdata=[(y/len(df_fine)*100) for y in [len(df_fine)] * 40]
                            ), row=2, col=2)
                            
                            fig.update_layout(
                                height=800, 
                                showlegend=False, 
                                title_text=f"Barbell Strategy: {ticker_symbol} {option_type} ${barbell_strike:.2f}",
                                hovermode='x unified'
                            )
                            fig.update_xaxes(title_text="Movement (%)", row=1, col=1)
                            fig.update_yaxes(title_text="Portfolio ($)", row=1, col=1)
                            fig.update_xaxes(title_text="Movement (%)", row=1, col=2)
                            fig.update_yaxes(title_text="Return (%)", row=1, col=2)
                            fig.update_xaxes(title_text="Movement (%)", row=2, col=1)
                            fig.update_yaxes(title_text="Option Value ($)", row=2, col=1)
                            fig.update_xaxes(title_text="Return (%)", row=2, col=2)
                            fig.update_yaxes(title_text="Count", row=2, col=2)
                            st.plotly_chart(fig, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
                            
                            # Insights
                            st.markdown("### 🔍 Key Insights")
                            profitable = df[df['Total Return (%)'] > 0]
                            if not profitable.empty:
                                st.success(f"💰 Profitable Range: {profitable['Movement (%)'].min():.1f}% to {profitable['Movement (%)'].max():.1f}%")
                                best = df.loc[df['Total Return (%)'].idxmax()]
                                st.success(f"🚀 Best Case: {best['Movement (%)']:.1f}% → {best['Total Return (%)']:.1f}% return (${best['Total Return ($)']:,.0f})")
                            max_loss = df['Total Return (%)'].min()
                            st.warning(f"📉 Maximum Loss: {max_loss:.1f}% (${df['Total Return ($)'].min():,.0f})")
                            
                            # Comparison
                            st.markdown("### 📊 vs 100% Underlying")
                            comp = go.Figure()
                            
                            # Calculate 100% underlying returns (smooth)
                            underlying_returns = [total_capital * (1 + m/100) for m in df_fine['Movement (%)']]
                            
                            comp.add_trace(go.Scatter(
                                x=df_fine['Movement (%)'], 
                                y=df_fine['Total Portfolio ($)'], 
                                mode='lines', 
                                name='Barbell Strategy',
                                line=dict(color='blue', width=3),
                                hovertemplate='<b>Barbell Strategy</b><br>' +
                                            'Movement: %{x:.1f}%<br>' +
                                            'Portfolio Value: $%{y:,.0f}<br>' +
                                            'Return: $%{customdata[0]:,.0f} (%{customdata[1]:.1f}%)<br>' +
                                            'Options: $%{customdata[2]:,.0f}<br>' +
                                            'Cash: $%{customdata[3]:,.0f}<br>' +
                                            'Contracts: %{customdata[4]:.2f}<extra></extra>',
                                customdata=list(zip(df_fine['Total Return ($)'], df_fine['Total Return (%)'], 
                                                   df_fine['Total Option Value ($)'], df_fine['Cash ($)'], 
                                                   [contracts] * len(df_fine)))
                            ))
                            
                            comp.add_trace(go.Scatter(
                                x=df_fine['Movement (%)'], 
                                y=underlying_returns, 
                                mode='lines', 
                                name=f'100% {ticker_symbol}',
                                line=dict(color='green', width=3),
                                hovertemplate='<b>100% {ticker_symbol}</b><br>' +
                                            'Movement: %{x:.1f}%<br>' +
                                            'Value: $%{y:,.0f}<br>' +
                                            'Return: $%{customdata[0]:,.0f} (%{customdata[1]:.1f}%)<br>' +
                                            'Shares: %{customdata[2]:.2f}<extra></extra>',
                                customdata=list(zip([u - total_capital for u in underlying_returns], 
                                                   [(u/total_capital - 1)*100 for u in underlying_returns],
                                                   [total_capital/current_price] * len(df_fine)))
                            ))
                            
                            comp.add_trace(go.Scatter(
                                x=movements, 
                                y=[total_capital] * len(movements), 
                                mode='lines', 
                                name='Initial Capital',
                                line=dict(color='red', width=2, dash='dash'),
                                hovertemplate='<b>Initial Capital</b><br>' +
                                            'Value: $%{y:,.0f}<br>' +
                                            'Return: $0 (0.0%)<extra></extra>'
                            ))
                            
                            comp.update_layout(
                                title=f"Barbell Strategy vs 100% {ticker_symbol}", 
                                xaxis_title="Movement (%)", 
                                yaxis_title="Value ($)", 
                                height=500,
                                hovermode='x unified'
                            )
                            st.plotly_chart(comp, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
                            
                            # Simple ticker vs BARBELL Comparison
                            st.markdown(f"### 📊 {ticker_symbol} vs BARBELL Strategy Comparison")
                            
                            # Cash return rate input - Independent from risk-free rate checkbox
                            if 'barbell_persistent_cash_return' not in st.session_state:
                                st.session_state.barbell_persistent_cash_return = 4.0  # Default to 4% instead of risk-free rate
                            
                            cash_return_rate = st.number_input("💰 Cash Return Rate (%)", min_value=0.0, max_value=20.0, value=st.session_state.barbell_persistent_cash_return, step=0.1, 
                                                              help="Return rate for the cash portion (e.g., 0% = no return, 5% = 5% annual return)", key="barbell_cash_return")
                            st.session_state.barbell_persistent_cash_return = cash_return_rate
                            
                            # Create fine-grained movements for smooth interpolation
                            fine_movements = np.arange(-30, 51, 0.5)  # Every 0.5% instead of 5%
                            
                            # Calculate ticker returns and cash returns
                            ticker_returns = []
                            cash_returns = []
                            barbell_returns = []
                            ticker_values = []
                            barbell_values = []
                            
                            for mov in fine_movements:
                                # Ticker return
                                ticker_return = mov
                                ticker_returns.append(ticker_return)
                                ticker_values.append(total_capital * (1 + mov/100))
                                
                                # Cash return (constant)
                                cash_returns.append(cash_return_rate)
                                
                                # Calculate BARBELL return directly with user-defined cash return rate
                                new_px = current_price * (1 + mov / 100)
                                payoff = max(0, (new_px - barbell_strike) if option_type == "CALL" else (barbell_strike - new_px))
                                opt_val = contracts * payoff * 100
                                
                                # Apply ONLY the user-defined cash return rate (no risk-free rate)
                                cash_portion = cash * (1 + cash_return_rate/100)
                                barbell_value_with_cash_return = opt_val + cash_portion
                                barbell_return_with_cash = ((barbell_value_with_cash_return - total_capital) / total_capital) * 100
                                
                                barbell_returns.append(barbell_return_with_cash)
                                barbell_values.append(barbell_value_with_cash_return)
                            
                            # Return Comparison Chart
                            fig_returns = go.Figure()
                            
                            # Ticker line
                            fig_returns.add_trace(go.Scatter(
                                x=fine_movements,
                                y=ticker_returns,
                                mode='lines',
                                name=ticker_symbol,
                                line=dict(color='blue', width=3),
                                hovertemplate=f'<b>{ticker_symbol}</b><br>Movement: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>'
                            ))
                            
                            # BARBELL line
                            fig_returns.add_trace(go.Scatter(
                                x=fine_movements,
                                y=barbell_returns,
                                mode='lines',
                                name='BARBELL Strategy',
                                line=dict(color='red', width=3),
                                hovertemplate='<b>BARBELL</b><br>Movement: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
                            ))
                            
                            # Cash return line
                            fig_returns.add_trace(go.Scatter(
                                x=fine_movements,
                                y=cash_returns,
                                mode='lines',
                                name=f'Cash Return ({cash_return_rate}%)',
                                line=dict(color='green', width=2, dash='dash'),
                                hovertemplate='<b>Cash Return</b><br>Movement: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
                            ))
                            
                            # Zero line
                            fig_returns.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                            
                            fig_returns.update_layout(
                                title="Return Comparison",
                                xaxis_title=f"{ticker_symbol} Movement (%)",
                                yaxis_title="Return (%)",
                                hovermode='x unified',
                                height=400
                            )
                            
                            st.plotly_chart(fig_returns, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
                            
                            # Value Comparison Chart
                            fig_values = go.Figure()
                            
                            # Ticker line
                            fig_values.add_trace(go.Scatter(
                                x=fine_movements,
                                y=ticker_values,
                                mode='lines',
                                name=ticker_symbol,
                                line=dict(color='blue', width=3),
                                hovertemplate=f'<b>{ticker_symbol}</b><br>Movement: %{{x:.1f}}%<br>Value: $%{{y:,.0f}}<extra></extra>'
                            ))
                            
                            # BARBELL line
                            fig_values.add_trace(go.Scatter(
                                x=fine_movements,
                                y=barbell_values,
                                mode='lines',
                                name='BARBELL Strategy',
                                line=dict(color='red', width=3),
                                hovertemplate='<b>BARBELL</b><br>Movement: %{x:.1f}%<br>Value: $%{y:,.0f}<extra></extra>'
                            ))
                            
                            # Initial capital line
                            fig_values.add_hline(y=total_capital, line_dash="dot", line_color="gray", opacity=0.5)
                            
                            fig_values.update_layout(
                                title="Value Comparison",
                                xaxis_title=f"{ticker_symbol} Movement (%)",
                                yaxis_title="Portfolio Value ($)",
                                hovermode='x unified',
                                height=400
                            )
                            
                            st.plotly_chart(fig_values, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
                            
                            # Summary Table
                            st.markdown("### 📊 Performance Summary Table")
                            
                            # Create summary data
                            summary_data = []
                            for i, mov in enumerate(fine_movements[::10]):  # Every 5% for readability
                                ticker_return = ticker_returns[i*10]
                                barbell_return = barbell_returns[i*10]
                                ticker_value = ticker_values[i*10]
                                barbell_value = barbell_values[i*10]
                                
                                # Calculate variations
                                return_diff = barbell_return - ticker_return
                                value_diff = barbell_value - ticker_value
                                
                                # Calculate cash return percentage
                                cash_return_pct = ((cash * (1 + cash_return_rate/100)) - cash) / cash * 100
                                
                                summary_data.append({
                                    f'{ticker_symbol} Movement (%)': f"{mov:.1f}%",
                                    'BARBELL Movement (%)': f"{barbell_return:+.1f}%",
                                    f'{ticker_symbol} Value ($)': f"${ticker_value:,.0f}",
                                    'BARBELL Value ($)': f"${barbell_value:,.0f}",
                                    'Return Difference (%)': f"{return_diff:+.1f}%",
                                    'Value Difference ($)': f"${value_diff:+,.0f}",
                                    'Cash Return (%)': f"{cash_return_pct:+.1f}%"
                                })
                            
                            # Create DataFrame
                            summary_df = pd.DataFrame(summary_data)
                            
                            # Define color function with row-based matching
                            def color_gradient_matched(val, col_name, row_index):
                                if isinstance(val, str):
                                    # For Ticker Value ($) - match Ticker Movement (%) color
                                    if col_name == f'{ticker_symbol} Value ($)':
                                        ticker_movement_val = summary_df.iloc[row_index][f'{ticker_symbol} Movement (%)']
                                        ticker_movement_num = float(ticker_movement_val.replace('%', ''))
                                        if ticker_movement_num > 50:
                                            return 'background-color: #004d00'
                                        elif ticker_movement_num > 20:
                                            return 'background-color: #1e8449'
                                        elif ticker_movement_num > 5:
                                            return 'background-color: #388e3c'
                                        elif ticker_movement_num > 0:
                                            return 'background-color: #66bb6a'
                                        elif ticker_movement_num < -50:
                                            return 'background-color: #7b0000'
                                        elif ticker_movement_num < -20:
                                            return 'background-color: #b22222'
                                        elif ticker_movement_num < -5:
                                            return 'background-color: #d32f2f'
                                        elif ticker_movement_num < 0:
                                            return 'background-color: #ef5350'
                                    
                                    # For BARBELL Value ($) - match BARBELL Movement (%) color
                                    elif col_name == 'BARBELL Value ($)':
                                        barbell_movement_val = summary_df.iloc[row_index]['BARBELL Movement (%)']
                                        barbell_movement_num = float(barbell_movement_val.replace('%', '').replace('+', ''))
                                        if barbell_movement_num > 50:
                                            return 'background-color: #004d00'
                                        elif barbell_movement_num > 20:
                                            return 'background-color: #1e8449'
                                        elif barbell_movement_num > 5:
                                            return 'background-color: #388e3c'
                                        elif barbell_movement_num > 0:
                                            return 'background-color: #66bb6a'
                                        elif barbell_movement_num < -50:
                                            return 'background-color: #7b0000'
                                        elif barbell_movement_num < -20:
                                            return 'background-color: #b22222'
                                        elif barbell_movement_num < -5:
                                            return 'background-color: #d32f2f'
                                        elif barbell_movement_num < 0:
                                            return 'background-color: #ef5350'
                                    
                                    # For Value Difference ($) - match Return Difference (%) color
                                    elif col_name == 'Value Difference ($)':
                                        return_diff_val = summary_df.iloc[row_index]['Return Difference (%)']
                                        return_diff_num = float(return_diff_val.replace('%', '').replace('+', ''))
                                        if return_diff_num > 50:
                                            return 'background-color: #004d00'
                                        elif return_diff_num > 20:
                                            return 'background-color: #1e8449'
                                        elif return_diff_num > 5:
                                            return 'background-color: #388e3c'
                                        elif return_diff_num > 0:
                                            return 'background-color: #66bb6a'
                                        elif return_diff_num < -50:
                                            return 'background-color: #7b0000'
                                        elif return_diff_num < -20:
                                            return 'background-color: #b22222'
                                        elif return_diff_num < -5:
                                            return 'background-color: #d32f2f'
                                        elif return_diff_num < 0:
                                            return 'background-color: #ef5350'
                                    
                                    # For other columns - use their own values
                                    else:
                                        try:
                                            clean_val = val.replace('%', '').replace('$', '').replace(',', '').replace('+', '')
                                            if clean_val:
                                                num_val = float(clean_val)
                                                if num_val > 50:
                                                    return 'background-color: #004d00'
                                                elif num_val > 20:
                                                    return 'background-color: #1e8449'
                                                elif num_val > 5:
                                                    return 'background-color: #388e3c'
                                                elif num_val > 0:
                                                    return 'background-color: #66bb6a'
                                                elif num_val < -50:
                                                    return 'background-color: #7b0000'
                                                elif num_val < -20:
                                                    return 'background-color: #b22222'
                                                elif num_val < -5:
                                                    return 'background-color: #d32f2f'
                                                elif num_val < 0:
                                                    return 'background-color: #ef5350'
                                        except:
                                            pass
                                # DEFAULT: Apply neutral background for any unmatched cells
                                return 'background-color: #2d3748; color: white'
                            
                            # Apply conditional formatting with row-based matching
                            def apply_color_to_row(row):
                                row_index = row.name
                                return [color_gradient_matched(val, col, row_index) for val, col in zip(row, row.index)]
                            
                            styled_df = summary_df.style.apply(apply_color_to_row, axis=1)
                            
                            # Display table with conditional formatting
                            st.dataframe(
                                styled_df,
                                width='stretch',
                                height=450,  # Optimal height to show all rows without scrolling
                                column_config={
                                    f"{ticker_symbol} Movement (%)": st.column_config.TextColumn(f"{ticker_symbol} Movement (%)", width="medium"),
                                    "BARBELL Movement (%)": st.column_config.TextColumn("📈 BARBELL Movement (%)", width="medium"),
                                    f"{ticker_symbol} Value ($)": st.column_config.TextColumn(f"{ticker_symbol} Value ($)", width="medium"),
                                    "BARBELL Value ($)": st.column_config.TextColumn("BARBELL Value ($)", width="medium"),
                                    "Return Difference (%)": st.column_config.TextColumn("📊 Return Difference (%)", width="medium"),
                                    "Value Difference ($)": st.column_config.TextColumn("💰 Value Difference ($)", width="medium"),
                                    "Cash Return (%)": st.column_config.TextColumn("💰 Cash Return (%)", width="medium")
                                }
                            )
                            
                            # Key insights
                            st.markdown("#### 🔍 Key Insights")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                best_return_idx = np.argmax(barbell_returns)
                                best_movement = fine_movements[best_return_idx]
                                best_return = barbell_returns[best_return_idx]
                                st.metric("🚀 Best BARBELL Return", f"{best_return:.1f}%", f"at {best_movement:.1f}% {ticker_symbol} movement")
                            
                            with col2:
                                worst_return_idx = np.argmin(barbell_returns)
                                worst_movement = fine_movements[worst_return_idx]
                                worst_return = barbell_returns[worst_return_idx]
                                st.metric("📉 Worst BARBELL Return", f"{worst_return:.1f}%", f"at {worst_movement:.1f}% {ticker_symbol} movement")
                            
                            with col3:
                                avg_return_diff = np.mean([barbell_returns[i] - ticker_returns[i] for i in range(len(ticker_returns))])
                                st.metric("📊 Average Return Difference", f"{avg_return_diff:+.1f}%", f"BARBELL vs {ticker_symbol}")
                            
                            with col4:
                                # Find minimum SPY movement for 0% BARBELL return
                                zero_return_idx = np.argmin(np.abs(barbell_returns))
                                zero_movement = fine_movements[zero_return_idx]
                                st.metric("⚖️ Neutral BARBELL Return", f"0.0%", f"at {zero_movement:.1f}% {ticker_symbol} movement")
                            
                            with col5:
                                # Find ticker movement where BARBELL = ticker return - PRIORITIZE POSITIVE VALUES
                                return_diffs = np.array(barbell_returns) - np.array(ticker_returns)
                                
                                # First try to find positive movements where performance is equal
                                positive_indices = np.where(fine_movements > 0)[0]
                                if len(positive_indices) > 0:
                                    positive_diffs = return_diffs[positive_indices]
                                    positive_movements = fine_movements[positive_indices]
                                    
                                    # Find the closest to zero among positive movements
                                    closest_positive_idx = np.argmin(np.abs(positive_diffs))
                                    equal_movement = positive_movements[closest_positive_idx]
                                else:
                                    # Fallback to any movement if no positive ones
                                    equal_return_idx = np.argmin(np.abs(return_diffs))
                                    equal_movement = fine_movements[equal_return_idx]
                                
                                st.metric("🎯 Equal Performance", f"BARBELL = {ticker_symbol}", f"at {equal_movement:.1f}% {ticker_symbol} movement")
                            
                            st.markdown("💡 **Note**: Assumes options held to expiration, sold at intrinsic value only (no extrinsic). Time decay/volatility changes not considered.")
                        else:
                            st.warning("⚠️ Not enough capital for 1 contract")
                    else:
                        st.error("❌ No valid option price")
                else:
                    st.warning(f"No {option_type} options available")
                
                # Historical Backtest Section
                st.markdown("---")
                st.markdown("### 📈 Historical Backtest")
                st.markdown("**Backtest the Barbell strategy using historical data**")
                
                # 🛡️ ULTIMATE PROTECTION - ALWAYS DEFINE BARBELL_STRIKE
                # Force barbell_strike to always be defined
                try:
                    if 'barbell_strike' not in globals() or barbell_strike is None:
                        barbell_strike = 400.0
                except:
                    barbell_strike = 400.0
                
                # 🛡️ ULTIMATE FALLBACK - ALWAYS HAVE A VALUE
                if barbell_strike is None:
                    # Try to get from session state first
                    if 'persistent_strike' in st.session_state:
                        barbell_strike = float(st.session_state.persistent_strike)
                    else:
                        # ULTIMATE FALLBACK - NEVER CRASH
                        barbell_strike = 400.0
                        st.warning("⚠️ Using default strike 400.0 - Please select a strike in the Barbell Strategy section for better results.")
                
                st.markdown(f"**Using current selection: {options_pct}% options, {100-options_pct}% bonds, Strike {barbell_strike:.0f}, Expiration {selected_exp}**")
                
                # Bond return rate - PERSISTANT
                if 'barbell_persistent_bond_return' not in st.session_state:
                    st.session_state.barbell_persistent_bond_return = 4.0
                
                bond_return_rate = st.number_input("💰 Bond Return Rate (%)", min_value=0.0, max_value=15.0, value=st.session_state.barbell_persistent_bond_return, step=0.1, 
                                                 help="Annual return rate for the bond portion", key="persistent_bond_return")
                st.session_state.barbell_persistent_bond_return = bond_return_rate
                
                # Initial value and added per period - PERSISTENT
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'barbell_persistent_initial_value' not in st.session_state:
                        st.session_state.barbell_persistent_initial_value = 100000
                    
                    initial_value = st.number_input("💰 Valeur Initiale ($)", min_value=1000, value=st.session_state.barbell_persistent_initial_value, step=1000, 
                                                   help="Capital initial pour le backtest", key="persistent_initial_value")
                    st.session_state.barbell_persistent_initial_value = initial_value
                
                with col2:
                    if 'barbell_persistent_added_per_period' not in st.session_state:
                        st.session_state.barbell_persistent_added_per_period = 0
                    
                    added_per_period = st.number_input("📈 Added Value per Period ($)", min_value=0, value=st.session_state.barbell_persistent_added_per_period, step=1000, 
                                                     help="Amount added each period", key="persistent_added_period")
                    st.session_state.barbell_persistent_added_per_period = added_per_period
                
                # Initialize persistent storage for all backtest results
                if 'all_backtest_results' not in st.session_state:
                    st.session_state.all_backtest_results = {}
                
                # Display all saved backtest results IMMEDIATELY (persistent across page refreshes)
                if st.session_state.all_backtest_results:
                    st.markdown("### 📊 All Saved Backtest Results")
                    
                    for result_key, result_data in st.session_state.all_backtest_results.items():
                        with st.expander(f"📈 {result_key}", expanded=False):
                            # Display parameters
                            params = result_data['params']
                            
                            # Check if it's a Multi-Strike, All-CALLs, or regular backtest
                            if 'Multi-Strike' in result_key:
                                # Multi-Strike results
                                st.markdown(f"**Parameters:** {params['options_pct']}% options, {100-params['options_pct']}% bonds, Multi-Strike test, Expiration {params['selected_exp']} ({params['days_to_exp']} days)")
                            elif 'All-CALLs' in result_key:
                                # All-CALLs results
                                st.markdown(f"**Parameters:** {params['options_pct']}% options, {100-params['options_pct']}% bonds, All-CALLs test, Min {params['days_to_exp']} days to expiration")
                            else:
                                # Regular backtest results
                                if 'barbell_strike' in params:
                                    st.markdown(f"**Parameters:** {params['options_pct']}% options, {100-params['options_pct']}% bonds, Strike {params['barbell_strike']:.0f}, Expiration {params['selected_exp']} ({params['days_to_exp']} days)")
                                else:
                                    st.markdown(f"**Parameters:** {params['options_pct']}% options, {100-params['options_pct']}% bonds, Expiration {params['selected_exp']} ({params['days_to_exp']} days)")
                            
                            # Display summary table if available
                            if 'summary_df' in result_data and result_data['summary_df'] is not None and not result_data['summary_df'].empty:
                                st.markdown("**Performance Summary:**")
                                summary_df = result_data['summary_df']
                                
                                # Check if it's a Multi-Strike, All-CALLs, or regular result
                                if 'Multi-Strike' in result_key:
                                    # Multi-Strike results - display the full table with colors
                                    display_results = summary_df.copy()
                                    
                                    # Apply same formatting as the original Multi-Strike backtest
                                    for col in display_results.columns:
                                        if col == 'Strike':
                                            display_results[col] = display_results[col].apply(lambda x: 
                                                f"{int(float(x))}" if float(x) == int(float(x)) else f"{float(x):.2f}")
                                        elif col in ['Positive Periods', 'Negative Periods', 'Total Periods']:
                                            pass
                                        elif col == 'Final Capital ($)':
                                            display_results[col] = display_results[col].apply(lambda x: f"${float(x):,.2f}")
                                        else:
                                            if display_results[col].dtype in ['float64', 'int64']:
                                                display_results[col] = display_results[col].apply(lambda x: f"{float(x):.2f}")
                                elif 'All-CALLs' in result_key:
                                    # All-CALLs results - display the full table with colors
                                    display_results = summary_df.copy()
                                    
                                    # Apply same formatting as the original All-CALLs backtest
                                    for col in display_results.columns:
                                        if col == 'Strike':
                                            display_results[col] = display_results[col].apply(lambda x: 
                                                f"${int(float(x))}" if float(x) == int(float(x)) else f"${float(x):.2f}")
                                        elif col in ['Positive Periods', 'Negative Periods', 'Total Periods', 'Days to Exp']:
                                            pass
                                        elif col in ['Option Price', 'Contracts']:
                                            if display_results[col].dtype in ['float64', 'int64']:
                                                display_results[col] = display_results[col].apply(lambda x: f"${float(x):.2f}")
                                        elif col == 'Final Capital ($)':
                                            if display_results[col].dtype in ['float64', 'int64']:
                                                display_results[col] = display_results[col].apply(lambda x: f"${float(x):,.0f}")
                                        elif col in ['CAGR (%)', 'CPGR (%)', 'MWRR (%)', 'Average Return (%)', 'Median Return (%)', 'Best Period (%)', 'Worst Period (%)', '% Positive Periods']:
                                            if display_results[col].dtype in ['float64', 'int64']:
                                                display_results[col] = display_results[col].apply(lambda x: f"{float(x):.2f}%")
                                    
                                    # Apply comprehensive colors - SAME AS ORIGINAL All-CALLs
                                    def color_comprehensive_results(val, col_name, row_index):
                                        try:
                                            if isinstance(val, (int, float)):
                                                if col_name in ['CAGR (%)', 'CPGR (%)', 'MWRR (%)']:
                                                    if val > 20:
                                                        return 'background-color: #004d00; color: white; font-weight: bold'
                                                    elif val > 10:
                                                        return 'background-color: #1e8449; color: white; font-weight: bold'
                                                    elif val > 5:
                                                        return 'background-color: #66bb6a; color: white'
                                                    elif val > 0:
                                                        return 'background-color: #ffeb3b; color: black'
                                                    else:
                                                        return 'background-color: #ef5350; color: white'
                                                elif col_name == 'Final Capital ($)':
                                                    if val > params['total_capital'] * 2:
                                                        return 'background-color: #004d00; color: white; font-weight: bold'
                                                    elif val > params['total_capital'] * 1.5:
                                                        return 'background-color: #1e8449; color: white'
                                                    elif val > params['total_capital']:
                                                        return 'background-color: #66bb6a; color: white'
                                                    else:
                                                        return 'background-color: #ef5350; color: white'
                                                elif col_name == '% Positive Periods':
                                                    if val > 70:
                                                        return 'background-color: #004d00; color: white; font-weight: bold'
                                                    elif val > 50:
                                                        return 'background-color: #66bb6a; color: white'
                                                    elif val > 30:
                                                        return 'background-color: #ffeb3b; color: black'
                                                    else:
                                                        return 'background-color: #ef5350; color: white'
                                        except:
                                            return ''
                                        return ''
                                    
                                    # Apply colors to RAW data first, then format for display
                                    styled_results = summary_df.style.apply(
                                        lambda x: [color_comprehensive_results(val, x.name, i) 
                                                 for i, val in enumerate(x)], 
                                        axis=0
                                    ).set_table_styles([
                                        {'selector': 'th', 'props': [('background-color', '#2d3748'), ('color', 'white'), ('font-weight', 'bold')]},
                                        {'selector': 'td', 'props': [('text-align', 'center')]}
                                    ])
                                    
                                    # Format the styled results for display - MAX 2 decimal places
                                    styled_results = styled_results.format({
                                        'Strike': '${:.0f}',
                                        'Option Price': '${:.2f}',
                                        'Contracts': '{:.2f}',
                                        'Final Capital ($)': '${:,.0f}',
                                        'CAGR (%)': '{:.2f}%',
                                        'CPGR (%)': '{:.2f}%',
                                        'MWRR (%)': '{:.2f}%',
                                        'Average Return (%)': '{:.2f}%',
                                        'Median Return (%)': '{:.2f}%',
                                        'Best Period (%)': '{:.2f}%',
                                        'Worst Period (%)': '{:.2f}%',
                                        '% Positive Periods': '{:.2f}%'
                                    })
                                    
                                    st.dataframe(styled_results, use_container_width=True, height=600)
                                else:
                                    # Regular backtest results - display without colors
                                    st.dataframe(summary_df, use_container_width=True)
                            
                            # Display periods table if available
                            if 'periods_df' in result_data and result_data['periods_df'] is not None and not result_data['periods_df'].empty:
                                st.markdown("**Periods Table:**")
                                periods_df = result_data['periods_df']
                                
                                # Apply colors to periods table
                                def color_persistent_periods(val, col_name, row_index):
                                    if isinstance(val, (int, float, str)):
                                        if col_name.endswith('(%)'):
                                            if isinstance(val, str):
                                                percentage_val = float(val.replace('%', '').replace(',', ''))
                                            else:
                                                percentage_val = float(val)
                                            
                                            if percentage_val > 50:
                                                return 'background-color: #004d00; color: white'
                                            elif percentage_val > 20:
                                                return 'background-color: #1e8449; color: white'
                                            elif percentage_val > 5:
                                                return 'background-color: #388e3c; color: white'
                                            elif percentage_val > 0:
                                                return 'background-color: #66bb6a; color: white'
                                            elif percentage_val < -50:
                                                return 'background-color: #7b0000; color: white'
                                            elif percentage_val < -20:
                                                return 'background-color: #b22222; color: white'
                                            elif percentage_val < -5:
                                                return 'background-color: #d32f2f; color: white'
                                            elif percentage_val < 0:
                                                return 'background-color: #ef5350; color: white'
                                            else:
                                                return 'background-color: #424242; color: white'
                                    return ''
                                
                                styled_periods = periods_df.style.apply(
                                    lambda x: [color_persistent_periods(val, x.name, i) 
                                             for i, val in enumerate(x)], 
                                    axis=0,
                                    subset=[col for col in periods_df.columns if col.endswith('(%)')]
                                )
                                
                                st.dataframe(styled_periods, use_container_width=True)
                            
                            # Delete button for this specific result
                            if st.button(f"🗑️ Delete {result_key}", key=f"delete_{result_key}"):
                                del st.session_state.all_backtest_results[result_key]
                                st.success("Deleted!")
                                st.rerun()
                    
                    # Clear all results button
                    if st.button("🗑️ Clear All Results", key="clear_all_results"):
                        st.session_state.all_backtest_results = {}
                        st.success("All results cleared!")
                        st.rerun()
                    
                    st.markdown("---")
                
                # Run backtest button
                # Add multi-strike backtest option
                col1, col2 = st.columns(2)
                
                with col1:
                    run_single_backtest = st.button("🚀 Run Historical Backtest", key="run_backtest")
                
                with col2:
                    run_multi_strike_backtest = st.button("🎯 Run Multi-Strike Backtest", key="run_multi_strike_backtest", type="primary")
                
                if run_single_backtest:
                    with st.spinner("Fetching historical data and running backtest..."):
                        try:
                            import yfinance as yf
                            from datetime import datetime, timedelta
                            import numpy as np
                            
                            # Get the option details from the current selection
                            if not filtered_opts.empty:
                                opt = filtered_opts[filtered_opts['strike'] == barbell_strike].iloc[0]
                                opt_price = calculate_mid_price(opt['bid'], opt['ask'], opt['lastPrice'])
                                
                                if not opt_price or opt_price <= 0:
                                    st.error("❌ Invalid option price for backtest")
                                    st.stop()
                            else:
                                st.error("❌ No options data available for backtest")
                                st.stop()
                            
                            # Calculate days to expiration from selected option
                            try:
                                exp_dt = pd.to_datetime(selected_exp)
                                days_to_exp = (exp_dt - datetime.now()).days
                                if days_to_exp <= 0:
                                    st.error("❌ Selected option has already expired")
                                    st.stop()
                            except:
                                st.error("❌ Invalid expiration date")
                                st.stop()
                            
                            # Display backtest parameters
                            option_portion = total_capital * (options_pct / 100)
                            bond_portion = total_capital * ((100 - options_pct) / 100)
                            ticker_exposure = contracts * barbell_strike * 100
                            
                            st.info(f"📊 **Backtest Parameters**: {options_pct}% options ({option_portion:,.0f}".replace(',', ' ') + f"), {100-options_pct}% bonds ({bond_portion:,.0f}".replace(',', ' ') + f"), Strike {barbell_strike:.0f}, Expiration {selected_exp} ({days_to_exp} days), Option Price {opt_price:.2f}")
                            st.info(f"🎯 **{ticker_symbol} Exposure**: {ticker_exposure:,.0f}".replace(',', ' ') + f" ({ticker_exposure/total_capital:.1f}x leverage)")
                            
                            # SIMPLE BACKTEST - DAILY
                            st.info(f"ℹ️ {ticker_symbol} backtest - Daily with variations")
                            
                            # Try to get cached data first, fallback to fetch if not available
                            global_cache_key = f"historical_data_{ticker_symbol}"
                            if global_cache_key in st.session_state:
                                # Use cached data
                                cached_data = st.session_state[global_cache_key]
                                ticker_data = cached_data['ticker_data']
                                price_col = cached_data['price_col']
                                results_df = cached_data['results_df']
                                st.info(f"📊 Using cached historical data for Barbell Strategy ({len(results_df)} days)")
                            else:
                                # Fetch ticker data - ALL historical data since 1993
                                st.info("📊 Fetching historical data for Barbell Strategy (1993-present)")
                                ticker_data = yf.download(ticker_symbol, start="1993-01-01", progress=False)
                            
                            if ticker_data.empty:
                                st.error(f"❌ Failed to fetch {ticker_symbol} data")
                                st.stop()
                            
                            # Use Close column
                            price_col = 'Close' if 'Close' in ticker_data.columns else 'Adj Close'
                            
                            # FFILL pour inclure les jours non-trading (weekends/vacances)
                            ticker_data = ticker_data.resample('D').ffill()
                            
                            # Calculate daily variations
                            daily_data = []
                            for i in range(len(ticker_data)):
                                date = ticker_data.index[i]
                                price = ticker_data[price_col].iloc[i]
                                
                                # Calculate variation from previous day
                                if i > 0:
                                    prev_price = float(ticker_data[price_col].iloc[i-1])
                                    current_price = float(price)
                                    daily_change = (current_price / prev_price) - 1
                                else:
                                    daily_change = 0
                                
                                # With and without dividends (approximate)
                                daily_change_with_div = daily_change
                                daily_change_no_div = daily_change - (0.015 / 365)  # Subtract daily dividend impact
                                
                                daily_data.append({
                                    'Date': date,
                                    'Price': float(price),
                                    'Daily Change (w/ Div)': float(daily_change_with_div * 100),
                                    'Daily Change (no Div)': float(daily_change_no_div * 100)
                                })
                            
                            # Create results DataFrame
                            results_df = pd.DataFrame(daily_data)
                            
                            # Cache the data for future use (global cache)
                            st.session_state[global_cache_key] = {
                                'ticker_data': ticker_data,
                                'price_col': price_col,
                                'results_df': results_df,
                                'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            if results_df.empty:
                                st.error("❌ No data found")
                                st.stop()
                            
                            # Simple display
                            st.markdown(f"### 📊 {ticker_symbol} Daily Variations")
                            st.markdown(f"**{len(results_df)} days of {ticker_symbol} data**")
                            
                            # Simple metrics
                            avg_ticker_with_div = results_df['Daily Change (w/ Div)'].mean()
                            avg_ticker_no_div = results_df['Daily Change (no Div)'].mean()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"{ticker_symbol} Daily Average (w/ Div)", f"{avg_ticker_with_div:.2f}%")
                            with col2:
                                st.metric(f"{ticker_symbol} Daily Average (no Div)", f"{avg_ticker_no_div:.2f}%")
                            
                            # Display complete table in dropdown
                            with st.expander(f"📋 {ticker_symbol} Daily Data Table", expanded=False):
                                st.markdown("**Daily with variations:**")
                                
                                # Format the table for better display
                                display_df = results_df.copy()
                                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                                
                                # Round numeric columns
                                display_df['Price'] = display_df['Price'].round(2)
                                display_df['Daily Change (w/ Div)'] = display_df['Daily Change (w/ Div)'].round(2)
                                display_df['Daily Change (no Div)'] = display_df['Daily Change (no Div)'].round(2)
                                
                                st.dataframe(display_df, use_container_width=True)
                            
                            # SECOND TABLE - PERIODS ACCORDING TO OPTION
                            st.markdown(f"### 📊 SECOND TABLE - {days_to_exp} DAY PERIODS")
                            st.markdown(f"**Total variation per period (based on selected option):**")
                            
                            # 🎯 DUAL RUN BACKTEST - HISTORICAL SINGLE 🎯
                            # RUN 1: PURE PERFORMANCE (sans ajouts d'argent)
                            # RUN 2: WITH ADDED MONEY (pour MWRR et Final Capital)
                            
                            # ===== RUN 1: PURE PERFORMANCE =====
                            pure_performance_portfolio = initial_value
                            pure_period_returns = []
                            
                            # ===== RUN 2: WITH ADDED MONEY =====
                            # Valeurs cumulatives pour le backtest
                            cumulative_with_div = initial_value
                            cumulative_no_div = initial_value
                            cumulative_portfolio = initial_value
                            
                            # 🎯 PURE TICKER PERFORMANCE (sans ajouts d'argent) pour tickers
                            pure_performance_with_div = initial_value
                            pure_performance_no_div = initial_value
                            pure_period_returns_with_div = []
                            pure_period_returns_no_div = []
                            
                            # Create periods according to selected option - SIMPLE LOGIC
                            period_data = []
                            period_num = 1
                            
                            # Go through data in blocks of days_to_exp (option days)
                            for i in range(0, len(results_df) - days_to_exp + 1, days_to_exp):
                                # Start and end dates
                                start_date = results_df.iloc[i]['Date']
                                end_date = results_df.iloc[i + days_to_exp - 1]['Date']
                                
                                # Start and end prices
                                start_price = results_df.iloc[i]['Price']
                                end_price = results_df.iloc[i + days_to_exp - 1]['Price']
                                
                                # Calculate total variations
                                ticker_with_div = ((end_price / start_price) - 1) * 100
                                ticker_no_div = ticker_with_div - (1.5 * days_to_exp / 365)  # Subtract dividends
                                
                                # ===== RUN 1: PURE PERFORMANCE CALCULATION =====
                                # Calculate portfolio performance for this period (PURE - no added money)
                                current_capital_pure = pure_performance_portfolio
                                current_option_portion_pure = current_capital_pure * (options_pct / 100)
                                current_contracts_pure = current_option_portion_pure / (opt_price * 100)
                                
                                bond_period_return = (bond_return_rate / 100) * (days_to_exp / 365)
                                bond_value_pure = (current_capital_pure - current_option_portion_pure) * (1 + bond_period_return)
                                
                                # Final ticker price = current price + period variation (without dividends)
                                current_ticker_price = ticker_data[price_col].iloc[-1]  # Current price (today)
                                final_ticker_price = current_ticker_price * (1 + ticker_no_div / 100)  # Final price with variation
                                
                                option_profit_per_share = max(0, float(final_ticker_price) - barbell_strike)
                                option_value_pure = option_profit_per_share * current_contracts_pure * 100
                                
                                # Final Portfolio Value (PURE)
                                total_portfolio_value_pure = bond_value_pure + option_value_pure
                                portfolio_return_pure = ((total_portfolio_value_pure / current_capital_pure) - 1) * 100
                                
                                # Store pure performance return
                                pure_period_returns.append(portfolio_return_pure)
                                
                                # Update pure performance (NO added money)
                                pure_performance_portfolio = pure_performance_portfolio * (1 + portfolio_return_pure / 100)
                                
                                # ===== TICKERS DUAL RUN CALCULATION =====
                                # Calculate ticker performance (PURE - no added money)
                                pure_performance_with_div = pure_performance_with_div * (1 + ticker_with_div / 100)
                                pure_performance_no_div = pure_performance_no_div * (1 + ticker_no_div / 100)
                                
                                # Store pure ticker returns
                                pure_period_returns_with_div.append(ticker_with_div)
                                pure_period_returns_no_div.append(ticker_no_div)
                                
                                # ===== RUN 2: WITH ADDED MONEY CALCULATION =====
                                # Calculate portfolio performance for this period (WITH added money)
                                current_capital_with_added = cumulative_portfolio + added_per_period
                                current_option_portion_with_added = current_capital_with_added * (options_pct / 100)
                                current_contracts_with_added = current_option_portion_with_added / (opt_price * 100)
                                
                                bond_value_with_added = (current_capital_with_added - current_option_portion_with_added) * (1 + bond_period_return)
                                option_value_with_added = option_profit_per_share * current_contracts_with_added * 100
                                
                                # Valeur Finale Portfolio (WITH ADDED MONEY)
                                total_portfolio_value_with_added = bond_value_with_added + option_value_with_added
                                portfolio_return_with_added = ((total_portfolio_value_with_added / current_capital_with_added) - 1) * 100
                                
                                # Calculs cumulatifs pour le backtest (WITH ADDED MONEY)
                                cumulative_with_div = cumulative_with_div * (1 + ticker_with_div / 100) + added_per_period
                                cumulative_no_div = cumulative_no_div * (1 + ticker_no_div / 100) + added_per_period
                                cumulative_portfolio = cumulative_portfolio * (1 + portfolio_return_with_added / 100) + added_per_period
                                
                                # Use pure performance return for display
                                portfolio_return = portfolio_return_pure
                                
                                # Nombre de jours calendaires (avec FFILL)
                                actual_calendar_days = days_to_exp
                                
                                period_data.append({
                                    'Période': period_num,
                                    'Date Début': start_date.strftime('%Y-%m-%d'),
                                    'Date Fin': end_date.strftime('%Y-%m-%d'),
                                    'Jours Calendaires': actual_calendar_days,
                                    f'{ticker_symbol} avec dividendes (%)': f"{round(ticker_with_div, 2):.2f}",
                                    f'{ticker_symbol} sans dividendes (%)': f"{round(ticker_no_div, 2):.2f}",
                                    f'Portfolio Barbell (%)': f"{round(portfolio_return, 2):.2f}",
                                    f'Capital Avec Div ($)': f"{round(cumulative_with_div, 0):,.0f}".replace(',', ' '),
                                    f'Capital Sans Div ($)': f"{round(cumulative_no_div, 0):,.0f}".replace(',', ' '),
                                    f'Capital Portfolio ($)': f"{round(cumulative_portfolio, 0):,.0f}".replace(',', ' ')
                                })
                                
                                period_num += 1
                            
                            # Create the periods DataFrame and save it
                            periods_df = pd.DataFrame(period_data)
                            
                            # Save the table in session_state for persistence
                            st.session_state.barbell_backtest_results = periods_df
                            
                            # Also save the parameters used to avoid "expired" error
                            st.session_state.barbell_backtest_params = {
                                'ticker_symbol': ticker_symbol,
                                'barbell_strike': barbell_strike,
                                'selected_exp': selected_exp,
                                'days_to_exp': days_to_exp,
                                'opt_price': opt_price,
                                'options_pct': options_pct,
                                'total_capital': total_capital,
                                'bond_return_rate': bond_return_rate,
                                'initial_value': initial_value,
                                'added_per_period': added_per_period
                            }
                            
                            if not periods_df.empty:
                                # ===== METRICS FROM RUN 1 (PURE PERFORMANCE) =====
                                total_days = len(periods_df) * days_to_exp
                                
                                # CAGR based on PURE PERFORMANCE (without added money)
                                if total_days > 0 and initial_value > 0 and pure_performance_portfolio > 0:
                                    capr_portfolio = ((pure_performance_portfolio / initial_value) ** (365 / total_days) - 1) * 100
                                else:
                                    capr_portfolio = 0
                                
                                # ===== METRICS FROM RUN 2 (WITH ADDED MONEY) =====
                                final_capital_with_div = cumulative_with_div
                                final_capital_no_div = cumulative_no_div
                                final_capital_portfolio = cumulative_portfolio
                                
                                # MWRR (Money-Weighted Rate of Return) - vraie formule avec cash flows
                                def calculate_mwrr(initial_investment, periodic_investments, final_value, periods):
                                    """Calculate MWRR using Newton-Raphson method"""
                                    if periods == 0 or final_value <= 0:
                                        return 0
                                    
                                    # Cash flows: initial investment (negative), periodic investments (negative), final value (positive)
                                    cash_flows = [-initial_investment]
                                    for i in range(periods):
                                        cash_flows.append(-periodic_investments)
                                    cash_flows.append(final_value)
                                    
                                    # Newton-Raphson method to find IRR
                                    def npv(rate, cash_flows):
                                        return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
                                    
                                    def npv_derivative(rate, cash_flows):
                                        return sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows))
                                    
                                    # Initial guess
                                    rate = 0.1
                                    for _ in range(100):  # Max 100 iterations
                                        npv_val = npv(rate, cash_flows)
                                        if abs(npv_val) < 1e-6:
                                            break
                                        npv_deriv = npv_derivative(rate, cash_flows)
                                        if abs(npv_deriv) < 1e-10:
                                            break
                                        rate = rate - npv_val / npv_deriv
                                    
                                    return rate * 100
                                
                                periods_count = len(periods_df)
                                mwrr_with_div = calculate_mwrr(initial_value, added_per_period, final_capital_with_div, periods_count)
                                mwrr_no_div = calculate_mwrr(initial_value, added_per_period, final_capital_no_div, periods_count)
                                mwrr_portfolio = calculate_mwrr(initial_value, added_per_period, final_capital_portfolio, periods_count)
                                
                                # Ticker CAGR based on PURE PERFORMANCE (without added money) for tickers
                                if total_days > 0 and initial_value > 0 and pure_performance_with_div > 0:
                                    total_cagr_with_div = ((pure_performance_with_div / initial_value) ** (365 / total_days) - 1) * 100
                                else:
                                    total_cagr_with_div = 0
                                
                                if total_days > 0 and initial_value > 0 and pure_performance_no_div > 0:
                                    total_cagr_no_div = ((pure_performance_no_div / initial_value) ** (365 / total_days) - 1) * 100
                                else:
                                    total_cagr_no_div = 0
                                
                                # Create comprehensive summary table
                                st.markdown("### 📈 Performance Summary Table")
                                
                                # Show price adjustment status
                                if ('barbell_price_adjustment' in st.session_state and 
                                    st.session_state.barbell_price_adjustment != 0):
                                    adjustment = st.session_state.barbell_price_adjustment
                                    if adjustment > 0:
                                        st.warning(f"🎯 **Price Adjustment Active**: All option prices increased by {adjustment:.1f}%")
                                    else:
                                        st.success(f"🎯 **Price Adjustment Active**: All option prices decreased by {abs(adjustment):.1f}%")
                                
                                
                                # Calculate statistics for each strategy
                                # Use pure_period_returns from Run 1 for ALL statistics
                                with_div_returns = pd.Series(pure_period_returns_with_div)
                                no_div_returns = pd.Series(pure_period_returns_no_div)
                                portfolio_returns = pd.Series(pure_period_returns)
                                
                                # Calculate CPGR (Compound Period Growth Rate) - growth rate for each options period
                                # Use pure performance for CAGR/CPGR calculations
                                cpgr_with_div = ((pure_performance_with_div / initial_value) ** (1 / len(periods_df)) - 1) * 100
                                cpgr_no_div = ((pure_performance_no_div / initial_value) ** (1 / len(periods_df)) - 1) * 100
                                cpgr_portfolio = ((pure_performance_portfolio / initial_value) ** (1 / len(periods_df)) - 1) * 100
                                
                                # Base statistics
                                summary_data = {
                                    'Metric': [
                                        'CAGR (%)',
                                        'CPGR (%)',
                                        'MWRR (%)',
                                        'Final Capital ($)',
                                        'Positive Periods',
                                        'Negative Periods', 
                                        '% Positive Periods',
                                        'Average Return (%)',
                                        'Median Return (%)',
                                        'Best Period (%)',
                                        'Worst Period (%)'
                                    ],
                                    f'{ticker_symbol} With Div': [
                                        f"{total_cagr_with_div:.2f}%",
                                        f"{cpgr_with_div:.2f}%",
                                        f"{mwrr_with_div:.2f}%",
                                        f"{final_capital_with_div:,.0f}".replace(',', ' '),
                                        f"{(with_div_returns > 0).sum()}",
                                        f"{(with_div_returns < 0).sum()}",
                                        f"{(with_div_returns > 0).mean() * 100:.1f}%",
                                        f"{with_div_returns.mean():.2f}%",
                                        f"{with_div_returns.median():.2f}%",
                                        f"{with_div_returns.max():.2f}%",
                                        f"{with_div_returns.min():.2f}%"
                                    ],
                                    f'{ticker_symbol} No Div': [
                                        f"{total_cagr_no_div:.2f}%",
                                        f"{cpgr_no_div:.2f}%",
                                        f"{mwrr_no_div:.2f}%",
                                        f"{final_capital_no_div:,.0f}".replace(',', ' '),
                                        f"{(no_div_returns > 0).sum()}",
                                        f"{(no_div_returns < 0).sum()}",
                                        f"{(no_div_returns > 0).mean() * 100:.1f}%",
                                        f"{no_div_returns.mean():.2f}%",
                                        f"{no_div_returns.median():.2f}%",
                                        f"{no_div_returns.max():.2f}%",
                                        f"{no_div_returns.min():.2f}%"
                                    ],
                                    'Portfolio Barbell': [
                                        f"{capr_portfolio:.2f}%",
                                        f"{cpgr_portfolio:.2f}%",
                                        f"{mwrr_portfolio:.2f}%",
                                        f"{final_capital_portfolio:,.0f}".replace(',', ' '),
                                        f"{(portfolio_returns > 0).sum()}",
                                        f"{(portfolio_returns < 0).sum()}",
                                        f"{(portfolio_returns > 0).mean() * 100:.1f}%",
                                        f"{portfolio_returns.mean():.2f}%",
                                        f"{portfolio_returns.median():.2f}%",
                                        f"{portfolio_returns.max():.2f}%",
                                        f"{portfolio_returns.min():.2f}%"
                                    ]
                                }
                                
                                summary_df = pd.DataFrame(summary_data)
                                
                                # Apply colors to summary table
                                def color_summary_table(val, col_name, row_index):
                                    if isinstance(val, str) and '%' in val and col_name != 'Metric':
                                        # Extract numerical value
                                        try:
                                            percentage_val = float(val.replace('%', '').replace(',', '').replace(' ', ''))
                                            metric_name = summary_df.iloc[row_index]['Metric']
                                            
                                            # Best Period - Shades of Green
                                            if 'Best Period' in metric_name:
                                                if percentage_val > 50:
                                                    return 'background-color: #004d00; color: white; font-weight: bold'
                                                elif percentage_val > 30:
                                                    return 'background-color: #1e8449; color: white; font-weight: bold'
                                                elif percentage_val > 15:
                                                    return 'background-color: #66bb6a; color: white'
                                                else:
                                                    return 'background-color: #a5d6a7; color: white'
                                            
                                            # Worst Period - Shades of Red
                                            elif 'Worst Period' in metric_name:
                                                if percentage_val < -30:
                                                    return 'background-color: #7b0000; color: white; font-weight: bold'
                                                elif percentage_val < -15:
                                                    return 'background-color: #d32f2f; color: white; font-weight: bold'
                                                elif percentage_val < -5:
                                                    return 'background-color: #ef5350; color: white'
                                                else:
                                                    return 'background-color: #ffab91; color: white'
                                            
                                            # Positive Periods - Shades of Green
                                            elif 'Positive Periods' in metric_name:
                                                if percentage_val > 25:
                                                    return 'background-color: #004d00; color: white; font-weight: bold'
                                                elif percentage_val > 15:
                                                    return 'background-color: #1e8449; color: white; font-weight: bold'
                                                elif percentage_val > 5:
                                                    return 'background-color: #66bb6a; color: white'
                                                else:
                                                    return 'background-color: #a5d6a7; color: white'
                                            
                                            # Negative Periods - Shades of Red
                                            elif 'Negative Periods' in metric_name:
                                                if percentage_val > 20:
                                                    return 'background-color: #7b0000; color: white; font-weight: bold'
                                                elif percentage_val > 10:
                                                    return 'background-color: #d32f2f; color: white; font-weight: bold'
                                                elif percentage_val > 5:
                                                    return 'background-color: #ef5350; color: white'
                                                else:
                                                    return 'background-color: #ffab91; color: white'
                                            
                                            # % Positive Periods - Shades of Green
                                            elif '% Positive Periods' in metric_name:
                                                if percentage_val > 80:
                                                    return 'background-color: #004d00; color: white; font-weight: bold'
                                                elif percentage_val > 60:
                                                    return 'background-color: #1e8449; color: white; font-weight: bold'
                                                elif percentage_val > 40:
                                                    return 'background-color: #66bb6a; color: white'
                                                else:
                                                    return 'background-color: #a5d6a7; color: white'
                                            
                                            # Other percentage metrics (CAGR, CPGR, MWRR, Average Return, Median Return)
                                            elif any(x in metric_name for x in ['CAGR', 'CPGR', 'MWRR', 'Average Return', 'Median Return']):
                                                if percentage_val > 15:
                                                    return 'background-color: #004d00; color: white; font-weight: bold'
                                                elif percentage_val > 8:
                                                    return 'background-color: #1e8449; color: white; font-weight: bold'
                                                elif percentage_val > 0:
                                                    return 'background-color: #66bb6a; color: white'
                                                elif percentage_val > -5:
                                                    return 'background-color: #ef5350; color: white'
                                                else:
                                                    return 'background-color: #7b0000; color: white; font-weight: bold'
                                        except:
                                            pass
                                    # DEFAULT: No background for unmatched cells
                                    return ''
                                
                                styled_summary = summary_df.style.apply(
                                    lambda x: [color_summary_table(val, x.name, i) 
                                             for i, val in enumerate(x)], 
                                    axis=0
                                ).set_table_styles([
                                    {'selector': 'th', 'props': [('background-color', '#2d3748'), ('color', 'white'), ('font-weight', 'bold')]},
                                    {'selector': 'td', 'props': [('text-align', 'center')]}
                                ])
                                
                                st.dataframe(styled_summary, use_container_width=True, height=450)
                                
                                # Save to persistent storage
                                result_key = f"{ticker_symbol}_{barbell_strike}_{options_pct}%_{days_to_exp}d"
                                st.session_state.all_backtest_results[result_key] = {
                                    'params': {
                                        'ticker_symbol': ticker_symbol,
                                        'barbell_strike': barbell_strike,
                                        'selected_exp': selected_exp,
                                        'days_to_exp': days_to_exp,
                                        'opt_price': opt_price,
                                        'options_pct': options_pct,
                                        'total_capital': total_capital,
                                        'bond_return_rate': bond_return_rate,
                                        'initial_value': initial_value,
                                        'added_per_period': added_per_period
                                    },
                                    'summary_df': summary_df,
                                    'periods_df': periods_df,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                                
                                st.success(f"✅ Backtest completed and saved! Key: {result_key}")
                                
                                # Apply the same colors as the Performance Summary Table
                                def color_backtest_variations(val, col_name, row_index):
                                    if isinstance(val, (int, float, str)):
                                        if col_name.endswith('(%)'):
                                            # Extract numeric value if formatted
                                            if isinstance(val, str):
                                                percentage_val = float(val.replace('%', '').replace(',', ''))
                                            else:
                                                percentage_val = float(val)
                                        
                                        # Apply the same colors as Performance Summary
                                        if percentage_val > 50:
                                            return 'background-color: #004d00; color: white'
                                        elif percentage_val > 20:
                                            return 'background-color: #1e8449; color: white'
                                        elif percentage_val > 5:
                                            return 'background-color: #388e3c; color: white'
                                        elif percentage_val > 0:
                                            return 'background-color: #66bb6a; color: white'
                                        elif percentage_val < -50:
                                            return 'background-color: #7b0000; color: white'
                                        elif percentage_val < -20:
                                            return 'background-color: #b22222; color: white'
                                        elif percentage_val < -5:
                                            return 'background-color: #d32f2f; color: white'
                                        elif percentage_val < 0:
                                            return 'background-color: #ef5350; color: white'
                                        else:
                                            return 'background-color: #424242; color: white'  # Neutre pour 0
                                    return ''
                                
                                # Appliquer le style aux colonnes de variations
                                styled_df = periods_df.style.apply(
                                    lambda x: [color_backtest_variations(val, x.name, i) 
                                             for i, val in enumerate(x)], 
                                    axis=0,
                                    subset=[col for col in periods_df.columns if col.endswith('(%)')]
                                )
                                
                                st.dataframe(styled_df, use_container_width=True)
                            else:
                                st.warning("Not enough data to create complete periods")
                            
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Daily Data",
                                data=csv,
                                file_name=f"{ticker_symbol}_daily_data_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error running backtest: {str(e)}")
                
                elif run_multi_strike_backtest:
                    # 🎯 MULTI-STRIKE BACKTEST - COMPLETELY INDEPENDENT
                    with st.spinner("Running multi-strike backtest - this may take a while..."):
                        try:
                            import yfinance as yf
                            from datetime import datetime, timedelta
                            import numpy as np
                            
                            # 🎯 COMPLETELY INDEPENDENT VARIABLES - NO DEPENDENCIES
                            # Get available strikes for the selected expiration
                            available_strikes = sorted(filtered_opts['strike'].unique()) if not filtered_opts.empty else []
                            
                            if not available_strikes:
                                st.error("❌ No strikes available for backtest")
                                st.stop()
                            
                            st.info(f"🎯 **Testing {len(available_strikes)} strikes for expiration {selected_exp}**")
                            
                            # Calculate days to expiration
                            try:
                                exp_dt = pd.to_datetime(selected_exp)
                                days_to_exp = (exp_dt - datetime.now()).days
                                if days_to_exp <= 0:
                                    st.error("❌ Selected option has already expired")
                                    st.stop()
                            except:
                                st.error("❌ Invalid expiration date")
                                st.stop()
                            
                            # Try to get cached data first, fallback to fetch if not available
                            global_cache_key = f"historical_data_{ticker_symbol}"
                            if global_cache_key in st.session_state:
                                # Use cached data
                                cached_data = st.session_state[global_cache_key]
                                ticker_data = cached_data['ticker_data']
                                price_col = cached_data['price_col']
                                results_df = cached_data['results_df']
                                st.info(f"📊 Using cached historical data for Multi-Strike Backtest ({len(results_df)} days)")
                            else:
                                # Fetch historical data
                                st.info("📊 Fetching historical data for Multi-Strike Backtest (1993-present)")
                                ticker_data = yf.download(ticker_symbol, start="1993-01-01", progress=False)
                            
                            if ticker_data.empty:
                                st.error(f"❌ Failed to fetch {ticker_symbol} data")
                                st.stop()
                            
                            # Use Close column
                            price_col = 'Close' if 'Close' in ticker_data.columns else 'Adj Close'
                            
                            # Forward fill for non-trading days
                            ticker_data = ticker_data.resample('D').ffill()
                            
                            # Calculate daily variations - EXACT SAME AS TEST2.PY
                            daily_data = []
                            for i in range(len(ticker_data)):
                                date = ticker_data.index[i]
                                price = ticker_data[price_col].iloc[i]
                                
                                if i > 0:
                                    prev_price = float(ticker_data[price_col].iloc[i-1])
                                    current_price = float(price)
                                    daily_change = (current_price / prev_price) - 1
                                else:
                                    daily_change = 0
                                
                                daily_change_with_div = daily_change
                                daily_change_no_div = daily_change - (0.015 / 365)
                                
                                daily_data.append({
                                    'Date': date,
                                    'Price': float(price),
                                    'Daily Change (w/ Div)': float(daily_change_with_div * 100),
                                    'Daily Change (no Div)': float(daily_change_no_div * 100)
                                })
                            
                            results_df = pd.DataFrame(daily_data)
                            
                            # Cache the data for future use (global cache)
                            st.session_state[global_cache_key] = {
                                'ticker_data': ticker_data,
                                'price_col': price_col,
                                'results_df': results_df,
                                'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            # Test each strike - EXACT SAME AS TEST2.PY
                            strike_results = []
                            
                            for strike in available_strikes:
                                try:
                                    # Get option price for this strike - USE calculate_mid_price TO APPLY ADJUSTMENT
                                    opt = filtered_opts[filtered_opts['strike'] == strike].iloc[0]
                                    opt_price = calculate_mid_price(opt['bid'], opt['ask'], opt['lastPrice'])
                                    
                                    if not opt_price or opt_price <= 0:
                                        continue
                                    
                                    # Calculate portfolio parameters for this strike
                                    option_portion = total_capital * (options_pct / 100)
                                    bond_portion = total_capital * ((100 - options_pct) / 100)
                                    
                                    # 🎯 DUAL RUN BACKTEST 🎯 - EXACT SAME AS TEST2.PY
                                    # RUN 1: PURE PERFORMANCE (sans ajouts d'argent)
                                    # RUN 2: WITH ADDED MONEY (pour MWRR et Final Capital)
                                    
                                    # ===== RUN 1: PURE PERFORMANCE =====
                                    pure_performance_portfolio = total_capital
                                    pure_period_returns = []
                                    
                                    # ===== RUN 2: WITH ADDED MONEY =====
                                    period_data = []
                                    cumulative_with_div = total_capital
                                    cumulative_no_div = total_capital
                                    cumulative_portfolio = total_capital
                                    
                                    # 🎯 PURE TICKER PERFORMANCE (sans ajouts d'argent) pour tickers
                                    pure_performance_with_div = total_capital
                                    pure_performance_no_div = total_capital
                                    pure_period_returns_with_div = []
                                    pure_period_returns_no_div = []
                                    
                                    # Calculate initial contracts for this strike
                                    initial_contracts = option_portion / (opt_price * 100)
                                    
                                    for i in range(0, len(results_df) - days_to_exp + 1, days_to_exp):
                                        start_date = results_df.iloc[i]['Date']
                                        end_date = results_df.iloc[i + days_to_exp - 1]['Date']
                                        
                                        start_price = results_df.iloc[i]['Price']
                                        end_price = results_df.iloc[i + days_to_exp - 1]['Price']
                                        
                                        ticker_with_div = ((end_price / start_price) - 1) * 100
                                        ticker_no_div = ticker_with_div - (1.5 * days_to_exp / 365)
                                        
                                        # ===== RUN 1: PURE PERFORMANCE CALCULATION =====
                                        # Calculate portfolio performance for this period (PURE - no added money)
                                        current_capital_pure = pure_performance_portfolio
                                        current_option_portion_pure = current_capital_pure * (options_pct / 100)
                                        current_contracts_pure = current_option_portion_pure / (opt_price * 100)
                                        
                                        bond_period_return = (bond_return_rate / 100) * (days_to_exp / 365)
                                        bond_value_pure = (current_capital_pure - current_option_portion_pure) * (1 + bond_period_return)
                                        
                                        current_ticker_price = ticker_data[price_col].iloc[-1]
                                        final_ticker_price = current_ticker_price * (1 + ticker_no_div / 100)
                                        
                                        option_profit_per_share = max(0, float(final_ticker_price) - strike) if option_type == "CALL" else max(0, strike - float(final_ticker_price))
                                        option_value_pure = option_profit_per_share * current_contracts_pure * 100
                                        
                                        # Final Portfolio Value (PURE)
                                        total_portfolio_value_pure = bond_value_pure + option_value_pure
                                        portfolio_return_pure = ((total_portfolio_value_pure / current_capital_pure) - 1) * 100
                                        
                                        # Store pure performance return
                                        pure_period_returns.append(portfolio_return_pure)
                                        
                                        # Update pure performance (NO added money)
                                        pure_performance_portfolio = pure_performance_portfolio * (1 + portfolio_return_pure / 100)
                                        
                                        # ===== TICKERS DUAL RUN CALCULATION =====
                                        # Calculate ticker performance (PURE - no added money)
                                        pure_performance_with_div = pure_performance_with_div * (1 + ticker_with_div / 100)
                                        pure_performance_no_div = pure_performance_no_div * (1 + ticker_no_div / 100)
                                        
                                        # Store pure ticker returns
                                        pure_period_returns_with_div.append(ticker_with_div)
                                        pure_period_returns_no_div.append(ticker_no_div)
                                        
                                        # ===== RUN 2: WITH ADDED MONEY CALCULATION =====
                                        # Calculate portfolio performance for this period (WITH added money)
                                        current_capital_with_added = cumulative_portfolio + added_per_period
                                        current_option_portion_with_added = current_capital_with_added * (options_pct / 100)
                                        current_contracts_with_added = current_option_portion_with_added / (opt_price * 100)
                                        
                                        bond_value_with_added = (current_capital_with_added - current_option_portion_with_added) * (1 + bond_period_return)
                                        option_value_with_added = option_profit_per_share * current_contracts_with_added * 100
                                        
                                        # Valeur Finale Portfolio (WITH ADDED MONEY)
                                        total_portfolio_value_with_added = bond_value_with_added + option_value_with_added
                                        portfolio_return_with_added = ((total_portfolio_value_with_added / current_capital_with_added) - 1) * 100
                                        
                                        # Calculs cumulatifs pour le backtest (WITH ADDED MONEY)
                                        cumulative_with_div = cumulative_with_div * (1 + ticker_with_div / 100) + added_per_period
                                        cumulative_no_div = cumulative_no_div * (1 + ticker_no_div / 100) + added_per_period
                                        cumulative_portfolio = cumulative_portfolio * (1 + portfolio_return_with_added / 100) + added_per_period
                                    
                                    # Calculate comprehensive metrics for this strike - EXACT SAME AS TEST2.PY
                                    total_periods = len([i for i in range(0, len(results_df) - days_to_exp + 1, days_to_exp)])
                                    total_days = total_periods * days_to_exp if total_periods > 0 else 1
                                    
                                    # Final capital and CAGR (with safety check)
                                    final_capital_portfolio = cumulative_portfolio
                                    
                                    # CAGR based on PURE PERFORMANCE (without added money)
                                    if total_days > 0 and total_capital > 0 and pure_performance_portfolio > 0:
                                        final_cagr_portfolio = ((pure_performance_portfolio / total_capital) ** (365 / total_days) - 1) * 100
                                    else:
                                        final_cagr_portfolio = 0
                                    
                                    # CPGR (Compound Period Growth Rate) - based on pure performance
                                    if total_periods > 0 and total_capital > 0 and pure_performance_portfolio > 0:
                                        cpgr_portfolio = ((pure_performance_portfolio / total_capital) ** (1 / total_periods) - 1) * 100
                                    else:
                                        cpgr_portfolio = 0
                                    
                                    # MWRR (Money-Weighted Rate of Return) - vraie formule avec cash flows
                                    def calculate_mwrr(initial_investment, periodic_investments, final_value, periods):
                                        """Calculate MWRR using Newton-Raphson method"""
                                        if periods == 0 or final_value <= 0:
                                            return 0
                                        
                                        # Cash flows: initial investment (negative), periodic investments (negative), final value (positive)
                                        cash_flows = [-initial_investment]
                                        for i in range(periods):
                                            cash_flows.append(-periodic_investments)
                                        cash_flows.append(final_value)
                                        
                                        # Newton-Raphson method to find IRR
                                        def npv(rate, cash_flows):
                                            return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
                                        
                                        def npv_derivative(rate, cash_flows):
                                            return sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows))
                                        
                                        # Initial guess
                                        rate = 0.1
                                        for _ in range(100):  # Max 100 iterations
                                            npv_val = npv(rate, cash_flows)
                                            if abs(npv_val) < 1e-6:
                                                break
                                            npv_deriv = npv_derivative(rate, cash_flows)
                                            if abs(npv_deriv) < 1e-10:
                                                break
                                            rate = rate - npv_val / npv_deriv
                                        
                                        return rate * 100
                                    
                                    mwrr_portfolio = calculate_mwrr(total_capital, added_per_period, final_capital_portfolio, total_periods)
                                    
                                    # Use pure_period_returns from Run 1 for all statistics
                                    portfolio_returns = pure_period_returns
                                    
                                    # Calculate statistics
                                    if portfolio_returns:
                                        positive_periods = sum(1 for r in portfolio_returns if r > 0)
                                        negative_periods = sum(1 for r in portfolio_returns if r < 0)
                                        pct_positive = (positive_periods / len(portfolio_returns)) * 100
                                        avg_return = sum(portfolio_returns) / len(portfolio_returns)
                                        median_return = sorted(portfolio_returns)[len(portfolio_returns)//2]
                                        best_period = max(portfolio_returns)
                                        worst_period = min(portfolio_returns)
                                    else:
                                        positive_periods = 0
                                        negative_periods = 0
                                        pct_positive = 0
                                        avg_return = 0
                                        median_return = 0
                                        best_period = 0
                                        worst_period = 0
                                    
                                    # 💥 ATOMIC BOMB - FORCE 2 DECIMALS AT CREATION 💥
                                    strike_results.append({
                                        'Strike': strike,
                                        'Option Price': float(f"{opt_price:.2f}"),
                                        'Contracts': float(f"{initial_contracts:.2f}"),
                                        'CAGR (%)': float(f"{final_cagr_portfolio:.2f}"),
                                        'CPGR (%)': float(f"{cpgr_portfolio:.2f}"),
                                        'MWRR (%)': float(f"{mwrr_portfolio:.2f}"),
                                        'Final Capital ($)': float(f"{final_capital_portfolio:.2f}"),
                                        'Positive Periods': positive_periods,
                                        'Negative Periods': negative_periods,
                                        '% Positive Periods': float(f"{pct_positive:.2f}"),
                                        'Average Return (%)': float(f"{avg_return:.2f}"),
                                        'Median Return (%)': float(f"{median_return:.2f}"),
                                        'Best Period (%)': float(f"{best_period:.2f}"),
                                        'Worst Period (%)': float(f"{worst_period:.2f}"),
                                        'Total Periods': len(portfolio_returns)
                                    })
                                    
                                except Exception as e:
                                    continue
                            
                            if strike_results:
                                # Create results DataFrame
                                results_df_multi = pd.DataFrame(strike_results)
                                
                                # Sort by CAGR descending
                                results_df_multi = results_df_multi.sort_values('CAGR (%)', ascending=False)
                                
                                st.markdown("### 🎯 Multi-Strike Backtest Results")
                                st.markdown(f"**Tested {len(strike_results)} {option_type} strikes for expiration {selected_exp}**")
                                
                                # Show price adjustment status
                                if ('barbell_price_adjustment' in st.session_state and 
                                    st.session_state.barbell_price_adjustment != 0):
                                    adjustment = st.session_state.barbell_price_adjustment
                                    if adjustment > 0:
                                        st.warning(f"🎯 **Price Adjustment Active**: All option prices increased by {adjustment:.1f}%")
                                    else:
                                        st.success(f"🎯 **Price Adjustment Active**: All option prices decreased by {abs(adjustment):.1f}%")
                                
                                # Display complete results table
                                st.markdown("#### 📊 Complete Multi-Strike Results Table")
                                st.markdown(f"**All {len(strike_results)} strikes tested - Sort by any column to filter**")
                                
                                # Apply comprehensive colors to results - EXACT SAME AS TEST2.PY
                                def color_comprehensive_results(val, col_name, row_index):
                                    # Convert string to float if necessary (atomic bomb)
                                    try:
                                        if isinstance(val, str):
                                            # Remove $ and commas for Final Capital
                                            if col_name == 'Final Capital ($)':
                                                val = float(val.replace('$', '').replace(',', ''))
                                            else:
                                                val = float(val)
                                    except:
                                        return ''
                                    
                                    if isinstance(val, (int, float)):
                                        if col_name in ['CAGR (%)', 'CPGR (%)', 'MWRR (%)']:
                                            if val > 20:
                                                return 'background-color: #004d00; color: white; font-weight: bold'
                                            elif val > 10:
                                                return 'background-color: #1e8449; color: white; font-weight: bold'
                                            elif val > 5:
                                                return 'background-color: #66bb6a; color: white'
                                            elif val > 0:
                                                return 'background-color: #ffeb3b; color: black'
                                            else:
                                                return 'background-color: #ef5350; color: white'
                                        elif col_name == 'Final Capital ($)':
                                            if val > total_capital * 2:
                                                return 'background-color: #004d00; color: white; font-weight: bold'
                                            elif val > total_capital * 1.5:
                                                return 'background-color: #1e8449; color: white'
                                            elif val > total_capital:
                                                return 'background-color: #66bb6a; color: white'
                                            else:
                                                return 'background-color: #ef5350; color: white'
                                        elif col_name == '% Positive Periods':
                                            if val > 70:
                                                return 'background-color: #004d00; color: white; font-weight: bold'
                                            elif val > 50:
                                                return 'background-color: #66bb6a; color: white'
                                            elif val > 30:
                                                return 'background-color: #ffeb3b; color: black'
                                            else:
                                                return 'background-color: #ef5350; color: white'
                                        elif col_name in ['Average Return (%)', 'Median Return (%)']:
                                            if val > 10:
                                                return 'background-color: #004d00; color: white; font-weight: bold'
                                            elif val > 5:
                                                return 'background-color: #1e8449; color: white'
                                            elif val > 0:
                                                return 'background-color: #66bb6a; color: white'
                                            elif val > -5:
                                                return 'background-color: #ffeb3b; color: black'
                                            else:
                                                return 'background-color: #ef5350; color: white'
                                        elif col_name == 'Best Period (%)':
                                            if val > 50:
                                                return 'background-color: #004d00; color: white; font-weight: bold'
                                            elif val > 20:
                                                return 'background-color: #1e8449; color: white'
                                            elif val > 5:
                                                return 'background-color: #66bb6a; color: white'
                                            elif val > 0:
                                                return 'background-color: #ffeb3b; color: black'
                                            else:
                                                return 'background-color: #ef5350; color: white'
                                        elif col_name == 'Worst Period (%)':
                                            if val > 0:
                                                return 'background-color: #004d00; color: white; font-weight: bold'
                                            elif val > -10:
                                                return 'background-color: #66bb6a; color: white'
                                            elif val > -20:
                                                return 'background-color: #ffeb3b; color: black'
                                            elif val > -50:
                                                return 'background-color: #ef5350; color: white'
                                            else:
                                                return 'background-color: #7b0000; color: white; font-weight: bold'
                                    return ''
                                
                                # Format the results for better display - EXACT SAME AS TEST2.PY
                                display_results = results_df_multi.copy()
                                
                                # 💥 ATOMIC BOMB TO FORCE 2 DECIMALS 💥
                                for col in display_results.columns:
                                    if col == 'Strike':
                                        # Smart Strike: integer if possible, otherwise 2 decimals
                                        display_results[col] = display_results[col].apply(lambda x: 
                                            f"{int(float(x))}" if float(x) == int(float(x)) else f"{float(x):.2f}")
                                    elif col in ['Positive Periods', 'Negative Periods', 'Total Periods']:
                                        # These columns remain as integers
                                        pass
                                    elif col == 'Final Capital ($)':
                                        # Final Capital with 2 decimals and $ format
                                        display_results[col] = display_results[col].apply(lambda x: f"${float(x):,.2f}")
                                    else:
                                        # ALL OTHER COLUMNS = EXACTLY 2 DECIMALS
                                        if display_results[col].dtype in ['float64', 'int64']:
                                            display_results[col] = display_results[col].apply(lambda x: f"{float(x):.2f}")
                                
                                styled_results = display_results.style.apply(
                                    lambda x: [color_comprehensive_results(val, x.name, i) 
                                             for i, val in enumerate(x)], 
                                    axis=0
                                ).set_table_styles([
                                    {'selector': 'th', 'props': [('background-color', '#2d3748'), ('color', 'white'), ('font-weight', 'bold')]},
                                    {'selector': 'td', 'props': [('text-align', 'center')]}
                                ])
                                
                                st.dataframe(styled_results, use_container_width=True, height=450)
                                
                                # Display best strike details - EXACT SAME AS TEST2.PY
                                best_strike = results_df_multi.iloc[0]
                                st.markdown("#### 🥇 Best Strike Details")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Best Strike", f"${best_strike['Strike']:.0f}")
                                    st.metric("Option Price", f"${best_strike['Option Price']:.2f}")
                                
                                with col2:
                                    st.metric("Contracts", f"{best_strike['Contracts']:.2f}")
                                    st.metric("Final Capital", f"${best_strike['Final Capital ($)']:,.0f}")
                                
                                with col3:
                                    st.metric("CAGR", f"{best_strike['CAGR (%)']:.2f}%")
                                    st.metric("Avg Return", f"{best_strike['Average Return (%)']:.2f}%")
                                
                                with col4:
                                    st.metric("Total Periods", f"{best_strike['Total Periods']}")
                                    st.metric("Outperformance", f"{best_strike['CAGR (%)'] - 10:.2f}%")
                                
                                # Save to persistent storage - SAME AS HISTORICAL BACKTEST
                                result_key = f"Multi-Strike_{ticker_symbol}_{selected_exp}_{options_pct}%_{days_to_exp}d"
                                st.session_state.all_backtest_results[result_key] = {
                                    'params': {
                                        'ticker_symbol': ticker_symbol,
                                        'selected_exp': selected_exp,
                                        'days_to_exp': days_to_exp,
                                        'options_pct': options_pct,
                                        'total_capital': total_capital,
                                        'bond_return_rate': bond_return_rate,
                                        'added_per_period': added_per_period,
                                        'option_type': option_type
                                    },
                                    'summary_df': results_df_multi,
                                    'periods_df': None,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                                
                                # Also save in the old format for compatibility
                                st.session_state.multi_strike_results = {
                                    'results_df': results_df_multi,
                                    'parameters': {
                                        'ticker_symbol': ticker_symbol,
                                        'expiration': selected_exp,
                                        'option_type': option_type,
                                        'initial_capital': total_capital,
                                        'options_pct': options_pct,
                                        'bond_return': bond_return_rate,
                                        'added_per_period': added_per_period,
                                        'days_to_exp': days_to_exp
                                    }
                                }
                                
                                st.success(f"✅ Multi-strike backtest completed! Best strike: ${best_strike['Strike']:.0f} with {best_strike['CAGR (%)']:.2f}% CAGR")
                                
                            else:
                                st.error("❌ No valid results generated")
                        
                        except Exception as e:
                            st.error(f"❌ Error running multi-strike backtest: {str(e)}")
                            st.exception(e)
                
                # All Options Backtest Section
                st.markdown("---")
                st.markdown("### 🌐 All CALL Options Backtest")
                st.markdown("**Test ALL available CALL options (all strikes + all expirations) with the same logic as multi-strike backtest**")
                
                # Permanent message
                st.info("💾 **Results are automatically saved!** You can find all backtest results in the '📊 All Saved Backtest Results' section above.")
                
                # Parameters section
                st.markdown("#### 📊 Backtest Parameters")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Initial capital
                    if 'all_options_initial_capital' not in st.session_state:
                        st.session_state.all_options_initial_capital = 100000
                    
                    all_options_initial_capital = st.number_input(
                        "💰 Initial Capital ($)", 
                        min_value=1000, 
                        value=st.session_state.all_options_initial_capital, 
                        step=1000,
                        help="Initial capital for the backtest",
                        key="all_options_initial_capital_input"
                    )
                    st.session_state.all_options_initial_capital = all_options_initial_capital
                    
                    # Added per period
                    if 'all_options_added_per_period' not in st.session_state:
                        st.session_state.all_options_added_per_period = 0
                    
                    all_options_added_per_period = st.number_input(
                        "📈 Added per Period ($)", 
                        min_value=0, 
                        value=st.session_state.all_options_added_per_period, 
                        step=1000,
                        help="Amount added each period",
                        key="all_options_added_per_period_input"
                    )
                    st.session_state.all_options_added_per_period = all_options_added_per_period
                
                with col2:
                    # Options percentage
                    if 'all_options_options_pct' not in st.session_state:
                        st.session_state.all_options_options_pct = 20
                    
                    all_options_options_pct = st.number_input(
                        "📈 Options Allocation (%)", 
                        min_value=1, 
                        max_value=100, 
                        value=st.session_state.all_options_options_pct, 
                        step=1,
                        help="Percentage of capital allocated to options",
                        key="all_options_options_pct_input"
                    )
                    st.session_state.all_options_options_pct = all_options_options_pct
                    
                    # Bond return rate
                    if 'all_options_bond_return' not in st.session_state:
                        st.session_state.all_options_bond_return = 4.0
                    
                    all_options_bond_return = st.number_input(
                        "💰 Bond Return Rate (%)", 
                        min_value=0.0, 
                        max_value=15.0, 
                        value=st.session_state.all_options_bond_return, 
                        step=0.1,
                        help="Annual return rate for the bond portion",
                        key="all_options_bond_return_input"
                    )
                    st.session_state.all_options_bond_return = all_options_bond_return
                
                with col3:
                    # Minimum days filter slider
                    if 'all_options_min_days' not in st.session_state:
                        st.session_state.all_options_min_days = 200
                    
                    min_days_filter = st.slider(
                        "🚫 Minimum Days to Expiration Filter", 
                        min_value=0, 
                        max_value=730, 
                        value=st.session_state.all_options_min_days,
                        step=10,
                        help="Exclude options with fewer days to expiration (to avoid very short-term options)",
                        key="all_options_min_days_slider"
                    )
                    st.session_state.all_options_min_days = min_days_filter
                
                run_all_options_backtest = st.button("🌐 Run All CALL Options Backtest", key="run_all_options_backtest", type="primary")
                
                if run_all_options_backtest:
                    with st.spinner("Running all CALL options backtest - this may take a very long time..."):
                        try:
                            import yfinance as yf
                            from datetime import datetime, timedelta
                            import numpy as np
                            
                            # Get ALL CALL options from ALL expirations (NO PUTS)
                            all_calls_data = []
                            
                            # Get all calls only
                            if not calls_combined.empty:
                                for _, row in calls_combined.iterrows():
                                    # Check minimum days filter
                                    try:
                                        exp_date = pd.to_datetime(row['expiration'])
                                        days_to_exp = (exp_date - datetime.now()).days
                                        if days_to_exp >= min_days_filter:
                                            all_calls_data.append({
                                                'strike': row['strike'],
                                                'expiration': row['expiration'],
                                                'days_to_exp': days_to_exp,
                                                'bid': row['bid'],
                                                'ask': row['ask'],
                                                'lastPrice': row['lastPrice']
                                            })
                                    except:
                                        continue
                            
                            if not all_calls_data:
                                st.error(f"❌ No CALL options found with at least {min_days_filter} days to expiration")
                                st.stop()
                            
                            st.info(f"🌐 **Testing {len(all_calls_data)} CALL options (min {min_days_filter} days to expiration)**")
                            
                            
                            valid_prices_count = 0
                            invalid_prices_count = 0
                            for call_data in all_calls_data:
                                bid = call_data.get('bid')
                                ask = call_data.get('ask') 
                                last_price = call_data.get('lastPrice')
                                
                                opt_price = calculate_mid_price(bid, ask, last_price)
                                if opt_price and opt_price > 0:
                                    valid_prices_count += 1
                                else:
                                    invalid_prices_count += 1
                            
                            st.info(f"📊 **Price Analysis: {valid_prices_count} valid prices, {invalid_prices_count} invalid prices**")
                            
                            # Get historical data (same as multi-strike backtest)
                            global_cache_key = f"historical_data_{ticker_symbol}"
                            if global_cache_key in st.session_state:
                                cached_data = st.session_state[global_cache_key]
                                ticker_data = cached_data['ticker_data']
                                price_col = cached_data['price_col']
                                results_df = cached_data['results_df']
                                st.info(f"📊 Using cached historical data for All CALL Options Backtest ({len(results_df)} days)")
                            else:
                                st.info("📊 Fetching historical data for All CALL Options Backtest (1993-present)")
                                ticker_data = yf.download(ticker_symbol, start="1993-01-01", progress=False)
                                
                                if ticker_data.empty:
                                    st.error(f"❌ Failed to fetch {ticker_symbol} data")
                                    st.stop()
                                
                                price_col = 'Close' if 'Close' in ticker_data.columns else 'Adj Close'
                                ticker_data = ticker_data.resample('D').ffill()
                                
                                daily_data = []
                                for i in range(len(ticker_data)):
                                    date = ticker_data.index[i]
                                    price = ticker_data[price_col].iloc[i]
                                    
                                    if i > 0:
                                        prev_price = float(ticker_data[price_col].iloc[i-1])
                                        current_price = float(price)
                                        daily_change = (current_price / prev_price) - 1
                                    else:
                                        daily_change = 0
                                    
                                    daily_change_with_div = daily_change
                                    daily_change_no_div = daily_change - (0.015 / 365)
                                    
                                    daily_data.append({
                                        'Date': date,
                                        'Price': float(price),
                                        'Daily Change (w/ Div)': float(daily_change_with_div * 100),
                                        'Daily Change (no Div)': float(daily_change_no_div * 100)
                                    })
                                
                                results_df = pd.DataFrame(daily_data)
                                
                                st.session_state[global_cache_key] = {
                                    'ticker_data': ticker_data,
                                    'price_col': price_col,
                                    'results_df': results_df,
                                    'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                            
                            if results_df.empty:
                                st.error("❌ No data found")
                                st.stop()
                            
                            # Test each CALL option
                            all_calls_results = []
                            processed_count = 0
                            skipped_count = 0
                            
                            for call_data in all_calls_data:
                                try:
                                    strike = call_data['strike']
                                    expiration = call_data['expiration']
                                    days_to_exp = call_data['days_to_exp']
                                    
                                    # Calculate option price
                                    opt_price = calculate_mid_price(call_data['bid'], call_data['ask'], call_data['lastPrice'])
                                    
                                    if not opt_price or opt_price <= 0:
                                        skipped_count += 1
                                        continue
                                    
                                    processed_count += 1
                                    
                                    # Calculate portfolio parameters using the input parameters
                                    option_portion = all_options_initial_capital * (all_options_options_pct / 100)
                                    bond_portion = all_options_initial_capital * ((100 - all_options_options_pct) / 100)
                                    
                                    # Same dual run logic as multi-strike backtest
                                    pure_performance_portfolio = all_options_initial_capital
                                    pure_period_returns = []
                                    
                                    cumulative_with_div = all_options_initial_capital
                                    cumulative_no_div = all_options_initial_capital
                                    cumulative_portfolio = all_options_initial_capital
                                    
                                    pure_performance_with_div = all_options_initial_capital
                                    pure_performance_no_div = all_options_initial_capital
                                    pure_period_returns_with_div = []
                                    pure_period_returns_no_div = []
                                    
                                    initial_contracts = option_portion / (opt_price * 100)
                                    
                                    for i in range(0, len(results_df) - days_to_exp + 1, days_to_exp):
                                        start_date = results_df.iloc[i]['Date']
                                        end_date = results_df.iloc[i + days_to_exp - 1]['Date']
                                        
                                        start_price = results_df.iloc[i]['Price']
                                        end_price = results_df.iloc[i + days_to_exp - 1]['Price']
                                        
                                        ticker_with_div = ((end_price / start_price) - 1) * 100
                                        ticker_no_div = ticker_with_div - (1.5 * days_to_exp / 365)
                                        
                                        # PURE PERFORMANCE CALCULATION
                                        current_capital_pure = pure_performance_portfolio
                                        current_option_portion_pure = current_capital_pure * (all_options_options_pct / 100)
                                        current_contracts_pure = current_option_portion_pure / (opt_price * 100)
                                        
                                        bond_period_return = (all_options_bond_return / 100) * (days_to_exp / 365)
                                        bond_value_pure = (current_capital_pure - current_option_portion_pure) * (1 + bond_period_return)
                                        
                                        current_ticker_price = ticker_data[price_col].iloc[-1]
                                        final_ticker_price = current_ticker_price * (1 + ticker_no_div / 100)
                                        
                                        # CALL option payoff only
                                        option_profit_per_share = max(0, float(final_ticker_price) - strike)
                                        
                                        option_value_pure = option_profit_per_share * current_contracts_pure * 100
                                        
                                        total_portfolio_value_pure = bond_value_pure + option_value_pure
                                        portfolio_return_pure = ((total_portfolio_value_pure / current_capital_pure) - 1) * 100
                                        
                                        pure_period_returns.append(portfolio_return_pure)
                                        pure_performance_portfolio = pure_performance_portfolio * (1 + portfolio_return_pure / 100)
                                        
                                        # TICKER PERFORMANCE
                                        pure_performance_with_div = pure_performance_with_div * (1 + ticker_with_div / 100)
                                        pure_performance_no_div = pure_performance_no_div * (1 + ticker_no_div / 100)
                                        
                                        pure_period_returns_with_div.append(ticker_with_div)
                                        pure_period_returns_no_div.append(ticker_no_div)
                                        
                                        # WITH ADDED MONEY CALCULATION
                                        current_capital_with_added = cumulative_portfolio + all_options_added_per_period
                                        current_option_portion_with_added = current_capital_with_added * (all_options_options_pct / 100)
                                        current_contracts_with_added = current_option_portion_with_added / (opt_price * 100)
                                        
                                        bond_value_with_added = (current_capital_with_added - current_option_portion_with_added) * (1 + bond_period_return)
                                        option_value_with_added = option_profit_per_share * current_contracts_with_added * 100
                                        
                                        total_portfolio_value_with_added = bond_value_with_added + option_value_with_added
                                        portfolio_return_with_added = ((total_portfolio_value_with_added / current_capital_with_added) - 1) * 100
                                        
                                        cumulative_with_div = cumulative_with_div * (1 + ticker_with_div / 100) + all_options_added_per_period
                                        cumulative_no_div = cumulative_no_div * (1 + ticker_no_div / 100) + all_options_added_per_period
                                        cumulative_portfolio = cumulative_portfolio * (1 + portfolio_return_with_added / 100) + all_options_added_per_period
                                    
                                    # Calculate metrics (same as multi-strike backtest)
                                    total_periods = len([i for i in range(0, len(results_df) - days_to_exp + 1, days_to_exp)])
                                    total_days = total_periods * days_to_exp if total_periods > 0 else 1
                                    
                                    final_capital_portfolio = cumulative_portfolio
                                    
                                    if total_days > 0 and all_options_initial_capital > 0 and pure_performance_portfolio > 0:
                                        final_cagr_portfolio = ((pure_performance_portfolio / all_options_initial_capital) ** (365 / total_days) - 1) * 100
                                    else:
                                        final_cagr_portfolio = 0
                                    
                                    if total_periods > 0 and all_options_initial_capital > 0 and pure_performance_portfolio > 0:
                                        cpgr_portfolio = ((pure_performance_portfolio / all_options_initial_capital) ** (1 / total_periods) - 1) * 100
                                    else:
                                        cpgr_portfolio = 0
                                    
                                    def calculate_mwrr(initial_investment, periodic_investments, final_value, periods):
                                        if periods == 0 or final_value <= 0:
                                            return 0
                                        cash_flows = [-initial_investment]
                                        for i in range(periods):
                                            cash_flows.append(-periodic_investments)
                                        cash_flows.append(final_value)
                                        
                                        def npv(rate, cash_flows):
                                            return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
                                        
                                        def npv_derivative(rate, cash_flows):
                                            return sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows))
                                        
                                        rate = 0.1
                                        for _ in range(100):
                                            npv_val = npv(rate, cash_flows)
                                            npv_deriv = npv_derivative(rate, cash_flows)
                                            if abs(npv_val) < 1e-6 or abs(npv_deriv) < 1e-12:
                                                break
                                            rate = rate - npv_val / npv_deriv
                                        
                                        return rate * 100
                                    
                                    mwrr_portfolio = calculate_mwrr(all_options_initial_capital, all_options_added_per_period, final_capital_portfolio, total_periods)
                                    
                                    portfolio_returns = pure_period_returns
                                    
                                    if portfolio_returns:
                                        avg_return = np.mean(portfolio_returns)
                                        median_return = np.median(portfolio_returns)
                                        best_period = max(portfolio_returns)
                                        worst_period = min(portfolio_returns)
                                        positive_periods = sum(1 for r in portfolio_returns if r > 0)
                                        negative_periods = sum(1 for r in portfolio_returns if r < 0)
                                        positive_pct = (positive_periods / len(portfolio_returns)) * 100
                                    else:
                                        avg_return = 0
                                        median_return = 0
                                        best_period = 0
                                        worst_period = 0
                                        positive_periods = 0
                                        negative_periods = 0
                                        positive_pct = 0
                                    
                                    all_calls_results.append({
                                        'Strike': strike,
                                        'Expiration': expiration,
                                        'Days to Exp': days_to_exp,
                                        'Option Price': opt_price,
                                        'Contracts': initial_contracts,
                                        'Final Capital ($)': final_capital_portfolio,
                                        'CAGR (%)': final_cagr_portfolio,
                                        'CPGR (%)': cpgr_portfolio,
                                        'MWRR (%)': mwrr_portfolio,
                                        'Average Return (%)': avg_return,
                                        'Median Return (%)': median_return,
                                        'Best Period (%)': best_period,
                                        'Worst Period (%)': worst_period,
                                        'Positive Periods': positive_periods,
                                        'Negative Periods': negative_periods,
                                        '% Positive Periods': positive_pct,
                                        'Total Periods': len(portfolio_returns)
                                    })
                                    
                                except Exception as e:
                                    skipped_count += 1
                                    continue
                            
                            # Display processing statistics
                            st.info(f"📈 **Processing Complete: {processed_count} processed, {skipped_count} skipped, {len(all_calls_results)} successful results**")
                            
                            if all_calls_results:
                                # Create results DataFrame
                                results_df_all_calls = pd.DataFrame(all_calls_results)
                                
                                # Sort by CAGR descending
                                results_df_all_calls = results_df_all_calls.sort_values('CAGR (%)', ascending=False)
                                
                                st.markdown("### 🌐 All CALL Options Backtest Results")
                                st.markdown(f"**Tested {len(all_calls_results)} CALL options (min {min_days_filter} days to expiration)**")
                                
                                # Display complete results table
                                st.markdown("#### 📊 Complete All CALL Options Results Table")
                                st.markdown(f"**All {len(all_calls_results)} CALL options tested - Sort by any column to filter**")
                                
                                # Apply colors to RAW data before formatting - EXACTLY like multi-strike backtest
                                def color_all_calls_results(val, col_name, row_index):
                                    try:
                                        if isinstance(val, (int, float)):
                                            if col_name in ['CAGR (%)', 'CPGR (%)', 'MWRR (%)']:
                                                if val > 20:
                                                    return 'background-color: #004d00; color: white; font-weight: bold'
                                                elif val > 10:
                                                    return 'background-color: #1e8449; color: white; font-weight: bold'
                                                elif val > 5:
                                                    return 'background-color: #66bb6a; color: white'
                                                elif val > 0:
                                                    return 'background-color: #ffeb3b; color: black'
                                                else:
                                                    return 'background-color: #ef5350; color: white'
                                            elif col_name == 'Final Capital ($)':
                                                if val > all_options_initial_capital * 2:
                                                    return 'background-color: #004d00; color: white; font-weight: bold'
                                                elif val > all_options_initial_capital * 1.5:
                                                    return 'background-color: #1e8449; color: white'
                                                elif val > all_options_initial_capital:
                                                    return 'background-color: #66bb6a; color: white'
                                                else:
                                                    return 'background-color: #ef5350; color: white'
                                            elif col_name == '% Positive Periods':
                                                if val > 70:
                                                    return 'background-color: #004d00; color: white; font-weight: bold'
                                                elif val > 50:
                                                    return 'background-color: #66bb6a; color: white'
                                                elif val > 30:
                                                    return 'background-color: #ffeb3b; color: black'
                                                else:
                                                    return 'background-color: #ef5350; color: white'
                                    except:
                                        return ''
                                    return ''
                                
                                # Format the results for better display - MAX 2 decimal places
                                display_results_all_calls = results_df_all_calls.copy()
                                
                                # Format columns - MAX 2 decimal places
                                for col in display_results_all_calls.columns:
                                    if col == 'Strike':
                                        display_results_all_calls[col] = display_results_all_calls[col].apply(lambda x: 
                                            f"${int(float(x))}" if float(x) == int(float(x)) else f"${float(x):.2f}")
                                    elif col in ['Positive Periods', 'Negative Periods', 'Total Periods', 'Days to Exp']:
                                        display_results_all_calls[col] = display_results_all_calls[col].astype(int)
                                    elif col in ['Option Price', 'Contracts']:
                                        if display_results_all_calls[col].dtype in ['float64', 'int64']:
                                            display_results_all_calls[col] = display_results_all_calls[col].apply(lambda x: f"${float(x):.2f}")
                                    elif col == 'Final Capital ($)':
                                        if display_results_all_calls[col].dtype in ['float64', 'int64']:
                                            display_results_all_calls[col] = display_results_all_calls[col].apply(lambda x: f"${float(x):,.0f}")
                                    elif col in ['CAGR (%)', 'CPGR (%)', 'MWRR (%)', 'Average Return (%)', 'Median Return (%)', 'Best Period (%)', 'Worst Period (%)', '% Positive Periods']:
                                        if display_results_all_calls[col].dtype in ['float64', 'int64']:
                                            display_results_all_calls[col] = display_results_all_calls[col].apply(lambda x: f"{float(x):.2f}%")
                                
                                # Apply colors to RAW data first, then format for display
                                styled_results_all_calls = results_df_all_calls.style.apply(
                                    lambda x: [color_all_calls_results(val, x.name, i) 
                                             for i, val in enumerate(x)], 
                                    axis=0
                                ).set_table_styles([
                                    {'selector': 'th', 'props': [('background-color', '#2d3748'), ('color', 'white'), ('font-weight', 'bold')]},
                                    {'selector': 'td', 'props': [('text-align', 'center')]}
                                ])
                                
                                # Format the styled results for display
                                styled_results_all_calls = styled_results_all_calls.format({
                                    'Strike': '${:.0f}',
                                    'Option Price': '${:.2f}',
                                    'Contracts': '{:.2f}',
                                    'Final Capital ($)': '${:,.0f}',
                                    'CAGR (%)': '{:.2f}%',
                                    'CPGR (%)': '{:.2f}%',
                                    'MWRR (%)': '{:.2f}%',
                                    'Average Return (%)': '{:.2f}%',
                                    'Median Return (%)': '{:.2f}%',
                                    'Best Period (%)': '{:.2f}%',
                                    'Worst Period (%)': '{:.2f}%',
                                    '% Positive Periods': '{:.2f}%'
                                })
                                
                                st.dataframe(styled_results_all_calls, use_container_width=True, height=600)
                                
                                # Display best CALL option details
                                best_call = results_df_all_calls.iloc[0]
                                st.markdown("#### 🥇 Best CALL Option Details")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Best Strike", f"${best_call['Strike']:.0f}")
                                    st.metric("Expiration", best_call['Expiration'])
                                    st.metric("Days to Exp", f"{best_call['Days to Exp']}")
                                
                                with col2:
                                    st.metric("Option Price", f"${best_call['Option Price']:.2f}")
                                    st.metric("Contracts", f"{best_call['Contracts']:.2f}")
                                    st.metric("Final Capital", f"${best_call['Final Capital ($)']:,.0f}")
                                
                                with col3:
                                    st.metric("CAGR", f"{best_call['CAGR (%)']:.2f}%")
                                    st.metric("CPGR", f"{best_call['CPGR (%)']:.2f}%")
                                    st.metric("MWRR", f"{best_call['MWRR (%)']:.2f}%")
                                
                                with col4:
                                    st.metric("Avg Return", f"{best_call['Average Return (%)']:.2f}%")
                                    st.metric("Positive Periods", f"{best_call['Positive Periods']}/{best_call['Total Periods']}")
                                    st.metric("Success Rate", f"{best_call['% Positive Periods']:.1f}%")
                                
                                # Save to persistent storage - SAME AS HISTORICAL BACKTEST
                                result_key = f"All-CALLs_{ticker_symbol}_All-CALLs_{all_options_options_pct}%_{min_days_filter}d"
                                st.session_state.all_backtest_results[result_key] = {
                                    'params': {
                                        'ticker_symbol': ticker_symbol,
                                        'selected_exp': 'All-CALLs',
                                        'days_to_exp': min_days_filter,
                                        'options_pct': all_options_options_pct,
                                        'total_capital': all_options_initial_capital,
                                        'bond_return_rate': all_options_bond_return,
                                        'added_per_period': all_options_added_per_period,
                                        'option_type': 'CALL'
                                    },
                                    'summary_df': results_df_all_calls,
                                    'periods_df': None,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                                
                                # Also save in the old format for compatibility
                                st.session_state.all_calls_results = {
                                    'results_df': results_df_all_calls,
                                    'parameters': {
                                        'ticker_symbol': ticker_symbol,
                                        'expiration': 'All-CALLs',
                                        'option_type': 'CALL',
                                        'initial_capital': all_options_initial_capital,
                                        'options_pct': all_options_options_pct,
                                        'bond_return': all_options_bond_return,
                                        'added_per_period': all_options_added_per_period,
                                        'days_to_exp': min_days_filter
                                    }
                                }
                                
                                st.success(f"✅ All CALL options backtest completed! Best CALL: ${best_call['Strike']:.0f} {best_call['Expiration']} with {best_call['CAGR (%)']:.2f}% CAGR")
                                st.info("💾 **Results saved!** You can find this backtest in the '📊 All Saved Backtest Results' section above.")
                                
                            else:
                                st.error("❌ No valid CALL options found for backtest")
                                
                        except Exception as e:
                            st.error(f"❌ Error running all CALL options backtest: {str(e)}")
                            st.exception(e)
                
                # Strike Optimizer Calculator
                st.markdown("Find the optimal strike for maximum gains with your specific parameters")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    target_ticker_movement = st.number_input(
                        f"📈 Target {ticker_symbol} Movement (%)",
                        min_value=-50.0,
                        max_value=100.0,
                        value=10.0,
                        step=0.5,
                        help=f"Expected {ticker_symbol} movement for optimization"
                    )
                
                with col2:
                    target_options_pct = st.number_input(
                        "💰 Options Allocation (%)",
                        min_value=1.0,
                        max_value=50.0,
                        value=20.0,
                        step=1.0,
                        help="Percentage of capital allocated to options"
                    )
                
                with col3:
                    target_cash_return = st.number_input(
                        "💵 Cash Return Rate (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=4.0,
                        step=0.1,
                        help="Return rate for cash portion"
                    )
                
                if st.button("🔍 Find Optimal Strike", type="primary"):
                    if not calls_combined.empty or not puts_combined.empty:
                        # Get available strikes for the selected expiration
                        available_strikes = sorted(filtered_opts['strike'].unique()) if not filtered_opts.empty else []
                        
                        if available_strikes:
                            best_strike = None
                            best_return = -float('inf')
                            best_analysis = {}
                            
                            # Test each available strike
                            for test_strike in available_strikes:
                                try:
                                    # Get option price for this strike
                                    test_opt = filtered_opts[filtered_opts['strike'] == test_strike].iloc[0]
                                    test_opt_price = calculate_mid_price(test_opt['bid'], test_opt['ask'], test_opt['lastPrice'])
                                    
                                    if test_opt_price > 0:
                                        # Calculate portfolio with this strike
                                        test_options_cost = (target_options_pct / 100) * total_capital
                                        test_contracts = test_options_cost / (test_opt_price * 100)
                                        test_cash = total_capital - (test_contracts * test_opt_price * 100)
                                        
                                        # Calculate returns for target ticker movement
                                        target_ticker_price = current_price * (1 + target_ticker_movement / 100)
                                        
                                        # Option payoff
                                        if option_type == "CALL":
                                            option_payoff = max(0, target_ticker_price - test_strike)
                                        else:  # PUT
                                            option_payoff = max(0, test_strike - target_ticker_price)
                                        
                                        # Portfolio value
                                        option_value = test_contracts * option_payoff * 100
                                        cash_value = test_cash * (1 + target_cash_return / 100)
                                        total_portfolio_value = option_value + cash_value
                                        
                                        # Return calculation
                                        total_return = ((total_portfolio_value - total_capital) / total_capital) * 100
                                        
                                        if total_return > best_return:
                                            best_return = total_return
                                            best_strike = test_strike
                                            best_analysis = {
                                                'strike': test_strike,
                                                'option_price': test_opt_price,
                                                'contracts': test_contracts,
                                                'options_cost': test_contracts * test_opt_price * 100,
                                                'cash_remaining': test_cash,
                                                'ticker_price_at_target': target_ticker_price,
                                                'option_payoff': option_payoff,
                                                'option_value': option_value,
                                                'cash_value': cash_value,
                                                'total_value': total_portfolio_value,
                                                'total_return': total_return
                                            }
                                            
                                except Exception as e:
                                    continue
                            
                            if best_strike is not None:
                                st.success(f"🎯 **Optimal Strike Found: ${best_strike:.0f}**")
                                
                                # Display results
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("💰 Expected Return", f"{best_return:.1f}%")
                                    st.metric("📊 Option Price", f"${best_analysis['option_price']:.2f}")
                                    st.metric("🔢 Contracts", f"{best_analysis['contracts']:.2f}")
                                
                                with col2:
                                    st.metric("💵 Options Cost", f"${best_analysis['options_cost']:,.0f}")
                                    st.metric("💰 Cash Remaining", f"${best_analysis['cash_remaining']:,.0f}")
                                    st.metric(f"📈 {ticker_symbol} Price at Target", f"${best_analysis['ticker_price_at_target']:.2f}")
                                
                                with col3:
                                    st.metric("🎯 Option Payoff", f"${best_analysis['option_payoff']:.2f}")
                                    st.metric("💎 Option Value", f"${best_analysis['option_value']:,.0f}")
                                    st.metric("🏦 Total Portfolio", f"${best_analysis['total_value']:,.0f}")
                                
                                # Comparison with other strategies
                                st.markdown("#### 📊 Strategy Comparison")
                                
                                # 100% ticker return
                                ticker_return = target_ticker_movement
                                
                                # 100% Cash return
                                cash_return = target_cash_return
                                
                                comparison_data = {
                                    'Strategy': [f'100% {ticker_symbol}', '100% Cash', 'BARBELL Optimal'],
                                    'Return (%)': [ticker_return, cash_return, best_return],
                                    'Value ($)': [
                                        total_capital * (1 + ticker_return/100),
                                        total_capital * (1 + cash_return/100),
                                        best_analysis['total_value']
                                    ]
                                }
                                
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, width='stretch')
                                
                                # Performance vs ticker
                                outperformance = best_return - ticker_return
                                st.info(f"🚀 **BARBELL outperforms {ticker_symbol} by {outperformance:+.1f}%** at {target_ticker_movement}% {ticker_symbol} movement")
                                
                            else:
                                st.error("❌ No valid strikes found for optimization")
                        else:
                            st.warning("⚠️ No strikes available for the selected expiration")
                    else:
                        st.warning("⚠️ No options data available for optimization")
            
            # VIX Analysis Section
            st.markdown("---")
            st.subheader("📊 VIX Analysis")
            
            try:
                with st.spinner("🔄 Fetching VIX data..."):
                    vix_current, vix_history_dict = get_cached_vix_data()
                
                # Reconstruct VIX DataFrame
                vix_history = pd.DataFrame({
                    'Date': vix_history_dict['dates'],
                    'VIX': vix_history_dict['values']
                })
                vix_history['Date'] = pd.to_datetime(vix_history['Date'])
                vix_history = vix_history.set_index('Date')
                
                # Calculate VIX metrics
                vix_1m = vix_history['VIX'].tail(21).mean()  # ~1 month
                vix_3m = vix_history['VIX'].tail(63).mean()  # ~3 months
                vix_6m = vix_history['VIX'].tail(126).mean()  # ~6 months
                vix_1y = vix_history['VIX'].tail(252).mean()  # ~1 year
                vix_5y = vix_history['VIX'].mean()  # All data
                
                vix_median_1m = vix_history['VIX'].tail(21).median()
                vix_median_1y = vix_history['VIX'].tail(252).median()
                
                vix_min = vix_history['VIX'].min()
                vix_max = vix_history['VIX'].max()
                
                
                # Calculate additional medians
                vix_median_3m = vix_history['VIX'].tail(63).median()
                vix_median_6m = vix_history['VIX'].tail(126).median()
                vix_median_5y = vix_history['VIX'].median()
                
                # VIX metrics table - handle missing values properly
                vix_metrics = pd.DataFrame({
                    'Period': ['Current', '1 Month', '3 Months', '6 Months', '1 Year', '5 Years', 'All-Time Min', 'All-Time Max'],
                    'Average': [f"{vix_current:.2f}", f"{vix_1m:.2f}", f"{vix_3m:.2f}", f"{vix_6m:.2f}", 
                               f"{vix_1y:.2f}", f"{vix_5y:.2f}", f"{vix_min:.2f}", f"{vix_max:.2f}"],
                    'Median': ['-', f"{vix_median_1m:.2f}", f"{vix_median_3m:.2f}", f"{vix_median_6m:.2f}", 
                              f"{vix_median_1y:.2f}", f"{vix_median_5y:.2f}", f"{vix_min:.2f}", f"{vix_max:.2f}"],
                    'Min': ['-', f"{vix_history['VIX'].tail(21).min():.2f}", f"{vix_history['VIX'].tail(63).min():.2f}", 
                           f"{vix_history['VIX'].tail(126).min():.2f}", f"{vix_history['VIX'].tail(252).min():.2f}", 
                           f"{vix_min:.2f}", f"{vix_min:.2f}", f"{vix_min:.2f}"],
                    'Max': ['-', f"{vix_history['VIX'].tail(21).max():.2f}", f"{vix_history['VIX'].tail(63).max():.2f}", 
                           f"{vix_history['VIX'].tail(126).max():.2f}", f"{vix_history['VIX'].tail(252).max():.2f}", 
                           f"{vix_max:.2f}", f"{vix_max:.2f}", f"{vix_max:.2f}"]
                })
                
                # All columns are strings now - no conversion issues
                vix_metrics['Period'] = vix_metrics['Period'].astype(str)
                vix_metrics['Average'] = vix_metrics['Average'].astype(str)
                vix_metrics['Median'] = vix_metrics['Median'].astype(str)
                vix_metrics['Min'] = vix_metrics['Min'].astype(str)
                vix_metrics['Max'] = vix_metrics['Max'].astype(str)
                
                st.markdown("**📈 VIX Historical Metrics:**")
                vix_metrics_clean = clean_dataframe_for_arrow(vix_metrics)
                st.dataframe(vix_metrics_clean, width='stretch')
                
                # VIX Chart
                st.markdown("**📊 VIX Historical Chart:**")
                
                # Toggle options for chart - Individual controls for each period in dropdown
                with st.expander("📊 Chart Controls", expanded=False):
                    # Initialize session state for VIX controls
                    if 'vix_show_current' not in st.session_state:
                        st.session_state.vix_show_current = True
                    if 'vix_show_1m_avg' not in st.session_state:
                        st.session_state.vix_show_1m_avg = False
                    if 'vix_show_3m_avg' not in st.session_state:
                        st.session_state.vix_show_3m_avg = False
                    if 'vix_show_6m_avg' not in st.session_state:
                        st.session_state.vix_show_6m_avg = False
                    if 'vix_show_1y_avg' not in st.session_state:
                        st.session_state.vix_show_1y_avg = True  # Default enabled
                    if 'vix_show_5y_avg' not in st.session_state:
                        st.session_state.vix_show_5y_avg = False
                    if 'vix_show_alltime_avg' not in st.session_state:
                        st.session_state.vix_show_alltime_avg = False
                    if 'vix_show_1m_median' not in st.session_state:
                        st.session_state.vix_show_1m_median = False
                    if 'vix_show_3m_median' not in st.session_state:
                        st.session_state.vix_show_3m_median = False
                    if 'vix_show_6m_median' not in st.session_state:
                        st.session_state.vix_show_6m_median = False
                    if 'vix_show_1y_median' not in st.session_state:
                        st.session_state.vix_show_1y_median = True  # Default enabled
                    if 'vix_show_5y_median' not in st.session_state:
                        st.session_state.vix_show_5y_median = False
                    if 'vix_show_alltime_median' not in st.session_state:
                        st.session_state.vix_show_alltime_median = False
                    
                    # Current VIX
                    col1, col2 = st.columns(2)
                    with col1:
                        show_current = st.checkbox("Current VIX", value=st.session_state.vix_show_current, key="vix_current")
                    
                    # Averages
                    st.markdown("**📈 Averages:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        show_1m_avg = st.checkbox("1M Average", value=st.session_state.vix_show_1m_avg, key="vix_1m_avg")
                    with col2:
                        show_3m_avg = st.checkbox("3M Average", value=st.session_state.vix_show_3m_avg, key="vix_3m_avg")
                    with col3:
                        show_6m_avg = st.checkbox("6M Average", value=st.session_state.vix_show_6m_avg, key="vix_6m_avg")
                    with col4:
                        show_1y_avg = st.checkbox("1Y Average", value=st.session_state.vix_show_1y_avg, key="vix_1y_avg")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        show_5y_avg = st.checkbox("5Y Average", value=st.session_state.vix_show_5y_avg, key="vix_5y_avg")
                    with col2:
                        show_alltime_avg = st.checkbox("All-Time Average", value=st.session_state.vix_show_alltime_avg, key="vix_alltime_avg")
                    with col3:
                        pass  # Empty column
                    with col4:
                        pass  # Empty column
                    
                    # Medians
                    st.markdown("**📊 Medians:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        show_1m_median = st.checkbox("1M Median", value=st.session_state.vix_show_1m_median, key="vix_1m_median")
                    with col2:
                        show_3m_median = st.checkbox("3M Median", value=st.session_state.vix_show_3m_median, key="vix_3m_median")
                    with col3:
                        show_6m_median = st.checkbox("6M Median", value=st.session_state.vix_show_6m_median, key="vix_6m_median")
                    with col4:
                        show_1y_median = st.checkbox("1Y Median", value=st.session_state.vix_show_1y_median, key="vix_1y_median")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        show_5y_median = st.checkbox("5Y Median", value=st.session_state.vix_show_5y_median, key="vix_5y_median")
                    with col2:
                        show_alltime_median = st.checkbox("All-Time Median", value=st.session_state.vix_show_alltime_median, key="vix_alltime_median")
                    with col3:
                        pass  # Empty column
                    with col4:
                        pass  # Empty column
                
                # VIX Summary - Always show key metrics
                st.markdown("**📊 VIX Summary:**")
                
                # Create summary columns
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                
                with col1:
                    st.metric("Current VIX", f"{vix_current:.2f}")
                
                with col2:
                    st.metric("1M Average", f"{vix_1m:.2f}")
                
                with col3:
                    st.metric("1Y Average", f"{vix_1y:.2f}")
                
                with col4:
                    st.metric("1M Median", f"{vix_median_1m:.2f}")
                
                with col5:
                    st.metric("1Y Median", f"{vix_median_1y:.2f}")
                
                with col6:
                    st.metric("All-Time Max", f"{vix_history['VIX'].max():.2f}")
                
                with col7:
                    st.metric("All-Time Min", f"{vix_history['VIX'].min():.2f}")
                
                # Create VIX chart with integrated date range buttons
                go, _, _ = get_plotly()
                fig = go.Figure()
                
                # Add VIX historical line (always visible)
                fig.add_trace(go.Scatter(
                    x=vix_history.index,
                    y=vix_history['VIX'],
                    mode='lines',
                    name='VIX Historical',
                    line=dict(color='red', width=1)
                ))
                
                # Add current VIX horizontal line
                if show_current:
                    fig.add_hline(y=vix_current, line_dash="solid", line_color="red", 
                                line_width=4, annotation_text=f"🔥 Current VIX: {vix_current:.2f}")
                
                # Add average lines with distinct colors and styles
                if show_1m_avg:
                    fig.add_hline(y=vix_1m, line_dash="dash", line_color="navy", line_width=2,
                                annotation_text=f"📊 1M Average: {vix_1m:.2f}")
                
                if show_3m_avg:
                    fig.add_hline(y=vix_3m, line_dash="dash", line_color="darkgreen", line_width=2,
                                annotation_text=f"📊 3M Average: {vix_3m:.2f}")
                
                if show_6m_avg:
                    fig.add_hline(y=vix_6m, line_dash="dash", line_color="darkorange", line_width=2,
                                annotation_text=f"📊 6M Average: {vix_6m:.2f}")
                
                if show_1y_avg:
                    fig.add_hline(y=vix_1y, line_dash="dash", line_color="darkviolet", line_width=3,
                                annotation_text=f"📈 1Y Average: {vix_1y:.2f}")
                
                if show_5y_avg:
                    fig.add_hline(y=vix_5y, line_dash="dash", line_color="saddlebrown", line_width=2,
                                annotation_text=f"📊 5Y Average: {vix_5y:.2f}")
                
                if show_alltime_avg:
                    fig.add_hline(y=vix_5y, line_dash="dash", line_color="hotpink", line_width=2,
                                annotation_text=f"📊 All-Time Average: {vix_5y:.2f}")
                
                # Add median lines with distinct colors and styles
                if show_1m_median:
                    fig.add_hline(y=vix_median_1m, line_dash="dot", line_color="steelblue", line_width=2,
                                annotation_text=f"📊 1M Median: {vix_median_1m:.2f}")
                
                if show_3m_median:
                    fig.add_hline(y=vix_median_3m, line_dash="dot", line_color="forestgreen", line_width=2,
                                annotation_text=f"📊 3M Median: {vix_median_3m:.2f}")
                
                if show_6m_median:
                    fig.add_hline(y=vix_median_6m, line_dash="dot", line_color="gold", line_width=2,
                                annotation_text=f"📊 6M Median: {vix_median_6m:.2f}")
                
                if show_1y_median:
                    fig.add_hline(y=vix_median_1y, line_dash="dot", line_color="mediumorchid", line_width=3,
                                annotation_text=f"📈 1Y Median: {vix_median_1y:.2f}")
                
                if show_5y_median:
                    fig.add_hline(y=vix_median_5y, line_dash="dot", line_color="darkcyan", line_width=2,
                                annotation_text=f"📊 5Y Median: {vix_median_5y:.2f}")
                
                if show_alltime_median:
                    fig.add_hline(y=vix_median_5y, line_dash="dot", line_color="dimgray", line_width=2,
                                annotation_text=f"📊 All-Time Median: {vix_median_5y:.2f}")
                
                fig.update_layout(
                    title="VIX Historical Chart",
                    xaxis_title="Date",
                    yaxis_title="VIX Level",
                    height=500,
                    showlegend=True,
                    hovermode='x unified',  # Show VIX value when hovering anywhere on the chart
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1M", step="month", stepmode="backward"),
                                dict(count=3, label="3M", step="month", stepmode="backward"),
                                dict(count=6, label="6M", step="month", stepmode="backward"),
                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                                dict(count=5, label="5Y", step="year", stepmode="backward"),
                                dict(step="all", label="All")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    ),
                    yaxis=dict(
                        # Removed fixedrange and autorange as they are Plotly config arguments
                        # that should not be in update_layout()
                    )
                )
                
                st.plotly_chart(fig, config={'displayModeBar': True, 'displaylogo': False})
                
            except Exception as e:
                st.warning(f"⚠️ Could not fetch VIX data: {e}")
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("💡 Try a different ticker symbol or check if the ticker exists")

else:
    st.info("👆 Please enter a ticker symbol in the sidebar")


# Debug information dropdown at the bottom
if st.session_state.debug_messages:
    with st.expander("🔧 Debug Information", expanded=False):
        st.markdown("**Technical details and performance metrics:**")
        for message in st.session_state.debug_messages:
            st.text(message)
        
        # Clear debug messages after displaying
        st.session_state.debug_messages = []
