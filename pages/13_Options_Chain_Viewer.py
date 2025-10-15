import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import concurrent.futures
import time
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Options Chain Viewer",
    page_icon="📊",
    layout="wide"
)

# Initialize debug messages list
if 'debug_messages' not in st.session_state:
    st.session_state.debug_messages = []

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
    """Calculate MID price from bid/ask, fallback to last price"""
    if pd.notna(bid) and pd.notna(ask) and bid is not None and ask is not None:
        return (bid + ask) / 2
    elif pd.notna(last_price) and last_price is not None:
        return last_price
    else:
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

# Initialize session state for ticker persistence - NO DEFAULT SPY
if 'persistent_ticker' not in st.session_state:
    st.session_state.persistent_ticker = ""

# Ticker input with persistence
ticker_input = st.sidebar.text_input(
    "📈 Enter Ticker Symbol:",
    value=st.session_state.persistent_ticker,
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
    ticker = yf.Ticker(ticker_symbol)
    try:
        # Single API call to get both price and options data
        current_price = float(ticker.history(period="1d")['Close'].iloc[-1])
        expirations = list(ticker.options)  # This is cached by yfinance internally
        fetch_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return current_price, expirations, fetch_timestamp
    except Exception as e:
        raise Exception(f"Error fetching {ticker_symbol}: {e}")

@st.cache_data(persist="disk")  # Disk persistence (TTL ignored when persist is set)
def get_cached_all_options_batch(ticker_symbol, expirations_list):
    """Cache ALL option chains in ONE API call - OPTIMIZED like page 1"""
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
            
        # Ensure current_price is a safe float for calculations
        current_price = safe_float_price(current_price)
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
                available_sections = ["📅 BY EXPIRATION", "📋 COMPLETE LIST", "📈 CALLS ONLY", "📉 PUTS ONLY", "📊 OPTION EVOLUTION"]
            elif show_calls:
                available_sections = ["📅 BY EXPIRATION", "📋 COMPLETE LIST", "📈 CALLS ONLY", "📊 OPTION EVOLUTION"]
            elif show_puts:
                available_sections = ["📅 BY EXPIRATION", "📋 COMPLETE LIST", "📉 PUTS ONLY", "📊 OPTION EVOLUTION"]
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
                    
                    # Expiration selector with days
                    selected_exp_full = st.selectbox(
                        "📅 Select Expiration Date:",
                        expiration_options,
                        index=0,
                        help="Choose which expiration date to display in the options chain format"
                    )
                    
                    # Extract just the date part
                    selected_exp = selected_exp_full.split(' (')[0] if selected_exp_full else all_expirations[0]
                    
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
                            st.dataframe(styled_df, width='stretch', height=800)
                            
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
                    st.dataframe(calls_display_clean, width='stretch', height=300)
                
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
                    st.dataframe(puts_display_clean, width='stretch', height=300)
            
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
                        
                        # Select columns to display
                        display_cols = ['expiration', 'daysToExp', 'strike', 'lastPrice', 'bid', 'ask', 
                                       'volume', 'openInterest', 'impliedVolatility']
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
                        st.dataframe(calls_display_clean, width='stretch', height=600)
                        
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
                        
                        # Select columns to display
                        display_cols = ['expiration', 'daysToExp', 'strike', 'lastPrice', 'bid', 'ask', 
                                       'volume', 'openInterest', 'impliedVolatility']
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
                        st.dataframe(puts_display_clean, width='stretch', height=600)
                        
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
                if show_calls and show_puts:
                    st.subheader("📊 Option Evolution Over Time")
                    st.markdown("**Track the same strike across different expirations**")
                    
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
                                # Manual input - highest priority
                                manual_strike = st.text_input(
                                    "🔍 Enter Strike Price:",
                                    value=st.session_state.persistent_strike,
                                    help="Type a strike price manually (e.g., 440.0 or 440,5 - comma will be converted to dot)",
                                    key="manual_strike_input"
                                )
                            
                            with col2:
                                # Dropdown selector
                                closest_strike_index = min(range(len(all_strikes)), 
                                                         key=lambda i: abs(all_strikes[i] - current_price))
                                
                                dropdown_strike = st.selectbox(
                                    "📋 Or Select from List:",
                                    all_strikes,
                                    index=closest_strike_index,
                                    help="Choose from dropdown list"
                                )
                            
                            # Determine final strike selection
                            selected_strike = None
                            
                            # Priority 1: Manual input if provided and valid
                            if manual_strike and manual_strike.strip():
                                try:
                                    # Convert comma to dot for decimal separator
                                    converted_strike = manual_strike.replace(',', '.')
                                    selected_strike = float(converted_strike)
                                    
                                    # Show conversion if different
                                    if converted_strike != manual_strike:
                                        st.info(f"🔄 Converted {manual_strike} to {converted_strike}")
                                    
                                    # Check if strike exists in available options
                                    if selected_strike not in all_strikes:
                                        closest_available = min(all_strikes, key=lambda x: abs(x - selected_strike))
                                        st.warning(f"Strike ${selected_strike:.1f} not found in available options. Using closest available strike: ${closest_available:.1f}")
                                        selected_strike = closest_available
                                    
                                    # Update session state with the final strike
                                    st.session_state.persistent_strike = str(selected_strike)
                                    st.session_state.manual_strike_entered = True  # Mark that user entered manually
                                    
                                except ValueError:
                                    st.error("Please enter a valid number for strike price.")
                                    selected_strike = dropdown_strike
                            else:
                                # Priority 2: Use dropdown selection
                                selected_strike = dropdown_strike
                                # Update session state with dropdown selection
                                st.session_state.persistent_strike = str(selected_strike)
                                st.session_state.manual_strike_entered = False  # Mark that user used dropdown
                        else:
                            selected_strike = None
                        
                        if selected_strike:
                            st.markdown(f"**Tracking Strike: ${selected_strike:.2f}**")
                            
                            # Filter data for selected strike
                            calls_strike = calls_combined[calls_combined['strike'] == selected_strike] if not calls_combined.empty else pd.DataFrame()
                            puts_strike = puts_combined[puts_combined['strike'] == selected_strike] if not puts_combined.empty else pd.DataFrame()
                            
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
                                        'Strike': f"{selected_strike:.0f}" if selected_strike == int(selected_strike) else f"{selected_strike:.2f}"
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
                                            call_display_price = f"${call_mid:.2f}"
                                        elif call_last is not None:
                                            call_mid = call_last
                                            call_display_price = f"${call_last:.2f} (Last)"
                                        else:
                                            call_mid = None
                                            call_display_price = '-'
                                        
                                        # Calculate Call price as % of current ticker price
                                        call_price_pct = None
                                        if call_mid is not None and call_mid != 0:
                                            safe_price = safe_float_price(current_price)
                                            call_price_pct = (call_mid / safe_price) * 100
                                        
                                        row.update({
                                            'Call Price': call_display_price,
                                            'Call Price %': f"{call_price_pct:.2f}%" if call_price_pct is not None else '-',
                                            'Call MID': f"{call_mid:.2f}" if call_mid is not None else '-',
                                            'Call Last': f"{call_last:.2f}" if call_last is not None else '-',
                                            'Call Bid': f"{call_bid:.2f}" if call_bid is not None else '-',
                                            'Call Ask': f"{call_ask:.2f}" if call_ask is not None else '-',
                                            'Call Volume': f"{call_row['volume']:.0f}" if pd.notna(call_row['volume']) else '0',
                                            'Call IV': f"{(call_row['impliedVolatility'] * 100):.1f}%" if pd.notna(call_row['impliedVolatility']) else '-'
                                        })
                                    else:
                                        row.update({
                                            'Call Price': '-', 'Call Price %': '-', 'Call MID': '-', 'Call Last': '-', 'Call Bid': '-', 'Call Ask': '-',
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
                                            put_display_price = f"${put_mid:.2f}"
                                        elif put_last is not None:
                                            put_mid = put_last
                                            put_display_price = f"${put_last:.2f} (Last)"
                                        else:
                                            put_mid = None
                                            put_display_price = '-'
                                        
                                        # Calculate Put price as % of current ticker price
                                        put_price_pct = None
                                        if put_mid is not None and put_mid != 0:
                                            safe_price = safe_float_price(current_price)
                                            put_price_pct = (put_mid / safe_price) * 100
                                        
                                        row.update({
                                            'Put Price': put_display_price,
                                            'Put Price %': f"{put_price_pct:.2f}%" if put_price_pct is not None else '-',
                                            'Put MID': f"{put_mid:.2f}" if put_mid is not None else '-',
                                            'Put Last': f"{put_last:.2f}" if put_last is not None else '-',
                                            'Put Bid': f"{put_bid:.2f}" if put_bid is not None else '-',
                                            'Put Ask': f"{put_ask:.2f}" if put_ask is not None else '-',
                                            'Put Volume': f"{put_row['volume']:.0f}" if pd.notna(put_row['volume']) else '0',
                                            'Put IV': f"{(put_row['impliedVolatility'] * 100):.1f}%" if pd.notna(put_row['impliedVolatility']) else '-'
                                        })
                                    else:
                                        row.update({
                                            'Put Price': '-', 'Put Price %': '-', 'Put MID': '-', 'Put Last': '-', 'Put Bid': '-', 'Put Ask': '-',
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
                                st.dataframe(styled_evolution_df, width='stretch', height=400)
                                
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
                                    # Use MID price for calls if available, otherwise last price
                                    call_price = None
                                    if row['Call MID'] != '-' and pd.notna(row['Call MID']):
                                        call_price = float(row['Call MID'])
                                    elif row['Call Last'] != '-' and pd.notna(row['Call Last']):
                                        call_price = float(row['Call Last'])
                                    
                                    if call_price is not None:
                                        safe_price = safe_float_price(current_price)
                                        call_intrinsic = max(0, safe_price - selected_strike)
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
                                    # Use MID price for puts if available, otherwise last price
                                    put_price = None
                                    if row['Put MID'] != '-' and pd.notna(row['Put MID']):
                                        put_price = float(row['Put MID'])
                                    elif row['Put Last'] != '-' and pd.notna(row['Put Last']):
                                        put_price = float(row['Put Last'])
                                    
                                    if put_price is not None:
                                        safe_price = safe_float_price(current_price)
                                        put_intrinsic = max(0, selected_strike - safe_price)
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
                                        title=f"Option Price Evolution - Strike ${selected_strike:.2f} | {ticker_symbol}: ${current_price:.2f} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                                        xaxis_title="Days to Expiration (Left: Far Future → Right: Near Expiration)",
                                        yaxis_title="Option Price ($)",
                                        height=500,
                                        showlegend=True,
                                        hovermode='x unified',  # Show both curves on hover
                                        xaxis=dict(
                                            autorange='reversed'  # Reverse x-axis: left=far future, right=near expiration
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, config={'displayModeBar': True, 'displaylogo': False})
                                else:
                                    st.warning("No price data available for chart")
                            else:
                                st.warning(f"No options found for strike ${selected_strike:.2f}")
                    else:
                        st.warning("No options data available")
            
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
                        fixedrange=False,  # Allow Y-axis zoom
                        autorange=True
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
