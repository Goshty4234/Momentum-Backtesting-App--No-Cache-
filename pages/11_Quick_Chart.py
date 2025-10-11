import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_ticker_data(ticker_symbol):
    """Download ticker data with caching - 2 hour cache for multiple MA testing"""
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period="max")
    return data

@st.cache_data(ttl=7200)  # Cache for 2 hours  
def get_processed_ticker_data(final_ticker):
    """Get processed ticker data with leverage/expense applied - 2 hour cache"""
    # Extract base ticker for yfinance (remove parameters)
    base_ticker = final_ticker.split('?')[0]
    
    # Download data with caching
    data = get_ticker_data(base_ticker)
    
    if data.empty:
        return None
    
    # Apply leverage and expense ratio if present
    if '?L=' in final_ticker and '?E=' in final_ticker:
        # Extract leverage and expense
        parts = final_ticker.split('?')
        leverage = 1
        expense_ratio = 0
        
        for part in parts[1:]:
            if part.startswith('L='):
                leverage = float(part.split('=')[1])
            elif part.startswith('E='):
                expense_ratio = float(part.split('=')[1])
        
        # Apply leverage to returns
        data['Returns'] = data['Close'].pct_change()
        data['Leveraged_Returns'] = data['Returns'] * leverage
        
        # Apply expense ratio (daily)
        daily_expense = expense_ratio / 252 / 100
        data['Leveraged_Returns'] = data['Leveraged_Returns'] - daily_expense
        
        # Recalculate close prices
        data['Close'] = (1 + data['Leveraged_Returns']).cumprod() * data['Close'].iloc[0]
        data['Open'] = data['Close'].shift(1)
        data['High'] = data['Close']
        data['Low'] = data['Close']
    
    return data

def update_ticker_input():
    """Callback function when ticker input changes - auto-capitalize and resolve aliases"""
    if 'ticker_input_key' in st.session_state:
        new_value = st.session_state.ticker_input_key.strip()
        
        # Apply transformations like in other pages
        # Convert commas to dots for decimal separators
        new_value = new_value.replace(",", ".")
        
        # Convert to uppercase
        new_value = new_value.upper()
        
        # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
        if new_value == 'BRK.B':
            new_value = 'BRK-B'
        elif new_value == 'BRK.A':
            new_value = 'BRK-A'
        
        # Alias resolution (same as in generate_chart)
        aliases = {
            'SPX': '^GSPC',
            'SPXTR': '^SP500TR',
            'SP500': '^GSPC',
            'SP500TR': '^SP500TR',
            'SPYTR': '^SP500TR',
            'NASDAQ': '^IXIC',
            'NDX': '^NDX',
            'TQQQND': '^NDX?L=3?E=0.95',
            'GOLDX': 'GOLD_COMPLETE',
            'TLTTR': 'TLT_COMPLETE',
            'GOLDSIM': 'GOLDSIM_COMPLETE',
            'SPYSIM': 'SPYSIM_COMPLETE',
        }
        
        resolved_value = aliases.get(new_value, new_value)
        
        # Update session state with resolved value for display
        st.session_state.ticker_input_key = resolved_value

def generate_chart(ticker_input, ma_window, ma_type):
    """Generate chart and save to session state - optimized for multiple MA testing"""
    try:
        # Process ticker input
        processed_ticker = ticker_input.strip().replace(",", ".").upper()
        
        # Berkshire conversion
        if processed_ticker == 'BRK.B':
            processed_ticker = 'BRK-B'
        elif processed_ticker == 'BRK.A':
            processed_ticker = 'BRK-A'
        
        # Alias resolution (simplified version)
        aliases = {
            'SPX': '^GSPC',
            'SPXTR': '^SP500TR',
            'SP500': '^GSPC',
            'SP500TR': '^SP500TR',
            'SPYTR': '^SP500TR',
            'NASDAQ': '^IXIC',
            'NDX': '^NDX',
            'TQQQND': '^NDX?L=3?E=0.95',
            'GOLDX': 'GOLD_COMPLETE',
            'TLTTR': 'TLT_COMPLETE',
            'GOLDSIM': 'GOLDSIM_COMPLETE',
            'SPYSIM': 'SPYSIM_COMPLETE',
        }
        
        final_ticker = aliases.get(processed_ticker, processed_ticker)
        
        # Get processed data with 2-hour cache
        data = get_processed_ticker_data(final_ticker)
        
        if data is None or data.empty:
            return None, f"No data found for {final_ticker}"
        
        # Calculate Moving Average (this is fast, no caching needed)
        if ma_type == "SMA":
            data[f'MA_{ma_window}'] = data['Close'].rolling(window=ma_window).mean()
        else:  # EMA
            data[f'MA_{ma_window}'] = data['Close'].ewm(span=ma_window).mean()
        
        # Create chart
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name=f'{final_ticker} Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[f'MA_{ma_window}'],
            mode='lines',
            name=f'{ma_type}({ma_window})',
            line=dict(color='red', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{final_ticker} Price vs {ma_type}({ma_window})",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            height=600,
            showlegend=True,
            template="plotly_white"
        )
        
        # Calculate metrics
        current_price = data['Close'].iloc[-1]
        current_ma = data[f'MA_{ma_window}'].iloc[-1]
        current_date = data.index[-1].strftime('%Y-%m-%d')
        
        chart_data = {
            'fig': fig,
            'final_ticker': final_ticker,
            'data': data,
            'current_price': current_price,
            'current_ma': current_ma,
            'current_date': current_date,
            'ma_window': ma_window,
            'ma_type': ma_type
        }
        
        return chart_data, None
        
    except Exception as e:
        return None, str(e)

st.set_page_config(page_title="Quick Chart", layout="wide")

st.title("📈 Quick Chart - Ticker & Moving Average")

# Input section
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    ticker_input = st.text_input(
        "Ticker Symbol", 
        key="ticker_input_key",
        value="SPY",
        help="Enter ticker symbol (e.g., SPY, QQQ, TQQQND, GOLDX, etc.)",
        placeholder="SPY",
        on_change=update_ticker_input
    )

with col2:
    ma_window = st.number_input(
        "MA Window", 
        min_value=5, 
        max_value=500, 
        value=200,
        help="Moving Average period"
    )

with col3:
    ma_type = st.selectbox(
        "MA Type",
        options=["SMA", "EMA"],
        help="Simple Moving Average or Exponential Moving Average"
    )

# Initialize session state for chart persistence
if 'quick_chart_data' not in st.session_state:
    st.session_state.quick_chart_data = None

# Process ticker input
if ticker_input:
    final_ticker = ticker_input  # Already processed by callback

    # Generate or update chart
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("📊 Generate Chart", type="primary"):
            # Check if we're using cache
            cache_key = f"processed_data_{final_ticker}"
            using_cache = cache_key in st.session_state
            
            if using_cache:
                with st.spinner(f"Using cached data for {final_ticker}..."):
                    chart_data, error = generate_chart(ticker_input, ma_window, ma_type)
                    if chart_data:
                        st.session_state.quick_chart_data = chart_data
                        st.success(f"⚡ Used cache - {len(chart_data['data'])} days (2h cache)")
                    else:
                        st.error(f"❌ Error: {error}")
            else:
                with st.spinner(f"Downloading fresh data for {final_ticker}..."):
                    chart_data, error = generate_chart(ticker_input, ma_window, ma_type)
                    if chart_data:
                        st.session_state.quick_chart_data = chart_data
                        st.success(f"✅ Downloaded fresh - {len(chart_data['data'])} days (2h cache)")
                    else:
                        st.error(f"❌ Error: {error}")
    
    # Display chart if available
    if st.session_state.quick_chart_data:
        chart_data = st.session_state.quick_chart_data
        
        # Show current values
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${chart_data['current_price']:.2f}")
        with col2:
            st.metric(f"{chart_data['ma_type']}({chart_data['ma_window']})", f"${chart_data['current_ma']:.2f}")
        with col3:
            if chart_data['current_price'] > chart_data['current_ma']:
                st.metric("Status", "📈 Above MA", delta=f"{((chart_data['current_price']/chart_data['current_ma']-1)*100):.2f}%")
            else:
                st.metric("Status", "📉 Below MA", delta=f"{((chart_data['current_price']/chart_data['current_ma']-1)*100):.2f}%")
        
        # Display chart
        st.plotly_chart(chart_data['fig'], use_container_width=True)
        
        # Show data summary
        with st.expander("📊 Data Summary"):
            data = chart_data['data']
            st.write(f"**Period:** {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            st.write(f"**Total Days:** {len(data)}")
            st.write(f"**Current Price:** ${chart_data['current_price']:.2f}")
            st.write(f"**Current {chart_data['ma_type']}({chart_data['ma_window']}):** ${chart_data['current_ma']:.2f}")
            st.write(f"**Price vs MA:** {((chart_data['current_price']/chart_data['current_ma']-1)*100):.2f}%")
            
            # Show last few rows
            st.write("**Last 5 days:**")
            display_data = data[['Close', f'MA_{chart_data["ma_window"]}']].tail()
            st.dataframe(display_data)

else:
    st.info("👆 Enter a ticker symbol to get started")

# Footer
st.markdown("---")
st.markdown("💡 **Tips:**")
st.markdown("- Use aliases like `SPYTR`, `TQQQND`, `GOLDX` for special tickers")
st.markdown("- Leveraged tickers (like `TQQQND`) include leverage and expense ratio")
st.markdown("- Berkshire tickers: use `BRK.A` or `BRK.B` (automatically converted)")
st.markdown("- **Cache**: Data cached for 2 hours - test multiple MA without re-downloading!")
st.markdown("- **Session**: Chart persists when switching pages")
