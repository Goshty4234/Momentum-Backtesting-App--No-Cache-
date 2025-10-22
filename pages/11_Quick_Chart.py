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

def calculate_ma_crossings(data, ma_column):
    """Calculate MA crossings with dates, direction, and duration"""
    crossings = []
    
    # Create a boolean series for price above MA
    above_ma = data['Close'] > data[ma_column]
    
    # Find crossing points (where the boolean series changes)
    crossing_points = above_ma.diff().fillna(0) != 0
    
    if not crossing_points.any():
        return crossings
    
    # Get crossing dates and directions
    crossing_dates = data.index[crossing_points]
    crossing_directions = above_ma[crossing_points]
    
    # Calculate duration for each crossing
    for i, (date, direction) in enumerate(zip(crossing_dates, crossing_directions)):
        # Find the next crossing or end of data
        if i < len(crossing_dates) - 1:
            next_date = crossing_dates[i + 1]
            duration = (next_date - date).days
        else:
            # Last crossing - duration to current date
            duration = (data.index[-1] - date).days
        
        crossings.append({
            'date': date.strftime('%Y-%m-%d'),
            'direction': 'Above MA' if direction else 'Below MA',
            'duration_days': duration,
            'price': data.loc[date, 'Close'],
            'ma_value': data.loc[date, ma_column]
        })
    
    return crossings

def calculate_cagr(returns):
    """Calculate Compound Annual Growth Rate"""
    if len(returns) == 0:
        return 0
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252  # Assuming 252 trading days per year
    if years == 0:
        return 0
    return (1 + total_return) ** (1 / years) - 1

def calculate_drawdown(returns):
    """Calculate maximum drawdown"""
    if len(returns) == 0:
        return 0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def analyze_sma_periods(data, min_period=10, max_period=200, 
                       bandwidth_pct=0, confirmation_days=0):
    """Analyze all SMA periods and find best CAGR and min drawdown periods"""
    results = []
    
    for period in range(min_period, max_period + 1):
        try:
            # Calculate SMA
            data_copy = data.copy()
            data_copy[f'MA_{period}'] = data_copy['Close'].rolling(window=period).mean()
            
            # Remove NaN values
            data_clean = data_copy.dropna()
            if len(data_clean) < 252:  # Need at least 1 year of data
                continue
            
            # Apply bandwidth filter if specified
            if bandwidth_pct > 0:
                bandwidth = data_clean[f'MA_{period}'] * (bandwidth_pct / 100)
                above_band = data_clean['Close'] > (data_clean[f'MA_{period}'] + bandwidth)
                below_band = data_clean['Close'] < (data_clean[f'MA_{period}'] - bandwidth)
            else:
                above_band = data_clean['Close'] > data_clean[f'MA_{period}']
                below_band = data_clean['Close'] < data_clean[f'MA_{period}']
            
            # Apply confirmation days filter if specified
            if confirmation_days > 0:
                # Require consecutive days above/below MA
                above_confirmed = above_band.rolling(window=confirmation_days).sum() == confirmation_days
                below_confirmed = below_band.rolling(window=confirmation_days).sum() == confirmation_days
                
                # Generate signals with confirmation
                signals = np.where(above_confirmed, 1, np.where(below_confirmed, 0, np.nan))
            else:
                # Simple signals
                signals = np.where(above_band, 1, 0)
            
            # Forward fill signals to maintain position until next signal
            signals = pd.Series(signals, index=data_clean.index).fillna(method='ffill')
            
            # Calculate returns based on signals
            returns = data_clean['Close'].pct_change()
            strategy_returns = signals.shift(1) * returns  # Shift to avoid look-ahead bias
            
            # Remove NaN from strategy returns
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) == 0:
                continue
                
            # Calculate metrics
            cagr = calculate_cagr(strategy_returns)
            max_dd = calculate_drawdown(strategy_returns)
            
            results.append({
                'period': period,
                'cagr': cagr * 100,  # Convert to percentage
                'max_drawdown': max_dd * 100,  # Convert to percentage
                'total_return': (1 + strategy_returns).prod() - 1,
                'sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0,
                'win_rate': (strategy_returns > 0).mean() * 100
            })
            
        except Exception as e:
            continue
    
    if not results:
        return None, "No valid results found"
    
    # Find best CAGR period
    best_cagr = max(results, key=lambda x: x['cagr'])
    
    # Find min drawdown period (least negative drawdown)
    best_drawdown = max(results, key=lambda x: x['max_drawdown'])  # max because drawdowns are negative
    
    return {
        'all_results': results,
        'best_cagr': best_cagr,
        'best_drawdown': best_drawdown
    }, None

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
        
        # Calculate MA crossings
        crossings = calculate_ma_crossings(data, f'MA_{ma_window}')
        
        chart_data = {
            'fig': fig,
            'final_ticker': final_ticker,
            'data': data,
            'current_price': current_price,
            'current_ma': current_ma,
            'current_date': current_date,
            'ma_window': ma_window,
            'ma_type': ma_type,
            'crossings': crossings
        }
        
        return chart_data, None
        
    except Exception as e:
        return None, str(e)

st.set_page_config(
    page_title="Quick Chart", 
    page_icon="📈", 
    layout="wide"
)

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
        
        # Show MA crossings table
        if 'crossings' in chart_data and chart_data['crossings']:
            with st.expander("🔄 MA Crossings Analysis", expanded=True):
                st.write(f"**Total Crossings:** {len(chart_data['crossings'])}")
                
                # Create crossings DataFrame
                crossings_df = pd.DataFrame(chart_data['crossings'])
                crossings_df['price'] = crossings_df['price'].round(2)
                crossings_df['ma_value'] = crossings_df['ma_value'].round(2)
                
                # Add color coding for direction
                def color_direction(val):
                    if val == 'Above MA':
                        return 'background-color: #2d5016; color: #ffffff'
                    else:
                        return 'background-color: #8b0000; color: #ffffff'
                
                styled_df = crossings_df.style.applymap(color_direction, subset=['direction'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    above_crossings = len([c for c in chart_data['crossings'] if c['direction'] == 'Above MA'])
                    st.metric("Above MA Crossings", above_crossings)
                with col2:
                    below_crossings = len([c for c in chart_data['crossings'] if c['direction'] == 'Below MA'])
                    st.metric("Below MA Crossings", below_crossings)
                with col3:
                    avg_duration = np.mean([c['duration_days'] for c in chart_data['crossings']])
                    st.metric("Avg Duration (days)", f"{avg_duration:.1f}")
                
                # Show longest periods
                st.write("**Longest Periods:**")
                sorted_crossings = sorted(chart_data['crossings'], key=lambda x: x['duration_days'], reverse=True)
                for i, crossing in enumerate(sorted_crossings[:5]):
                    st.write(f"{i+1}. {crossing['direction']} for {crossing['duration_days']:.0f} days (from {crossing['date']})")
        elif 'crossings' in chart_data:
            st.info("No MA crossings detected in the data period")
        else:
            st.info("Generate a chart first to see MA crossings analysis")

else:
    st.info("👆 Enter a ticker symbol to get started")

# SMA Analyzer section (Advanced - at the end)
st.markdown("---")
st.subheader("🎯 SMA Analyzer (Advanced)")

# Parameters in two rows
col1, col2, col3, col4 = st.columns(4)

with col1:
    min_period = st.number_input(
        "Min Period",
        min_value=5,
        max_value=100,
        value=10,
        help="Minimum SMA period to test"
    )

with col2:
    max_period = st.number_input(
        "Max Period", 
        min_value=50,
        max_value=500,
        value=200,
        help="Maximum SMA period to test"
    )

with col3:
    bandwidth_pct = st.number_input(
        "Bandwidth %",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Price must cross MA by this % to trigger signal (anti-whiplash)"
    )

with col4:
    confirmation_days = st.number_input(
        "Confirmation Days",
        min_value=0,
        max_value=10,
        value=0,
        help="Require N consecutive days above/below MA to confirm signal"
    )

# Analysis button
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 Analyze SMA Periods", type="secondary"):
        if ticker_input:
            # Get data for analysis
            final_ticker = ticker_input
            data = get_processed_ticker_data(final_ticker)
            
            if data is not None and not data.empty:
                with st.spinner(f"Analyzing SMA periods for {final_ticker}..."):
                    analysis_result, error = analyze_sma_periods(
                        data, 
                        min_period=min_period, 
                        max_period=max_period, 
                        bandwidth_pct=bandwidth_pct,
                        confirmation_days=confirmation_days
                    )
                    
                    if analysis_result:
                        st.session_state.sma_analysis = analysis_result
                        st.success(f"✅ Analysis complete!")
                    else:
                        st.error(f"❌ Error: {error}")
            else:
                st.error("❌ No data available for analysis")
        else:
            st.warning("⚠️ Enter a ticker symbol first")

with col2:
    if st.button("🗑️ Clear Results", type="secondary"):
        if 'sma_analysis' in st.session_state:
            del st.session_state.sma_analysis
            st.rerun()

with col3:
    st.write("")  # Empty space for alignment

with col4:
    st.write("")  # Empty space for alignment

# Display SMA Analysis Results
if 'sma_analysis' in st.session_state and st.session_state.sma_analysis:
    analysis = st.session_state.sma_analysis
    
    st.markdown("---")
    st.subheader("📊 SMA Analysis Results")
    
    # Show best results for both metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Best CAGR Period", f"{analysis['best_cagr']['period']}")
    with col2:
        st.metric("Best CAGR", f"{analysis['best_cagr']['cagr']:.2f}%")
    with col3:
        st.metric("Min Drawdown Period", f"{analysis['best_drawdown']['period']}")
    with col4:
        st.metric("Min Drawdown", f"{analysis['best_drawdown']['max_drawdown']:.2f}%")
    with col5:
        st.metric("Win Rate (Best CAGR)", f"{analysis['best_cagr']['win_rate']:.1f}%")
    
    # Show results table
    results_df = pd.DataFrame(analysis['all_results'])
    
    # Format all numeric columns to show exactly 2 decimal places
    numeric_columns = ['cagr', 'max_drawdown', 'total_return', 'sharpe', 'win_rate']
    for col in numeric_columns:
        if col in results_df.columns:
            results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)
    
    # Sort by CAGR descending
    results_df = results_df.sort_values('cagr', ascending=False)
    
    # Highlight the best results
    def highlight_best(row):
        if row['period'] == analysis['best_cagr']['period']:
            return ['background-color: #2d5016; color: white'] * len(row)  # Green for best CAGR
        elif row['period'] == analysis['best_drawdown']['period']:
            return ['background-color: #8b0000; color: white'] * len(row)  # Red for min drawdown
        return [''] * len(row)
    
    styled_df = results_df.style.apply(highlight_best, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # Show summary
    st.write("**🎯 Summary:**")
    st.write(f"• **Best CAGR**: Period {analysis['best_cagr']['period']} with {analysis['best_cagr']['cagr']:.2f}% CAGR")
    st.write(f"• **Min Drawdown**: Period {analysis['best_drawdown']['period']} with {analysis['best_drawdown']['max_drawdown']:.2f}% drawdown")
