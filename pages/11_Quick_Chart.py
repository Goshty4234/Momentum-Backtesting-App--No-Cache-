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

def generate_chart(ticker_input, ma_window, ma_type, calendar_multiplier=1.48):
    """Generate chart with forward-filled calendar days data"""
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
        
        # Create a complete date range including weekends and holidays
        start_date = data.index.min()
        end_date = data.index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Forward fill the data to include all calendar days
        data_full = data.reindex(full_date_range, method='ffill')
        
        # Calculate adjusted window for calendar days
        adjusted_window = int(ma_window * calendar_multiplier)
        
        # Calculate forward-filled MA on all calendar days first
        if ma_type == "SMA":
            data_full[f'MA_{ma_window}_ffill'] = data_full['Close'].rolling(window=adjusted_window).mean()
        else:  # EMA
            data_full[f'MA_{ma_window}_ffill'] = data_full['Close'].ewm(span=adjusted_window).mean()
        
        # Keep only the original trading days for display
        data = data_full.reindex(data.index)
        
        # Calculate regular MA on trading days only
        if ma_type == "SMA":
            data[f'MA_{ma_window}_regular'] = data['Close'].rolling(window=ma_window).mean()
        else:  # EMA
            data[f'MA_{ma_window}_regular'] = data['Close'].ewm(span=ma_window).mean()
        
        # Create chart
        fig = go.Figure()
        
        # Add background zones based on price vs SMA relationship
        # Create zones where price is above/below SMA
        above_sma = data['Close'] > data[f'MA_{ma_window}_regular']
        
        # Find transition points
        transitions = above_sma.diff().fillna(0) != 0
        transition_dates = data.index[transitions]
        
        # Prepare shapes for background zones
        shapes = []
        if len(transition_dates) > 0:
            current_zone = above_sma.iloc[0]
            start_date = data.index[0]
            
            for i, transition_date in enumerate(transition_dates):
                # Add zone from start to transition
                if current_zone:
                    # Bullish zone (green background)
                    shapes.append(dict(
                        type="rect",
                        x0=start_date, x1=transition_date,
                        y0=0, y1=1,
                        yref="paper",
                        fillcolor="rgba(0, 255, 0, 0.15)",
                        layer="below",
                        line=dict(width=0)
                    ))
                else:
                    # Bearish zone (red background)
                    shapes.append(dict(
                        type="rect",
                        x0=start_date, x1=transition_date,
                        y0=0, y1=1,
                        yref="paper",
                        fillcolor="rgba(255, 0, 0, 0.15)",
                        layer="below",
                        line=dict(width=0)
                    ))
                
                start_date = transition_date
                current_zone = not current_zone
            
            # Add final zone
            if current_zone:
                shapes.append(dict(
                    type="rect",
                    x0=start_date, x1=data.index[-1],
                    y0=0, y1=1,
                    yref="paper",
                    fillcolor="rgba(0, 255, 0, 0.15)",
                    layer="below",
                    line=dict(width=0)
                ))
            else:
                shapes.append(dict(
                    type="rect",
                    x0=start_date, x1=data.index[-1],
                    y0=0, y1=1,
                    yref="paper",
                    fillcolor="rgba(255, 0, 0, 0.15)",
                    layer="below",
                    line=dict(width=0)
                ))
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name=f'{final_ticker} Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add regular moving average
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[f'MA_{ma_window}_regular'],
            mode='lines',
            name=f'{ma_type}({ma_window}) - Trading Days',
            line=dict(color='red', width=2)
        ))
        
        # Add forward-filled moving average
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[f'MA_{ma_window}_ffill'],
            mode='lines',
            name=f'{ma_type}({ma_window}) - Calendar Days (FFill)',
            line=dict(color='green', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{final_ticker} Price vs {ma_type}({ma_window}) - Trading vs Calendar Days",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            height=600,
            showlegend=True,
            template="plotly_white",
            shapes=shapes
        )
        
        # Customize hover template to show full date and prevent text truncation
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>" +
                         "Date: %{x|%Y-%m-%d}<br>" +
                         "Price: $%{y:.2f}<br>" +
                         "<extra></extra>"
        )
        
        # Calculate metrics for both MAs
        current_price = data['Close'].iloc[-1]
        current_ma_regular = data[f'MA_{ma_window}_regular'].iloc[-1]
        current_ma_ffill = data[f'MA_{ma_window}_ffill'].iloc[-1]
        current_date = data.index[-1].strftime('%Y-%m-%d')
        
        # Calculate MA crossings for both MAs
        crossings_regular = calculate_ma_crossings(data, f'MA_{ma_window}_regular')
        # For calendar days, calculate crossings on the full forward-filled data
        crossings_ffill = calculate_ma_crossings(data_full, f'MA_{ma_window}_ffill')
        
        chart_data = {
            'fig': fig,
            'final_ticker': final_ticker,
            'data': data,
            'current_price': current_price,
            'current_ma_regular': current_ma_regular,
            'current_ma_ffill': current_ma_ffill,
            'current_date': current_date,
            'ma_window': ma_window,
            'ma_type': ma_type,
            'crossings_regular': crossings_regular,
            'crossings_ffill': crossings_ffill
        }
        
        return chart_data, None
        
    except Exception as e:
        return None, str(e)

st.set_page_config(
    page_title="Quick Chart Calendar", 
    page_icon="📅", 
    layout="wide"
)

st.title("📅 Quick Chart - Calendar Days (Forward Fill)")

# Input section
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

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

with col4:
    calendar_multiplier = st.number_input(
        "Calendar Multiplier",
        min_value=1.0,
        max_value=3.0,
        value=1.48,
        step=0.01,
        help="Multiplier to convert trading days to calendar days (e.g., 1.48 means 148 calendar days ≈ 100 trading days)"
    )

# Initialize session state for chart persistence
if 'quick_chart_calendar_data' not in st.session_state:
    st.session_state.quick_chart_calendar_data = None

# Process ticker input
if ticker_input:
    final_ticker = ticker_input  # Already processed by callback

    # Generate or update chart
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("📊 Generate Calendar Chart", type="primary"):
            # Check if we're using cache
            cache_key = f"processed_data_{final_ticker}_calendar"
            using_cache = cache_key in st.session_state
            
            if using_cache:
                with st.spinner(f"Using cached calendar data for {final_ticker}..."):
                    chart_data, error = generate_chart(ticker_input, ma_window, ma_type, calendar_multiplier)
                    if chart_data:
                        st.session_state.quick_chart_calendar_data = chart_data
                        st.success(f"⚡ Used cache - {len(chart_data['data'])} days (2h cache)")
                    else:
                        st.error(f"❌ Error: {error}")
            else:
                with st.spinner(f"Downloading fresh calendar data for {final_ticker}..."):
                    chart_data, error = generate_chart(ticker_input, ma_window, ma_type, calendar_multiplier)
                    if chart_data:
                        st.session_state.quick_chart_calendar_data = chart_data
                        st.success(f"✅ Downloaded fresh - {len(chart_data['data'])} days (2h cache)")
                    else:
                        st.error(f"❌ Error: {error}")
    
    # Display chart if available
    if st.session_state.quick_chart_calendar_data:
        chart_data = st.session_state.quick_chart_calendar_data
        
        # Show current values for both MAs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${chart_data['current_price']:.2f}")
        with col2:
            st.metric(f"{chart_data['ma_type']}({chart_data['ma_window']}) - Trading", f"${chart_data['current_ma_regular']:.2f}")
        with col3:
            st.metric(f"{chart_data['ma_type']}({chart_data['ma_window']}) - Calendar", f"${chart_data['current_ma_ffill']:.2f}")
        with col4:
            diff_pct = ((chart_data['current_ma_ffill']/chart_data['current_ma_regular']-1)*100)
            st.metric("Difference", f"{diff_pct:.2f}%", delta=f"{diff_pct:.2f}%")
        
        # Display chart
        st.plotly_chart(chart_data['fig'], use_container_width=True)
        
        # Show data summary
        with st.expander("📊 Data Summary"):
            data = chart_data['data']
            st.write(f"**Period:** {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            st.write(f"**Total Days:** {len(data)}")
            st.write(f"**Current Price:** ${chart_data['current_price']:.2f}")
            st.write(f"**Trading Days MA:** ${chart_data['current_ma_regular']:.2f}")
            st.write(f"**Calendar Days MA:** ${chart_data['current_ma_ffill']:.2f}")
            st.write(f"**Difference:** {((chart_data['current_ma_ffill']/chart_data['current_ma_regular']-1)*100):.2f}%")
            st.write(f"**Calendar Multiplier:** {calendar_multiplier}")
            st.write(f"**Adjusted Window:** {int(ma_window * calendar_multiplier)} calendar days")
            
            # Show last few rows
            st.write("**Last 5 days:**")
            display_data = data[['Close', f'MA_{chart_data["ma_window"]}_regular', f'MA_{chart_data["ma_window"]}_ffill']].tail()
            st.dataframe(display_data)
        
        # Show MA crossings table for both MAs - Full width layout
        if 'crossings_regular' in chart_data and chart_data['crossings_regular']:
            with st.expander("🔄 Trading Days MA Crossings", expanded=True):
                st.write(f"**Total Crossings:** {len(chart_data['crossings_regular'])}")
                
                # Create crossings DataFrame
                crossings_df = pd.DataFrame(chart_data['crossings_regular'])
                crossings_df['price'] = crossings_df['price'].apply(lambda x: f"{x:.2f}")
                crossings_df['ma_value'] = crossings_df['ma_value'].apply(lambda x: f"{x:.2f}")
                crossings_df['duration_days'] = crossings_df['duration_days'].astype(int)
                
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
                    above_crossings = len([c for c in chart_data['crossings_regular'] if c['direction'] == 'Above MA'])
                    st.metric("Above MA", above_crossings)
                with col2:
                    below_crossings = len([c for c in chart_data['crossings_regular'] if c['direction'] == 'Below MA'])
                    st.metric("Below MA", below_crossings)
                with col3:
                    avg_duration = np.mean([c['duration_days'] for c in chart_data['crossings_regular']])
                    st.metric("Avg Duration", f"{avg_duration:.1f}")
        else:
            st.info("No Trading Days MA crossings detected")
        
        if 'crossings_ffill' in chart_data and chart_data['crossings_ffill']:
            with st.expander("🔄 Calendar Days MA Crossings", expanded=True):
                st.write(f"**Total Crossings:** {len(chart_data['crossings_ffill'])}")
                st.info("ℹ️ Note: Duration shows calendar days. Price crossings only occur on trading days, so durations may appear similar to Trading Days MA.")
                
                # Create crossings DataFrame
                crossings_df = pd.DataFrame(chart_data['crossings_ffill'])
                crossings_df['price'] = crossings_df['price'].apply(lambda x: f"{x:.2f}")
                crossings_df['ma_value'] = crossings_df['ma_value'].apply(lambda x: f"{x:.2f}")
                crossings_df['duration_days'] = crossings_df['duration_days'].astype(int)
                
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
                    above_crossings = len([c for c in chart_data['crossings_ffill'] if c['direction'] == 'Above MA'])
                    st.metric("Above MA", above_crossings)
                with col2:
                    below_crossings = len([c for c in chart_data['crossings_ffill'] if c['direction'] == 'Below MA'])
                    st.metric("Below MA", below_crossings)
                with col3:
                    avg_duration = np.mean([c['duration_days'] for c in chart_data['crossings_ffill']])
                    st.metric("Avg Duration", f"{avg_duration:.1f}")
        else:
            st.info("No Calendar Days MA crossings detected")

else:
    st.info("👆 Enter a ticker symbol to get started")

# SMA Analyzer section (Advanced - at the end)
if st.session_state.quick_chart_calendar_data:
    st.markdown("---")
    
    # Trading Days Analyzer
    st.subheader("📊 SMA Analyzer - Trading Days")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_period_trading = st.number_input("Min Period", min_value=5, max_value=500, value=10, key="sma_min_trading")
    with col2:
        max_period_trading = st.number_input("Max Period", min_value=10, max_value=1000, value=200, key="sma_max_trading")
    with col3:
        bandwidth_trading = st.number_input("Bandwidth %", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="sma_bandwidth_trading")
    with col4:
        confirmation_trading = st.number_input("Confirmation Days", min_value=0, max_value=10, value=0, key="sma_confirmation_trading")
    
    if st.button("Analyze Trading Days SMA", key="analyze_trading"):
        with st.spinner("Analyzing Trading Days SMA..."):
            chart_data = st.session_state.quick_chart_calendar_data
            data = chart_data['data']
            
            try:
                results, error = analyze_sma_periods(data, int(min_period_trading), int(max_period_trading), float(bandwidth_trading), int(confirmation_trading))
                if results:
                    st.session_state.sma_analysis_trading = results
                    st.success("✅ Trading Days analysis complete!")
                else:
                    st.error(f"Error: {error}")
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")
    
    # Display Trading Days results
    if st.session_state.get('sma_analysis_trading'):
        results = st.session_state.sma_analysis_trading
        
        if 'all_results' in results and results['all_results']:
            # Create DataFrame
            results_df = pd.DataFrame(results['all_results'])
            results_df['period'] = results_df['period'].astype(int)
            results_df['cagr'] = results_df['cagr'].apply(lambda x: f"{x:.2f}")
            results_df['max_drawdown'] = results_df['max_drawdown'].apply(lambda x: f"{x:.2f}")
            results_df['sharpe'] = results_df['sharpe'].apply(lambda x: f"{x:.2f}")
            
            # Highlight best periods
            def highlight_best(row):
                styles = [''] * len(row)
                if row['period'] == results['best_cagr']['period']:
                    styles[1] = 'background-color: #2d5016; color: #ffffff'  # CAGR column (green)
                if row['period'] == results['best_drawdown']['period']:
                    styles[2] = 'background-color: #8b0000; color: #ffffff'  # Max Drawdown column (red)
                return styles
            
            styled_df = results_df.style.apply(highlight_best, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Show best periods
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best CAGR Period", f"{results['best_cagr']['period']}", f"{results['best_cagr']['cagr']:.2f}%")
            with col2:
                st.metric("Best Drawdown Period", f"{results['best_drawdown']['period']}", f"{results['best_drawdown']['max_drawdown']:.2f}%")
        else:
            st.error("No results available for Trading Days Analysis")
    
    st.markdown("---")
    
    # Calendar Days Analyzer
    st.subheader("📅 SMA Analyzer - Calendar Days")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        min_period_calendar = st.number_input("Min Period", min_value=5, max_value=500, value=10, key="sma_min_calendar")
    with col2:
        max_period_calendar = st.number_input("Max Period", min_value=10, max_value=1000, value=200, key="sma_max_calendar")
    with col3:
        bandwidth_calendar = st.number_input("Bandwidth %", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="sma_bandwidth_calendar")
    with col4:
        confirmation_calendar = st.number_input("Confirmation Days", min_value=0, max_value=10, value=0, key="sma_confirmation_calendar")
    with col5:
        calendar_multiplier = st.number_input("Calendar Multiplier", min_value=1.0, max_value=3.0, value=1.48, step=0.01, key="sma_multiplier")
    
    if st.button("Analyze Calendar Days SMA", key="analyze_calendar"):
        with st.spinner("Analyzing Calendar Days SMA..."):
            chart_data = st.session_state.quick_chart_calendar_data
            data = chart_data['data']
            
            try:
                # Forward-fill data for calendar days analysis
                start_date = data.index.min()
                end_date = data.index.max()
                full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                data_full = data.reindex(full_date_range, method='ffill')
                
                # Analyze with calendar multiplier
                results = []
                for period in range(int(min_period_calendar), int(max_period_calendar) + 1):
                    try:
                        adjusted_window = int(period * calendar_multiplier)
                        data_full[f'MA_{period}'] = data_full['Close'].rolling(window=adjusted_window).mean()
                        analysis_data = data_full.reindex(data.index)
                        
                        # Calculate strategy returns
                        analysis_data['Position'] = 0
                        analysis_data['Returns'] = analysis_data['Close'].pct_change()
                        
                        # Apply bandwidth filter if specified
                        if bandwidth_calendar > 0:
                            bandwidth = analysis_data['Close'] * bandwidth_calendar / 100
                            above_band = analysis_data['Close'] > (analysis_data[f'MA_{period}'] + bandwidth)
                            below_band = analysis_data['Close'] < (analysis_data[f'MA_{period}'] - bandwidth)
                        else:
                            above_band = analysis_data['Close'] > analysis_data[f'MA_{period}']
                            below_band = analysis_data['Close'] < analysis_data[f'MA_{period}']
                        
                        # Apply confirmation days filter if specified
                        if confirmation_calendar > 0:
                            above_confirmed = above_band.rolling(window=confirmation_calendar).sum() >= confirmation_calendar
                            below_confirmed = below_band.rolling(window=confirmation_calendar).sum() >= confirmation_calendar
                        else:
                            above_confirmed = above_band
                            below_confirmed = below_band
                        
                        # Set positions
                        analysis_data.loc[above_confirmed, 'Position'] = 1
                        analysis_data.loc[below_confirmed, 'Position'] = 0
                        
                        # Calculate strategy returns
                        analysis_data['Strategy_Returns'] = analysis_data['Position'].shift(1) * analysis_data['Returns']
                        
                        # Calculate metrics
                        total_return = (1 + analysis_data['Strategy_Returns']).prod() - 1
                        years = len(analysis_data) / 252
                        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
                        
                        # Calculate drawdown
                        cumulative = (1 + analysis_data['Strategy_Returns']).cumprod()
                        rolling_max = cumulative.expanding().max()
                        drawdown = (cumulative - rolling_max) / rolling_max
                        max_drawdown = drawdown.min()
                        
                        # Calculate Sharpe ratio
                        strategy_returns = analysis_data['Strategy_Returns'].dropna()
                        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
                            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                        else:
                            sharpe_ratio = 0
                        
                        results.append({
                            'period': period,
                            'cagr': cagr * 100,
                            'max_drawdown': max_drawdown * 100,
                            'sharpe_ratio': sharpe_ratio
                        })
                        
                    except Exception as e:
                        continue
                
                if results:
                    best_cagr = max(results, key=lambda x: x['cagr'])
                    best_drawdown = max(results, key=lambda x: x['max_drawdown'])
                    
                    analysis_results = {
                        'results': results,
                        'best_cagr_period': best_cagr['period'],
                        'best_cagr': best_cagr['cagr'],
                        'best_drawdown_period': best_drawdown['period'],
                        'best_drawdown': best_drawdown['max_drawdown']
                    }
                    
                    st.session_state.sma_analysis_calendar = analysis_results
                    st.success("✅ Calendar Days analysis complete!")
                else:
                    st.error("No results generated")
                    
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")
    
    # Display Calendar Days results
    if st.session_state.get('sma_analysis_calendar'):
        results = st.session_state.sma_analysis_calendar
        
        if 'results' in results and results['results']:
            # Create DataFrame
            results_df = pd.DataFrame(results['results'])
            results_df['period'] = results_df['period'].astype(int)
            results_df['cagr'] = results_df['cagr'].apply(lambda x: f"{x:.2f}")
            results_df['max_drawdown'] = results_df['max_drawdown'].apply(lambda x: f"{x:.2f}")
            results_df['sharpe_ratio'] = results_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
            
            # Highlight best periods
            def highlight_best(row):
                styles = [''] * len(row)
                if row['period'] == results['best_cagr_period']:
                    styles[1] = 'background-color: #2d5016; color: #ffffff'  # CAGR column (green)
                if row['period'] == results['best_drawdown_period']:
                    styles[2] = 'background-color: #8b0000; color: #ffffff'  # Max Drawdown column (red)
                return styles
            
            styled_df = results_df.style.apply(highlight_best, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Show best periods
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best CAGR Period", f"{results['best_cagr_period']}", f"{results['best_cagr']:.2f}%")
            with col2:
                st.metric("Best Drawdown Period", f"{results['best_drawdown_period']}", f"{results['best_drawdown']:.2f}%")
        else:
            st.error("No results available for Calendar Days Analysis")

# SMA Analyzer section (Advanced - at the end)

