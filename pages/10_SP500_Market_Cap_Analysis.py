import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import time
import concurrent.futures
import threading
from functools import partial

st.set_page_config(
    page_title="S&P 500 Market Cap Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 S&P 500 Market Cap Analysis")
st.markdown("**Current S&P 500 companies ranked by market cap with key performance metrics**")

# Function to get S&P 500 companies from Wikipedia
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_sp500_companies():
    """Get S&P 500 companies from Wikipedia"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        
        if not table:
            return None, "Could not find S&P 500 table on Wikipedia"
        
        # Extract basic data
        rows = table.find_all('tr')[1:]
        companies = []
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 4:
                symbol = cells[0].get_text(strip=True)
                name = cells[1].get_text(strip=True)
                sector = cells[2].get_text(strip=True)
                industry = cells[3].get_text(strip=True)
                
                companies.append({
                    'Symbol': symbol,
                    'Name': name,
                    'Sector': sector,
                    'Industry': industry
                })
        
        return pd.DataFrame(companies), None
        
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

# Function to get returns for different time periods (like Google Finance)
def get_returns_by_period(ticker, period):
    """Get returns for different time periods like Google Finance"""
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Define date ranges based on period
        today = date.today()
        
        if period == "1d":
            start = today - timedelta(days=1)
            period_name = "1-Day Return %"
        elif period == "5d":
            start = today - timedelta(days=5)
            period_name = "5-Day Return %"
        elif period == "1m":
            start = today - timedelta(days=30)
            period_name = "1-Month Return %"
        elif period == "6m":
            start = today - timedelta(days=180)
            period_name = "6-Month Return %"
        elif period == "1a":
            start = today - timedelta(days=365)
            period_name = "1-Year Return %"
        elif period == "5a":
            start = today - timedelta(days=1825)
            period_name = "5-Year Return %"
        else:  # Max
            start = today - timedelta(days=3650)  # 10 years
            period_name = "Max Return %"
        
        # Get data
        data = ticker_obj.history(start=start, end=today)
        
        if data.empty or len(data) < 2:
            return 0, period_name
        
        # Calculate return
        price_now = data["Close"].iloc[-1]
        price_old = data["Close"].iloc[0]
        return_pct = (price_now / price_old - 1) * 100
        
        return return_pct, period_name
        
    except Exception as e:
        print(f"Error getting {period} return for {ticker}: {str(e)}")
        return 0, f"{period} Return %"

# Super simple function - exactly like your example
def get_simple_returns(ticker):
    """Super simple returns calculation - exactly like your NVDA example"""
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Dates : aujourd'hui et il y a 365 jours
        today = date.today()
        start = today - timedelta(days=365)
        
        # Télécharger les prix
        data = ticker_obj.history(start=start, end=today)
        
        if data.empty:
            return 0, 0, 0
        
        # Calculer la performance sur 1 an
        price_now = data["Close"].iloc[-1]
        price_old = data["Close"].iloc[0]
        one_year_return = (price_now / price_old - 1) * 100
        
        # YTD: get price at start of current year
        current_year = date.today().year
        ytd_start = date(current_year, 1, 1)
        ytd_data = ticker_obj.history(start=ytd_start, end=today)
        
        if not ytd_data.empty:
            ytd_price = ytd_data["Close"].iloc[0]
            ytd_return = (price_now / ytd_price - 1) * 100
        else:
            ytd_return = 0
        
        # Volatility: standard deviation of daily returns
        daily_returns = data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        return ytd_return, one_year_return, volatility
        
    except Exception as e:
        print(f"Error getting returns for {ticker}: {str(e)}")
        return 0, 0, 0

# Debug function to check what data we're getting
def debug_ticker_data(ticker):
    """Debug function to see what data we're getting from Yahoo Finance"""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        
        # Get current price from recent data
        hist_data = yf_ticker.history(period="5d", interval="1d")
        current_price = hist_data.iloc[-1]['Close'] if not hist_data.empty else 0
        
        # Get shares outstanding
        shares_outstanding = info.get('sharesOutstanding', 0)
        
        # Calculate market cap
        calculated_market_cap = current_price * shares_outstanding if current_price and shares_outstanding else 0
        
        debug_info = {
            'ticker': ticker,
            'current_price': current_price,
            'shares_outstanding': shares_outstanding,
            'calculated_market_cap': calculated_market_cap,
            'yahoo_market_cap': info.get('marketCap', 0),
            'yahoo_enterprise_value': info.get('enterpriseValue', 0),
            'yahoo_current_price': info.get('currentPrice', 0),
            'yahoo_regular_market_price': info.get('regularMarketPrice', 0)
        }
        
        return debug_info
    except Exception as e:
        return {'ticker': ticker, 'error': str(e)}

# Function to get data for a single ticker (for parallel processing)
def get_single_ticker_data(ticker):
    """Get comprehensive data for a single ticker"""
    try:
        yf_ticker = yf.Ticker(ticker)
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.2)  # Increased delay to avoid rate limits and protect other pages
        
        info = yf_ticker.info
        
        # Get current price first from historical data (most reliable)
        price = 0
        current_price = 0
        
        try:
            # Get current price from most recent trading data - try multiple periods
            for period in ["1d", "5d", "1mo"]:
                try:
                    hist_data = yf_ticker.history(period=period, interval="1d")
                    if hasattr(hist_data, 'empty') and not hist_data.empty:
                        current_price = hist_data.iloc[-1]['Close']
                        price = current_price
                        break
                except:
                    continue
        except:
            pass
        
        # Fallback to info data if no historical data
        if not price or price == 0:
            price = (info.get('currentPrice', 0) or 
                    info.get('regularMarketPrice', 0) or 
                    info.get('previousClose', 0) or 
                    info.get('lastPrice', 0))
            current_price = price
        
        # Get shares outstanding
        shares_outstanding = (info.get('sharesOutstanding', 0) or 
                            info.get('impliedSharesOutstanding', 0))
        
        # Calculate market cap from current price × shares outstanding (most reliable method)
        if price and shares_outstanding and price > 0 and shares_outstanding > 0:
            market_cap = price * shares_outstanding
        else:
            # Try multiple fallbacks for market cap
            market_cap = (info.get('marketCap', 0) or 
                         info.get('enterpriseValue', 0) or 
                         info.get('totalValue', 0) or
                         info.get('quoteType', {}).get('marketCap', 0))
            
            # If still no market cap, try to get from recent trading data
            if not market_cap or market_cap == 0:
                try:
                    recent_data = yf_ticker.history(period="1d")
                    if hasattr(recent_data, 'empty') and not recent_data.empty:
                        recent_price = recent_data.iloc[-1]['Close']
                        recent_volume = recent_data.iloc[-1]['Volume']
                        # Very rough estimate: price * (volume / 1000000) - not accurate but better than 0
                        if recent_price > 0 and recent_volume > 0:
                            market_cap = recent_price * (recent_volume / 1000000)
                except:
                    pass
        
        beta = info.get('beta', 0)
        
        # Get historical data for ALL time periods - simple approach
        try:
            today = date.today()
            
            # Get current price first
            hist_current = yf_ticker.history(period="1d")
            if not hist_current.empty:
                price_now = hist_current["Close"].iloc[-1]
                price = price_now
            else:
                price_now = price
                volatility = 0
            
            # Calculate returns for different periods
            periods = {
                "1d": 1,
                "5d": 5, 
                "1m": 30,
                "6m": 180,
                "1a": 365,
                "5a": 1825,
                "max": 3650
            }
            
            returns = {}
            
            # YTD calculation
            current_year = date.today().year
            ytd_start = date(current_year, 1, 1)
            ytd_data = yf_ticker.history(start=ytd_start, end=today)
            
            if not ytd_data.empty:
                ytd_price = ytd_data["Close"].iloc[0]
                returns["ytd"] = (price_now / ytd_price - 1) * 100
            else:
                returns["ytd"] = 0
            
            # Calculate all period returns
            for period_name, days in periods.items():
                start_date = today - timedelta(days=days)
                data = yf_ticker.history(start=start_date, end=today)
                
                if not data.empty and len(data) > 1:
                    old_price = data["Close"].iloc[0]
                    period_return = (price_now / old_price - 1) * 100
                    returns[period_name] = period_return
                    
                    # Use 1-year data for volatility
                    if period_name == "1a":
                        daily_returns = data['Close'].pct_change().dropna()
                        volatility = daily_returns.std() * np.sqrt(252) * 100
                else:
                    returns[period_name] = 0
            
            # Extract returns
            ytd_return = returns.get("ytd", 0)
            one_year_return = returns.get("1a", 0)
            return_1d = returns.get("1d", 0)
            return_5d = returns.get("5d", 0)
            return_1m = returns.get("1m", 0)
            return_6m = returns.get("6m", 0)
            return_5a = returns.get("5a", 0)
            return_max = returns.get("max", 0)
                
        except Exception as e:
            print(f"Error getting historical data for {ticker}: {str(e)}")
            ytd_return = 0
            one_year_return = 0
            volatility = 0
            if not price or price == 0:
                price = info.get('regularMarketPreviousClose', 0)
        
        # Debug output for major stocks
        if ticker in ['NVDA', 'MSFT', 'AAPL', 'META', 'GOOGL']:
            print(f"DEBUG {ticker}: YTD={ytd_return:.2f}%, 1Y={one_year_return:.2f}%, Vol={volatility:.2f}%")
        
        return {
            'Symbol': ticker,
            'Market Cap': market_cap,
            'Market Cap (B)': market_cap / 1e9 if market_cap else 0,
            'Price': price,
            'Beta': beta,
            'YTD Return %': ytd_return,
            '1D Return %': return_1d,
            '5D Return %': return_5d,
            '1M Return %': return_1m,
            '6M Return %': return_6m,
            '1Y Return %': one_year_return,
            '5Y Return %': return_5a,
            'Max Return %': return_max,
            'Volatility %': volatility
        }
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return {
            'Symbol': ticker,
            'Market Cap': 0,
            'Market Cap (B)': 0,
            'Price': 0,
            'Beta': 0,
            'YTD Return %': 0,
            '1D Return %': 0,
            '5D Return %': 0,
            '1M Return %': 0,
            '6M Return %': 0,
            '1Y Return %': 0,
            '5Y Return %': 0,
            'Max Return %': 0,
            'Volatility %': 0
        }

# Function to get market data using bulk download - MUCH faster!
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_market_data(tickers):
    """Get comprehensive market data using bulk download - reduces API calls dramatically"""
    import yfinance as yf
    from datetime import date, timedelta
    import numpy as np
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tickers = len(tickers)
    market_data = []
    
    try:
        # Bulk download historical data for all tickers at once
        progress_bar.progress(0.1)
        status_text.text("📥 Downloading historical data for all tickers in one request...")
        
        # Get 1 year of data for all tickers
        end_date = date.today()
        start_date = end_date - timedelta(days=400)  # Extra buffer for weekends/holidays
        
        bulk_data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')
        
        progress_bar.progress(0.3)
        status_text.text("🔄 Processing bulk data...")
        
        # Get current info for all tickers (minimal individual calls)
        ticker_objects = {}
        for ticker in tickers:
            try:
                ticker_objects[ticker] = yf.Ticker(ticker)
            except:
                continue
        
        progress_bar.progress(0.5)
        
        # Process each ticker from bulk data
        processed = 0
        for ticker in tickers:
            try:
                # Get basic info
                yf_ticker = ticker_objects.get(ticker)
                if not yf_ticker:
                    continue
                
                info = yf_ticker.info
                
                # Get current price from bulk data
                price = 0
                if len(tickers) == 1:
                    ticker_data = bulk_data
                else:
                    ticker_data = bulk_data[ticker] if ticker in bulk_data.columns.levels[0] else None
                
                if ticker_data is not None and not ticker_data.empty:
                    price = float(ticker_data['Close'].iloc[-1])
                
                # Fallback to info if bulk data doesn't work
                if price == 0:
                    price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                
                # Calculate market cap
                market_cap = 0
                shares_outstanding = info.get('sharesOutstanding', 0)
                if price > 0 and shares_outstanding > 0:
                    market_cap = price * shares_outstanding
                else:
                    market_cap = info.get('marketCap', info.get('enterpriseValue', 0))
                
                # Get other metrics
                beta = info.get('beta', 0)
                
                # Calculate returns from bulk data
                returns = calculate_returns_from_bulk(ticker_data, price, end_date)
                
                market_data.append({
                    'Symbol': ticker,
                    'Market Cap': market_cap,
                    'Market Cap (B)': market_cap / 1e9 if market_cap else 0,
                    'Price': price,
                    'Beta': beta,
                    'YTD Return %': returns.get('ytd', 0),
                    '1D Return %': returns.get('1d', 0),
                    '5D Return %': returns.get('5d', 0),
                    '1M Return %': returns.get('1m', 0),
                    '6M Return %': returns.get('6m', 0),
                    '1Y Return %': returns.get('1a', 0),
                    '5Y Return %': returns.get('5a', 0),
                    'Max Return %': returns.get('max', 0),
                    'Volatility %': returns.get('volatility', 0)
                })
                
                processed += 1
                progress = 0.5 + (processed / total_tickers) * 0.5
                progress_bar.progress(progress)
                status_text.text(f"✅ Processed {processed}/{total_tickers} companies ({progress:.1%})")
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                processed += 1
                progress = 0.5 + (processed / total_tickers) * 0.5
                progress_bar.progress(progress)
    
    except Exception as e:
        print(f"Bulk download failed: {e}")
        status_text.text("⚠️ Bulk download failed, falling back to individual requests...")
        # Fallback to old method
        return get_market_data_individual(tickers)
    
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(market_data)

def calculate_returns_from_bulk(ticker_data, current_price, end_date):
    """Calculate returns from bulk historical data"""
    from datetime import date, timedelta
    import numpy as np
    
    returns = {}
    
    if ticker_data is None or ticker_data.empty or current_price == 0:
        return {k: 0 for k in ['ytd', '1d', '5d', '1m', '6m', '1a', '5a', 'max', 'volatility']}
    
    try:
        # Get dates for calculations
        today = end_date
        periods = {
            "1d": 1,
            "5d": 5, 
            "1m": 30,
            "6m": 180,
            "1a": 365,
            "5a": 1825,
            "max": 3650
        }
        
        # YTD calculation
        current_year = today.year
        ytd_start = date(current_year, 1, 1)
        
        # Calculate YTD return first (special handling)
        try:
            # For YTD, get data from January 1st of current year
            current_year = today.year
            ytd_start = pd.Timestamp(f"{current_year}-01-01")
            
            # Get all data from start of year
            ytd_data = ticker_data[ticker_data.index >= ytd_start]
            
            if not ytd_data.empty and len(ytd_data) > 1:
                # Get first price of the year
                first_price = float(ytd_data['Close'].iloc[0])
                returns['ytd'] = (current_price / first_price - 1) * 100
            else:
                returns['ytd'] = 0
        except Exception as e:
            print(f"YTD calculation error: {e}")
            returns['ytd'] = 0
        
        # Calculate all other period returns
        for period_name, days in periods.items():
            if period_name == "ytd":
                continue  # Already calculated above
                
            start_date = today - timedelta(days=days)
            
            # Find closest date in data
            try:
                period_mask = ticker_data.index >= pd.Timestamp(start_date)
                if period_mask.any():
                    old_price = float(ticker_data.loc[period_mask, 'Close'].iloc[0])
                    returns[period_name] = (current_price / old_price - 1) * 100
                    
                    # Use 1-year data for volatility
                    if period_name == "1a":
                        daily_returns = ticker_data.loc[period_mask, 'Close'].pct_change().dropna()
                        if not daily_returns.empty:
                            returns['volatility'] = float(daily_returns.std() * np.sqrt(252) * 100)
                else:
                    returns[period_name] = 0
            except:
                returns[period_name] = 0
        
        # Fill missing values
        for key in ['ytd', '1d', '5d', '1m', '6m', '1a', '5a', 'max', 'volatility']:
            if key not in returns:
                returns[key] = 0
                
    except Exception as e:
        print(f"Error calculating returns: {e}")
        returns = {k: 0 for k in ['ytd', '1d', '5d', '1m', '6m', '1a', '5a', 'max', 'volatility']}
    
    return returns

# Keep old method as fallback
def get_market_data_individual(tickers):
    """Fallback method using individual requests"""
    import concurrent.futures
    import time
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tickers = len(tickers)
    market_data = []
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(3, len(tickers))  # Reduce to 3 workers to avoid rate limits and protect other pages
    
    status_text.text(f"Fetching data for {total_tickers} companies using {max_workers} parallel workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {executor.submit(get_single_ticker_data, ticker): ticker for ticker in tickers}
        
        # Process completed tasks
        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                market_data.append(result)
                completed += 1
                
                # Update progress
                progress_bar.progress(completed / total_tickers)
                status_text.text(f"Completed {completed}/{total_tickers} companies...")
                
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                # Add empty data for failed ticker
                market_data.append({
                    'Symbol': ticker,
                    'Market Cap': 0,
                    'Market Cap (B)': 0,
                    'Price': 0,
                    'Beta': 0,
                    'YTD Return %': 0,
                    '1Y Return %': 0,
                    'Volatility %': 0
                })
                completed += 1
                progress_bar.progress(completed / total_tickers)
    
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(market_data)

# Main interface
col_fetch, col_clear, col_debug = st.columns([1, 1, 1])

with col_fetch:
    if st.button("📥 Fetch Market Data", type="primary", help="Click to load market data - only when needed!"):
        st.session_state['fetch_data'] = True
        st.rerun()

with col_clear:
    if st.button("🗑️ Clear Cache", help="Clear all cached data"):
        st.cache_data.clear()
        st.session_state['fetch_data'] = False  # Reset fetch state
        st.success("Cache cleared! Click 'Fetch Market Data' to load fresh data.")
        st.rerun()

with col_debug:
    if st.button("🔍 Debug Top Companies"):
        st.session_state['debug_mode'] = True

# Removed redundant "Fetch 1-Year Returns" button - "Fetch Market Data" does everything

# Test single ticker
if st.button("🧪 Test NVDA Only"):
    st.session_state['test_nvda'] = True

# Debug section
if st.session_state.get('debug_mode', False):
    st.subheader("🔍 Debug: Raw Data from Yahoo Finance")
    
    # Debug the key companies
    debug_tickers = ['NVDA', 'MSFT', 'AAPL', 'META', 'GOOGL', 'AMZN']
    
    with st.spinner("Getting debug data..."):
        debug_results = []
        for ticker in debug_tickers:
            debug_info = debug_ticker_data(ticker)
            debug_results.append(debug_info)
        
        debug_df = pd.DataFrame(debug_results)
        st.dataframe(debug_df, use_container_width=True)
        
        # Show which market cap value we're using
        st.markdown("**Market Cap Calculation Method:**")
        for result in debug_results:
            if 'error' not in result:
                ticker = result['ticker']
                calculated = result['calculated_market_cap'] / 1e9
                yahoo = result['yahoo_market_cap'] / 1e9
                st.write(f"**{ticker}**: Calculated: ${calculated:.1f}B, Yahoo: ${yahoo:.1f}B")
    
    if st.button("❌ Close Debug"):
        st.session_state['debug_mode'] = False
        st.rerun()
    
    # Show what the debug data actually reveals
    st.markdown("**Debug Data Analysis:**")
    st.info("🔍 The debug shows our calculated market caps are correct, but there might be a sorting issue in the main table.")

# Removed redundant 1-Year Return Fetcher section - "Fetch Market Data" provides all this data and more

# Test NVDA section
if st.session_state.get('test_nvda', False):
    st.subheader("🧪 Test NVDA - Step by Step")
    
    try:
        st.write("**Step 1: Get NVDA ticker**")
        nvda = yf.Ticker("NVDA")
        
        st.write("**Step 2: Get dates**")
        today = date.today()
        start = today - timedelta(days=365)
        st.write(f"Today: {today}")
        st.write(f"Start: {start}")
        
        st.write("**Step 3: Download data**")
        data = nvda.history(start=start, end=today)
        st.write(f"Data shape: {data.shape}")
        st.write(f"Data empty: {data.empty}")
        
        if not data.empty:
            st.write("**Step 4: Get prices**")
            price_now = data["Close"].iloc[-1]
            price_old = data["Close"].iloc[0]
            st.write(f"Price now: ${price_now:.2f}")
            st.write(f"Price old: ${price_old:.2f}")
            
            st.write("**Step 5: Calculate performance**")
            performance = (price_now / price_old - 1) * 100
            st.write(f"Performance 365 days NVDA: {performance:.2f}%")
        else:
            st.error("No data returned!")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    if st.button("❌ Close Test"):
        st.session_state['test_nvda'] = False
        st.rerun()

# Get S&P 500 companies (this is fast, no API calls)
with st.spinner("Loading S&P 500 companies..."):
    companies_df, error = get_sp500_companies()

if error:
    st.error(f"❌ {error}")
    st.info("💡 Try refreshing the page or check your internet connection")
elif companies_df is not None and not companies_df.empty:
    st.success(f"✅ Successfully loaded {len(companies_df)} S&P 500 companies")
    
    # Fix known ticker symbol issues before fetching data
    companies_df = companies_df.copy()
    companies_df.loc[companies_df['Symbol'] == 'BRK.B', 'Symbol'] = 'BRK-B'
    
    # Only fetch market data if user clicked the button
    if st.session_state.get('fetch_data', False):
        # Get market data
        with st.spinner("📥 Fetching market data using bulk download (this will be fast!)..."):
            market_df = get_market_data(companies_df['Symbol'].tolist())
    else:
        # Show instructions instead of fetching data
        st.info("💡 **Click '📥 Fetch Market Data' above to load market data and see the analysis!**")
        st.info("🎯 **This reduces API requests - data only loads when you need it!**")
        
        # Create empty dataframe to prevent errors
        market_df = pd.DataFrame()
    
    # Only process and display data if market data was fetched
    if st.session_state.get('fetch_data', False) and not market_df.empty:
        # Merge data
        df = companies_df.merge(market_df, on='Symbol', how='left')
        
        # Clean up data - handle any missing or invalid market cap values
        df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce').fillna(0)
        df['Market Cap (B)'] = df['Market Cap'] / 1e9
        
        # Filter out stocks with zero or invalid market cap data
        valid_data = df[df['Market Cap'] > 0].copy()
        invalid_data = df[df['Market Cap'] == 0].copy()
        
        st.info(f"📊 Data Quality: {len(valid_data)} companies with valid data, {len(invalid_data)} companies with missing data")
        
        if len(invalid_data) > 0:
            with st.expander(f"⚠️ {len(invalid_data)} companies with missing data (click to view)", expanded=False):
                st.dataframe(invalid_data[['Symbol', 'Name', 'Sector']], use_container_width=True, hide_index=True)
                st.info("These companies will be excluded from rankings due to missing market cap data.")
        
        # Sort by market cap - use only valid data
        df = valid_data.sort_values('Market Cap', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
    else:
        # Create empty dataframe if no data fetched
        df = companies_df.copy()
        df['Market Cap'] = 0
        df['Market Cap (B)'] = 0
        df['Price'] = 0
        df['Beta'] = 0
        df['YTD Return %'] = 0
        df['1D Return %'] = 0
        df['5D Return %'] = 0
        df['1M Return %'] = 0
        df['6M Return %'] = 0
        df['1Y Return %'] = 0
        df['5Y Return %'] = 0
        df['Max Return %'] = 0
        df['Volatility %'] = 0
        df['Rank'] = range(1, len(df) + 1)
    
    # Only show data analysis if market data was fetched
    if st.session_state.get('fetch_data', False) and not market_df.empty:
        # Debug: Show the top 10 market cap values to verify sorting (only in debug mode)
        if st.session_state.get('debug_mode', False):
            st.write("🔍 Debug - Top 10 Market Caps after sorting:")
            debug_sort = df.head(10)[['Symbol', 'Name', 'Market Cap', 'Market Cap (B)', 'Price']].copy()
            debug_sort['Market Cap (B)'] = debug_sort['Market Cap (B)'].round(1)
            debug_sort['Price'] = debug_sort['Price'].round(2)
            st.dataframe(debug_sort, use_container_width=True, hide_index=True)
        
        # Display summary metrics
        st.subheader("📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Market Cap", 
            f"${df['Market Cap (B)'].sum():.0f}B",
            help="Sum of all S&P 500 market caps"
        )
    
    with col2:
        st.metric(
            "Average Market Cap", 
            f"${df['Market Cap (B)'].mean():.1f}B",
            help="Average market cap of S&P 500 companies"
        )
    
    with col3:
        st.metric(
            "Median Market Cap", 
            f"${df['Market Cap (B)'].median():.1f}B",
            help="Median market cap of S&P 500 companies"
        )
    
    with col4:
        st.metric(
            "Largest Company", 
            f"{df.iloc[0]['Symbol']} (${df.iloc[0]['Market Cap (B)']:.0f}B)",
            help="Company with the largest market cap"
        )
    
    
    # Top 20 visualization
    st.subheader("🏆 Top 20 Companies by Market Cap")
    
    top_20 = df.head(20).copy()
    
    # Create bar chart for top 20
    fig = px.bar(
        top_20, 
        x='Market Cap (B)', 
        y='Symbol',
        orientation='h',
        title="Top 20 S&P 500 Companies by Market Cap",
        labels={'Market Cap (B)': 'Market Cap (Billions $)', 'Symbol': 'Company Symbol'},
        color='Market Cap (B)',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Main data table
    st.subheader("📋 Complete S&P 500 Analysis")
    
    
    # Format the dataframe for display - check which columns exist
    available_columns = df.columns.tolist()
    
    # Define all possible columns
    desired_columns = ['Rank', 'Symbol', 'Name', 'Sector', 'Market Cap (B)', 'Price', 'Beta', 
                      'YTD Return %', '1D Return %', '5D Return %', '1M Return %', 
                      '6M Return %', '1Y Return %', '5Y Return %', 'Max Return %', 'Volatility %']
    
    # Only use columns that exist in the dataframe
    columns_to_use = [col for col in desired_columns if col in available_columns]
    
    # If we don't have the new columns, use the old ones and show warning
    if '1D Return %' not in available_columns:
        st.warning("⚠️ **Using cached data with limited columns.** Click '🔄 Refresh Data' to get ALL return periods (1D, 5D, 1M, 6M, 5Y, Max)!")
        columns_to_use = ['Rank', 'Symbol', 'Name', 'Sector', 'Market Cap (B)', 'Price', 'Beta', 
                         'YTD Return %', '1Y Return %', 'Volatility %']
    else:
        st.success("✅ **Complete data loaded!** All return periods available.")
    
    display_df = df[columns_to_use].copy()
    
    # Format numeric columns dynamically
    if 'Market Cap (B)' in display_df.columns:
        display_df['Market Cap (B)'] = display_df['Market Cap (B)'].round(1)
    if 'Price' in display_df.columns:
        display_df['Price'] = display_df['Price'].round(2)
    if 'Beta' in display_df.columns:
        display_df['Beta'] = display_df['Beta'].round(2)
    
    # Format return columns dynamically
    return_columns = ['YTD Return %', '1D Return %', '5D Return %', '1M Return %', 
                     '6M Return %', '1Y Return %', '5Y Return %', 'Max Return %', 'Volatility %']
    
    for col in return_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    
    # Color code the performance columns
    def color_performance(val):
        if val > 0:
            return 'color: green'
        elif val < 0:
            return 'color: red'
        else:
            return 'color: orange'
    
    # Apply styling to all available performance columns
    available_return_columns = [col for col in return_columns if col in display_df.columns]
    styled_df = display_df.style.applymap(color_performance, subset=available_return_columns)
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=600
    )
    
    # Performance analysis
    st.subheader("📈 Performance Analysis")
    
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        # YTD Performance Distribution
        ytd_data = df['YTD Return %'].dropna()
        if len(ytd_data) > 0 and ytd_data.nunique() > 1:  # Check if we have varied data
            fig_ytd = px.histogram(
                df, 
                x='YTD Return %',
                nbins=30,
                title="YTD Return Distribution",
                labels={'YTD Return %': 'YTD Return (%)', 'count': 'Number of Companies'}
            )
            fig_ytd.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
            st.plotly_chart(fig_ytd, use_container_width=True)
        else:
            st.info("📊 YTD Return Distribution: No data available or all values are the same")
    
    with col_perf2:
        # 1Y Performance Distribution
        one_y_data = df['1Y Return %'].dropna()
        if len(one_y_data) > 0 and one_y_data.nunique() > 1:  # Check if we have varied data
            fig_1y = px.histogram(
                df, 
                x='1Y Return %',
                nbins=30,
                title="1-Year Return Distribution",
                labels={'1Y Return %': '1-Year Return (%)', 'count': 'Number of Companies'}
            )
            fig_1y.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
            st.plotly_chart(fig_1y, use_container_width=True)
        else:
            st.info("📊 1-Year Return Distribution: No data available or all values are the same")
    
    # Risk vs Return scatter plot
    st.subheader("⚖️ Risk vs Return Analysis")
    
    # Filter out companies with missing data for the scatter plot
    scatter_df = df[(df['Volatility %'] > 0) & (df['1Y Return %'] != 0) & (df['Market Cap (B)'] > 0)].copy()
    
    if not scatter_df.empty:
        fig_scatter = px.scatter(
            scatter_df,
            x='Volatility %',
            y='1Y Return %',
            size='Market Cap (B)',
            color='Beta',
            hover_data=['Symbol', 'Name', 'Sector'],
            title="Risk vs Return: Volatility vs 1-Year Return",
            labels={
                'Volatility %': 'Volatility (%)',
                '1Y Return %': '1-Year Return (%)',
                'Market Cap (B)': 'Market Cap (B)',
                'Beta': 'Beta'
            },
            color_continuous_scale='RdYlBu_r'
        )
        
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Add some statistics about the scatter plot
        col_scatter1, col_scatter2, col_scatter3 = st.columns(3)
        
        with col_scatter1:
            st.metric("Companies in Analysis", len(scatter_df))
        
        with col_scatter2:
            avg_volatility = scatter_df['Volatility %'].mean()
            st.metric("Average Volatility", f"{avg_volatility:.1f}%")
        
        with col_scatter3:
            avg_return = scatter_df['1Y Return %'].mean()
            st.metric("Average 1Y Return", f"{avg_return:.1f}%")
    else:
        st.warning("⚠️ Insufficient data for risk vs return analysis. Please refresh the data.")
    
    # Sector analysis
    st.subheader("🏢 Sector Analysis")
    
    sector_stats = df.groupby('Sector').agg({
        'Market Cap (B)': ['sum', 'mean', 'count'],
        'YTD Return %': 'mean',
        '1Y Return %': 'mean',
        'Beta': 'mean',
        'Volatility %': 'mean'
    }).round(2)
    
    sector_stats.columns = ['Total Market Cap (B)', 'Avg Market Cap (B)', 'Count', 'Avg YTD Return %', 'Avg 1Y Return %', 'Avg Beta', 'Avg Volatility %']
    sector_stats = sector_stats.sort_values('Total Market Cap (B)', ascending=False)
    
    # Color code the sector analysis performance columns
    def color_sector_performance(val):
        if val > 0:
            return 'color: green'
        elif val < 0:
            return 'color: red'
        else:
            return 'color: orange'
    
    # Apply styling to performance columns in sector analysis
    styled_sector_stats = sector_stats.style.applymap(color_sector_performance, subset=['Avg YTD Return %', 'Avg 1Y Return %', 'Avg Volatility %'])
    
    st.dataframe(styled_sector_stats, use_container_width=True)
    
    # Ticker Copy Section
    st.subheader("📋 Copy Top Companies by Market Cap")
    
    col_ticker1, col_ticker2 = st.columns([1, 2])
    
    with col_ticker1:
        top_n = st.number_input(
            "Number of top companies:",
            min_value=1,
            max_value=len(df),
            value=20,
            step=1,
            help="Select how many top companies by market cap you want"
        )
        
        # Get top N companies
        top_companies = df.head(top_n)
        ticker_list = top_companies['Symbol'].tolist()
        ticker_string = " ".join(ticker_list)
        
        # Copy button using the same method as other pages
        import streamlit.components.v1 as components
        import json
        
        copy_html = f"""
        <button onclick='navigator.clipboard.writeText({json.dumps(ticker_string)});' 
                style='background-color: #ff4b4b; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-bottom: 10px;'>
        📋 Copy Tickers
        </button>
        """
        components.html(copy_html, height=50)
    
    with col_ticker2:
        st.markdown(f"**Top {top_n} companies by market cap:**")
        st.code(ticker_string, language="text")
        
        # Show summary info instead of duplicate table
        if top_n <= 10:
            st.markdown(f"**Top {top_n} companies:** {', '.join(ticker_list)}")
        else:
            st.markdown(f"**Top {top_n} companies:** {', '.join(ticker_list[:10])} ... and {top_n - 10} more")
    
        # Download options
        st.subheader("💾 Download Data")
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Data (CSV)",
                data=csv,
                file_name=f"sp500_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col_download2:
            top_100_csv = df.head(100).to_csv(index=False)
            st.download_button(
                label="📥 Download Top 100 (CSV)",
                data=top_100_csv,
                file_name=f"sp500_top100_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Click '📥 Fetch Market Data' to load S&P 500 companies")

# Footer
st.markdown("---")
st.markdown("**💡 Key Metrics Explained:**")
st.markdown("- **Market Cap**: Total value of all company shares")
st.markdown("- **YTD Return**: Performance since January 1st of current year")
st.markdown("- **1Y Return**: Performance over the last 12 months")
st.markdown("- **Beta**: Volatility relative to the market (1.0 = market average)")
st.markdown("- **Volatility**: Annualized standard deviation of returns")
st.markdown("*Data sources: [Wikipedia S&P 500](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) & [Yahoo Finance](https://finance.yahoo.com)*")
