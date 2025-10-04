import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

def create_bitcoin_complete_ticker():
    """
    Create a complete Bitcoin ticker combining historical CSV data with Yahoo Finance data.
    Similar to GOLD, TLT, and ZROZ complete tickers.
    """
    try:
        print("Loading Bitcoin historical data...")
        
        # Load historical Bitcoin data from CSV
        historical_data = pd.read_csv('Complete_Tickers/Historical CSV/BITCOIN Hisotrical Data Monthly.csv', 
                                    sep='\t', parse_dates=[0], index_col=0)
        
        print(f"CSV columns: {historical_data.columns.tolist()}")
        print(f"CSV shape: {historical_data.shape}")
        print(f"CSV head: {historical_data.head()}")
        print(f"CSV tail: {historical_data.tail()}")
        
        # The CSV has columns: Return%, Price (after setting Date as index)
        # We need the price column (2nd column, index 1)
        historical_prices = historical_data.iloc[:, 1]  # Price column (2nd column)
        
        print(f"Raw price column length: {len(historical_prices)}")
        print(f"Raw price column sample: {historical_prices.head(10)}")
        print(f"Raw price column tail: {historical_prices.tail(10)}")
        
        # Convert price strings to float - remove commas first
        historical_prices = historical_prices.astype(str).str.replace(',', '').str.replace('$', '')
        historical_prices = pd.to_numeric(historical_prices, errors='coerce')
        
        print(f"After numeric conversion: {len(historical_prices)}")
        print(f"Non-null values: {historical_prices.notna().sum()}")
        print(f"Null values: {historical_prices.isna().sum()}")
        
        historical_prices = historical_prices.dropna()
        
        print(f"Historical Bitcoin data: {len(historical_prices)} records")
        print(f"Date range: {historical_prices.index[0]} to {historical_prices.index[-1]}")
        print(f"Price range: ${historical_prices.min():,.2f} to ${historical_prices.max():,.2f}")
        
        # Get recent Bitcoin data from Yahoo Finance (BTC-USD)
        print("Fetching recent Bitcoin data from Yahoo Finance...")
        btc_ticker = yf.Ticker("BTC-USD")
        btc_data = btc_ticker.history(period="max")
        
        if btc_data.empty:
            print("No recent Bitcoin data found, using historical data only")
            return historical_prices
        
        # Use Close prices from Yahoo Finance
        btc_close = btc_data['Close'].dropna()
        
        # Remove timezone info from Yahoo Finance data to match historical data
        btc_close.index = btc_close.index.tz_localize(None)
        
        print(f"Recent Bitcoin data: {len(btc_close)} records")
        print(f"Date range: {btc_close.index[0]} to {btc_close.index[-1]}")
        
        # Find where Yahoo Finance data starts to have reliable data
        # Use Yahoo Finance as the primary source and fill early gaps with CSV data
        yahoo_start_date = btc_close.index[0]
        
        print(f"Historical CSV data: {len(historical_prices)} records")
        print(f"Yahoo Finance data: {len(btc_close)} records")
        print(f"Yahoo Finance starts: {yahoo_start_date}")
        
        # Get historical CSV data that comes before Yahoo Finance data
        historical_before_yahoo = historical_prices[historical_prices.index < yahoo_start_date]
        
        if len(historical_before_yahoo) > 0:
            print(f"Using {len(historical_before_yahoo)} historical records before Yahoo Finance")
            
            # Find the overlap point to calculate scaling
            # Use the last historical price and first Yahoo Finance price
            last_historical_price = historical_before_yahoo.iloc[-1]
            first_yahoo_price = btc_close.iloc[0]
            
            # Calculate scaling to match Yahoo Finance at the transition
            scaling_factor = first_yahoo_price / last_historical_price
            
            print(f"Scaling factor: {scaling_factor:.6f}")
            print(f"Last historical: ${last_historical_price:,.2f}")
            print(f"First Yahoo price: ${first_yahoo_price:,.2f}")
            print(f"Scaled historical: ${last_historical_price * scaling_factor:,.2f}")
            
            # Scale historical data to match Yahoo Finance
            historical_scaled = historical_before_yahoo * scaling_factor
            
            # Combine scaled historical + Yahoo Finance data
            complete_bitcoin = pd.concat([historical_scaled, btc_close])
            complete_bitcoin = complete_bitcoin.sort_index().drop_duplicates()
        else:
            print("No historical data before Yahoo Finance, using Yahoo Finance only")
            complete_bitcoin = btc_close
        
        print(f"Complete Bitcoin ticker created: {len(complete_bitcoin)} total records")
        print(f"Full date range: {complete_bitcoin.index[0]} to {complete_bitcoin.index[-1]}")
        print(f"Final price range: ${complete_bitcoin.min():,.2f} to ${complete_bitcoin.max():,.2f}")
        
        # Create a debug plot to show the transition point
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot scaled historical data (if it exists)
        if len(historical_before_yahoo) > 0:
            ax.plot(historical_scaled.index, historical_scaled.values, 
                    color='blue', linewidth=2, label='Scaled Historical CSV Data', alpha=0.8)
        
        # Plot Yahoo Finance data
        ax.plot(btc_close.index, btc_close.values, 
                color='red', linewidth=2, label='Yahoo Finance Data', alpha=0.8)
        
        # Add vertical line at transition point (if historical data exists)
        if len(historical_before_yahoo) > 0:
            transition_date = historical_scaled.index[-1]
            ax.axvline(x=transition_date, color='green', linestyle='--', linewidth=3, 
                       label=f'Transition Point: {transition_date.strftime("%Y-%m-%d")}')
            
            # Add transition point values
            ax.annotate(f'Scaled Historical: ${last_historical_price * scaling_factor:,.2f}', 
                       xy=(transition_date, last_historical_price * scaling_factor), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7),
                       fontsize=10, color='white')
            
            ax.annotate(f'Yahoo Finance: ${first_yahoo_price:,.2f}', 
                       xy=(transition_date, first_yahoo_price), 
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                       fontsize=10, color='white')
        
        ax.set_title('Bitcoin Complete Ticker - Transition Point Verification', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Bitcoin Price ($)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return complete_bitcoin
        
    except Exception as e:
        print(f"Error creating Bitcoin complete ticker: {str(e)}")
        return None

# Test the function
if __name__ == "__main__":
    bitcoin_data = create_bitcoin_complete_ticker()
    if bitcoin_data is not None:
        print(f"\nBitcoin ticker test successful!")
        print(f"Total records: {len(bitcoin_data)}")
        print(f"Date range: {bitcoin_data.index[0]} to {bitcoin_data.index[-1]}")
        print(f"Latest price: ${bitcoin_data.iloc[-1]:,.2f}")
    else:
        print("Bitcoin ticker test failed!")
