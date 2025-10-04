import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

def create_dbmf_complete_ticker():
    """
    Create a complete DBMF ticker combining historical CSV data with Yahoo Finance data.
    DBMF = iMGP DBi Managed Futures Strategy ETF
    """
    try:
        print("Loading DBMF historical data...")
        
        # Load historical DBMF data from CSV (no header)
        historical_data = pd.read_csv('Complete_Tickers/Historical CSV/DBMF Historical Data Monthly.csv', 
                                    sep='\t', header=None, parse_dates=[0], index_col=0)
        
        # Clean and convert price column (remove $ and % signs)
        historical_data[2] = historical_data[2].astype(str).str.replace('$', '')
        historical_prices = pd.to_numeric(historical_data[2], errors='coerce')
        
        print(f"CSV columns: {historical_data.columns.tolist()}")
        print(f"CSV shape: {historical_data.shape}")
        
        # The CSV has columns: Return%, Price (after setting Date as index)
        # We need the price column (2nd column, index 1)
        historical_prices = historical_data.iloc[:, 1]  # Price column (2nd column)
        
        # Convert price strings to float - remove commas and dollar signs first
        historical_prices = historical_prices.astype(str).str.replace(',', '').str.replace('$', '')
        historical_prices = pd.to_numeric(historical_prices, errors='coerce').dropna()
        
        print(f"Historical DBMF data: {len(historical_prices)} records")
        print(f"Date range: {historical_prices.index[0]} to {historical_prices.index[-1]}")
        print(f"Price range: ${historical_prices.min():,.2f} to ${historical_prices.max():,.2f}")
        
        # Get recent DBMF data from Yahoo Finance
        print("Fetching recent DBMF data from Yahoo Finance...")
        dbmf_ticker = yf.Ticker("DBMF")
        dbmf_data = dbmf_ticker.history(period="max")
        
        if dbmf_data.empty:
            print("No recent DBMF data found, using historical data only")
            return historical_prices
        
        # Use Close prices from Yahoo Finance
        dbmf_close = dbmf_data['Close'].dropna()
        
        # Remove timezone info from Yahoo Finance data to match historical data
        dbmf_close.index = dbmf_close.index.tz_localize(None)
        
        print(f"Recent DBMF data: {len(dbmf_close)} records")
        print(f"Date range: {dbmf_close.index[0]} to {dbmf_close.index[-1]}")
        
        # Find where Yahoo Finance data starts to have reliable data
        yahoo_start_date = dbmf_close.index[0]
        
        print(f"Historical CSV data: {len(historical_prices)} records")
        print(f"Yahoo Finance data: {len(dbmf_close)} records")
        print(f"Yahoo Finance starts: {yahoo_start_date}")
        
        # Get historical CSV data that comes before Yahoo Finance data
        historical_before_yahoo = historical_prices[historical_prices.index < yahoo_start_date]
        
        if len(historical_before_yahoo) > 0:
            print(f"Using {len(historical_before_yahoo)} historical records before Yahoo Finance")
            
            # Find the overlap point to calculate scaling
            last_historical_price = historical_before_yahoo.iloc[-1]
            first_yahoo_price = dbmf_close.iloc[0]
            
            # Calculate scaling to match Yahoo Finance at the transition
            scaling_factor = first_yahoo_price / last_historical_price
            
            print(f"Scaling factor: {scaling_factor:.6f}")
            print(f"Last historical: ${last_historical_price:,.2f}")
            print(f"First Yahoo price: ${first_yahoo_price:,.2f}")
            print(f"Scaled historical: ${last_historical_price * scaling_factor:,.2f}")
            
            # Scale historical data to match Yahoo Finance
            historical_scaled = historical_before_yahoo * scaling_factor
            
            # Combine scaled historical + Yahoo Finance data
            complete_dbmf = pd.concat([historical_scaled, dbmf_close])
            complete_dbmf = complete_dbmf.sort_index().drop_duplicates()
        else:
            print("No historical data before Yahoo Finance, using Yahoo Finance only")
            complete_dbmf = dbmf_close
        
        print(f"Complete DBMF ticker created: {len(complete_dbmf)} total records")
        print(f"Full date range: {complete_dbmf.index[0]} to {complete_dbmf.index[-1]}")
        print(f"Final price range: ${complete_dbmf.min():,.2f} to ${complete_dbmf.max():,.2f}")
        
        return complete_dbmf
        
    except Exception as e:
        print(f"Error creating DBMF complete ticker: {str(e)}")
        return None

# Test the function
if __name__ == "__main__":
    dbmf_data = create_dbmf_complete_ticker()
    if dbmf_data is not None:
        print(f"\nDBMF ticker test successful!")
        print(f"Total records: {len(dbmf_data)}")
        print(f"Date range: {dbmf_data.index[0]} to {dbmf_data.index[-1]}")
        print(f"Latest price: ${dbmf_data.iloc[-1]:,.2f}")
    else:
        print("DBMF ticker test failed!")
