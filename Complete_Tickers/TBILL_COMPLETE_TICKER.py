import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

def create_tbill_complete_ticker():
    """
    Create a complete TBILL ticker combining historical CSV data with Yahoo Finance data.
    TBILL = Short-term US Treasury Bills
    Transitions to SGOV (iShares 0-3 Month Treasury Bond ETF) for recent data
    """
    try:
        print("Loading TBILL historical data...")
        
        # Load historical TBILL data from CSV
        historical_data = pd.read_csv('Complete_Tickers/Historical CSV/TBILL Historical Data Monthly.csv', 
                                    sep='\t', parse_dates=[0], index_col=0)
        
        print(f"CSV columns: {historical_data.columns.tolist()}")
        print(f"CSV shape: {historical_data.shape}")
        
        # The CSV has columns: Return%, Price (after setting Date as index)
        # We need the price column (2nd column, index 1)
        historical_prices = historical_data.iloc[:, 1]  # Price column (2nd column)
        
        # Convert price strings to float - remove commas and dollar signs first
        historical_prices = historical_prices.astype(str).str.replace(',', '').str.replace('$', '')
        historical_prices = pd.to_numeric(historical_prices, errors='coerce').dropna()
        
        print(f"Historical TBILL data: {len(historical_prices)} records")
        print(f"Date range: {historical_prices.index[0]} to {historical_prices.index[-1]}")
        print(f"Price range: ${historical_prices.min():,.2f} to ${historical_prices.max():,.2f}")
        
        # Get recent SGOV data from Yahoo Finance (Short-term Treasury ETF)
        print("Fetching recent SGOV data from Yahoo Finance...")
        sgov_ticker = yf.Ticker("SGOV")
        sgov_data = sgov_ticker.history(period="max")
        
        if sgov_data.empty:
            print("No recent SGOV data found, using historical data only")
            return historical_prices
        
        # Use Close prices from Yahoo Finance
        sgov_close = sgov_data['Close'].dropna()
        
        # Remove timezone info from Yahoo Finance data to match historical data
        sgov_close.index = sgov_close.index.tz_localize(None)
        
        print(f"Recent SGOV data: {len(sgov_close)} records")
        print(f"Date range: {sgov_close.index[0]} to {sgov_close.index[-1]}")
        
        # Find where Yahoo Finance data starts to have reliable data
        yahoo_start_date = sgov_close.index[0]
        
        print(f"Historical CSV data: {len(historical_prices)} records")
        print(f"Yahoo Finance data: {len(sgov_close)} records")
        print(f"Yahoo Finance starts: {yahoo_start_date}")
        
        # Get historical CSV data that comes before Yahoo Finance data
        historical_before_yahoo = historical_prices[historical_prices.index < yahoo_start_date]
        
        if len(historical_before_yahoo) > 0:
            print(f"Using {len(historical_before_yahoo)} historical records before Yahoo Finance")
            
            # Find the overlap point to calculate scaling
            last_historical_price = historical_before_yahoo.iloc[-1]
            first_yahoo_price = sgov_close.iloc[0]
            
            # Calculate scaling to match Yahoo Finance at the transition
            scaling_factor = first_yahoo_price / last_historical_price
            
            print(f"Scaling factor: {scaling_factor:.6f}")
            print(f"Last historical: ${last_historical_price:,.2f}")
            print(f"First Yahoo price: ${first_yahoo_price:,.2f}")
            print(f"Scaled historical: ${last_historical_price * scaling_factor:,.2f}")
            
            # Scale historical data to match Yahoo Finance
            historical_scaled = historical_before_yahoo * scaling_factor
            
            # Combine scaled historical + Yahoo Finance data
            complete_tbill = pd.concat([historical_scaled, sgov_close])
            complete_tbill = complete_tbill.sort_index().drop_duplicates()
        else:
            print("No historical data before Yahoo Finance, using Yahoo Finance only")
            complete_tbill = sgov_close
        
        print(f"Complete TBILL ticker created: {len(complete_tbill)} total records")
        print(f"Full date range: {complete_tbill.index[0]} to {complete_tbill.index[-1]}")
        print(f"Final price range: ${complete_tbill.min():,.2f} to ${complete_tbill.max():,.2f}")
        
        return complete_tbill
        
    except Exception as e:
        print(f"Error creating TBILL complete ticker: {str(e)}")
        return None

# Test the function
if __name__ == "__main__":
    tbill_data = create_tbill_complete_ticker()
    if tbill_data is not None:
        print(f"\nTBILL ticker test successful!")
        print(f"Total records: {len(tbill_data)}")
        print(f"Date range: {tbill_data.index[0]} to {tbill_data.index[-1]}")
        print(f"Latest price: ${tbill_data.iloc[-1]:,.2f}")
    else:
        print("TBILL ticker test failed!")
