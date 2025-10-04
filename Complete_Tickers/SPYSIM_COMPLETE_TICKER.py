import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

def create_spysim_complete_ticker():
    """
    Create a complete SPYSIM ticker combining historical CSV data with Yahoo Finance SPYTR data.
    SPYSIM = Complete S&P 500 Simulation (1885+) - Historical + SPYTR
    """
    try:
        print("Loading SPY historical data...")
        
        # Load historical SPY data from CSV (no header)
        historical_data = pd.read_csv('Complete_Tickers/Historical CSV/SPY Historical Data Monthly.csv', 
                                    sep='\t', header=None)
        
        # Parse dates manually to ensure they're all datetime
        historical_data[0] = pd.to_datetime(historical_data[0], errors='coerce')
        historical_data = historical_data.set_index(0).dropna()
        
        # Clean and convert price column (remove $, % signs, and commas)
        historical_data[2] = historical_data[2].astype(str).str.replace('$', '').str.replace(',', '')
        historical_prices = pd.to_numeric(historical_data[2], errors='coerce')
        
        print(f"CSV columns: {historical_data.columns.tolist()}")
        print(f"CSV shape: {historical_data.shape}")
        
        # Create historical price series
        historical_series = pd.Series(historical_prices.values, index=historical_data.index)
        historical_series = historical_series.dropna()
        historical_series.index = pd.to_datetime(historical_series.index)  # Ensure datetime index
        
        print(f"Historical SPY data: {len(historical_series)} records")
        print(f"Date range: {historical_series.index[0]} to {historical_series.index[-1]}")
        print(f"Price range: ${float(historical_series.min()):.2f} to ${float(historical_series.max()):.2f}")
        
        # Fetch recent SPYTR data from Yahoo Finance
        print("Fetching recent SPYTR data from Yahoo Finance...")
        spytr = yf.download("^SP500TR", period="max", progress=False)
        
        if spytr.empty:
            print("No SPYTR data available from Yahoo Finance")
            return historical_series
        
        # Use Close price for SPYTR (SP500TR doesn't have Adj Close)
        spytr_data = spytr['Close'].dropna()
        spytr_data.index = spytr_data.index.tz_localize(None)  # Remove timezone for compatibility
        spytr_data.index = pd.to_datetime(spytr_data.index)  # Ensure datetime index
        
        print(f"Recent SPYTR data: {len(spytr_data)} records")
        print(f"Date range: {spytr_data.index[0]} to {spytr_data.index[-1]}")
        
        print(f"Historical CSV data: {len(historical_series)} records")
        print(f"Yahoo Finance data: {len(spytr_data)} records")
        print(f"Yahoo Finance starts: {spytr_data.index[0]}")
        
        # Find overlap period and calculate scaling factor
        overlap_start = max(historical_series.index[0], spytr_data.index[0])
        overlap_end = min(historical_series.index[-1], spytr_data.index[-1])
        
        if overlap_start <= overlap_end:
            # Find the last historical date before Yahoo Finance data starts
            historical_before_yahoo = historical_series[historical_series.index < spytr_data.index[0]]
            
            if len(historical_before_yahoo) > 0:
                # Use the last historical price before Yahoo Finance starts
                last_historical_price = historical_before_yahoo.iloc[-1]
                first_yahoo_price = spytr_data.iloc[0]
                
                # Calculate scaling factor to match prices at transition
                scaling_factor = float(first_yahoo_price) / float(last_historical_price)
                
                print(f"Using {len(historical_before_yahoo)} historical records before Yahoo Finance")
                print(f"Scaling factor: {float(scaling_factor):.6f}")
                print(f"Last historical: ${float(last_historical_price):.2f}")
                print(f"First Yahoo price: ${float(first_yahoo_price):.2f}")
                print(f"Scaled historical: ${float(last_historical_price * scaling_factor):.2f}")
                
                # Scale historical data to match Yahoo Finance at transition point
                # Create a clean copy to avoid index corruption
                scaled_values = historical_before_yahoo.values * scaling_factor
                scaled_index = historical_before_yahoo.index.copy()
                
                # Create clean series with proper datetime indexes
                scaled_series = pd.Series(scaled_values, index=scaled_index)
                spytr_series = pd.Series(spytr_data.values.flatten(), index=pd.to_datetime(spytr_data.index))
                
                # Combine scaled historical data with Yahoo Finance data
                complete_series = pd.concat([scaled_series, spytr_series])
                # Sort by index to ensure proper chronological order
                complete_series = complete_series.sort_index()
                
                print(f"Complete SPYSIM ticker created: {len(complete_series)} total records")
                print(f"Full date range: {complete_series.index[0]} to {complete_series.index[-1]}")
                # print(f"Final price range: ${complete_series.min():.2f} to ${complete_series.max():.2f}")
                
                return complete_series
            else:
                # No historical data before Yahoo Finance, use Yahoo Finance only
                print("No historical data before Yahoo Finance, using Yahoo Finance only")
                return spytr_data
        else:
            # No overlap, use Yahoo Finance data only
            print("No overlap period found, using Yahoo Finance data only")
            return spytr_data
            
    except Exception as e:
        import traceback
        print(f"Error creating SPYSIM complete ticker: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    # Test the ticker
    print("Testing SPYSIM Complete Ticker...")
    spysim_data = create_spysim_complete_ticker()
    
    if spysim_data is not None and not spysim_data.empty:
        print(f"\nSPYSIM ticker test successful!")
        print(f"Total records: {len(spysim_data)}")
        print(f"Date range: {spysim_data.index[0]} to {spysim_data.index[-1]}")
        # print(f"Latest price: ${spysim_data.iloc[-1]:.2f}")
        
        # Transition plot would be created here if needed
        print("Transition plot would show seamless blend from 1885 to 2025")
    else:
        print("SPYSIM ticker test failed!")
