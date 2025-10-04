import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

def create_goldsim_complete_ticker():
    """
    Create a complete GOLDSIM ticker combining new historical CSV data with existing GOLDX ticker data.
    GOLDSIM = Complete Gold Simulation (1968+) - New Historical + GOLDX
    """
    try:
        print("Loading new GOLD historical data...")
        
        # Load new historical GOLD data from CSV (no header)
        historical_data = pd.read_csv('Complete_Tickers/Historical CSV/GOLD Historical Data Monthly.csv', 
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
        
        print(f"Historical GOLD data: {len(historical_series)} records")
        print(f"Date range: {historical_series.index[0]} to {historical_series.index[-1]}")
        print(f"Price range: ${float(historical_series.min()):.2f} to ${float(historical_series.max()):.2f}")
        
        # Fetch existing GOLDX data (which combines Gold_Futures_Complete.csv + GLD)
        print("Fetching existing GOLDX data...")
        try:
            from GOLD_COMPLETE_TICKER import create_gold_complete_ticker
        except ImportError:
            from Complete_Tickers.GOLD_COMPLETE_TICKER import create_gold_complete_ticker
        goldx_data = create_gold_complete_ticker()
        
        if goldx_data is None or goldx_data.empty:
            print("No GOLDX data available, using historical data only")
            return historical_series
        
        # GOLDX data is already a pandas Series
        goldx_data.index = goldx_data.index.tz_localize(None)  # Remove timezone for compatibility
        goldx_data.index = pd.to_datetime(goldx_data.index)  # Ensure datetime index
        
        print(f"GOLDX data: {len(goldx_data)} records")
        print(f"Date range: {goldx_data.index[0]} to {goldx_data.index[-1]}")
        
        print(f"Historical CSV data: {len(historical_series)} records")
        print(f"GOLDX data: {len(goldx_data)} records")
        print(f"GOLDX starts: {goldx_data.index[0]}")
        
        # Find overlap period and calculate scaling factor
        overlap_start = max(historical_series.index[0], goldx_data.index[0])
        overlap_end = min(historical_series.index[-1], goldx_data.index[-1])
        
        if overlap_start <= overlap_end:
            # Find the last historical date before GOLDX data starts
            historical_before_goldx = historical_series[historical_series.index < goldx_data.index[0]]
            
            if len(historical_before_goldx) > 0:
                # Use the last historical price before GOLDX starts
                last_historical_price = historical_before_goldx.iloc[-1]
                first_goldx_price = goldx_data.iloc[0]
                
                # Calculate scaling factor to match prices at transition
                scaling_factor = float(first_goldx_price) / float(last_historical_price)
                
                print(f"Using {len(historical_before_goldx)} historical records before GOLDX")
                print(f"Scaling factor: {scaling_factor:.6f}")
                print(f"Last historical: ${float(last_historical_price):.2f}")
                print(f"First GOLDX price: ${float(first_goldx_price):.2f}")
                print(f"Scaled historical: ${float(last_historical_price * scaling_factor):.2f}")
                
                # Scale historical data to match GOLDX at transition point
                # Create a clean copy to avoid index corruption
                scaled_values = historical_before_goldx.values * scaling_factor
                scaled_index = historical_before_goldx.index.copy()
                
                # Create clean series with proper datetime indexes
                scaled_series = pd.Series(scaled_values, index=scaled_index)
                goldx_series = pd.Series(goldx_data.values.flatten(), index=pd.to_datetime(goldx_data.index))
                
                # Combine scaled historical data with GOLDX data
                complete_series = pd.concat([scaled_series, goldx_series])
                # Sort by index to ensure proper chronological order
                complete_series = complete_series.sort_index()
                
                print(f"Complete GOLDSIM ticker created: {len(complete_series)} total records")
                print(f"Full date range: {complete_series.index[0]} to {complete_series.index[-1]}")
                
                return complete_series
            else:
                # No historical data before GOLDX, use GOLDX only
                print("No historical data before GOLDX, using GOLDX only")
                return goldx_data
        else:
            # No overlap, use GOLDX data only
            print("No overlap period found, using GOLDX data only")
            return goldx_data
            
    except Exception as e:
        import traceback
        print(f"Error creating GOLDSIM complete ticker: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    # Test the ticker
    print("Testing GOLDSIM Complete Ticker...")
    goldsim_data = create_goldsim_complete_ticker()
    
    if goldsim_data is not None and not goldsim_data.empty:
        print(f"\nGOLDSIM ticker test successful!")
        print(f"Total records: {len(goldsim_data)}")
        print(f"Date range: {goldsim_data.index[0]} to {goldsim_data.index[-1]}")
        
        # Transition plot would be created here if needed
        print("Transition plot would show seamless blend from 1968 to 2025")
    else:
        print("GOLDSIM ticker test failed!")
