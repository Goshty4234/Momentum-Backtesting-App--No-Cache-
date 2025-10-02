import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import warnings
import urllib3
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_complete_zroz_data():
    """Load the complete ZROZ historical data from CSV"""
    try:
        # Load the monthly ZROZ data (no headers, tab-separated)
        df = pd.read_csv('Complete_Tickers/Historical CSV/ZROZ Historical Data Monthly.csv', header=None, sep='\t',
                        names=['Date', 'Percentage_Change', 'Dollar_Value'])
        
        # Convert Date column to datetime (add day 01 for monthly data)
        df['Date'] = pd.to_datetime(df['Date'] + '-01')
        df.set_index('Date', inplace=True)
        
        # Clean and convert percentage change column
        df['Percentage_Change'] = df['Percentage_Change'].str.replace('%', '').astype(float)
        
        # Clean and convert dollar value column
        df['Dollar_Value'] = df['Dollar_Value'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Create a price series from the dollar values
        # We'll use the dollar value as our price proxy
        df['Close'] = df['Dollar_Value']
        
        print(f"Loaded ZROZ historical data: {len(df)} monthly records")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Sample data:")
        print(df.head())
        
        return df[['Close', 'Percentage_Change']]
        
    except Exception as e:
        print(f"Error loading ZROZ historical data: {e}")
        return None

def get_recent_zroz_data_safe(period="max"):
    """Safely fetch recent ZROZ data from Yahoo Finance"""
    try:
        print("Fetching recent ZROZ data from Yahoo Finance...")
        
        # Fetch ZROZ data from Yahoo Finance
        zroz_data = yf.download(
            'ZROZ', 
            period=period,
            timeout=5,
            threads=False
        )
        
        if zroz_data.empty:
            print("No ZROZ data received from Yahoo Finance")
            return None
            
        # Flatten multi-level columns if they exist
        if isinstance(zroz_data.columns, pd.MultiIndex):
            zroz_data.columns = zroz_data.columns.get_level_values(0)
        
        print(f"Fetched ZROZ data: {len(zroz_data)} records")
        print(f"Date range: {zroz_data.index.min()} to {zroz_data.index.max()}")
        
        return zroz_data
        
    except Exception as e:
        print(f"Error fetching ZROZ data: {e}")
        return None

def create_safe_zroz_ticker():
    """Create a complete ZROZ ticker combining historical CSV data with live Yahoo Finance data"""
    try:
        print("Creating complete ZROZ ticker...")
        
        # Load historical data
        historical_data = load_complete_zroz_data()
        if historical_data is None:
            print("Failed to load historical data")
            return None
        
        # Get recent ZROZ data
        recent_data = get_recent_zroz_data_safe()
        if recent_data is None:
            print("Failed to fetch recent data, using historical data only")
            return historical_data
        
        # ZROZ IPO date (November 4, 2009)
        zroz_start_date = pd.Timestamp('2009-11-04')
        
        # Split historical data
        historical_before_2009 = historical_data[historical_data.index < zroz_start_date]
        historical_on_2009 = historical_data[historical_data.index.date == zroz_start_date.date()]
        historical_after_2009 = historical_data[historical_data.index > zroz_start_date]
        
        print(f"Historical data before 2009: {len(historical_before_2009)} records")
        print(f"Historical data on 2009: {len(historical_on_2009)} records")
        print(f"Historical data after 2009: {len(historical_after_2009)} records")
        
        # Get the last historical price before ZROZ IPO
        if not historical_before_2009.empty:
            last_historical_date = historical_before_2009.index[-1]
            last_historical_price = historical_before_2009['Close'].iloc[-1]
            print(f"Last historical price before ZROZ IPO: ${last_historical_price:.2f} on {last_historical_date.date()}")
        else:
            print("No historical data before ZROZ IPO")
            return recent_data
        
        # Get the first ZROZ price on IPO date
        first_zroz_data = recent_data[recent_data.index.date == zroz_start_date.date()]
        if first_zroz_data.empty:
            # If no data on exact IPO date, get the first available data
            first_zroz_data = recent_data.head(1)
            first_zroz_date = recent_data.index[0]
        else:
            first_zroz_date = first_zroz_data.index[0]
        
        first_zroz_price = first_zroz_data['Close'].iloc[0]
        print(f"First ZROZ price on IPO: ${first_zroz_price:.2f} on {first_zroz_date.date()}")
        
        # Calculate scaling factor to match prices
        scaling_factor = last_historical_price / first_zroz_price
        print(f"Scaling factor: {scaling_factor:.6f}")
        
        # Scale the recent data
        recent_data_scaled = recent_data.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in recent_data_scaled.columns:
                recent_data_scaled[col] = recent_data_scaled[col] * scaling_factor
        
        first_zroz_price_scaled = first_zroz_price * scaling_factor
        print(f"First scaled ZROZ price: ${first_zroz_price_scaled:.2f}")
        print(f"Price difference: ${abs(first_zroz_price_scaled - last_historical_price):.2f}")
        
        # Combine historical data (before 2009) with scaled recent data
        combined_data = pd.concat([historical_before_2009, recent_data_scaled])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data = combined_data.sort_index()
        
        # Fill any gaps with forward fill
        combined_data = combined_data.reindex(
            pd.date_range(combined_data.index.min(), combined_data.index.max(), freq='D')
        ).fillna(method='ffill')
        
        print(f"Combined data: {len(combined_data)} records")
        print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        
        # Verify transition day continuity
        if not historical_before_2009.empty and not recent_data_scaled.empty:
            last_hist_price = historical_before_2009['Close'].iloc[-1]
            first_zroz_scaled = recent_data_scaled['Close'].iloc[0]
            print(f"Transition verification:")
            print(f"  Last historical: ${last_hist_price:.2f}")
            print(f"  First ZROZ scaled: ${first_zroz_scaled:.2f}")
            print(f"  Difference: ${abs(last_hist_price - first_zroz_scaled):.2f}")
        
        return combined_data
        
    except Exception as e:
        print(f"Error creating complete ZROZ ticker: {e}")
        return None

def plot_zroz_data(data):
    """Plot the ZROZ data to visualize the transition"""
    try:
        if data is None or data.empty:
            print("No data to plot")
            return
        
        plt.figure(figsize=(15, 8))
        
        # ZROZ IPO date
        zroz_start_date = pd.Timestamp('2009-11-04')
        
        # Split data for different colors
        historical_data = data[data.index < zroz_start_date]
        zroz_data = data[data.index >= zroz_start_date]
        
        # Plot historical data in gold color
        if not historical_data.empty:
            plt.plot(historical_data.index, historical_data['Close'], 
                   color='gold', linewidth=2, label='Historical ZROZ Data (Monthly)')
        
        # Plot ZROZ ETF data in blue
        if not zroz_data.empty:
            plt.plot(zroz_data.index, zroz_data['Close'], 
                   color='blue', linewidth=2, label='ZROZ ETF Data (Daily)')
        
        # Add vertical line at transition
        plt.axvline(x=zroz_start_date, color='red', linestyle='--', 
                   linewidth=2, label='ZROZ ETF IPO (2009-11-04)')
        
        plt.title('Complete ZROZ Historical Data (1962-2025)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("ZROZ data plot displayed successfully!")
        
    except Exception as e:
        print(f"Error plotting ZROZ data: {e}")

def test_safe_zroz_ticker():
    """Test the complete ZROZ ticker"""
    print("=" * 60)
    print("TESTING COMPLETE ZROZ TICKER")
    print("=" * 60)
    
    # Create the ticker
    zroz_data = create_safe_zroz_ticker()
    
    if zroz_data is not None:
        print(f"\nSUCCESS: ZROZ ticker created successfully!")
        print(f"Total records: {len(zroz_data)}")
        print(f"Date range: {zroz_data.index.min()} to {zroz_data.index.max()}")
        print(f"Latest price: ${zroz_data['Close'].iloc[-1]:.2f}")
        
        # Show sample data
        print(f"\nSample data (first 3 rows):")
        print(zroz_data.head(3))
        print(f"\nSample data (last 3 rows):")
        print(zroz_data.tail(3))
        
        # Plot the data
        plot_zroz_data(zroz_data)
        
    else:
        print("ERROR: Failed to create ZROZ ticker")

if __name__ == "__main__":
    test_safe_zroz_ticker()
