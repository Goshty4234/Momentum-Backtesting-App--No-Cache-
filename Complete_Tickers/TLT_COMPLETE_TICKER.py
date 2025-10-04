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

def load_complete_tlt_data():
    """Load the complete TLT historical data from CSV"""
    try:
        # Load the monthly TLT data (no headers, tab-separated)
        df = pd.read_csv('Complete_Tickers/Historical CSV/TLT Historical Data Monthly.csv', header=None, sep='\t',
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
        
        print(f"Loaded TLT historical data: {len(df)} monthly records")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Sample data:")
        print(df.head())
        
        return df[['Close', 'Percentage_Change']]
        
    except Exception as e:
        print(f"Error loading TLT historical data: {e}")
        return None

def get_recent_tlt_data_safe(period="max"):
    """Safely fetch recent TLT data from Yahoo Finance"""
    try:
        print("Fetching recent TLT data from Yahoo Finance...")
        
        # Fetch TLT data from Yahoo Finance
        tlt_data = yf.download(
            'TLT', 
            period=period,
            timeout=5,
            threads=False
        )
        
        if tlt_data.empty:
            print("No TLT data received from Yahoo Finance")
            return None
            
        # Flatten multi-level columns if they exist
        if isinstance(tlt_data.columns, pd.MultiIndex):
            tlt_data.columns = tlt_data.columns.get_level_values(0)
        
        print(f"Fetched TLT data: {len(tlt_data)} records")
        print(f"Date range: {tlt_data.index.min()} to {tlt_data.index.max()}")
        
        return tlt_data
        
    except Exception as e:
        print(f"Error fetching TLT data: {e}")
        return None

def create_safe_tlt_ticker():
    """Create a complete TLT ticker combining historical CSV data with live Yahoo Finance data"""
    try:
        print("Creating complete TLT ticker...")
        
        # Load historical data
        historical_data = load_complete_tlt_data()
        if historical_data is None:
            print("Failed to load historical data")
            return None
        
        # Get recent TLT data
        recent_data = get_recent_tlt_data_safe()
        if recent_data is None:
            print("Failed to fetch recent data, using historical data only")
            return historical_data
        
        # TLT IPO date (July 30, 2002)
        tlt_start_date = pd.Timestamp('2002-07-30')
        
        # Split historical data
        historical_before_2002 = historical_data[historical_data.index < tlt_start_date]
        historical_on_2002 = historical_data[historical_data.index.date == tlt_start_date.date()]
        historical_after_2002 = historical_data[historical_data.index > tlt_start_date]
        
        print(f"Historical data before 2002: {len(historical_before_2002)} records")
        print(f"Historical data on 2002: {len(historical_on_2002)} records")
        print(f"Historical data after 2002: {len(historical_after_2002)} records")
        
        # Get the last historical price before TLT IPO
        if not historical_before_2002.empty:
            last_historical_date = historical_before_2002.index[-1]
            last_historical_price = historical_before_2002['Close'].iloc[-1]
            print(f"Last historical price before TLT IPO: ${last_historical_price:.2f} on {last_historical_date.date()}")
        else:
            print("No historical data before TLT IPO")
            return recent_data
        
        # Get the first TLT price on IPO date
        first_tlt_data = recent_data[recent_data.index.date == tlt_start_date.date()]
        if first_tlt_data.empty:
            # If no data on exact IPO date, get the first available data
            first_tlt_data = recent_data.head(1)
            first_tlt_date = recent_data.index[0]
        else:
            first_tlt_date = first_tlt_data.index[0]
        
        first_tlt_price = first_tlt_data['Close'].iloc[0]
        print(f"First TLT price on IPO: ${first_tlt_price:.2f} on {first_tlt_date.date()}")
        
        # Calculate scaling factor to match prices
        scaling_factor = last_historical_price / first_tlt_price
        print(f"Scaling factor: {scaling_factor:.6f}")
        
        # Scale the recent data
        recent_data_scaled = recent_data.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in recent_data_scaled.columns:
                recent_data_scaled[col] = recent_data_scaled[col] * scaling_factor
        
        first_tlt_price_scaled = first_tlt_price * scaling_factor
        print(f"First scaled TLT price: ${first_tlt_price_scaled:.2f}")
        print(f"Price difference: ${abs(first_tlt_price_scaled - last_historical_price):.2f}")
        
        # Combine historical data (before 2002) with scaled recent data
        combined_data = pd.concat([historical_before_2002, recent_data_scaled])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data = combined_data.sort_index()
        
        # Fill any gaps with forward fill
        combined_data = combined_data.reindex(
            pd.date_range(combined_data.index.min(), combined_data.index.max(), freq='D')
        ).fillna(method='ffill')
        
        print(f"Combined data: {len(combined_data)} records")
        print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        
        # Verify transition day continuity
        if not historical_before_2002.empty and not recent_data_scaled.empty:
            last_hist_price = historical_before_2002['Close'].iloc[-1]
            first_tlt_scaled = recent_data_scaled['Close'].iloc[0]
            print(f"Transition verification:")
            print(f"  Last historical: ${last_hist_price:.2f}")
            print(f"  First TLT scaled: ${first_tlt_scaled:.2f}")
            print(f"  Difference: ${abs(last_hist_price - first_tlt_scaled):.2f}")
        
        return combined_data
        
    except Exception as e:
        print(f"Error creating complete TLT ticker: {e}")
        return None

def plot_tlt_data(data):
    """Plot the TLT data to visualize the transition"""
    try:
        if data is None or data.empty:
            print("No data to plot")
            return
        
        plt.figure(figsize=(15, 8))
        
        # TLT IPO date
        tlt_start_date = pd.Timestamp('2002-07-30')
        
        # Split data for different colors
        historical_data = data[data.index < tlt_start_date]
        tlt_data = data[data.index >= tlt_start_date]
        
        # Plot historical data in gold color
        if not historical_data.empty:
            plt.plot(historical_data.index, historical_data['Close'], 
                   color='gold', linewidth=2, label='Historical TLT Data (Monthly)')
        
        # Plot TLT ETF data in blue
        if not tlt_data.empty:
            plt.plot(tlt_data.index, tlt_data['Close'], 
                   color='blue', linewidth=2, label='TLT ETF Data (Daily)')
        
        # Add vertical line at transition
        plt.axvline(x=tlt_start_date, color='red', linestyle='--', 
                   linewidth=2, label='TLT ETF IPO (2002-07-30)')
        
        plt.title('Complete TLT Historical Data (1962-2025)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("TLT data plot displayed successfully!")
        
    except Exception as e:
        print(f"Error plotting TLT data: {e}")

def test_safe_tlt_ticker():
    """Test the complete TLT ticker"""
    print("=" * 60)
    print("TESTING COMPLETE TLT TICKER")
    print("=" * 60)
    
    # Create the ticker
    tlt_data = create_safe_tlt_ticker()
    
    if tlt_data is not None:
        print(f"\nSUCCESS: TLT ticker created successfully!")
        print(f"Total records: {len(tlt_data)}")
        print(f"Date range: {tlt_data.index.min()} to {tlt_data.index.max()}")
        print(f"Latest price: ${tlt_data['Close'].iloc[-1]:.2f}")
        
        # Show sample data
        print(f"\nSample data (first 3 rows):")
        print(tlt_data.head(3))
        print(f"\nSample data (last 3 rows):")
        print(tlt_data.tail(3))
        
        # Plot the data
        plot_tlt_data(tlt_data)
        
    else:
        print("ERROR: Failed to create TLT ticker")

if __name__ == "__main__":
    test_safe_tlt_ticker()
