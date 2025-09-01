# Quantitative Portfolio Momentum Backtest & Analytics

A comprehensive Streamlit application for backtesting momentum-based investment strategies with advanced portfolio analysis tools.

## Features

- **Multi-Portfolio Backtesting**: Compare multiple investment strategies simultaneously
- **Advanced Momentum Strategies**: Classic, relative, rank-based, and volatility-adjusted momentum
- **Real-time Data**: Uses Yahoo Finance for live market data
- **Comprehensive Analytics**: Sharpe ratio, Sortino ratio, maximum drawdown, and more
- **Interactive Visualizations**: Plotly charts with detailed performance metrics
- **Allocation Tracking**: Monitor and analyze portfolio allocations over time

## Setup Instructions

### 1. Virtual Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# Method 1: Direct Streamlit command
streamlit run Backtest_Engine.py

# Method 2: Using the startup script
python run_app.py
```

The application will be available at: http://localhost:8501

## Application Structure

- `Backtest_Engine.py` - Main application with single portfolio backtesting
- `pages/1_Multi_Backtest.py` - Multi-portfolio comparison tool
- `pages/2_Allocations.py` - Portfolio allocation analysis and tracking
- `pages/3_Strategy_Comparison.py` - Advanced strategy comparison and analysis
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration

## Usage

### Single Portfolio Backtesting
1. Navigate to the main page
2. Configure your portfolio with tickers and allocations
3. Set momentum strategy parameters
4. Run backtest and analyze results

### Multi-Portfolio Comparison
1. Go to "Multi-Backtest" page
2. Create multiple portfolio configurations
3. Compare performance across different strategies
4. Analyze relative performance metrics

### Allocation Tracking
1. Visit "Allocations" page
2. Set up your current portfolio
3. Track allocation changes over time
4. Monitor rebalancing needs

## Key Features

### Momentum Strategies
- **Classic Momentum**: Traditional price-based momentum
- **Relative Momentum**: Rank-based momentum scoring
- **Volatility-Adjusted**: Momentum adjusted for risk
- **Dual Momentum**: Combines absolute and relative momentum

### Risk Metrics
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Ulcer Index
- Beta calculation
- Volatility analysis

### Data Sources
- Yahoo Finance (yfinance)
- Real-time market data
- Historical price data
- Dividend information

## Deployment

### Local Deployment
The application is ready for local deployment with the provided configuration.

### Cloud Deployment
For cloud deployment (Streamlit Cloud, Heroku, etc.):
1. Ensure `requirements.txt` is in the root directory
2. Set up the `.streamlit/config.toml` for production settings
3. Deploy using your preferred platform

## Troubleshooting

### Common Issues
1. **Virtual Environment**: Always activate the virtual environment before running
2. **Port Conflicts**: Change the port in `.streamlit/config.toml` if 8501 is busy
3. **Data Loading**: Ensure internet connection for Yahoo Finance data

### Performance Tips
- Use appropriate date ranges for faster backtesting
- Limit the number of assets for quicker calculations
- Consider using cached data for repeated backtests

## Contributing

Feel free to contribute to this project by:
- Adding new momentum strategies
- Improving visualization features
- Enhancing performance metrics
- Bug fixes and optimizations

## License

This project is open source and available under the MIT License.

