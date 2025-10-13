# NO CACHE VERSION - ALL @st.cache_data decorators removed - ZERO CACHE ANYWHERE
import streamlit as st
import datetime
from datetime import timedelta, time
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
import contextlib
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors as reportlab_colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import base64
import os
import signal
import sys
import threading
warnings.filterwarnings('ignore')

# Handle rerun flag for smooth UI updates - must be at the very top
if st.session_state.get('alloc_rerun_flag', False):
    st.session_state.alloc_rerun_flag = False
    st.rerun()

# =============================================================================
# HARD KILL FUNCTIONS
# =============================================================================
def hard_kill_process():
    """Completely kill the current process and all background threads"""
    try:
        # Kill all background threads
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join(timeout=0.1)

        # Force garbage collection
        import gc
        gc.collect()

        # On Windows, use os._exit for immediate termination
        if os.name == 'nt':
            os._exit(1)
        else:
            # On Unix-like systems, use os.kill
            os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        # Last resort - force exit
        os._exit(1)

def check_kill_request():
    """Check if user has requested a hard kill"""
    if st.session_state.get('hard_kill_requested', False):
        st.error("🛑 **HARD KILL REQUESTED** - Terminating all processes...")
        st.stop()

def emergency_kill():
    """Emergency kill function that stops backtest without crashing the app"""
    st.error("🛑 **EMERGENCY KILL** - Forcing immediate backtest termination...")
    st.session_state.hard_kill_requested = True
    st.rerun()

# =============================================================================
# TICKER ALIASES FUNCTIONS
# =============================================================================

def get_ticker_aliases():
    """Define ticker aliases for easier entry"""
    return {
        # Stock Market Indices
        'SPX': '^GSPC',           # S&P 500 (price only, no dividends) - 1927+
        'SPXTR': '^SP500TR',      # S&P 500 Total Return (with dividends) - 1988+
        'SP500': '^GSPC',         # S&P 500 (price only, no dividends) - 1927+
        'SP500TR': '^SP500TR',    # S&P 500 Total Return (with dividends) - 1988+
        'SPYTR': '^SP500TR',      # S&P 500 Total Return (with dividends) - 1988+
        'NASDAQ': '^IXIC',        # NASDAQ Composite (price only, no dividends) - 1971+
        'NDX': '^NDX',           # NASDAQ 100 (price only, no dividends) - 1985+
        'QQQTR': '^IXIC',        # NASDAQ Composite (price only, no dividends) - 1971+
        'DOW': '^DJI',           # Dow Jones Industrial Average (price only, no dividends) - 1992+
        
        # Treasury Yield Indices (LONGEST HISTORY - 1960s+)
        'TNX': '^TNX',           # 10-Year Treasury Yield (1962+) - Price only, no coupons
        'TYX': '^TYX',           # 30-Year Treasury Yield (1977+) - Price only, no coupons
        'FVX': '^FVX',           # 5-Year Treasury Yield (1962+) - Price only, no coupons
        'IRX': '^IRX',           # 3-Month Treasury Yield (1960+) - Price only, no coupons
        
        # Treasury Bond ETFs (MODERN - WITH COUPONS/DIVIDENDS)
        'TLTETF': 'TLT',          # 20+ Year Treasury Bond ETF (2002+) - With coupons
        'IEFETF': 'IEF',          # 7-10 Year Treasury Bond ETF (2002+) - With coupons
        'SHY': 'SHY',            # 1-3 Year Treasury Bond ETF (2002+) - With coupons
        'BIL': 'BIL',            # 1-3 Month T-Bill ETF (2007+) - With coupons
        'GOVT': 'GOVT',          # US Treasury Bond ETF (2012+) - With coupons
        'SPTL': 'SPTL',          # Long Term Treasury ETF (2007+) - With coupons
        'SPTS': 'SPTS',          # Short Term Treasury ETF (2011+) - With coupons
        'SPTI': 'SPTI',          # Intermediate Term Treasury ETF (2007+) - With coupons
        
        # Cash/Zero Return
        'ZEROX': 'ZEROX',        # Zero-cost portfolio (literally cash doing nothing)
        
        # Gold & Commodities
        'GOLDX': 'GOLDX',        # Fidelity Gold Fund (1994+) - With dividends
        'GLD': 'GLD',            # SPDR Gold Trust ETF (2004+) - With dividends
        'IAU': 'IAU',            # iShares Gold Trust ETF (2005+) - With dividends
        'GOLDF': 'GC=F',         # Gold Futures (2000+) - No dividends
        'SILVER': 'SI=F',        # Silver Futures (2000+) - No dividends
        'OIL': 'CL=F',           # Crude Oil Futures (2000+) - No dividends
        'NATGAS': 'NG=F',        # Natural Gas Futures (2000+) - No dividends
        'CORN': 'ZC=F',          # Corn Futures (2000+) - No dividends
        'SOYBEAN': 'ZS=F',       # Soybean Futures (2000+) - No dividends
        'COFFEE': 'KC=F',        # Coffee Futures (2000+) - No dividends
        'SUGAR': 'SB=F',         # Sugar Futures (2000+) - No dividends
        'COTTON': 'CT=F',        # Cotton Futures (2000+) - No dividends
        'COPPER': 'HG=F',        # Copper Futures (2000+) - No dividends
        'PLATINUM': 'PL=F',      # Platinum Futures (1997+) - No dividends
        'PALLADIUM': 'PA=F',     # Palladium Futures (1998+) - No dividends
        
        # Cryptocurrency
        'BITCOIN': 'BTC-USD',    # Bitcoin (2014+) - No dividends
        
        # Leveraged & Inverse ETFs (Synthetic Aliases)
        'TQQQTR': '^IXIC?L=3?E=0.95',    # 3x NASDAQ Composite (price only) - 1971+
        'TQQQND': '^NDX?L=3?E=0.95',     # 3x NASDAQ-100 (price only) - 1985+
        'SPXLTR': '^SP500TR?L=3?E=1.00', # 3x S&P 500 (with dividends)
        'UPROTR': '^SP500TR?L=3?E=0.91', # 3x S&P 500 (with dividends)
        'QLDTR': '^IXIC?L=2?E=0.95',     # 2x NASDAQ Composite (price only) - 1971+
        'SSOTR': '^SP500TR?L=2?E=0.91',  # 2x S&P 500 (with dividends)
        'SHTR': '^GSPC?L=-1?E=0.89',     # -1x S&P 500 (price only, no dividends) - 1927+
        'PSQTR': '^IXIC?L=-1?E=0.95',    # -1x NASDAQ Composite (price only, no dividends) - 1971+
        'SDSTR': '^GSPC?L=-2?E=0.91',    # -2x S&P 500 (price only, no dividends) - 1927+
        'QIDTR': '^IXIC?L=-2?E=0.95',    # -2x NASDAQ Composite (price only, no dividends) - 1971+
        'SPXUTR': '^GSPC?L=-3?E=1.00',   # -3x S&P 500 (price only, no dividends) - 1927+
        'SQQQTR': '^IXIC?L=-3?E=0.95',   # -3x NASDAQ Composite (price only, no dividends) - 1971+
        
        # Synthetic Complete Tickers
        'SPYSIM': 'SPYSIM_COMPLETE',  # Complete S&P 500 Simulation (1885+) - Historical + SPYTR
        'GOLDSIM': 'GOLDSIM_COMPLETE',  # Complete Gold Simulation (1968+) - New Historical + GOLDX
        'GOLDX': 'GOLD_COMPLETE',  # Complete Gold Dataset (1975+) - Historical + GLD
        'ZROZX': 'ZROZ_COMPLETE',  # Complete ZROZ Dataset (1962+) - Historical + ZROZ
        'TLTTR': 'TLT_COMPLETE',  # Complete TLT Dataset (1962+) - Historical + TLT
        'BITCOINX': 'BTC_COMPLETE',  # Complete Bitcoin Dataset (2010+) - Historical + BTC-USD
        'KMLMX': 'KMLM_COMPLETE',  # Complete KMLM Dataset (1992+) - Historical + KMLM
        'IEFTR': 'IEF_COMPLETE',  # Complete IEF Dataset (1962+) - Historical + IEF
        'DBMFX': 'DBMF_COMPLETE',  # Complete DBMF Dataset (2000+) - Historical + DBMF
        'TBILL': 'TBILL_COMPLETE',  # Complete TBILL Dataset (1948+) - Historical + SGOV
        
        # Canadian Ticker Mappings (USD OTC -> Canadian Exchange)
        'MDALF': 'MDA.TO',          # MDA Ltd - USD OTC -> Canadian TSX
        'KRKNF': 'PNG.V',           # Kraken Robotics - USD OTC -> Canadian Venture
        'CNSWF': 'CSU.TO',          # Constellation Software - USD OTC -> Canadian TSX
        'TOITF': 'TOI.V',           # Topicus - USD OTC -> Canadian Venture
        'LMGIF': 'LMN.V',           # Lumine Group - USD OTC -> Canadian Venture
        'DLMAF': 'DOL.TO',          # Dollarama - USD OTC -> Canadian TSX
        'FRFHF': 'FFH.TO',          # Fairfax Financial - USD OTC -> Canadian TSX
    }

def get_leveraged_ticker_underlying():
    """Map leveraged tickers to their underlying tickers for valuation
    
    This mapping is used ONLY for valuation tables (P/E, Market Cap, etc.)
    Backtests still use the leveraged ticker for accurate price/return data.
    """
    return {
        # Berkshire Hathaway
        'BRKU': 'BRK-B',          # 2x Berkshire Hathaway
        
        # Booking
        'BKNU': 'BKNG',           # 2x Booking
        
        # UnitedHealth
        'UNHG': 'UNH',            # 2x UnitedHealth
        
        # Microsoft
        'MSFU': 'MSFT',           # 2x Microsoft
        'MSFL': 'MSFT',           # 1.5x Microsoft
        
        # Meta
        'METU': 'META',           # 2x Meta
        'FBL': 'META',            # 1.5x Meta
        
        # Google
        'GGLL': 'GOOGL',          # 2x Google
        'ALPU': 'GOOGL',          # 1.5x Google
        
        # Taiwan Semiconductor
        'TSMG': 'TSM',            # 2x Taiwan Semi
        'TSMU': 'TSM',            # 1.5x Taiwan Semi
        'TSMX': 'TSM',            # 3x Taiwan Semi
        
        # ASML
        'ASMG': 'ASML',           # 2x ASML
        
        # Amazon
        'AMZU': 'AMZN',           # 2x Amazon
        
        # Oracle
        'ORCX': 'ORCL',           # 2x Oracle
        
        # Broadcom
        'AVGU': 'AVGO',           # 2x Broadcom
        'AVL': 'AVGO',            # 1.5x Broadcom
        'AVGG': 'AVGO',           # 3x Broadcom
        'AVGX': 'AVGO',           # 4x Broadcom
        
        # NVIDIA
        'NVDL': 'NVDA',           # 2x NVIDIA
        'NVDU': 'NVDA',           # 3x NVIDIA
        'NVDG': 'NVDA',           # 4x NVIDIA
        'NVD': 'NVDA',            # Alternative NVIDIA leveraged
        
        # Netflix
        'NFXL': 'NFLX',           # 2x Netflix
        'NFLU': 'NFLX',           # 1.5x Netflix
        
        # Arista Networks
        'ANEL': 'ANET',           # 2x Arista
        
        # Super Micro Computer
        'SMCX': 'SMCI',           # 2x Super Micro
        'SMCL': 'SMCI',           # 1.5x Super Micro
        
        # Apple
        'AAPB': 'AAPL',           # 2x Apple
        'AAPU': 'AAPL',           # 1.5x Apple
    }

def resolve_ticker_alias(ticker):
    """Resolve ticker alias to actual ticker symbol"""
    aliases = get_ticker_aliases()
    
    # Extract base ticker before any parameters (e.g., CNSWF?L=2?E=1 -> CNSWF)
    base_ticker = ticker.split('?')[0].upper()
    
    # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
    if base_ticker == 'BRK.B':
        base_ticker = 'BRK-B'
    elif base_ticker == 'BRK.A':
        base_ticker = 'BRK-A'
    
    # Get the Canadian ticker if available
    resolved_base = aliases.get(base_ticker, base_ticker)
    
    # If we have parameters, add them back to the resolved ticker
    if '?' in ticker:
        parameters = ticker.split('?', 1)[1]  # Get everything after the first ?
        return f"{resolved_base}?{parameters}"
    else:
        return resolved_base

def resolve_index_to_etf_for_stats(ticker):
    """Convert indices to equivalent ETFs for better statistics data
    
    This function converts raw indices (like ^SP500-45) to their equivalent ETFs (like XLK)
    for statistics calculations, as ETFs have more comprehensive market data.
    """
    # Extract base ticker before any parameters (e.g., ^IXIC?L=3?E=0.95 -> ^IXIC)
    base_ticker = ticker.split('?')[0].upper()
    
    # Mapping from indices to equivalent ETFs
    index_to_etf_mapping = {
        # S&P 500 Sector Indices -> Sector ETFs
        '^SP500-45': 'XLK',    # Technology
        '^SP500-35': 'XLV',    # Healthcare
        '^SP500-30': 'XLP',    # Consumer Staples
        '^SP500-40': 'XLF',    # Financials
        '^SP500-10': 'XLE',    # Energy
        '^SP500-20': 'XLI',    # Industrials
        '^SP500-25': 'XLY',    # Consumer Discretionary
        '^SP500-15': 'XLB',    # Materials
        '^SP500-55': 'XLU',    # Utilities
        '^SP500-60': 'XLRE',   # Real Estate
        '^SP500-50': 'XLC',    # Communication Services
        
        # Major Indices -> Major ETFs
        '^IXIC': 'QQQ',        # NASDAQ Composite -> NASDAQ-100 ETF
        '^NDX': 'QQQ',         # NASDAQ-100 -> NASDAQ-100 ETF
        '^GSPC': 'SPY',        # S&P 500 -> S&P 500 ETF
        '^SP500TR': 'SPY',     # S&P 500 Total Return -> S&P 500 ETF
        '^DJI': 'DIA',         # Dow Jones -> Dow Jones ETF
    }
    
    # Check if this is an index that should be converted to ETF
    if base_ticker in index_to_etf_mapping:
        etf_ticker = index_to_etf_mapping[base_ticker]
        
        # If we have parameters (like leverage), add them back to the ETF
        if '?' in ticker:
            parameters = ticker.split('?', 1)[1]  # Get everything after the first ?
            return f"{etf_ticker}?{parameters}"
        else:
            return etf_ticker
    
    # If not an index or no mapping found, return original ticker
    return ticker

def get_custom_sector_for_ticker(ticker):
    """Get custom sector for ETFs and special tickers that don't have traditional sectors
    
    Returns custom sector name for special tickers, or None to use Yahoo Finance sector.
    """
    # Check if ticker has leverage parameter (?L=)
    has_leverage = '?L=' in ticker.upper()
    
    # Extract base ticker (remove leverage parameters)
    base_ticker = ticker.split('?')[0].upper()
    
    # List of tickers that are indices/ETFs (not individual stocks)
    index_etf_tickers = {
        'SPY', 'VOO', 'IVV', 'VTI', 'ITOT', 'SCHB', 'VT', 'VXUS', 'IXUS', 'QQQ', 'QQQM', 'DIA',
        '^GSPC', '^SP500TR', '^IXIC', '^NDX', '^DJI', 'SPYSIM', 'SPYSIM_COMPLETE',
        'MTUM', 'SPMO', 'VFMO', 'QMOM', 'IMOM', 'JMOM', 'SEIM', 'FDMO', 'FPMO', 'IWMO', 'UMMT', 'LRGF', 'MOM', 'USMC', 'PDP', 'DWAQ', 'DWAS',
        'VIG', 'SCHD', 'DGRO', 'HDV', 'SPYD', 'DVY', 'SDY', 'VYM', 'RDVY', 'FDL', 'FVD', 'NOBL', 'DHS', 'FDVV', 'DGRW', 'DIVO', 'LVHD', 'SPHD', 'OUSA', 'PID', 'PEY', 'DON', 'FQAL', 'RDIV', 'VYMI', 'IDV', 'DGT', 'DWX',
        'VUG', 'MGK', 'IWF', 'RPG', 'SCHG', 'QGRO', 'JKE', 'IVW', 'TGRW', 'SPYG', 'SYG', 'GFG', 'GXG', 'GGRO', 'XLG',
        'SSO', 'QLD', 'SPXL', 'UPRO', 'TQQQ', 'TMF', 'SOXL', 'TNA', 'CURE', 'FAS', 'LABU', 'TECL',
        'TQQQTR', 'SPXLTR', 'UPROTR', 'QLDTR', 'SSOTR', 'SHTR', 'PSQTR', 'SDSTR', 'QIDTR', 'SPXUTR', 'SQQQTR',
        'TLT', 'IEF', 'SHY', 'BIL', 'ZROZ', 'TLH', 'IEI', 'SHV', 'VGSH', 'VGIT', 'VGLT', 'GOVT', 'SPTL', 'SPTS', 'SPTI',
        'TLTTR', 'TLT_COMPLETE', 'ZROZX', 'ZROZ_COMPLETE', 'IEFTR', 'IEF_COMPLETE', 'TBILL', 'TBILL_COMPLETE',
        '^TNX', '^TYX', '^FVX', '^IRX', 'AGG', 'BND', 'BNDX', 'LQD', 'HYG', 'JNK', 'MUB', 'TIP', 'VTIP',
        'GLD', 'GOLDX', 'GOLDSIM', 'IAU', 'IAUM', 'GLDM', 'SGOL', 'GOLD_COMPLETE', 'GOLDSIM_COMPLETE', 'GC=F', 'GOLDF',
        'GDX', 'GDXJ', 'SLV', 'SI=F', 'SILVER', 'PPLT', 'PL=F', 'PLATINUM', 'PALL', 'PA=F', 'PALLADIUM',
        'USO', 'CL=F', 'OIL', 'UNG', 'NG=F', 'NATGAS', 'DBA', 'ZC=F', 'CORN', 'ZS=F', 'SOYBEAN', 'KC=F', 'COFFEE', 'SB=F', 'SUGAR', 'CT=F', 'COTTON',
        'DBC', 'GSG', 'PDBC', 'BCI', 'HG=F', 'COPPER', 'KMLM', 'DBMF', 'KMLMX', 'KMLM_COMPLETE', 'DBMFX', 'DBMF_COMPLETE',
        'BTC-USD', 'ETH-USD', 'BITO', 'GBTC', 'ETHE', 'BITCOINX', 'BTC_COMPLETE',
        'XLK', 'VGT', 'FTEC', 'IGM', 'SMH', 'SOXX', 'SOXS', 'USD',
        '^SP500-45', '^SP500-35', '^SP500-30', '^SP500-40', '^SP500-10', '^SP500-20', '^SP500-25', '^SP500-15', '^SP500-55', '^SP500-60', '^SP500-50',
        'XLF', 'XLV', 'XLP', 'XLE', 'XLI', 'XLY', 'XLB', 'XLU', 'XLC',
        'VNQ', 'IYR', 'SCHH', 'RWR', 'XLRE',
        'VXX', 'UVXY', 'SVXY',
        'CASH', 'SGOV', 'USFR', 'ZEROX'
    }
    
    # If ticker has leverage parameter AND is in our index/ETF list, return appropriate leveraged sector
    if has_leverage and base_ticker in index_etf_tickers:
        # Check for gold/precious metals tickers
        if base_ticker in ['GLD', 'GOLDX', 'GOLDSIM', 'IAU', 'IAUM', 'GLDM', 'SGOL', 'GOLD_COMPLETE', 'GOLDSIM_COMPLETE', 'GC=F', 'GOLDF', 'GDX', 'GDXJ', 'SLV', 'SI=F', 'SILVER', 'PPLT', 'PL=F', 'PLATINUM', 'PALL', 'PA=F', 'PALLADIUM']:
            return 'LEVERAGED PRECIOUS METALS'
        
        # Check for treasury/bond tickers
        if base_ticker in ['TLT', 'ZROZ', 'VGLT', 'SPTL', 'TLH', 'TLTTR', 'TLT_COMPLETE', 'ZROZX', 'ZROZ_COMPLETE', 'IEF', 'VGIT', 'SPTI', 'IEI', 'IEFTR', 'IEF_COMPLETE', 'SHY', 'BIL', 'VGSH', 'SPTS', 'SHV', 'TBILL', 'TBILL_COMPLETE', '^TNX', '^TYX', '^FVX', '^IRX', 'AGG', 'BND', 'BNDX', 'LQD', 'HYG', 'JNK', 'MUB', 'TIP', 'VTIP', 'GOVT']:
            return 'LEVERAGED TREASURIES'
        
        # Check for cryptocurrency
        if base_ticker in ['BTC-USD', 'ETH-USD', 'BITO', 'GBTC', 'ETHE', 'BITCOINX', 'BTC_COMPLETE']:
            return 'LEVERAGED CRYPTOCURRENCY'
        
        # Check for commodities (excluding precious metals)
        if base_ticker in ['USO', 'CL=F', 'OIL', 'UNG', 'NG=F', 'NATGAS', 'DBA', 'ZC=F', 'CORN', 'ZS=F', 'SOYBEAN', 'KC=F', 'COFFEE', 'SB=F', 'SUGAR', 'CT=F', 'COTTON', 'DBC', 'GSG', 'PDBC', 'BCI', 'HG=F', 'COPPER', 'KMLM', 'DBMF', 'KMLMX', 'KMLM_COMPLETE', 'DBMFX', 'DBMF_COMPLETE']:
            return 'LEVERAGED COMMODITIES'
        
        # Check for real estate
        if base_ticker in ['VNQ', 'IYR', 'SCHH', 'RWR', 'XLRE']:
            return 'LEVERAGED REAL ESTATE'
        
        # Check for volatility
        if base_ticker in ['VXX', 'UVXY', 'SVXY']:
            return 'LEVERAGED VOLATILITY'
        
        # Default to LEVERAGED ETF for all other index/ETF tickers
        return 'LEVERAGED ETF'
    
    # Custom sector mappings for ETFs and special tickers
    custom_sectors = {
        # Broad Market Index ETFs
        'SPY': 'INDEX FUND',
        'VOO': 'INDEX FUND',
        'IVV': 'INDEX FUND',
        'VTI': 'INDEX FUND',
        'ITOT': 'INDEX FUND',
        'SCHB': 'INDEX FUND',
        'VT': 'INDEX FUND',
        'VXUS': 'INDEX FUND',
        'IXUS': 'INDEX FUND',
        'QQQ': 'INDEX FUND',
        'QQQM': 'INDEX FUND',
        'DIA': 'INDEX FUND',
        '^GSPC': 'INDEX FUND',
        '^SP500TR': 'INDEX FUND',
        '^IXIC': 'INDEX FUND',
        '^NDX': 'INDEX FUND',
        '^DJI': 'INDEX FUND',
        'SPYSIM': 'INDEX FUND',
        'SPYSIM_COMPLETE': 'INDEX FUND',
        
        # Momentum Index ETFs
        'MTUM': 'INDEX FUND',
        'SPMO': 'INDEX FUND',
        'VFMO': 'INDEX FUND',
        'QMOM': 'INDEX FUND',
        'IMOM': 'INDEX FUND',
        'JMOM': 'INDEX FUND',
        'SEIM': 'INDEX FUND',
        'FDMO': 'INDEX FUND',
        'FPMO': 'INDEX FUND',
        'IWMO': 'INDEX FUND',
        'UMMT': 'INDEX FUND',
        'LRGF': 'INDEX FUND',
        'MOM': 'INDEX FUND',
        'USMC': 'INDEX FUND',
        'PDP': 'INDEX FUND',
        'DWAQ': 'INDEX FUND',
        'DWAS': 'INDEX FUND',
        
        # Dividend Index ETFs
        'VIG': 'INDEX FUND',
        'SCHD': 'INDEX FUND',
        'DGRO': 'INDEX FUND',
        'HDV': 'INDEX FUND',
        'SPYD': 'INDEX FUND',
        'DVY': 'INDEX FUND',
        'SDY': 'INDEX FUND',
        'VYM': 'INDEX FUND',
        'RDVY': 'INDEX FUND',
        'FDL': 'INDEX FUND',
        'FVD': 'INDEX FUND',
        'NOBL': 'INDEX FUND',
        'DHS': 'INDEX FUND',
        'FDVV': 'INDEX FUND',
        'DGRW': 'INDEX FUND',
        'DIVO': 'INDEX FUND',
        'LVHD': 'INDEX FUND',
        'SPHD': 'INDEX FUND',
        'OUSA': 'INDEX FUND',
        'PID': 'INDEX FUND',
        'PEY': 'INDEX FUND',
        'DON': 'INDEX FUND',
        'FQAL': 'INDEX FUND',
        'RDIV': 'INDEX FUND',
        'VYMI': 'INDEX FUND',
        'IDV': 'INDEX FUND',
        'DGT': 'INDEX FUND',
        'DWX': 'INDEX FUND',
        
        # Growth Index ETFs
        'VUG': 'INDEX FUND',
        'MGK': 'INDEX FUND',
        'IWF': 'INDEX FUND',
        'RPG': 'INDEX FUND',
        'SCHG': 'INDEX FUND',
        'QGRO': 'INDEX FUND',
        'JKE': 'INDEX FUND',
        'IVW': 'INDEX FUND',
        'TGRW': 'INDEX FUND',
        'SPYG': 'INDEX FUND',
        'SYG': 'INDEX FUND',
        'GFG': 'INDEX FUND',
        'GXG': 'INDEX FUND',
        'GGRO': 'INDEX FUND',
        'XLG': 'INDEX FUND',
        
        # Leveraged Index ETFs
        'SSO': 'LEVERAGED ETF',
        'QLD': 'LEVERAGED ETF',
        'SPXL': 'LEVERAGED ETF',
        'UPRO': 'LEVERAGED ETF',
        'TQQQ': 'LEVERAGED ETF',
        'TMF': 'LEVERAGED ETF',
        'SOXL': 'LEVERAGED ETF',
        'TNA': 'LEVERAGED ETF',
        'CURE': 'LEVERAGED ETF',
        'FAS': 'LEVERAGED ETF',
        'LABU': 'LEVERAGED ETF',
        'TECL': 'LEVERAGED ETF',
        'TQQQTR': 'LEVERAGED ETF',
        'SPXLTR': 'LEVERAGED ETF',
        'UPROTR': 'LEVERAGED ETF',
        'QLDTR': 'LEVERAGED ETF',
        'SSOTR': 'LEVERAGED ETF',
        'SHTR': 'LEVERAGED ETF',
        'PSQTR': 'LEVERAGED ETF',
        'SDSTR': 'LEVERAGED ETF',
        'QIDTR': 'LEVERAGED ETF',
        'SPXUTR': 'LEVERAGED ETF',
        'SQQQTR': 'LEVERAGED ETF',
        
        # Treasury/Bond ETFs
        'TLT': 'TREASURIES',
        'IEF': 'TREASURIES',
        'SHY': 'TREASURIES',
        'BIL': 'TREASURIES',
        'ZROZ': 'TREASURIES',
        'TLH': 'TREASURIES',
        'IEI': 'TREASURIES',
        'SHV': 'TREASURIES',
        'VGSH': 'TREASURIES',
        'VGIT': 'TREASURIES',
        'VGLT': 'TREASURIES',
        'GOVT': 'TREASURIES',
        'SPTL': 'TREASURIES',
        'SPTS': 'TREASURIES',
        'SPTI': 'TREASURIES',
        'TLTTR': 'TREASURIES',
        'TLT_COMPLETE': 'TREASURIES',
        'ZROZX': 'TREASURIES',
        'ZROZ_COMPLETE': 'TREASURIES',
        'IEFTR': 'TREASURIES',
        'IEF_COMPLETE': 'TREASURIES',
        'TBILL': 'TREASURIES',
        'TBILL_COMPLETE': 'TREASURIES',
        '^TNX': 'TREASURIES',
        '^TYX': 'TREASURIES',
        '^FVX': 'TREASURIES',
        '^IRX': 'TREASURIES',
        'AGG': 'BONDS',
        'BND': 'BONDS',
        'BNDX': 'BONDS',
        'LQD': 'BONDS',
        'HYG': 'BONDS',
        'JNK': 'BONDS',
        'MUB': 'BONDS',
        'TIP': 'BONDS',
        'VTIP': 'BONDS',
        
        # Gold/Precious Metals
        'GLD': 'GOLD',
        'GOLDX': 'GOLD',
        'GOLDSIM': 'GOLD',
        'IAU': 'GOLD',
        'IAUM': 'GOLD',
        'GLDM': 'GOLD',
        'SGOL': 'GOLD',
        'GOLD_COMPLETE': 'GOLD',
        'GOLDSIM_COMPLETE': 'GOLD',
        'GC=F': 'GOLD',
        'GOLDF': 'GOLD',
        'GDX': 'GOLD MINERS',
        'GDXJ': 'GOLD MINERS',
        'SLV': 'SILVER',
        'SI=F': 'SILVER',
        'SILVER': 'SILVER',
        'PPLT': 'PLATINUM',
        'PL=F': 'PLATINUM',
        'PLATINUM': 'PLATINUM',
        'PALL': 'PALLADIUM',
        'PA=F': 'PALLADIUM',
        'PALLADIUM': 'PALLADIUM',
        
        # Commodities
        'USO': 'OIL',
        'CL=F': 'OIL',
        'OIL': 'OIL',
        'UNG': 'NATURAL GAS',
        'NG=F': 'NATURAL GAS',
        'NATGAS': 'NATURAL GAS',
        'DBA': 'AGRICULTURE',
        'ZC=F': 'AGRICULTURE',
        'CORN': 'AGRICULTURE',
        'ZS=F': 'AGRICULTURE',
        'SOYBEAN': 'AGRICULTURE',
        'KC=F': 'AGRICULTURE',
        'COFFEE': 'AGRICULTURE',
        'SB=F': 'AGRICULTURE',
        'SUGAR': 'AGRICULTURE',
        'CT=F': 'AGRICULTURE',
        'COTTON': 'AGRICULTURE',
        'DBC': 'COMMODITIES',
        'GSG': 'COMMODITIES',
        'PDBC': 'COMMODITIES',
        'BCI': 'COMMODITIES',
        'HG=F': 'COMMODITIES',
        'COPPER': 'COMMODITIES',
        'KMLM': 'MANAGED FUTURES',
        'DBMF': 'MANAGED FUTURES',
        'KMLMX': 'MANAGED FUTURES',
        'KMLM_COMPLETE': 'MANAGED FUTURES',
        'DBMFX': 'MANAGED FUTURES',
        'DBMF_COMPLETE': 'MANAGED FUTURES',
        
        # Cryptocurrency
        'BTC-USD': 'CRYPTOCURRENCY',
        'ETH-USD': 'CRYPTOCURRENCY',
        'BITO': 'CRYPTOCURRENCY',
        'GBTC': 'CRYPTOCURRENCY',
        'ETHE': 'CRYPTOCURRENCY',
        'BITCOINX': 'CRYPTOCURRENCY',
        'BTC_COMPLETE': 'CRYPTOCURRENCY',
        'BITCOIN': 'CRYPTOCURRENCY',
        
        # Bitcoin ETFs
        'IBIT': 'CRYPTOCURRENCY',      # iShares Bitcoin Trust (BlackRock)
        'FBTC': 'CRYPTOCURRENCY',      # Fidelity Wise Origin Bitcoin Fund
        'BITB': 'CRYPTOCURRENCY',      # Bitwise Bitcoin ETF Trust
        'ARKB': 'CRYPTOCURRENCY',      # ARK 21Shares Bitcoin ETF
        'BTCO': 'CRYPTOCURRENCY',      # Invesco Galaxy Bitcoin ETF
        'HODL': 'CRYPTOCURRENCY',      # 21Shares Crypto Basket 10 ETP
        'EZBC': 'CRYPTOCURRENCY',      # ETC Group Bitcoin ETP
        'XBTF': 'CRYPTOCURRENCY',      # ProShares Bitcoin Futures Strategy ETF
        'BTF': 'CRYPTOCURRENCY',       # Valkyrie Bitcoin Strategy ETF
        'BTCC': 'CRYPTOCURRENCY',      # Purpose Bitcoin ETF (Canada)
        
        # Technology
        'XLK': 'TECHNOLOGY INDEX',
        'VGT': 'TECHNOLOGY INDEX',
        'FTEC': 'TECHNOLOGY INDEX',
        'IGM': 'TECHNOLOGY INDEX',
        'SMH': 'TECHNOLOGY',
        'SOXX': 'TECHNOLOGY',
        'SOXS': 'LEVERAGED ETF',
        'USD': 'LEVERAGED ETF',
        
        # S&P 500 Sector Indices
        '^SP500-45': 'TECHNOLOGY INDEX',
        '^SP500-35': 'HEALTHCARE INDEX',
        '^SP500-30': 'CONSUMER STAPLES INDEX',
        '^SP500-40': 'FINANCIAL INDEX',
        '^SP500-10': 'ENERGY INDEX',
        '^SP500-20': 'INDUSTRIALS INDEX',
        '^SP500-25': 'CONSUMER DISCRETIONARY INDEX',
        '^SP500-15': 'MATERIALS INDEX',
        '^SP500-55': 'UTILITIES INDEX',
        '^SP500-60': 'REAL ESTATE INDEX',
        '^SP500-50': 'COMMUNICATION INDEX',
        'XLF': 'FINANCIAL INDEX',
        'XLV': 'HEALTHCARE INDEX',
        'XLP': 'CONSUMER STAPLES INDEX',
        'XLE': 'ENERGY INDEX',
        'XLI': 'INDUSTRIALS INDEX',
        'XLY': 'CONSUMER DISCRETIONARY INDEX',
        'XLB': 'MATERIALS INDEX',
        'XLU': 'UTILITIES INDEX',
        'XLC': 'COMMUNICATION INDEX',
        
        # Real Estate
        'VNQ': 'REAL ESTATE',
        'IYR': 'REAL ESTATE',
        'SCHH': 'REAL ESTATE',
        'RWR': 'REAL ESTATE',
        'XLRE': 'REAL ESTATE',
        
        # Volatility
        'VXX': 'VOLATILITY',
        'UVXY': 'VOLATILITY',
        'SVXY': 'VOLATILITY',
        
        # Cash Equivalents
        'CASH': 'CASH',
        'SGOV': 'CASH EQUIVALENT',
        'USFR': 'CASH EQUIVALENT',
        'ZEROX': 'CASH',
    }
    
    return custom_sectors.get(base_ticker, None)


def get_custom_industry_for_ticker(ticker):
    """Get custom industry for ETFs and special tickers that don't have traditional industries
    
    Returns custom industry name for special tickers, or None to use Yahoo Finance industry.
    """
    # Check if ticker has leverage parameter (?L=)
    has_leverage = '?L=' in ticker.upper()
    
    # Extract base ticker (remove leverage parameters)
    base_ticker = ticker.split('?')[0].upper()
    
    # List of tickers that are indices/ETFs (not individual stocks) - same as in get_custom_sector_for_ticker
    index_etf_tickers = {
        'SPY', 'VOO', 'IVV', 'VTI', 'ITOT', 'SCHB', 'VT', 'VXUS', 'IXUS', 'QQQ', 'QQQM', 'DIA',
        '^GSPC', '^SP500TR', '^IXIC', '^NDX', '^DJI', 'SPYSIM', 'SPYSIM_COMPLETE',
        'MTUM', 'SPMO', 'VFMO', 'QMOM', 'IMOM', 'JMOM', 'SEIM', 'FDMO', 'FPMO', 'IWMO', 'UMMT', 'LRGF', 'MOM', 'USMC', 'PDP', 'DWAQ', 'DWAS',
        'VIG', 'SCHD', 'DGRO', 'HDV', 'SPYD', 'DVY', 'SDY', 'VYM', 'RDVY', 'FDL', 'FVD', 'NOBL', 'DHS', 'FDVV', 'DGRW', 'DIVO', 'LVHD', 'SPHD', 'OUSA', 'PID', 'PEY', 'DON', 'FQAL', 'RDIV', 'VYMI', 'IDV', 'DGT', 'DWX',
        'VUG', 'MGK', 'IWF', 'RPG', 'SCHG', 'QGRO', 'JKE', 'IVW', 'TGRW', 'SPYG', 'SYG', 'GFG', 'GXG', 'GGRO', 'XLG',
        'SSO', 'QLD', 'SPXL', 'UPRO', 'TQQQ', 'TMF', 'SOXL', 'TNA', 'CURE', 'FAS', 'LABU', 'TECL',
        'TQQQTR', 'SPXLTR', 'UPROTR', 'QLDTR', 'SSOTR', 'SHTR', 'PSQTR', 'SDSTR', 'QIDTR', 'SPXUTR', 'SQQQTR',
        'TLT', 'IEF', 'SHY', 'BIL', 'ZROZ', 'TLH', 'IEI', 'SHV', 'VGSH', 'VGIT', 'VGLT', 'GOVT', 'SPTL', 'SPTS', 'SPTI',
        'TLTTR', 'TLT_COMPLETE', 'ZROZX', 'ZROZ_COMPLETE', 'IEFTR', 'IEF_COMPLETE', 'TBILL', 'TBILL_COMPLETE',
        '^TNX', '^TYX', '^FVX', '^IRX', 'AGG', 'BND', 'BNDX', 'LQD', 'HYG', 'JNK', 'MUB', 'TIP', 'VTIP',
        'GLD', 'GOLDX', 'GOLDSIM', 'IAU', 'IAUM', 'GLDM', 'SGOL', 'GOLD_COMPLETE', 'GOLDSIM_COMPLETE', 'GC=F', 'GOLDF',
        'GDX', 'GDXJ', 'SLV', 'SI=F', 'SILVER', 'PPLT', 'PL=F', 'PLATINUM', 'PALL', 'PA=F', 'PALLADIUM',
        'USO', 'CL=F', 'OIL', 'UNG', 'NG=F', 'NATGAS', 'DBA', 'ZC=F', 'CORN', 'ZS=F', 'SOYBEAN', 'KC=F', 'COFFEE', 'SB=F', 'SUGAR', 'CT=F', 'COTTON',
        'DBC', 'GSG', 'PDBC', 'BCI', 'HG=F', 'COPPER', 'KMLM', 'DBMF', 'KMLMX', 'KMLM_COMPLETE', 'DBMFX', 'DBMF_COMPLETE',
        'BTC-USD', 'ETH-USD', 'BITO', 'GBTC', 'ETHE', 'BITCOINX', 'BTC_COMPLETE',
        'XLK', 'VGT', 'FTEC', 'IGM', 'SMH', 'SOXX', 'SOXS', 'USD',
        '^SP500-45', '^SP500-35', '^SP500-30', '^SP500-40', '^SP500-10', '^SP500-20', '^SP500-25', '^SP500-15', '^SP500-55', '^SP500-60', '^SP500-50',
        'XLF', 'XLV', 'XLP', 'XLE', 'XLI', 'XLY', 'XLB', 'XLU', 'XLC',
        'VNQ', 'IYR', 'SCHH', 'RWR', 'XLRE',
        'VXX', 'UVXY', 'SVXY',
        'CASH', 'SGOV', 'USFR', 'ZEROX'
    }
    
    # If ticker has leverage parameter AND is in our index/ETF list, determine industry
    if has_leverage and base_ticker in index_etf_tickers:
        # Check for specific semiconductor tickers (SMH, SOXX)
        if base_ticker in ['SMH', 'SOXX']:
            return 'LEVERAGED SEMICONDUCTORS'
        
        # Check for gold/precious metals tickers
        if base_ticker in ['GLD', 'GOLDX', 'GOLDSIM', 'IAU', 'IAUM', 'GLDM', 'SGOL', 'GOLD_COMPLETE', 'GOLDSIM_COMPLETE', 'GC=F', 'GOLDF']:
            return 'LEVERAGED GOLD'
        if base_ticker in ['GDX', 'GDXJ']:
            return 'LEVERAGED GOLD MINERS'
        if base_ticker in ['SLV', 'SI=F', 'SILVER', 'PPLT', 'PL=F', 'PLATINUM', 'PALL', 'PA=F', 'PALLADIUM']:
            return 'LEVERAGED PRECIOUS METALS'
        
        # Check for treasury/bond tickers
        if base_ticker in ['TLT', 'ZROZ', 'VGLT', 'SPTL', 'TLH', 'TLTTR', 'TLT_COMPLETE', 'ZROZX', 'ZROZ_COMPLETE']:
            return 'LEVERAGED LONG TERM TREASURIES'
        if base_ticker in ['IEF', 'VGIT', 'SPTI', 'IEI', 'IEFTR', 'IEF_COMPLETE']:
            return 'LEVERAGED INTERMEDIATE TREASURIES'
        if base_ticker in ['SHY', 'BIL', 'VGSH', 'SPTS', 'SHV', 'TBILL', 'TBILL_COMPLETE']:
            return 'LEVERAGED SHORT TERM TREASURIES'
        if base_ticker in ['AGG', 'BND', 'BNDX', 'GOVT', 'LQD', 'HYG', 'JNK', 'MUB', 'TIP', 'VTIP']:
            return 'LEVERAGED BONDS'
        
        # Check for commodities
        if base_ticker in ['USO', 'CL=F', 'OIL', 'UNG', 'NG=F', 'NATGAS']:
            return 'LEVERAGED ENERGY COMMODITIES'
        if base_ticker in ['DBA', 'ZC=F', 'CORN', 'ZS=F', 'SOYBEAN', 'KC=F', 'COFFEE', 'SB=F', 'SUGAR', 'CT=F', 'COTTON']:
            return 'LEVERAGED AGRICULTURAL COMMODITIES'
        if base_ticker in ['DBC', 'GSG', 'PDBC', 'BCI', 'HG=F', 'COPPER']:
            return 'LEVERAGED BROAD COMMODITIES'
        if base_ticker in ['KMLM', 'DBMF', 'KMLMX', 'KMLM_COMPLETE', 'DBMFX', 'DBMF_COMPLETE']:
            return 'LEVERAGED MANAGED FUTURES'
        
        # Check for cryptocurrency
        if base_ticker in ['BTC-USD', 'ETH-USD', 'BITO', 'GBTC', 'ETHE', 'BITCOINX', 'BTC_COMPLETE']:
            return 'LEVERAGED DIGITAL ASSETS'
        
        # Check for real estate
        if base_ticker in ['VNQ', 'IYR', 'SCHH', 'RWR', 'XLRE']:
            return 'LEVERAGED REIT INDEX'
        
        # Check for volatility
        if base_ticker in ['VXX', 'UVXY', 'SVXY']:
            return 'LEVERAGED VOLATILITY INDEX'
        
        # Check if it's inverse leverage (negative L value)
        if '?L=-' in ticker.upper():
            # Determine specific inverse type based on base ticker
            if base_ticker in ['^IXIC', '^NDX']:
                return 'INVERSE LEVERAGED NASDAQ'
            else:
                return 'INVERSE LEVERAGED INDEX'
        else:
            # Determine specific leveraged type based on base ticker
            if base_ticker in ['^IXIC', '^NDX']:
                return 'LEVERAGED NASDAQ INDEX'
            elif base_ticker in ['^GSPC', '^SP500TR']:
                return 'LEVERAGED SP500 INDEX'
            else:
                return 'LEVERAGED INDEX FUND'
    
    # Custom industry mappings for ETFs and special tickers
    custom_industries = {
        # Broad Market Index ETFs
        'SPY': 'BROAD MARKET INDEX',
        'VOO': 'BROAD MARKET INDEX',
        'IVV': 'BROAD MARKET INDEX',
        'VTI': 'BROAD MARKET INDEX',
        'ITOT': 'BROAD MARKET INDEX',
        'SCHB': 'BROAD MARKET INDEX',
        'VT': 'BROAD MARKET INDEX',
        'VXUS': 'BROAD MARKET INDEX',
        'IXUS': 'BROAD MARKET INDEX',
        'DIA': 'BROAD MARKET INDEX',
        '^GSPC': 'BROAD MARKET INDEX',
        '^SP500TR': 'BROAD MARKET INDEX',
        '^DJI': 'BROAD MARKET INDEX',
        'SPYSIM': 'SP500 SIMULATION',
        'SPYSIM_COMPLETE': 'SP500 SIMULATION',
        
        # NASDAQ Index ETFs
        'QQQ': 'NASDAQ INDEX',
        'QQQM': 'NASDAQ INDEX',
        '^IXIC': 'NASDAQ INDEX',
        '^NDX': 'NASDAQ INDEX',
        
        # Momentum Index ETFs
        'MTUM': 'MOMENTUM INDEX',
        'SPMO': 'MOMENTUM INDEX',
        'VFMO': 'MOMENTUM INDEX',
        'QMOM': 'MOMENTUM INDEX',
        'IMOM': 'MOMENTUM INDEX',
        'JMOM': 'MOMENTUM INDEX',
        'SEIM': 'MOMENTUM INDEX',
        'FDMO': 'MOMENTUM INDEX',
        'FPMO': 'MOMENTUM INDEX',
        'IWMO': 'MOMENTUM INDEX',
        'UMMT': 'MOMENTUM INDEX',
        'LRGF': 'MOMENTUM INDEX',
        'MOM': 'MOMENTUM INDEX',
        'USMC': 'MOMENTUM INDEX',
        'PDP': 'MOMENTUM INDEX',
        'DWAQ': 'MOMENTUM INDEX',
        'DWAS': 'MOMENTUM INDEX',
        
        # Dividend Index ETFs
        'VIG': 'DIVIDEND INDEX',
        'SCHD': 'DIVIDEND INDEX',
        'DGRO': 'DIVIDEND INDEX',
        'HDV': 'DIVIDEND INDEX',
        'SPYD': 'DIVIDEND INDEX',
        'DVY': 'DIVIDEND INDEX',
        'SDY': 'DIVIDEND INDEX',
        'VYM': 'DIVIDEND INDEX',
        'RDVY': 'DIVIDEND INDEX',
        'FDL': 'DIVIDEND INDEX',
        'FVD': 'DIVIDEND INDEX',
        'NOBL': 'DIVIDEND INDEX',
        'DHS': 'DIVIDEND INDEX',
        'FDVV': 'DIVIDEND INDEX',
        'DGRW': 'DIVIDEND INDEX',
        'DIVO': 'DIVIDEND INDEX',
        'LVHD': 'DIVIDEND INDEX',
        'SPHD': 'DIVIDEND INDEX',
        'OUSA': 'DIVIDEND INDEX',
        'PID': 'DIVIDEND INDEX',
        'PEY': 'DIVIDEND INDEX',
        'DON': 'DIVIDEND INDEX',
        'FQAL': 'DIVIDEND INDEX',
        'RDIV': 'DIVIDEND INDEX',
        'VYMI': 'DIVIDEND INDEX',
        'IDV': 'DIVIDEND INDEX',
        'DGT': 'DIVIDEND INDEX',
        'DWX': 'DIVIDEND INDEX',
        
        # Growth Index ETFs
        'VUG': 'GROWTH INDEX',
        'MGK': 'GROWTH INDEX',
        'IWF': 'GROWTH INDEX',
        'RPG': 'GROWTH INDEX',
        'SCHG': 'GROWTH INDEX',
        'QGRO': 'GROWTH INDEX',
        'JKE': 'GROWTH INDEX',
        'IVW': 'GROWTH INDEX',
        'TGRW': 'GROWTH INDEX',
        'SPYG': 'GROWTH INDEX',
        'SYG': 'GROWTH INDEX',
        'GFG': 'GROWTH INDEX',
        'GXG': 'GROWTH INDEX',
        'GGRO': 'GROWTH INDEX',
        'XLG': 'GROWTH INDEX',
        
        # Leveraged Index ETFs
        'SSO': 'LEVERAGED INDEX FUND',
        'QLD': 'LEVERAGED INDEX FUND',
        'SPXL': 'LEVERAGED INDEX FUND',
        'UPRO': 'LEVERAGED INDEX FUND',
        'TQQQ': 'LEVERAGED INDEX FUND',
        'TMF': 'LEVERAGED INDEX FUND',
        'SOXL': 'LEVERAGED INDEX FUND',
        'TNA': 'LEVERAGED INDEX FUND',
        'CURE': 'LEVERAGED INDEX FUND',
        'FAS': 'LEVERAGED INDEX FUND',
        'LABU': 'LEVERAGED INDEX FUND',
        'TECL': 'LEVERAGED INDEX FUND',
        'TQQQTR': 'LEVERAGED INDEX FUND',
        'SPXLTR': 'LEVERAGED INDEX FUND',
        'UPROTR': 'LEVERAGED INDEX FUND',
        'QLDTR': 'LEVERAGED INDEX FUND',
        'SSOTR': 'LEVERAGED INDEX FUND',
        'SHTR': 'INVERSE LEVERAGED INDEX',
        'PSQTR': 'INVERSE LEVERAGED INDEX',
        'SDSTR': 'INVERSE LEVERAGED INDEX',
        'QIDTR': 'INVERSE LEVERAGED INDEX',
        'SPXUTR': 'INVERSE LEVERAGED INDEX',
        'SQQQTR': 'INVERSE LEVERAGED INDEX',
        
        # Treasury/Bond ETFs
        'TLT': 'LONG TERM TREASURIES',
        'ZROZ': 'LONG TERM TREASURIES',
        'VGLT': 'LONG TERM TREASURIES',
        'SPTL': 'LONG TERM TREASURIES',
        'TLH': 'LONG TERM TREASURIES',
        'TLTTR': 'LONG TERM TREASURIES',
        'TLT_COMPLETE': 'LONG TERM TREASURIES',
        'ZROZX': 'LONG TERM TREASURIES',
        'ZROZ_COMPLETE': 'LONG TERM TREASURIES',
        'IEF': 'INTERMEDIATE TREASURIES',
        'VGIT': 'INTERMEDIATE TREASURIES',
        'SPTI': 'INTERMEDIATE TREASURIES',
        'IEI': 'INTERMEDIATE TREASURIES',
        'IEFTR': 'INTERMEDIATE TREASURIES',
        'IEF_COMPLETE': 'INTERMEDIATE TREASURIES',
        'SHY': 'SHORT TERM TREASURIES',
        'BIL': 'SHORT TERM TREASURIES',
        'VGSH': 'SHORT TERM TREASURIES',
        'SPTS': 'SHORT TERM TREASURIES',
        'SHV': 'SHORT TERM TREASURIES',
        'TBILL': 'SHORT TERM TREASURIES',
        'TBILL_COMPLETE': 'SHORT TERM TREASURIES',
        '^TNX': 'TREASURY YIELDS',
        '^TYX': 'TREASURY YIELDS',
        '^FVX': 'TREASURY YIELDS',
        '^IRX': 'TREASURY YIELDS',
        'AGG': 'TOTAL BOND MARKET',
        'BND': 'TOTAL BOND MARKET',
        'BNDX': 'TOTAL BOND MARKET',
        'GOVT': 'TOTAL BOND MARKET',
        'LQD': 'CORPORATE BONDS',
        'HYG': 'CORPORATE BONDS',
        'JNK': 'CORPORATE BONDS',
        'MUB': 'CORPORATE BONDS',
        'TIP': 'CORPORATE BONDS',
        'VTIP': 'CORPORATE BONDS',
        
        # Gold/Precious Metals
        'GLD': 'PRECIOUS METALS',
        'GOLDX': 'PRECIOUS METALS',
        'GOLDSIM': 'PRECIOUS METALS',
        'IAU': 'PRECIOUS METALS',
        'IAUM': 'PRECIOUS METALS',
        'GLDM': 'PRECIOUS METALS',
        'SGOL': 'PRECIOUS METALS',
        'GOLD_COMPLETE': 'PRECIOUS METALS',
        'GOLDSIM_COMPLETE': 'PRECIOUS METALS',
        'GC=F': 'GOLD FUTURES',
        'GOLDF': 'GOLD FUTURES',
        'GDX': 'GOLD MINERS',
        'GDXJ': 'GOLD MINERS',
        'SLV': 'SILVER/PLATINUM/PALLADIUM',
        'SI=F': 'SILVER/PLATINUM/PALLADIUM',
        'SILVER': 'SILVER/PLATINUM/PALLADIUM',
        'PPLT': 'SILVER/PLATINUM/PALLADIUM',
        'PL=F': 'SILVER/PLATINUM/PALLADIUM',
        'PLATINUM': 'SILVER/PLATINUM/PALLADIUM',
        'PALL': 'SILVER/PLATINUM/PALLADIUM',
        'PA=F': 'SILVER/PLATINUM/PALLADIUM',
        'PALLADIUM': 'SILVER/PLATINUM/PALLADIUM',
        
        # Commodities
        'USO': 'ENERGY COMMODITIES',
        'CL=F': 'ENERGY COMMODITIES',
        'OIL': 'ENERGY COMMODITIES',
        'UNG': 'ENERGY COMMODITIES',
        'NG=F': 'ENERGY COMMODITIES',
        'NATGAS': 'ENERGY COMMODITIES',
        'DBA': 'AGRICULTURAL COMMODITIES',
        'ZC=F': 'AGRICULTURAL COMMODITIES',
        'CORN': 'AGRICULTURAL COMMODITIES',
        'ZS=F': 'AGRICULTURAL COMMODITIES',
        'SOYBEAN': 'AGRICULTURAL COMMODITIES',
        'KC=F': 'AGRICULTURAL COMMODITIES',
        'COFFEE': 'AGRICULTURAL COMMODITIES',
        'SB=F': 'AGRICULTURAL COMMODITIES',
        'SUGAR': 'AGRICULTURAL COMMODITIES',
        'CT=F': 'AGRICULTURAL COMMODITIES',
        'COTTON': 'AGRICULTURAL COMMODITIES',
        'DBC': 'BROAD COMMODITIES',
        'GSG': 'BROAD COMMODITIES',
        'PDBC': 'BROAD COMMODITIES',
        'BCI': 'BROAD COMMODITIES',
        'HG=F': 'INDUSTRIAL METALS',
        'COPPER': 'INDUSTRIAL METALS',
        'KMLM': 'MANAGED FUTURES',
        'DBMF': 'MANAGED FUTURES',
        'KMLMX': 'MANAGED FUTURES',
        'KMLM_COMPLETE': 'MANAGED FUTURES',
        'DBMFX': 'MANAGED FUTURES',
        'DBMF_COMPLETE': 'MANAGED FUTURES',
        
        # Cryptocurrency
        'BTC-USD': 'DIGITAL ASSETS',
        'ETH-USD': 'DIGITAL ASSETS',
        'BITO': 'DIGITAL ASSETS',
        'GBTC': 'DIGITAL ASSETS',
        'ETHE': 'DIGITAL ASSETS',
        'BITCOINX': 'DIGITAL ASSETS',
        'BTC_COMPLETE': 'DIGITAL ASSETS',
        'BITCOIN': 'DIGITAL ASSETS',
        
        # Bitcoin ETFs
        'IBIT': 'DIGITAL ASSETS',      # iShares Bitcoin Trust (BlackRock)
        'FBTC': 'DIGITAL ASSETS',      # Fidelity Wise Origin Bitcoin Fund
        'BITB': 'DIGITAL ASSETS',      # Bitwise Bitcoin ETF Trust
        'ARKB': 'DIGITAL ASSETS',      # ARK 21Shares Bitcoin ETF
        'BTCO': 'DIGITAL ASSETS',      # Invesco Galaxy Bitcoin ETF
        'HODL': 'DIGITAL ASSETS',      # 21Shares Crypto Basket 10 ETP
        'EZBC': 'DIGITAL ASSETS',      # ETC Group Bitcoin ETP
        'XBTF': 'DIGITAL ASSETS',      # ProShares Bitcoin Futures Strategy ETF
        'BTF': 'DIGITAL ASSETS',       # Valkyrie Bitcoin Strategy ETF
        'BTCC': 'DIGITAL ASSETS',      # Purpose Bitcoin ETF (Canada)
        
        # Technology
        'XLK': 'TECHNOLOGY INDEX FUND',
        'VGT': 'TECHNOLOGY INDEX FUND',
        'FTEC': 'TECHNOLOGY INDEX FUND',
        'IGM': 'TECHNOLOGY INDEX FUND',
        'SMH': 'SEMICONDUCTORS',
        'SOXX': 'SEMICONDUCTORS',
        'SOXS': 'LEVERAGED SEMICONDUCTORS',
        'USD': 'LEVERAGED SEMICONDUCTORS',
        
        # S&P 500 Sector Indices
        '^SP500-45': 'SECTOR INDEX FUND',
        '^SP500-35': 'SECTOR INDEX FUND',
        '^SP500-30': 'SECTOR INDEX FUND',
        '^SP500-40': 'SECTOR INDEX FUND',
        '^SP500-10': 'SECTOR INDEX FUND',
        '^SP500-20': 'SECTOR INDEX FUND',
        '^SP500-25': 'SECTOR INDEX FUND',
        '^SP500-15': 'SECTOR INDEX FUND',
        '^SP500-55': 'SECTOR INDEX FUND',
        '^SP500-60': 'SECTOR INDEX FUND',
        '^SP500-50': 'SECTOR INDEX FUND',
        'XLF': 'SECTOR INDEX FUND',
        'XLV': 'SECTOR INDEX FUND',
        'XLP': 'SECTOR INDEX FUND',
        'XLE': 'SECTOR INDEX FUND',
        'XLI': 'SECTOR INDEX FUND',
        'XLY': 'SECTOR INDEX FUND',
        'XLB': 'SECTOR INDEX FUND',
        'XLU': 'SECTOR INDEX FUND',
        'XLC': 'SECTOR INDEX FUND',
        
        # Real Estate
        'VNQ': 'REIT INDEX',
        'IYR': 'REIT INDEX',
        'SCHH': 'REIT INDEX',
        'RWR': 'REIT INDEX',
        'XLRE': 'REIT INDEX',
        
        # Volatility
        'VXX': 'VOLATILITY INDEX',
        'UVXY': 'VOLATILITY INDEX',
        'SVXY': 'VOLATILITY INDEX',
        
        # Cash Equivalents
        'CASH': 'CASH EQUIVALENT',
        'SGOV': 'CASH EQUIVALENT',
        'USFR': 'CASH EQUIVALENT',
        'ZEROX': 'CASH EQUIVALENT',
    }
    
    return custom_industries.get(base_ticker, None)

# =============================================================================
# PERFORMANCE OPTIMIZATION: CACHING FUNCTIONS
# =============================================================================

# =============================================================================
# STANDALONE LEVERAGE FUNCTIONS (Independent from Backtest_Engine)
# =============================================================================

def _ensure_naive_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Return a copy of obj with a tz-naive DatetimeIndex."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        return obj
    idx = obj.index
    if getattr(idx, "tz", None) is not None:
        obj = obj.copy()
        obj.index = idx.tz_convert(None)
    return obj

def _get_default_risk_free_rate(dates):
    """Get default risk-free rate when all other methods fail."""
    default_daily = (1 + 0.02) ** (1 / 365.25) - 1
    result = pd.Series(default_daily, index=pd.to_datetime(dates))
    # Ensure the result is timezone-naive
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_convert(None)
    return result

def get_risk_free_rate_robust(dates):
    """Simple risk-free rate fetcher using Yahoo Finance treasury data."""
    try:
        dates = pd.to_datetime(dates)
        if isinstance(dates, pd.DatetimeIndex):
            if getattr(dates, "tz", None) is not None:
                dates = dates.tz_convert(None)
        
        # Get treasury data - use ^IRX (13-week treasury) as primary for leverage calculations
        # Fallback hierarchy: ^IRX → ^FVX → ^TNX → ^TYX
        symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]
        ticker = None
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="max", auto_adjust=False)
                if hist is not None and not hist.empty and 'Close' in hist.columns:
                    break
            except Exception:
                continue
        
        if ticker is None:
            # Final fallback to ^TNX
            ticker = yf.Ticker("^TNX")
        hist = ticker.history(period="max", auto_adjust=False)
        
        if hist is not None and not hist.empty and 'Close' in hist.columns:
            # Filter valid data
            valid_data = hist[hist['Close'].notnull() & (hist['Close'] > 0)]
            
            if not valid_data.empty:
                # Convert annual percentage to daily rate
                annual_rates = valid_data['Close'] / 100.0
                daily_rates = (1 + annual_rates) ** (1 / 365.25) - 1.0
                
                # Create series with timezone-naive index
                daily_rate_series = pd.Series(daily_rates.values, index=daily_rates.index)
                if getattr(daily_rate_series.index, "tz", None) is not None:
                    daily_rate_series.index = daily_rate_series.index.tz_convert(None)
                
                # For each target date, use the most recent available rate
                result = pd.Series(index=dates, dtype=float)
                
                for i, target_date in enumerate(dates):
                    # Find the most recent treasury date <= target_date
                    valid_dates = daily_rate_series.index[daily_rate_series.index <= target_date]
                    
                    if len(valid_dates) > 0:
                        closest_date = valid_dates.max()
                        result.iloc[i] = daily_rate_series.loc[closest_date]
                    else:
                        # If no data before target date, use the earliest available
                        result.iloc[i] = daily_rate_series.iloc[0]
                
                # Handle any remaining NaN values
                if result.isna().any():
                    result = result.fillna(method='ffill').fillna(method='bfill')
                    if result.isna().any():
                        result = result.fillna(0.000105)  # Default daily rate
                
                return result
        
        # Fallback to default if all else fails
        return _get_default_risk_free_rate(dates)
        
    except Exception:
        return _get_default_risk_free_rate(dates)

def parse_ticker_parameters(ticker_symbol: str) -> tuple[str, float, float]:
    """
    Parse ticker symbol to extract base ticker, leverage multiplier, and expense ratio.
    
    Args:
        ticker_symbol: Ticker symbol with optional parameters (e.g., "SPY?L=3?E=0.84")
        
    Returns:
        tuple: (base_ticker, leverage_multiplier, expense_ratio)
        
    Examples:
        "SPY" -> ("SPY", 1.0, 0.0)
        "SPY?L=3" -> ("SPY", 3.0, 0.0)
        "QQQ?L=3?E=0.84" -> ("QQQ", 3.0, 0.84)
        "QQQ?E=1?L=2" -> ("QQQ", 2.0, 1.0)  # Order doesn't matter
    """
    # Convert commas to dots for decimal separators (like case conversion)
    ticker_symbol = ticker_symbol.replace(",", ".")
    base_ticker = ticker_symbol
    leverage = 1.0
    expense_ratio = 0.0

    # Parse leverage parameter first
    if "?L=" in base_ticker:
        try:
            parts = base_ticker.split("?L=", 1)
            base_ticker = parts[0]
            leverage_part = parts[1]
            
            # Check if there are more parameters after leverage
            if "?" in leverage_part:
                leverage_str, remaining = leverage_part.split("?", 1)
                leverage = float(leverage_str)
                base_ticker += "?" + remaining
            else:
                leverage = float(leverage_part)
                
            # Leverage validation removed - allow any leverage value for testing
        except (ValueError, IndexError) as e:
            leverage = 1.0
    
    # Parse expense ratio parameter
    if "?E=" in base_ticker:
        try:
            parts = base_ticker.split("?E=", 1)
            base_ticker = parts[0]
            expense_part = parts[1]
            
            # Check if there are more parameters after expense ratio
            if "?" in expense_part:
                expense_str, remaining = expense_part.split("?", 1)
                expense_ratio = float(expense_str)
                base_ticker += "?" + remaining
            else:
                expense_ratio = float(expense_part)
                
            # Expense ratio validation removed - allow any expense ratio value for testing
        except (ValueError, IndexError) as e:
            expense_ratio = 0.0
            
    return base_ticker.strip(), leverage, expense_ratio

def parse_leverage_ticker(ticker_symbol: str) -> tuple[str, float]:
    """
    Parse ticker symbol to extract base ticker and leverage multiplier.
    Backward compatibility wrapper for parse_ticker_parameters.
    
    Args:
        ticker_symbol: Ticker symbol, potentially with leverage (e.g., "SPY?L=3")
        
    Returns:
        tuple: (base_ticker, leverage_multiplier)
        
    Examples:
        "SPY" -> ("SPY", 1.0)
        "SPY?L=3" -> ("SPY", 3.0)
        "QQQ?L=2" -> ("QQQ", 2.0)
    """
    base_ticker, leverage, _ = parse_ticker_parameters(ticker_symbol)
    return base_ticker, leverage

def apply_daily_leverage(price_data: pd.DataFrame, leverage: float, expense_ratio: float = 0.0) -> pd.DataFrame:
    """
    Apply daily leverage multiplier and expense ratio to price data, simulating leveraged ETF behavior.
    
    Leveraged ETFs reset daily, so we apply the leverage to daily returns and then
    compound the results to get the leveraged price series. Includes daily cost drag
    equivalent to (leverage - 1) × risk_free_rate plus daily expense ratio drag.
    
    Args:
        price_data: DataFrame with 'Close' column containing price data
        leverage: Leverage multiplier (e.g., 3.0 for 3x leverage)
        expense_ratio: Annual expense ratio in percentage (e.g., 1.0 for 1% annual expense)
        
    Returns:
        DataFrame with leveraged price data including cost drag and expense ratio drag
    """
    if leverage == 1.0 and expense_ratio == 0.0:
        return price_data.copy()
    
    # Create a copy to avoid modifying original data
    leveraged_data = price_data.copy()
    
    # Get time-varying risk-free rates for the entire period
    try:
        risk_free_rates = get_risk_free_rate_robust(price_data.index)
        # Ensure risk-free rates are timezone-naive to match price_data
        if getattr(risk_free_rates.index, "tz", None) is not None:
            risk_free_rates.index = risk_free_rates.index.tz_localize(None)
    except Exception as e:
        raise
    
    # Calculate daily cost drag: (leverage - 1) × risk_free_rate
    # risk_free_rates is already in daily format, so we don't need to divide by 365.25
    try:
        daily_cost_drag = (leverage - 1) * risk_free_rates
    except Exception as e:
        raise
    
    # Calculate daily expense ratio drag: expense_ratio / 100 / 365.25 (annual to daily)
    daily_expense_drag = expense_ratio / 100.0 / 365.25
    
    # VECTORIZED APPROACH - 100-1000x faster than for loop!
    # Calculate daily returns using vectorized operations
    prices = price_data['Close'].values  # Convert to NumPy array for speed
    daily_returns = np.zeros(len(prices))
    daily_returns[1:] = prices[1:] / prices[:-1] - 1  # Vectorized returns calculation
    
    # Apply leverage to returns and subtract cost drag and expense ratio drag
    leveraged_returns = (daily_returns * leverage) - daily_cost_drag.values - daily_expense_drag
    leveraged_returns[0] = 0  # First day has no return
    
    # Compound the leveraged returns to get prices (cumulative product)
    # Using np.cumprod for vectorized compounding
    leveraged_prices = prices[0] * np.cumprod(1 + leveraged_returns)
    
    # Convert back to pandas Series with proper index
    leveraged_prices = pd.Series(leveraged_prices, index=price_data.index)
    
    # Update the Close price with leveraged prices
    leveraged_data['Close'] = leveraged_prices
    
    # Recalculate price changes with the new leveraged prices
    leveraged_data['Price_change'] = leveraged_data['Close'].pct_change(fill_method=None)
    
    # IMPORTANT: Preserve all other columns (like Dividend_per_share) from original data
    # The dividends should remain at their original values (not leveraged) for leveraged ETFs
    # This is correct behavior - leveraged ETFs don't multiply dividends
    
    return leveraged_data

def apply_leverage_to_hist_data(hist_data, leverage):
    """Apply leverage to historical data"""
    if leverage == 1.0:
        return hist_data
    
    # Create a copy to avoid modifying original
    leveraged_data = hist_data.copy()
    
    # Apply leverage to price columns
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        if col in leveraged_data.columns:
            leveraged_data[col] = leveraged_data[col] * leverage
    
    # Recalculate price changes with the new leveraged prices
    leveraged_data['Price_change'] = leveraged_data['Close'].pct_change(fill_method=None)
    
    return leveraged_data

def get_ticker_data_for_valuation(ticker_symbol, period="max", auto_adjust=False):
    """Get ticker data specifically for valuation tables
    
    This function handles two special cases:
    1. Canadian tickers: Converts USD OTC to Canadian exchange (CNSWF → CSU.TO)
    2. Leveraged tickers: Uses underlying ticker for valuation (NVDL → NVDA)
    
    Args:
        ticker_symbol: Stock ticker symbol
        period: Data period
        auto_adjust: Auto-adjust setting
    """
    try:
        # Parse leverage from ticker symbol
        base_ticker, leverage = parse_leverage_ticker(ticker_symbol)
        
        # Check if this is a leveraged ticker (for valuation stats only)
        leveraged_map = get_leveraged_ticker_underlying()
        if base_ticker.upper() in leveraged_map:
            underlying_ticker = leveraged_map[base_ticker.upper()]
            print(f"📊 VALUATION: Using {underlying_ticker} stats for leveraged ticker {base_ticker}")
            resolved_ticker = underlying_ticker
        else:
            # Resolve ticker alias for valuation tables (converts USD OTC to Canadian exchange, indices to ETFs)
            resolved_ticker = resolve_index_to_etf_for_stats(resolve_ticker_alias(base_ticker))
        
        # Special handling for synthetic complete tickers
        if resolved_ticker == "ZEROX":
            return generate_zero_return_data(period)
        if resolved_ticker == "SPYSIM_COMPLETE":
            return get_spysim_complete_data(period)
        if resolved_ticker == "GOLDSIM_COMPLETE":
            return get_goldsim_complete_data(period)
        if resolved_ticker == "TBILL_COMPLETE":
            return get_tbill_complete_data(period)
        if resolved_ticker == "IEF_COMPLETE":
            return get_ief_complete_data(period)
        if resolved_ticker == "TLT_COMPLETE":
            return get_tlt_complete_data(period)
        if resolved_ticker == "ZROZ_COMPLETE":
            return get_zroz_complete_data(period)
        if resolved_ticker == "BTC_COMPLETE":
            return get_bitcoin_complete_data(period)
        if resolved_ticker == "BTC-USD":
            # Fallback: If BTC-USD fails, try BTC_COMPLETE data
            try:
                return get_bitcoin_complete_data(period)
            except Exception:
                # Final fallback: use yfinance BTC-USD
                pass
        if resolved_ticker == "KMLM_COMPLETE":
            return get_kmlm_complete_data(period)
        if resolved_ticker == "DBMF_COMPLETE":
            return get_dbmf_complete_data(period)
        
        # Create ticker object with resolved ticker
        ticker_obj = yf.Ticker(resolved_ticker)
        
        # Get historical data
        hist = ticker_obj.history(period=period, auto_adjust=auto_adjust)
        
        if hist.empty:
            return None
            
        # Apply leverage if specified
        if leverage != 1.0:
            hist = apply_leverage_to_hist_data(hist, leverage)
            
        return hist
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {str(e)}")
        return None

def get_multiple_tickers_batch(ticker_list, period="max", auto_adjust=False):
    """
    Smart batch download with fallback to individual downloads.
    
    Strategy:
    1. Try batch download (fast - 1 API call for all tickers)
    2. If batch fails → fallback to individual downloads (reliable)
    3. Invalid tickers are skipped, others continue
    
    Args:
        ticker_list: List of ticker symbols (can include leverage format)
        period: Data period
        auto_adjust: Auto-adjust setting
    
    Returns:
        Dict[ticker_symbol, DataFrame]: Data for each ticker
    """
    if not ticker_list:
        return {}
    
    results = {}
    yahoo_tickers = []
    
    for ticker_symbol in ticker_list:
        # Parse parameters from ticker if it has ?L= or ?E= format
        base_ticker = ticker_symbol
        leverage = 1.0
        expense_ratio = 0.0
        
        if '?L=' in ticker_symbol or '?E=' in ticker_symbol:
            parts = ticker_symbol.split('?')
            base_ticker = parts[0]
            for part in parts[1:]:
                if part.startswith('L='):
                    try:
                        leverage = float(part[2:])
                    except:
                        pass
                elif part.startswith('E='):
                    try:
                        expense_ratio = float(part[2:])
                    except:
                        pass
        
        resolved = resolve_index_to_etf_for_stats(resolve_ticker_alias(base_ticker))
        print(f"[BATCH DEBUG] {ticker_symbol} -> base={base_ticker}, resolved={resolved}, L={leverage}, E={expense_ratio}")
        yahoo_tickers.append((ticker_symbol, resolved, leverage, expense_ratio))
    
    # Extract unique resolved tickers for batch download (exclude _COMPLETE tickers and ZEROX)
    resolved_list = list(set([resolved for _, resolved, _, _ in yahoo_tickers if not resolved.endswith('_COMPLETE') and resolved != 'ZEROX']))
    print(f"[BATCH DEBUG] Resolved tickers to download: {resolved_list}")
    
    try:
        # BATCH DOWNLOAD - Fast path (1 API call for all)
        if len(resolved_list) > 1:
            batch_data = yf.download(
                resolved_list,
                period=period,
                auto_adjust=auto_adjust,
                progress=False,
                group_by='ticker'
            )
            print(f"[BATCH DEBUG] Batch download result columns: {batch_data.columns.tolist() if not batch_data.empty else 'EMPTY'}")
            
            # Process batch data
            if not batch_data.empty:
                for ticker_symbol, resolved, leverage, expense_ratio in yahoo_tickers:
                    # Skip _COMPLETE tickers and ZEROX (they will be handled in fallback section)
                    if resolved.endswith('_COMPLETE') or resolved == 'ZEROX':
                        continue
                    
                    try:
                        if len(resolved_list) > 1:
                            ticker_data = batch_data[resolved][['Close', 'Dividends']] if resolved in batch_data else pd.DataFrame()
                        else:
                            ticker_data = batch_data[['Close', 'Dividends']]
                        
                        print(f"[BATCH DEBUG] Processing {ticker_symbol}: data_empty={ticker_data.empty}, shape={ticker_data.shape if not ticker_data.empty else 'N/A'}")
                        
                        if not ticker_data.empty:
                            # Apply leverage/expense if needed
                            if leverage != 1.0 or expense_ratio != 0.0:
                                print(f"[BATCH DEBUG] Applying leverage L={leverage}, E={expense_ratio} to {ticker_symbol}")
                                ticker_data = apply_daily_leverage(ticker_data, leverage, expense_ratio)
                            results[ticker_symbol] = ticker_data
                            print(f"[BATCH DEBUG] ✓ {ticker_symbol} added to results")
                        else:
                            results[ticker_symbol] = pd.DataFrame()
                            print(f"[BATCH DEBUG] ✗ {ticker_symbol} is EMPTY")
                    except Exception as e:
                        print(f"[BATCH DEBUG] ✗ Error processing {ticker_symbol} from batch: {e}")
                        pass
            else:
                raise Exception("Batch download returned empty")
                
    except Exception:
        # FALLBACK - Batch failed, download individually
        pass
    
    # Download any missing tickers individually (fallback or single ticker)
    for ticker_symbol, resolved, leverage, expense_ratio in yahoo_tickers:
        if ticker_symbol not in results or results[ticker_symbol].empty:
            try:
                # Handle special complete tickers
                if resolved == "ZEROX":
                    hist = generate_zero_return_data(period)
                elif resolved == "SPYSIM_COMPLETE":
                    hist = get_spysim_complete_data(period)
                elif resolved == "GOLDSIM_COMPLETE":
                    hist = get_goldsim_complete_data(period)
                elif resolved == "GOLD_COMPLETE":
                    hist = get_gold_complete_data(period)
                elif resolved == "ZROZ_COMPLETE":
                    hist = get_zroz_complete_data(period)
                elif resolved == "TLT_COMPLETE":
                    hist = get_tlt_complete_data(period)
                elif resolved == "BTC_COMPLETE":
                    hist = get_bitcoin_complete_data(period)
                elif resolved == "KMLM_COMPLETE":
                    hist = get_kmlm_complete_data(period)
                elif resolved == "IEF_COMPLETE":
                    hist = get_ief_complete_data(period)
                elif resolved == "DBMF_COMPLETE":
                    hist = get_dbmf_complete_data(period)
                elif resolved == "TBILL_COMPLETE":
                    hist = get_tbill_complete_data(period)
                else:
                    # Regular Yahoo Finance ticker
                    ticker = yf.Ticker(resolved)
                    hist = ticker.history(period=period, auto_adjust=auto_adjust)[["Close", "Dividends"]]
                
                if hist is not None and not hist.empty:
                    if leverage != 1.0 or expense_ratio != 0.0:
                        hist = apply_daily_leverage(hist, leverage, expense_ratio)
                    results[ticker_symbol] = hist
                else:
                    results[ticker_symbol] = pd.DataFrame()
            except Exception as e:
                print(f"Error downloading {ticker_symbol}: {e}")
                results[ticker_symbol] = pd.DataFrame()
    
    return results

def get_ticker_data(ticker_symbol, period="max", auto_adjust=False):
    """Get ticker data (NO CACHE for maximum freshness)
    
    Args:
        ticker_symbol: Stock ticker symbol (supports leverage format like SPY?L=3)
        period: Data period
        auto_adjust: Auto-adjust setting
    """
    try:
        # Parse leverage from ticker symbol
        base_ticker, leverage = parse_leverage_ticker(ticker_symbol)
        
        # Use original ticker for backtests and calculations (NO conversion)
        resolved_ticker = base_ticker
        
        # Special handling for synthetic complete tickers
        if resolved_ticker == "ZEROX":
            return generate_zero_return_data(period)
        if resolved_ticker == "SPYSIM_COMPLETE":
            return get_spysim_complete_data(period)
        if resolved_ticker == "GOLDSIM_COMPLETE":
            return get_goldsim_complete_data(period)
        if resolved_ticker == "GOLD_COMPLETE":
            return get_gold_complete_data(period)
        if resolved_ticker == "ZROZ_COMPLETE":
            return get_zroz_complete_data(period)
        if resolved_ticker == "TLT_COMPLETE":
            return get_tlt_complete_data(period)
        if resolved_ticker == "BTC_COMPLETE":
            return get_bitcoin_complete_data(period)
        if resolved_ticker == "BTC-USD":
            # Fallback: If BTC-USD fails, try BTC_COMPLETE data
            try:
                return get_bitcoin_complete_data(period)
            except Exception:
                # Final fallback: use yfinance BTC-USD
                pass
        if resolved_ticker == "KMLM_COMPLETE":
            return get_kmlm_complete_data(period)
        if resolved_ticker == "IEF_COMPLETE":
            return get_ief_complete_data(period)
        if resolved_ticker == "DBMF_COMPLETE":
            return get_dbmf_complete_data(period)
        if resolved_ticker == "TBILL_COMPLETE":
            return get_tbill_complete_data(period)
        
        ticker = yf.Ticker(resolved_ticker)
        hist = ticker.history(period=period, auto_adjust=auto_adjust)[["Close", "Dividends"]]
        
        if hist.empty:
            return hist
            
        # Apply leverage if specified
        if leverage != 1.0:
            hist = apply_daily_leverage(hist, leverage)
            
        return hist
    except Exception:
        return pd.DataFrame()

# Synthetic Complete Ticker Functions
def get_spysim_complete_data(period="max"):
    """Get complete SPYSIM data from our custom SPYSIM ticker"""
    try:
        from Complete_Tickers.SPYSIM_COMPLETE_TICKER import create_spysim_complete_ticker
        spysim_data = create_spysim_complete_ticker()
        if spysim_data is not None and not spysim_data.empty:
            result = pd.DataFrame({
                'Close': spysim_data,
                'Dividends': [0.0] * len(spysim_data)
            }, index=spysim_data.index)
            return result
        else:
            return None
    except Exception as e:
        try:
            ticker = yf.Ticker("^SP500TR")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_goldsim_complete_data(period="max"):
    """Get complete GOLDSIM data from our custom GOLDSIM ticker"""
    try:
        from Complete_Tickers.GOLDSIM_COMPLETE_TICKER import create_goldsim_complete_ticker
        goldsim_data = create_goldsim_complete_ticker()
        if goldsim_data is not None and not goldsim_data.empty:
            result = pd.DataFrame({
                'Close': goldsim_data,
                'Dividends': [0.0] * len(goldsim_data)
            }, index=goldsim_data.index)
            return result
        else:
            print("⚠️ WARNING: GOLDSIM ticker returned empty data, falling back to GLD")
            ticker = yf.Ticker("GLD")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
    except Exception as e:
        print(f"⚠️ WARNING: GOLDSIM error: {e}, falling back to GLD")
        try:
            ticker = yf.Ticker("GLD")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def generate_zero_return_data(period="max"):
    """Generate synthetic zero return data for ZEROX ticker"""
    try:
        ref_ticker = yf.Ticker("SPY")
        ref_hist = ref_ticker.history(period=period)
        if ref_hist.empty:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=365)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            dates = ref_hist.index
        zero_data = pd.DataFrame({
            'Close': [100.0] * len(dates),
            'Dividends': [0.0] * len(dates)
        }, index=dates)
        return zero_data
    except Exception:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        zero_data = pd.DataFrame({
            'Close': [100.0] * len(dates),
            'Dividends': [0.0] * len(dates)
        }, index=dates)
        return zero_data

def get_gold_complete_data(period="max"):
    """Get complete gold data from our custom gold ticker"""
    try:
        from Complete_Tickers.GOLD_COMPLETE_TICKER import create_gold_complete_ticker
        gold_data = create_gold_complete_ticker()
        if gold_data is not None and not gold_data.empty:
            result = pd.DataFrame({
                'Close': gold_data,
                'Dividends': [0.0] * len(gold_data)
            }, index=gold_data.index)
            return result
        else:
            print("⚠️ WARNING: GOLD_COMPLETE ticker returned empty data, falling back to GLD")
            ticker = yf.Ticker("GLD")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
    except Exception as e:
        print(f"⚠️ WARNING: GOLD_COMPLETE error: {e}, falling back to GLD")
        try:
            ticker = yf.Ticker("GLD")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_zroz_complete_data(period="max"):
    """Get complete ZROZ data from our custom ZROZ ticker"""
    try:
        from Complete_Tickers.ZROZ_COMPLETE_TICKER import create_safe_zroz_ticker
        zroz_data = create_safe_zroz_ticker()
        if zroz_data is not None and not zroz_data.empty:
            result = pd.DataFrame({
                'Close': zroz_data,
                'Dividends': [0.0] * len(zroz_data)
            }, index=zroz_data.index)
            return result
        else:
            return None
    except Exception as e:
        try:
            ticker = yf.Ticker("ZROZ")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_tlt_complete_data(period="max"):
    """Get complete TLT data from our custom TLT ticker"""
    try:
        from Complete_Tickers.TLT_COMPLETE_TICKER import create_safe_tlt_ticker
        tlt_data = create_safe_tlt_ticker()
        if tlt_data is not None and not tlt_data.empty:
            result = pd.DataFrame({
                'Close': tlt_data,
                'Dividends': [0.0] * len(tlt_data)
            }, index=tlt_data.index)
            return result
        else:
            return None
    except Exception as e:
        try:
            ticker = yf.Ticker("TLT")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_bitcoin_complete_data(period="max"):
    """Get complete Bitcoin data from our custom Bitcoin ticker"""
    try:
        from Complete_Tickers.BITCOIN_COMPLETE_TICKER import create_bitcoin_complete_ticker
        bitcoin_data = create_bitcoin_complete_ticker()
        if bitcoin_data is not None and not bitcoin_data.empty:
            result = pd.DataFrame({
                'Close': bitcoin_data,
                'Dividends': [0.0] * len(bitcoin_data)
            }, index=bitcoin_data.index)
            return result
        else:
            return None
    except Exception as e:
        try:
            ticker = yf.Ticker("BTC-USD")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_kmlm_complete_data(period="max"):
    """Get complete KMLM data from our custom KMLM ticker"""
    try:
        from Complete_Tickers.KMLM_COMPLETE_TICKER import create_kmlm_complete_ticker
        kmlm_data = create_kmlm_complete_ticker()
        if kmlm_data is not None and not kmlm_data.empty:
            result = pd.DataFrame({
                'Close': kmlm_data,
                'Dividends': [0.0] * len(kmlm_data)
            }, index=kmlm_data.index)
            return result
        else:
            return None
    except Exception as e:
        try:
            ticker = yf.Ticker("KMLM")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_ief_complete_data(period="max"):
    """Get complete IEF data from our custom IEF ticker"""
    try:
        from Complete_Tickers.IEF_COMPLETE_TICKER import create_ief_complete_ticker
        ief_data = create_ief_complete_ticker()
        if ief_data is not None and not ief_data.empty:
            result = pd.DataFrame({
                'Close': ief_data,
                'Dividends': [0.0] * len(ief_data)
            }, index=ief_data.index)
            return result
        else:
            return None
    except Exception as e:
        try:
            ticker = yf.Ticker("IEF")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_dbmf_complete_data(period="max"):
    """Get complete DBMF data from our custom DBMF ticker"""
    try:
        from Complete_Tickers.DBMF_COMPLETE_TICKER import create_dbmf_complete_ticker
        dbmf_data = create_dbmf_complete_ticker()
        if dbmf_data is not None and not dbmf_data.empty:
            result = pd.DataFrame({
                'Close': dbmf_data,
                'Dividends': [0.0] * len(dbmf_data)
            }, index=dbmf_data.index)
            return result
        else:
            return None
    except Exception as e:
        try:
            ticker = yf.Ticker("DBMF")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_tbill_complete_data(period="max"):
    """Get complete TBILL data from our custom TBILL ticker"""
    try:
        from Complete_Tickers.TBILL_COMPLETE_TICKER import create_tbill_complete_ticker
        tbill_data = create_tbill_complete_ticker()
        if tbill_data is not None and not tbill_data.empty:
            result = pd.DataFrame({
                'Close': tbill_data,
                'Dividends': [0.0] * len(tbill_data)
            }, index=tbill_data.index)
            return result
        else:
            return None
    except Exception as e:
        try:
            ticker = yf.Ticker("SGOV")
            return ticker.history(period=period, auto_adjust=True)[["Close", "Dividends"]]
        except:
            return pd.DataFrame()

def get_ticker_info(ticker_symbol):
    """Get ticker info (NO CACHE for maximum freshness)
    
    This function handles two special cases:
    1. Canadian tickers: Converts USD OTC to Canadian exchange (CNSWF → CSU.TO)
    2. Leveraged tickers: Uses underlying ticker for info (NVDL → NVDA)
    """
    try:
        # Parse leverage from ticker symbol
        base_ticker, leverage = parse_leverage_ticker(ticker_symbol)
        
        # Check if this is a leveraged ticker (for valuation stats only)
        leveraged_map = get_leveraged_ticker_underlying()
        if base_ticker.upper() in leveraged_map:
            underlying_ticker = leveraged_map[base_ticker.upper()]
            resolved_ticker = underlying_ticker
        else:
            # Resolve ticker alias for valuation tables (converts USD OTC to Canadian exchange, indices to ETFs)
            resolved_ticker = resolve_index_to_etf_for_stats(resolve_ticker_alias(base_ticker))
        
        stock = yf.Ticker(resolved_ticker)
        info = stock.info
        return info
    except Exception:
        return {}

def get_multiple_tickers_info_batch(ticker_list):
    """
    Batch download ticker info for multiple tickers to improve performance.
    
    This is much faster than calling get_ticker_info() one by one.
    Uses yf.download() to get basic price data in one call, then fetches
    individual info only for tickers that need detailed stats.
    
    Args:
        ticker_list: List of ticker symbols
        
    Returns:
        Dict[ticker_symbol, dict]: Info dict for each ticker
    """
    if not ticker_list:
        return {}
    
    results = {}
    
    # Resolve all tickers first
    resolved_map = {}  # Maps original ticker -> resolved ticker
    leveraged_map = get_leveraged_ticker_underlying()
    
    for ticker_symbol in ticker_list:
        base_ticker, leverage = parse_leverage_ticker(ticker_symbol)
        
        # Check if leveraged ticker
        if base_ticker.upper() in leveraged_map:
            resolved = leveraged_map[base_ticker.upper()]
        else:
            resolved = resolve_index_to_etf_for_stats(resolve_ticker_alias(base_ticker))
        
        resolved_map[ticker_symbol] = resolved
    
    # Get unique resolved tickers to minimize API calls
    unique_resolved = list(set(resolved_map.values()))
    
    # Fetch info for all unique tickers in parallel using threading
    import concurrent.futures
    
    def fetch_single_info(resolved_ticker):
        try:
            stock = yf.Ticker(resolved_ticker)
            return resolved_ticker, stock.info
        except:
            return resolved_ticker, {}
    
    # Use ThreadPoolExecutor for parallel fetching (much faster)
    info_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_single_info, resolved) for resolved in unique_resolved]
        for future in concurrent.futures.as_completed(futures):
            resolved_ticker, info = future.result()
            info_results[resolved_ticker] = info
    
    # Map back to original ticker symbols
    for ticker_symbol in ticker_list:
        resolved = resolved_map[ticker_symbol]
        results[ticker_symbol] = info_results.get(resolved, {})
    
    return results

def calculate_portfolio_metrics(portfolio_config, allocation_data):
    """Calculate portfolio metrics (NO CACHE for maximum freshness)"""
    # This will calculate portfolio metrics fresh every time
    # Note: The actual calculation logic remains unchanged
    return portfolio_config, allocation_data  # Placeholder - will be filled in by calling functions

def optimize_data_loading():
    """Session state optimization to prevent redundant operations - PAGE-SPECIFIC"""
    # Use page-specific keys to prevent conflicts between pages
    page_prefix = "alloc_page_"
    
    # Initialize performance flags if not present
    if f'{page_prefix}data_loaded' not in st.session_state:
        st.session_state[f'{page_prefix}data_loaded'] = False
    if f'{page_prefix}last_refresh' not in st.session_state:
        st.session_state[f'{page_prefix}last_refresh'] = None
    
    # Check if data needs refresh (5 minutes)
    current_time = datetime.datetime.now()
    if (st.session_state[f'{page_prefix}last_refresh'] is None or 
        (current_time - st.session_state[f'{page_prefix}last_refresh']).seconds > 300):
        st.session_state[f'{page_prefix}data_loaded'] = False
        st.session_state[f'{page_prefix}last_refresh'] = current_time
    
    return st.session_state[f'{page_prefix}data_loaded']

def create_safe_hash_key(data):
    """Create a safe, consistent hash key from complex data structures"""
    import hashlib
    import json
    try:
        # Convert to JSON string and hash for consistent hash keys
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    except Exception:
        # Fallback to string representation
        return hashlib.md5(str(data).encode()).hexdigest()

def run_fresh_backtest(portfolios_config_hash, start_date_str, end_date_str, benchmark_str, page_id="allocations"):
    """Run fresh backtest calculations (NO CACHE for maximum freshness)
    
    Args:
        portfolios_config_hash: Hash of portfolio configurations to detect changes
        start_date_str: Start date as string
        end_date_str: End date as string  
        benchmark_str: Benchmark ticker as string
        page_id: Page identifier to prevent cross-page conflicts
    """
    # This will be called by the actual backtest functions when needed
    # The caching key includes all parameters that affect the backtest result
    return portfolios_config_hash, start_date_str, end_date_str, benchmark_str, page_id  # Placeholder

def check_currency_warning(tickers):
    """
    Check if any tickers are non-USD and display a warning.
    """
    non_usd_suffixes = ['.TO', '.V', '.CN', '.AX', '.L', '.PA', '.AS', '.SW', '.T', '.HK', '.KS', '.TW', '.JP']
    non_usd_tickers = []
    
    for ticker in tickers:
        if any(ticker.endswith(suffix) for suffix in non_usd_suffixes):
            non_usd_tickers.append(ticker)
    
    if non_usd_tickers:
        st.warning(f"⚠️ **Currency Warning**: The following tickers are not in USD: {', '.join(non_usd_tickers)}. "
                  f"Currency conversion is not taken into account, which may affect allocation accuracy. "
                  f"Consider using USD equivalents for more accurate results.")
st.set_page_config(layout="wide", page_title="Portfolio Allocation Analysis", page_icon="📈")
st.markdown("""
<style>
    /* Global Styles for the App */
    .st-emotion-cache-1f87s81 {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .st-emotion-cache-1v0bb62 button {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .st-emotion-cache-1v0bb62 button:hover {
        background-color: #0056b3 !important;
        border-color: #0056b3 !important;
        transform: translateY(-2px);
    }
    /* Fix for the scrollable dataframe - forces it to be non-scrollable */
    div.st-emotion-cache-1ftv8z > div {
        overflow: visible !important;
        max-height: none !important;
    }
    /* Make the 'View Details' button more obvious */
    button[aria-label="View Details"] {
        background-color: #0ea5e9 !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: 0 4px 8px rgba(14,165,233,0.16) !important;
    }
    button[aria-label="View Details"]:hover {
        background-color: #0891b2 !important;
    }
</style>
<a id="top"></a>
<button id="back-to-top" onclick="window.scrollTo(0, 0);">⬆️</button>
<style>
    #back-to-top {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        opacity: 0.7;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 24px;
        cursor: pointer;
        display: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: opacity 0.3s;
    }
    #back-to-top:hover {
        opacity: 1;
    }
</style>
<script>
    window.onscroll = function() {
        var button = document.getElementById("back-to-top");
        if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {
            button.style.display = "block";
        } else {
            button.style.display = "none";
        }
    };
</script>
""", unsafe_allow_html=True)



# ...existing code...

# ==============================================================================
# PAGE-SCOPED SESSION STATE INITIALIZATION - ALLOCATIONS PAGE
# ==============================================================================
# Ensure complete independence from other pages by using page-specific session keys
if 'allocations_page_initialized' not in st.session_state:
    st.session_state.allocations_page_initialized = True

# Initialize page-specific session state with default configurations
if 'alloc_portfolio_configs' not in st.session_state:
    # Default configuration for allocations page
    st.session_state.alloc_portfolio_configs = [
        {
            'name': 'Allocation Portfolio',
            'stocks': [
                {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True, 'include_in_sma_filter': True, 'max_allocation_percent': None},
                {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True, 'include_in_sma_filter': True, 'max_allocation_percent': None},
                {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True, 'include_in_sma_filter': True, 'max_allocation_percent': None},
                {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True, 'include_in_sma_filter': True, 'max_allocation_percent': None},
            ],
            'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
                          'added_amount': 0,
              'added_frequency': 'none',
              'rebalancing_frequency': 'Monthly',
              'start_date_user': None,
              'end_date_user': None,
              'start_with': 'oldest',
            'use_momentum': True,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [
                {"lookback": 365, "exclude": 30, "weight": 0.5},
                {"lookback": 180, "exclude": 30, "weight": 0.3},
                {"lookback": 120, "exclude": 30, "weight": 0.2},
            ],
            'calc_beta': False,
            'calc_volatility': False,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
            'use_sma_filter': False,
            'sma_window': 200,
            'ma_type': 'SMA',
        }
    ]
if 'alloc_active_portfolio_index' not in st.session_state:
    st.session_state.alloc_active_portfolio_index = 0
if 'alloc_rerun_flag' not in st.session_state:
    st.session_state.alloc_rerun_flag = False

# Clean up any existing portfolio configs to remove unused settings
if 'alloc_portfolio_configs' in st.session_state:
    for config in st.session_state.alloc_portfolio_configs:
        config.pop('use_relative_momentum', None)
        config.pop('equal_if_all_negative', None)
        
        # Ensure MA filter fields exist with default values
        if 'use_sma_filter' not in config:
            config['use_sma_filter'] = False
        if 'sma_window' not in config:
            config['sma_window'] = 200
        if 'ma_type' not in config:
            config['ma_type'] = 'SMA'
        
        # Ensure all stocks have include_in_sma_filter and ma_reference_ticker settings
        for stock in config.get('stocks', []):
            if 'include_in_sma_filter' not in stock:
                stock['include_in_sma_filter'] = True
            if 'ma_reference_ticker' not in stock:
                stock['ma_reference_ticker'] = ''  # Empty = use ticker's own MA
            if 'max_allocation_percent' not in stock:
                stock['max_allocation_percent'] = None
if 'alloc_paste_json_text' not in st.session_state:
    st.session_state.alloc_paste_json_text = ""

# ==============================================================================
# END PAGE-SCOPED SESSION STATE INITIALIZATION
# ==============================================================================

# Use page-scoped active portfolio for the allocations page
active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index] if 'alloc_portfolio_configs' in st.session_state and 'alloc_active_portfolio_index' in st.session_state else None

# NUCLEAR SYNC: FORCE momentum widgets to sync with the active portfolio
if active_portfolio:
    # NUCLEAR APPROACH: FORCE momentum session state widget to sync
    st.session_state['alloc_active_use_momentum'] = active_portfolio.get('use_momentum', False)

if active_portfolio:
    # Removed duplicate Portfolio Name input field
    if st.session_state.get('alloc_rerun_flag', False):
        st.session_state.alloc_rerun_flag = False
        st.rerun()

import numpy as np
import pandas as pd
def calculate_mwrr(values, cash_flows, dates):
    # Exact logic from app.py for MWRR calculation
    try:
        from scipy.optimize import brentq
        values = pd.Series(values).dropna()
        flows = pd.Series(cash_flows).reindex(values.index, fill_value=0.0)
        if len(values) < 2:
            return np.nan
        dates = pd.to_datetime(values.index)
        start_date = dates[0]
        time_periods = np.array([(d - start_date).days / 365.25 for d in dates])
        initial_investment = -values.iloc[0]
        significant_flows = flows[flows != 0]
        cash_flow_dates = [start_date]
        cash_flow_amounts = [initial_investment]
        cash_flow_times = [0.0]
        for date, flow in significant_flows.items():
            if date != dates[0] and date != dates[-1]:
                cash_flow_dates.append(pd.to_datetime(date))
                cash_flow_amounts.append(flow)
                cash_flow_times.append((pd.to_datetime(date) - start_date).days / 365.25)
        cash_flow_dates.append(dates[-1])
        cash_flow_amounts.append(values.iloc[-1])
        cash_flow_times.append((dates[-1] - start_date).days / 365.25)
        cash_flow_amounts = np.array(cash_flow_amounts)
        cash_flow_times = np.array(cash_flow_times)
        def npv(rate):
            return np.sum(cash_flow_amounts / (1 + rate) ** cash_flow_times)
        try:
            irr = brentq(npv, -0.999, 10)
            return irr * 100
        except (ValueError, RuntimeError):
            return np.nan
    except Exception:
        return np.nan
    # Exact logic from app.py for MWRR calculation
    try:
        from scipy.optimize import brentq
        values = pd.Series(values).dropna()
        flows = pd.Series(cash_flows).reindex(values.index, fill_value=0.0)
        if len(values) < 2:
            return np.nan
        dates = pd.to_datetime(values.index)
        start_date = dates[0]
        time_periods = np.array([(d - start_date).days / 365.25 for d in dates])
        initial_investment = -values.iloc[0]
        significant_flows = flows[flows != 0]
        cash_flow_dates = [start_date]
        cash_flow_amounts = [initial_investment]
        cash_flow_times = [0.0]
        for date, flow in significant_flows.items():
            if date != dates[0] and date != dates[-1]:
                cash_flow_dates.append(pd.to_datetime(date))
                cash_flow_amounts.append(flow)
                cash_flow_times.append((pd.to_datetime(date) - start_date).days / 365.25)
        cash_flow_dates.append(dates[-1])
        cash_flow_amounts.append(values.iloc[-1])
        cash_flow_times.append((dates[-1] - start_date).days / 365.25)
        cash_flow_amounts = np.array(cash_flow_amounts)
        cash_flow_times = np.array(cash_flow_times)
        def npv(rate):
            return np.sum(cash_flow_amounts / (1 + rate) ** cash_flow_times)
        try:
            irr = brentq(npv, -0.999, 10)
            return irr
        except (ValueError, RuntimeError):
            return np.nan
    except Exception:
        return np.nan
# Backtest_Engine.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io
import contextlib
import json
from datetime import datetime, timedelta
from warnings import warn
from scipy.optimize import newton, brentq, root_scalar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# Custom CSS for a better layout, a distinct primary button, and the fixed 'Back to Top' button
st.markdown("""
<style>
    /* Global Styles for the App */
    .st-emotion-cache-1f87s81 {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .st-emotion-cache-1v0bb62 button {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .st-emotion-cache-1v0bb62 button:hover {
        background-color: #0056b3 !important;
        border-color: #0056b3 !important;
        transform: translateY(-2px);
    }
    /* Fix for the scrollable dataframe - forces it to be non-scrollable */
    div.st-emotion-cache-1ftv8z > div {
        overflow: visible !important;
        max-height: none !important;
    }
</style>
<a id="top"></a>
<button id="back-to-top" onclick="window.scrollTo(0, 0);">⬆️</button>
<style>
    #back-to-top {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        opacity: 0.7;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 24px;
        cursor: pointer;
        display: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: opacity 0.3s;
    }
    #back-to-top:hover {
        opacity: 1;
    }
</style>
<script>
    window.onscroll = function() {
        var button = document.getElementById("back-to-top");
        if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {
            button.style.display = "block";
        } else {
            button.style.display = "none";
        }
    };
</script>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="Portfolio Allocation Analysis")

st.title("Portfolio Allocations")
st.markdown("Use the forms below to configure and run backtests to obtain allocation insights.")

# Portfolio Name
if 'alloc_portfolio_name' not in st.session_state:
    st.session_state.alloc_portfolio_name = "Allocation Portfolio"
alloc_portfolio_name = st.text_input("Portfolio Name", value=st.session_state.alloc_portfolio_name, key="alloc_portfolio_name_input")
st.session_state.alloc_portfolio_name = alloc_portfolio_name

# Sync portfolio name with active portfolio configuration
if 'alloc_active_portfolio_index' in st.session_state:
    active_idx = st.session_state.alloc_active_portfolio_index
    if 'alloc_portfolio_configs' in st.session_state:
        if active_idx < len(st.session_state.alloc_portfolio_configs):
            st.session_state.alloc_portfolio_configs[active_idx]['name'] = alloc_portfolio_name

# -----------------------
# Default JSON configs (for initialization)
# -----------------------
default_configs = [
    # 1) Benchmark only (SPY) - yearly rebalancing and yearly additions
    {
        'name': 'Benchmark Only (SPY)',
        'stocks': [
            {'ticker': 'SPY', 'allocation': 1.0, 'include_dividends': True},
        ],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 10000,
        'added_frequency': 'Annually',
        'rebalancing_frequency': 'Annually',
    'start_date_user': None,
    'end_date_user': None,
    'start_with': 'oldest',
        'use_momentum': False,
        'momentum_windows': [],
    'calc_beta': False,
    'calc_volatility': False,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
                    'vol_window_days': 365,
            'exclude_days_vol': 30,
            'use_minimal_threshold': False,
            'minimal_threshold_percent': 4.0,
            'use_max_allocation': False,
            'max_allocation_percent': 20.0,
        },
]

# -----------------------
# Helper functions
# -----------------------
def get_trading_days(start_date, end_date):
    return pd.bdate_range(start=start_date, end=end_date)

def get_dates_by_freq(freq, start, end, market_days):
    market_days = sorted(market_days)
    
    # Ensure market_days are timezone-naive for consistent comparison
    market_days_naive = [d.tz_localize(None) if d.tz is not None else d for d in market_days]
    
    if freq == "market_day":
        return set(market_days)
    elif freq == "calendar_day":
        return set(pd.date_range(start=start, end=end, freq='D'))
    elif freq == "Weekly":
        base = pd.date_range(start=start, end=end, freq='W-MON')
    elif freq == "Biweekly":
        base = pd.date_range(start=start, end=end, freq='2W-MON')
    elif freq == "Monthly":
        # Fixed calendar dates: 1st of each month
        monthly = []
        for y in range(start.year, end.year + 1):
            for m in range(1, 13):
                monthly.append(pd.Timestamp(year=y, month=m, day=1))
        base = pd.DatetimeIndex(monthly)
    elif freq == "Quarterly":
        # Fixed calendar dates: 1st of each quarter (Jan 1, Apr 1, Jul 1, Oct 1)
        quarterly = []
        for y in range(start.year, end.year + 1):
            for m in [1, 4, 7, 10]:  # Q1, Q2, Q3, Q4
                quarterly.append(pd.Timestamp(year=y, month=m, day=1))
        base = pd.DatetimeIndex(quarterly)
    elif freq == "Semiannually":
        # First day of Jan and Jul each year
        semi = []
        for y in range(start.year, end.year + 1):
            for m in [1, 7]:
                semi.append(pd.Timestamp(year=y, month=m, day=1))
        base = pd.DatetimeIndex(semi)
    elif freq == "Annually":
        base = pd.date_range(start=start, end=end, freq='YS')
    elif freq == "Never" or freq == "none" or freq is None:
        return set()
    else:
        raise ValueError(f"Unknown frequency: {freq}")

    dates = []
    for d in base:
        # Ensure d is timezone-naive for comparison
        d_naive = d.tz_localize(None) if d.tz is not None else d
        idx = np.searchsorted(market_days_naive, d_naive, side='right')
        if idx > 0 and market_days_naive[idx-1] >= d_naive:
            dates.append(market_days[idx-1])  # Use original market_days for return
        elif idx < len(market_days_naive):
            dates.append(market_days[idx])  # Use original market_days for return
    return set(dates)

def calculate_cagr(values, dates):
    if len(values) < 2:
        return np.nan
    start_val = values[0]
    end_val = values[-1]
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0 or start_val == 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1

def calculate_max_drawdown(values):
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / np.where(peak == 0, 1, peak)
    return np.nanmin(drawdowns), drawdowns

def calculate_volatility(returns):
    # Annualized volatility
    return np.std(returns) * np.sqrt(365) if len(returns) > 1 else np.nan

def calculate_beta(returns, benchmark_returns):
    # Use exact logic from app.py
    portfolio_returns = pd.Series(returns)
    benchmark_returns = pd.Series(benchmark_returns)
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 2:
        return np.nan
    pr = portfolio_returns.reindex(common_idx).dropna()
    br = benchmark_returns.reindex(common_idx).dropna()
    # Re-align after dropping NAs
    common_idx = pr.index.intersection(br.index)
    if len(common_idx) < 2 or br.loc[common_idx].var() == 0:
        return np.nan
    cov = pr.loc[common_idx].cov(br.loc[common_idx])
    var = br.loc[common_idx].var()
    return cov / var

# FIXED: Correct Sortino Ratio calculation
def calculate_sortino(returns, risk_free_rate=0):
    # Annualized Sortino ratio
    target_return = risk_free_rate / 365.25  # Daily target
    downside_returns = returns[returns < target_return]
    if len(downside_returns) < 2:
        return np.nan
    downside_std = np.std(downside_returns) * np.sqrt(365.25)
    if downside_std == 0:
        return np.nan
    expected_return = returns.mean() * 365.25
    return (expected_return - risk_free_rate) / downside_std

# FIXED: Correct Ulcer Index calculation
def calculate_ulcer_index(values):
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    peak[peak == 0] = 1 # Avoid division by zero
    drawdown_sq = ((values - peak) / peak)**2
    return np.sqrt(np.mean(drawdown_sq)) if len(drawdown_sq) > 0 else np.nan

# FIXED: Correct UPI calculation
def calculate_upi(cagr, ulcer_index, risk_free_rate=0):
    if pd.isna(cagr) or pd.isna(ulcer_index) or ulcer_index == 0:
        return np.nan
    return (cagr - risk_free_rate) / ulcer_index

# -----------------------
# Timer function for next rebalance date
# -----------------------
def calculate_next_rebalance_date(rebalancing_frequency, last_rebalance_date):
    """
    Calculate the next rebalance date based on rebalancing frequency and last rebalance date.
    Excludes today and yesterday as mentioned in the requirements.
    """
    if not last_rebalance_date or rebalancing_frequency == 'none':
        return None, None, None
    
    # Convert to datetime if it's a pandas Timestamp
    if hasattr(last_rebalance_date, 'to_pydatetime'):
        last_rebalance_date = last_rebalance_date.to_pydatetime()
    
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    # If last rebalance was today or yesterday, use the day before yesterday as base
    if last_rebalance_date.date() >= yesterday:
        base_date = yesterday - timedelta(days=1)
    else:
        base_date = last_rebalance_date.date()
    
    def add_months_safely(date, months_to_add):
        """Safely add months to a date, handling day overflow"""
        year = date.year + (date.month + months_to_add - 1) // 12
        month = ((date.month + months_to_add - 1) % 12) + 1
        
        # Find the last day of the target month
        if month == 12:
            last_day_of_month = 31
        else:
            # Get the first day of the next month and subtract 1 day
            next_month_first = datetime(year, month + 1, 1).date()
            last_day_of_month = (next_month_first - timedelta(days=1)).day
        
        # Use the minimum of the original day or the last day of the target month
        safe_day = min(date.day, last_day_of_month)
        
        return date.replace(year=year, month=month, day=safe_day)
    
    # Calculate next rebalance date based on frequency
    if rebalancing_frequency == 'market_day':
        # Next market day (simplified - just next day for now)
        next_date = base_date + timedelta(days=1)
    elif rebalancing_frequency == 'calendar_day':
        next_date = base_date + timedelta(days=1)
    elif rebalancing_frequency == 'week':
        next_date = base_date + timedelta(weeks=1)
    elif rebalancing_frequency == '2weeks':
        next_date = base_date + timedelta(weeks=2)
    elif rebalancing_frequency == 'month':
        # Add one month safely
        next_date = add_months_safely(base_date, 1)
    elif rebalancing_frequency == '3months':
        # Add three months safely
        next_date = add_months_safely(base_date, 3)
    elif rebalancing_frequency == '6months':
        # Add six months safely
        next_date = add_months_safely(base_date, 6)
    elif rebalancing_frequency == 'year':
        next_date = base_date.replace(year=base_date.year + 1)
    else:
        return None, None, None
    
    # Calculate time until next rebalance
    now = datetime.now()
    # Ensure both datetimes are offset-naive for comparison and subtraction
    if hasattr(next_date, 'tzinfo') and next_date.tzinfo is not None:
        next_date = next_date.replace(tzinfo=None)
    next_rebalance_datetime = datetime.combine(next_date, time(9, 30))  # Assume 9:30 AM market open
    if hasattr(next_rebalance_datetime, 'tzinfo') and next_rebalance_datetime.tzinfo is not None:
        next_rebalance_datetime = next_rebalance_datetime.replace(tzinfo=None)
    if hasattr(now, 'tzinfo') and now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    # If next rebalance is in the past, calculate the next one iteratively instead of recursively
    max_iterations = 10  # Prevent infinite loops
    iteration = 0
    while next_rebalance_datetime <= now and iteration < max_iterations:
        iteration += 1
        if rebalancing_frequency in ['market_day', 'calendar_day']:
            next_date = next_date + timedelta(days=1)
        elif rebalancing_frequency == 'week':
            next_date = next_date + timedelta(weeks=1)
        elif rebalancing_frequency == '2weeks':
            next_date = next_date + timedelta(weeks=2)
        elif rebalancing_frequency == 'month':
            # Add one month safely
            next_date = add_months_safely(next_date, 1)
        elif rebalancing_frequency == '3months':
            # Add three months safely
            next_date = add_months_safely(next_date, 3)
        elif rebalancing_frequency == '6months':
            # Add six months safely
            next_date = add_months_safely(next_date, 6)
        elif rebalancing_frequency == 'year':
            next_date = next_date.replace(year=next_date.year + 1)
        
        next_rebalance_datetime = datetime.combine(next_date, time(9, 30))
    
    time_until = next_rebalance_datetime - now
    
    return next_date, time_until, next_rebalance_datetime

def format_time_until(time_until):
    """Format the time until next rebalance in a human-readable format."""
    if not time_until:
        return "Unknown"
    
    total_seconds = int(time_until.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    
    if days > 0:
        return f"{days} days, {hours} hours, {minutes} minutes"
    elif hours > 0:
        return f"{hours} hours, {minutes} minutes"
    else:
        return f"{minutes} minutes"

# -----------------------
# PDF Generation Functions
# -----------------------
def plotly_to_matplotlib_figure(plotly_fig, title="", width_inches=8, height_inches=6):
    """
    Convert a Plotly figure to a matplotlib figure for PDF generation
    """
    try:
        # Extract data from Plotly figure
        fig_data = plotly_fig.data
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Define a color palette for different traces
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_index = 0
        
        # Process each trace
        for trace in fig_data:
            if trace.type == 'scatter':
                x_data = trace.x
                y_data = trace.y
                name = trace.name if hasattr(trace, 'name') and trace.name else f'Trace {color_index}'
                
                # Get color from trace or use palette
                if hasattr(trace, 'line') and hasattr(trace.line, 'color') and trace.line.color:
                    color = trace.line.color
                else:
                    color = colors[color_index % len(colors)]
                
                # Plot the line
                ax.plot(x_data, y_data, label=name, linewidth=2, color=color)
                color_index += 1
                
            elif trace.type == 'bar':
                x_data = trace.x
                y_data = trace.y
                name = trace.name if hasattr(trace, 'name') and trace.name else f'Bar {color_index}'
                
                # Get color from trace or use palette
                if hasattr(trace, 'marker') and hasattr(trace.marker, 'color') and trace.marker.color:
                    color = trace.marker.color
                else:
                    color = colors[color_index % len(colors)]
                
                # Plot the bars
                ax.bar(x_data, y_data, label=name, color=color, alpha=0.7)
                color_index += 1
                
            elif trace.type == 'pie':
                # Handle pie charts - SIMPLE PERFECT CIRCLE SOLUTION
                labels = trace.labels if hasattr(trace, 'labels') else []
                values = trace.values if hasattr(trace, 'values') else []
                
                if labels and values:
                    # Create a slightly wider figure to ensure perfect circle
                    fig_pie, ax_pie = plt.subplots(figsize=(8.5, 8))
                    
                    # Format long titles to break into multiple lines using textwrap
                    import textwrap
                    formatted_title = textwrap.fill(title, width=40, break_long_words=True, break_on_hyphens=False)
                    ax_pie.set_title(formatted_title, fontsize=14, fontweight='bold', pad=40, y=0.95)
                    
                    # Create pie chart with smart percentage display - hide small ones to prevent overlap
                    def smart_autopct(pct):
                        return f'{pct:.1f}%' if pct > 3 else ''  # Only show percentages > 3%
                    
                    wedges, texts, autotexts = ax_pie.pie(
                        values, 
                        labels=labels, 
                        autopct=smart_autopct,  # Use smart percentage display
                        startangle=90, 
                        colors=colors[:len(values)]
                    )
                    
                    # Create legend labels with percentages
                    legend_labels = []
                    for i, label in enumerate(labels):
                        percentage = (values[i] / sum(values)) * 100
                        legend_labels.append(f"{label} ({percentage:.1f}%)")
                    
                    # Add legend with SPECIFIC POSITIONING to prevent overlap
                    ax_pie.legend(wedges, legend_labels, title="Categories", 
                                loc="center left", bbox_to_anchor=(1.15, 0.5))
                    
                    # This is the magic - force perfect circle
                    ax_pie.axis('equal')
                    
                    # Add extra spacing to prevent overlap
                    plt.subplots_adjust(right=0.8)
                    
                    return fig_pie
        
        # Format the plot
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:  # Only add legend if there are labels
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Format x-axis for dates if needed
        if fig_data and len(fig_data) > 0 and hasattr(fig_data[0], 'x') and fig_data[0].x is not None:
            try:
                # Try to parse as dates
                dates = pd.to_datetime(fig_data[0].x)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator(interval=2))  # Show every 2 years
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            except:
                pass
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.text(0.5, 0.5, f'Error converting plot: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig

def create_matplotlib_table(data, headers=None):
    """Create a matplotlib table for PDF generation."""
    try:
        if data is None or len(data) == 0:
            return None
        
        # Convert data to proper format
        if isinstance(data, pd.DataFrame):
            table_data = data.values.tolist()
            if headers is None:
                headers = data.columns.tolist()
        else:
            table_data = data
            if headers is None:
                headers = [f'Col {i+1}' for i in range(len(data[0]))]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        return fig
    except Exception as e:
        print(f"Error creating matplotlib table: {e}")
        return None

def generate_allocations_pdf(custom_name=""):
    """Generate PDF report for allocations page."""
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("📄 Initializing PDF document...")
        
        # Get active portfolio configuration
        active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
        
        # Create PDF document
        buffer = io.BytesIO()
        
        # Add proper PDF metadata
        if custom_name.strip():
            title = f"Allocations Report - {custom_name.strip()}"
            subject = f"Portfolio Allocation Analysis: {custom_name.strip()}"
        else:
            title = "Allocations Report"
            subject = "Portfolio Allocation and Asset Distribution Analysis"
        
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=72, 
            leftMargin=72, 
            topMargin=72, 
            bottomMargin=72,
            title=title,
            author="Portfolio Backtest System",
            subject=subject,
            creator="Allocations Application"
        )
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER,
            leftIndent=0,
            rightIndent=0,
            firstLineIndent=0,
            wordWrap=False
        )
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15
        )
        
        # Title page (no page break before)
        title_style = ParagraphStyle(
            'TitlePage',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=reportlab_colors.Color(0.2, 0.4, 0.6),
            alignment=1  # Center alignment
        )
        
        subtitle_style = ParagraphStyle(
            'SubtitlePage',
            parent=styles['Normal'],
            fontSize=16,
            spaceAfter=40,
            textColor=reportlab_colors.Color(0.4, 0.6, 0.8),
            alignment=1  # Center alignment
        )
        
        # Main title - use custom name if provided
        if custom_name.strip():
            main_title = f"Allocations Report - {custom_name.strip()}"
            subtitle = f"Portfolio Allocation Analysis: {custom_name.strip()}"
        else:
            main_title = "Portfolio Allocations Report"
            subtitle = "Comprehensive Investment Portfolio Analysis"
        
        story.append(Paragraph(main_title, title_style))
        story.append(Paragraph(subtitle, subtitle_style))
        
        # Document metadata is set in SimpleDocTemplate creation above
        
        # Report metadata
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated on: {current_time}", styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Table of contents
        toc_style = ParagraphStyle(
            'TOC',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=reportlab_colors.Color(0.3, 0.5, 0.7)
        )
        
        story.append(Paragraph("Table of Contents", toc_style))
        toc_points = [
            "Portfolio Configurations & Parameters",
            "Target Allocation if Rebalanced Today",
            "Portfolio-Weighted Summary Statistics",
            "Portfolio Composition Analysis"
        ]
        
        for i, point in enumerate(toc_points, 1):
            story.append(Paragraph(f"{i}. {point}", styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Report overview
        overview_style = ParagraphStyle(
            'Overview',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=reportlab_colors.Color(0.3, 0.5, 0.7)
        )
        
        story.append(Paragraph("Report Overview", overview_style))
        story.append(Paragraph("This report provides comprehensive analysis of investment portfolios, including:", styles['Normal']))
        
        # Overview bullet points
        overview_points = [
            "Detailed portfolio configurations with all parameters and strategies",
            "Current allocations and rebalancing countdown timers"
        ]
        
        for point in overview_points:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        
        story.append(PageBreak())
        
        # Update progress
        progress_bar.progress(20)
        status_text.text("📊 Adding portfolio configurations...")
        
        # SECTION 1: Portfolio Configurations & Parameters
        story.append(Paragraph("1. Portfolio Configurations & Parameters", heading_style))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph(f"Portfolio: {active_portfolio.get('name', 'Unknown')}", subheading_style))
        story.append(Spacer(1, 10))
        
        # Create configuration table with all parameters
        config_data = [
            ['Parameter', 'Value', 'Description'],
            ['Initial Value', f"${active_portfolio.get('initial_value', 0):,.2f}", 'Starting portfolio value'],
            ['Added Amount', f"${active_portfolio.get('added_amount', 0):,.2f}", 'Regular contribution amount'],
            ['Added Frequency', active_portfolio.get('added_frequency', 'N/A'), 'How often contributions are made'],
            ['Rebalancing Frequency', active_portfolio.get('rebalancing_frequency', 'N/A'), 'How often portfolio is rebalanced'],
            ['Benchmark', active_portfolio.get('benchmark_ticker', 'N/A'), 'Performance comparison index'],
            ['Use Momentum', 'Yes' if active_portfolio.get('use_momentum', False) else 'No', 'Whether momentum strategy is enabled'],
            ['Momentum Strategy', active_portfolio.get('momentum_strategy', 'N/A'), 'Type of momentum calculation'],
            ['Negative Momentum Strategy', active_portfolio.get('negative_momentum_strategy', 'N/A'), 'How to handle negative momentum'],
            ['Calculate Beta', 'Yes' if active_portfolio.get('calc_beta', False) else 'No', 'Include beta in momentum weighting'],
            ['Calculate Volatility', 'Yes' if active_portfolio.get('calc_volatility', False) else 'No', 'Include volatility in momentum weighting'],
            ['Start Strategy', active_portfolio.get('start_with', 'N/A'), 'Initial allocation strategy'],
            ['Beta Lookback', f"{active_portfolio.get('beta_window_days', 0)} days", 'Days for beta calculation'],
            ['Beta Exclude', f"{active_portfolio.get('exclude_days_beta', 0)} days", 'Days excluded from beta calculation'],
            ['Volatility Lookback', f"{active_portfolio.get('vol_window_days', 0)} days", 'Days for volatility calculation'],
            ['Volatility Exclude', f"{active_portfolio.get('exclude_days_vol', 0)} days", 'Days excluded from volatility calculation'],
            ['Minimal Threshold', f"{active_portfolio.get('minimal_threshold_percent', 2.0):.1f}%" if active_portfolio.get('use_minimal_threshold', False) else 'Disabled', 'Minimum allocation percentage threshold'],
            ['Max Allocation', f"{active_portfolio.get('max_allocation_percent', 10.0):.1f}%" if active_portfolio.get('use_max_allocation', False) else 'Disabled', 'Maximum allocation percentage per stock']
        ]
        
        # Add momentum windows if they exist
        momentum_windows = active_portfolio.get('momentum_windows', [])
        if momentum_windows:
            for i, window in enumerate(momentum_windows, 1):
                lookback = window.get('lookback', 0)
                weight = window.get('weight', 0)
                config_data.append([
                    f'Momentum Window {i}',
                    f"{lookback} days, {weight:.2f}",
                    f"Lookback: {lookback} days, Weight: {weight:.2f}"
                ])
        
        # Add tickers with enhanced information
        tickers_data = [['Ticker', 'Allocation %', 'Reinvest Dividends']]
        for ticker_config in active_portfolio.get('stocks', []):
            tickers_data.append([
                ticker_config['ticker'],
                f"{ticker_config['allocation']*100:.1f}%",
                "✓" if ticker_config['include_dividends'] else "✗"
            ])
        
        # Create tables with proper column widths to prevent text overflow
        config_table = Table(config_data, colWidths=[2.2*inch, 1.8*inch, 2.5*inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('WORDWRAP', (0, 0), (-1, -1), True),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3)
        ]))
        
        tickers_table = Table(tickers_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
        tickers_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
        ]))
        
        story.append(config_table)
        story.append(PageBreak())
        # Show ticker allocations table, but hide Allocation % column if momentum is enabled
        if not active_portfolio.get('use_momentum', True):
            story.append(Paragraph("Initial Ticker Allocations (Entered by User):", styles['Heading3']))
            story.append(Paragraph("Note: These are the initial allocations entered by the user, not rebalanced allocations.", styles['Normal']))
            story.append(tickers_table)
            story.append(Spacer(1, 15))
        else:
            story.append(Paragraph("Initial Ticker Allocations:", styles['Heading3']))
            story.append(Paragraph("Note: Momentum strategy is enabled - ticker allocations are calculated dynamically based on momentum scores.", styles['Normal']))
            
            # Create modified table without Allocation % column for momentum strategies
            tickers_data_momentum = [['Ticker', 'Reinvest Dividends']]
            for ticker_config in active_portfolio.get('stocks', []):
                tickers_data_momentum.append([
                    ticker_config['ticker'],
                    "✓" if ticker_config['include_dividends'] else "✗"
                ])
            
            tickers_table_momentum = Table(tickers_data_momentum, colWidths=[2.25*inch, 2.25*inch])
            tickers_table_momentum.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
            ]))
            
            story.append(tickers_table_momentum)
            story.append(Spacer(1, 15))
        
        # Update progress
        progress_bar.progress(40)
        status_text.text("🎯 Adding allocation charts and timers...")
        
        # SECTION 2: Target Allocation if Rebalanced Today
        story.append(PageBreak())
        current_date_str = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"2. Target Allocation if Rebalanced Today ({current_date_str})", heading_style))
        story.append(Spacer(1, 2))
        
        # Get the allocation data from your existing UI - fetch the existing allocation data
        if 'alloc_snapshot_data' in st.session_state:
            snapshot = st.session_state.alloc_snapshot_data
            today_weights_map = snapshot.get('today_weights_map', {})
            
            # Process ALL portfolios, not just the active one
            portfolio_count = 0
            for portfolio_name, today_weights in today_weights_map.items():
                if today_weights:
                    # Add page break for all portfolios except the first one
                    if portfolio_count > 0:
                        story.append(PageBreak())
                    
                    portfolio_count += 1
                    
                    # Add portfolio header
                    story.append(Paragraph(f"Portfolio: {portfolio_name}", subheading_style))
                    story.append(Spacer(1, 2))
                    
                    # Create pie chart for this portfolio
                    try:
                        # Create labels and values for the plot
                        labels_today = [k for k, v in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
                        vals_today = [float(today_weights[k]) * 100 for k in labels_today]
                        
                        # Handle case where momentum goes to cash (all assets have negative momentum)
                        # If no labels or all values are very small, show 100% CASH
                        if not labels_today or sum(vals_today) < 0.1:
                            labels_today = ['CASH']
                            vals_today = [100.0]
                        
                        if labels_today and vals_today:
                            # Create matplotlib pie chart (same format as sector/industry)
                            fig, ax_target = plt.subplots(1, 1, figsize=(10, 10))
                            
                            # Create pie chart with smart percentage display - hide small ones to prevent overlap
                            def smart_autopct(pct):
                                return f'{pct:.1f}%' if pct > 3 else ''  # Only show percentages > 3%
                            
                            wedges_target, texts_target, autotexts_target = ax_target.pie(vals_today, autopct=smart_autopct, 
                                                                                         startangle=90)
                            
                            # Add ticker names with percentages outside the pie chart slices for allocations > 1.8%
                            for i, (wedge, ticker, alloc) in enumerate(zip(wedges_target, labels_today, vals_today)):
                                # Only show tickers above 1.8%
                                if alloc > 1.8:
                                    # Calculate position for the text (middle of the slice)
                                    angle = (wedge.theta1 + wedge.theta2) / 2
                                    # Convert angle to radians and calculate position
                                    rad = np.radians(angle)
                                    # Position text outside the pie chart at 1.4 radius (farther away)
                                    x = 1.4 * np.cos(rad)
                                    y = 1.4 * np.sin(rad)
                                    
                                    # Add ticker name with percentage under it (e.g., "ORLY 5%")
                                    ax_target.text(x, y, f"{ticker}\n{alloc:.1f}%", ha='center', va='center', 
                                                 fontsize=8, fontweight='bold', 
                                                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                                    
                                    # Add leader line from slice edge to label
                                    # Start from edge of pie chart (radius 1.0)
                                    line_start_x = 1.0 * np.cos(rad)
                                    line_start_y = 1.0 * np.sin(rad)
                                    # End at label position
                                    line_end_x = 1.25 * np.cos(rad)
                                    line_end_y = 1.25 * np.sin(rad)
                                    
                                    ax_target.plot([line_start_x, line_end_x], [line_start_y, line_end_y], 
                                                 'k-', linewidth=0.5, alpha=0.6)
                            
                            # Create legend with percentages - positioned farther to the right to avoid overlap
                            legend_labels = [f"{ticker} ({alloc:.1f}%)" for ticker, alloc in zip(labels_today, vals_today)]
                            ax_target.legend(wedges_target, legend_labels, title="Tickers", loc="center left", bbox_to_anchor=(1.15, 0, 0.5, 1), fontsize=10)
                            
                            # Wrap long titles to prevent them from going out of bounds
                            title_text = f'Target Allocation - {portfolio_name}'
                            # Use textwrap for proper word-based wrapping
                            import textwrap
                            wrapped_title = textwrap.fill(title_text, width=40, break_long_words=True, break_on_hyphens=False)
                            ax_target.set_title(wrapped_title, fontsize=14, fontweight='bold', pad=80)
                            # Force perfectly circular shape
                            ax_target.set_aspect('equal')
                            # Use tighter axis limits to make pie chart appear larger within its space
                            ax_target.set_xlim(-1.2, 1.2)
                            ax_target.set_ylim(-1.2, 1.2)
                            
                            # Adjust layout to accommodate legend and title (better spacing to prevent title cutoff)
                            # Use more aggressive spacing like sector/industry charts for bigger pie chart
                            plt.subplots_adjust(left=0.1, right=0.7, top=0.95, bottom=0.05)
                            
                            # Save to buffer
                            target_img_buffer = io.BytesIO()
                            fig.savefig(target_img_buffer, format='png', dpi=300, facecolor='white')
                            target_img_buffer.seek(0)
                            plt.close(fig)
                            
                            # Add to PDF - reduce pie chart size to fit everything on one page
                            story.append(Image(target_img_buffer, width=5.5*inch, height=5.5*inch))
                            
                            # Add Next Rebalance Timer information - calculate from portfolio data
                            story.append(Paragraph(f"Next Rebalance Timer - {portfolio_name}", subheading_style))
                            story.append(Spacer(1, 1))
                            
                            # Calculate timer information from portfolio configuration and allocation data
                            try:
                                # Get portfolio configuration
                                portfolio_cfg = None
                                if 'alloc_snapshot_data' in st.session_state:
                                    snapshot = st.session_state['alloc_snapshot_data']
                                    portfolio_configs = snapshot.get('portfolio_configs', [])
                                    portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == portfolio_name), None)
                                
                                if portfolio_cfg:
                                    rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'Monthly')
                                    initial_value = portfolio_cfg.get('initial_value', 10000)
                                    
                                    # Get last rebalance date from allocation data
                                    all_allocations = snapshot.get('all_allocations', {})
                                    portfolio_allocations = all_allocations.get(portfolio_name, {})
                                    
                                    if portfolio_allocations:
                                        alloc_dates = sorted(list(portfolio_allocations.keys()))
                                        if len(alloc_dates) > 1:
                                            last_rebal_date = alloc_dates[-2]  # Second to last date
                                        else:
                                            last_rebal_date = alloc_dates[-1] if alloc_dates else None
                                        
                                        if last_rebal_date:
                                            # Map frequency to function expectations
                                            frequency_mapping = {
                                                'monthly': 'month',
                                                'weekly': 'week',
                                                'bi-weekly': '2weeks',
                                                'biweekly': '2weeks',
                                                'quarterly': '3months',
                                                'semi-annually': '6months',
                                                'semiannually': '6months',
                                                'annually': 'year',
                                                'yearly': 'year',
                                                'market_day': 'market_day',
                                                'calendar_day': 'calendar_day',
                                                'never': 'none',
                                                'none': 'none'
                                            }
                                            mapped_frequency = frequency_mapping.get(rebalancing_frequency.lower(), rebalancing_frequency.lower())
                                            
                                            # Calculate next rebalance date
                                            
                                            next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                                                mapped_frequency, last_rebal_date
                                            )
                                            
                                            if next_date and time_until:
                                                story.append(Paragraph(f"Time Until Next Rebalance: {format_time_until(time_until)}", styles['Normal']))
                                                story.append(Paragraph(f"Target Rebalance Date: {next_date.strftime('%B %d, %Y')}", styles['Normal']))
                                                story.append(Paragraph(f"Rebalancing Frequency: {rebalancing_frequency}", styles['Normal']))
                                                story.append(Paragraph(f"Portfolio Value: ${initial_value:,.2f}", styles['Normal']))
                                            else:
                                                story.append(Paragraph("Next rebalance date calculation not available", styles['Normal']))
                                        else:
                                            story.append(Paragraph("No rebalancing history available", styles['Normal']))
                                    else:
                                        story.append(Paragraph("No allocation data available for timer calculation", styles['Normal']))
                                else:
                                    story.append(Paragraph("Portfolio configuration not found for timer calculation", styles['Normal']))
                            except Exception as e:
                                story.append(Paragraph(f"Error calculating timer information: {str(e)}", styles['Normal']))
                            
                            # Add page break after pie plot + timer to separate from allocation table
                            story.append(PageBreak())
                            
                            # Now add the allocation table on the next page
                            story.append(Paragraph(f"Allocation Details for {portfolio_name}", subheading_style))
                            story.append(Spacer(1, 10))
                            
                            # Create comprehensive allocation table with all columns
                            try:
                                if today_weights:
                                    # Get portfolio value and raw data for price calculations
                                    portfolio_value = 10000  # Default value
                                    raw_data = {}
                                    
                                    if 'alloc_snapshot_data' in st.session_state:
                                        snapshot = st.session_state['alloc_snapshot_data']
                                        portfolio_configs = snapshot.get('portfolio_configs', [])
                                        portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == portfolio_name), None)
                                        if portfolio_cfg:
                                            portfolio_value = portfolio_cfg.get('initial_value', 10000)
                                        raw_data = snapshot.get('raw_data', {})
                                    
                                    # Create comprehensive table with all columns
                                    headers = ['Asset', 'Allocation %', 'Price ($)', 'Shares', 'Total Value ($)', '% of Portfolio']
                                    table_rows = []
                                    
                                    for asset, weight in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])):
                                        if float(weight) > 0:
                                            alloc_pct = float(weight) * 100
                                            allocation_value = portfolio_value * float(weight)
                                            
                                            # Get current price
                                            current_price = None
                                            shares = 0.0
                                            if asset != 'CASH' and asset in raw_data:
                                                try:
                                                    df = raw_data[asset]
                                                    if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                        current_price = float(df['Close'].iloc[-1])
                                                        if current_price and current_price > 0:
                                                            shares = round(allocation_value / current_price, 1)
                                                except Exception:
                                                    pass
                                            
                                            # Calculate total value
                                            if current_price and current_price > 0:
                                                total_val = shares * current_price
                                            else:
                                                total_val = allocation_value
                                            
                                            # Calculate percentage of portfolio
                                            pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                                            
                                            # Format values for table
                                            price_str = f"${current_price:,.2f}" if current_price else "N/A"
                                            shares_str = f"{shares:,.1f}" if shares > 0 else "0.0"
                                            total_val_str = f"${total_val:,.2f}"
                                            
                                            table_rows.append([
                                                asset,
                                                f"{alloc_pct:.2f}%",
                                                price_str,
                                                shares_str,
                                                total_val_str,
                                                f"{pct_of_port:.2f}%"
                                            ])
                                    
                                    if table_rows:
                                        # Calculate total values for summary row
                                        total_alloc_pct = sum(float(row[1].rstrip('%')) for row in table_rows)
                                        total_value = sum(float(row[4].replace('$', '').replace(',', '')) for row in table_rows)
                                        total_port_pct = sum(float(row[5].rstrip('%')) for row in table_rows)
                                        
                                        # Add total row
                                        total_row = [
                                            'TOTAL',
                                            f"{total_alloc_pct:.2f}%",
                                            '',
                                            '',
                                            f"${total_value:,.2f}",
                                            f"{total_port_pct:.2f}%"
                                        ]
                                        
                                        # Create table with proper column widths
                                        page_width = 7.5*inch
                                        col_widths = [1.2*inch, 1.0*inch, 1.2*inch, 1.0*inch, 1.5*inch, 1.0*inch]
                                        alloc_table = Table([headers] + table_rows + [total_row], colWidths=col_widths)
                                        alloc_table.setStyle(TableStyle([
                                            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                                            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                            ('FONTSIZE', (0, 0), (-1, -1), 8),
                                            ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                                            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                            ('LEFTPADDING', (0, 0), (-1, -1), 3),
                                            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                                            ('TOPPADDING', (0, 0), (-1, -1), 2),
                                            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                                            ('WORDWRAP', (0, 0), (-1, -1), True),
                                            # Style the total row
                                            ('BACKGROUND', (0, -1), (-1, -1), reportlab_colors.Color(0.2, 0.4, 0.6)),
                                            ('TEXTCOLOR', (0, -1), (-1, -1), reportlab_colors.whitesmoke),
                                            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
                                        ]))
                                        story.append(alloc_table)
                                        story.append(Spacer(1, 5))
                                    else:
                                        story.append(Paragraph("No allocation data available", styles['Normal']))
                                else:
                                    story.append(Paragraph("No allocation data available", styles['Normal']))
                            except Exception as e:
                                story.append(Paragraph(f"Error creating allocation table: {str(e)}", styles['Normal']))
                            
                            story.append(Spacer(1, 5))
                        else:
                            story.append(Paragraph(f"No allocation data available for {portfolio_name}", styles['Normal']))
                    except Exception as e:
                        story.append(Paragraph(f"Error creating pie chart for {portfolio_name}: {str(e)}", styles['Normal']))
        else:
            story.append(Paragraph("Allocation data not available. Please run the allocation analysis first.", styles['Normal']))
            story.append(Spacer(1, 5))
        
        # Update progress
        progress_bar.progress(70)
        status_text.text("📊 Adding portfolio-weighted summary statistics...")
        
        # SECTION 3: Portfolio-Weighted Summary Statistics
        story.append(PageBreak())
        story.append(Paragraph("3. Portfolio-Weighted Summary Statistics", heading_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Metrics weighted by portfolio allocation - represents the total portfolio characteristics", styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Add data accuracy warning
        warning_style = ParagraphStyle(
            'WarningStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=reportlab_colors.Color(0.8, 0.4, 0.2),
            leftIndent=20,
            rightIndent=20,
            spaceAfter=15
        )
        story.append(Paragraph("⚠️ <b>Data Accuracy Notice:</b> Portfolio metrics (PE, Beta, etc.) are calculated from available data and may not accurately represent the portfolio if some ticker data is missing, outdated, or incorrect. These metrics should be used as indicative values for portfolio analysis.", warning_style))
        story.append(Spacer(1, 15))
        
        # Get portfolio-weighted metrics using the same approach as the main UI
        active_name = active_portfolio.get('name', 'Unknown')
        
        # Create summary statistics table
        summary_data = []
        
        # Get portfolio metrics from session state (these are calculated and stored in the main UI)
        portfolio_pe = getattr(st.session_state, 'portfolio_pe', None)
        portfolio_forward_pe = getattr(st.session_state, 'portfolio_forward_pe', None)
        portfolio_pb = getattr(st.session_state, 'portfolio_pb', None)
        portfolio_peg = getattr(st.session_state, 'portfolio_peg', None)
        portfolio_ps = getattr(st.session_state, 'portfolio_ps', None)
        portfolio_ev_ebitda = getattr(st.session_state, 'portfolio_ev_ebitda', None)
        portfolio_beta = getattr(st.session_state, 'portfolio_beta', None)
        portfolio_roe = getattr(st.session_state, 'portfolio_roe', None)
        portfolio_roa = getattr(st.session_state, 'portfolio_roa', None)
        portfolio_profit_margin = getattr(st.session_state, 'portfolio_profit_margin', None)
        portfolio_operating_margin = getattr(st.session_state, 'portfolio_operating_margin', None)
        portfolio_gross_margin = getattr(st.session_state, 'portfolio_gross_margin', None)
        portfolio_revenue_growth = getattr(st.session_state, 'portfolio_revenue_growth', None)
        portfolio_earnings_growth = getattr(st.session_state, 'portfolio_earnings_growth', None)
        portfolio_eps_growth = getattr(st.session_state, 'portfolio_eps_growth', None)
        portfolio_dividend_yield = getattr(st.session_state, 'portfolio_dividend_yield', None)
        portfolio_payout_ratio = getattr(st.session_state, 'portfolio_payout_ratio', None)
        portfolio_market_cap = getattr(st.session_state, 'portfolio_market_cap', None)
        portfolio_enterprise_value = getattr(st.session_state, 'portfolio_enterprise_value', None)
        
        # Valuation metrics
        if portfolio_pe is not None and not pd.isna(portfolio_pe):
            summary_data.append(["Valuation", "P/E Ratio", f"{portfolio_pe:.2f}", "Price-to-Earnings ratio weighted by portfolio allocation"])
        if portfolio_forward_pe is not None and not pd.isna(portfolio_forward_pe):
            summary_data.append(["Valuation", "Forward P/E", f"{portfolio_forward_pe:.2f}", "Forward Price-to-Earnings ratio weighted by portfolio allocation"])
        if portfolio_pb is not None and not pd.isna(portfolio_pb):
            summary_data.append(["Valuation", "Price/Book", f"{portfolio_pb:.2f}", "Price-to-Book ratio weighted by portfolio allocation"])
        if portfolio_peg is not None and not pd.isna(portfolio_peg):
            summary_data.append(["Valuation", "PEG Ratio", f"{portfolio_peg:.2f}", "P/E to Growth ratio weighted by portfolio allocation"])
        if portfolio_ps is not None and not pd.isna(portfolio_ps):
            summary_data.append(["Valuation", "Price/Sales", f"{portfolio_ps:.2f}", "Price-to-Sales ratio weighted by portfolio allocation"])
        if portfolio_ev_ebitda is not None and not pd.isna(portfolio_ev_ebitda):
            summary_data.append(["Valuation", "EV/EBITDA", f"{portfolio_ev_ebitda:.2f}", "EV/EBITDA ratio weighted by portfolio allocation"])
        
        # Risk metrics
        if portfolio_beta is not None and not pd.isna(portfolio_beta):
            summary_data.append(["Risk", "Beta", f"{portfolio_beta:.2f}", "Portfolio volatility relative to market (1.0 = market average)"])
        
        # Profitability metrics
        if portfolio_roe is not None and not pd.isna(portfolio_roe):
            summary_data.append(["Profitability", "ROE (%)", f"{portfolio_roe:.2f}%", "Return on Equity weighted by portfolio allocation"])
        if portfolio_roa is not None and not pd.isna(portfolio_roa):
            summary_data.append(["Profitability", "ROA (%)", f"{portfolio_roa:.2f}%", "Return on Assets weighted by portfolio allocation"])
        if portfolio_profit_margin is not None and not pd.isna(portfolio_profit_margin):
            summary_data.append(["Profitability", "Profit Margin (%)", f"{portfolio_profit_margin:.2f}%", "Net profit margin weighted by portfolio allocation"])
        if portfolio_operating_margin is not None and not pd.isna(portfolio_operating_margin):
            summary_data.append(["Profitability", "Operating Margin (%)", f"{portfolio_operating_margin:.2f}%", "Operating profit margin weighted by portfolio allocation"])
        if portfolio_gross_margin is not None and not pd.isna(portfolio_gross_margin):
            summary_data.append(["Profitability", "Gross Margin (%)", f"{portfolio_gross_margin:.2f}%", "Gross profit margin weighted by portfolio allocation"])
        
        # Growth metrics
        if portfolio_revenue_growth is not None and not pd.isna(portfolio_revenue_growth):
            summary_data.append(["Growth", "Revenue Growth (%)", f"{portfolio_revenue_growth:.2f}%", "Revenue growth rate weighted by portfolio allocation"])
        if portfolio_earnings_growth is not None and not pd.isna(portfolio_earnings_growth):
            summary_data.append(["Growth", "Earnings Growth (%)", f"{portfolio_earnings_growth:.2f}%", "Earnings growth rate weighted by portfolio allocation"])
        if portfolio_eps_growth is not None and not pd.isna(portfolio_eps_growth):
            summary_data.append(["Growth", "EPS Growth (%)", f"{portfolio_eps_growth:.2f}%", "Earnings per share growth rate weighted by portfolio allocation"])
        
        # Dividend metrics
        if portfolio_dividend_yield is not None and not pd.isna(portfolio_dividend_yield):
            summary_data.append(["Dividends", "Dividend Yield (%)", f"{portfolio_dividend_yield:.2f}%", "Dividend yield weighted by portfolio allocation"])
        if portfolio_payout_ratio is not None and not pd.isna(portfolio_payout_ratio):
            summary_data.append(["Dividends", "Payout Ratio (%)", f"{portfolio_payout_ratio:.2f}%", "Dividend payout ratio weighted by portfolio allocation"])
        
        # Size metrics
        if portfolio_market_cap is not None and not pd.isna(portfolio_market_cap):
            summary_data.append(["Size", "Market Cap ($B)", f"${portfolio_market_cap:.2f}B", "Market capitalization weighted by portfolio allocation"])
        if portfolio_enterprise_value is not None and not pd.isna(portfolio_enterprise_value):
            summary_data.append(["Size", "Enterprise Value ($B)", f"${portfolio_enterprise_value:.2f}B", "Enterprise value weighted by portfolio allocation"])
        
        if summary_data:
            # Create summary table
            summary_headers = ['Category', 'Metric', 'Value', 'Description']
            summary_table = Table([summary_headers] + summary_data, colWidths=[0.8*inch, 1.2*inch, 0.8*inch, 3.2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('WORDWRAP', (0, 0), (-1, -1), True),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 15))
            
            # Add interpretation
            story.append(Paragraph("Portfolio Interpretation:", subheading_style))
            story.append(Spacer(1, 10))
            
            if portfolio_beta is not None and not pd.isna(portfolio_beta):
                if portfolio_beta < 0.8:
                    story.append(Paragraph(f"• Low Risk Portfolio - Beta {portfolio_beta:.2f} indicates lower volatility than market", styles['Normal']))
                elif portfolio_beta < 1.2:
                    story.append(Paragraph(f"• Moderate Risk Portfolio - Beta {portfolio_beta:.2f} indicates market-average volatility", styles['Normal']))
                else:
                    story.append(Paragraph(f"• High Risk Portfolio - Beta {portfolio_beta:.2f} indicates higher volatility than market", styles['Normal']))
            
            if portfolio_pe is not None and not pd.isna(portfolio_pe):
                if portfolio_pe < 15:
                    story.append(Paragraph(f"• Undervalued Portfolio - P/E {portfolio_pe:.2f} suggests attractive valuations", styles['Normal']))
                elif portfolio_pe < 25:
                    story.append(Paragraph(f"• Fairly Valued Portfolio - P/E {portfolio_pe:.2f} suggests reasonable valuations", styles['Normal']))
                else:
                    story.append(Paragraph(f"• Potentially Overvalued Portfolio - P/E {portfolio_pe:.2f} suggests high valuations", styles['Normal']))
        else:
            story.append(Paragraph("No portfolio-weighted metrics available for display.", styles['Normal']))
        
        # Update progress
        progress_bar.progress(80)
        status_text.text("🏢 Adding portfolio composition analysis...")
        
        # SECTION 4: Portfolio Composition Analysis
        story.append(PageBreak())
        story.append(Paragraph("4. Portfolio Composition Analysis", heading_style))
        story.append(Spacer(1, 15))
        
        # Get sector and industry data from session state (these are calculated and stored in the main UI)
        sector_data = getattr(st.session_state, 'sector_data', pd.Series(dtype=float))
        industry_data = getattr(st.session_state, 'industry_data', pd.Series(dtype=float))
        
        # Sector Allocation
        if not sector_data.empty:
            story.append(Paragraph("Sector Allocation", subheading_style))
            story.append(Spacer(1, 10))
            
            # Create sector table
            sector_table_data = [['Sector', 'Allocation (%)']]
            for sector, allocation in sector_data.items():
                sector_table_data.append([sector, f"{allocation:.2f}%"])
            
            sector_table = Table(sector_table_data, colWidths=[3.0*inch, 1.5*inch])
            sector_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
            ]))
            story.append(sector_table)
            story.append(Spacer(1, 15))
        
        # Industry Allocation
        if not industry_data.empty:
            story.append(Paragraph("Industry Allocation", subheading_style))
            story.append(Spacer(1, 10))
            
            # Create industry table
            industry_table_data = [['Industry', 'Allocation (%)']]
            for industry, allocation in industry_data.items():
                industry_table_data.append([industry, f"{allocation:.2f}%"])
            
            industry_table = Table(industry_table_data, colWidths=[3.0*inch, 1.5*inch])
            industry_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
            ]))
            story.append(industry_table)
            story.append(Spacer(1, 15))
        
        # Add page break and create charts page
        if not sector_data.empty or not industry_data.empty:
            story.append(PageBreak())
            story.append(Paragraph("Portfolio Distribution Charts", heading_style))
            story.append(Spacer(1, 15))
            
            # Check if we actually have any allocation data to display
            has_sector_data = not sector_data.empty and len(sector_data) > 0 and sector_data.sum() > 0
            has_industry_data = not industry_data.empty and len(industry_data) > 0 and industry_data.sum() > 0
            
            if not has_sector_data and not has_industry_data:
                # Portfolio is all cash - show message instead of charts
                story.append(Paragraph("This portfolio is currently allocated 100% to cash. No sector or industry distribution charts are available.", styles['Normal']))
                story.append(Spacer(1, 15))
            else:
                # Create combined figure with both pie charts one above the other
                try:
                    # Create figure with two subplots - square subplots to ensure circular pie charts
                    fig, (ax_sector, ax_industry) = plt.subplots(2, 1, figsize=(12, 12))
                
                    # Define consistent color palette for both charts
                    consistent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    
                    # Sector pie chart
                    if has_sector_data:
                        sectors = sector_data.index.tolist()
                        allocations = sector_data.values.tolist()
                        
                        # Create pie chart with percentage labels but only for larger slices to avoid overlap
                        def make_autopct(values):
                            def my_autopct(pct):
                                if pd.isna(pct) or pct is None:
                                    return ''
                                total = sum(values)
                                if pd.isna(total) or total == 0:
                                    return ''
                                val = int(round(pct*total/100.0))
                                # Only show percentage if slice is large enough (>5%)
                                return f'{pct:.1f}%' if pct > 5 else ''
                            return my_autopct
                        
                        wedges_sector, texts_sector, autotexts_sector = ax_sector.pie(allocations, autopct=make_autopct(allocations), 
                                                                                     startangle=90, textprops={'fontsize': 10},
                                                                                     colors=consistent_colors[:len(allocations)])
                        
                        # Create legend with percentages - positioned further to the right
                        legend_labels = [f"{sector} ({alloc:.1f}%)" for sector, alloc in zip(sectors, allocations)]
                        ax_sector.legend(wedges_sector, legend_labels, title="Sectors", loc="center left", bbox_to_anchor=(1.05, 0, 0.5, 1), fontsize=10)
                        
                        # Wrap long titles to prevent them from going out of bounds
                        title_text = f'Sector Allocation - {portfolio_name}'
                        # Use textwrap for proper word-based wrapping
                        import textwrap
                        wrapped_title = textwrap.fill(title_text, width=40, break_long_words=True, break_on_hyphens=False)
                        ax_sector.set_title(wrapped_title, fontsize=14, fontweight='bold')
                        # Force perfectly circular shape
                        ax_sector.set_aspect('equal')
                        ax_sector.set_xlim(-1.2, 1.2)
                        ax_sector.set_ylim(-1.2, 1.2)
                    else:
                        # No sector data - show placeholder
                        ax_sector.text(0.5, 0.5, 'No sector data available', 
                                     horizontalalignment='center', verticalalignment='center', 
                                     transform=ax_sector.transAxes, fontsize=12)
                        # Wrap long titles to prevent them from going out of bounds
                        title_text = f'Sector Allocation - {portfolio_name}'
                        # Use textwrap for proper word-based wrapping
                        import textwrap
                        wrapped_title = textwrap.fill(title_text, width=40, break_long_words=True, break_on_hyphens=False)
                        ax_sector.set_title(wrapped_title, fontsize=14, fontweight='bold')
                    
                    # Industry pie chart
                    if has_industry_data:
                        industries = industry_data.index.tolist()
                        allocations = industry_data.values.tolist()
                        
                        # Create pie chart with percentage labels but only for larger slices to avoid overlap
                        def make_autopct(values):
                            def my_autopct(pct):
                                if pd.isna(pct) or pct is None:
                                    return ''
                                total = sum(values)
                                if pd.isna(total) or total == 0:
                                    return ''
                                val = int(round(pct*total/100.0))
                                # Only show percentage if slice is large enough (>5%)
                                return f'{pct:.1f}%' if pct > 5 else ''
                            return my_autopct
                        
                        wedges_industry, texts_industry, autotexts_industry = ax_industry.pie(allocations, autopct=make_autopct(allocations), 
                                                                                             startangle=90, textprops={'fontsize': 10},
                                                                                             colors=consistent_colors[:len(allocations)])
                        
                        # Create legend with percentages - positioned further to the right
                        legend_labels = [f"{industry} ({alloc:.1f}%)" for industry, alloc in zip(industries, allocations)]
                        ax_industry.legend(wedges_industry, legend_labels, title="Industries", loc="center left", bbox_to_anchor=(1.05, 0, 0.5, 1), fontsize=10)
                        
                        # Wrap long titles to prevent them from going out of bounds
                        title_text = f'Industry Allocation - {portfolio_name}'
                        # Use textwrap for proper word-based wrapping
                        import textwrap
                        wrapped_title = textwrap.fill(title_text, width=40, break_long_words=True, break_on_hyphens=False)
                        ax_industry.set_title(wrapped_title, fontsize=14, fontweight='bold')
                        # Force perfectly circular shape
                        ax_industry.set_aspect('equal')
                        ax_industry.set_xlim(-1.2, 1.2)
                        ax_industry.set_ylim(-1.2, 1.2)
                    else:
                        # No industry data - show placeholder
                        ax_industry.text(0.5, 0.5, 'No industry data available', 
                                       horizontalalignment='center', verticalalignment='center', 
                                       transform=ax_industry.transAxes, fontsize=12)
                        # Wrap long titles to prevent them from going out of bounds
                        title_text = f'Industry Allocation - {portfolio_name}'
                        # Use textwrap for proper word-based wrapping
                        import textwrap
                        wrapped_title = textwrap.fill(title_text, width=40, break_long_words=True, break_on_hyphens=False)
                        ax_industry.set_title(wrapped_title, fontsize=14, fontweight='bold')
                    
                    # Adjust layout to maintain circular shapes and accommodate legends
                    plt.subplots_adjust(hspace=0.4, left=0.1, right=0.7, top=0.95, bottom=0.05)
                    
                    # Save to buffer - don't use bbox_inches='tight' to preserve aspect ratio
                    combined_img_buffer = io.BytesIO()
                    fig.savefig(combined_img_buffer, format='png', dpi=300, facecolor='white')
                    combined_img_buffer.seek(0)
                    plt.close(fig)
                    
                    # Add to PDF - maintain square aspect ratio for circular charts
                    story.append(Image(combined_img_buffer, width=8*inch, height=8*inch))
                    story.append(Spacer(1, 15))
                    
                except Exception as e:
                    story.append(Paragraph(f"Error creating charts: {str(e)}", styles['Normal']))
        
        # Add page break and move Portfolio Risk Metrics Summary to next page
        story.append(PageBreak())
        story.append(Paragraph("Portfolio Risk Metrics Summary", subheading_style))
        story.append(Spacer(1, 10))
        
        risk_metrics_data = []
        
        # Beta
        if portfolio_beta is not None and not pd.isna(portfolio_beta):
            if portfolio_beta < 0.8:
                beta_risk = "Low Risk"
            elif portfolio_beta < 1.2:
                beta_risk = "Balanced Risk"
            elif portfolio_beta < 1.5:
                beta_risk = "Moderate Risk"
            else:
                beta_risk = "High Risk"
            risk_metrics_data.append(["Portfolio Risk Level", beta_risk, f"Beta: {portfolio_beta:.2f}"])
        else:
            risk_metrics_data.append(["Portfolio Risk Level", "NA", "Beta: NA"])
        
        # P/E
        if portfolio_pe is not None and not pd.isna(portfolio_pe):
            if portfolio_pe < 15:
                pe_rating = "Undervalued"
            elif portfolio_pe < 25:
                pe_rating = "Fair Value"
            elif portfolio_pe < 35:
                pe_rating = "Expensive"
            else:
                pe_rating = "Overvalued"
            risk_metrics_data.append(["Current P/E Rating", pe_rating, f"P/E: {portfolio_pe:.2f}"])
        else:
            risk_metrics_data.append(["Current P/E Rating", "NA", "P/E: NA"])
        
        # Forward P/E
        if portfolio_forward_pe is not None and not pd.isna(portfolio_forward_pe):
            if portfolio_forward_pe < 15:
                fpe_rating = "Undervalued"
            elif portfolio_forward_pe < 25:
                fpe_rating = "Fair Value"
            elif portfolio_forward_pe < 35:
                fpe_rating = "Expensive"
            else:
                fpe_rating = "Overvalued"
            risk_metrics_data.append(["Forward P/E Rating", fpe_rating, f"Forward P/E: {portfolio_forward_pe:.2f}"])
        else:
            risk_metrics_data.append(["Forward P/E Rating", "NA", "Forward P/E: NA"])
        
        # Dividend
        if portfolio_dividend_yield is not None and not pd.isna(portfolio_dividend_yield):
            if portfolio_dividend_yield > 5:
                div_rating = "Very High Yield"
            elif portfolio_dividend_yield > 3:
                div_rating = "Good Yield"
            elif portfolio_dividend_yield > 1.5:
                div_rating = "Moderate Yield"
            else:
                div_rating = "Low Yield"
            risk_metrics_data.append(["Dividend Rating", div_rating, f"Yield: {portfolio_dividend_yield:.2f}%"])
        else:
            risk_metrics_data.append(["Dividend Rating", "NA", "Yield: NA"])
        
        if risk_metrics_data:
            risk_headers = ['Metric', 'Rating', 'Value']
            risk_table = Table([risk_headers] + risk_metrics_data, colWidths=[2.0*inch, 1.5*inch, 1.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
            ]))
            story.append(risk_table)
        
        # Add page break before detailed financial indicators
        story.append(PageBreak())
        
        # Add detailed financial indicators tables covering all 5 sections
        if hasattr(st.session_state, 'sector_data') and not st.session_state.sector_data.empty:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Detailed Financial Indicators for Each Position", heading_style))
            story.append(Spacer(1, 10))
            
            # Get the comprehensive portfolio data from session state
            try:
                if hasattr(st.session_state, 'df_comprehensive'):
                    df_comprehensive = st.session_state.df_comprehensive
                    
                    # Helper function to create wrapped text for PDF with enhanced company name, sector, and industry handling
                    def wrap_text_for_pdf(text, max_length=25):
                        """Wrap long text to fit in PDF cells with intelligent line breaks for company names, sectors, and industries"""
                        if pd.isna(text) or text is None:
                            return 'N/A'
                        text_str = str(text)
                        if len(text_str) <= max_length:
                            return text_str
                        
                        # For company names, sector, and industry, prioritize showing the full name with line breaks
                        if 'Name' in str(text) or 'Sector' in str(text) or 'Industry' in str(text) or len(text_str) > max_length * 1.2:
                            # Try to break at spaces first for better readability
                            words = text_str.split()
                            if len(words) > 1:
                                # Find the best break point that maximizes text visibility
                                best_break = 0
                                for i, word in enumerate(words):
                                    if len(' '.join(words[:i+1])) <= max_length:
                                        best_break = i + 1
                                    else:
                                        break
                                
                                if best_break > 0:
                                    # Use line break for better PDF formatting
                                    first_line = ' '.join(words[:best_break])
                                    second_line = ' '.join(words[best_break:])
                                    
                                    # For company names, sector, and industry, allow longer second lines to show more text
                                    if len(second_line) > max_length:
                                        # Instead of truncating, try to break again
                                        second_words = second_line.split()
                                        if len(second_words) > 1:
                                            # Find another break point in the second line
                                            second_break = 0
                                            for j, word in enumerate(second_words):
                                                if len(' '.join(second_words[:j+1])) <= max_length:
                                                    second_break = j + 1
                                                else:
                                                    break
                                            if second_break > 0:
                                                second_line = ' '.join(second_words[:second_break])
                                                third_line = ' '.join(second_words[second_break:])
                                                if len(third_line) > max_length:
                                                    third_line = third_line[:max_length-3] + '...'
                                                return first_line + '\n' + second_line + '\n' + third_line
                                            else:
                                                # If we can't break further, truncate the second line
                                                second_line = second_line[:max_length-3] + '...'
                                        else:
                                            # Single long word in second line, truncate
                                            second_line = second_line[:max_length-3] + '...'
                                    
                                    return first_line + '\n' + second_line
                                else:
                                    # If we can't break at words, truncate with ellipsis
                                    return text_str[:max_length-3] + '...'
                        
                        # For other columns, try to break at spaces or special characters
                        words = text_str.split()
                        if len(words) == 1:
                            # Single long word, break at max_length
                            return text_str[:max_length-3] + '...'
                        # Try to fit as many words as possible
                        result = ''
                        for word in words:
                            if len(result + ' ' + word) <= max_length:
                                result += (' ' + word) if result else word
                            else:
                                break
                        if not result:
                            result = text_str[:max_length-3] + '...'
                        return result
                    
                    # Helper function to create a focused table
                    def create_focused_table(title, columns, data_subset, col_widths):
                        """Create a focused table with specific columns"""
                        story.append(Paragraph(f"<b>{title}</b>", styles['Normal']))
                        story.append(Spacer(1, 5))
                        
                        # Filter dataframe to only include available columns
                        available_columns = [col for col in columns if col in data_subset.columns]
                        if not available_columns:
                            return
                        
                        df_subset = data_subset[available_columns].copy()
                        
                        # Convert to list format for PDF table with text wrapping
                        pdf_data = [available_columns]  # Headers
                        for _, row in df_subset.iterrows():
                            pdf_row = []
                            for col in available_columns:
                                value = row[col]
                                if pd.isna(value) or value is None:
                                    pdf_row.append('N/A')
                                else:
                                    # Apply text wrapping based on column type
                                    if 'Name' in col:
                                        pdf_row.append(wrap_text_for_pdf(value, 22))  # Increased for better company name display
                                    elif 'Sector' in col or 'Industry' in col:
                                        pdf_row.append(wrap_text_for_pdf(value, 25))  # Increased for better sector/industry display
                                    elif 'Ticker' in col:
                                        pdf_row.append(wrap_text_for_pdf(value, 8))
                                    else:
                                        pdf_row.append(wrap_text_for_pdf(value, 15))
                            pdf_data.append(pdf_row)
                        
                        # Create table with specified column widths and enhanced styling for text wrapping
                        table = Table(pdf_data, colWidths=col_widths[:len(available_columns)])
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, -1), 6),  # Very small font to fit more data
                            ('GRID', (0, 0), (-1, -1), 0.5, reportlab_colors.grey),
                            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [reportlab_colors.Color(0.98, 0.98, 0.98), reportlab_colors.Color(0.95, 0.95, 0.95)]),
                            ('WORDWRAP', (0, 0), (-1, -1), True),  # Enable word wrapping for all cells
                            ('LEFTPADDING', (0, 0), (-1, -1), 3),  # Add some padding for wrapped text
                            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                            ('MINIMUMHEIGHT', (0, 0), (-1, -1), 15)  # Ensure minimum height for wrapped text, especially for company names
                        ]))
                        
                        story.append(table)
                        story.append(Spacer(1, 10))
                    
                    # Section 1: Overview & Basic Info - Company Name column optimized to 1.5" for balance
                    # Industry column optimized to 1.1" to prevent truncation while ensuring table fits within page
                    # Enhanced text wrapping for Company Name, Sector, and Industry columns to ensure full text visibility
                    # Total table width: 7.1 inches (ensures small margin on each side of 8.5" page)
                    overview_cols = ['Ticker', 'Company Name', 'Sector', 'Industry', 'Current Price ($)', 'Allocation %', 'Shares', 'Total Value ($)', '% of Portfolio']
                    overview_widths = [0.6*inch, 1.5*inch, 1.0*inch, 1.1*inch, 0.8*inch, 0.7*inch, 0.6*inch, 1.0*inch, 0.8*inch]
                    create_focused_table("📊 Overview & Basic Information", overview_cols, df_comprehensive, overview_widths)
                    
                    # Section 2: Valuation Metrics
                    valuation_cols = ['Ticker', 'Market Cap ($B)', 'Enterprise Value ($B)', 'P/E Ratio', 'Forward P/E', 'PEG Ratio', 'Price/Book', 'Price/Sales', 'EV/EBITDA']
                    valuation_widths = [0.6*inch, 0.8*inch, 1.0*inch, 0.6*inch, 0.7*inch, 0.6*inch, 0.7*inch, 0.7*inch, 0.7*inch]
                    create_focused_table("💰 Valuation Metrics", valuation_cols, df_comprehensive, valuation_widths)
                    
                    # Section 3: Financial Health - Increased column widths to prevent title overlap
                    health_cols = ['Ticker', 'Debt/Equity', 'Current Ratio', 'Quick Ratio', 'ROE (%)', 'ROA (%)', 'ROIC (%)', 'Profit Margin (%)', 'Operating Margin (%)']
                    health_widths = [0.6*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.0*inch, 1.0*inch]
                    create_focused_table("🏥 Financial Health", health_cols, df_comprehensive, health_widths)
                    
                    # Section 4: Growth & Dividends - Increased column widths to prevent title overlap
                    growth_cols = ['Ticker', 'Revenue Growth (%)', 'Earnings Growth (%)', 'EPS Growth (%)', 'Dividend Yield (%)', 'Dividend Rate ($)', 'Payout Ratio (%)', '5Y Dividend Growth (%)']
                    growth_widths = [0.6*inch, 1.0*inch, 1.0*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.8*inch, 1.0*inch]
                    create_focused_table("📈 Growth & Dividends", growth_cols, df_comprehensive, growth_widths)
                    
                    # Section 5: Technical & Trading
                    technical_cols = ['Ticker', '52W High ($)', '52W Low ($)', '50D MA ($)', '200D MA ($)', 'Beta', 'Volume', 'Avg Volume', 'Analyst Rating']
                    technical_widths = [0.6*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.5*inch, 0.7*inch, 0.7*inch, 0.7*inch]
                    create_focused_table("📊 Technical & Trading", technical_cols, df_comprehensive, technical_widths)
                    
                    story.append(Spacer(1, 10))
                    story.append(Paragraph("Note: This comprehensive analysis covers all 5 sections (Overview, Valuation, Financial Health, Growth & Dividends, Technical) with the most important metrics for each position. For complete data and interactive analysis, run the allocation analysis in the Streamlit interface.", styles['Normal']))
                    
                else:
                    # Fallback: create basic table from current allocations
                    basic_data = []
                    for ticker, alloc_pct in alloc_dict.items():
                        if ticker != 'CASH':
                            basic_data.append([
                                ticker,
                                f"{alloc_pct * 100:.2f}%",
                                f"${portfolio_value * alloc_pct:,.2f}",
                                f"{(portfolio_value * alloc_pct / portfolio_value * 100):.2f}%"
                            ])
                    
                    if basic_data:
                        # Create basic table headers
                        basic_headers = ['Ticker', 'Allocation %', 'Total Value ($)', '% of Portfolio']
                        basic_table = Table([basic_headers] + basic_data, colWidths=[1.5*inch, 1.5*inch, 2.0*inch, 1.5*inch])
                        basic_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, -1), 8),
                            ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
                        ]))
                        story.append(basic_table)
                        story.append(Spacer(1, 10))
                        story.append(Paragraph("Note: Basic allocation data shown. For comprehensive financial indicators, run the allocation analysis in the Streamlit interface.", styles['Normal']))
            
            except Exception as e:
                story.append(Paragraph(f"Note: Detailed financial indicators table could not be generated: {str(e)}", styles['Normal']))
        
        # Update progress
        progress_bar.progress(90)
        status_text.text("💾 Finalizing PDF...")
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Update progress
        progress_bar.progress(100)
        status_text.text("✅ PDF generated successfully!")
        
        # Store PDF data in session state for download button
        st.session_state['pdf_buffer'] = pdf_data
        
        return True
        
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return False

# -----------------------
# MA Filter Functions - COPIED FROM PAGE 1
# -----------------------
def calculate_ema(df, window):
    """
    Calculate Exponential Moving Average for a given window.
    
    Args:
        df: DataFrame with 'Close' column
        window: Number of periods for EMA calculation
        
    Returns:
        Series with EMA values
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    if 'Close' not in df.columns:
        return None
    # EMA uses standard formula: multiplier = 2 / (window + 1)
    return df['Close'].ewm(span=window, adjust=False, min_periods=window).mean()

def calculate_sma(df, window):
    """
    Calculate Simple Moving Average for a given window.
    
    Args:
        df: DataFrame with 'Close' column
        window: Number of periods for SMA calculation
        
    Returns:
        Series with SMA values
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    if 'Close' not in df.columns:
        return None
    return df['Close'].rolling(window=window, min_periods=window).mean()

def filter_assets_by_ma(valid_assets, reindexed_data, date, ma_window, ma_type='SMA', config=None, stocks_config=None):
    """
    Filter out assets that are below their Moving Average (SMA or EMA).
    Now supports using a different ticker's MA as reference!
    
    Args:
        valid_assets: List of tickers to filter
        reindexed_data: Dict of ticker -> DataFrame
        date: Current date for filtering
        ma_window: MA window in days (e.g., 200 for 200-day MA)
        ma_type: Type of moving average - 'SMA' or 'EMA'
        config: Optional config dict
        stocks_config: Optional stocks config for include_in_sma_filter and ma_reference_ticker settings
        
    Returns:
        filtered_assets: List of tickers above their MA
        excluded_assets: Dict of excluded tickers with reasons
    """
    filtered_assets = []
    excluded_assets = {}
    tickers_with_enough_data = []
    
    # Create mappings from stocks_config
    include_in_ma = {}
    ma_reference = {}
    if stocks_config:
        for stock in stocks_config:
            ticker = stock.get('ticker')
            if ticker:
                include_in_ma[ticker] = stock.get('include_in_sma_filter', True)
                # Get MA reference ticker (empty or None means use ticker itself)
                ref = stock.get('ma_reference_ticker', '').strip()
                # Apply same transformations as regular tickers for consistency
                if ref:
                    ref = ref.replace(",", ".").upper()
                    # Special conversion for Berkshire Hathaway
                    if ref == 'BRK.B':
                        ref = 'BRK-B'
                    elif ref == 'BRK.A':
                        ref = 'BRK-A'
                    # Resolve alias (e.g., TLTTR -> TLT_COMPLETE, GOLDX -> GOLD_COMPLETE)
                    ref = resolve_ticker_alias(ref)
                ma_reference[ticker] = ref if ref else ticker
    
    for ticker in valid_assets:
        is_included = include_in_ma.get(ticker, True)
        # Check if this ticker should be excluded from MA filter (not included)
        if not is_included:
            filtered_assets.append(ticker)
            continue
            
        # Get MA reference ticker (default to self)
        reference_ticker = ma_reference.get(ticker, ticker)
        
        # Get ticker's price data
        df = reindexed_data.get(ticker)
        if df is None or not isinstance(df, pd.DataFrame):
            continue
        
        # Get reference ticker's data for MA calculation
        df_ref = reindexed_data.get(reference_ticker)
        if df_ref is None or not isinstance(df_ref, pd.DataFrame):
            # Reference ticker not available, fallback to using ticker itself
            df_ref = df
            reference_ticker = ticker
        
        # Get data up to current date
        df_up_to_date = df[df.index <= date]
        df_ref_up_to_date = df_ref[df_ref.index <= date]
        
        if len(df_ref_up_to_date) < ma_window:
            # Not enough data to calculate MA on reference, include by default (no filter)
            filtered_assets.append(ticker)
            continue
        
        # Mark that this ticker has enough data for MA calculation
        tickers_with_enough_data.append(ticker)
        
        # Calculate MA on the REFERENCE ticker
        if ma_type == 'EMA':
            ma = calculate_ema(df_ref_up_to_date, ma_window)
        else:  # Default to SMA
            ma = calculate_sma(df_ref_up_to_date, ma_window)
            
        if ma is None:
            filtered_assets.append(ticker)
            continue
        
        try:
            # Get current price of REFERENCE TICKER and MA value of REFERENCE
            # CRITICAL: Use reference ticker's price, not the ticker itself!
            # Compare: reference price vs reference MA (same price scale)
            if len(df_ref_up_to_date) == 0:
                filtered_assets.append(ticker)
                continue
            current_price = df_ref_up_to_date['Close'].iloc[-1]
            current_ma = ma.iloc[-1] if hasattr(ma, 'iloc') else ma.loc[date]
            
            # Include only if REFERENCE ticker's price is above REFERENCE ticker's MA
            if current_price >= current_ma:
                filtered_assets.append(ticker)
            else:
                ref_label = f" (using {reference_ticker})" if reference_ticker != ticker else ""
                excluded_assets[ticker] = f"Below {ma_window}-day {ma_type}{ref_label} ({current_price:.2f} < {current_ma:.2f})"
        except:
            # If any error, include by default
            filtered_assets.append(ticker)
    
    # SIMPLE LOGIC: Return whatever assets are still active (above MA or excluded from filter)
    # NO "go to cash" logic - that's handled in the rebalancing
    return filtered_assets, excluded_assets

# -----------------------
# Single-backtest core (adapted from your code, robust)
# -----------------------
def single_backtest(config, sim_index, reindexed_data):
    print(f"[THRESHOLD DEBUG] single_backtest called for portfolio: {config.get('name', 'Unknown')}")
    stocks_list = config.get('stocks', [])
    raw_tickers = [s.get('ticker') for s in stocks_list if s.get('ticker')]
    # Filter out tickers not present in reindexed_data to avoid crashes for invalid tickers
    if reindexed_data:
        tickers = [t for t in raw_tickers if t in reindexed_data]
    else:
        tickers = raw_tickers[:]
    missing_tickers = [t for t in raw_tickers if t not in tickers]
    if missing_tickers:
        # Log a warning and ignore unknown tickers
        print(f"[ALLOC WARN] Ignoring unknown or missing tickers: {missing_tickers}")
    # Handle duplicate tickers by summing their allocations
    allocations = {}
    include_dividends = {}
    for s in stocks_list:
        if s.get('ticker') and s.get('ticker') in tickers:
            ticker = s.get('ticker')
            allocation = s.get('allocation', 0)
            include_div = s.get('include_dividends', False)
            
            if ticker in allocations:
                # If ticker already exists, add the allocation
                allocations[ticker] += allocation
                # For include_dividends, use True if any instance has it True
                include_dividends[ticker] = include_dividends[ticker] or include_div
            else:
                # First occurrence of this ticker
                allocations[ticker] = allocation
                include_dividends[ticker] = include_div
    
    # Update tickers to only include unique tickers after deduplication
    tickers = list(allocations.keys())
    
    # Apply threshold filters to initial allocations for non-momentum strategies
    use_momentum = config.get('use_momentum', True)
    print(f"[THRESHOLD DEBUG] Portfolio: {config.get('name', 'Unknown')}, use_momentum: {use_momentum}, allocations: {allocations}")
    
    if not use_momentum and allocations:
        # Apply MA filter first (for non-momentum strategies) - COPIED FROM PAGE 1
        if config.get('use_sma_filter', False):
            print(f"🔍 DEBUG MA FILTER PAGE 2: Portfolio {config.get('name', 'Unknown')} - MA filter enabled")
            ma_window = config.get('sma_window', 200)
            ma_type = config.get('ma_type', 'SMA')
            # Get list of current tickers (excluding CASH)
            current_tickers = [t for t in tickers if t != 'CASH']
            print(f"🔍 DEBUG MA FILTER PAGE 2: Current tickers: {current_tickers}")
            print(f"🔍 DEBUG MA FILTER PAGE 2: Before filtering - allocations: {allocations}")
            
            # Use the simulation start date for MA filtering (like page 1)
            filtered_tickers, excluded_assets = filter_assets_by_ma(current_tickers, reindexed_data, sim_index[0], ma_window, ma_type, config, config.get('stocks', []))
            print(f"🔍 DEBUG MA FILTER PAGE 2: After filtering - filtered_tickers: {filtered_tickers}, excluded_assets: {excluded_assets}")
            
            # Redistribute allocations of excluded tickers proportionally among remaining tickers - EXACTLY LIKE PAGE 1
            if excluded_assets:
                excluded_ticker_list = list(excluded_assets.keys())
                
                # Calculate total allocation of excluded tickers
                excluded_allocation = sum(allocations.get(t, 0) for t in excluded_ticker_list)
                
                # Remove excluded tickers from current allocations
                for excluded_ticker in excluded_ticker_list:
                    if excluded_ticker in allocations:
                        del allocations[excluded_ticker]
                
                # If there are remaining tickers (excluding CASH), redistribute proportionally - EXACTLY LIKE PAGE 1
                remaining_tickers = [t for t in allocations.keys() if t != 'CASH']
                if remaining_tickers:
                    # Calculate total allocation of remaining tickers (excluding CASH)
                    remaining_allocation = sum(allocations.get(t, 0) for t in remaining_tickers)
                    
                    if remaining_allocation > 0:
                        # Redistribute excluded allocation proportionally
                        for ticker in remaining_tickers:
                            proportion = allocations[ticker] / remaining_allocation
                            allocations[ticker] += excluded_allocation * proportion
                    else:
                        # Equal distribution if no remaining allocation
                        equal_allocation = excluded_allocation / len(remaining_tickers)
                        for ticker in remaining_tickers:
                            allocations[ticker] = equal_allocation
                else:
                    # No remaining tickers, all goes to CASH
                    allocations = {'CASH': 1.0}
            
            print(f"🔍 DEBUG MA FILTER PAGE 2: After redistribution - allocations: {allocations}")
        
        use_max_allocation = config.get('use_max_allocation', False)
        max_allocation_percent = config.get('max_allocation_percent', 10.0)
        use_threshold = config.get('use_minimal_threshold', False)
        threshold_percent = config.get('minimal_threshold_percent', 2.0)
        
        # Build dictionary of individual ticker caps from stock configs
        individual_caps = {}
        for stock in config.get('stocks', []):
            ticker = stock.get('ticker', '')
            individual_cap = stock.get('max_allocation_percent', None)
            if individual_cap is not None and individual_cap > 0:
                individual_caps[ticker] = individual_cap / 100.0
        
        # Debug output
        print(f"[THRESHOLD DEBUG] Non-momentum portfolio: use_threshold={use_threshold}, threshold_percent={threshold_percent}, use_max_allocation={use_max_allocation}, max_allocation_percent={max_allocation_percent}")
        print(f"[THRESHOLD DEBUG] Individual caps: {individual_caps}")
        print(f"[THRESHOLD DEBUG] Original allocations: {allocations}")
        
        # Apply allocation filters in correct order: Max Allocation -> Min Threshold -> Max Allocation (two-pass system)
        if (use_max_allocation or individual_caps):
            max_allocation_decimal = max_allocation_percent / 100.0
            
            # FIRST PASS: Apply maximum allocation filter (EXCLUDE CASH from max_allocation limit)
            capped_allocations = {}
            excess_allocation = 0.0
            
            for ticker, allocation in allocations.items():
                # CASH is exempt from max_allocation limit to prevent money loss
                if ticker == 'CASH':
                    capped_allocations[ticker] = allocation
                else:
                    # Use individual cap if available, otherwise use global cap
                    ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                    
                    if allocation > ticker_cap:
                        # Cap the allocation and collect excess
                        capped_allocations[ticker] = ticker_cap
                        excess_allocation += (allocation - ticker_cap)
                    else:
                        capped_allocations[ticker] = allocation
            
            # Redistribute excess allocation proportionally to stocks below the cap
            if excess_allocation > 0:
                # Find stocks below the cap (include CASH as eligible for redistribution)
                below_cap_stocks = {}
                for ticker, allocation in capped_allocations.items():
                    if ticker == 'CASH':
                        below_cap_stocks[ticker] = allocation
                    else:
                        ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                        if allocation < ticker_cap:
                            below_cap_stocks[ticker] = allocation
                
                if below_cap_stocks:
                    total_below_cap = sum(below_cap_stocks.values())
                    if total_below_cap > 0:
                        # Redistribute excess proportionally
                        for ticker in below_cap_stocks:
                            proportion = below_cap_stocks[ticker] / total_below_cap
                            new_allocation = capped_allocations[ticker] + (excess_allocation * proportion)
                            # CASH can receive unlimited allocation, other stocks are capped
                            if ticker == 'CASH':
                                capped_allocations[ticker] = new_allocation
                            else:
                                ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                                capped_allocations[ticker] = min(new_allocation, ticker_cap)
            
            allocations = capped_allocations
        
        # Apply minimal threshold filter
        if use_threshold:
            threshold_decimal = threshold_percent / 100.0
            
            # First: Filter out stocks below threshold
            filtered_allocations = {}
            for ticker, allocation in allocations.items():
                if allocation >= threshold_decimal:
                    # Keep stocks above or equal to threshold
                    filtered_allocations[ticker] = allocation
            
            # Then: Normalize remaining stocks to sum to 1
            if filtered_allocations:
                total_allocation = sum(filtered_allocations.values())
                if total_allocation > 0:
                    allocations = {ticker: allocation / total_allocation for ticker, allocation in filtered_allocations.items()}
                else:
                    allocations = {}
            else:
                # If no stocks meet threshold, keep original allocations
                pass  # allocations remain unchanged
        
        # SECOND PASS: Apply maximum allocation filter again (in case normalization created new excess)
        # Run if global cap is enabled OR any individual caps exist (parity with Page 1)
        if (use_max_allocation or individual_caps):
            max_allocation_decimal = max_allocation_percent / 100.0
            
            # Check if any stocks exceed the cap after threshold filtering and normalization (EXCLUDE CASH)
            capped_allocations = {}
            excess_allocation = 0.0
            
            for ticker, allocation in allocations.items():
                # CASH is exempt from max_allocation limit to prevent money loss
                if ticker == 'CASH':
                    capped_allocations[ticker] = allocation
                else:
                    # Use individual cap if available, otherwise use global cap
                    ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                    
                    if allocation > ticker_cap:
                        # Cap the allocation and collect excess
                        capped_allocations[ticker] = ticker_cap
                        excess_allocation += (allocation - ticker_cap)
                    else:
                        capped_allocations[ticker] = allocation
            
            # Redistribute excess allocation proportionally to stocks below the cap
            if excess_allocation > 0:
                # Find stocks below the cap (include CASH as eligible for redistribution)
                below_cap_stocks = {}
                for ticker, allocation in capped_allocations.items():
                    if ticker == 'CASH':
                        below_cap_stocks[ticker] = allocation
                    else:
                        ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                        if allocation < ticker_cap:
                            below_cap_stocks[ticker] = allocation
                
                if below_cap_stocks:
                    total_below_cap = sum(below_cap_stocks.values())
                    if total_below_cap > 0:
                        # Redistribute excess proportionally
                        for ticker in below_cap_stocks:
                            proportion = below_cap_stocks[ticker] / total_below_cap
                            new_allocation = capped_allocations[ticker] + (excess_allocation * proportion)
                            # CASH can receive unlimited allocation, other stocks are capped
                            if ticker == 'CASH':
                                capped_allocations[ticker] = new_allocation
                            else:
                                ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                                capped_allocations[ticker] = min(new_allocation, ticker_cap)
            
            allocations = capped_allocations
            
            # Final normalization to 100% in case not enough stocks to distribute excess
            total_alloc = sum(allocations.values())
            if total_alloc > 0:
                allocations = {ticker: allocation / total_alloc for ticker, allocation in allocations.items()}
        
        # Update tickers list to only include tickers with non-zero allocations
        tickers = [ticker for ticker, allocation in allocations.items() if allocation > 0]
        
        # Debug output after filtering
        print(f"[THRESHOLD DEBUG] Filtered allocations: {allocations}")
        print(f"[THRESHOLD DEBUG] Final tickers: {tickers}")
    
    benchmark_ticker = config.get('benchmark_ticker')
    initial_value = config.get('initial_value', 0)
    # Allocation tracker: ignore added cash for this mode. Use initial_value as current portfolio value.
    added_amount = 0
    added_frequency = 'none'
    # Map frequency to ensure compatibility with get_dates_by_freq
    raw_rebalancing_frequency = config.get('rebalancing_frequency', 'none')
    def map_frequency_for_backtest(freq):
        if freq is None:
            return 'Never'
        freq_map = {
            'Never': 'Never',
            'Weekly': 'Weekly',
            'Biweekly': 'Biweekly',
            'Monthly': 'Monthly',
            'Quarterly': 'Quarterly',
            'Semiannually': 'Semiannually',
            'Annually': 'Annually',
            # Legacy format mapping
            'none': 'Never',
            'week': 'Weekly',
            '2weeks': 'Biweekly',
            'month': 'Monthly',
            '3months': 'Quarterly',
            '6months': 'Semiannually',
            'year': 'Annually'
        }
        return freq_map.get(freq, 'Monthly')
    
    rebalancing_frequency = map_frequency_for_backtest(raw_rebalancing_frequency)
    use_momentum = config.get('use_momentum', True)
    momentum_windows = config.get('momentum_windows', [])
    calc_beta = config.get('calc_beta', False)
    calc_volatility = config.get('calc_volatility', False)
    beta_window_days = config.get('beta_window_days', 365)
    exclude_days_beta = config.get('exclude_days_beta', 30)
    vol_window_days = config.get('vol_window_days', 365)
    exclude_days_vol = config.get('exclude_days_vol', 30)
    current_data = {t: reindexed_data[t] for t in tickers + [benchmark_ticker] if t in reindexed_data}
    # Respect start_with setting: 'oldest' (default) or 'all' (wait for all assets)
    start_with = config.get('start_with', 'oldest')
    # Precompute first-valid dates for each ticker to decide availability
    start_dates_config = {}
    for t in tickers:
        if t in reindexed_data and isinstance(reindexed_data.get(t), pd.DataFrame):
            fd = reindexed_data[t].first_valid_index()
            start_dates_config[t] = fd if fd is not None else pd.NaT
        else:
            start_dates_config[t] = pd.NaT
    dates_added = set()
    dates_rebal = sorted(get_dates_by_freq(rebalancing_frequency, sim_index[0], sim_index[-1], sim_index))

    # Dictionaries to store historical data for new tables
    historical_allocations = {}
    historical_metrics = {}

    def calculate_momentum(date, current_assets, momentum_windows, stocks_config=None):
        cumulative_returns, valid_assets = {}, []
        
        # Apply MA filter BEFORE calculating momentum (more efficient) - COPIED FROM PAGE 1
        assets_to_calculate = current_assets
        if config.get('use_sma_filter', False):
            ma_window = config.get('sma_window', 200)
            ma_type = config.get('ma_type', 'SMA')
            filtered_assets, excluded_assets = filter_assets_by_ma(list(current_assets), reindexed_data, date, ma_window, ma_type, config, stocks_config or config['stocks'])
            
            # If no assets remain after MA filtering, go to cash immediately
            if not filtered_assets:
                return {}, []
            
            # Only calculate momentum for filtered assets
            assets_to_calculate = filtered_assets
        else:
            # No MA filter - use all assets
            pass
        
        filtered_windows = [w for w in momentum_windows if w.get("weight", 0) > 0]
        # Normalize weights so they sum to 1 (same as app.py)
        total_weight = sum(w.get("weight", 0) for w in filtered_windows)
        if total_weight == 0:
            normalized_weights = [0 for _ in filtered_windows]
        else:
            normalized_weights = [w.get("weight", 0) / total_weight for w in filtered_windows]
        # Only consider assets that exist in current_data (filtered earlier)
        candidate_assets = [t for t in assets_to_calculate if t in current_data]
        for t in candidate_assets:
            is_valid, asset_returns = True, 0.0
            df_t = current_data.get(t)
            if not (isinstance(df_t, pd.DataFrame) and 'Close' in df_t.columns and not df_t['Close'].dropna().empty):
                # no usable data for this ticker
                continue
            for idx, window in enumerate(filtered_windows):
                lookback, exclude = window.get("lookback", 0), window.get("exclude", 0)
                weight = normalized_weights[idx]
                start_mom = date - pd.Timedelta(days=lookback)
                end_mom = date - pd.Timedelta(days=exclude)
                sd = start_dates_config.get(t, pd.NaT)
                # If no start date or asset starts after required lookback, mark invalid
                if pd.isna(sd) or sd > start_mom:
                    is_valid = False
                    break
                try:
                    price_start_index = df_t.index.asof(start_mom)
                    price_end_index = df_t.index.asof(end_mom)
                except Exception:
                    is_valid = False
                    break
                if pd.isna(price_start_index) or pd.isna(price_end_index):
                    is_valid = False
                    break
                price_start = df_t.loc[price_start_index, "Close"]
                price_end = df_t.loc[price_end_index, "Close"]
                if pd.isna(price_start) or pd.isna(price_end) or price_start == 0:
                    is_valid = False
                    break
                
                # ACADEMIC FIX: Include dividends in momentum calculation if configured (Jegadeesh & Titman 1993)
                if include_dividends.get(t, False):
                    # Calculate cumulative dividends in the momentum window
                    divs_in_period = df_t.loc[price_start_index:price_end_index, "Dividends"].fillna(0).sum()
                    ret = ((price_end + divs_in_period) - price_start) / price_start
                else:
                    ret = (price_end - price_start) / price_start
                asset_returns += ret * weight
            if is_valid:
                cumulative_returns[t] = asset_returns
                valid_assets.append(t)
        return cumulative_returns, valid_assets

    def calculate_momentum_weights(returns, valid_assets, date, momentum_strategy='Classic', negative_momentum_strategy='Cash'):
        if not valid_assets: return {}, {}
        rets = {t: returns[t] for t in valid_assets if not pd.isna(returns[t])}
        if not rets: return {}, {}
        beta_vals, vol_vals = {}, {}
        metrics = {t: {} for t in tickers}
        if calc_beta or calc_volatility:
            df_bench = current_data.get(benchmark_ticker)
            if calc_beta:
                start_beta = date - pd.Timedelta(days=beta_window_days)
                end_beta = date - pd.Timedelta(days=exclude_days_beta)
            if calc_volatility:
                start_vol = date - pd.Timedelta(days=vol_window_days)
                end_vol = date - pd.Timedelta(days=exclude_days_vol)
            for t in valid_assets:
                df_t = current_data[t]
                if calc_beta and df_bench is not None:
                    mask_beta = (df_t.index >= start_beta) & (df_t.index <= end_beta)
                    returns_t_beta = df_t.loc[mask_beta, "Price_change"]
                    mask_bench_beta = (df_bench.index >= start_beta) & (df_bench.index <= end_beta)
                    returns_bench_beta = df_bench.loc[mask_bench_beta, "Price_change"]
                    if len(returns_t_beta) < 2 or len(returns_bench_beta) < 2:
                        beta_vals[t] = np.nan
                    else:
                        covariance = np.cov(returns_t_beta, returns_bench_beta)[0,1]
                        variance = np.var(returns_bench_beta)
                        beta_vals[t] = covariance/variance if variance>0 else np.nan
                    metrics[t]['Beta'] = beta_vals[t]
                if calc_volatility:
                    mask_vol = (df_t.index >= start_vol) & (df_t.index <= end_vol)
                    returns_t_vol = df_t.loc[mask_vol, "Price_change"]
                    if len(returns_t_vol) < 2:
                        vol_vals[t] = np.nan
                    else:
                        vol_vals[t] = returns_t_vol.std() * np.sqrt(365)
                    metrics[t]['Volatility'] = vol_vals[t]
        
        for t in rets:
            metrics[t]['Momentum'] = rets[t]

        # Compute initial weights from raw momentum scores (relative/classic) then apply
        # post-filtering by inverse volatility and inverse absolute beta (app.py approach).
        weights = {}
        # raw momentum values
        rets_keys = list(rets.keys())
        all_negative = all(r <= 0 for r in rets.values())

        # Helper: detect relative mode from momentum_strategy string
        relative_mode = isinstance(momentum_strategy, str) and momentum_strategy.lower().startswith('relat')

        if all_negative:
            if negative_momentum_strategy == 'Cash':
                weights = {t: 0 for t in rets_keys}
            elif negative_momentum_strategy == 'Equal weight':
                weights = {t: 1 / len(rets_keys) for t in rets_keys}
            elif negative_momentum_strategy == 'Relative momentum':
                min_score = min(rets.values())
                offset = -min_score + 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
        else:
            if relative_mode:
                min_score = min(rets.values())
                offset = -min_score + 0.01 if min_score < 0 else 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
            else:
                positive_scores = {t: s for t, s in rets.items() if s > 0}
                if positive_scores:
                    sum_positive = sum(positive_scores.values())
                    weights = {t: positive_scores[t] / sum_positive for t in positive_scores}
                    for t in [t for t in rets_keys if rets.get(t, 0) <= 0]:
                        weights[t] = 0
                else:
                    weights = {t: 0 for t in rets_keys}

        # Apply post-filtering using inverse volatility and inverse absolute beta (like app.py)
        fallback_mode = all_negative and negative_momentum_strategy == 'Equal weight'
        if weights and (calc_volatility or calc_beta) and not fallback_mode:
            filtered_weights = {}
            for t, w in weights.items():
                if w > 0:
                    score = 1.0
                    if calc_volatility:
                        v = vol_vals.get(t, np.nan)
                        if not pd.isna(v) and v > 0:
                            score *= (1.0 / v)
                        else:
                            score *= 0
                    if calc_beta:
                        b = beta_vals.get(t, np.nan)
                        if not pd.isna(b):
                            abs_beta = abs(b)
                            if abs_beta > 0:
                                score *= (1.0 / abs_beta)
                            else:
                                score *= 1.0
                    filtered_weights[t] = w * score
            total_filtered = sum(filtered_weights.values())
            if total_filtered > 0:
                weights = {t: v / total_filtered for t, v in filtered_weights.items()}
            else:
                weights = {}

        # Apply allocation filters in correct order: Max Allocation -> Min Threshold -> Max Allocation (two-pass system)
        use_max_allocation = config.get('use_max_allocation', False)
        max_allocation_percent = config.get('max_allocation_percent', 10.0)
        use_threshold = config.get('use_minimal_threshold', False)
        threshold_percent = config.get('minimal_threshold_percent', 2.0)
        
        # Build dictionary of individual ticker caps from stock configs
        individual_caps = {}
        for stock in config.get('stocks', []):
            ticker = stock.get('ticker', '')
            individual_cap = stock.get('max_allocation_percent', None)
            if individual_cap is not None and individual_cap > 0:
                individual_caps[ticker] = individual_cap / 100.0
        
        # Apply caps if either global cap is enabled OR any individual caps exist
        if (use_max_allocation or individual_caps) and weights:
            max_allocation_decimal = max_allocation_percent / 100.0
            
            # FIRST PASS: Apply maximum allocation filter
            capped_weights = {}
            excess_weight = 0.0
            
            for ticker, weight in weights.items():
                # CASH is exempt from max_allocation limit
                if ticker == 'CASH':
                    capped_weights[ticker] = weight
                else:
                    # Use individual cap if available, otherwise use global cap
                    ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                    
                    if weight > ticker_cap:
                        # Cap the weight and collect excess
                        capped_weights[ticker] = ticker_cap
                        excess_weight += (weight - ticker_cap)
                    else:
                        # Keep original weight
                        capped_weights[ticker] = weight
            
            # Redistribute excess weight proportionally among stocks that are below the cap
            if excess_weight > 0:
                # Find stocks that can receive more weight (below their individual cap) - include CASH as eligible
                eligible_stocks = {}
                for ticker, weight in capped_weights.items():
                    if ticker == 'CASH':
                        eligible_stocks[ticker] = weight
                    else:
                        ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                        if weight < ticker_cap:
                            eligible_stocks[ticker] = weight
                
                if eligible_stocks:
                    # Calculate total weight of eligible stocks
                    total_eligible_weight = sum(eligible_stocks.values())
                    
                    if total_eligible_weight > 0:
                        # Redistribute excess proportionally
                        for ticker in eligible_stocks:
                            proportion = eligible_stocks[ticker] / total_eligible_weight
                            additional_weight = excess_weight * proportion
                            new_weight = capped_weights[ticker] + additional_weight
                            
                            # CASH can receive unlimited weight, other stocks are capped
                            if ticker == 'CASH':
                                capped_weights[ticker] = new_weight
                            else:
                                # Make sure we don't exceed the individual ticker's cap
                                ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                                capped_weights[ticker] = min(new_weight, ticker_cap)
            
            weights = capped_weights
            
            # Final normalization to 100% in case not enough stocks to distribute excess
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
        
        # Apply minimal threshold filter if enabled
        if use_threshold and weights:
            threshold_decimal = threshold_percent / 100.0
            
            # Check which stocks are below threshold after max allocation redistribution
            filtered_weights = {}
            for ticker, weight in weights.items():
                if weight >= threshold_decimal:
                    # Keep stocks above or equal to threshold (remove stocks below threshold)
                    filtered_weights[ticker] = weight
            
            # Normalize remaining stocks to sum to 1.0
            if filtered_weights:
                total_weight = sum(filtered_weights.values())
                if total_weight > 0:
                    weights = {ticker: weight / total_weight for ticker, weight in filtered_weights.items()}
                else:
                    weights = {}
            else:
                # If no stocks meet threshold, keep original weights
                weights = weights
        
        # SECOND PASS: Apply maximum allocation filter again (in case normalization created new excess)
        if (use_max_allocation or individual_caps) and weights:
            max_allocation_decimal = max_allocation_percent / 100.0
            
            # Check if any stocks exceed the cap after threshold filtering and normalization
            capped_weights = {}
            excess_weight = 0.0
            
            for ticker, weight in weights.items():
                # CASH is exempt from max_allocation limit
                if ticker == 'CASH':
                    capped_weights[ticker] = weight
                else:
                    # Use individual cap if available, otherwise use global cap
                    ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                    
                    if weight > ticker_cap:
                        # Cap the weight and collect excess
                        capped_weights[ticker] = ticker_cap
                        excess_weight += (weight - ticker_cap)
                    else:
                        # Keep original weight
                        capped_weights[ticker] = weight
            
            # Redistribute excess weight proportionally among stocks that are below the cap
            if excess_weight > 0:
                # Find stocks that can receive more weight (below their individual cap) - include CASH as eligible
                eligible_stocks = {}
                for ticker, weight in capped_weights.items():
                    if ticker == 'CASH':
                        eligible_stocks[ticker] = weight
                    else:
                        ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                        if weight < ticker_cap:
                            eligible_stocks[ticker] = weight
                
                if eligible_stocks:
                    # Calculate total weight of eligible stocks
                    total_eligible_weight = sum(eligible_stocks.values())
                    
                    if total_eligible_weight > 0:
                        # Redistribute excess proportionally
                        for ticker in eligible_stocks:
                            proportion = eligible_stocks[ticker] / total_eligible_weight
                            additional_weight = excess_weight * proportion
                            new_weight = capped_weights[ticker] + additional_weight
                            
                            # CASH can receive unlimited weight, other stocks are capped
                            if ticker == 'CASH':
                                capped_weights[ticker] = new_weight
                            else:
                                # Make sure we don't exceed the individual ticker's cap
                                ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                                capped_weights[ticker] = min(new_weight, ticker_cap)
            
            weights = capped_weights
            
            # Final normalization to 100% in case not enough stocks to distribute excess
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {ticker: weight / total_weight for ticker, weight in weights.items()}

        for t in weights:
            metrics[t]['Calculated_Weight'] = weights.get(t, 0)

        # Debug: print metrics summary for this rebal date when beta/vol modifiers are active
        if calc_beta or calc_volatility:
            try:
                debug_lines = [
                    f"[MOM DEBUG] Date: {date} | Ticker: {t} | Momentum: {metrics[t].get('Momentum')} | Beta: {metrics[t].get('Beta')} | Vol: {metrics[t].get('Volatility')} | Weight: {weights.get(t, metrics[t].get('Calculated_Weight'))}"
                    for t in rets_keys
                ]
                for ln in debug_lines:
                    print(ln)
            except Exception as e:
                print(f"[MOM DEBUG] Error printing debug metrics: {e}")

        return weights, metrics
        # --- MODIFIED LOGIC END ---

    values = {t: [0.0] for t in tickers}
    unallocated_cash = [0.0]
    unreinvested_cash = [0.0]
    portfolio_no_additions = [initial_value]
    
    # Initial allocation and metric storage
    if not use_momentum:
        # If start_with is 'oldest', only allocate to tickers that are available at the simulation start
        if start_with == 'oldest':
            available_at_start = [t for t in tickers if start_dates_config.get(t, pd.Timestamp.max) <= sim_index[0]]
            current_allocations = {t: allocations.get(t, 0) if t in available_at_start else 0 for t in tickers}
        else:
            current_allocations = {t: allocations.get(t,0) for t in tickers}
        
        # Apply MA filter even when momentum is disabled - COPIED FROM PAGE 1
        if config.get('use_sma_filter', False):
            ma_window = config.get('sma_window', 200)
            ma_type = config.get('ma_type', 'SMA')
            # Get list of current tickers (excluding CASH)
            current_tickers = [t for t in tickers if t != 'CASH']
            filtered_tickers, excluded_assets = filter_assets_by_ma(current_tickers, reindexed_data, sim_index[0], ma_window, ma_type, config, config['stocks'])
            
            # If no assets remain after MA filtering, go to cash
            if not filtered_tickers:
                current_allocations = {t: 0 for t in tickers}
            else:
                # Only keep allocations for filtered tickers, set others to 0
                filtered_allocations = {}
                for t in tickers:
                    if t in filtered_tickers:
                        filtered_allocations[t] = current_allocations.get(t, 0)
                    else:
                        filtered_allocations[t] = 0
                
                # Normalize filtered allocations to sum to 1
                total_filtered = sum(filtered_allocations.values())
                if total_filtered > 0:
                    current_allocations = {t: allocation / total_filtered for t, allocation in filtered_allocations.items()}
                else:
                    # If no filtered allocations, use equal weights for filtered tickers
                    equal_weight = 1.0 / len(filtered_tickers) if filtered_tickers else 0
                    current_allocations = {t: equal_weight if t in filtered_tickers else 0 for t in tickers}
        
        # Apply allocation filters in correct order: Max Allocation -> Min Threshold -> Max Allocation (two-pass system)
        use_max_allocation = config.get('use_max_allocation', False)
        max_allocation_percent = config.get('max_allocation_percent', 10.0)
        use_threshold = config.get('use_minimal_threshold', False)
        threshold_percent = config.get('minimal_threshold_percent', 2.0)
        
        if use_max_allocation and current_allocations:
            max_allocation_decimal = max_allocation_percent / 100.0
            
            # FIRST PASS: Apply maximum allocation filter
            capped_allocations = {}
            excess_allocation = 0.0
            
            for ticker, allocation in current_allocations.items():
                if allocation > max_allocation_decimal:
                    # Cap the allocation and collect excess
                    capped_allocations[ticker] = max_allocation_decimal
                    excess_allocation += (allocation - max_allocation_decimal)
                else:
                    # Keep original allocation
                    capped_allocations[ticker] = allocation
            
            # Redistribute excess allocation proportionally among stocks that are below the cap
            if excess_allocation > 0:
                # Find stocks that can receive more allocation (below the cap)
                eligible_stocks = {ticker: allocation for ticker, allocation in capped_allocations.items() 
                                 if allocation < max_allocation_decimal}
                
                if eligible_stocks:
                    # Calculate total allocation of eligible stocks
                    total_eligible_allocation = sum(eligible_stocks.values())
                    
                    if total_eligible_allocation > 0:
                        # Redistribute excess proportionally
                        for ticker in eligible_stocks:
                            proportion = eligible_stocks[ticker] / total_eligible_allocation
                            additional_allocation = excess_allocation * proportion
                            new_allocation = capped_allocations[ticker] + additional_allocation
                            
                            # Make sure we don't exceed the cap
                            capped_allocations[ticker] = min(new_allocation, max_allocation_decimal)
            
            current_allocations = capped_allocations
        
        # Apply minimal threshold filter for non-momentum strategies
        if use_threshold and current_allocations:
            threshold_decimal = threshold_percent / 100.0
            
            # First: Filter out stocks below threshold
            filtered_allocations = {}
            for ticker, allocation in current_allocations.items():
                if allocation >= threshold_decimal:
                    # Keep stocks above or equal to threshold
                    filtered_allocations[ticker] = allocation
            
            # Then: Normalize remaining stocks to sum to 1
            if filtered_allocations:
                total_allocation = sum(filtered_allocations.values())
                if total_allocation > 0:
                    current_allocations = {ticker: allocation / total_allocation for ticker, allocation in filtered_allocations.items()}
                else:
                    current_allocations = {}
            else:
                # If no stocks meet threshold, keep original allocations
                current_allocations = current_allocations
        
        # SECOND PASS: Apply maximum allocation filter again (in case normalization created new excess)
        if use_max_allocation and current_allocations:
            max_allocation_decimal = max_allocation_percent / 100.0
            
            # Check if any stocks exceed the cap after threshold filtering and normalization
            capped_allocations = {}
            excess_allocation = 0.0
            
            for ticker, allocation in current_allocations.items():
                if allocation > max_allocation_decimal:
                    # Cap the allocation and collect excess
                    capped_allocations[ticker] = max_allocation_decimal
                    excess_allocation += (allocation - max_allocation_decimal)
                else:
                    # Keep original allocation
                    capped_allocations[ticker] = allocation
            
            # Redistribute excess allocation proportionally among stocks that are below the cap
            if excess_allocation > 0:
                # Find stocks that can receive more allocation (below the cap)
                eligible_stocks = {ticker: allocation for ticker, allocation in capped_allocations.items() 
                                 if allocation < max_allocation_decimal}
                
                if eligible_stocks:
                    # Calculate total allocation of eligible stocks
                    total_eligible_allocation = sum(eligible_stocks.values())
                    
                    if total_eligible_allocation > 0:
                        # Redistribute excess proportionally
                        for ticker in eligible_stocks:
                            proportion = eligible_stocks[ticker] / total_eligible_allocation
                            additional_allocation = excess_allocation * proportion
                            new_allocation = capped_allocations[ticker] + additional_allocation
                            
                            # Make sure we don't exceed the cap
                            capped_allocations[ticker] = min(new_allocation, max_allocation_decimal)
            
            current_allocations = capped_allocations
    else:
        returns, valid_assets = calculate_momentum(sim_index[0], set(tickers), momentum_windows, config['stocks'])
        current_allocations, metrics_on_rebal = calculate_momentum_weights(
            returns, valid_assets, date=sim_index[0],
            momentum_strategy=config.get('momentum_strategy', 'Classic'),
            negative_momentum_strategy=config.get('negative_momentum_strategy', 'Cash')
        )
        historical_metrics[sim_index[0]] = metrics_on_rebal
    
    sum_alloc = sum(current_allocations.get(t,0) for t in tickers)
    if sum_alloc > 0:
        for t in tickers:
            values[t][0] = initial_value * current_allocations.get(t,0) / sum_alloc
        unallocated_cash[0] = 0
    else:
        unallocated_cash[0] = initial_value
    
    historical_allocations[sim_index[0]] = {t: values[t][0] / initial_value if initial_value > 0 else 0 for t in tickers}
    historical_allocations[sim_index[0]]['CASH'] = unallocated_cash[0] / initial_value if initial_value > 0 else 0
    
    for i in range(len(sim_index)):
        date = sim_index[i]
        if i == 0: continue
        
        date_prev = sim_index[i-1]
        total_unreinvested_dividends = 0
        total_portfolio_prev = sum(values[t][-1] for t in tickers) + unreinvested_cash[-1]
        daily_growth_factor = 1
        if total_portfolio_prev > 0:
            total_portfolio_current_before_changes = 0
            for t in tickers:
                df = reindexed_data[t]
                price_prev = df.loc[date_prev, "Close"]
                val_prev = values[t][-1]
                nb_shares = val_prev / price_prev if price_prev > 0 else 0
                # --- Dividend fix: find the correct trading day for dividend ---
                div = 0.0
                # CRITICAL FIX: For leveraged tickers, get dividends from the base ticker, not the leveraged ticker
                if "?L=" in t:
                    # For leveraged tickers, get dividend data from the base ticker
                    base_ticker, leverage = parse_leverage_ticker(t)
                    if base_ticker in reindexed_data:
                        base_df = reindexed_data[base_ticker]
                        if "Dividends" in base_df.columns:
                            if date in base_df.index:
                                div = base_df.loc[date, "Dividends"]
                            else:
                                # Find next trading day in index after 'date'
                                future_dates = base_df.index[base_df.index > date]
                                if len(future_dates) > 0:
                                    div = base_df.loc[future_dates[0], "Dividends"]
                else:
                    # For regular tickers, get dividend data normally
                    if "Dividends" in df.columns:
                        if date in df.index:
                            div = df.loc[date, "Dividends"]
                        else:
                            # Find next trading day in index after 'date'
                            future_dates = df.index[df.index > date]
                            if len(future_dates) > 0:
                                div = df.loc[future_dates[0], "Dividends"]
                var = df.loc[date, "Price_change"] if date in df.index else 0.0
                
                # Expense ratio is already applied in apply_daily_leverage() when data is fetched
                # No need to re-apply it here (would be double application + slow loop)
                
                if include_dividends.get(t, False):
                    # CRITICAL FIX: For leveraged tickers, dividends should be handled differently
                    # When simulating leveraged ETFs, the dividend RATE should be the same as the base asset
                    if "?L=" in t:
                        # For leveraged tickers, get the base ticker's dividend rate (not amount)
                        base_ticker, leverage = parse_leverage_ticker(t)
                        if base_ticker in reindexed_data:
                            base_df = reindexed_data[base_ticker]
                            base_price_prev = base_df.loc[date_prev, "Close"]
                            # Use the base ticker's dividend rate, not the leveraged amount
                            dividend_rate = div / base_price_prev if base_price_prev > 0 else 0
                            rate_of_return = var + dividend_rate
                        else:
                            # Fallback: use leveraged price (may cause issues)
                            rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                    else:
                        # For regular tickers, use normal dividend reinvestment
                        rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                    val_new = val_prev * (1 + rate_of_return)
                else:
                    val_new = val_prev * (1 + var)
                    # If dividends are not included, do NOT add to unreinvested cash or anywhere else
                total_portfolio_current_before_changes += val_new
            total_portfolio_current_before_changes += unreinvested_cash[-1] + total_unreinvested_dividends
            daily_growth_factor = total_portfolio_current_before_changes / total_portfolio_prev
        for t in tickers:
            df = reindexed_data[t]
            price_prev = df.loc[date_prev, "Close"]
            val_prev = values[t][-1]
            # --- Dividend fix: find the correct trading day for dividend ---
            div = 0.0
            # CRITICAL FIX: For leveraged tickers, get dividends from the base ticker, not the leveraged ticker
            if "?L=" in t:
                # For leveraged tickers, get dividend data from the base ticker
                base_ticker, leverage = parse_leverage_ticker(t)
                if base_ticker in reindexed_data:
                    base_df = reindexed_data[base_ticker]
                    if "Dividends" in base_df.columns:
                        if date in base_df.index:
                            div = base_df.loc[date, "Dividends"]
                        else:
                            # Find next trading day in index after 'date'
                            future_dates = base_df.index[base_df.index > date]
                            if len(future_dates) > 0:
                                div = base_df.loc[future_dates[0], "Dividends"]
            else:
                # For regular tickers, get dividend data normally
                if "Dividends" in df.columns:
                    if date in df.index:
                        div = df.loc[date, "Dividends"]
                    else:
                        # Find next trading day in index after 'date'
                        future_dates = df.index[df.index > date]
                        if len(future_dates) > 0:
                            div = df.loc[future_dates[0], "Dividends"]
            var = df.loc[date, "Price_change"] if date in df.index else 0.0
            
            # Expense ratio is already applied in apply_daily_leverage() when data is fetched
            # No need to re-apply it here (would be double application + slow loop)
            
            if include_dividends.get(t, False):
                # CRITICAL FIX: For leveraged tickers, dividends should be handled differently
                # When simulating leveraged ETFs, the dividend RATE should be the same as the base asset
                if "?L=" in t:
                    # For leveraged tickers, get the base ticker's dividend rate (not amount)
                    base_ticker, leverage = parse_leverage_ticker(t)
                    if base_ticker in reindexed_data:
                        base_df = reindexed_data[base_ticker]
                        base_price_prev = base_df.loc[date_prev, "Close"]
                        # Use the base ticker's dividend rate, not the leveraged amount
                        dividend_rate = div / base_price_prev if base_price_prev > 0 else 0
                        rate_of_return = var + dividend_rate
                    else:
                        # Fallback: use leveraged price (may cause issues)
                        rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                else:
                    # For regular tickers, use normal dividend reinvestment
                    rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                val_new = val_prev * (1 + rate_of_return)
            else:
                val_new = val_prev * (1 + var)
            values[t].append(val_new)
        unallocated_cash.append(unallocated_cash[-1])
        unreinvested_cash.append(unreinvested_cash[-1] + total_unreinvested_dividends)
        portfolio_no_additions.append(portfolio_no_additions[-1] * daily_growth_factor)
        
        current_total = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
        
        # Check if we should rebalance
        should_rebalance = False
        
        # First check if it's a regular rebalancing date - COPIED FROM PAGE 1
        # Normalize dates for comparison (remove timezone and time components)
        date_normalized = pd.Timestamp(date).normalize()
        dates_rebal_normalized = {pd.Timestamp(d).normalize() for d in dates_rebal}
        
        if date_normalized in dates_rebal_normalized and set(tickers):
            # If targeted rebalancing is enabled, check thresholds first
            if config.get('use_targeted_rebalancing', False):
                # Calculate current allocations as percentages
                current_total = sum(values[t][-1] for t in tickers)
                if current_total > 0:
                    current_allocations = {t: values[t][-1] / current_total for t in tickers}
                    
                    # Check if any ticker exceeds its targeted rebalancing thresholds
                    targeted_settings = config.get('targeted_rebalancing_settings', {})
                    threshold_exceeded = False
                    
                    for ticker in tickers:
                        if ticker in targeted_settings and targeted_settings[ticker].get('enabled', False):
                            current_allocation_pct = current_allocations.get(ticker, 0) * 100
                            max_threshold = targeted_settings[ticker].get('max_allocation', 100.0)
                            min_threshold = targeted_settings[ticker].get('min_allocation', 0.0)
                            
                            # Check if allocation exceeds max or falls below min threshold
                            if current_allocation_pct > max_threshold or current_allocation_pct < min_threshold:
                                threshold_exceeded = True
                                break
                    
                    # Only rebalance if thresholds are exceeded
                    should_rebalance = threshold_exceeded
                else:
                    # If no current value, don't rebalance
                    should_rebalance = False
            else:
                # Regular rebalancing - always rebalance on scheduled dates
                should_rebalance = True
        
        if should_rebalance and set(tickers):
            if use_momentum:
                returns, valid_assets = calculate_momentum(date, set(tickers), momentum_windows, config['stocks'])
                if valid_assets:
                    weights, metrics_on_rebal = calculate_momentum_weights(
                        returns, valid_assets, date=date,
                        momentum_strategy=config.get('momentum_strategy', 'Classic'),
                        negative_momentum_strategy=config.get('negative_momentum_strategy', 'Cash')
                    )
                    historical_metrics[date] = metrics_on_rebal
                    if all(w == 0 for w in weights.values()):
                        # All cash: move total to unallocated_cash, set asset values to zero
                        for t in tickers:
                            values[t][-1] = 0
                        unallocated_cash[-1] = current_total
                        unreinvested_cash[-1] = 0
                    else:
                        # For Buy & Hold strategies with momentum, only distribute new cash
                        if rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"]:
                            # Calculate current proportions for Buy & Hold, or use momentum weights for Buy & Hold (Target)
                            if rebalancing_frequency == "Buy & Hold":
                                # Use current proportions from existing holdings
                                current_total_value = sum(values[t][-1] for t in tickers)
                                if current_total_value > 0:
                                    current_proportions = {t: values[t][-1] / current_total_value for t in tickers}
                                else:
                                    # If no current holdings, use equal weights
                                    current_proportions = {t: 1.0 / len(tickers) for t in tickers}
                            else:  # "Buy & Hold (Target)"
                                # Use momentum weights
                                current_proportions = weights
                            
                            # Only distribute the new cash (unallocated_cash + unreinvested_cash)
                            cash_to_distribute = unallocated_cash[-1] + unreinvested_cash[-1]
                            for t in tickers:
                                # Add new cash proportionally to existing holdings
                                values[t][-1] += cash_to_distribute * current_proportions.get(t, 0)
                            unreinvested_cash[-1] = 0
                            unallocated_cash[-1] = 0
                        else:
                            # Normal momentum rebalancing: replace all holdings
                            for t in tickers:
                                values[t][-1] = current_total * weights.get(t, 0)
                            unreinvested_cash[-1] = 0
                            unallocated_cash[-1] = 0
            else:
                # Non-momentum rebalancing: respect 'start_with' option
                
                # ALWAYS start with original allocations (like page 1)
                rebalance_allocations = {t: allocations.get(t, 0) for t in tickers}
                
                # Apply MA filter if enabled (for non-momentum strategies) - COPIED FROM PAGE 1
                if config.get('use_sma_filter', False):
                     print(f"🔍 DEBUG MA FILTER PAGE 2 REBALANCING: Portfolio {config.get('name', 'Unknown')} - MA filter enabled at {date}")
                     ma_window = config.get('sma_window', 200)
                     ma_type = config.get('ma_type', 'SMA')
                     # Get list of current tickers (excluding CASH)
                     current_tickers = [t for t in tickers if t != 'CASH']
                     print(f"🔍 DEBUG MA FILTER PAGE 2 REBALANCING: Current tickers: {current_tickers}")
                     
                     filtered_tickers, excluded_assets = filter_assets_by_ma(current_tickers, reindexed_data, date, ma_window, ma_type, config, config.get('stocks', []))
                     print(f"🔍 DEBUG MA FILTER PAGE 2 REBALANCING: After filtering - filtered_tickers: {filtered_tickers}, excluded_assets: {excluded_assets}")
                     
                     # Redistribute allocations of excluded tickers proportionally among remaining tickers - EXACTLY LIKE PAGE 1
                     if excluded_assets:
                         excluded_ticker_list = list(excluded_assets.keys())
                         
                         # Calculate total allocation of excluded tickers
                         excluded_allocation = sum(rebalance_allocations.get(t, 0) for t in excluded_ticker_list)
                         
                         # Remove excluded tickers from current allocations
                         for excluded_ticker in excluded_ticker_list:
                             if excluded_ticker in rebalance_allocations:
                                 del rebalance_allocations[excluded_ticker]
                         
                         # If there are remaining tickers (excluding CASH), redistribute proportionally - EXACTLY LIKE PAGE 1
                         remaining_tickers = [t for t in rebalance_allocations.keys() if t != 'CASH']
                         if remaining_tickers:
                             # Calculate total allocation of remaining tickers (excluding CASH)
                             remaining_allocation = sum(rebalance_allocations.get(t, 0) for t in remaining_tickers)
                             
                             if remaining_allocation > 0:
                                 # Redistribute excluded allocation proportionally
                                 for ticker in remaining_tickers:
                                     proportion = rebalance_allocations[ticker] / remaining_allocation
                                     rebalance_allocations[ticker] += excluded_allocation * proportion
                             else:
                                 # Equal distribution if no remaining allocation
                                 equal_allocation = excluded_allocation / len(remaining_tickers)
                                 for ticker in remaining_tickers:
                                     rebalance_allocations[ticker] = equal_allocation
                         else:
                             # No remaining tickers, go to cash
                             # Put everything in unallocated_cash
                             for t in tickers:
                                 values[t][-1] = 0
                             unallocated_cash[-1] = current_total
                             unreinvested_cash[-1] = 0
                             # Clear rebalance_allocations so rest of rebalancing logic is skipped
                             rebalance_allocations = {t: 0 for t in tickers}
                     
                     print(f"🔍 DEBUG MA FILTER PAGE 2 REBALANCING: After redistribution - rebalance_allocations: {rebalance_allocations}")
                     # DO NOT UPDATE allocations HERE - we need to keep original allocations for next rebalancing
                     # The filtered allocations are already in rebalance_allocations and will be used below
                
                if start_with == 'oldest':
                    # Only consider tickers that have data by this rebalancing date
                    available = [t for t in tickers if start_dates_config.get(t, pd.Timestamp.max) <= date]
                    sum_alloc_avail = sum(rebalance_allocations.get(t,0) for t in available)
                    if sum_alloc_avail > 0:
                        # For Buy & Hold strategies, only distribute new cash without touching existing holdings
                        if rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"]:
                            # Calculate current proportions for Buy & Hold, or use target allocations for Buy & Hold (Target)
                            if rebalancing_frequency == "Buy & Hold":
                                # Use current proportions from existing holdings
                                current_total_value = sum(values[t][-1] for t in tickers)
                                if current_total_value > 0:
                                    current_proportions = {t: values[t][-1] / current_total_value for t in tickers}
                                else:
                                    # If no current holdings, use equal weights
                                    current_proportions = {t: 1.0 / len(tickers) for t in tickers}
                            else:  # "Buy & Hold (Target)"
                                # Use target allocations (from rebalance_allocations which includes MA filtering)
                                current_proportions = {t: rebalance_allocations.get(t,0)/sum_alloc_avail for t in available}
                            
                            # Only distribute the new cash (unallocated_cash + unreinvested_cash)
                            cash_to_distribute = unallocated_cash[-1] + unreinvested_cash[-1]
                            for t in tickers:
                                if t in available:
                                    # Add new cash proportionally to existing holdings
                                    values[t][-1] += cash_to_distribute * current_proportions.get(t, 0)
                                else:
                                    # Keep existing value for unavailable tickers
                                    pass
                            unreinvested_cash[-1] = 0
                            unallocated_cash[-1] = 0
                        else:
                            # Normal rebalancing: replace all holdings
                            # For targeted rebalancing, rebalance TO THE THRESHOLD LIMITS, not to base allocations
                            if config.get('use_targeted_rebalancing', False):
                                targeted_settings = config.get('targeted_rebalancing_settings', {})
                                target_allocations = {}
                                
                                # Calculate target allocations based on threshold limits
                                for t in tickers:
                                    if t in available:
                                        if t in targeted_settings and targeted_settings[t].get('enabled', False):
                                            current_allocation_pct = (values[t][-1] / current_total) * 100 if current_total > 0 else 0
                                            max_threshold = targeted_settings[t].get('max_allocation', 100.0)
                                            min_threshold = targeted_settings[t].get('min_allocation', 0.0)
                                            
                                            # Rebalance to the threshold limit that was exceeded
                                            if current_allocation_pct > max_threshold:
                                                target_allocations[t] = max_threshold / 100.0
                                            elif current_allocation_pct < min_threshold:
                                                target_allocations[t] = min_threshold / 100.0
                                            else:
                                                # Within bounds - keep current allocation
                                                target_allocations[t] = current_allocation_pct / 100.0
                                        else:
                                            # Not in targeted settings - keep current allocation
                                            target_allocations[t] = (values[t][-1] / current_total) if current_total > 0 else 0
                                    else:
                                        target_allocations[t] = 0
                                
                                # Calculate remaining allocation for non-targeted tickers
                                total_targeted = sum(target_allocations.values())
                                remaining_allocation = 1.0 - total_targeted
                                
                                # Get non-targeted tickers
                                non_targeted_tickers = [t for t in tickers if t in available and (t not in targeted_settings or not targeted_settings[t].get('enabled', False))]
                                
                                # Distribute remaining allocation PROPORTIONALLY to base allocations (not equally) - COPIED FROM PAGE 1
                                if non_targeted_tickers and remaining_allocation > 0:
                                    non_targeted_base_sum = sum(rebalance_allocations.get(t, 0) for t in non_targeted_tickers)
                                    if non_targeted_base_sum > 0:
                                        # Distribute proportionally to base allocations (from rebalance_allocations which includes MA filtering)
                                        for t in non_targeted_tickers:
                                            base_proportion = rebalance_allocations.get(t, 0) / non_targeted_base_sum
                                            target_allocations[t] = base_proportion * remaining_allocation
                                    else:
                                        # If no base allocations, distribute equally
                                        allocation_per_ticker = remaining_allocation / len(non_targeted_tickers)
                                        for t in non_targeted_tickers:
                                            target_allocations[t] = allocation_per_ticker
                                
                                # Apply target allocations
                                for t in tickers:
                                    values[t][-1] = current_total * target_allocations.get(t, 0)
                            else:
                                # Regular rebalancing - use base allocations (from rebalance_allocations which includes MA filtering)
                                for t in tickers:
                                    if t in available:
                                        weight = rebalance_allocations.get(t,0)/sum_alloc_avail
                                        values[t][-1] = current_total * weight
                                    else:
                                        values[t][-1] = 0
                            
                            unreinvested_cash[-1] = 0
                            unallocated_cash[-1] = 0
                    else:
                        # No assets available yet — keep everything as cash
                        for t in tickers:
                            values[t][-1] = 0
                        unreinvested_cash[-1] = 0
                        unallocated_cash[-1] = current_total
                else:
                    # NOTE: rebalance_allocations has already been initialized with MA filter applied at line 4933-4988
                    # We should NOT reapply MA filter here or reinitialize rebalance_allocations from allocations
                    
                    # Apply threshold filter for non-momentum strategies during rebalancing
                    use_threshold = config.get('use_minimal_threshold', False)
                    threshold_percent = config.get('minimal_threshold_percent', 2.0)
                    
                    if use_threshold:
                        threshold_decimal = threshold_percent / 100.0
                        
                        # First: Filter out stocks below threshold (use rebalance_allocations which already has MA filter applied)
                        filtered_allocations = {}
                        for t in tickers:
                            allocation = rebalance_allocations.get(t, 0)
                            if allocation >= threshold_decimal:
                                # Keep stocks above or equal to threshold
                                filtered_allocations[t] = allocation
                        
                        # Then: Normalize remaining stocks to sum to 1
                        if filtered_allocations:
                            total_allocation = sum(filtered_allocations.values())
                            if total_allocation > 0:
                                rebalance_allocations = {t: allocation / total_allocation for t, allocation in filtered_allocations.items()}
                            else:
                                rebalance_allocations = {}
                        else:
                            # If no stocks meet threshold, keep existing rebalance_allocations
                            pass
                    
                    # Apply maximum allocation filter during rebalancing
                    use_max_allocation = config.get('use_max_allocation', False)
                    max_allocation_percent = config.get('max_allocation_percent', 10.0)
                    
                    if use_max_allocation and rebalance_allocations:
                        max_allocation_decimal = max_allocation_percent / 100.0
                        
                        # Cap individual stock allocations at maximum
                        capped_rebalance_allocations = {}
                        excess_allocation = 0.0
                        
                        for t in tickers:
                            allocation = rebalance_allocations.get(t, 0)
                            if allocation > max_allocation_decimal:
                                # Cap the allocation and collect excess
                                capped_rebalance_allocations[t] = max_allocation_decimal
                                excess_allocation += (allocation - max_allocation_decimal)
                            else:
                                # Keep original allocation
                                capped_rebalance_allocations[t] = allocation
                        
                        # Redistribute excess allocation proportionally among stocks that are below the cap
                        if excess_allocation > 0:
                            # Find stocks that can receive more allocation (below the cap)
                            eligible_stocks = {t: allocation for t, allocation in capped_rebalance_allocations.items() 
                                             if allocation < max_allocation_decimal}
                            
                            if eligible_stocks:
                                # Calculate total allocation of eligible stocks
                                total_eligible_allocation = sum(eligible_stocks.values())
                                
                                if total_eligible_allocation > 0:
                                    # Redistribute excess proportionally
                                    for t in eligible_stocks:
                                        proportion = eligible_stocks[t] / total_eligible_allocation
                                        additional_allocation = excess_allocation * proportion
                                        new_allocation = capped_rebalance_allocations[t] + additional_allocation
                                        
                                        # Make sure we don't exceed the cap
                                        capped_rebalance_allocations[t] = min(new_allocation, max_allocation_decimal)
                        
                        rebalance_allocations = capped_rebalance_allocations
                    
                    # Apply targeted rebalancing if enabled and thresholds are violated
                    if config.get('use_targeted_rebalancing', False) and should_rebalance:
                        targeted_settings = config.get('targeted_rebalancing_settings', {})
                        current_asset_values = {t: values[t][-1] for t in tickers}
                        current_total_value = sum(current_asset_values.values())
                        
                        if current_total_value > 0:
                            current_allocations = {t: v / current_total_value for t, v in current_asset_values.items()}
                            
                            # Apply targeted rebalancing
                            new_allocations = {}
                            
                            # Set allocations for tickers with targeted rebalancing
                            for ticker in tickers:
                                if ticker in targeted_settings and targeted_settings[ticker].get('enabled', False):
                                    settings = targeted_settings[ticker]
                                    min_alloc = settings.get('min_allocation', 0.0) / 100.0
                                    max_alloc = settings.get('max_allocation', 100.0) / 100.0
                                    current_alloc = current_allocations.get(ticker, 0.0)
                                    
                                    if current_alloc > max_alloc:
                                        new_allocations[ticker] = max_alloc
                                    elif current_alloc < min_alloc:
                                        new_allocations[ticker] = min_alloc
                                    else:
                                        new_allocations[ticker] = current_alloc
                                else:
                                    new_allocations[ticker] = current_allocations.get(ticker, 0.0)
                            
                            # Normalize to ensure allocations sum to 1.0
                            total_alloc = sum(new_allocations.values())
                            if total_alloc > 0:
                                rebalance_allocations = {t: alloc / total_alloc for t, alloc in new_allocations.items()}
                    
                    sum_alloc = sum(rebalance_allocations.values())
                    if sum_alloc > 0:
                        # For Buy & Hold strategies, only distribute new cash without touching existing holdings
                        if rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"]:
                            # Calculate current proportions for Buy & Hold, or use target allocations for Buy & Hold (Target)
                            if rebalancing_frequency == "Buy & Hold":
                                # Use current proportions from existing holdings
                                current_total_value = sum(values[t][-1] for t in tickers)
                                if current_total_value > 0:
                                    current_proportions = {t: values[t][-1] / current_total_value for t in tickers}
                                else:
                                    # If no current holdings, use equal weights
                                    current_proportions = {t: 1.0 / len(tickers) for t in tickers}
                            else:  # "Buy & Hold (Target)"
                                # Use target allocations
                                current_proportions = {t: rebalance_allocations.get(t, 0) / sum_alloc for t in tickers}
                            
                            # Only distribute the new cash (unallocated_cash + unreinvested_cash)
                            cash_to_distribute = unallocated_cash[-1] + unreinvested_cash[-1]
                            for t in tickers:
                                # Add new cash proportionally to existing holdings
                                values[t][-1] += cash_to_distribute * current_proportions.get(t, 0)
                            unreinvested_cash[-1] = 0
                            unallocated_cash[-1] = 0
                        else:
                            # Normal rebalancing: replace all holdings
                            # For targeted rebalancing, rebalance TO THE THRESHOLD LIMITS, not to base allocations
                            if config.get('use_targeted_rebalancing', False):
                                targeted_settings = config.get('targeted_rebalancing_settings', {})
                                target_allocations = {}
                                
                                # Calculate target allocations based on threshold limits
                                for t in tickers:
                                    if t in targeted_settings and targeted_settings[t].get('enabled', False):
                                        current_allocation_pct = (values[t][-1] / current_total) * 100 if current_total > 0 else 0
                                        max_threshold = targeted_settings[t].get('max_allocation', 100.0)
                                        min_threshold = targeted_settings[t].get('min_allocation', 0.0)
                                        
                                        # Rebalance to the threshold limit that was exceeded
                                        if current_allocation_pct > max_threshold:
                                            target_allocations[t] = max_threshold / 100.0
                                        elif current_allocation_pct < min_threshold:
                                            target_allocations[t] = min_threshold / 100.0
                                        else:
                                            # Within bounds - keep current allocation
                                            target_allocations[t] = current_allocation_pct / 100.0
                                    else:
                                        # Not in targeted settings - keep current allocation
                                        target_allocations[t] = (values[t][-1] / current_total) if current_total > 0 else 0
                                
                                # Calculate remaining allocation for non-targeted tickers
                                total_targeted = sum(target_allocations.values())
                                remaining_allocation = 1.0 - total_targeted
                                
                                # Get non-targeted tickers
                                non_targeted_tickers = [t for t in tickers if (t not in targeted_settings or not targeted_settings[t].get('enabled', False))]
                                
                                # Distribute remaining allocation PROPORTIONALLY to base allocations (not equally) - COPIED FROM PAGE 1
                                if non_targeted_tickers and remaining_allocation > 0:
                                    non_targeted_base_sum = sum(rebalance_allocations.get(t, 0) for t in non_targeted_tickers)
                                    if non_targeted_base_sum > 0:
                                        # Distribute proportionally to base allocations
                                        for t in non_targeted_tickers:
                                            base_proportion = rebalance_allocations.get(t, 0) / non_targeted_base_sum
                                            target_allocations[t] = base_proportion * remaining_allocation
                                    else:
                                        # If no base allocations, distribute equally
                                        allocation_per_ticker = remaining_allocation / len(non_targeted_tickers)
                                        for t in non_targeted_tickers:
                                            target_allocations[t] = allocation_per_ticker
                                
                                # Apply target allocations
                                for t in tickers:
                                    values[t][-1] = current_total * target_allocations.get(t, 0)
                            else:
                                # Regular rebalancing - use base allocations
                                for t in tickers:
                                    weight = rebalance_allocations.get(t, 0) / sum_alloc
                                    values[t][-1] = current_total * weight
                            
                            unreinvested_cash[-1] = 0
                            unallocated_cash[-1] = 0
            
            # Store allocations at rebalancing date
            current_total_after_rebal = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
            if current_total_after_rebal > 0:
                allocs = {t: values[t][-1] / current_total_after_rebal for t in tickers}
                allocs['CASH'] = unallocated_cash[-1] / current_total_after_rebal if current_total_after_rebal > 0 else 0
                historical_allocations[date] = allocs
            else:
                allocs = {t: 0 for t in tickers}
                allocs['CASH'] = 0
                historical_allocations[date] = allocs

    # Store last allocation - ONLY APPLY MA FILTERS IF LAST DATE IS A REBALANCING DATE
    last_date = sim_index[-1]
    last_total = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
    
    # Check if last date is a rebalancing date
    date_normalized = pd.Timestamp(last_date).normalize()
    dates_rebal_normalized = {pd.Timestamp(d).normalize() for d in dates_rebal}
    is_rebalancing_date = date_normalized in dates_rebal_normalized
    
    if last_total > 0:
        # Only apply MA filter if last date is actually a rebalancing date
        if config.get('use_sma_filter', False) and is_rebalancing_date:
            ma_window = config.get('sma_window', 200)
            ma_type = config.get('ma_type', 'SMA')
            # Get list of current tickers (excluding CASH)
            current_tickers = [t for t in tickers if t != 'CASH']
            filtered_tickers, excluded_assets = filter_assets_by_ma(current_tickers, reindexed_data, last_date, ma_window, ma_type, config, config['stocks'])
            
            # If no assets remain after MA filtering, go to cash
            if not filtered_tickers:
                last_allocs = {t: 0 for t in tickers}
                last_allocs['CASH'] = 1.0  # All cash
            else:
                # Only keep allocations for filtered tickers, set others to 0
                filtered_last_allocs = {}
                for t in tickers:
                    if t in filtered_tickers:
                        filtered_last_allocs[t] = values[t][-1] / last_total
                    else:
                        filtered_last_allocs[t] = 0
                
                # Normalize filtered allocations to sum to 1
                total_filtered = sum(filtered_last_allocs.values())
                if total_filtered > 0:
                    last_allocs = {t: allocation / total_filtered for t, allocation in filtered_last_allocs.items()}
                else:
                    # If no filtered allocations, use equal weights for filtered tickers
                    equal_weight = 1.0 / len(filtered_tickers) if filtered_tickers else 0
                    last_allocs = {t: equal_weight if t in filtered_tickers else 0 for t in tickers}
                
                last_allocs['CASH'] = unallocated_cash[-1] / last_total if last_total > 0 else 0
        else:
            # No MA filter or not a rebalancing date - use raw allocations (reflects last rebalancing)
            last_allocs = {t: values[t][-1] / last_total for t in tickers}
            last_allocs['CASH'] = unallocated_cash[-1] / last_total if last_total > 0 else 0
        
        historical_allocations[last_date] = last_allocs
    else:
        historical_allocations[last_date] = {t: 0 for t in tickers}
        historical_allocations[last_date]['CASH'] = 0
    
    # Store last metrics: always add a last-rebalance snapshot so the UI has a metrics row
    # If momentum is used, compute metrics; otherwise build metrics from the last allocation snapshot
    if use_momentum:
        returns, valid_assets = calculate_momentum(last_date, set(tickers), momentum_windows, config['stocks'])
        weights, metrics_on_rebal = calculate_momentum_weights(
            returns, valid_assets, date=last_date,
            momentum_strategy=config.get('momentum_strategy', 'Classic'),
            negative_momentum_strategy=config.get('negative_momentum_strategy', 'Cash')
        )
        # Add CASH line to metrics
        cash_weight = 1.0 if all(w == 0 for w in weights.values()) else 0.0
        metrics_on_rebal['CASH'] = {'Calculated_Weight': cash_weight}
        historical_metrics[last_date] = metrics_on_rebal
    else:
        # Build a metrics snapshot from the last allocation so there's always a 'last rebalance' metrics entry
        if last_date in historical_allocations:
            alloc_snapshot = historical_allocations.get(last_date, {})
            metrics_on_rebal = {}
            for ticker_sym, alloc_val in alloc_snapshot.items():
                metrics_on_rebal[ticker_sym] = {'Calculated_Weight': alloc_val}
            # Ensure CASH entry exists
            if 'CASH' not in metrics_on_rebal:
                metrics_on_rebal['CASH'] = {'Calculated_Weight': alloc_snapshot.get('CASH', 0)}
            # Only set if not already present
            if last_date not in historical_metrics:
                historical_metrics[last_date] = metrics_on_rebal

    results = pd.DataFrame(index=sim_index)
    for t in tickers:
        results[f"Value_{t}"] = values[t]
    results["Unallocated_cash"] = unallocated_cash
    results["Unreinvested_cash"] = unreinvested_cash
    results["Total_assets"] = results[[f"Value_{t}" for t in tickers]].sum(axis=1)
    results["Total_with_dividends_plus_cash"] = results["Total_assets"] + results["Unallocated_cash"] + results["Unreinvested_cash"]
    results['Portfolio_Value_No_Additions'] = portfolio_no_additions

    return results["Total_with_dividends_plus_cash"], results['Portfolio_Value_No_Additions'], historical_allocations, historical_metrics


# -----------------------
# Main App Logic
# -----------------------

from copy import deepcopy
# Use page-scoped session keys so this page does not share state with other pages
if 'alloc_portfolio_configs' not in st.session_state:
    # initialize from existing global configs if present, but deep-copy to avoid shared references
            st.session_state.alloc_portfolio_configs = deepcopy(st.session_state.get('alloc_portfolio_configs', default_configs))
if 'alloc_active_portfolio_index' not in st.session_state:
    st.session_state.alloc_active_portfolio_index = 0
if 'alloc_paste_json_text' not in st.session_state:
    st.session_state.alloc_paste_json_text = ""
if 'alloc_rerun_flag' not in st.session_state:
    st.session_state.alloc_rerun_flag = False

def add_portfolio_callback():
    new_portfolio = default_configs[1].copy()
    new_portfolio['name'] = f"New Portfolio {len(st.session_state.alloc_portfolio_configs) + 1}"
    st.session_state.alloc_portfolio_configs.append(new_portfolio)
    st.session_state.alloc_active_portfolio_index = len(st.session_state.alloc_portfolio_configs) - 1
    st.session_state.alloc_rerun_flag = True

def remove_portfolio_callback():
    if len(st.session_state.alloc_portfolio_configs) > 1:
        st.session_state.alloc_portfolio_configs.pop(st.session_state.alloc_active_portfolio_index)
        st.session_state.alloc_active_portfolio_index = max(0, st.session_state.alloc_active_portfolio_index - 1)
        st.session_state.alloc_rerun_flag = True

def add_stock_callback():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'].append({'ticker': '', 'allocation': 0.0, 'include_dividends': True, 'include_in_sma_filter': True, 'max_allocation_percent': None})
    st.session_state.alloc_rerun_flag = True

def remove_stock_callback(ticker):
    """Immediate stock removal callback"""
    try:
        active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
        stocks = active_portfolio['stocks']
        
        # Find and remove the stock with matching ticker
        for i, stock in enumerate(stocks):
            if stock['ticker'] == ticker:
                stocks.pop(i)
                # If this was the last stock, add an empty one
                if len(stocks) == 0:
                    stocks.append({'ticker': '', 'allocation': 0.0, 'include_dividends': True})
                st.session_state.alloc_rerun_flag = True
                break
    except (IndexError, KeyError):
        pass

def normalize_stock_allocations_callback():
    if 'alloc_portfolio_configs' not in st.session_state or 'alloc_active_portfolio_index' not in st.session_state:
        return
    stocks = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks']
    valid_stocks = [s for s in stocks if s['ticker']]
    total_alloc = sum(s['allocation'] for s in valid_stocks)
    if total_alloc > 0:
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] /= total_alloc
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = int(s['allocation'] * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] = stocks
    st.session_state.alloc_rerun_flag = True

def equal_stock_allocation_callback():
    if 'alloc_portfolio_configs' not in st.session_state or 'alloc_active_portfolio_index' not in st.session_state:
        return
    stocks = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks']
    valid_stocks = [s for s in stocks if s['ticker']]
    if valid_stocks:
        equal_weight = 1.0 / len(valid_stocks)
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] = equal_weight
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = int(equal_weight * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] = stocks
    st.session_state.alloc_rerun_flag = True
    
def reset_portfolio_callback():
    current_name = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['name']
    default_cfg_found = next((cfg for cfg in default_configs if cfg['name'] == current_name), None)
    if default_cfg_found is None:
        default_cfg_found = default_configs[1].copy()
        default_cfg_found['name'] = current_name
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index] = default_cfg_found
    st.session_state.alloc_rerun_flag = True

def reset_stock_selection_callback():
    current_name = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['name']
    default_cfg_found = next((cfg for cfg in default_configs if cfg['name'] == current_name), None)
    if default_cfg_found is None:
        default_cfg_found = default_configs[1].copy()
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] = default_cfg_found['stocks']
    st.session_state.alloc_rerun_flag = True

def reset_momentum_windows_callback():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = [
        {"lookback": 365, "exclude": 30, "weight": 0.5},
        {"lookback": 180, "exclude": 30, "weight": 0.3},
        {"lookback": 120, "exclude": 30, "weight": 0.2},
    ]
    st.session_state.alloc_rerun_flag = True

def reset_beta_callback():
    # Reset beta lookback/exclude to defaults and enable beta calculation for alloc page
    idx = st.session_state.alloc_active_portfolio_index
    st.session_state.alloc_portfolio_configs[idx]['beta_window_days'] = 365
    st.session_state.alloc_portfolio_configs[idx]['exclude_days_beta'] = 30
    # Ensure checkbox state reflects enabled
    st.session_state.alloc_portfolio_configs[idx]['calc_beta'] = True
    st.session_state['alloc_active_calc_beta'] = True
    st.session_state.alloc_rerun_flag = True

def reset_vol_callback():
    # Reset volatility lookback/exclude to defaults and enable volatility calculation
    idx = st.session_state.alloc_active_portfolio_index
    st.session_state.alloc_portfolio_configs[idx]['vol_window_days'] = 365
    st.session_state.alloc_portfolio_configs[idx]['exclude_days_vol'] = 30
    st.session_state.alloc_portfolio_configs[idx]['calc_volatility'] = True
    st.session_state['alloc_active_calc_vol'] = True
    st.session_state.alloc_rerun_flag = True

def add_momentum_window_callback():
    # Append a new momentum window with modest defaults (alloc page)
    idx = st.session_state.alloc_active_portfolio_index
    cfg = st.session_state.alloc_portfolio_configs[idx]
    if 'momentum_windows' not in cfg:
        cfg['momentum_windows'] = []
    # default new window
    cfg['momentum_windows'].append({"lookback": 90, "exclude": 30, "weight": 0.1})
    st.session_state.alloc_portfolio_configs[idx] = cfg
    st.session_state.alloc_rerun_flag = True

def remove_momentum_window_callback():
    idx = st.session_state.alloc_active_portfolio_index
    cfg = st.session_state.alloc_portfolio_configs[idx]
    if 'momentum_windows' in cfg and cfg['momentum_windows']:
        cfg['momentum_windows'].pop()
        st.session_state.alloc_portfolio_configs[idx] = cfg
        st.session_state.alloc_rerun_flag = True

def normalize_momentum_weights_callback():
    # Use page-scoped configs for allocations page
    if 'alloc_portfolio_configs' not in st.session_state or 'alloc_active_portfolio_index' not in st.session_state:
        return
    active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
    total_weight = sum(w['weight'] for w in active_portfolio.get('momentum_windows', []))
    if total_weight > 0:
        for idx, w in enumerate(active_portfolio.get('momentum_windows', [])):
            w['weight'] /= total_weight
            weight_key = f"alloc_weight_input_active_{st.session_state.alloc_active_portfolio_index}_{idx}"
            # Sanitize weight to prevent StreamlitValueAboveMaxError
            weight = w['weight']
            if isinstance(weight, (int, float)):
                # Convert decimal to percentage, ensuring it's within bounds
                weight_percentage = max(0.0, min(weight * 100.0, 100.0))
            else:
                # Invalid weight, set to default
                weight_percentage = 10.0
            st.session_state[weight_key] = int(weight_percentage)
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = active_portfolio.get('momentum_windows', [])
    st.session_state.alloc_rerun_flag = True

def paste_json_callback():
    try:
        # Use the SAME parsing logic as successful PDF extraction
        raw_text = st.session_state.get('alloc_paste_json_text', '{}')
        
        # STEP 1: Try the exact same approach as PDF extraction (simple strip + parse)
        try:
            cleaned_text = raw_text.strip()
            json_data = json.loads(cleaned_text)
            st.success("✅ JSON parsed successfully using PDF-style parsing!")
        except json.JSONDecodeError:
            # STEP 2: If that fails, apply our advanced cleaning (fallback)
            st.info("🔧 Simple parsing failed, applying advanced PDF extraction fixes...")
            
            json_text = raw_text
            import re
            
            # Fix common PDF extraction issues
            # Pattern to find broken portfolio name lines like: "name": "Some Name "stocks":
            broken_pattern = r'"name":\s*"([^"]*?)"\s*"stocks":'
            # Replace with proper JSON structure: "name": "Some Name", "stocks":
            json_text = re.sub(broken_pattern, r'"name": "\1", "stocks":', json_text)
            
            # Fix truncated names that end abruptly without closing quote
            # Pattern: "name": "Some text without closing quote "stocks":
            truncated_pattern = r'"name":\s*"([^"]*?)\s+"stocks":'
            json_text = re.sub(truncated_pattern, r'"name": "\1", "stocks":', json_text)
            
            # Fix missing opening brace for portfolio objects
            # Pattern: }, "name": should be }, { "name":
            missing_brace_pattern = r'(},)\s*("name":)'
            json_text = re.sub(missing_brace_pattern, r'\1 {\n \2', json_text)
            
            json_data = json.loads(json_text)
            st.success("✅ JSON parsed successfully using advanced cleaning!")
        
        # Add missing fields for compatibility if they don't exist
        if 'collect_dividends_as_cash' not in json_data:
            json_data['collect_dividends_as_cash'] = False
        if 'exclude_from_cashflow_sync' not in json_data:
            json_data['exclude_from_cashflow_sync'] = False
        if 'exclude_from_rebalancing_sync' not in json_data:
            json_data['exclude_from_rebalancing_sync'] = False
        if 'use_minimal_threshold' not in json_data:
            json_data['use_minimal_threshold'] = False
        if 'minimal_threshold_percent' not in json_data:
            json_data['minimal_threshold_percent'] = 4.0
        if 'use_max_allocation' not in json_data:
            json_data['use_max_allocation'] = False
        if 'max_allocation_percent' not in json_data:
            json_data['max_allocation_percent'] = 20.0
        
        # Debug: Show what we received
        st.info(f"Received JSON keys: {list(json_data.keys())}")
        if 'tickers' in json_data:
            st.info(f"Tickers in JSON: {json_data['tickers']}")
        if 'stocks' in json_data:
            st.info(f"Stocks in JSON: {json_data['stocks']}")
        if 'momentum_windows' in json_data:
            st.info(f"Momentum windows in JSON: {json_data['momentum_windows']}")
        if 'use_momentum' in json_data:
            st.info(f"Use momentum in JSON: {json_data['use_momentum']}")
        
        # Handle momentum strategy value mapping from other pages
        momentum_strategy = json_data.get('momentum_strategy', 'Classic')
        if momentum_strategy == 'Classic momentum':
            momentum_strategy = 'Classic'
        elif momentum_strategy == 'Relative momentum':
            momentum_strategy = 'Relative Momentum'
        elif momentum_strategy not in ['Classic', 'Relative Momentum']:
            momentum_strategy = 'Classic'  # Default fallback
        
        # Handle negative momentum strategy value mapping from other pages
        negative_momentum_strategy = json_data.get('negative_momentum_strategy', 'Cash')
        if negative_momentum_strategy == 'Go to cash':
            negative_momentum_strategy = 'Cash'
        elif negative_momentum_strategy not in ['Cash', 'Equal weight', 'Relative momentum']:
            negative_momentum_strategy = 'Cash'  # Default fallback
        
        # Handle stocks field - convert from legacy format if needed
        stocks = json_data.get('stocks', [])
        if not stocks and 'tickers' in json_data:
            # Convert legacy format (tickers, allocs, divs) to stocks format
            tickers = json_data.get('tickers', [])
            allocs = json_data.get('allocs', [])
            divs = json_data.get('divs', [])
            stocks = []
            
            # Ensure we have valid arrays
            if tickers and isinstance(tickers, list):
                for i in range(len(tickers)):
                    if tickers[i] and tickers[i].strip():  # Check for non-empty ticker
                        # Convert allocation from percentage (0-100) to decimal (0.0-1.0) format
                        allocation = 0.0
                        if i < len(allocs) and allocs[i] is not None:
                            alloc_value = float(allocs[i])
                            if alloc_value > 1.0:
                                # Already in percentage format, convert to decimal
                                allocation = alloc_value / 100.0
                            else:
                                # Already in decimal format, use as is
                                allocation = alloc_value
                        
                        # Resolve the alias to the actual Yahoo ticker
                        resolved_ticker = resolve_ticker_alias(tickers[i].strip())
                        stock = {
                            'ticker': resolved_ticker,  # Use resolved ticker
                            'allocation': allocation,
                            'include_dividends': bool(divs[i]) if i < len(divs) and divs[i] is not None else True
                        }
                        stocks.append(stock)
            
            # Debug output
            st.info(f"Converted {len(stocks)} stocks from legacy format: {[s['ticker'] for s in stocks]}")
        
        # Ensure all stocks have max_allocation_percent field
        for stock in stocks:
            if 'max_allocation_percent' not in stock:
                stock['max_allocation_percent'] = None
            if 'include_in_sma_filter' not in stock:
                stock['include_in_sma_filter'] = True
        
        # Sanitize momentum window weights to prevent StreamlitValueAboveMaxError
        momentum_windows = json_data.get('momentum_windows', [])
        for window in momentum_windows:
            if 'weight' in window:
                weight = window['weight']
                # If weight is a percentage (e.g., 50 for 50%), convert to decimal
                if isinstance(weight, (int, float)) and weight > 1.0:
                    # Cap at 100% and convert to decimal
                    weight = min(weight, 100.0) / 100.0
                elif isinstance(weight, (int, float)) and weight <= 1.0:
                    # Already in decimal format, ensure it's valid
                    weight = max(0.0, min(weight, 1.0))
                else:
                    # Invalid weight, set to default
                    weight = 0.1
                window['weight'] = weight
        
        # Map frequency values from app.py format to Allocations format
        def map_frequency(freq):
            if freq is None:
                return 'Never'
            freq_map = {
                'Never': 'Never',
                'Weekly': 'Weekly',
                'Biweekly': 'Biweekly',
                'Monthly': 'Monthly',
                'Quarterly': 'Quarterly',
                'Semiannually': 'Semiannually',
                'Annually': 'Annually',
                # Legacy format mapping
                'none': 'Never',
                'week': 'Weekly',
                '2weeks': 'Biweekly',
                'month': 'Monthly',
                '3months': 'Quarterly',
                '6months': 'Semiannually',
                'year': 'Annually'
            }
            return freq_map.get(freq, 'Monthly')
        
        # Allocations page specific: ensure all required fields are present
        # and ignore fields that are specific to other pages
        allocations_config = {
            'name': json_data.get('name', 'Allocation Portfolio'),
            'stocks': stocks,
            'benchmark_ticker': json_data.get('benchmark_ticker', '^GSPC'),
            'initial_value': json_data.get('initial_value', 10000),
            'added_amount': json_data.get('added_amount', 0),  # Allocations page typically doesn't use additions
            'added_frequency': map_frequency(json_data.get('added_frequency', 'Never')),  # Allocations page typically doesn't use additions
            'rebalancing_frequency': map_frequency(json_data.get('rebalancing_frequency', 'Monthly')),
            'start_date_user': json_data.get('start_date_user'),
            'end_date_user': json_data.get('end_date_user'),
            'start_with': json_data.get('start_with', 'oldest'),
            'use_momentum': json_data.get('use_momentum', True),
            'momentum_strategy': momentum_strategy,
            'negative_momentum_strategy': negative_momentum_strategy,
            'momentum_windows': momentum_windows,
            'use_minimal_threshold': json_data.get('use_minimal_threshold', False),
            'minimal_threshold_percent': json_data.get('minimal_threshold_percent', 4.0),
            'use_max_allocation': json_data.get('use_max_allocation', False),
            'max_allocation_percent': json_data.get('max_allocation_percent', 20.0),
            'calc_beta': json_data.get('calc_beta', True),
            'calc_volatility': json_data.get('calc_volatility', True),
            'beta_window_days': json_data.get('beta_window_days', 365),
            'exclude_days_beta': json_data.get('exclude_days_beta', 30),
            'vol_window_days': json_data.get('vol_window_days', 365),
            'exclude_days_vol': json_data.get('exclude_days_vol', 30),
            'use_targeted_rebalancing': json_data.get('use_targeted_rebalancing', False),
            'targeted_rebalancing_settings': json_data.get('targeted_rebalancing_settings', {}),
            'use_sma_filter': json_data.get('use_sma_filter', False),
            'sma_window': json_data.get('sma_window', 200),
            'ma_type': json_data.get('ma_type', 'SMA'),
        }
        
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index] = allocations_config
        
        # Update session state for threshold settings
        st.session_state['alloc_active_use_threshold'] = allocations_config.get('use_minimal_threshold', False)
        st.session_state['alloc_active_threshold_percent'] = allocations_config.get('minimal_threshold_percent', 4.0)
        st.session_state['alloc_active_use_max_allocation'] = allocations_config.get('use_max_allocation', False)
        st.session_state['alloc_active_max_allocation_percent'] = allocations_config.get('max_allocation_percent', 20.0)
        
        # Update session state for MA filter settings
        st.session_state['alloc_active_use_sma_filter'] = allocations_config.get('use_sma_filter', False)
        st.session_state['alloc_active_sma_window'] = allocations_config.get('sma_window', 200)
        st.session_state['alloc_active_ma_type'] = allocations_config.get('ma_type', 'SMA')
        
        # Update session state for momentum settings (FIX FOR VISUAL BUG)
        st.session_state['alloc_active_use_momentum'] = allocations_config.get('use_momentum', True)
        st.session_state['alloc_active_momentum_strategy'] = allocations_config.get('momentum_strategy', 'Classic')
        st.session_state['alloc_active_negative_momentum_strategy'] = allocations_config.get('negative_momentum_strategy', 'Cash')
        st.session_state['alloc_active_calc_beta'] = allocations_config.get('calc_beta', False)
        st.session_state['alloc_active_calc_vol'] = allocations_config.get('calc_volatility', False)
        
        # Update portfolio name input field to match the imported portfolio
        st.session_state.alloc_portfolio_name = allocations_config.get('name', 'Allocation Portfolio')
        
        st.success("Portfolio configuration updated from JSON (Allocations page).")
        st.info(f"Final stocks list: {[s['ticker'] for s in allocations_config['stocks']]}")
        st.info(f"Final momentum windows: {allocations_config['momentum_windows']}")
        st.info(f"Final use_momentum: {allocations_config['use_momentum']}")
        st.info(f"Final threshold settings: use={allocations_config.get('use_minimal_threshold', False)}, percent={allocations_config.get('minimal_threshold_percent', 2.0)}")
        st.info(f"Final max allocation settings: use={allocations_config.get('use_max_allocation', False)}, percent={allocations_config.get('max_allocation_percent', 10.0)}")
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check the text and try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    st.session_state.alloc_rerun_flag = True

def update_active_portfolio_index():
    # Allocation page: keep a page-scoped index. If a selector exists, respect it; otherwise default to 0
    selected_name = st.session_state.get('alloc_portfolio_selector', None)
    portfolio_configs = st.session_state.get('alloc_portfolio_configs', [])
    portfolio_names = [cfg.get('name', '') for cfg in portfolio_configs]
    if selected_name and selected_name in portfolio_names:
        st.session_state.alloc_active_portfolio_index = portfolio_names.index(selected_name)
    else:
        st.session_state.alloc_active_portfolio_index = 0 if portfolio_names else None
    
    # Update portfolio name input field to match the active portfolio
    if st.session_state.alloc_active_portfolio_index is not None and portfolio_configs:
        active_portfolio = portfolio_configs[st.session_state.alloc_active_portfolio_index]
        st.session_state.alloc_portfolio_name = active_portfolio.get('name', 'Allocation Portfolio')
    
    st.session_state.alloc_rerun_flag = True

def update_name():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['name'] = st.session_state.get('alloc_active_name', '')

def update_initial():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['initial_value'] = st.session_state.get('alloc_active_initial', 0)

def update_added_amount():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['added_amount'] = st.session_state.get('alloc_active_added_amount', 0)

def update_add_freq():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['added_frequency'] = st.session_state.get('alloc_active_add_freq', 'none')

def update_rebal_freq():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['rebalancing_frequency'] = st.session_state.get('alloc_active_rebal_freq', 'none')

def update_benchmark():
    # Convert benchmark ticker to uppercase and resolve alias
    benchmark_val = st.session_state.get('alloc_active_benchmark', '')
    # Convert commas to dots for decimal separators (like case conversion)
    converted_benchmark = benchmark_val.replace(",", ".")
    upper_benchmark = converted_benchmark.upper()
    # Keep original benchmark ticker in UI (NO conversion here)
    resolved_benchmark = upper_benchmark
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['benchmark_ticker'] = resolved_benchmark
    # Update the widget to show original ticker
    st.session_state['alloc_active_benchmark'] = resolved_benchmark

def update_use_momentum():
    current_val = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('use_momentum', True)
    new_val = st.session_state.get('alloc_active_use_momentum', True)
    if current_val != new_val:
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['use_momentum'] = new_val
        if new_val:
            # When momentum is enabled, keep existing beta and volatility settings
            pass
            st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = [
                {"lookback": 365, "exclude": 30, "weight": 0.5},
                {"lookback": 180, "exclude": 30, "weight": 0.3},
                {"lookback": 120, "exclude": 30, "weight": 0.2},
            ]
        else:
            st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = []
        st.session_state.alloc_rerun_flag = True

def update_use_sma_filter():
    """Callback function for MA filter checkbox"""
    current_val = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('use_sma_filter', False)
    new_val = st.session_state.alloc_active_use_sma_filter
    
    if current_val != new_val:
        portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
        portfolio['use_sma_filter'] = new_val
        
        # If enabling MA filter, disable targeted rebalancing (mutually exclusive)
        if new_val:
            portfolio['use_targeted_rebalancing'] = False
            st.session_state['alloc_active_use_targeted_rebalancing'] = False
        
        st.session_state.alloc_rerun_flag = True

def update_use_targeted_rebalancing():
    """Callback function for targeted rebalancing checkbox - COPIED FROM PAGE 4"""
    current_val = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('use_targeted_rebalancing', False)
    new_val = st.session_state.get('alloc_active_use_targeted_rebalancing', False)
    
    if current_val != new_val:
        portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
        portfolio['use_targeted_rebalancing'] = new_val
        
        # If enabling targeted rebalancing, disable momentum and MA filter (mutually exclusive)
        if new_val:
            portfolio['use_momentum'] = False
            st.session_state['alloc_active_use_momentum'] = False
            portfolio['use_sma_filter'] = False
            st.session_state['alloc_active_use_sma_filter'] = False
        
        st.session_state.alloc_rerun_flag = True




def update_calc_beta():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['calc_beta'] = st.session_state.get('alloc_active_calc_beta', True)

def update_beta_window():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['beta_window_days'] = st.session_state.get('alloc_active_beta_window', 365)

def update_beta_exclude():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['exclude_days_beta'] = st.session_state.get('alloc_active_beta_exclude', 30)

def update_calc_vol():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['calc_volatility'] = st.session_state.get('alloc_active_calc_vol', True)

def update_vol_window():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['vol_window_days'] = st.session_state.get('alloc_active_vol_window', 365)

def update_vol_exclude():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['exclude_days_vol'] = st.session_state.get('alloc_active_vol_exclude', 30)

def update_use_threshold():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['use_minimal_threshold'] = st.session_state.alloc_active_use_threshold

def update_threshold_percent():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['minimal_threshold_percent'] = st.session_state.alloc_active_threshold_percent
def update_use_max_allocation():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['use_max_allocation'] = st.session_state.alloc_active_use_max_allocation

def update_max_allocation_percent():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['max_allocation_percent'] = st.session_state.alloc_active_max_allocation_percent

# Sidebar simplified for single-portfolio allocation tracker
st.sidebar.title("Allocation Tracker")


# Work with the first portfolio as active (single-portfolio mode). Keep inputs accessible.
active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
# Do not show portfolio name in allocation tracker. Keep a page-scoped session key for compatibility.
if "alloc_active_name" not in st.session_state:
    st.session_state["alloc_active_name"] = active_portfolio['name']

col_left, col_right = st.columns([1, 1])
with col_left:
    if "alloc_active_initial" not in st.session_state:
        # Treat this as the current portfolio value (not a backtest initial cash)
        st.session_state["alloc_active_initial"] = int(active_portfolio.get('initial_value', 0))
    st.number_input("Portfolio Value ($)", min_value=0, step=1000, format="%d", key="alloc_active_initial", on_change=update_initial, help="Current total portfolio value used to compute required shares.")
# Removed Added Amount / Added Frequency UI - allocation tracker is not running periodic additions

# Swap positions: show Rebalancing Frequency first, then Added Frequency.
# Use two equal-width columns and make selectboxes use the container width so they match visually.
col_freq_rebal, col_freq_add = st.columns([1, 1])
freq_options = ["Never", "Weekly", "Biweekly", "Monthly", "Quarterly", "Semiannually", "Annually"]
with col_freq_rebal:
    if "alloc_active_rebal_freq" not in st.session_state:
        st.session_state["alloc_active_rebal_freq"] = active_portfolio['rebalancing_frequency']
    st.selectbox("Rebalancing Frequency", freq_options, key="alloc_active_rebal_freq", on_change=update_rebal_freq, help="How often the portfolio is rebalanced.")
# Note: Added Frequency removed for allocation tracker

# Rebalancing and Added Frequency explanation removed for allocation tracker UI

if "alloc_active_benchmark" not in st.session_state:
    st.session_state["alloc_active_benchmark"] = active_portfolio['benchmark_ticker']
st.text_input("Benchmark Ticker (default: ^GSPC, starts 1927-12-30, used for beta calculation. Use SPYSIM for earlier dates, starts 1885-03-01)", key="alloc_active_benchmark", on_change=update_benchmark)

st.subheader("Tickers")
col_ticker_buttons = st.columns([0.3, 0.3, 0.3, 0.1])
with col_ticker_buttons[0]:
    if st.button("Normalize Tickers %", on_click=normalize_stock_allocations_callback, use_container_width=True):
        pass
with col_ticker_buttons[1]:
    if st.button("Equal Allocation %", on_click=equal_stock_allocation_callback, use_container_width=True):
        pass
with col_ticker_buttons[2]:
    if st.button("Reset Tickers", on_click=reset_stock_selection_callback, use_container_width=True):
        pass

# Calculate live total ticker allocation
valid_tickers = [s for s in st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] if s['ticker']]
total_ticker_allocation = sum(s['allocation'] for s in valid_tickers)

if active_portfolio['use_momentum']:
    st.info("Ticker allocations are not used directly for Momentum strategies.")
else:
    if abs(total_ticker_allocation - 1.0) > 0.001:
        st.warning(f"Total ticker allocation is {total_ticker_allocation*100:.2f}%, not 100%. Click 'Normalize' to fix.")
    else:
        st.success(f"Total ticker allocation is {total_ticker_allocation*100:.2f}%.")

def update_stock_allocation(index):
    try:
        key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][index]['allocation'] = float(val) / 100.0
    except Exception:
        # Ignore transient errors (e.g., active_portfolio_index changed); UI will reflect state on next render
        return


def update_stock_ticker(index):
    try:
        key = f"alloc_ticker_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            # key not yet initialized (race condition). Skip update; the widget's key will be present on next rerender.
            return
        
        
        # Convert commas to dots for decimal separators (like case conversion)
        converted_val = val.replace(",", ".")
        
        # Convert the input value to uppercase
        upper_val = converted_val.upper()
        
        # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
        if upper_val == 'BRK.B':
            upper_val = 'BRK-B'
        elif upper_val == 'BRK.A':
            upper_val = 'BRK-A'

        # CRITICAL: Resolve ticker alias BEFORE storing in portfolio config
        resolved_ticker = resolve_ticker_alias(upper_val)
        
        # Update the portfolio configuration with the resolved ticker (with leverage/expense)
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][index]['ticker'] = resolved_ticker
        
        # IMPORTANT: Force UI update by setting the widget's session_state value
        # This ensures the resolved ticker is displayed immediately in the text_input
        st.session_state[key] = resolved_ticker
        
        # Auto-disable dividends for negative leverage (inverse ETFs)
        if '?L=-' in resolved_ticker:
            st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][index]['include_dividends'] = False
            # Also update the checkbox UI state
            div_key = f"alloc_div_{st.session_state.alloc_active_portfolio_index}_{index}"
            st.session_state[div_key] = False
        
        # Use the EXACT same method as "Special Long-Term Tickers" buttons (line 10001)
        # Set rerun flag instead of calling st.rerun() directly
        st.session_state.alloc_rerun_flag = True
    except Exception:
        # Defensive: if portfolio index or structure changed, skip silently
        return


def update_ma_reference_ticker(stock_index):
    """Callback function when MA reference ticker changes"""
    ma_ref_key = f"alloc_ma_reference_{st.session_state.alloc_active_portfolio_index}_{stock_index}"
    new_value = st.session_state.get(ma_ref_key, '').strip()
    
    # Apply EXACTLY the same transformations as regular tickers
    # Convert commas to dots for decimal separators
    new_value = new_value.replace(",", ".")
    
    # Convert to uppercase
    new_value = new_value.upper()
    
    # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
    if new_value == 'BRK.B':
        new_value = 'BRK-B'
    elif new_value == 'BRK.A':
        new_value = 'BRK-A'
    
    # CRITICAL: Resolve ticker alias (GOLDX → GOLD_COMPLETE, SPYTR → ^SP500TR, etc.)
    if new_value:  # Only resolve if not empty
        resolved_value = resolve_ticker_alias(new_value)
    else:
        resolved_value = new_value
    
    # Update session state with resolved value for display
    st.session_state[ma_ref_key] = resolved_value
    
    # Update the stock config
    portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
    if stock_index < len(portfolio['stocks']):
        old_value = portfolio['stocks'][stock_index].get('ma_reference_ticker', '')
        if resolved_value != old_value:
            portfolio['stocks'][stock_index]['ma_reference_ticker'] = resolved_value
            st.session_state.alloc_rerun_flag = True


def update_stock_dividends(index):
    try:
        key = f"alloc_div_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][index]['include_dividends'] = bool(val)
    except Exception:
        return

# Update active_portfolio
active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]

# Initialize MA filter state
if "alloc_active_use_sma_filter" not in st.session_state:
    st.session_state["alloc_active_use_sma_filter"] = active_portfolio.get('use_sma_filter', False)
if "alloc_active_sma_window" not in st.session_state:
    st.session_state["alloc_active_sma_window"] = active_portfolio.get('sma_window', 200)
 
for i in range(len(active_portfolio['stocks'])):
    stock = active_portfolio['stocks'][i]
    col_t, col_a, col_d, col_sma, col_b = st.columns([0.2, 0.2, 0.25, 0.25, 0.1])
    with col_t:
        ticker_key = f"alloc_ticker_{st.session_state.alloc_active_portfolio_index}_{i}"
        # Always sync the session state with the portfolio config to show resolved ticker
        st.session_state[ticker_key] = stock['ticker']
        st.text_input("Ticker", key=ticker_key, label_visibility="visible", on_change=update_stock_ticker, args=(i,))
    with col_a:
        use_mom = st.session_state.get('alloc_active_use_momentum', active_portfolio.get('use_momentum', True))
        if not use_mom:
            alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{i}"
            if alloc_key not in st.session_state:
                st.session_state[alloc_key] = int(stock['allocation'] * 100)
            st.number_input("Allocation %", min_value=0, step=1, format="%d", key=alloc_key, label_visibility="visible", on_change=update_stock_allocation, args=(i,))
            if st.session_state[alloc_key] != int(stock['allocation'] * 100):
                st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][i]['allocation'] = st.session_state[alloc_key] / 100.0
        else:
            # Show Max Cap % field when momentum is active
            max_cap_key = f"alloc_max_cap_{st.session_state.alloc_active_portfolio_index}_{i}"
            # Ensure max_allocation_percent key exists
            if 'max_allocation_percent' not in stock:
                stock['max_allocation_percent'] = None
            
            if max_cap_key not in st.session_state:
                st.session_state[max_cap_key] = int(stock['max_allocation_percent']) if stock['max_allocation_percent'] is not None else 0
            
            max_cap_value = st.number_input(
                "Max Cap %", 
                min_value=0, 
                max_value=100,
                step=1, 
                format="%d", 
                key=max_cap_key, 
                label_visibility="visible",
                help="Individual cap for this ticker (0 = no cap, uses global cap if enabled)"
            )
            
            # Update the portfolio config
            if max_cap_value > 0:
                st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][i]['max_allocation_percent'] = float(max_cap_value)
            else:
                st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][i]['max_allocation_percent'] = None
    with col_d:
        div_key = f"alloc_div_{st.session_state.alloc_active_portfolio_index}_{i}"
        # Ensure include_dividends key exists with default value
        if 'include_dividends' not in stock:
            stock['include_dividends'] = True
        
        # Auto-disable dividends for negative leverage (inverse ETFs) ONLY on first display
        # Don't override if user has explicitly set a value
        if '?L=-' in stock['ticker'] and div_key not in st.session_state:
            stock['include_dividends'] = False
        
        if div_key not in st.session_state:
            st.session_state[div_key] = stock['include_dividends']
        st.checkbox("Reinvest Dividends", key=div_key)
        if st.session_state[div_key] != stock['include_dividends']:
            st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][i]['include_dividends'] = st.session_state[div_key]
        
    with col_sma:
        # MA Filter selection - EXACT SAME LOGIC AS PAGE 1
        if st.session_state.get("alloc_active_use_sma_filter", False):
            sma_key = f"alloc_include_sma_{st.session_state.alloc_active_portfolio_index}_{i}"
            # Ensure include_in_sma_filter key exists with default value
            if 'include_in_sma_filter' not in stock:
                stock['include_in_sma_filter'] = True
            
            if sma_key not in st.session_state:
                st.session_state[sma_key] = stock['include_in_sma_filter']
            st.checkbox("Include in MA Filter", key=sma_key, help="Uncheck to exclude this ticker from the Moving Average filter")
            if st.session_state[sma_key] != stock['include_in_sma_filter']:
                st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][i]['include_in_sma_filter'] = st.session_state[sma_key]
            
            # MA Reference Ticker - allows using another ticker's MA for filtering
            ma_ref_key = f"alloc_ma_reference_{st.session_state.alloc_active_portfolio_index}_{i}"
            if 'ma_reference_ticker' not in stock:
                stock['ma_reference_ticker'] = ""  # Empty = use own ticker
            
            if ma_ref_key not in st.session_state:
                st.session_state[ma_ref_key] = stock.get('ma_reference_ticker', '')
            
            # Always sync the session state with the portfolio config to show resolved ticker
            st.session_state[ma_ref_key] = stock.get('ma_reference_ticker', '')
            
            st.text_input(
                "MA Reference Ticker",
                key=ma_ref_key,
                placeholder=f"Leave empty for {stock['ticker']}",
                help=f"Optional: Use another ticker's MA (e.g., SPY for SSO, QQQ for TQQQ). Leave empty to use {stock['ticker']}'s own MA.",
                label_visibility="visible",
                on_change=update_ma_reference_ticker,
                args=(i,)
            )
            
            if st.session_state[ma_ref_key] != stock.get('ma_reference_ticker', ''):
                st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][i]['ma_reference_ticker'] = st.session_state[ma_ref_key]
            
        else:
            st.write("")
        
    with col_b:
        st.write("")
        if st.button("Remove", key=f"alloc_rem_stock_{st.session_state.alloc_active_portfolio_index}_{i}_{stock['ticker']}_{id(stock)}", on_click=remove_stock_callback, args=(stock['ticker'],)):
            pass

if st.button("Add Ticker", on_click=add_stock_callback):
    pass

# Bulk Leverage Controls
with st.expander("🔧 Bulk Leverage Controls", expanded=False):
    def apply_bulk_leverage_callback():
        """Apply leverage and expense ratio to selected tickers in the current portfolio"""
        try:
            portfolio_index = st.session_state.alloc_active_portfolio_index
            portfolio = st.session_state.alloc_portfolio_configs[portfolio_index]
            
            leverage_value = st.session_state.get('bulk_leverage_value', 1.0)
            expense_ratio_value = st.session_state.get('bulk_expense_ratio_value', 1.0)
            selected_tickers = st.session_state.get('bulk_selected_tickers', [])
            
            # Check if any tickers are selected
            if not selected_tickers:
                st.toast("⚠️ Please select at least one ticker to apply leverage to.")
                return
            
            applied_count = 0
            for i, stock in enumerate(portfolio['stocks']):
                current_ticker = stock['ticker']
                
                # Check if this ticker should be modified
                base_ticker, _, _ = parse_ticker_parameters(current_ticker)
                if base_ticker in selected_tickers or current_ticker in selected_tickers:
                    # Parse current ticker to get base ticker
                    base_ticker, _, _ = parse_ticker_parameters(current_ticker)
                    
                    # Create new ticker with leverage and expense ratio
                    new_ticker = base_ticker
                    if leverage_value != 1.0:
                        new_ticker += f"?L={leverage_value}"
                    if expense_ratio_value != 0.0:
                        new_ticker += f"?E={expense_ratio_value}"
                    
                    # Update the ticker in the portfolio
                    st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'][i]['ticker'] = new_ticker
                    
                    # Update the session state for the text input
                    ticker_key = f"alloc_ticker_{portfolio_index}_{i}"
                    st.session_state[ticker_key] = new_ticker
                    
                    # If leverage is negative (short position), uncheck dividends checkbox
                    # User can manually re-check it if desired
                    if leverage_value < 0:
                        st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'][i]['include_dividends'] = False
                        div_key = f"alloc_div_{portfolio_index}_{i}"
                        st.session_state[div_key] = False
                    
                    applied_count += 1
            
            if applied_count > 0:
                st.toast(f"✅ Applied {leverage_value}x leverage and {expense_ratio_value}% expense ratio to {applied_count} ticker(s)!")
            else:
                st.warning("⚠️ No tickers were selected for modification.")
            
        except Exception as e:
            st.error(f"Error applying bulk leverage: {str(e)}")

    def remove_bulk_leverage_callback():
        """Remove all leverage and expense ratio from selected tickers"""
        try:
            portfolio_index = st.session_state.alloc_active_portfolio_index
            portfolio = st.session_state.alloc_portfolio_configs[portfolio_index]
            selected_tickers = st.session_state.get('bulk_selected_tickers', [])
            
            # Check if any tickers are selected
            if not selected_tickers:
                st.toast("⚠️ Please select at least one ticker to remove leverage from.")
                return
            
            removed_count = 0
            for i, stock in enumerate(portfolio['stocks']):
                current_ticker = stock['ticker']
                
                # Check if this ticker should be modified
                base_ticker, _, _ = parse_ticker_parameters(current_ticker)
                if base_ticker in selected_tickers or current_ticker in selected_tickers:
                    # Parse current ticker to get base ticker
                    base_ticker, _, _ = parse_ticker_parameters(current_ticker)
                    
                    # Update the ticker to base ticker (no leverage, no expense ratio)
                    st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'][i]['ticker'] = base_ticker
                    
                    # Update the session state for the text input
                    ticker_key = f"alloc_ticker_{portfolio_index}_{i}"
                    st.session_state[ticker_key] = base_ticker
                    
                    removed_count += 1
            
            if removed_count > 0:
                st.toast(f"✅ Removed leverage and expense ratio from {removed_count} ticker(s)!")
            else:
                st.warning("⚠️ No tickers were selected for modification.")
            
        except Exception as e:
            st.error(f"Error removing leverage: {str(e)}")

    # Get current portfolio tickers for selection
    portfolio_index = st.session_state.alloc_active_portfolio_index
    portfolio = st.session_state.alloc_portfolio_configs[portfolio_index]
    available_tickers = [stock['ticker'] for stock in portfolio['stocks']]
    
    # Initialize selected tickers if not exists
    if 'bulk_selected_tickers' not in st.session_state:
        st.session_state.bulk_selected_tickers = []
    
    # Ticker selection interface
    st.markdown("**Select Tickers to Modify:**")
    
    # Quick selection buttons
    col_quick1, col_quick2 = st.columns([1, 1])
    
    with col_quick1:
        if st.button("Select All", key="page2_select_all_tickers", use_container_width=True):
            st.session_state.bulk_selected_tickers = available_tickers.copy()
            st.rerun()
    
    with col_quick2:
        if st.button("Clear Selection", key="page2_clear_all_tickers", use_container_width=True):
            st.session_state.bulk_selected_tickers = []
            st.rerun()
    
    # Individual ticker selection
    if available_tickers:
        st.markdown("**Individual Ticker Selection:**")
        
        # Create checkboxes for each ticker
        for i, ticker in enumerate(available_tickers):
            base_ticker, leverage, expense = parse_ticker_parameters(ticker)
            display_text = f"{base_ticker}"
            if leverage != 1.0 or expense > 0.0:
                display_text += f" (L:{leverage}x, E:{expense}%)"
            
            # Use checkbox state directly
            checkbox_key = f"page2_bulk_ticker_select_{i}"
            is_checked = st.checkbox(
                display_text, 
                value=ticker in st.session_state.bulk_selected_tickers,
                key=checkbox_key
            )
            
            # Update selection based on checkbox state
            if is_checked and ticker not in st.session_state.bulk_selected_tickers:
                st.session_state.bulk_selected_tickers.append(ticker)
            elif not is_checked and ticker in st.session_state.bulk_selected_tickers:
                st.session_state.bulk_selected_tickers.remove(ticker)
    else:
        st.info("No tickers available in the current portfolio.")
    
    # Show selected tickers count
    selected_count = len(st.session_state.bulk_selected_tickers)
    if selected_count > 0:
        st.success(f"📊 {selected_count} ticker(s) selected for bulk operations")
    else:
        st.warning("⚠️ No tickers selected - please select tickers before applying bulk operations")

    # Bulk leverage controls
    st.markdown("---")
    st.markdown("**Leverage & Expense Ratio Settings:**")
    
    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1, 1])

    with col1:
        st.number_input(
            "Leverage",
            value=2.0,
            step=0.1,
            format="%.1f",
            key="bulk_leverage_value",
            help="Leverage multiplier (e.g., 2.0 for 2x leverage, -3.0 for -3x inverse)"
        )

    with col2:
        st.number_input(
            "Expense Ratio (%)",
            value=1.0,
            step=0.01,
            format="%.2f",
            key="bulk_expense_ratio_value",
            help="Annual expense ratio in percentage (e.g., 0.84 for 0.84%, can be negative)"
        )

    with col3:
        if st.button("Apply to Selected", on_click=apply_bulk_leverage_callback, type="primary"):
            pass

    with col4:
        if st.button("Remove from Selected", on_click=remove_bulk_leverage_callback, type="secondary"):
            pass


# Special tickers and leverage guide sections
with st.expander("📈 Broad Long-Term Tickers", expanded=False):
    st.markdown("""
    **Recommended tickers for long-term strategies:**
    
    **Core ETFs:**
    - **SPY** - S&P 500 (0.09% expense ratio)
    - **QQQ** - NASDAQ-100 (0.20% expense ratio)  
    - **VTI** - Total Stock Market (0.03% expense ratio)
    - **VEA** - Developed Markets (0.05% expense ratio)
    - **VWO** - Emerging Markets (0.10% expense ratio)
    
    **Sector ETFs:**
    - **XLK** - Technology (0.10% expense ratio)
    - **XLF** - Financials (0.10% expense ratio)
    - **XLE** - Energy (0.10% expense ratio)
    - **XLV** - Healthcare (0.10% expense ratio)
    - **XLI** - Industrials (0.10% expense ratio)
    
    **Bond ETFs:**
    - **TLT** - 20+ Year Treasury (0.15% expense ratio)
    - **IEF** - 7-10 Year Treasury (0.15% expense ratio)
    - **LQD** - Investment Grade Corporate (0.14% expense ratio)
    - **HYG** - High Yield Corporate (0.49% expense ratio)
    
    **Commodity ETFs:**
    - **GLD** - Gold (0.40% expense ratio)
    - **SLV** - Silver (0.50% expense ratio)
    - **DBA** - Agriculture (0.93% expense ratio)
    - **USO** - Oil (0.60% expense ratio)
    """)

# Special Tickers Section
# Use session state to control expander state
if 'alloc_special_tickers_expanded' not in st.session_state:
    st.session_state.alloc_special_tickers_expanded = False

with st.expander("🎯 Special Long-Term Tickers", expanded=st.session_state.alloc_special_tickers_expanded):
    st.markdown("**Quick access to ticker aliases that the system accepts:**")
    
    # Get the actual ticker aliases from the function
    aliases = get_ticker_aliases()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📈 Stock Indices**")
        stock_mapping = {
            'S&P 500 (No Dividend) (1927+)': ('SPYND', '^GSPC'),
            'S&P 500 (Total Return) (1988+)': ('SPYTR', '^SP500TR'), 
            'NASDAQ (No Dividend) (1971+)': ('QQQND', '^IXIC'),
            'NASDAQ 100 (1985+)': ('NDX', '^NDX'),
            'Dow Jones (1992+)': ('DOW', '^DJI')
        }
        
        for name, (alias, ticker) in stock_mapping.items():
            if st.button(f"➕ {name}", key=f"add_stock_{ticker}", help=f"Add {alias} → {ticker}"):
                # Ensure portfolio configs exist
                if 'alloc_portfolio_configs' not in st.session_state:
                    st.session_state.alloc_portfolio_configs = default_configs
                if 'alloc_active_portfolio_index' not in st.session_state:
                    st.session_state.alloc_active_portfolio_index = 0
                
                portfolio_index = st.session_state.alloc_active_portfolio_index
                # Resolve the alias to the actual Yahoo ticker before storing
                resolved_ticker = resolve_ticker_alias(alias)
                st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'].append({
                    'ticker': resolved_ticker,  # Add the resolved Yahoo ticker
                    'allocation': 0.0, 
                    'include_dividends': True,
                    'include_in_sma_filter': True,
                    'max_allocation_percent': None
                })
                # Keep expander open and rerun immediately
                st.session_state.alloc_special_tickers_expanded = True
                st.rerun()
    
    with col2:
        st.markdown("**🏭 Sector Indices**")
        sector_mapping = {
            'Technology (XLK) (1990+)': ('XLKND', '^SP500-45'),
            'Healthcare (XLV) (1990+)': ('XLVND', '^SP500-35'),
            'Consumer Staples (XLP) (1990+)': ('XLPND', '^SP500-30'),
            'Financials (XLF) (1990+)': ('XLFND', '^SP500-40'),
            'Energy (XLE) (1990+)': ('XLEND', '^SP500-10'),
            'Industrials (XLI) (1990+)': ('XLIND', '^SP500-20'),
            'Consumer Discretionary (XLY) (1990+)': ('XLYND', '^SP500-25'),
            'Materials (XLB) (1990+)': ('XLBND', '^SP500-15'),
            'Utilities (XLU) (1990+)': ('XLUND', '^SP500-55'),
            'Real Estate (XLRE) (1990+)': ('XLREND', '^SP500-60'),
            'Communication Services (XLC) (1990+)': ('XLCND', '^SP500-50')
        }
        
        for name, (alias, ticker) in sector_mapping.items():
            if st.button(f"➕ {name}", key=f"add_sector_{ticker}", help=f"Add {alias} → {ticker}"):
                 # Ensure portfolio configs exist
                 if 'alloc_portfolio_configs' not in st.session_state:
                     st.session_state.alloc_portfolio_configs = default_configs
                 if 'alloc_active_portfolio_index' not in st.session_state:
                     st.session_state.alloc_active_portfolio_index = 0
                 
                 portfolio_index = st.session_state.alloc_active_portfolio_index
                 # Resolve the alias to the actual Yahoo ticker before storing
                 resolved_ticker = resolve_ticker_alias(alias)
                 st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'].append({
                     'ticker': resolved_ticker,  # Add the resolved Yahoo ticker
                     'allocation': 0.0, 
                     'include_dividends': True
                 })
                 st.rerun()
    
    with col3:
        st.markdown("**🔬 Synthetic Tickers**")
        synthetic_tickers = {
            # Ordered by asset class: Stocks → Bonds → Gold → Managed Futures → Bitcoin
            'Complete S&P 500 Simulation (1885+)': ('SPYSIM', 'SPYSIM_COMPLETE'),
            'Dynamic S&P 500 Top 20 (Historical)': ('SP500TOP20', 'SP500TOP20'),
            'Cash Simulator (ZEROX)': ('ZEROX', 'ZEROX'),
            'Complete TBILL Dataset (1948+)': ('TBILL', 'TBILL_COMPLETE'),
            'Complete IEF Dataset (1962+)': ('IEFTR', 'IEF_COMPLETE'),
            'Complete TLT Dataset (1962+)': ('TLTTR', 'TLT_COMPLETE'),
            'Complete ZROZ Dataset (1962+)': ('ZROZX', 'ZROZ_COMPLETE'),
            'Complete Gold Simulation (1968+)': ('GOLDSIM', 'GOLDSIM_COMPLETE'),
            'Complete Gold Dataset (1975+)': ('GOLDX', 'GOLD_COMPLETE'),
            'Complete KMLM Dataset (1992+)': ('KMLMX', 'KMLM_COMPLETE'),
            'Complete DBMF Dataset (2000+)': ('DBMFX', 'DBMF_COMPLETE'),
            'Complete Bitcoin Dataset (2010+)': ('BITCOINX', 'BTC_COMPLETE'),
            
            # Leveraged & Inverse ETFs (Synthetic) - NASDAQ-100 versions
            'Simulated TQQQ (3x QQQ) (1985+)': ('TQQQND', '^NDX?L=3?E=0.95'),
            'Simulated QLD (2x QQQ) (1985+)': ('QLDND', '^NDX?L=2?E=0.95'),
            'Simulated PSQ (-1x QQQ) (1985+)': ('PSQND', '^NDX?L=-1?E=0.95'),
            'Simulated QID (-2x QQQ) (1985+)': ('QIDND', '^NDX?L=-2?E=0.95'),
            'Simulated SQQQ (-3x QQQ) (1985+)': ('SQQQND', '^NDX?L=-3?E=0.95'),
            
            # Leveraged & Inverse ETFs (Synthetic) - NASDAQ Composite versions (longer history)
            'Simulated TQQQ-IXIC (3x IXIC) (1971+)': ('TQQQIXIC', '^IXIC?L=3?E=0.95'),
            'Simulated QLD-IXIC (2x IXIC) (1971+)': ('QLDIXIC', '^IXIC?L=2?E=0.95'),
            'Simulated PSQ-IXIC (-1x IXIC) (1971+)': ('PSQIXIC', '^IXIC?L=-1?E=0.95'),
            'Simulated QID-IXIC (-2x IXIC) (1971+)': ('QIDIXIC', '^IXIC?L=-2?E=0.95'),
            'Simulated SQQQ-IXIC (-3x IXIC) (1971+)': ('SQQQIXIC', '^IXIC?L=-3?E=0.95'),
            
            # S&P 500 leveraged/inverse (unchanged)
            'Simulated SPXL (3x SPY) (1988+)': ('SPXLTR', '^SP500TR?L=3?E=1.00'),
            'Simulated UPRO (3x SPY) (1988+)': ('UPROTR', '^SP500TR?L=3?E=0.91'),
            'Simulated SSO (2x SPY) (1988+)': ('SSOTR', '^SP500TR?L=2?E=0.91'),
            'Simulated SH (-1x SPY) (1927+)': ('SHND', '^GSPC?L=-1?E=0.89'),
            'Simulated SDS (-2x SPY) (1927+)': ('SDSND', '^GSPC?L=-2?E=0.91'),
            'Simulated SPXU (-3x SPY) (1927+)': ('SPXUND', '^GSPC?L=-3?E=1.00')
        }
        
        for name, (alias, ticker) in synthetic_tickers.items():
            # Custom help text for different ticker types
            if alias == 'SP500TOP20':
                help_text = "Add SP500TOP20 → SP500TOP20 - BETA ticker: Dynamic portfolio of top 20 S&P 500 companies rebalanced annually based on historical market cap data"
            elif alias == 'ZEROX':
                help_text = "Add ZEROX → ZEROX - Cash Simulator: Simulates a cash position that does nothing (no price movement, no dividends)"
            elif 'IXIC' in ticker:
                # Special warning for IXIC versions
                help_text = f"Add {alias} → {ticker} ⚠️ WARNING: This tracks NASDAQ Composite (broader index), NOT NASDAQ-100 like the real ETF!"
            else:
                help_text = f"Add {alias} → {ticker}"
            
            if st.button(f"➕ {name}", key=f"add_synthetic_{ticker}", help=help_text):
                # Ensure portfolio configs exist
                if 'alloc_portfolio_configs' not in st.session_state:
                    st.session_state.alloc_portfolio_configs = default_configs
                if 'alloc_active_portfolio_index' not in st.session_state:
                    st.session_state.alloc_active_portfolio_index = 0
                
                portfolio_index = st.session_state.alloc_active_portfolio_index
                # Resolve the alias to the actual ticker before storing
                resolved_ticker = resolve_ticker_alias(alias)
                # Auto-disable dividends for negative leverage (inverse ETFs)
                include_divs = False if '?L=-' in resolved_ticker else True
                st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'].append({
                    'ticker': resolved_ticker,  # Add the resolved ticker
                    'allocation': 0.0, 
                    'include_dividends': include_divs,
                    'include_in_sma_filter': True,
                    'max_allocation_percent': None
                })
                # Keep expander open and rerun immediately
                st.session_state.alloc_special_tickers_expanded = True
                st.rerun()
    
    st.markdown("---")
    
    # Ticker Aliases Section INSIDE the expander
    st.markdown("**💡 Ticker Aliases:** You can also use these shortcuts in the text input below:")
    st.markdown("- `SPX` → `^GSPC` (S&P 500 Price, 1927+), `SPXTR` → `^SP500TR` (S&P 500 Total Return, 1988+)")
    st.markdown("- `SPYTR` → `^SP500TR` (S&P 500 Total Return, 1988+), `QQQTR` → `^NDX` (NASDAQ 100, 1985+)")
    st.markdown("- `TLTETF` → `TLT` (20+ Year Treasury ETF, 2002+), `IEFETF` → `IEF` (7-10 Year Treasury ETF, 2002+)")
    st.markdown("- `ZROZX` → `ZROZ` (25+ Year Zero Coupon Treasury, 2009+), `GOVZTR` → `GOVZ` (25+ Year Treasury STRIPS, 2020+)")
    st.markdown("- `TNX` → `^TNX` (10Y Treasury Yield, 1962+), `TYX` → `^TYX` (30Y Treasury Yield, 1977+)")
    st.markdown("- `TBILL3M` → `^IRX` (3M Treasury Yield, 1960+), `SHY` → `SHY` (1-3 Year Treasury ETF, 2002+)")
    st.markdown("- `ZEROX` (Cash doing nothing - zero return), `GOLDX` → `GC=F` (Gold Futures, 2000+), `XAU` → `^XAU` (Gold & Silver Index, 1983+)")
    st.markdown("**🇨🇦 Canadian Ticker Mappings:** USD OTC → Canadian TSX (for better data quality):")
    st.markdown("- `MDALF` → `MDA.TO` (MDA Ltd), `KRKNF` → `PNG.TO` (Kraken Robotics)")
    st.markdown("- `CNSWF` → `TOI.TO` (Constellation Software), `TOITF` → `TOI.TO` (Constellation Software)")
    st.markdown("- `LMGIF` → `LMN.TO` (Lumine Group), `DLMAF` → `DOL.TO` (Dollarama)")
    st.markdown("- `FRFHF` → `FFH.TO` (Fairfax Financial)")


with st.expander("⚡ Leverage & Expense Ratio Guide", expanded=False):
    st.markdown("""
    **Leverage Format:** Use `TICKER?L=N` where N is the leverage multiplier
    **Expense Ratio Format:** Use `TICKER?E=N` where N is the annual expense ratio percentage
    
    **Examples:**
    - **SPY?L=2** - 2x leveraged S&P 500
    - **QQQ?L=3?E=0.84** - 3x leveraged NASDAQ-100 with 0.84% expense ratio (like TQQQ)
    - **QQQ?E=1** - QQQ with 1% expense ratio
    - **TLT?L=2?E=0.5** - 2x leveraged 20+ Year Treasury with 0.5% expense ratio
    - **SPY?E=2?L=3** - Order doesn't matter: 3x leveraged S&P 500 with 2% expense ratio
    - **QQQ?E=5** - QQQ with 5% expense ratio (high fee for testing)
    
    **Parameter Combinations:**
    - **QQQ?L=3?E=0.84** - Simulates TQQQ (3x QQQ with 0.84% expense ratio)
    - **SPY?L=2?E=0.95** - Simulates SSO (2x SPY with 0.95% expense ratio)
    - **QQQ?E=0.2** - Simulates QQQ with 0.2% expense ratio
    
    **Important Notes:**
    - **Daily Reset:** Leverage resets daily (like real leveraged ETFs)
    - **Cost Drag:** Includes daily cost drag = (leverage - 1) × risk-free rate
    - **Expense Drag:** Daily expense ratio drag = annual_expense_ratio / 365.25
    - **Volatility Decay:** High volatility can cause significant decay over time
    - **Risk Warning:** Leveraged products are high-risk and can lose value quickly
    
    **Real Leveraged ETFs for Reference:**
    - **SSO** - 2x S&P 500 (ProShares)
    - **UPRO** - 3x S&P 500 (ProShares)
    - **TQQQ** - 3x NASDAQ-100 (ProShares)
    - **TMF** - 3x 20+ Year Treasury (Direxion)
    
    **Best Practices:**
    - Use for short-term strategies or hedging
    - Avoid holding for extended periods due to decay
    - Consider the underlying asset's volatility
    - Monitor risk-free rate changes affecting cost drag
    """)

# Bulk ticker input section
with st.expander("📝 Bulk Ticker Input", expanded=False):
    st.markdown("**Enter multiple tickers separated by spaces or commas:**")
    
    # Initialize bulk ticker input in session state
    if 'alloc_bulk_tickers' not in st.session_state:
        st.session_state.alloc_bulk_tickers = ""
    
    # Auto-populate bulk ticker input with current tickers (only if user hasn't entered anything)
    portfolio_index = st.session_state.alloc_active_portfolio_index
    current_tickers = [stock['ticker'] for stock in st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'] if stock['ticker']]
    if current_tickers:
        current_ticker_string = ' '.join(current_tickers)
        # Only auto-populate if the bulk ticker field is empty or matches the current portfolio
        if not st.session_state.alloc_bulk_tickers or st.session_state.alloc_bulk_tickers == current_ticker_string:
            st.session_state.alloc_bulk_tickers = current_ticker_string
    
    # Text area for bulk ticker input
    bulk_tickers = st.text_area(
        "Tickers (e.g., SPY QQQ GLD TLT or SPY,QQQ,GLD,TLT)",
        value=st.session_state.alloc_bulk_tickers,
        key="alloc_bulk_ticker_input",
        height=100,
        help="Enter ticker symbols separated by spaces or commas. Choose 'Replace All' to replace all tickers or 'Add to Existing' to add new tickers."
    )
    
    # Action buttons
    col_replace, col_add, col_fetch, col_copy = st.columns([1, 1, 1, 1])
    
    with col_replace:
        if st.button("🔄 Replace All", key="alloc_fill_tickers_btn", type="secondary"):
            if bulk_tickers.strip():
                # Parse tickers (split by comma or space)
                ticker_list = []
            for ticker in bulk_tickers.replace(',', ' ').split():
                ticker = ticker.strip().upper()
                if ticker:
                        # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
                        if ticker == 'BRK.B':
                            ticker = 'BRK-B'
                        elif ticker == 'BRK.A':
                            ticker = 'BRK-A'
                        ticker_list.append(ticker)
            
            if ticker_list:
                portfolio_index = st.session_state.alloc_active_portfolio_index
                current_stocks = st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'].copy()
                
                # Replace tickers - new ones get 0% allocation
                new_stocks = []
                
                for i, ticker in enumerate(ticker_list):
                    # Resolve the alias to the actual Yahoo ticker
                    resolved_ticker = resolve_ticker_alias(ticker)
                    if i < len(current_stocks):
                        # Use existing allocation if available
                        new_stocks.append({
                            'ticker': resolved_ticker,  # Use resolved ticker
                            'allocation': current_stocks[i]['allocation'],
                            'include_dividends': current_stocks[i]['include_dividends']
                        })
                    else:
                        # New tickers get 0% allocation
                        new_stocks.append({
                            'ticker': resolved_ticker,  # Use resolved ticker
                            'allocation': 0.0,
                            'include_dividends': True
                        })
                
                # Update the portfolio with new stocks
                st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'] = new_stocks
                
                # Update the active_portfolio reference to match session state
                active_portfolio['stocks'] = new_stocks
                
                # Clear any existing session state keys for individual ticker inputs to force refresh
                for key in list(st.session_state.keys()):
                    if key.startswith(f"alloc_ticker_{portfolio_index}_") or key.startswith(f"alloc_input_alloc_{portfolio_index}_"):
                        del st.session_state[key]
                
                    st.success(f"✅ Replaced all tickers with: {', '.join(ticker_list)}")
                st.info("💡 **Note:** Existing allocations preserved. Adjust allocations manually if needed.")
                
                # Force immediate rerun to refresh the UI
                st.rerun()
            else:
                st.warning("⚠️ No valid tickers found in input.")
    
    with col_add:
        if st.button("➕ Add to Existing", key="alloc_add_tickers_btn", type="secondary"):
            if bulk_tickers.strip():
                # Parse tickers (split by comma or space)
                ticker_list = []
                for ticker in bulk_tickers.replace(',', ' ').split():
                    ticker = ticker.strip().upper()
                    if ticker:
                        # Special conversion for Berkshire Hathaway tickers for Yahoo Finance compatibility
                        if ticker == 'BRK.B':
                            ticker = 'BRK-B'
                        elif ticker == 'BRK.A':
                            ticker = 'BRK-A'
                        ticker_list.append(ticker)
                
                if ticker_list:
                    portfolio_index = st.session_state.alloc_active_portfolio_index
                    current_stocks = st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'].copy()
                    
                    # Add new tickers to existing ones
                    for ticker in ticker_list:
                        # Resolve the alias to the actual Yahoo ticker
                        resolved_ticker = resolve_ticker_alias(ticker)
                        # Check if ticker already exists
                        ticker_exists = any(stock['ticker'] == resolved_ticker for stock in current_stocks)
                        if not ticker_exists:
                            current_stocks.append({
                                'ticker': resolved_ticker,  # Use resolved ticker
                                'allocation': 0.0,
                                'include_dividends': True
                            })
                    
                    # Update the portfolio with combined stocks
                    st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'] = current_stocks
                    
                    # Update the active_portfolio reference to match session state
                    active_portfolio['stocks'] = current_stocks
                    
                    # Clear any existing session state keys for individual ticker inputs to force refresh
                    for key in list(st.session_state.keys()):
                        if key.startswith(f"alloc_ticker_{portfolio_index}_") or key.startswith(f"alloc_input_alloc_{portfolio_index}_"):
                            del st.session_state[key]
                    
                    st.success(f"✅ Added new tickers: {', '.join(ticker_list)}")
                    st.info("💡 **Note:** New tickers added with 0% allocation. Adjust allocations manually if needed.")
                    
                    # Force immediate rerun to refresh the UI
                    st.rerun()
                else:
                    st.warning("⚠️ No valid tickers found in input.")
    
    with col_fetch:
        if st.button("🔍 Fetch Tickers", key="alloc_fetch_tickers_btn", type="secondary"):
            # Get current tickers from the active portfolio
            portfolio_index = st.session_state.alloc_active_portfolio_index
            current_tickers = [stock['ticker'] for stock in st.session_state.alloc_portfolio_configs[portfolio_index]['stocks'] if stock['ticker']]
            
            if current_tickers:
                # Update the bulk ticker input with current tickers
                current_ticker_string = ' '.join(current_tickers)
                st.session_state.alloc_bulk_tickers = current_ticker_string
                st.success(f"✅ Fetched {len(current_tickers)} tickers: {current_ticker_string}")
                st.rerun()
            else:
                st.warning("⚠️ No tickers found in the current portfolio.")
    
    with col_copy:
        if bulk_tickers.strip():
            # Create a custom button with direct copy functionality
            import streamlit.components.v1 as components
            
            # JavaScript function to copy and show feedback
            copy_js = f"""
            <script>
            function copyTickers() {{
                navigator.clipboard.writeText({json.dumps(bulk_tickers.strip())}).then(function() {{
                    // Show success feedback
                    const button = document.querySelector('#copy-tickers-btn');
                    const originalText = button.innerHTML;
                    button.innerHTML = '✅ Copied!';
                    button.style.backgroundColor = '#28a745';
                    setTimeout(function() {{
                        button.innerHTML = originalText;
                        button.style.backgroundColor = '';
                    }}, 2000);
                }}).catch(function(err) {{
                    alert('Failed to copy: ' + err);
                }});
            }}
            </script>
            <button id="copy-tickers-btn" onclick="copyTickers()" style="
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                width: 100%;
                font-size: 14px;
            ">📋 Copy</button>
            """
            components.html(copy_js, height=50)
        else:
            st.button("📋 Copy", key="alloc_copy_tickers_btn", type="secondary", disabled=True)
            st.warning("⚠️ No tickers to copy. Please enter some tickers first.")

# Leverage Summary Section
leveraged_tickers = []
for stock in active_portfolio['stocks']:
    if "?L=" in stock['ticker'] or "?E=" in stock['ticker']:
        try:
            base_ticker, leverage, expense_ratio = parse_ticker_parameters(stock['ticker'])
            leveraged_tickers.append((base_ticker, leverage))
        except:
            pass

if leveraged_tickers:
    st.markdown("---")
    st.markdown("### 🚀 Leverage Summary")
    
    # Get risk-free rate for drag calculation
    try:
        risk_free_rates = get_risk_free_rate_robust([pd.Timestamp.now()])
        daily_rf = risk_free_rates.iloc[0] if len(risk_free_rates) > 0 else 0.000105
        annual_rf = daily_rf * 365.25 * 100  # Convert daily to annual percentage
    except:
        daily_rf = 0.000105  # fallback
        annual_rf = 3.86  # fallback annual rate
    
    # Group by leverage level
    leverage_groups = {}
    for base_ticker, leverage in leveraged_tickers:
        if leverage not in leverage_groups:
            leverage_groups[leverage] = []
        leverage_groups[leverage].append(base_ticker)
    
    for leverage in sorted(leverage_groups.keys()):
        base_tickers = leverage_groups[leverage]
        daily_drag = (leverage - 1) * daily_rf * 100
        st.markdown(f"🚀 **{leverage}x leverage** on {', '.join(base_tickers)}")
        st.markdown(f"📉 **Daily drag:** {daily_drag:.3f}% (RF: {annual_rf:.2f}%)")

st.subheader("Strategy")
if "alloc_active_use_momentum" not in st.session_state:
    st.session_state["alloc_active_use_momentum"] = active_portfolio['use_momentum']
if "alloc_active_use_threshold" not in st.session_state:
    st.session_state["alloc_active_use_threshold"] = active_portfolio.get('use_minimal_threshold', False)
if "alloc_active_threshold_percent" not in st.session_state:
    st.session_state["alloc_active_threshold_percent"] = active_portfolio.get('minimal_threshold_percent', 4.0)
if "alloc_active_use_max_allocation" not in st.session_state:
    st.session_state["alloc_active_use_max_allocation"] = active_portfolio.get('use_max_allocation', False)
if "alloc_active_max_allocation_percent" not in st.session_state:
    st.session_state["alloc_active_max_allocation_percent"] = active_portfolio.get('max_allocation_percent', 20.0)
# Only show momentum strategy if targeted rebalancing is disabled
if not active_portfolio.get('use_targeted_rebalancing', False):
    st.checkbox("Use Momentum Strategy", key="alloc_active_use_momentum", on_change=update_use_momentum, help="Enables momentum-based weighting of stocks.")
else:
    # Hide momentum strategy when targeted rebalancing is enabled
    st.session_state["alloc_active_use_momentum"] = False

if active_portfolio['use_momentum']:
    st.markdown("---")
    col_mom_options, col_beta_vol = st.columns(2)
    with col_mom_options:
        st.markdown("**Momentum Strategy Options**")
        
        # Initialize or sync with imported values
        momentum_key = f"momentum_strategy_{st.session_state.alloc_active_portfolio_index}"
        negative_momentum_key = f"negative_momentum_strategy_{st.session_state.alloc_active_portfolio_index}"
        
        # FORCE sync with session state if it was updated by JSON import
        if 'alloc_active_momentum_strategy' in st.session_state:
            st.session_state[momentum_key] = st.session_state['alloc_active_momentum_strategy']
            active_portfolio['momentum_strategy'] = st.session_state['alloc_active_momentum_strategy']
        
        if 'alloc_active_negative_momentum_strategy' in st.session_state:
            st.session_state[negative_momentum_key] = st.session_state['alloc_active_negative_momentum_strategy']
            active_portfolio['negative_momentum_strategy'] = st.session_state['alloc_active_negative_momentum_strategy']
        
        momentum_strategy = st.selectbox(
            "Momentum strategy when NOT all negative:",
            ["Classic", "Relative Momentum"],
            index=["Classic", "Relative Momentum"].index(active_portfolio.get('momentum_strategy', 'Classic')),
            key=momentum_key
        )
        negative_momentum_strategy = st.selectbox(
            "Strategy when ALL momentum scores are negative:",
            ["Cash", "Equal weight", "Relative momentum"],
            index=["Cash", "Equal weight", "Relative momentum"].index(active_portfolio.get('negative_momentum_strategy', 'Cash')),
            key=negative_momentum_key
        )
        active_portfolio['momentum_strategy'] = momentum_strategy
        active_portfolio['negative_momentum_strategy'] = negative_momentum_strategy
        st.markdown("💡 **Note:** These options control how weights are assigned based on momentum scores.")

    with col_beta_vol:
        if "alloc_active_calc_beta" not in st.session_state:
            st.session_state["alloc_active_calc_beta"] = active_portfolio.get('calc_beta', False)
        st.checkbox("Include Beta in momentum weighting", key="alloc_active_calc_beta", on_change=update_calc_beta, help="Incorporates a stock's Beta (volatility relative to the benchmark) into its momentum score.")
        if st.session_state.get('alloc_active_calc_beta', False):
            if "alloc_active_beta_window" not in st.session_state:
                st.session_state["alloc_active_beta_window"] = active_portfolio['beta_window_days']
            if "alloc_active_beta_exclude" not in st.session_state:
                st.session_state["alloc_active_beta_exclude"] = active_portfolio['exclude_days_beta']
            st.number_input("Beta Lookback (days)", min_value=1, key="alloc_active_beta_window", on_change=update_beta_window)
            st.number_input("Beta Exclude (days)", min_value=0, key="alloc_active_beta_exclude", on_change=update_beta_exclude)
            if st.button("Reset Beta", on_click=reset_beta_callback):
                pass
        if "alloc_active_calc_vol" not in st.session_state:
            st.session_state["alloc_active_calc_vol"] = active_portfolio.get('calc_volatility', False)
        st.checkbox("Include Volatility in momentum weighting", key="alloc_active_calc_vol", on_change=update_calc_vol, help="Incorporates a stock's volatility (standard deviation of returns) into its momentum score.")
        if st.session_state.get('alloc_active_calc_vol', False):
            if "alloc_active_vol_window" not in st.session_state:
                st.session_state["alloc_active_vol_window"] = active_portfolio['vol_window_days']
            if "alloc_active_vol_exclude" not in st.session_state:
                st.session_state["alloc_active_vol_exclude"] = active_portfolio['exclude_days_vol']
            st.number_input("Volatility Lookback (days)", min_value=1, key="alloc_active_vol_window", on_change=update_vol_window)
            st.number_input("Volatility Exclude (days)", min_value=0, key="alloc_active_vol_exclude", on_change=update_vol_exclude)
            if st.button("Reset Volatility", on_click=reset_vol_callback):
                pass
    
    
    st.markdown("---")
    st.subheader("Momentum Windows")
    col_reset, col_norm, col_addrem = st.columns([0.4, 0.4, 0.2])
    with col_reset:
        if st.button("Reset Momentum Windows", on_click=reset_momentum_windows_callback):
            pass
    with col_norm:
        if st.button("Normalize Weights to 100%", on_click=normalize_momentum_weights_callback):
            pass
    with col_addrem:
        if st.button("Add Window", on_click=add_momentum_window_callback):
            pass
        if st.button("Remove Window", on_click=remove_momentum_window_callback):
            pass

    total_weight = sum(w['weight'] for w in active_portfolio['momentum_windows'])
    if abs(total_weight - 1.0) > 0.001:
        st.warning(f"Current total weight is {total_weight*100:.2f}%, not 100%. Click 'Normalize Weights' to fix.")
    else:
        st.success(f"Total weight is {total_weight*100:.2f}%.")

    def update_momentum_lookback(index):
        key = f"alloc_lookback_active_{st.session_state.alloc_active_portfolio_index}_{index}"
        momentum_windows = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('momentum_windows', [])
        if index < len(momentum_windows):
            momentum_windows[index]['lookback'] = st.session_state.get(key, None)

    def update_momentum_exclude(index):
        key = f"alloc_exclude_active_{st.session_state.alloc_active_portfolio_index}_{index}"
        momentum_windows = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('momentum_windows', [])
        if index < len(momentum_windows):
            momentum_windows[index]['exclude'] = st.session_state.get(key, None)
    
    def update_momentum_weight(index):
        key = f"alloc_weight_input_active_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        momentum_windows = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('momentum_windows', [])
        if index < len(momentum_windows):
            momentum_windows[index]['weight'] = val / 100.0

    # Allow the user to remove momentum windows down to zero.
    # Previously the UI forced a minimum of 3 windows which prevented removing them.
    # If no windows exist, show an informational message and allow adding via the button.
    if len(active_portfolio.get('momentum_windows', [])) == 0:
        st.info("No momentum windows configured. Click 'Add Window' to create momentum lookback windows.")
    col_headers = st.columns(3)
    with col_headers[0]:
        st.markdown("**Lookback (days)**")
    with col_headers[1]:
        st.markdown("**Exclude (days)**")
    with col_headers[2]:
        st.markdown("**Weight %**")

    for j in range(len(active_portfolio.get('momentum_windows', []))):
        with st.container():
            col_mw1, col_mw2, col_mw3 = st.columns(3)
            lookback_key = f"alloc_lookback_active_{st.session_state.alloc_active_portfolio_index}_{j}"
            exclude_key = f"alloc_exclude_active_{st.session_state.alloc_active_portfolio_index}_{j}"
            weight_key = f"alloc_weight_input_active_{st.session_state.alloc_active_portfolio_index}_{j}"
            
            # Initialize session state values if not present
            if lookback_key not in st.session_state:
                st.session_state[lookback_key] = int(active_portfolio['momentum_windows'][j]['lookback'])
            if exclude_key not in st.session_state:
                st.session_state[exclude_key] = int(active_portfolio['momentum_windows'][j]['exclude'])
            if weight_key not in st.session_state:
                # Sanitize weight to prevent StreamlitValueAboveMaxError
                weight = active_portfolio['momentum_windows'][j]['weight']
                if isinstance(weight, (int, float)):
                    # If weight is already a percentage (e.g., 50 for 50%), use it directly
                    if weight > 1.0:
                        # Cap at 100% and use as percentage
                        weight_percentage = min(weight, 100.0)
                    else:
                        # Convert decimal to percentage
                        weight_percentage = weight * 100.0
                else:
                    # Invalid weight, set to default
                    weight_percentage = 10.0
                st.session_state[weight_key] = int(weight_percentage)
            
            with col_mw1:
                st.number_input(f"Lookback {j+1}", min_value=1, key=lookback_key, label_visibility="collapsed", on_change=update_momentum_lookback, args=(j,))
            with col_mw2:
                st.number_input(f"Exclude {j+1}", min_value=0, key=exclude_key, label_visibility="collapsed", on_change=update_momentum_exclude, args=(j,))
            with col_mw3:
                st.number_input(f"Weight {j+1}", min_value=0, max_value=100, step=1, format="%d", key=weight_key, label_visibility="collapsed", on_change=update_momentum_weight, args=(j,))
else:
    # Don't clear momentum_windows - they should persist when momentum is disabled
    # so they're available when momentum is re-enabled or for variant generation
    
    active_portfolio['momentum_windows'] = []

# Minimal Threshold Filter Section (only available when momentum is enabled)
if active_portfolio['use_momentum']:
    st.markdown("---")
    st.subheader("Minimal Threshold Filter")

    # ALWAYS sync threshold settings from portfolio (not just if not present)
    # Only sync if session state doesn't exist or if we're not in the middle of an import
    if "alloc_active_use_threshold" not in st.session_state or not st.session_state.get('alloc_rerun_flag', False):
        st.session_state["alloc_active_use_threshold"] = active_portfolio.get('use_minimal_threshold', False)
        st.session_state["alloc_active_threshold_percent"] = active_portfolio.get('minimal_threshold_percent', 4.0)

    st.checkbox(
        "Enable Minimal Threshold Filter", 
        key="alloc_active_use_threshold", 
        on_change=update_use_threshold,
        help="Exclude stocks with allocations below the threshold percentage and normalize remaining allocations to 100%"
    )

    if st.session_state.get("alloc_active_use_threshold", False):
        st.number_input(
            "Minimal Threshold (%)", 
            min_value=0.1, 
            max_value=50.0, 
            step=0.1,
            key="alloc_active_threshold_percent", 
            on_change=update_threshold_percent,
            help="Stocks with allocations below this percentage will be excluded and their weight redistributed to remaining stocks"
        )

    # Maximum Allocation Filter Section (only available when momentum is enabled)
    st.markdown("---")
    st.subheader("Maximum Allocation Filter")

    # ALWAYS sync maximum allocation settings from portfolio (not just if not present)
    # Only sync if session state doesn't exist or if we're not in the middle of an import
    if "alloc_active_use_max_allocation" not in st.session_state or not st.session_state.get('alloc_rerun_flag', False):
        st.session_state["alloc_active_use_max_allocation"] = active_portfolio.get('use_max_allocation', False)
        st.session_state["alloc_active_max_allocation_percent"] = active_portfolio.get('max_allocation_percent', 20.0)

    st.checkbox(
        "Enable Maximum Allocation Filter", 
        key="alloc_active_use_max_allocation", 
        on_change=update_use_max_allocation,
        help="Cap individual stock allocations at the maximum percentage and redistribute excess weight to other stocks"
    )

    if st.session_state.get("alloc_active_use_max_allocation", False):
        st.number_input(
            "Maximum Allocation (%)", 
            min_value=1.0, 
            max_value=100.0, 
            step=0.1,
            key="alloc_active_max_allocation_percent", 
            on_change=update_max_allocation_percent,
            help="Individual stocks will be capped at this maximum allocation percentage"
        )

# MA Filter Section (SMA/EMA) - EXACTLY SAME AS PAGE 1
# Only show MA filter if targeted rebalancing is disabled
if not st.session_state.get("alloc_active_use_targeted_rebalancing", False):
    st.markdown("---")
    st.subheader("MA Filter")

    # Initialize MA filter state
    if "alloc_active_use_sma_filter" not in st.session_state:
        st.session_state["alloc_active_use_sma_filter"] = active_portfolio.get('use_sma_filter', False)

    st.checkbox(
        "Enable MA Filter", 
        key="alloc_active_use_sma_filter",
        on_change=update_use_sma_filter,
        help="Enable the Moving Average filter"
    )

    # MA controls (only show when MA filter is enabled)
    if st.session_state.get("alloc_active_use_sma_filter", False):
        col_ma_type, col_ma_window = st.columns([1, 1])
        
        with col_ma_type:
            # MA type selector
            # Initialize MA type state
            if "alloc_active_ma_type" not in st.session_state:
                st.session_state["alloc_active_ma_type"] = active_portfolio.get('ma_type', 'SMA')
            
            ma_type = st.selectbox("MA Type", 
                                   options=["SMA", "EMA"], 
                                   index=0 if st.session_state.get("alloc_active_ma_type", "SMA") == "SMA" else 1,
                                   key="alloc_active_ma_type",
                                   help="Select the type of moving average")
            active_portfolio['ma_type'] = ma_type
        
        with col_ma_window:
            # MA window input
            # Initialize MA window state
            if "alloc_active_sma_window" not in st.session_state:
                st.session_state["alloc_active_sma_window"] = active_portfolio.get('sma_window', 200)
            
            ma_window = st.number_input(
                "MA Window (days)",
                min_value=10,
                max_value=500,
                value=st.session_state.get("alloc_active_sma_window", 200),
                step=10,
                key="alloc_active_sma_window",
                help="Number of days for the Moving Average calculation"
            )
            active_portfolio['sma_window'] = ma_window

    # Store MA filter state
    active_portfolio['use_sma_filter'] = st.session_state.get('alloc_active_use_sma_filter', False)
else:
    # Hide MA filter when targeted rebalancing is enabled
    # Don't modify session state directly - let the checkbox handle it
    active_portfolio['use_sma_filter'] = False

# Targeted Rebalancing Section (COPIED FROM PAGE 4)
# Only show targeted rebalancing if momentum AND MA filter are disabled
if not st.session_state.get('alloc_active_use_momentum', False) and not st.session_state.get("alloc_active_use_sma_filter", False):
    st.markdown("---")
    st.subheader("Targeted Rebalancing")

    # Initialize targeted rebalancing state (COPIED FROM PAGE 4)
    if "alloc_active_use_targeted_rebalancing" not in st.session_state:
        st.session_state["alloc_active_use_targeted_rebalancing"] = active_portfolio.get('use_targeted_rebalancing', False)

    st.checkbox(
        "Enable Targeted Rebalancing", 
        key="alloc_active_use_targeted_rebalancing", 
        on_change=update_use_targeted_rebalancing,
        help="Automatically rebalance when ticker allocations exceed min/max thresholds"
    )
    
    # Update active portfolio with current targeted rebalancing state
    active_portfolio['use_targeted_rebalancing'] = st.session_state.get("alloc_active_use_targeted_rebalancing", False)
else:
    # Hide targeted rebalancing when momentum or MA filter is enabled
    # Don't modify session state directly - let the checkbox handle it
    active_portfolio['use_targeted_rebalancing'] = False

if st.session_state.get("alloc_active_use_targeted_rebalancing", False):
    st.markdown("**Configure allocation limits for each ticker:**")
    
    # Get current tickers
    stocks_list = active_portfolio.get('stocks', [])
    current_tickers = [s['ticker'] for s in stocks_list if s.get('ticker')]
    
    if current_tickers:
        # Initialize settings if not exists
        if 'targeted_rebalancing_settings' not in active_portfolio:
            active_portfolio['targeted_rebalancing_settings'] = {}
        
        # Create columns for ticker settings
        cols = st.columns(min(len(current_tickers), 3))
        
        for i, ticker in enumerate(current_tickers):
            with cols[i % 3]:
                st.markdown(f"**{ticker}**")
                
                # Initialize default settings for this ticker
                if ticker not in active_portfolio['targeted_rebalancing_settings']:
                    active_portfolio['targeted_rebalancing_settings'][ticker] = {
                        'enabled': False,
                        'min_allocation': 0.0,
                        'max_allocation': 100.0
                    }
                
                # Enable/disable checkbox
                enabled = st.checkbox(
                    "Enable", 
                    value=active_portfolio['targeted_rebalancing_settings'][ticker]['enabled'],
                    key=f"targeted_enabled_{ticker}",
                    help=f"Enable targeted rebalancing for {ticker}"
                )
                active_portfolio['targeted_rebalancing_settings'][ticker]['enabled'] = enabled
                
                if enabled:
                    # Max allocation (on top)
                    max_alloc = st.number_input(
                        "Max %", 
                        min_value=0.0, 
                        max_value=100.0, 
                        step=0.1,
                        value=active_portfolio['targeted_rebalancing_settings'][ticker]['max_allocation'],
                        key=f"targeted_max_{ticker}",
                        help=f"Maximum allocation percentage for {ticker}"
                    )
                    active_portfolio['targeted_rebalancing_settings'][ticker]['max_allocation'] = max_alloc
                    
                    # Min allocation (below)
                    min_alloc = st.number_input(
                        "Min %", 
                        min_value=0.0, 
                        max_value=100.0, 
                        step=0.1,
                        value=active_portfolio['targeted_rebalancing_settings'][ticker]['min_allocation'],
                        key=f"targeted_min_{ticker}",
                        help=f"Minimum allocation percentage for {ticker}"
                    )
                    active_portfolio['targeted_rebalancing_settings'][ticker]['min_allocation'] = min_alloc
                    
                    # Validation
                    if min_alloc >= max_alloc:
                        st.error(f"Min must be less than Max for {ticker}")
    else:
        st.info("Add tickers to configure targeted rebalancing settings.")

with st.expander("JSON Configuration (Copy & Paste)", expanded=False):
    # Clean portfolio config for export by removing unused settings
    cleaned_config = active_portfolio.copy()
    cleaned_config.pop('use_relative_momentum', None)
    cleaned_config.pop('equal_if_all_negative', None)
    
    # Ensure targeted rebalancing settings are included
    cleaned_config['use_targeted_rebalancing'] = active_portfolio.get('use_targeted_rebalancing', False)
    cleaned_config['targeted_rebalancing_settings'] = active_portfolio.get('targeted_rebalancing_settings', {})
    
    # Ensure MA filter settings are included
    cleaned_config['use_sma_filter'] = st.session_state.get('alloc_active_use_sma_filter', False)
    cleaned_config['sma_window'] = st.session_state.get('alloc_active_sma_window', 200)
    cleaned_config['ma_type'] = st.session_state.get('alloc_active_ma_type', 'SMA')
    
    # Also update the active portfolio to keep it in sync
    active_portfolio['use_sma_filter'] = st.session_state.get('alloc_active_use_sma_filter', False)
    active_portfolio['sma_window'] = st.session_state.get('alloc_active_sma_window', 200)
    active_portfolio['ma_type'] = st.session_state.get('alloc_active_ma_type', 'SMA')
    
    config_json = json.dumps(cleaned_config, indent=4)
    st.code(config_json, language='json')
    # Fixed JSON copy button
    import streamlit.components.v1 as components
    copy_html = f"""
    <button onclick='navigator.clipboard.writeText({json.dumps(config_json)});' style='margin-bottom:10px;'>Copy to Clipboard</button>
    """
    components.html(copy_html, height=40)
    
    # Add PDF download button for JSON
    def generate_json_pdf(custom_name=""):
        """Generate a PDF with pure JSON content only for easy CTRL+A / CTRL+V copying."""
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Preformatted
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        import io
        from datetime import datetime
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Add proper PDF metadata
        portfolio_name = active_portfolio.get('name', 'Portfolio')
        
        # Use custom name if provided, otherwise use portfolio name
        if custom_name.strip():
            title = f"Allocations - {custom_name.strip()} - JSON Configuration"
            subject = f"JSON Configuration: {custom_name.strip()}"
        else:
            title = f"Allocations - {portfolio_name} - JSON Configuration"
            subject = f"JSON Configuration for {portfolio_name}"
        
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=36, 
            leftMargin=36, 
            topMargin=36, 
            bottomMargin=36,
            title=title,
            author="Portfolio Backtest System",
            subject=subject,
            creator="Allocations Application"
        )
        story = []
        
        # Pure JSON style - just monospace text
        json_style = ParagraphStyle(
            'PureJSONStyle',
            fontName='Courier',
            fontSize=10,
            leading=12,
            leftIndent=0,
            rightIndent=0,
            spaceAfter=0,
            spaceBefore=0
        )
        
        # Add only the JSON content - no headers, no instructions, just pure JSON
        json_lines = config_json.split('\n')
        for line in json_lines:
            story.append(Preformatted(line, json_style))
        
        # Build PDF
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    # Optional custom PDF name for individual portfolio
    custom_individual_pdf_name = st.text_input(
        "📝 Custom Portfolio JSON PDF Name (optional):", 
        value="",
        placeholder=f"e.g., {active_portfolio.get('name', 'Portfolio')} Allocation Config, Asset Setup Analysis",
        help="Leave empty to use automatic naming based on portfolio name",
        key="alloc_individual_custom_pdf_name"
    )
    
    if st.button("📄 Download JSON as PDF", help="Download a PDF containing the JSON configuration for easy copying", key="alloc_json_pdf_btn"):
        try:
            pdf_data = generate_json_pdf(custom_individual_pdf_name)
            
            # Generate filename based on custom name or default
            if custom_individual_pdf_name.strip():
                clean_name = custom_individual_pdf_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
                filename = f"{clean_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            else:
                filename = f"allocations_config_{active_portfolio.get('name', 'portfolio').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            st.download_button(
                label="💾 Download Allocations JSON PDF",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf",
                key="alloc_json_pdf_download",
            )
            st.success("PDF generated successfully! Click the download button above.")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
    

    st.text_area("Paste JSON Here to Update Portfolio", key="alloc_paste_json_text", height=200)
    st.button("Update with Pasted JSON", on_click=paste_json_callback)
    
    # Add PDF drag and drop functionality
    st.markdown("**OR** 📎 **Drag & Drop JSON PDF:**")
    
    def extract_json_from_pdf_alloc(pdf_file):
        """Extract JSON content from a PDF file."""
        try:
            # Try pdfplumber first (more reliable)
            try:
                import pdfplumber
                import io
                
                # Read PDF content with pdfplumber
                pdf_bytes = io.BytesIO(pdf_file.read())
                text_content = ""
                
                with pdfplumber.open(pdf_bytes) as pdf:
                    for page in pdf.pages:
                        text_content += page.extract_text() or ""
                        
            except ImportError:
                # Fallback to PyPDF2 if pdfplumber not available
                try:
                    import PyPDF2
                    import io
                    
                    # Reset file pointer
                    pdf_file.seek(0)
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
                    
                    # Extract text from all pages
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text()
                        
                except ImportError:
                    return None, "PDF extraction libraries not available. Please install 'pip install PyPDF2' or 'pip install pdfplumber'"
            
            # Clean up the text and try to parse as JSON
            cleaned_text = text_content.strip()
            
            # Try to parse as JSON
            import json
            json_data = json.loads(cleaned_text)
            return json_data, None
            
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON in PDF: {str(e)}"
        except Exception as e:
            return None, str(e)
    
    uploaded_pdf = st.file_uploader(
        "Drop your JSON PDF here", 
        type=['pdf'], 
        help="Upload a JSON PDF file generated by this app to automatically load the configuration",
        key="alloc_individual_pdf_upload"
    )
    
    if uploaded_pdf is not None:
        json_data, error = extract_json_from_pdf_alloc(uploaded_pdf)
        if json_data:
            # Store the extracted JSON in a different session state key to avoid widget conflicts
            st.session_state["alloc_extracted_json"] = json.dumps(json_data, indent=4)
            st.success(f"✅ Successfully extracted JSON from {uploaded_pdf.name}")
            st.info("👇 Click the button below to load the JSON into the text area.")
            def load_extracted_json():
                st.session_state["alloc_paste_json_text"] = st.session_state["alloc_extracted_json"]
            
            st.button("📋 Load Extracted JSON", key="load_extracted_json", on_click=load_extracted_json)
        else:
            st.error(f"❌ Failed to extract JSON from PDF: {error}")
            st.info("💡 Make sure the PDF contains valid JSON content (generated by this app)")

# Validation constants
_TOTAL_TOL = 1.0
_ALLOC_TOL = 1.0

# Clear all portfolios button - quick access for single portfolio pages
if st.sidebar.button("🗑️ Clear All Portfolios", key="alloc_clear_all_portfolios_immediate", 
                    help="Delete ALL portfolios and create a blank one", use_container_width=True):
    # Clear all portfolios and create a single blank portfolio
    st.session_state.alloc_portfolio_configs = [{
        'name': 'New Portfolio 1',
        'stocks': [],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 0,
        'added_frequency': 'none',
        'rebalancing_frequency': 'Monthly',
        'start_with': 'oldest',
        'first_rebalance_strategy': 'rebalancing_date',
        'use_momentum': False,
        'momentum_strategy': 'Classic',
        'negative_momentum_strategy': 'Cash',
        'momentum_windows': [
            {"lookback": 365, "exclude": 30, "weight": 1.0}
        ],
        'calc_beta': False,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'calc_volatility': False,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
        'use_minimal_threshold': False,
        'minimal_threshold_percent': 4.0,
        'use_max_allocation': False,
        'max_allocation_percent': 20.0,
        'collect_dividends_as_cash': False,
        'start_date_user': None,
        'end_date_user': None,
        'fusion_portfolio': {'enabled': False, 'selected_portfolios': [], 'allocations': {}}
    }]
    st.session_state.alloc_active_portfolio_index = 0
    st.success("✅ All portfolios cleared! Created 'New Portfolio 1'")
    st.rerun()

# Clear All Outputs Function
def clear_all_outputs():
    """Clear all backtest results and outputs while preserving portfolio configurations"""
    # Clear all result data
    st.session_state.multi_all_results = None
    st.session_state.multi_all_allocations = None
    st.session_state.multi_all_metrics = None
    st.session_state.multi_backtest_all_drawdowns = None
    st.session_state.multi_backtest_stats_df_display = None
    st.session_state.multi_backtest_all_years = None
    st.session_state.multi_backtest_portfolio_key_map = {}
    st.session_state.multi_backtest_ran = False
    
    # Clear allocations page specific data
    st.session_state.alloc_all_allocations = None
    st.session_state.alloc_all_metrics = None
    st.session_state.alloc_snapshot_data = None
    
    # Clear any processing flags
    for key in list(st.session_state.keys()):
        if key.startswith("processing_portfolio_"):
            del st.session_state[key]
    
    # Clear any stored data
    if 'raw_data' in st.session_state:
        del st.session_state['raw_data']
    
    st.success("✅ All outputs cleared! Portfolio configurations preserved.")

# Clear All Outputs Button
if st.sidebar.button("🗑️ Clear All Outputs", type="secondary", help="Clear all charts and results while keeping portfolio configurations", use_container_width=True):
    clear_all_outputs()
    st.rerun()

# Cancel Run Button
if st.sidebar.button("🛑 Cancel Run", type="secondary", help="Stop current backtest execution gracefully", use_container_width=True):
    st.session_state.hard_kill_requested = True
    st.toast("🛑 **CANCELLING** - Stopping backtest execution...", icon="⏹️")
    st.rerun()

# Emergency Kill Button
if st.sidebar.button("🚨 EMERGENCY KILL", type="secondary", help="Force terminate all processes immediately - Use for crashes, freezes, or unresponsive states", use_container_width=True):
    st.toast("🚨 **EMERGENCY KILL** - Force terminating all processes...", icon="💥")
    emergency_kill()

def calculate_minimum_lookback_days(portfolios):
    """
    Calculate the minimum data period needed for a backtest.
    Returns number of days to fetch (instead of period="max").
    """
    max_lookback = 0
    
    for config in portfolios:
        # Check momentum windows
        if config.get('use_momentum') and config.get('momentum_windows'):
            for window in config['momentum_windows']:
                lookback = window.get('lookback', 0)
                if lookback > max_lookback:
                    max_lookback = lookback
        
        # Check beta window
        if config.get('calc_beta'):
            beta_lookback = config.get('beta_window_days', 0)
            if beta_lookback > max_lookback:
                max_lookback = beta_lookback
        
        # Check volatility window
        if config.get('calc_volatility'):
            vol_lookback = config.get('vol_window_days', 0)
            if vol_lookback > max_lookback:
                max_lookback = vol_lookback
    
    # Add buffer: max lookback + 700 days extra for safety
    # This ensures we have enough data even with excludes, market holidays, and recent tickers
    total_days_needed = max_lookback + 700
    
    return total_days_needed

# Move Run Backtest to the first sidebar to make it conspicuous and separate from config
if st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True):
    # Reset kill request when starting new backtest
    st.session_state.hard_kill_requested = False
    print(f"[THRESHOLD DEBUG] Run Backtest button clicked!")
    
    # Update active portfolio config with current session state values before running backtest
    active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
    active_portfolio['use_minimal_threshold'] = st.session_state.get('alloc_active_use_threshold', False)
    active_portfolio['minimal_threshold_percent'] = st.session_state.get('alloc_active_threshold_percent', 4.0)
    active_portfolio['use_max_allocation'] = st.session_state.get('alloc_active_use_max_allocation', False)
    active_portfolio['max_allocation_percent'] = st.session_state.get('alloc_active_max_allocation_percent', 20.0)
    
    # Debug output
    print(f"[THRESHOLD DEBUG] Before backtest - Portfolio: {active_portfolio.get('name', 'Unknown')}")
    print(f"[THRESHOLD DEBUG] use_minimal_threshold: {active_portfolio.get('use_minimal_threshold', False)}")
    print(f"[THRESHOLD DEBUG] minimal_threshold_percent: {active_portfolio.get('minimal_threshold_percent', 2.0)}")
    print(f"[THRESHOLD DEBUG] use_max_allocation: {active_portfolio.get('use_max_allocation', False)}")
    print(f"[THRESHOLD DEBUG] max_allocation_percent: {active_portfolio.get('max_allocation_percent', 10.0)}")
    print(f"[THRESHOLD DEBUG] use_momentum: {active_portfolio.get('use_momentum', True)}")
    
    # Pre-backtest validation check for all portfolios
    # Prefer the allocations page configs when present so this page's edits are included
    configs_to_run = st.session_state.get('alloc_portfolio_configs', [])
    # Local alias used throughout the run block
    portfolio_list = configs_to_run
    # Set flag to show metrics after running backtest
    st.session_state['alloc_backtest_run'] = True
    valid_configs = True
    validation_errors = []
    
    for cfg in configs_to_run:
        if cfg['use_momentum']:
            total_momentum_weight = sum(w['weight'] for w in cfg['momentum_windows'])
            if abs(total_momentum_weight - 1.0) > (_TOTAL_TOL / 100.0):
                validation_errors.append(f"Portfolio '{cfg['name']}' has momentum enabled but the total momentum weight is {total_momentum_weight*100:.2f}% (must be 100%)")
                valid_configs = False
        else:
            valid_stocks_for_cfg = [s for s in cfg['stocks'] if s['ticker']]
            total_stock_allocation = sum(s['allocation'] for s in valid_stocks_for_cfg)
            if abs(total_stock_allocation - 1.0) > (_ALLOC_TOL / 100.0):
                validation_errors.append(f"Portfolio '{cfg['name']}' is not using momentum, but the total ticker allocation is {total_ticker_allocation*100:.2f}% (must be 100%)")
                valid_configs = False
                
    # Initialize progress bar
    progress_bar = st.empty()
    
    if not valid_configs:
        for error in validation_errors:
            st.error(error)
        # Don't run the backtest, but continue showing the UI
        progress_bar.empty()
        st.stop()
    else:
        # Show standalone popup notification that code is really running
        st.toast("**Code is running!** Starting backtest...", icon="🚀")
        
        progress_bar.progress(0, text="Starting backtest...")
        
        # Check for kill request
        check_kill_request()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        all_tickers = sorted(list(set(s['ticker'] for cfg in portfolio_list for s in cfg['stocks'] if s['ticker']) | set(cfg.get('benchmark_ticker') for cfg in portfolio_list if 'benchmark_ticker' in cfg)))
        all_tickers = [t for t in all_tickers if t]
        
        # CRITICAL FIX: Add base tickers for leveraged tickers to ensure dividend data is available
        base_tickers_to_add = set()
        for ticker in all_tickers:
            if "?L=" in ticker:
                base_ticker, leverage = parse_leverage_ticker(ticker)
                base_tickers_to_add.add(base_ticker)

        # Add base tickers to the list if they're not already there
        for base_ticker in base_tickers_to_add:
            if base_ticker not in all_tickers:
                all_tickers.append(base_ticker)
        
        # CRITICAL FIX: Add MA reference tickers to ensure they are downloaded
        ma_reference_tickers_to_add = set()
        for cfg in portfolio_list:
            # Only collect MA reference tickers if MA filter is enabled
            if cfg.get('use_sma_filter', False):
                for stock in cfg.get('stocks', []):
                    ma_ref_ticker = stock.get('ma_reference_ticker', '').strip()
                    # If a custom reference ticker is specified (not empty)
                    if ma_ref_ticker:
                        # Resolve aliases (e.g., TLTTR -> TLT_COMPLETE, GOLDX -> GOLD_COMPLETE)
                        resolved_ma_ref = resolve_ticker_alias(ma_ref_ticker)
                        if resolved_ma_ref not in all_tickers:
                            ma_reference_tickers_to_add.add(resolved_ma_ref)
        
        # Add MA reference tickers to the download list
        for ma_ref_ticker in ma_reference_tickers_to_add:
            if ma_ref_ticker not in all_tickers:
                all_tickers.append(ma_ref_ticker)
        
        print("Downloading data for all tickers...")
        data = {}
        invalid_tickers = []
        # OPTIMIZED: Batch download with smart fallback
        progress_text = f"Downloading data for {len(all_tickers)} tickers (batch mode)..."
        progress_bar.progress(0.1, text=progress_text)
        
        # Check for kill request before batch
        check_kill_request()
        
        # OPTIMIZATION: Calculate minimum lookback period needed
        min_days_needed = calculate_minimum_lookback_days(portfolio_list)
        print(f"📊 OPTIMIZATION: Only loading last {min_days_needed} days of data (instead of all history)")
        print(f"   Max lookback window: {min_days_needed - 365} days + 365 days buffer")
        
        # Convert days to period string for yfinance
        # yfinance accepts: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
        if min_days_needed <= 5:
            period_to_use = "5d"
        elif min_days_needed <= 30:
            period_to_use = "1mo"
        elif min_days_needed <= 90:
            period_to_use = "3mo"
        elif min_days_needed <= 180:
            period_to_use = "6mo"
        elif min_days_needed <= 365:
            period_to_use = "1y"
        elif min_days_needed <= 730:
            period_to_use = "2y"
        elif min_days_needed <= 1825:
            period_to_use = "5y"
        elif min_days_needed <= 3650:
            period_to_use = "10y"
        else:
            period_to_use = "max"  # For very long lookback windows
        
        print(f"   Using period: '{period_to_use}' for yfinance")
        
        # Use batch download for all tickers (much faster!)
        batch_results = get_multiple_tickers_batch(list(all_tickers), period=period_to_use, auto_adjust=False)
        
        # Process batch results
        for i, t in enumerate(all_tickers):
            progress_text = f"Processing {t} ({i+1}/{len(all_tickers)})..."
            progress_bar.progress((i + 1) / (len(all_tickers) + len(portfolio_list)), text=progress_text)
            
            hist = batch_results.get(t, pd.DataFrame())
            
            if hist.empty:
                print(f"No data available for {t}")
                invalid_tickers.append(t)
                continue
            
            try:
                # Force tz-naive for hist (like Backtest_Engine.py)
                hist = hist.copy()
                hist.index = hist.index.tz_localize(None)
                
                hist["Price_change"] = hist["Close"].pct_change(fill_method=None).fillna(0)
                data[t] = hist
                print(f"Data loaded for {t} from {data[t].index[0].date()}")
            except Exception as e:
                print(f"Error loading {t}: {e}")
                invalid_tickers.append(t)
        
        # Display invalid ticker warnings in Streamlit UI
        if invalid_tickers:
            # Separate portfolio tickers from benchmark tickers
            portfolio_tickers = set(s['ticker'] for cfg in portfolio_list for s in cfg['stocks'] if s['ticker'])
            benchmark_tickers = set(cfg.get('benchmark_ticker') for cfg in portfolio_list if 'benchmark_ticker' in cfg)
            
            portfolio_invalid = [t for t in invalid_tickers if t in portfolio_tickers]
            benchmark_invalid = [t for t in invalid_tickers if t in benchmark_tickers]
            
            if portfolio_invalid:
                st.warning(f"The following portfolio tickers are invalid and will be skipped: {', '.join(portfolio_invalid)}")
            if benchmark_invalid:
                st.warning(f"The following benchmark tickers are invalid and will be skipped: {', '.join(benchmark_invalid)}")
        
        # BULLETPROOF VALIDATION: Check for valid tickers and stop gracefully if none
        if not data:
            if invalid_tickers and len(invalid_tickers) == len(all_tickers):
                st.error(f"❌ **No valid tickers found!** All tickers are invalid: {', '.join(invalid_tickers)}. Please check your ticker symbols and try again.")
            else:
                st.error("❌ **No valid tickers found!** No data downloaded; aborting.")
            progress_bar.empty()
            st.session_state.alloc_all_results = None
            st.session_state.alloc_all_allocations = None
            st.session_state.alloc_all_metrics = None
            st.stop()
        else:
            # Check if any portfolio has valid tickers
            all_portfolio_tickers = set()
            for cfg in portfolio_list:
                portfolio_tickers = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                all_portfolio_tickers.update(portfolio_tickers)
            
            # Check for non-USD tickers and display currency warning
            check_currency_warning(list(all_portfolio_tickers))
            
            valid_portfolio_tickers = [t for t in all_portfolio_tickers if t in data]
            if not valid_portfolio_tickers:
                st.error(f"❌ **No valid tickers found!** No valid portfolio tickers found. Invalid tickers: {', '.join(all_portfolio_tickers)}. Please check your ticker symbols and try again.")
                progress_bar.empty()
                st.session_state.alloc_all_results = None
                st.session_state.alloc_all_allocations = None
                st.session_state.alloc_all_metrics = None
                st.stop()
            else:
                # Persist raw downloaded price data so later recomputations can access benchmark series
                st.session_state.alloc_raw_data = data
                common_start = max(df.first_valid_index() for df in data.values())
                common_end = min(df.last_valid_index() for df in data.values())
                print()
                all_results = {}
                all_drawdowns = {}
                all_stats = {}
                all_allocations = {}
                all_metrics = {}
                # Map portfolio index (0-based) to the unique key used in the result dicts
                portfolio_key_map = {}
                
                for i, cfg in enumerate(portfolio_list, start=1):
                    progress_text = f"Running backtest for {cfg.get('name', f'Backtest {i}')} ({i}/{len(portfolio_list)})..."
                    progress_bar.progress((len(all_tickers) + i) / (len(all_tickers) + len(portfolio_list)), text=progress_text)
                    name = cfg.get('name', f'Backtest {i}')
                    # Ensure unique key for storage to avoid overwriting when duplicate names exist
                    base_name = name
                    unique_name = base_name
                    suffix = 1
                    while unique_name in all_results or unique_name in all_allocations:
                        unique_name = f"{base_name} ({suffix})"
                        suffix += 1
                    print(f"\nRunning backtest {i}/{len(portfolio_list)}: {name}")
                    # Separate asset tickers from benchmark. Do NOT use benchmark when
                    # computing start/end/simulation dates or available-rebalance logic.
                    asset_tickers = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                    asset_tickers = [t for t in asset_tickers if t in data and t is not None]
                    benchmark_local = cfg.get('benchmark_ticker')
                    benchmark_in_data = benchmark_local if benchmark_local in data else None
                    tickers_for_config = asset_tickers
                    # Build the list of tickers whose data we will reindex (include benchmark if present)
                    data_tickers = list(asset_tickers)
                    if benchmark_in_data:
                        data_tickers.append(benchmark_in_data)
                    if not tickers_for_config:
                        # Check if this is because all tickers are invalid
                        original_asset_tickers = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                        missing_tickers = [t for t in original_asset_tickers if t not in data]
                        if missing_tickers:
                            print(f"  No available asset tickers for {name}; invalid tickers: {missing_tickers}. Skipping.")
                        else:
                            print(f"  No available asset tickers for {name}; skipping.")
                        continue
                    if cfg.get('start_with') == 'all':
                        # Start only when all asset tickers have data
                        final_start = max(data[t].first_valid_index() for t in tickers_for_config)
                    else:
                        # 'oldest' -> start at the earliest asset ticker date so assets can be added over time
                        final_start = min(data[t].first_valid_index() for t in tickers_for_config)
                    if cfg.get('start_date_user'):
                        user_start = pd.to_datetime(cfg['start_date_user'])
                        final_start = max(final_start, user_start)
                    # Preserve previous global alignment only for 'all' mode; do NOT force 'oldest' back to global latest
                    if cfg.get('start_with') == 'all':
                        final_start = max(final_start, common_start)
                    if cfg.get('end_date_user'):
                        final_end = min(pd.to_datetime(cfg['end_date_user']), min(data[t].last_valid_index() for t in tickers_for_config))
                    else:
                        final_end = min(data[t].last_valid_index() for t in tickers_for_config)
                    if final_start > final_end:
                        print(f"  Start date {final_start.date()} is after end date {final_end.date()}. Skipping {name}.")
                        continue
                    
                    simulation_index = pd.date_range(start=final_start, end=final_end, freq='D')
                    print(f"  Simulation period for {name}: {final_start.date()} to {final_end.date()}\n")
                    data_reindexed_for_config = {}
                    invalid_tickers = []
                    for t in data_tickers:
                        if t in data:  # Only process tickers that have data
                            df = data[t].reindex(simulation_index)
                            df["Close"] = df["Close"].ffill()
                            df["Dividends"] = df["Dividends"].fillna(0)
                            df["Price_change"] = df["Close"].pct_change(fill_method=None).fillna(0)
                            data_reindexed_for_config[t] = df
                        else:
                            invalid_tickers.append(t)
                            print(f"Warning: Invalid ticker '{t}' - no data available, skipping reindexing")
                    
                    # Display invalid ticker warnings in Streamlit UI
                    if invalid_tickers:
                        st.warning(f"The following tickers are invalid and will be skipped: {', '.join(invalid_tickers)}")
                    total_series, total_series_no_additions, historical_allocations, historical_metrics = single_backtest(cfg, simulation_index, data_reindexed_for_config)
                    # Store both series under the unique key for later use
                    all_results[unique_name] = {
                        'no_additions': total_series_no_additions,
                        'with_additions': total_series
                    }
                    all_allocations[unique_name] = historical_allocations
                    all_metrics[unique_name] = historical_metrics
                    # Remember mapping from portfolio index (0-based) to unique key
                    portfolio_key_map[i-1] = unique_name
                # --- PATCHED CASH FLOW LOGIC ---
                # Track cash flows as pandas Series indexed by date
                cash_flows = pd.Series(0.0, index=total_series.index)
                # Initial investment: negative cash flow on first date
                if len(total_series.index) > 0:
                    cash_flows.iloc[0] = -cfg.get('initial_value', 0)
                # No periodic additions for allocation tracker
                # Final value: positive cash flow on last date for MWRR
                if len(total_series.index) > 0:
                    cash_flows.iloc[-1] += total_series.iloc[-1]
                # Get benchmark returns for stats calculation
                benchmark_returns = None
                if cfg['benchmark_ticker'] and cfg['benchmark_ticker'] in data_reindexed_for_config:
                    benchmark_returns = data_reindexed_for_config[cfg['benchmark_ticker']]['Price_change']
                # Ensure benchmark_returns is a pandas Series aligned to total_series
                if benchmark_returns is not None:
                    benchmark_returns = pd.Series(benchmark_returns, index=total_series.index).dropna()
                # Ensure cash_flows is a pandas Series indexed by date, with initial investment and additions
                cash_flows = pd.Series(cash_flows, index=total_series.index)
                # Align for stats calculation
                # Track cash flows for MWRR exactly as in app.py
                # Initial investment: negative cash flow on first date
                mwrr_cash_flows = pd.Series(0.0, index=total_series.index)
                if len(total_series.index) > 0:
                    mwrr_cash_flows.iloc[0] = -cfg.get('initial_value', 0)
                # Periodic additions: negative cash flow on their respective dates
                dates_added = get_dates_by_freq(cfg.get('added_frequency'), total_series.index[0], total_series.index[-1], total_series.index)
                for d in dates_added:
                    if d in mwrr_cash_flows.index and d != mwrr_cash_flows.index[0]:
                        mwrr_cash_flows.loc[d] -= cfg.get('added_amount', 0)
                # Final value: positive cash flow on last date for MWRR
                if len(total_series.index) > 0:
                    mwrr_cash_flows.iloc[-1] += total_series.iloc[-1]

                # Use the no-additions series returned by single_backtest (do NOT reconstruct it here)
                # total_series_no_additions is returned by single_backtest and already represents the portfolio value without added cash.

                # Calculate statistics
                # Use total_series_no_additions for all stats except MWRR
                stats_values = total_series_no_additions.values
                stats_dates = total_series_no_additions.index
                stats_returns = pd.Series(stats_values, index=stats_dates).pct_change().fillna(0)
                cagr = calculate_cagr(stats_values, stats_dates)
                max_dd, drawdowns = calculate_max_drawdown(stats_values)
                vol = calculate_volatility(stats_returns)
                sharpe = np.nan if stats_returns.std() == 0 else stats_returns.mean() * 365 / (stats_returns.std() * np.sqrt(365))
                sortino = calculate_sortino(stats_returns)
                ulcer = calculate_ulcer_index(stats_values)
                upi = calculate_upi(cagr, ulcer)
                # --- Beta calculation (copied from app.py) ---
                beta = np.nan
                if benchmark_returns is not None:
                    portfolio_returns = stats_returns.copy()
                    benchmark_returns_series = pd.Series(benchmark_returns, index=stats_dates).dropna()
                    common_idx = portfolio_returns.index.intersection(benchmark_returns_series.index)
                    if len(common_idx) >= 2:
                        pr = portfolio_returns.reindex(common_idx).dropna()
                        br = benchmark_returns_series.reindex(common_idx).dropna()
                        common_idx = pr.index.intersection(br.index)
                        if len(common_idx) >= 2 and br.loc[common_idx].var() != 0:
                            cov = pr.loc[common_idx].cov(br.loc[common_idx])
                            var = br.loc[common_idx].var()
                            beta = cov / var
                # MWRR uses the full backtest with additions
                mwrr = calculate_mwrr(total_series, mwrr_cash_flows, total_series.index)
                def scale_pct(val):
                    if val is None or np.isnan(val):
                        return np.nan
                    # Only scale if value is between -1 and 1 (decimal)
                    if -1.5 < val < 1.5:
                        return val * 100
                    return val

                def clamp_stat(val, stat_type):
                    if val is None or np.isnan(val):
                        return "N/A"
                    v = scale_pct(val)
                    # Clamp ranges for each stat type
                    if stat_type in ["CAGR", "Volatility", "MWRR"]:
                        if v > 100:
                            return "N/A"
                    if stat_type == "MaxDrawdown":
                        if v < -100 or v > 0:
                            return "N/A"
                    return f"{v:.2f}%" if stat_type in ["CAGR", "MaxDrawdown", "Volatility", "MWRR"] else f"{v:.3f}" if isinstance(v, float) else v

                stats = {
                    "CAGR": clamp_stat(cagr, "CAGR"),
                    "MaxDrawdown": clamp_stat(max_dd, "MaxDrawdown"),
                    "Volatility": clamp_stat(vol, "Volatility"),
                    "Sharpe": clamp_stat(sharpe / 100 if isinstance(sharpe, (int, float)) and pd.notna(sharpe) else sharpe, "Sharpe"),
                    "Sortino": clamp_stat(sortino / 100 if isinstance(sortino, (int, float)) and pd.notna(sortino) else sortino, "Sortino"),
                    "UlcerIndex": clamp_stat(ulcer, "UlcerIndex"),
                    "UPI": clamp_stat(upi / 100 if isinstance(upi, (int, float)) and pd.notna(upi) else upi, "UPI"),
                    "Beta": clamp_stat(beta / 100 if isinstance(beta, (int, float)) and pd.notna(beta) else beta, "Beta"),
                    "MWRR": clamp_stat(mwrr, "MWRR"),
                }
                all_stats[name] = stats
                all_drawdowns[name] = pd.Series(drawdowns, index=stats_dates)
            progress_bar.progress(100, text="Backtests complete!")
            progress_bar.empty()
            print("\n" + "="*80)
            print(" " * 25 + "FINAL PERFORMANCE STATISTICS")
            print("="*80 + "\n")
            stats_df = pd.DataFrame(all_stats).T
            def fmt_pct(x):
                if isinstance(x, (int, float)) and pd.notna(x):
                    return f"{x*100:.2f}%"
                if isinstance(x, str):
                    return x
                return "N/A"
            def fmt_num(x, prec=3):
                if isinstance(x, (int, float)) and pd.notna(x):
                    return f"{x:.3f}"
                if isinstance(x, str):
                    return x
                return "N/A"
            if not stats_df.empty:
                stats_df_display = stats_df.copy()
                stats_df_display.rename(columns={'MaxDrawdown': 'Max Drawdown', 'UlcerIndex': 'Ulcer Index'}, inplace=True)
                stats_df_display['CAGR'] = stats_df_display['CAGR'].apply(lambda x: fmt_pct(x))
                stats_df_display['Max Drawdown'] = stats_df_display['Max Drawdown'].apply(lambda x: fmt_pct(x))
                stats_df_display['Volatility'] = stats_df_display['Volatility'].apply(lambda x: fmt_pct(x))
                # Ensure MWRR is the last column, Beta immediately before it
                if 'Beta' in stats_df_display.columns and 'MWRR' in stats_df_display.columns:
                    cols = list(stats_df_display.columns)
                    # Remove Beta and MWRR
                    beta_col = cols.pop(cols.index('Beta'))
                    mwrr_col = cols.pop(cols.index('MWRR'))
                    # Insert Beta before MWRR at the end
                    cols.append(beta_col)
                    cols.append(mwrr_col)
                    stats_df_display = stats_df_display[cols]
                stats_df_display['MWRR'] = stats_df_display['MWRR'].apply(lambda x: fmt_pct(x))
                stats_df_display['Sharpe'] = stats_df_display['Sharpe'].apply(lambda x: fmt_num(x))
                stats_df_display['Sortino'] = stats_df_display['Sortino'].apply(lambda x: fmt_num(x))
                stats_df_display['Ulcer Index'] = stats_df_display['Ulcer Index'].apply(lambda x: fmt_num(x))
                stats_df_display['UPI'] = stats_df_display['UPI'].apply(lambda x: fmt_num(x))
                if 'Beta' in stats_df_display.columns:
                    stats_df_display['Beta'] = stats_df_display['Beta'].apply(lambda x: fmt_num(x))
                print(stats_df_display.to_string())
            else:
                print("No stats to display.")
            # Yearly performance section (interactive table below)
            all_years = {}
            for name, ser in all_results.items():
                # Use the with-additions series for yearly performance (user requested)
                yearly = ser['with_additions'].resample('YE').last()
                all_years[name] = yearly
            years = sorted(list(set(y.year for ser in all_years.values() for y in ser.index)))
            names = list(all_years.keys())
            
            # Print console log yearly table correctly
            col_width = 22
            header_format = "{:<6} |" + "".join([" {:^" + str(col_width*2+1) + "} |" for _ in names])
            row_format = "{:<6} |" + "".join([" {:>" + str(col_width) + "} {:>" + str(col_width) + "} |" for _ in names])
            
            print(header_format.format("Year", *names))
            print("-" * (6 + 3 + (col_width*2+1 + 3)*len(names)))
            print(row_format.format(" ", *[item for pair in [('% Change', 'Final Value')] * len(names) for item in pair]))
            print("=" * (6 + 3 + (col_width*2+1 + 3)*len(names)))
            
            for y in years:
                row_items = [f"{y}"]
                for nm in names:
                    ser = all_years[nm]
                    ser_year = ser[ser.index.year == y]
                    
                    # Corrected logic for yearly performance calculation
                    start_val_for_year = None
                    if y == min(years):
                        config_for_name = next((c for c in portfolio_list if c['name'] == nm), None)
                        if config_for_name:
                            initial_val_of_config = config_for_name['initial_value']
                            if initial_val_of_config > 0:
                                start_val_for_year = initial_val_of_config
                    else:
                        prev_year = y - 1
                        prev_ser_year = all_years[nm][all_years[nm].index.year == prev_year]
                        if not prev_ser_year.empty:
                            start_val_for_year = prev_ser_year.iloc[-1]
                        
                    if not ser_year.empty and start_val_for_year is not None:
                        end_val = ser_year.iloc[-1]
                        if start_val_for_year > 0:
                            pct = f"{(end_val - start_val_for_year) / start_val_for_year * 100:.2f}%"
                            final_val = f"${end_val:,.2f}"
                        else:
                            pct = "N/A"
                            final_val = "N/A"
                    else:
                        pct = "N/A"
                        final_val = "N/A"
                        
                    row_items.extend([pct, final_val])
                print(row_format.format(*row_items))
            print("\n" + "="*80)
    
            # console output captured previously is no longer shown on the page
            st.session_state.alloc_all_results = all_results
            st.session_state.alloc_all_drawdowns = all_drawdowns
            if 'stats_df_display' in locals():
                st.session_state.alloc_stats_df_display = stats_df_display
            st.session_state.alloc_all_years = all_years
            st.session_state.alloc_all_allocations = all_allocations
            # Save a snapshot used by the allocations UI so charts/tables remain static until rerun
            try:
                # compute today_weights_map (target weights as-if rebalanced at final snapshot date)
                today_weights_map = {}
                for pname, allocs in all_allocations.items():
                    try:
                        alloc_dates = sorted(list(allocs.keys()))
                        final_d = alloc_dates[-1]
                        metrics_local = all_metrics.get(pname, {})
                        
                        # Check if momentum is used for this portfolio
                        portfolio_cfg = next((cfg for cfg in portfolio_list if cfg.get('name') == pname), None)
                        use_momentum = portfolio_cfg.get('use_momentum', True) if portfolio_cfg else True
                        
                        if final_d in metrics_local:
                            if use_momentum:
                                # extract Calculated_Weight if present (momentum-based)
                                weights = {t: v.get('Calculated_Weight', 0) for t, v in metrics_local[final_d].items()}
                                # normalize (ensure sums to 1 excluding CASH)
                                sumw = sum(w for k, w in weights.items() if k != 'CASH')
                                if sumw > 0:
                                    norm = {k: (w / sumw) if k != 'CASH' else weights.get('CASH', 0) for k, w in weights.items()}
                                else:
                                    norm = weights
                                today_weights_map[pname] = norm
                            else:
                                # Use user-defined allocations from portfolio config - COPIED FROM PAGE 1
                                weights = {}
                                for stock in portfolio_cfg.get('stocks', []):
                                    ticker = stock.get('ticker', '').strip()
                                    if ticker:
                                        weights[ticker] = stock.get('allocation', 0)
                                
                                # Apply MA filter even when momentum is disabled
                                if portfolio_cfg.get('use_sma_filter', False):
                                    ma_window = portfolio_cfg.get('sma_window', 200)
                                    ma_type = portfolio_cfg.get('ma_type', 'SMA')
                                    # Get list of current tickers (excluding CASH)
                                    current_tickers = [t for t in weights.keys() if t != 'CASH']
                                    
                                    # Apply MA filter using data
                                    try:
                                        filtered_tickers, excluded_assets = filter_assets_by_ma(current_tickers, data, final_d, ma_window, ma_type, portfolio_cfg, portfolio_cfg.get('stocks', []))
                                        
                                        # Redistribute allocations of excluded tickers
                                        if excluded_assets:
                                            excluded_ticker_list = list(excluded_assets.keys())
                                            excluded_allocation = sum(weights.get(t, 0) for t in excluded_ticker_list)
                                            
                                            # Remove excluded tickers
                                            for excluded_ticker in excluded_ticker_list:
                                                if excluded_ticker in weights:
                                                    del weights[excluded_ticker]
                                            
                                            # Redistribute to remaining tickers
                                            remaining_tickers = [t for t in weights.keys() if t != 'CASH']
                                            if remaining_tickers:
                                                remaining_allocation = sum(weights.get(t, 0) for t in remaining_tickers)
                                                if remaining_allocation > 0:
                                                    for ticker in remaining_tickers:
                                                        proportion = weights[ticker] / remaining_allocation
                                                        weights[ticker] += excluded_allocation * proportion
                                                else:
                                                    # Equal distribution
                                                    equal_allocation = excluded_allocation / len(remaining_tickers)
                                                    for ticker in remaining_tickers:
                                                        weights[ticker] = equal_allocation
                                            else:
                                                # No remaining tickers, all goes to CASH
                                                weights = {'CASH': 1.0}
                                    except Exception as e:
                                        # If MA filter fails, keep original allocations
                                        pass
                                
                                # Add CASH if needed (after MA filter)
                                if 'CASH' not in weights:
                                    total_alloc = sum(weights.values())
                                    if total_alloc < 1.0:
                                        weights['CASH'] = 1.0 - total_alloc
                                    else:
                                        weights['CASH'] = 0
                                
                                # For targeted rebalancing: check if rebalancing would be triggered today
                                # If no rebalancing needed, show current allocation instead of target allocation
                                if portfolio_cfg.get('use_targeted_rebalancing', False):
                                    # Get current allocation from historical_allocations (drifted)
                                    current_alloc = allocs.get(final_d, {})
                                    
                                    # Check if any threshold is exceeded
                                    targeted_settings = portfolio_cfg.get('targeted_rebalancing_settings', {})
                                    threshold_exceeded = False
                                    
                                    for ticker in current_alloc.keys():
                                        if ticker != 'CASH' and ticker in targeted_settings and targeted_settings[ticker].get('enabled', False):
                                            current_allocation_pct = current_alloc.get(ticker, 0) * 100
                                            max_threshold = targeted_settings[ticker].get('max_allocation', 100.0)
                                            min_threshold = targeted_settings[ticker].get('min_allocation', 0.0)
                                            
                                            # Check if allocation exceeds max or falls below min threshold
                                            if current_allocation_pct > max_threshold or current_allocation_pct < min_threshold:
                                                threshold_exceeded = True
                                                break
                                    
                                    # If no threshold exceeded, use current (drifted) allocation instead of target
                                    if not threshold_exceeded and current_alloc:
                                        weights = current_alloc.copy()
                                
                                today_weights_map[pname] = weights
                        else:
                            # fallback: use allocation snapshot at final date but convert market-value alloc to target weights (exclude CASH then renormalize)
                            final_alloc = allocs.get(final_d, {})
                            noncash = {k: v for k, v in final_alloc.items() if k != 'CASH'}
                            s = sum(noncash.values())
                            if s > 0:
                                norm = {k: (v / s) for k, v in noncash.items()}
                                norm['CASH'] = final_alloc.get('CASH', 0)
                            else:
                                norm = final_alloc
                            today_weights_map[pname] = norm
                    except Exception:
                        today_weights_map[pname] = {}

                st.session_state.alloc_snapshot_data = {
                    'raw_data': data,
                    'portfolio_configs': portfolio_list,
                    'all_allocations': all_allocations,
                    'all_metrics': all_metrics,
                    'today_weights_map': today_weights_map
                }
            except Exception:
                pass
            st.session_state.alloc_all_metrics = all_metrics
            # Save portfolio index -> unique key mapping so UI selectors can reference results reliably
            st.session_state.alloc_portfolio_key_map = portfolio_key_map
            st.session_state.alloc_backtest_run = True
            
            # ===================================================================
            # Calculate and store PE immediately after backtest
            # ===================================================================
            try:
                # Get the today_weights for the active portfolio
                active_name = active_portfolio.get('name')
                if active_name in today_weights_map:
                    today_weights = today_weights_map[active_name]
                    
                    # Get portfolio value
                    active_idx = st.session_state.alloc_active_portfolio_index
                    portfolio_value = float(st.session_state.get('alloc_active_initial', active_portfolio.get('initial_value', 0) or 0))
                    
                    # Get ticker info in batch
                    tickers = [tk for tk in today_weights.keys() if tk != 'CASH']
                    if tickers:
                        all_infos = get_multiple_tickers_info_batch(tickers)
                        
                        # Build rows with PE data
                        rows = []
                        for ticker in tickers:
                            info = all_infos.get(ticker, {})
                            alloc_pct = float(today_weights.get(ticker, 0))
                            allocation_value = portfolio_value * alloc_pct
                            total_val = allocation_value
                            pct_of_portfolio = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                            
                            rows.append({
                                'Ticker': ticker,
                                'P/E Ratio': info.get('trailingPE'),
                                'Forward P/E': info.get('forwardPE'),
                                'Beta': info.get('beta'),
                                '% of Portfolio': pct_of_portfolio
                            })
                        
                        if rows:
                            df_temp = pd.DataFrame(rows)
                            
                            # Calculate weighted averages
                            def weighted_average(df, column, weight_column='% of Portfolio'):
                                if column not in df.columns or weight_column not in df.columns:
                                    return None
                                # Convert column to numeric, coercing errors to NaN
                                df_numeric = df.copy()
                                df_numeric[column] = pd.to_numeric(df_numeric[column], errors='coerce')
                                df_numeric[weight_column] = pd.to_numeric(df_numeric[weight_column], errors='coerce')
                                
                                # First filter for non-null values
                                valid_mask = df_numeric[column].notna() & df_numeric[weight_column].notna()
                                # Then apply numeric filters only on valid (non-null) rows
                                if 'P/E' in column or 'PE' in column:
                                    valid_mask = valid_mask & (df_numeric[column] > 0) & (df_numeric[column] <= 1000)
                                elif column == 'Beta':
                                    valid_mask = valid_mask & (df_numeric[column] >= -5) & (df_numeric[column] <= 5)
                                if valid_mask.sum() == 0:
                                    return None
                                valid_df = df_numeric[valid_mask]
                                result = (valid_df[column] * valid_df[weight_column] / 100).sum() / (valid_df[weight_column].sum() / 100)
                                return result
                            
                            # Calculate and store PE, Forward PE, and Beta
                            st.session_state.portfolio_pe = weighted_average(df_temp, 'P/E Ratio')
                            st.session_state.portfolio_forward_pe = weighted_average(df_temp, 'Forward P/E')
                            st.session_state.portfolio_beta = weighted_average(df_temp, 'Beta')
                            
                            # Store snapshot of tickers for photo fixe
                            st.session_state.backtest_stocks_snapshot = tickers
                    
            except Exception as e:
                pass

# Sidebar JSON export/import for ALL portfolios
def paste_all_json_callback():
    txt = st.session_state.get('alloc_paste_all_json_text', '')
    if not txt:
        st.warning('No JSON provided')
        return
    try:
        # Use the SAME parsing logic as successful PDF extraction
        raw_text = txt
        
        # STEP 1: Try the exact same approach as PDF extraction (simple strip + parse)
        try:
            cleaned_text = raw_text.strip()
            obj = json.loads(cleaned_text)
            st.success("✅ Multi-portfolio JSON parsed successfully using PDF-style parsing!")
        except json.JSONDecodeError:
            # STEP 2: If that fails, apply our advanced cleaning (fallback)
            st.info("🔧 Simple parsing failed, applying advanced PDF extraction fixes...")
            
            json_text = raw_text
            import re
            
            # Fix broken portfolio name lines
            broken_pattern = r'"name":\s*"([^"]*?)"\s*"stocks":'
            json_text = re.sub(broken_pattern, r'"name": "\1", "stocks":', json_text)
            
            # Fix truncated names
            truncated_pattern = r'"name":\s*"([^"]*?)\s+"stocks":'
            json_text = re.sub(truncated_pattern, r'"name": "\1", "stocks":', json_text)
            
            # Fix missing opening brace for portfolio objects
            missing_brace_pattern = r'(},)\s*("name":)'
            json_text = re.sub(missing_brace_pattern, r'\1 {\n \2', json_text)
            
            obj = json.loads(json_text)
            st.success("✅ Multi-portfolio JSON parsed successfully using advanced cleaning!")
        
        # Add missing fields for compatibility if they don't exist
        if isinstance(obj, list):
            for portfolio in obj:
                # Add missing fields with default values
                if 'collect_dividends_as_cash' not in portfolio:
                    portfolio['collect_dividends_as_cash'] = False
                if 'exclude_from_cashflow_sync' not in portfolio:
                    portfolio['exclude_from_cashflow_sync'] = False
                if 'exclude_from_rebalancing_sync' not in portfolio:
                    portfolio['exclude_from_rebalancing_sync'] = False
        if isinstance(obj, list):
            # Process each portfolio configuration for Allocations page
            processed_configs = []
            for cfg in obj:
                if not isinstance(cfg, dict) or 'name' not in cfg:
                    st.error('Invalid portfolio configuration structure.')
                    return
                
                # Handle momentum strategy value mapping from other pages
                momentum_strategy = cfg.get('momentum_strategy', 'Classic')
                if momentum_strategy == 'Classic momentum':
                    momentum_strategy = 'Classic'
                elif momentum_strategy == 'Relative momentum':
                    momentum_strategy = 'Relative Momentum'
                elif momentum_strategy not in ['Classic', 'Relative Momentum']:
                    momentum_strategy = 'Classic'  # Default fallback
                
                # Handle negative momentum strategy value mapping from other pages
                negative_momentum_strategy = cfg.get('negative_momentum_strategy', 'Cash')
                if negative_momentum_strategy == 'Go to cash':
                    negative_momentum_strategy = 'Cash'
                elif negative_momentum_strategy not in ['Cash', 'Equal weight', 'Relative momentum']:
                    negative_momentum_strategy = 'Cash'  # Default fallback
                
                # Handle stocks field - convert from legacy format if needed
                stocks = cfg.get('stocks', [])
                if not stocks and 'tickers' in cfg:
                    # Convert legacy format (tickers, allocs, divs) to stocks format
                    tickers = cfg.get('tickers', [])
                    allocs = cfg.get('allocs', [])
                    divs = cfg.get('divs', [])
                    stocks = []
                    
                    # Ensure we have valid arrays
                    if tickers and isinstance(tickers, list):
                        for i in range(len(tickers)):
                            if tickers[i] and tickers[i].strip():  # Check for non-empty ticker
                                # Convert allocation from percentage (0-100) to decimal (0.0-1.0) format
                                allocation = 0.0
                                if i < len(allocs) and allocs[i] is not None:
                                    alloc_value = float(allocs[i])
                                    if alloc_value > 1.0:
                                        # Already in percentage format, convert to decimal
                                        allocation = alloc_value / 100.0
                                    else:
                                        # Already in decimal format, use as is
                                        allocation = alloc_value
                                
                                # Resolve the alias to the actual Yahoo ticker
                                resolved_ticker = resolve_ticker_alias(tickers[i].strip())
                                stock = {
                                    'ticker': resolved_ticker,  # Use resolved ticker
                                    'allocation': allocation,
                                    'include_dividends': bool(divs[i]) if i < len(divs) and divs[i] is not None else True
                                }
                                stocks.append(stock)
            
            # Process each portfolio configuration for Allocations page (existing logic)
            processed_configs = []
            for cfg in obj:
                if not isinstance(cfg, dict) or 'name' not in cfg:
                    st.error('Invalid portfolio configuration structure.')
                    return
                
                # Handle momentum strategy value mapping from other pages
                momentum_strategy = cfg.get('momentum_strategy', 'Classic')
                if momentum_strategy == 'Classic momentum':
                    momentum_strategy = 'Classic'
                elif momentum_strategy == 'Relative momentum':
                    momentum_strategy = 'Relative Momentum'
                elif momentum_strategy not in ['Classic', 'Relative Momentum']:
                    momentum_strategy = 'Classic'  # Default fallback
                
                # Handle negative momentum strategy value mapping from other pages
                negative_momentum_strategy = cfg.get('negative_momentum_strategy', 'Cash')
                if negative_momentum_strategy == 'Go to cash':
                    negative_momentum_strategy = 'Cash'
                elif negative_momentum_strategy not in ['Cash', 'Equal weight', 'Relative momentum']:
                    negative_momentum_strategy = 'Cash'  # Default fallback
                

                
                # Sanitize momentum window weights to prevent StreamlitValueAboveMaxError
                momentum_windows = cfg.get('momentum_windows', [])
                for window in momentum_windows:
                    if 'weight' in window:
                        weight = window['weight']
                        # If weight is a percentage (e.g., 50 for 50%), convert to decimal
                        if isinstance(weight, (int, float)) and weight > 1.0:
                            # Cap at 100% and convert to decimal
                            weight = min(weight, 100.0) / 100.0
                        elif isinstance(weight, (int, float)) and weight <= 1.0:
                            # Already in decimal format, ensure it's valid
                            weight = max(0.0, min(weight, 1.0))
                        else:
                            # Invalid weight, set to default
                            weight = 0.1
                        window['weight'] = weight
                
                # Debug: Show what we received for this portfolio
                if 'momentum_windows' in cfg:
                    st.info(f"Momentum windows for {cfg.get('name', 'Unknown')}: {cfg['momentum_windows']}")
                if 'use_momentum' in cfg:
                    st.info(f"Use momentum for {cfg.get('name', 'Unknown')}: {cfg['use_momentum']}")
                
                # Map frequency values from app.py format to Allocations format
                def map_frequency(freq):
                    if freq is None:
                        return 'Never'
                    freq_map = {
                        'Never': 'Never',
                        'Weekly': 'Weekly',
                        'Biweekly': 'Biweekly',
                        'Monthly': 'Monthly',
                        'Quarterly': 'Quarterly',
                        'Semiannually': 'Semiannually',
                        'Annually': 'Annually',
                        # Legacy format mapping
                        'none': 'Never',
                        'week': 'Weekly',
                        '2weeks': 'Biweekly',
                        'month': 'Monthly',
                        '3months': 'Quarterly',
                        '6months': 'Semiannually',
                        'year': 'Annually'
                    }
                    return freq_map.get(freq, 'Monthly')
                
                # Allocations page specific: ensure all required fields are present
                # and ignore fields that are specific to other pages
                allocations_config = {
                    'name': cfg.get('name', 'Allocation Portfolio'),
                    'stocks': stocks,
                    'benchmark_ticker': cfg.get('benchmark_ticker', '^GSPC'),
                    'initial_value': cfg.get('initial_value', 10000),
                    'added_amount': cfg.get('added_amount', 0),  # Allocations page typically doesn't use additions
                    'added_frequency': map_frequency(cfg.get('added_frequency', 'Never')),  # Allocations page typically doesn't use additions
                                          'rebalancing_frequency': map_frequency(cfg.get('rebalancing_frequency', 'Monthly')),
                      'start_date_user': cfg.get('start_date_user'),
                      'end_date_user': cfg.get('end_date_user'),
                      'start_with': cfg.get('start_with', 'oldest'),
                      'use_momentum': cfg.get('use_momentum', True),
                    'momentum_strategy': momentum_strategy,
                    'negative_momentum_strategy': negative_momentum_strategy,
                    'momentum_windows': momentum_windows,
                    'calc_beta': cfg.get('calc_beta', True),
                    'calc_volatility': cfg.get('calc_volatility', True),
                    'beta_window_days': cfg.get('beta_window_days', 365),
                    'exclude_days_beta': cfg.get('exclude_days_beta', 30),
                    'vol_window_days': cfg.get('vol_window_days', 365),
                    'exclude_days_vol': cfg.get('exclude_days_vol', 30),
                    'use_targeted_rebalancing': cfg.get('use_targeted_rebalancing', False),
                    'targeted_rebalancing_settings': cfg.get('targeted_rebalancing_settings', {}),
                }
                processed_configs.append(allocations_config)
            
            st.session_state.alloc_portfolio_configs = processed_configs
            # Reset active selection and derived mappings so the UI reflects the new configs
            if processed_configs:
                st.session_state.alloc_active_portfolio_index = 0
                st.session_state.alloc_portfolio_selector = processed_configs[0].get('name', '')
                # Update portfolio name input field to match the first imported portfolio
                st.session_state.alloc_portfolio_name = processed_configs[0].get('name', 'Allocation Portfolio')
                # Mirror several active_* widget defaults so the UI selectboxes/inputs update
                st.session_state['alloc_active_name'] = processed_configs[0].get('name', '')
                st.session_state['alloc_active_initial'] = int(processed_configs[0].get('initial_value', 0) or 0)
                st.session_state['alloc_active_added_amount'] = int(processed_configs[0].get('added_amount', 0) or 0)
                st.session_state['alloc_active_rebal_freq'] = processed_configs[0].get('rebalancing_frequency', 'month')
                st.session_state['alloc_active_add_freq'] = processed_configs[0].get('added_frequency', 'none')
                st.session_state['alloc_active_benchmark'] = processed_configs[0].get('benchmark_ticker', '')
                st.session_state['alloc_active_use_momentum'] = bool(processed_configs[0].get('use_momentum', True))
                
            else:
                st.session_state.alloc_active_portfolio_index = None
                st.session_state.alloc_portfolio_selector = ''
            st.session_state.alloc_portfolio_key_map = {}
            st.session_state.alloc_backtest_run = False
            st.success('All portfolio configurations updated from JSON (Allocations page).')
            # Debug: Show final momentum windows for first portfolio
            if processed_configs:
                st.info(f"Final momentum windows for first portfolio: {processed_configs[0]['momentum_windows']}")
                st.info(f"Final use_momentum for first portfolio: {processed_configs[0]['use_momentum']}")
            # Force a rerun so widgets rebuild with the new configs
            try:
                st.experimental_rerun()
            except Exception:
                # In some environments experimental rerun may raise; setting a rerun flag is a fallback
                st.session_state.alloc_rerun_flag = True
        else:
            st.error('JSON must be a list of portfolio configurations.')
    except Exception as e:
        st.error(f'Failed to parse JSON: {e}')





# Simplified display for allocation tracker: only allocation pies and rebalancing metrics are shown
active_name = active_portfolio.get('name')
if st.session_state.get('alloc_backtest_run', False):
    st.subheader("Allocation & Rebalancing Metrics")
    

    allocs_for_portfolio = st.session_state.get('alloc_all_allocations', {}).get(active_name) if st.session_state.get('alloc_all_allocations') else None
    metrics_for_portfolio = st.session_state.get('alloc_all_metrics', {}).get(active_name) if st.session_state.get('alloc_all_metrics') else None

    if not allocs_for_portfolio and not metrics_for_portfolio:
        st.info("No allocation or rebalancing history available. If you have precomputed allocation snapshots, store them in session state keys `alloc_all_allocations` and `alloc_all_metrics` under this portfolio name.")
    else:
        # --- Calculate timer variables for rebalancing timer ---
        last_rebal_date = None
        rebalancing_frequency = 'none'
        
        if allocs_for_portfolio and active_portfolio:
            try:
                # Get the last rebalance date from allocation history
                alloc_dates = sorted(list(allocs_for_portfolio.keys()))
                if len(alloc_dates) > 1:
                    last_rebal_date = alloc_dates[-2]  # Second to last date (excluding today/yesterday)
                else:
                    last_rebal_date = alloc_dates[-1] if alloc_dates else None
                
                # Get rebalancing frequency from active portfolio
                rebalancing_frequency = active_portfolio.get('rebalancing_frequency', 'none')
                # Convert to lowercase and map to function expectations
                rebalancing_frequency = rebalancing_frequency.lower()
                # Map frequency names to what the function expects
                frequency_mapping = {
                    'monthly': 'month',
                    'weekly': 'week',
                    'bi-weekly': '2weeks',
                    'biweekly': '2weeks',
                    'quarterly': '3months',
                    'semi-annually': '6months',
                    'semiannually': '6months',
                    'annually': 'year',
                    'yearly': 'year',
                    'market_day': 'market_day',
                    'calendar_day': 'calendar_day',
                    'never': 'none',
                    'none': 'none'
                }
                rebalancing_frequency = frequency_mapping.get(rebalancing_frequency, rebalancing_frequency)
            except Exception as e:
                pass  # Silently ignore timer calculation errors
        
        # --- Rebalance as of Today (static snapshot from last Run Backtests) ---
        snapshot = st.session_state.get('alloc_snapshot_data', {})
        today_weights_map = snapshot.get('today_weights_map', {}) if snapshot else {}
        
        # Check if momentum is used for this portfolio
        use_momentum = active_portfolio.get('use_momentum', True) if active_portfolio else True
        
        if snapshot and active_name in today_weights_map:
            today_weights = today_weights_map.get(active_name, {})
            
            # If momentum is not used, use the initial target allocation from portfolio configuration
            if not use_momentum and (not today_weights or all(v == 0 for v in today_weights.values())):
                # Use the target allocation from portfolio configuration (same as initial allocation)
                if active_portfolio and active_portfolio.get('stocks'):
                    today_weights = {}
                    total_allocation = sum(stock.get('allocation', 0) for stock in active_portfolio['stocks'] if stock.get('ticker'))
                    
                    if total_allocation > 0:
                        # Build raw allocations from portfolio config
                        raw_allocations = {}
                        for stock in active_portfolio['stocks']:
                            ticker = stock.get('ticker', '').strip()
                            allocation = stock.get('allocation', 0)
                            if ticker and allocation > 0:
                                raw_allocations[ticker] = allocation / total_allocation
                        
                        # Apply threshold filters to the raw allocations
                        use_max_allocation = active_portfolio.get('use_max_allocation', False)
                        max_allocation_percent = active_portfolio.get('max_allocation_percent', 10.0)
                        use_threshold = active_portfolio.get('use_minimal_threshold', False)
                        threshold_percent = active_portfolio.get('minimal_threshold_percent', 2.0)
                        
                        print(f"[THRESHOLD DEBUG] Rebalance as of Today - use_threshold: {use_threshold}, threshold_percent: {threshold_percent}, use_max_allocation: {use_max_allocation}, max_allocation_percent: {max_allocation_percent}")
                        print(f"[THRESHOLD DEBUG] Raw allocations: {raw_allocations}")
                        
                        # Build dictionary of individual ticker caps from stock configs
                        individual_caps = {}
                        for stock in active_portfolio.get('stocks', []):
                            ticker = stock.get('ticker', '')
                            individual_cap = stock.get('max_allocation_percent', None)
                            if individual_cap is not None and individual_cap > 0:
                                individual_caps[ticker] = individual_cap / 100.0
                        
                        # Apply allocation filters in correct order: Max Allocation -> Min Threshold -> Max Allocation (two-pass system)
                        filtered_allocations = raw_allocations.copy()
                        
                        if (use_max_allocation or individual_caps):
                            max_allocation_decimal = max_allocation_percent / 100.0
                            
                            # FIRST PASS: Apply maximum allocation filter
                            capped_allocations = {}
                            excess_allocation = 0.0
                            
                            for ticker, allocation in filtered_allocations.items():
                                # Use individual cap if available, otherwise use global cap
                                ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                                
                                if allocation > ticker_cap:
                                    # Cap the allocation and collect excess
                                    capped_allocations[ticker] = ticker_cap
                                    excess_allocation += (allocation - ticker_cap)
                                else:
                                    capped_allocations[ticker] = allocation
                            
                            # Redistribute excess allocation proportionally to stocks below the cap
                            if excess_allocation > 0:
                                # Find stocks below the cap (using individual caps)
                                below_cap_stocks = {}
                                for ticker, allocation in capped_allocations.items():
                                    ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                                    if allocation < ticker_cap:
                                        below_cap_stocks[ticker] = allocation
                                
                                if below_cap_stocks:
                                    total_below_cap = sum(below_cap_stocks.values())
                                    if total_below_cap > 0:
                                        # Redistribute excess proportionally
                                        for ticker in below_cap_stocks:
                                            proportion = below_cap_stocks[ticker] / total_below_cap
                                            new_allocation = capped_allocations[ticker] + (excess_allocation * proportion)
                                            ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                                            capped_allocations[ticker] = min(new_allocation, ticker_cap)
                            
                            filtered_allocations = capped_allocations
                        
                        # Apply minimal threshold filter
                        if use_threshold:
                            threshold_decimal = threshold_percent / 100.0
                            
                            # First: Filter out stocks below threshold
                            threshold_filtered_allocations = {}
                            for ticker, allocation in filtered_allocations.items():
                                if allocation >= threshold_decimal:
                                    # Keep stocks above or equal to threshold
                                    threshold_filtered_allocations[ticker] = allocation
                            
                            # Then: Normalize remaining stocks to sum to 1
                            if threshold_filtered_allocations:
                                total_allocation = sum(threshold_filtered_allocations.values())
                                if total_allocation > 0:
                                    filtered_allocations = {ticker: allocation / total_allocation for ticker, allocation in threshold_filtered_allocations.items()}
                                else:
                                    filtered_allocations = {}
                            else:
                                # If no stocks meet threshold, keep original allocations
                                pass  # filtered_allocations remain unchanged
                        
                        # SECOND PASS: Apply maximum allocation filter again (in case normalization created new excess)
                        if (use_max_allocation or individual_caps):
                            max_allocation_decimal = max_allocation_percent / 100.0
                            
                            # Check if any stocks exceed the cap after threshold filtering and normalization
                            capped_allocations = {}
                            excess_allocation = 0.0
                            
                            for ticker, allocation in filtered_allocations.items():
                                # Use individual cap if available, otherwise use global cap
                                ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                                
                                if allocation > ticker_cap:
                                    # Cap the allocation and collect excess
                                    capped_allocations[ticker] = ticker_cap
                                    excess_allocation += (allocation - ticker_cap)
                                else:
                                    capped_allocations[ticker] = allocation
                            
                            # Redistribute excess allocation proportionally to stocks below the cap
                            if excess_allocation > 0:
                                # Find stocks below the cap (using individual caps)
                                below_cap_stocks = {}
                                for ticker, allocation in capped_allocations.items():
                                    ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                                    if allocation < ticker_cap:
                                        below_cap_stocks[ticker] = allocation
                                
                                if below_cap_stocks:
                                    total_below_cap = sum(below_cap_stocks.values())
                                    if total_below_cap > 0:
                                        # Redistribute excess proportionally
                                        for ticker in below_cap_stocks:
                                            proportion = below_cap_stocks[ticker] / total_below_cap
                                            new_allocation = capped_allocations[ticker] + (excess_allocation * proportion)
                                            ticker_cap = individual_caps.get(ticker, max_allocation_decimal if use_max_allocation else float('inf'))
                                            capped_allocations[ticker] = min(new_allocation, ticker_cap)
                            
                            filtered_allocations = capped_allocations
                        
                        # Use the filtered allocations as today_weights
                        today_weights = filtered_allocations
                        print(f"[THRESHOLD DEBUG] Filtered allocations for Rebalance as of Today: {today_weights}")
                    
                    # If no valid stocks or allocations, leave today_weights empty (will show info message)
            
            labels_today = [k for k, v in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
            vals_today = [float(today_weights[k]) * 100 for k in labels_today]
            
            # Handle case where momentum goes to cash (all assets have negative momentum)
            # If no labels or all values are very small, show 100% CASH
            if not labels_today or sum(vals_today) < 0.1:
                labels_today = ['CASH']
                vals_today = [100.0]
            
            # --- REBALANCING TIMER SECTION (moved above pie chart) ---
            if last_rebal_date and rebalancing_frequency != 'none':
                # Ensure last_rebal_date is a naive datetime object
                import pandas as pd
                if isinstance(last_rebal_date, str):
                    last_rebal_date = pd.to_datetime(last_rebal_date)
                if hasattr(last_rebal_date, 'tzinfo') and last_rebal_date.tzinfo is not None:
                    last_rebal_date = last_rebal_date.replace(tzinfo=None)
                
                next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                    rebalancing_frequency, last_rebal_date
                )
                
                if next_date and time_until:
                    st.markdown("---")
                    st.markdown("**⏰ Next Rebalance Timer**")
                    
                    # Create columns for timer display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Time Until Next Rebalance",
                            value=format_time_until(time_until),
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            label="Target Rebalance Date",
                            value=next_date.strftime("%B %d, %Y"),
                            delta=None
                        )
                    
                    with col3:
                        st.metric(
                            label="Rebalancing Frequency",
                            value=rebalancing_frequency.replace('_', ' ').title(),
                            delta=None
                        )
                    
                    # Add a progress bar showing progress to next rebalance
                    if rebalancing_frequency in ['week', '2weeks', 'month', '3months', '6months', 'year']:
                        # Calculate progress percentage
                        if hasattr(last_rebal_date, 'to_pydatetime'):
                            last_rebal_datetime = last_rebal_date.to_pydatetime()
                        else:
                            last_rebal_datetime = last_rebal_date
                        
                        total_period = (next_rebalance_datetime - last_rebal_datetime).total_seconds()
                        elapsed_period = (datetime.now() - last_rebal_datetime).total_seconds()
                        progress = min(max(elapsed_period / total_period, 0), 1)
                        
                        st.progress(progress, text=f"Progress to next rebalance: {progress:.1%}")
            
            if labels_today and vals_today:
                st.markdown(f"## Rebalance as of Today ({pd.Timestamp.now().strftime('%Y-%m-%d')})")
                fig_today = go.Figure(data=[go.Pie(
                    labels=labels_today,
                    values=vals_today,
                    hole=0.35
                )])
                fig_today.update_traces(textinfo='percent+label')
                fig_today.update_layout(template='plotly_dark', margin=dict(t=10), height=600)
                st.plotly_chart(fig_today, key=f"alloc_today_chart_{active_name}")
            
            # static shares table

            # Define build_table_from_alloc before usage
            def build_table_from_alloc(alloc_dict, price_date, label):
                rows = []
                # Use portfolio_value from session state or active_portfolio (current portfolio value)
                try:
                    portfolio_value = float(st.session_state.get('alloc_active_initial', active_portfolio.get('initial_value', 0) or 0))
                except Exception:
                    portfolio_value = active_portfolio.get('initial_value', 0) or 0
                # Use raw_data from snapshot or session state
                snapshot = st.session_state.get('alloc_snapshot_data', {})
                raw_data = snapshot.get('raw_data') if snapshot and snapshot.get('raw_data') is not None else st.session_state.get('alloc_raw_data', {})
                def _price_on_or_before(df, target_date):
                    try:
                        idx = df.index[df.index <= pd.to_datetime(target_date)]
                        if len(idx) == 0:
                            return None
                        return float(df.loc[idx[-1], 'Close'])
                    except Exception:
                        return None
                for tk in sorted(alloc_dict.keys()):
                    alloc_pct = float(alloc_dict.get(tk, 0))
                    if tk == 'CASH':
                        price = None
                        shares = 0.0
                        total_val = portfolio_value * alloc_pct
                    else:
                        df = raw_data.get(tk)
                        price = None
                        # Ensure df is a valid DataFrame with Close prices before accessing
                        if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                            if price_date is None:
                                try:
                                    price = float(df['Close'].iloc[-1])
                                except Exception:
                                    price = None
                            else:
                                price = _price_on_or_before(df, price_date)
                        try:
                            if price and price > 0:
                                allocation_value = portfolio_value * alloc_pct
                                # allow fractional shares shown to 1 decimal place
                                shares = round(allocation_value / price, 1)
                                total_val = shares * price
                            else:
                                shares = 0.0
                                total_val = portfolio_value * alloc_pct
                        except Exception:
                            shares = 0.0
                            total_val = portfolio_value * alloc_pct
                    
                    pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                    rows.append({
                        'Ticker': tk,
                        'Allocation %': alloc_pct * 100,
                        'Price ($)': price if price is not None else float('nan'),
                        'Shares': shares,
                        'Total Value ($)': total_val,
                        '% of Portfolio': pct_of_port,
                    })
                df_table = pd.DataFrame(rows).set_index('Ticker')
                # Decide whether to show CASH row: hide if Total Value is zero or Shares zero/NaN
                df_display = df_table.copy()
                show_cash = False
                if 'CASH' in df_display.index:
                    cash_val = None
                    if 'Total Value ($)' in df_display.columns:
                        cash_val = df_display.at['CASH', 'Total Value ($)']
                    elif 'Shares' in df_display.columns:
                        cash_val = df_display.at['CASH', 'Shares']
                    try:
                        show_cash = bool(cash_val and not pd.isna(cash_val) and cash_val != 0)
                    except Exception:
                        show_cash = False
                    if not show_cash:
                        df_display = df_display.drop('CASH')
                
                # Add total row
                total_alloc_pct = df_display['Allocation %'].sum()
                total_value = df_display['Total Value ($)'].sum()
                total_port_pct = df_display['% of Portfolio'].sum()
                
                total_row = pd.DataFrame({
                    'Allocation %': [total_alloc_pct],
                    'Price ($)': [float('nan')],
                    'Shares': [float('nan')],
                    'Total Value ($)': [total_value],
                    '% of Portfolio': [total_port_pct]
                }, index=['TOTAL'])
                
                df_display = pd.concat([df_display, total_row])

                fmt = {
                    'Allocation %': '{:,.1f}%',
                    'Price ($)': '${:,.2f}',
                    'Shares': '{:,.1f}',
                    'Total Value ($)': '${:,.2f}',
                    '% of Portfolio': '{:,.2f}%'
                }
                try:
                    st.markdown(f"**{label}**")
                    sty = df_display.style.format(fmt)
                    
                    # Highlight CASH row if present
                    if 'CASH' in df_table.index and show_cash:
                        def _highlight_cash_row(s):
                            if s.name == 'CASH':
                                return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                            return [''] * len(s)
                        sty = sty.apply(_highlight_cash_row, axis=1)
                    
                    # Highlight TOTAL row
                    def _highlight_total_row(s):
                        if s.name == 'TOTAL':
                            return ['background-color: #1f4e79; color: white; font-weight: bold;' for _ in s]
                        return [''] * len(s)
                    sty = sty.apply(_highlight_total_row, axis=1)
                    
                    st.dataframe(sty, )
                except Exception:
                    st.dataframe(df_display, )
                
                # Add comprehensive portfolio data table right after the main allocation table
                build_comprehensive_portfolio_table(alloc_dict, portfolio_value)
            
            # Add comprehensive portfolio data table
            def build_comprehensive_portfolio_table(alloc_dict, portfolio_value):
                """
                Build a comprehensive table with all available financial indicators from Yahoo Finance
                """
                st.markdown("### 📊 Comprehensive Portfolio Data")
                st.markdown("#### Detailed financial indicators for each position")
                
                # Get current date for data freshness
                current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
                
                # Create progress bar for data fetching
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                rows = []
                tickers = [tk for tk in alloc_dict.keys() if tk != 'CASH']
                total_tickers = len(tickers)
                
                # OPTIMIZATION: Batch fetch all ticker infos at once (much faster!)
                status_text.text(f"Fetching data for {total_tickers} tickers in batch...")
                progress_bar.progress(0.1)
                all_infos = get_multiple_tickers_info_batch(tickers)
                
                for i, ticker in enumerate(tickers):
                    status_text.text(f"Processing {ticker}... ({i+1}/{total_tickers})")
                    progress_bar.progress((i + 1) / total_tickers)
                    
                    try:
                        # Get info from batch results
                        info = all_infos.get(ticker, {})
                        
                        # Get current price
                        current_price = info.get('currentPrice', info.get('regularMarketPrice', None))
                        if current_price is None:
                            # Try to get from historical data using get_ticker_data_for_valuation
                            # This ensures leveraged tickers (NVDL) use underlying data (NVDA) for price
                            hist_data = get_ticker_data_for_valuation(ticker, period='1d')
                            if hist_data is not None and not hist_data.empty:
                                current_price = hist_data['Close'].iloc[-1]
                        
                        # Calculate allocation values
                        alloc_pct = float(alloc_dict.get(ticker, 0))
                        allocation_value = portfolio_value * alloc_pct
                        shares = round(allocation_value / current_price, 1) if current_price and current_price > 0 else 0
                        total_val = shares * current_price if current_price else allocation_value
                        
                        # Determine if this is an ETF or stock for intelligent data handling
                        quote_type = info.get('quoteType', '').lower()
                        is_etf = quote_type in ['etf', 'fund']
                        is_commodity = ticker in ['GLD', 'SLV', 'USO', 'UNG'] or 'gold' in info.get('longName', '').lower()
                        
                        # Extract all available financial indicators with intelligent handling
                        # Get custom sector and industry for special tickers (ETFs, indices, etc.)
                        custom_sector = get_custom_sector_for_ticker(ticker)
                        sector = custom_sector if custom_sector else info.get('sector', 'N/A')
                        
                        custom_industry = get_custom_industry_for_ticker(ticker)
                        industry = custom_industry if custom_industry else info.get('industry', 'N/A')
                        
                        row = {
                            'Ticker': ticker,
                            'Company Name': info.get('longName', info.get('shortName', 'N/A')),
                            'Sector': sector,
                            'Industry': industry,
                            'Current Price ($)': current_price,
                            'Allocation %': alloc_pct * 100,
                            'Shares': shares,
                            'Total Value ($)': total_val,
                            '% of Portfolio': (total_val / portfolio_value * 100) if portfolio_value > 0 else 0,
                            
                            # Valuation Metrics (not applicable for commodities/ETFs)
                            'Market Cap ($B)': info.get('marketCap', 0) / 1e9 if info.get('marketCap') and not is_commodity else None,
                            'Enterprise Value ($B)': info.get('enterpriseValue', 0) / 1e9 if info.get('enterpriseValue') and not is_commodity else None,
                            'P/E Ratio': info.get('trailingPE', None) if not is_commodity else None,
                            'Forward P/E': info.get('forwardPE', None) if not is_commodity else None,
                            'PEG Ratio': None,  # Will be calculated below with fallback strategies
                            'Price/Book': None,  # Will be calculated manually below
                            'Price/Sales': info.get('priceToSalesTrailing12Months', None) if not is_commodity else None,
                            'Price/Cash Flow': info.get('priceToCashflow', None) if not is_commodity else None,
                            'EV/EBITDA': info.get('enterpriseToEbitda', None) if not is_commodity else None,
                            
                            # Financial Health (not applicable for commodities/ETFs)
                            'Debt/Equity': info.get('debtToEquity', None) if not is_commodity else None,
                            'Current Ratio': info.get('currentRatio', None) if not is_commodity else None,
                            'Quick Ratio': info.get('quickRatio', None) if not is_commodity else None,
                            'ROE (%)': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') and not is_commodity else None,
                            'ROA (%)': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') and not is_commodity else None,
                            'ROIC (%)': info.get('returnOnInvestedCapital', 0) * 100 if info.get('returnOnInvestedCapital') and not is_commodity else None,
                            
                            # Growth Metrics (not applicable for commodities/ETFs)
                            'Revenue Growth (%)': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') and not is_commodity else None,
                            'Earnings Growth (%)': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') and not is_commodity else None,
                            'EPS Growth (%)': info.get('earningsQuarterlyGrowth', 0) * 100 if info.get('earningsQuarterlyGrowth') and not is_commodity else None,
                            
                            # Dividend Information (available for ETFs and some stocks)
                            # Yahoo Finance returns dividend yield as decimal (0.0002 for 0.02%)
                            'Dividend Yield (%)': info.get('dividendYield', 0) if info.get('dividendYield') else None,
                            'Dividend Rate ($)': info.get('dividendRate', None),
                            'Payout Ratio (%)': info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') and not is_commodity else None,
                            '5Y Dividend Growth (%)': info.get('fiveYearAvgDividendYield', 0) if info.get('fiveYearAvgDividendYield') else None,
                            
                            # Trading Metrics (available for all securities)
                            '52W High ($)': info.get('fiftyTwoWeekHigh', None),
                            '52W Low ($)': info.get('fiftyTwoWeekLow', None),
                            '50D MA ($)': info.get('fiftyDayAverage', None),
                            '200D MA ($)': info.get('twoHundredDayAverage', None),
                            'Beta': info.get('beta', None),
                            'Volume': info.get('volume', None),
                            'Avg Volume': info.get('averageVolume', None),
                            
                            # Analyst Ratings (not available for commodities/ETFs)
                            'Analyst Rating': info.get('recommendationKey', 'N/A').title() if info.get('recommendationKey') and not is_commodity else 'N/A',
                            'Target Price ($)': info.get('targetMeanPrice', None) if not is_commodity else None,
                            'Target High ($)': info.get('targetHighPrice', None) if not is_commodity else None,
                            'Target Low ($)': info.get('targetLowPrice', None) if not is_commodity else None,
                            
                            # Additional Metrics (not applicable for commodities/ETFs)
                            'Book Value ($)': info.get('bookValue', None) if not is_commodity else None,
                            'Cash per Share ($)': info.get('totalCashPerShare', None) if not is_commodity else None,
                            'Revenue per Share ($)': info.get('revenuePerShare', None) if not is_commodity else None,
                            'Profit Margin (%)': info.get('profitMargins', 0) * 100 if info.get('profitMargins') and not is_commodity else None,
                            'Operating Margin (%)': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') and not is_commodity else None,
                            'Gross Margin (%)': info.get('grossMargins', 0) * 100 if info.get('grossMargins') and not is_commodity else None,
                        }
                        
                        # Simple PEG Ratio calculation: P/E ÷ Earnings Growth
                        pe_ratio = info.get('trailingPE')
                        earnings_growth = info.get('earningsGrowth')
                        peg_ratio = None
                        peg_source = "N/A"
                        
                        if not is_commodity and pe_ratio and pe_ratio > 0 and earnings_growth and earnings_growth > 0:
                            # Standard PEG Ratio calculation: P/E ÷ Earnings Growth Rate
                            # Yahoo Finance returns growth as decimal (0.15 = 15% growth)
                            # We need to convert to percentage for PEG calculation
                            growth_percentage = earnings_growth * 100
                            
                            peg_ratio = pe_ratio / growth_percentage
                            peg_source = "P/E ÷ Earnings Growth"
                        
                        # Price/Book will be calculated AFTER dual-class adjustment
                        
                        # Calculate EV/EBITDA manually to ensure consistency
                        enterprise_value = row.get('Enterprise Value ($B)')
                        ebitda = info.get('ebitda')
                        if enterprise_value and ebitda and ebitda > 0:
                            # Convert Enterprise Value from billions to actual value for calculation
                            ev_actual = enterprise_value * 1e9
                            row['EV/EBITDA'] = ev_actual / ebitda
                        
                        # Update the row with calculated PEG ratio and source
                        row['PEG Ratio'] = peg_ratio
                        row['PEG Source'] = peg_source
                        
                        # Define per-share fields for dual-class share adjustments
                        per_share_fields = ['Book Value ($)', 'Cash per Share ($)', 'Revenue per Share ($)']
                        
                        # Fix for dual-class shares - detect and adjust for share class differences
                        # This handles cases where Yahoo Finance returns same per-share data for different share classes
                        if current_price and current_price > 0:
                            # Store the current row data for comparison with other shares of same company
                            company_base = ticker.split('-')[0] if '-' in ticker else ticker
                            if company_base not in st.session_state:
                                st.session_state[company_base] = {}
                            
                            # Store reference data for this share class
                            st.session_state[company_base][ticker] = {
                                'price': current_price,
                                'per_share_data': {field: row[field] for field in per_share_fields if row[field]},
                                'enterprise_value': row.get('Enterprise Value ($B)')
                            }
                            
                            # Check if we have data for another share class of the same company
                            for other_ticker, other_data in st.session_state[company_base].items():
                                if other_ticker != ticker and other_data['price'] > 0:
                                    price_ratio = current_price / other_data['price']
                                    
                                    # If price ratio is significantly different (>10x), adjust per-share metrics
                                    # Use the higher-priced share class as reference to avoid scaling issues
                                    if price_ratio > 10 or price_ratio < 0.1:
                                        # Only adjust if current share class has lower price (use higher price as reference)
                                        if current_price and other_data['price'] and current_price < other_data['price']:
                                            for field in per_share_fields:
                                                if row[field] and other_data['per_share_data'].get(field):
                                                    # Use the price ratio to adjust the per-share values
                                                    row[field] = other_data['per_share_data'][field] * price_ratio
                                            
                                            # Also ensure Enterprise Value is consistent between share classes
                                            # Enterprise Value should be the same for both share classes of the same company
                                            if other_data.get('enterprise_value') is not None:
                                                row['Enterprise Value ($B)'] = other_data['enterprise_value']
                                                # Also recalculate EV/EBITDA with the corrected Enterprise Value
                                                ebitda = info.get('ebitda')
                                                if ebitda and ebitda > 0:
                                                    ev_actual = row['Enterprise Value ($B)'] * 1e9
                                                    row['EV/EBITDA'] = ev_actual / ebitda
                                        break
                        
                        # Calculate Price/Book ratio AFTER dual-class adjustment
                        book_value = row.get('Book Value ($)')
                        if current_price and book_value and book_value > 0:
                            row['Price/Book'] = current_price / book_value
                        elif current_price and book_value == 0:
                            row['Price/Book'] = None  # Explicitly set to None if book value is 0
                        
                        # Use Yahoo Finance data directly - no manual scaling fixes
                        # This ensures we get the most accurate data from Yahoo Finance
                        
                        # No manual calculations needed - use Yahoo Finance data directly
                        rows.append(row)
                        
                    except Exception as e:
                        st.warning(f"Error fetching data for {ticker}: {str(e)}")
                        # Add basic row with available data
                        alloc_pct = float(alloc_dict.get(ticker, 0))
                        allocation_value = portfolio_value * alloc_pct
                        rows.append({
                            'Ticker': ticker,
                            'Company Name': 'Error fetching data',
                            'Current Price ($)': None,
                            'Allocation %': alloc_pct * 100,
                            'Shares': 0,
                            'Total Value ($)': allocation_value,
                            '% of Portfolio': (allocation_value / portfolio_value * 100) if portfolio_value > 0 else 0,
                        })
                
                # Clear progress indicators
                progress_bar.empty()
                if 'status_text' in locals():
                    status_text.empty()
                
                if rows:
                    df_comprehensive = pd.DataFrame(rows)
                    # Store in session state for PDF generation
                    st.session_state.df_comprehensive = df_comprehensive
                    

                    
                    # Calculate portfolio-weighted metrics BEFORE formatting
                    def weighted_average(df, column, weight_column='% of Portfolio'):
                        """Calculate weighted average, handling NaN values and filtering invalid values"""
                        # Check if columns exist in DataFrame
                        if column not in df.columns or weight_column not in df.columns:
                            return None
                        
                        # Base mask for valid (non-NaN) values
                        valid_mask = df[column].notna() & df[weight_column].notna()
                        
                        # Special handling for PE ratios and similar valuation metrics
                        if 'P/E' in column or 'PE' in column or 'PEG' in column:
                            # Filter out negative or extremely high PE values that are likely errors
                            # Negative PE means negative earnings (company losing money)
                            # Very high PE (>1000) is usually a data error or near-zero earnings
                            valid_mask = valid_mask & (df[column] > 0) & (df[column] <= 1000)
                        elif column == 'Beta':
                            # Filter out extreme beta values (likely data errors)
                            valid_mask = valid_mask & (df[column] >= -5) & (df[column] <= 5)
                        elif 'Ratio' in column and column != 'PEG Ratio':
                            # For other ratios, filter out negative values (usually data errors)
                            valid_mask = valid_mask & (df[column] >= 0)
                        
                        if valid_mask.sum() == 0:
                            return None
                            
                        valid_df = df[valid_mask]
                        # Since weight_column is already in percentage, we divide by 100 to get decimal weights
                        result = (valid_df[column] * valid_df[weight_column] / 100).sum() / (valid_df[weight_column].sum() / 100)
                        
                        return result
                    
                    # Calculate all portfolio-weighted metrics first
                    portfolio_pe = weighted_average(df_comprehensive, 'P/E Ratio')
                    portfolio_pb = weighted_average(df_comprehensive, 'Price/Book')
                    portfolio_beta = weighted_average(df_comprehensive, 'Beta')
                    portfolio_peg = weighted_average(df_comprehensive, 'PEG Ratio')
                    portfolio_ps = weighted_average(df_comprehensive, 'Price/Sales')
                    portfolio_ev_ebitda = weighted_average(df_comprehensive, 'EV/EBITDA')
                    portfolio_roe = weighted_average(df_comprehensive, 'ROE (%)')
                    portfolio_roa = weighted_average(df_comprehensive, 'ROA (%)')
                    portfolio_roic = weighted_average(df_comprehensive, 'ROIC (%)')
                    portfolio_debt_equity = weighted_average(df_comprehensive, 'Debt/Equity')
                    portfolio_current_ratio = weighted_average(df_comprehensive, 'Current Ratio')
                    portfolio_quick_ratio = weighted_average(df_comprehensive, 'Quick Ratio')
                    portfolio_profit_margin = weighted_average(df_comprehensive, 'Profit Margin (%)')
                    portfolio_operating_margin = weighted_average(df_comprehensive, 'Operating Margin (%)')
                    portfolio_gross_margin = weighted_average(df_comprehensive, 'Gross Margin (%)')
                    portfolio_revenue_growth = weighted_average(df_comprehensive, 'Revenue Growth (%)')
                    portfolio_earnings_growth = weighted_average(df_comprehensive, 'Earnings Growth (%)')
                    portfolio_eps_growth = weighted_average(df_comprehensive, 'EPS Growth (%)')
                    portfolio_dividend_yield = weighted_average(df_comprehensive, 'Dividend Yield (%)')
                    
                    portfolio_payout_ratio = weighted_average(df_comprehensive, 'Payout Ratio (%)')
                    portfolio_dividend_growth = weighted_average(df_comprehensive, '5Y Dividend Growth (%)')
                    portfolio_market_cap = weighted_average(df_comprehensive, 'Market Cap ($B)')
                    portfolio_enterprise_value = weighted_average(df_comprehensive, 'Enterprise Value ($B)')
                    portfolio_forward_pe = weighted_average(df_comprehensive, 'Forward P/E')
                    
                    # Store portfolio metrics in session state for PDF generation
                    st.session_state.portfolio_pe = portfolio_pe
                    st.session_state.portfolio_pb = portfolio_pb
                    st.session_state.portfolio_beta = portfolio_beta
                    st.session_state.portfolio_peg = portfolio_peg
                    st.session_state.portfolio_ps = portfolio_ps
                    st.session_state.portfolio_ev_ebitda = portfolio_ev_ebitda
                    st.session_state.portfolio_roe = portfolio_roe
                    st.session_state.portfolio_roa = portfolio_roa
                    st.session_state.portfolio_roic = portfolio_roic
                    st.session_state.portfolio_debt_equity = portfolio_debt_equity
                    st.session_state.portfolio_current_ratio = portfolio_current_ratio
                    st.session_state.portfolio_quick_ratio = portfolio_quick_ratio
                    st.session_state.portfolio_profit_margin = portfolio_profit_margin
                    st.session_state.portfolio_operating_margin = portfolio_operating_margin
                    st.session_state.portfolio_gross_margin = portfolio_gross_margin
                    st.session_state.portfolio_revenue_growth = portfolio_revenue_growth
                    st.session_state.portfolio_earnings_growth = portfolio_earnings_growth
                    st.session_state.portfolio_eps_growth = portfolio_eps_growth
                    st.session_state.portfolio_dividend_yield = portfolio_dividend_yield
                    st.session_state.portfolio_payout_ratio = portfolio_payout_ratio
                    st.session_state.portfolio_dividend_growth = portfolio_dividend_growth
                    st.session_state.portfolio_market_cap = portfolio_market_cap
                    st.session_state.portfolio_enterprise_value = portfolio_enterprise_value
                    st.session_state.portfolio_forward_pe = portfolio_forward_pe
                    
                    # Calculate sector and industry breakdowns using ACTUAL portfolio allocations, not market values
                    sector_data = pd.Series(dtype=float)
                    industry_data = pd.Series(dtype=float)
                    
                    # Use 'Allocation %' column instead of '% of Portfolio' for accurate sector/industry breakdown
                    if 'Sector' in df_comprehensive.columns and 'Allocation %' in df_comprehensive.columns:
                        sector_data = df_comprehensive.groupby('Sector')['Allocation %'].sum().sort_values(ascending=False)
                    
                    if 'Industry' in df_comprehensive.columns and 'Allocation %' in df_comprehensive.columns:
                        industry_data = df_comprehensive.groupby('Industry')['Allocation %'].sum().sort_values(ascending=False)
                    
                    # Store sector and industry data in session state for PDF generation
                    st.session_state.sector_data = sector_data
                    st.session_state.industry_data = industry_data
                    
                    # Format the dataframe with safe formatting
                    def safe_format(value, format_str):
                        """Safely format values, handling NaN and None"""
                        if pd.isna(value) or value is None:
                            return 'N/A'
                        try:
                            if format_str.startswith('${:,.2f}'):
                                return f"${value:,.2f}"
                            elif format_str.startswith('{:,.2f}%'):
                                return f"{value:,.2f}%"
                            elif format_str.startswith('{:,.0f}'):
                                return f"{value:,.0f}"
                            elif format_str.startswith('{:,.2f}'):
                                return f"{value:,.2f}"
                            else:
                                return str(value)
                        except (ValueError, TypeError):
                            return 'N/A'
                    
                    # Apply safe formatting to all numeric columns
                    for col in df_comprehensive.columns:
                        if col in ['Ticker', 'Company Name', 'Sector', 'Industry', 'Analyst Rating']:
                            continue
                        elif col in ['Allocation %', 'ROE (%)', 'ROA (%)', 'ROIC (%)', 'Revenue Growth (%)', 
                                   'Earnings Growth (%)', 'EPS Growth (%)', 'Dividend Yield (%)', 
                                   'Payout Ratio (%)', '5Y Dividend Growth (%)', 'Profit Margin (%)', 
                                   'Operating Margin (%)', 'Gross Margin (%)', '% of Portfolio']:
                            df_comprehensive[col] = df_comprehensive[col].apply(lambda x: safe_format(x, '{:,.2f}%'))
                        elif col in ['Current Price ($)', 'Total Value ($)', '52W High ($)', '52W Low ($)', 
                                   '50D MA ($)', '200D MA ($)', 'Target Price ($)', 'Target High ($)', 
                                   'Target Low ($)', 'Book Value ($)', 'Cash per Share ($)', 
                                   'Revenue per Share ($)', 'Dividend Rate ($)']:
                            df_comprehensive[col] = df_comprehensive[col].apply(lambda x: safe_format(x, '${:,.2f}'))
                        elif col in ['Market Cap ($B)', 'Enterprise Value ($B)']:
                            df_comprehensive[col] = df_comprehensive[col].apply(lambda x: safe_format(x, '${:,.2f}B'))
                        elif col in ['Volume', 'Avg Volume']:
                            df_comprehensive[col] = df_comprehensive[col].apply(lambda x: safe_format(x, '{:,.0f}'))
                        else:
                            df_comprehensive[col] = df_comprehensive[col].apply(lambda x: safe_format(x, '{:,.2f}'))
                    
                    # Display the comprehensive table
                    st.markdown(f"**Data as of {current_date}**")
                    
                    # Create tabs for different metric categories
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "💰 Valuation", "🏥 Financial Health", "📊 Growth & Dividends", "📈 Technical"])
                    
                    with tab1:
                        # Overview tab - basic info and key metrics
                        overview_cols = ['Ticker', 'Company Name', 'Sector', 'Industry', 'Current Price ($)', 
                                       'Allocation %', 'Shares', 'Total Value ($)', '% of Portfolio', 
                                       'Market Cap ($B)', 'P/E Ratio', 'PEG Ratio', 'PEG Source', 'Beta', 'Analyst Rating']
                        df_overview = df_comprehensive[overview_cols].copy()
                        st.dataframe(df_overview, )
                    
                    with tab2:
                        # Valuation tab - all valuation metrics
                        valuation_cols = ['Ticker', 'Current Price ($)', 'Market Cap ($B)', 'Enterprise Value ($B)',
                                        'P/E Ratio', 'Forward P/E', 'PEG Ratio', 'PEG Source', 'Price/Book', 'Price/Sales',
                                        'Price/Cash Flow', 'EV/EBITDA', 'Book Value ($)', 'Cash per Share ($)',
                                        'Revenue per Share ($)', 'Target Price ($)', 'Target High ($)', 'Target Low ($)']
                        df_valuation = df_comprehensive[valuation_cols].copy()
                        st.dataframe(df_valuation, )
                    
                    with tab3:
                        # Financial Health tab - ratios and margins
                        health_cols = ['Ticker', 'Debt/Equity', 'Current Ratio', 'Quick Ratio', 'ROE (%)', 
                                     'ROA (%)', 'ROIC (%)', 'Profit Margin (%)', 'Operating Margin (%)', 
                                     'Gross Margin (%)']
                        df_health = df_comprehensive[health_cols].copy()
                        st.dataframe(df_health, )
                    
                    with tab4:
                        # Growth & Dividends tab
                        growth_cols = ['Ticker', 'Revenue Growth (%)', 'Earnings Growth (%)', 'EPS Growth (%)',
                                     'Dividend Yield (%)', 'Dividend Rate ($)', 'Payout Ratio (%)', 
                                     '5Y Dividend Growth (%)']
                        df_growth = df_comprehensive[growth_cols].copy()
                        st.dataframe(df_growth, )
                    
                    with tab5:
                        # Technical tab - price levels and volume
                        technical_cols = ['Ticker', 'Current Price ($)', '52W High ($)', '52W Low ($)', 
                                        '50D MA ($)', '200D MA ($)', 'Beta', 'Volume', 'Avg Volume']
                        df_technical = df_comprehensive[technical_cols].copy()
                        st.dataframe(df_technical, )
                    
                    # Add portfolio-weighted summary statistics in collapsible section
                    with st.expander("📊 Portfolio-Weighted Summary Statistics", expanded=True):
                        st.markdown("*Metrics weighted by portfolio allocation - represents the total portfolio characteristics*")
                        
                        # Add data accuracy warning
                        st.warning("⚠️ **Data Accuracy Notice:** Portfolio metrics (PE, Beta, etc.) are calculated from available data and may not accurately represent the portfolio if some ticker data is missing, outdated, or incorrect. These metrics should be used as indicative values for portfolio analysis.")
                        
                        # Create a comprehensive summary table
                        summary_data = []
                        
                        # Valuation metrics
                        if portfolio_pe is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "P/E Ratio", "Value": f"{portfolio_pe:.2f}", "Description": "Price-to-Earnings ratio weighted by portfolio allocation"})
                        if portfolio_forward_pe is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "Forward P/E", "Value": f"{portfolio_forward_pe:.2f}", "Description": "Forward Price-to-Earnings ratio weighted by portfolio allocation"})
                        if portfolio_pb is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "Price/Book", "Value": f"{portfolio_pb:.2f}", "Description": "Price-to-Book ratio weighted by portfolio allocation"})
                        if portfolio_peg is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "PEG Ratio", "Value": f"{portfolio_peg:.2f}", "Description": "P/E to Growth ratio weighted by portfolio allocation (calculated from best available source)"})
                        if portfolio_ps is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "Price/Sales", "Value": f"{portfolio_ps:.2f}", "Description": "Price-to-Sales ratio weighted by portfolio allocation"})
                        if portfolio_ev_ebitda is not None:
                            summary_data.append({"Category": "Valuation", "Metric": "EV/EBITDA", "Value": f"{portfolio_ev_ebitda:.2f}", "Description": "Enterprise Value to EBITDA ratio weighted by portfolio allocation"})
                        
                        # Risk metrics
                        if portfolio_beta is not None:
                            summary_data.append({"Category": "Risk", "Metric": "Beta", "Value": f"{portfolio_beta:.2f}", "Description": "Portfolio volatility relative to market (1.0 = market average)"})
                        
                        # Profitability metrics
                        if portfolio_roe is not None:
                            summary_data.append({"Category": "Profitability", "Metric": "ROE (%)", "Value": f"{portfolio_roe:.2f}%", "Description": "Return on Equity weighted by portfolio allocation"})
                        if portfolio_roa is not None:
                            summary_data.append({"Category": "Profitability", "Metric": "ROA (%)", "Value": f"{portfolio_roa:.2f}%", "Description": "Return on Assets weighted by portfolio allocation"})
                        if portfolio_profit_margin is not None:
                            summary_data.append({"Category": "Profitability", "Metric": "Profit Margin (%)", "Value": f"{portfolio_profit_margin:.2f}%", "Description": "Net profit margin weighted by portfolio allocation"})
                        if portfolio_operating_margin is not None:
                            summary_data.append({"Category": "Profitability", "Metric": "Operating Margin (%)", "Value": f"{portfolio_operating_margin:.2f}%", "Description": "Operating profit margin weighted by portfolio allocation"})
                        if portfolio_gross_margin is not None:
                            summary_data.append({"Category": "Profitability", "Metric": "Gross Margin (%)", "Value": f"{portfolio_gross_margin:.2f}%", "Description": "Gross profit margin weighted by portfolio allocation"})
                        
                        # Growth metrics
                        if portfolio_revenue_growth is not None:
                            summary_data.append({"Category": "Growth", "Metric": "Revenue Growth (%)", "Value": f"{portfolio_revenue_growth:.2f}%", "Description": "Revenue growth rate weighted by portfolio allocation"})
                        if portfolio_earnings_growth is not None:
                            summary_data.append({"Category": "Growth", "Metric": "Earnings Growth (%)", "Value": f"{portfolio_earnings_growth:.2f}%", "Description": "Earnings growth rate weighted by portfolio allocation"})
                        if portfolio_eps_growth is not None:
                            summary_data.append({"Category": "Growth", "Metric": "EPS Growth (%)", "Value": f"{portfolio_eps_growth:.2f}%", "Description": "Earnings per share growth rate weighted by portfolio allocation"})
                        
                        # Dividend metrics
                        if portfolio_dividend_yield is not None:
                            summary_data.append({"Category": "Dividends", "Metric": "Dividend Yield (%)", "Value": f"{portfolio_dividend_yield:.2f}%", "Description": "Dividend yield weighted by portfolio allocation"})
                        if portfolio_payout_ratio is not None:
                            summary_data.append({"Category": "Dividends", "Metric": "Payout Ratio (%)", "Value": f"{portfolio_payout_ratio:.2f}%", "Description": "Dividend payout ratio weighted by portfolio allocation"})
                        
                        # Size metrics
                        if portfolio_market_cap is not None:
                            summary_data.append({"Category": "Size", "Metric": "Market Cap ($B)", "Value": f"${portfolio_market_cap:.2f}B", "Description": "Market capitalization weighted by portfolio allocation"})
                        if portfolio_enterprise_value is not None:
                            summary_data.append({"Category": "Size", "Metric": "Enterprise Value ($B)", "Value": f"${portfolio_enterprise_value:.2f}B", "Description": "Enterprise value weighted by portfolio allocation"})
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, hide_index=True)
                            
                            # Add interpretation
                            st.markdown("**📈 Portfolio Interpretation:**")
                            if portfolio_beta is not None:
                                if portfolio_beta < 0.8:
                                    st.success(f"**Low Risk Portfolio** - Beta {portfolio_beta:.2f} indicates lower volatility than market")
                                elif portfolio_beta < 1.2:
                                    st.info(f"**Moderate Risk Portfolio** - Beta {portfolio_beta:.2f} indicates market-average volatility")
                                else:
                                    st.warning(f"**High Risk Portfolio** - Beta {portfolio_beta:.2f} indicates higher volatility than market")
                            
                            if portfolio_pe is not None:
                                if portfolio_pe < 15:
                                    st.success(f"**Undervalued Portfolio** - P/E {portfolio_pe:.2f} suggests attractive valuations")
                                elif portfolio_pe < 25:
                                    st.info(f"**Fairly Valued Portfolio** - P/E {portfolio_pe:.2f} suggests reasonable valuations")
                                else:
                                    st.warning(f"**Potentially Overvalued Portfolio** - P/E {portfolio_pe:.2f} suggests high valuations")
                        else:
                            st.warning("No portfolio-weighted metrics available for display.")
                    
                    # Add portfolio composition analysis
                    st.markdown("### 🏢 Portfolio Composition Analysis")
                    
                    # Create a nice table for sector and industry breakdown
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sector breakdown with table and pie chart
                        if not sector_data.empty:
                            st.markdown("**📊 Sector Allocation**")
                            
                            # Create sector table
                            sector_df = pd.DataFrame({
                                'Sector': sector_data.index,
                                'Allocation (%)': sector_data.values
                            }).round(2)
                            
                            # Display table
                            st.dataframe(sector_df, hide_index=True)
                            
                            # Create pie chart for sectors (filter out 0% allocations)
                            if len(sector_data) > 0:
                                # Filter out sectors with 0% allocation
                                sector_data_filtered = sector_data[sector_data > 0]
                                
                                if len(sector_data_filtered) > 0:
                                    fig_sector = go.Figure(data=[go.Pie(
                                        labels=sector_data_filtered.index,
                                        values=sector_data_filtered.values
                                    )])
                                    fig_sector.update_traces(textinfo='percent+label')
                                    fig_sector.update_layout(
                                        title="Sector Distribution",
                                        height=400,
                                        showlegend=True,
                                        margin=dict(t=50, b=50),
                                        template='plotly_dark'
                                    )
                                    st.plotly_chart(fig_sector, )
                    
                    with col2:
                        # Industry breakdown with table and pie chart
                        if not industry_data.empty:
                            st.markdown("**🏭 Industry Allocation**")
                            
                            # Create industry table
                            industry_df = pd.DataFrame({
                                'Industry': industry_data.index,
                                'Allocation (%)': industry_data.values
                            }).round(2)
                            
                            # Display table
                            st.dataframe(industry_df, hide_index=True)
                            
                            # Create pie chart for industries (filter out 0% allocations)
                            if len(industry_data) > 0:
                                # Filter out industries with 0% allocation
                                industry_data_filtered = industry_data[industry_data > 0]
                                
                                if len(industry_data_filtered) > 0:
                                    fig_industry = go.Figure(data=[go.Pie(
                                        labels=industry_data_filtered.index,
                                        values=industry_data_filtered.values
                                    )])
                                    fig_industry.update_traces(textinfo='percent+label')
                                    fig_industry.update_layout(
                                        title="Industry Distribution",
                                        height=400,
                                        showlegend=True,
                                        margin=dict(t=50, b=50),
                                        template='plotly_dark'
                                    )
                                    st.plotly_chart(fig_industry, )
                    
                    # Portfolio risk metrics
                    st.markdown("### ⚠️ Portfolio Risk Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # Beta analysis
                        if portfolio_beta is not None and not pd.isna(portfolio_beta):
                            if portfolio_beta < 0.8:
                                beta_risk = "Low Risk"
                                beta_color = "green"
                            elif portfolio_beta < 1.2:
                                beta_risk = "Balanced Risk"
                                beta_color = "green"
                            elif portfolio_beta < 1.5:
                                beta_risk = "Moderate Risk"
                                beta_color = "orange"
                            else:
                                beta_risk = "High Risk"
                                beta_color = "red"
                            st.metric("Portfolio Risk Level", beta_risk)
                            st.markdown(f"<span style='color: {beta_color}'>Beta: {portfolio_beta:.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.metric("Portfolio Risk Level", "NA")
                            st.markdown("<span style='color: gray'>Beta: NA</span>", unsafe_allow_html=True)
                    
                    with col2:
                        # Current P/E analysis
                        if portfolio_pe is not None and not pd.isna(portfolio_pe):
                            if portfolio_pe < 15:
                                pe_rating = "Undervalued"
                                pe_color = "green"
                            elif portfolio_pe < 25:
                                pe_rating = "Fair Value"
                                pe_color = "lime"
                            elif portfolio_pe < 35:
                                pe_rating = "Expensive"
                                pe_color = "orange"
                            else:
                                pe_rating = "Overvalued"
                                pe_color = "red"
                            st.metric("Current P/E Rating", pe_rating)
                            st.markdown(f"<span style='color: {pe_color}'>P/E: {portfolio_pe:.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.metric("Current P/E Rating", "NA")
                            st.markdown("<span style='color: gray'>P/E: NA</span>", unsafe_allow_html=True)
                    
                    with col3:
                        # Forward P/E analysis
                        if portfolio_forward_pe is not None and not pd.isna(portfolio_forward_pe):
                            if portfolio_forward_pe < 15:
                                fpe_rating = "Undervalued"
                                fpe_color = "green"
                            elif portfolio_forward_pe < 25:
                                fpe_rating = "Fair Value"
                                fpe_color = "lime"
                            elif portfolio_forward_pe < 35:
                                fpe_rating = "Expensive"
                                fpe_color = "orange"
                            else:
                                fpe_rating = "Overvalued"
                                fpe_color = "red"
                            st.metric("Forward P/E Rating", fpe_rating)
                            st.markdown(f"<span style='color: {fpe_color}'>Forward P/E: {portfolio_forward_pe:.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.metric("Forward P/E Rating", "NA")
                            st.markdown("<span style='color: gray'>Forward P/E: NA</span>", unsafe_allow_html=True)
                    
                    with col4:
                        # Dividend analysis
                        if portfolio_dividend_yield is not None and not pd.isna(portfolio_dividend_yield):
                            if portfolio_dividend_yield > 5:
                                div_rating = "Very High Yield"
                            elif portfolio_dividend_yield > 3:
                                div_rating = "Good Yield"
                            elif portfolio_dividend_yield > 1.5:
                                div_rating = "Moderate Yield"
                            else:
                                div_rating = "Low Yield"
                            st.metric("Dividend Rating", div_rating)
                            st.write(f"Yield: {portfolio_dividend_yield:.2f}%")
                        else:
                            st.metric("Dividend Rating", "NA")
                            st.write("Yield: NA")
                
                else:
                    st.warning("No portfolio data available to display.")
                
                # Add professional data explanation
                with st.expander("🔍 Data Sources & Methodology", expanded=False):
                    st.markdown("### Financial Data Information")
                    
                    st.markdown("**📊 Data Source**: All financial metrics are sourced directly from Yahoo Finance API")
                    st.markdown("**📈 Portfolio Metrics**: Weighted averages calculated based on portfolio allocation percentages")
                    st.markdown("**📊 Data Availability**: Some metrics may show N/A for securities where data is unavailable")
                    
                    st.markdown("**📊 Valuation Guidelines:**")
                    st.markdown("- **P/E Ratio**: <15 = undervalued, 15-25 = fair, >25 = potentially overvalued")
                    st.markdown("- **PEG Ratio**: <1 = undervalued, 1-2 = fair, >2 = potentially overvalued")
                    st.markdown("- **Price/Book**: <1 = potentially undervalued, 1-3 = fair, >3 = potentially overvalued")
                    st.markdown("- **Dividend Yield**: Low yield is not necessarily bad (growth stocks often have low yields)")
                    
                    st.markdown("**🔍 PEG Ratio Calculation:**")
                    st.markdown("- **Formula**: P/E Ratio ÷ Earnings Growth Rate")
                    st.markdown("- **What it measures**: Price relative to earnings growth (lower = better value)")
                    st.markdown("- **Source**: Direct calculation using Yahoo Finance data")
                    st.markdown("- **Realistic ranges**: <1.0 (undervalued), 1.0-1.5 (fair), >2.0 (overvalued)")
            
        
        # Add Portfolio Weighted Returns before Shares table
        st.markdown("### 📈 **Portfolio Weighted Returns**")
        
        # PE is now calculated directly and always up-to-date - no warning needed
        
        def calculate_portfolio_weighted_returns(available_data=None):
            """Calculate weighted portfolio returns for different periods"""
            try:
                # Get raw data
                snapshot = st.session_state.get('alloc_snapshot_data', {})
                raw_data = snapshot.get('raw_data') if snapshot and snapshot.get('raw_data') is not None else st.session_state.get('alloc_raw_data', {})
                
                if not raw_data:
                    return None
                
                today = pd.Timestamp.now().date()
                
                # Calculate different period returns using calendar-day lookbacks
                # This ensures assets that trade 7 days/week (e.g., BITCOIN) are correct
                periods = {
                    '1W': 7,      # 7 calendar days
                    '1M': 30,     # ~30 calendar days per month
                    '3M': 90,     # ~90 calendar days per quarter
                    '6M': 180,    # ~180 calendar days per half year
                    '1Y': 365     # ~365 calendar days per year
                }
                
                def _get_value_days_ago(series, days):
                    """Return value at or before last_date - days from a datetime-indexed Series."""
                    if series is None or len(series) == 0:
                        return None
                    last_date = pd.to_datetime(series.index[-1])
                    target_date = last_date - pd.Timedelta(days=days)
                    # Ensure datetime index
                    series.index = pd.to_datetime(series.index)
                    prior = series.loc[:target_date]
                    if len(prior) == 0:
                        return series.iloc[0]
                    return prior.iloc[-1]

                def get_value_days_ago(series, days):
                    """Return the series value at or before (last_date - days)."""
                    if series is None or len(series) == 0:
                        return None
                    last_date = series.index[-1]
                    target_date = last_date - pd.Timedelta(days=days)
                    # Slice up to target_date, pick last available
                    prior = series.loc[:target_date]
                    if len(prior) == 0:
                        # Not enough history; return first value
                        return series.iloc[0]
                    return prior.iloc[-1]
                
                
                portfolio_data = []
                
                # 1. PORTFOLIO (Historical) - Use backtest results directly
                historical_returns = {}
                try:
                    all_results = st.session_state.get('alloc_all_results', {})
                    if active_name in all_results:
                        portfolio_result = all_results[active_name]
                        if 'no_additions' in portfolio_result:
                            portfolio_values = portfolio_result['no_additions']
                            
                            for period_name, days in periods.items():
                                try:
                                    # Ensure we have enough data points
                                    if len(portfolio_values) < days + 1:
                                        historical_returns[period_name] = 'N/A'
                                        continue
                                    
                                    # Get current and past values safely
                                    current_value = portfolio_values.iloc[-1]
                                    past_value = portfolio_values.iloc[-(days + 1)]
                                    
                                    if past_value > 0:
                                        return_pct = ((current_value - past_value) / past_value) * 100
                                        historical_returns[period_name] = f"{return_pct:+.2f}%"
                                    else:
                                        historical_returns[period_name] = 'N/A'
                                        
                                except (IndexError, KeyError):
                                    historical_returns[period_name] = 'N/A'
                        else:
                            for period_name in periods.keys():
                                historical_returns[period_name] = 'N/A'
                    else:
                        for period_name in periods.keys():
                            historical_returns[period_name] = 'N/A'
                            
                except Exception as e:
                    print(f"[PORTFOLIO DEBUG] Error getting historical results: {e}")
                    for period_name in periods.keys():
                        historical_returns[period_name] = 'N/A'
                
                # NUCLEAR OPTION: Use backtest results directly (same as performance calculations)
                portfolio_pe_calculated = 'N/A'
                try:
                    # Use the SAME data source as performance calculations: alloc_all_results
                    all_results = st.session_state.get('alloc_all_results', {})
                    if active_name in all_results:
                        # Get portfolio info directly from backtest results
                        # Try to get PE from session state (calculated during backtest)
                        session_pe = getattr(st.session_state, 'portfolio_pe', None)
                        if session_pe is not None and not pd.isna(session_pe):
                            portfolio_pe_calculated = f"{session_pe:.2f}"
                        else:
                            # EMERGENCY FALLBACK: Calculate PE directly from portfolio config (like performance does)
                            if active_portfolio and 'stocks' in active_portfolio:
                                portfolio_tickers = [stock['ticker'] for stock in active_portfolio['stocks'] if stock.get('ticker')]
                                portfolio_allocations = {stock['ticker']: stock.get('allocation', 0) for stock in active_portfolio['stocks'] if stock.get('ticker')}
                                
                                if portfolio_tickers:
                                    # Fetch fresh info for portfolio tickers
                                    portfolio_info = get_multiple_tickers_info_batch(portfolio_tickers)
                                    
                                    # Calculate weighted PE
                                    total_weighted_pe = 0.0
                                    total_weight = 0.0
                                    valid_pe_count = 0
                                    
                                    for ticker in portfolio_tickers:
                                        weight = portfolio_allocations.get(ticker, 0)
                                        if weight <= 0:
                                            continue
                                            
                                        info = portfolio_info.get(ticker, {})
                                        pe = info.get('trailingPE')
                                        
                                        if pe is not None and pe > 0 and pe <= 1000:
                                            total_weighted_pe += pe * weight
                                            total_weight += weight
                                            valid_pe_count += 1
                                    
                                    if total_weight > 0 and valid_pe_count > 0:
                                        weighted_pe = total_weighted_pe / total_weight
                                        portfolio_pe_calculated = f"{weighted_pe:.2f}"
                    else:
                        pass
                    
                    # Fallback to df_comprehensive if available (but this is secondary)
                    if portfolio_pe_calculated == 'N/A' and hasattr(st.session_state, 'df_comprehensive') and st.session_state.df_comprehensive is not None:
                        df_comp = st.session_state.df_comprehensive
                        if not df_comp.empty and 'P/E Ratio' in df_comp.columns and '% of Portfolio' in df_comp.columns:
                            # Convert PE Ratio to numeric, replacing 'N/A' with NaN
                            pe_numeric = pd.to_numeric(df_comp['P/E Ratio'], errors='coerce')
                            
                            # Convert % of Portfolio to numeric, handling percentage strings like '24.83%'
                            portfolio_pct_str = df_comp['% of Portfolio'].astype(str)
                            portfolio_pct_numeric = portfolio_pct_str.str.replace('%', '').apply(pd.to_numeric, errors='coerce')
                            
                            # Apply same logic as weighted_average function
                            valid_mask = pe_numeric.notna() & portfolio_pct_numeric.notna()
                            valid_mask = valid_mask & (pe_numeric > 0) & (pe_numeric <= 1000)
                            
                            if valid_mask.sum() > 0:
                                valid_pe_numeric = pe_numeric[valid_mask]
                                valid_portfolio_pct = portfolio_pct_numeric[valid_mask]
                                
                                # Calculate weighted average (weights are already in percentage)
                                weighted_pe = (valid_pe_numeric * valid_portfolio_pct / 100).sum() / (valid_portfolio_pct.sum() / 100)
                                portfolio_pe_calculated = f"{weighted_pe:.2f}"
                            else:
                                pass
                        else:
                            pass
                    else:
                        # TEMPORARY: Check if we have session state PE as fallback
                        session_pe = getattr(st.session_state, 'portfolio_pe', None)
                        if session_pe is not None and not pd.isna(session_pe):
                            portfolio_pe_calculated = f"{session_pe:.2f}"
                        else:
                            pass
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    pass
                
                # Calculate Volatility and Beta for historical portfolio (last 252 trading days / ~1 year)
                portfolio_volatility = 'N/A'
                portfolio_beta = 'N/A'
                try:
                    all_results = st.session_state.get('alloc_all_results', {})
                    if active_name in all_results:
                        portfolio_result = all_results[active_name]
                        if 'no_additions' in portfolio_result:
                            portfolio_values = portfolio_result['no_additions']
                            if len(portfolio_values) >= 252:
                                # Use only last 252 trading days
                                portfolio_values_1y = portfolio_values.iloc[-252:]
                                portfolio_returns = portfolio_values_1y.pct_change().dropna()
                                volatility = portfolio_returns.std() * np.sqrt(252) * 100
                                portfolio_volatility = f"{volatility:.2f}%"
                                
                                # Simple Beta calculation - use pandas correlation and volatility ratio
                                try:
                                    # Use available_data for consistency with Benchmark table
                                    if 'SPY' in available_data and not available_data['SPY'].empty:
                                        spy_data = available_data['SPY'].copy()
                                        if 'Close' in spy_data.columns and len(spy_data) >= 252:
                                            spy_close = spy_data['Close'].iloc[-252:]
                                            spy_returns = spy_close.pct_change().dropna()
                                            
                                            # Take same length for both
                                            min_len = min(len(portfolio_returns), len(spy_returns))
                                            if min_len >= 200:
                                                port_ret = portfolio_returns.iloc[-min_len:]
                                                spy_ret = spy_returns.iloc[-min_len:]
                                                
                                                # Simple beta = correlation * (portfolio_vol / market_vol)
                                                correlation = port_ret.corr(spy_ret)
                                                port_vol = port_ret.std()
                                                spy_vol = spy_ret.std()
                                                
                                                if spy_vol > 0 and not np.isnan(correlation):
                                                    beta = correlation * (port_vol / spy_vol)
                                                    portfolio_beta = f"{beta:.2f}"
                                                else:
                                                    portfolio_beta = "1.00"
                                except Exception:
                                    pass
                except Exception:
                    pass
                
                historical_returns['Ticker'] = 'PORTFOLIO (Historical)'
                historical_returns['PE'] = portfolio_pe_calculated
                historical_returns['Volatility'] = portfolio_volatility
                historical_returns['Beta'] = portfolio_beta
                portfolio_data.append(historical_returns)
                
                # 2. PORTFOLIO (Current) - What current allocations would have done
                current_returns = {}
                current_weights = {**today_weights, 'CASH': today_weights.get('CASH', 0)}
                
                for period_name, days in periods.items():
                    try:
                        weighted_return = 0.0
                        total_weight = 0.0
                        
                        for ticker, weight in current_weights.items():
                            if ticker == 'CASH' or weight <= 0:
                                continue
                                
                            if ticker in raw_data and not raw_data[ticker].empty:
                                df = raw_data[ticker].copy()
                                if 'Close' not in df.columns or len(df) < days + 1:
                                    continue
                                
                                try:
                                    # Ensure datetime index
                                    df.index = pd.to_datetime(df.index)
                                    current_price = df['Close'].iloc[-1]
                                    past_price = _get_value_days_ago(df['Close'], days)
                                    
                                    if past_price is not None and past_price > 0:
                                        return_pct = ((current_price - past_price) / past_price) * 100
                                        weighted_return += return_pct * weight
                                        total_weight += weight
                                except Exception:
                                    continue
                        
                        if total_weight > 0:
                            # Normalize by total weight to get weighted average return
                            final_weighted_return = weighted_return / total_weight
                            current_returns[period_name] = f"{final_weighted_return:+.2f}%"
                        else:
                            current_returns[period_name] = 'N/A'
                            
                    except Exception:
                        current_returns[period_name] = 'N/A'
                
                # Calculate Volatility and Beta for current portfolio (last 252 trading days / ~1 year)
                current_volatility = 'N/A'
                current_beta = 'N/A'
                try:
                    weighted_volatility = 0.0
                    weighted_beta = 0.0
                    total_weight_vol = 0.0
                    
                    for ticker, weight in current_weights.items():
                        if ticker == 'CASH' or weight <= 0:
                            continue
                            
                        if ticker in raw_data and not raw_data[ticker].empty:
                            df = raw_data[ticker].copy()
                            if 'Close' in df.columns and len(df) >= 252:
                                # Use only last 252 trading days
                                ticker_returns = df['Close'].iloc[-252:].pct_change().dropna()
                                if len(ticker_returns) >= 200:  # Allow some flexibility
                                    ticker_vol = ticker_returns.std() * np.sqrt(252) * 100
                                    weighted_volatility += ticker_vol * weight
                                    
                                    # Simple Beta calculation for this ticker
                                    try:
                                        # Use available_data for consistency
                                        if 'SPY' in available_data and not available_data['SPY'].empty:
                                            spy_data = available_data['SPY'].copy()
                                            if 'Close' in spy_data.columns and len(spy_data) >= 252:
                                                spy_close = spy_data['Close'].iloc[-252:]
                                                spy_returns = spy_close.pct_change().dropna()
                                                
                                                # Take same length for both
                                                min_len = min(len(ticker_returns), len(spy_returns))
                                                if min_len >= 200:
                                                    ticker_ret = ticker_returns.iloc[-min_len:]
                                                    spy_ret = spy_returns.iloc[-min_len:]
                                                    
                                                    # Simple beta = correlation * (ticker_vol / market_vol)
                                                    correlation = ticker_ret.corr(spy_ret)
                                                    ticker_vol = ticker_ret.std()
                                                    spy_vol = spy_ret.std()
                                                    
                                                    if spy_vol > 0 and not np.isnan(correlation):
                                                        ticker_beta = correlation * (ticker_vol / spy_vol)
                                                        weighted_beta += ticker_beta * weight
                                    except Exception:
                                        pass
                                    
                                    total_weight_vol += weight
                    
                    if total_weight_vol > 0:
                        final_volatility = weighted_volatility / total_weight_vol
                        current_volatility = f"{final_volatility:.2f}%"
                        final_beta = weighted_beta / total_weight_vol
                        current_beta = f"{final_beta:.2f}"
                except Exception:
                    pass
                
                current_returns['Ticker'] = 'PORTFOLIO (Current)'
                current_returns['PE'] = portfolio_pe_calculated
                current_returns['Volatility'] = current_volatility
                current_returns['Beta'] = current_beta
                portfolio_data.append(current_returns)
                
                return portfolio_data
                
            except Exception as e:
                print(f"[PORTFOLIO RETURNS DEBUG] Error calculating portfolio returns: {e}")
                return None
        
        # Get available data for both functions
        snapshot = st.session_state.get('alloc_snapshot_data', {})
        raw_data = snapshot.get('raw_data') if snapshot and snapshot.get('raw_data') is not None else st.session_state.get('alloc_raw_data', {})
        
        # Prepare available_data for both portfolio and benchmark calculations
        available_data = {}
        benchmark_tickers = ['SPY', 'QQQ', 'SPMO', 'VTI', 'VT', 'SSO', 'QLD', 'BITCOIN']
        
        for ticker in benchmark_tickers:
            if raw_data and ticker in raw_data and not raw_data[ticker].empty:
                available_data[ticker] = raw_data[ticker].copy()
        
        # Download missing benchmarks if needed
        missing_tickers = [ticker for ticker in benchmark_tickers if ticker not in available_data]
        if missing_tickers:
            try:
                # Handle BITCOIN specially
                if 'BITCOIN' in missing_tickers:
                    try:
                        from Complete_Tickers.BITCOIN_COMPLETE_TICKER import create_bitcoin_complete_ticker
                        bitcoin_data = create_bitcoin_complete_ticker()
                        if bitcoin_data is not None and not bitcoin_data.empty:
                            available_data['BITCOIN'] = bitcoin_data
                        missing_tickers.remove('BITCOIN')
                    except Exception:
                        pass
                
                # Download other missing tickers
                if missing_tickers:
                    import yfinance as yf
                    batch_data = yf.download(missing_tickers, period="2y", interval="1d", progress=False, group_by='ticker')
                    if not batch_data.empty:
                        for ticker in missing_tickers:
                            if ticker in batch_data.columns.get_level_values(0):
                                df = batch_data[ticker].copy()
                                if df is not None and not df.empty and 'Close' in df.columns:
                                    available_data[ticker] = df
            except Exception:
                pass
        
        portfolio_returns_data = calculate_portfolio_weighted_returns(available_data)
        if portfolio_returns_data:
            # Create DataFrame from the list of portfolio returns
            df_portfolio_returns = pd.DataFrame(portfolio_returns_data)
            
            # Reorder columns to put Ticker first, then PE, then periods, then Volatility and Beta at the end
            period_cols = [col for col in df_portfolio_returns.columns if col not in ['Ticker', 'PE', 'Volatility', 'Beta']]
            columns = ['Ticker', 'PE'] + period_cols + ['Volatility', 'Beta']
            df_portfolio_returns = df_portfolio_returns[columns]
            
            # Style the dataframe
            styled_portfolio_returns = df_portfolio_returns.style
            
            # Apply coloring to each column separately
            for col in df_portfolio_returns.columns:
                if col == 'PE':
                    def style_pe(val):
                        if isinstance(val, str) and val != 'N/A' and not val.endswith('%'):
                            try:
                                pe_val = float(val)
                                if pe_val >= 35:
                                    return 'color: #ff4444; font-weight: bold'  # Red for PE >= 35 (Overvalued)
                                elif pe_val >= 25:
                                    return 'color: #ffaa00; font-weight: bold'  # Orange for PE 25-35 (Expensive)
                                elif pe_val >= 15:
                                    return 'color: #00ff00; font-weight: bold'  # Green for PE 15-25 (Fair Value)
                                else:
                                    return 'color: #00ff00; font-weight: bold'  # Green for PE < 15 (Undervalued)
                            except:
                                pass
                        return ''
                    styled_portfolio_returns = styled_portfolio_returns.applymap(style_pe, subset=[col])
                elif col not in ['Ticker', 'Beta', 'Volatility']:
                    def style_returns(val):
                        if isinstance(val, str) and val.endswith('%'):
                            try:
                                num_val = float(val.replace('%', '').replace('+', ''))
                                if num_val > 0:
                                    return 'color: #00ff00; font-weight: bold'
                                elif num_val < 0:
                                    return 'color: #ff4444; font-weight: bold'
                            except:
                                pass
                        return ''
                    styled_portfolio_returns = styled_portfolio_returns.applymap(style_returns, subset=[col])
            
            # Highlight the PORTFOLIO rows
            def highlight_portfolio_rows(row):
                if 'PORTFOLIO' in row['Ticker']:
                    return ['background-color: #333333; font-weight: bold; border: 2px solid #ffff00' for _ in row]
                return ['' for _ in row]
            
            # Apply row highlighting
            styled_portfolio_returns = styled_portfolio_returns.apply(highlight_portfolio_rows, axis=1)
            
            # Add custom CSS for uniform column widths
            st.markdown("""
            <style>
            /* Uniform column widths for Portfolio Returns table */
            .stDataFrame table {
                table-layout: fixed !important;
                width: 100% !important;
            }
            .stDataFrame table th:nth-child(1),
            .stDataFrame table td:nth-child(1) {
                width: 20% !important; /* Ticker column */
            }
            .stDataFrame table th:nth-child(2),
            .stDataFrame table td:nth-child(2) {
                width: 10% !important; /* PE column */
            }
            .stDataFrame table th:nth-child(n+3),
            .stDataFrame table td:nth-child(n+3) {
                width: 11.4% !important; /* Period columns (7 columns = 80% / 7) */
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(styled_portfolio_returns, )
        else:
            st.info("Portfolio returns data not available.")
        
        # Add Benchmark Comparison Table
        st.markdown("### 📊 **Benchmark Comparison**")
        # Clarify that figures are approximate snapshots
        st.caption("Approximate price-return snapshots using calendar lookbacks; may differ from total-return or month-end sources.")
        
        # PE is now calculated directly and always up-to-date - no warning needed
        
        def calculate_benchmark_returns(available_data=None, preloaded_info=None):
            """Calculate returns for benchmark tickers"""
            try:
                # Use the same active_name as Portfolio Weighted Returns
                active_name = active_portfolio.get('name') if active_portfolio else None
                
                # Get raw data
                snapshot = st.session_state.get('alloc_snapshot_data', {})
                raw_data = snapshot.get('raw_data') if snapshot and snapshot.get('raw_data') is not None else st.session_state.get('alloc_raw_data', {})
                
                today = pd.Timestamp.now().date()
                
                # Calculate different period returns using calendar-day lookbacks
                periods = {
                    '1W': 7,      # 7 calendar days
                    '1M': 30,     # ~30 calendar days per month
                    '3M': 90,     # ~90 calendar days per quarter
                    '6M': 180,    # ~180 calendar days per half year
                    '1Y': 365     # ~365 calendar days per year
                }

                def get_value_days_ago(series, days):
                    """Return the value at or before last_date - days from a datetime-indexed Series/DataFrame column."""
                    if series is None or len(series) == 0:
                        return None
                    last_date = pd.to_datetime(series.index[-1])
                    target_date = last_date - pd.Timedelta(days=days)
                    # Ensure datetime index
                    idx = pd.to_datetime(series.index)
                    series.index = idx
                    prior = series.loc[:target_date]
                    if len(prior) == 0:
                        return series.iloc[0]
                    return prior.iloc[-1]

                # Benchmark tickers to compare (in specific order)
                benchmark_tickers = ['SPY', 'QQQ', 'SPMO', 'VTI', 'VT', 'SSO', 'QLD', 'BITCOIN']
                
                # Use preloaded info if available (for performance)
                if preloaded_info is None:
                    preloaded_info = {}
                
                benchmark_data = []
                
                # Use available_data passed as parameter (already prepared outside)
                
                # available_data is already prepared outside this function
                
                # Add PORTFOLIO row first for comparison (using backtest results directly)
                portfolio_returns_dict = {}
                
                # Get the backtest results for this portfolio
                try:
                    # Get the portfolio value series from backtest results
                    all_results = st.session_state.get('alloc_all_results', {})
                    if active_name in all_results:
                        portfolio_result = all_results[active_name]
                        if 'no_additions' in portfolio_result:
                            portfolio_values = portfolio_result['no_additions']
                            
                            # Calculate period returns using calendar-day lookbacks
                            # Ensure index is datetime
                            pv = portfolio_values.copy()
                            pv.index = pd.to_datetime(pv.index)
                            for period_name, days in periods.items():
                                try:
                                    current_value = pv.iloc[-1]
                                    past_value = get_value_days_ago(pv, days)
                                    if past_value is not None and past_value > 0:
                                        return_pct = ((current_value - past_value) / past_value) * 100
                                        portfolio_returns_dict[period_name] = f"{return_pct:+.2f}%"
                                    else:
                                        portfolio_returns_dict[period_name] = 'N/A'
                                except Exception:
                                    portfolio_returns_dict[period_name] = 'N/A'
                        else:
                            # Fallback: all N/A if no portfolio data
                            for period_name in periods.keys():
                                portfolio_returns_dict[period_name] = 'N/A'
                    else:
                        # Fallback: all N/A if no portfolio data
                        for period_name in periods.keys():
                            portfolio_returns_dict[period_name] = 'N/A'
                            
                except Exception as e:
                    print(f"[PORTFOLIO DEBUG] Error getting backtest results: {e}")
                    # Fallback: all N/A if error
                    for period_name in periods.keys():
                        portfolio_returns_dict[period_name] = 'N/A'
                
                # Add PORTFOLIO as first row
                portfolio_returns_dict['Ticker'] = 'PORTFOLIO'
                
                # NUCLEAR OPTION: Use backtest results directly (same as performance calculations)
                portfolio_pe_calculated = 'N/A'
                try:
                    # Use the SAME data source as performance calculations: alloc_all_results
                    all_results = st.session_state.get('alloc_all_results', {})
                    if active_name in all_results:
                        # Get portfolio info directly from backtest results
                        # Try to get PE from session state (calculated during backtest)
                        session_pe = getattr(st.session_state, 'portfolio_pe', None)
                        if session_pe is not None and not pd.isna(session_pe):
                            portfolio_pe_calculated = f"{session_pe:.2f}"
                        else:
                            # EMERGENCY FALLBACK: Calculate PE directly from portfolio config (like performance does)
                            if active_portfolio and 'stocks' in active_portfolio:
                                portfolio_tickers = [stock['ticker'] for stock in active_portfolio['stocks'] if stock.get('ticker')]
                                portfolio_allocations = {stock['ticker']: stock.get('allocation', 0) for stock in active_portfolio['stocks'] if stock.get('ticker')}
                                
                                if portfolio_tickers:
                                    # Fetch fresh info for portfolio tickers
                                    portfolio_info = get_multiple_tickers_info_batch(portfolio_tickers)
                                    
                                    # Calculate weighted PE
                                    total_weighted_pe = 0.0
                                    total_weight = 0.0
                                    valid_pe_count = 0
                                    
                                    for ticker in portfolio_tickers:
                                        weight = portfolio_allocations.get(ticker, 0)
                                        if weight <= 0:
                                            continue
                                            
                                        info = portfolio_info.get(ticker, {})
                                        pe = info.get('trailingPE')
                                        
                                        if pe is not None and pe > 0 and pe <= 1000:
                                            total_weighted_pe += pe * weight
                                            total_weight += weight
                                            valid_pe_count += 1
                                    
                                    if total_weight > 0 and valid_pe_count > 0:
                                        weighted_pe = total_weighted_pe / total_weight
                                        portfolio_pe_calculated = f"{weighted_pe:.2f}"
                    else:
                        pass
                    
                    # Fallback to df_comprehensive if available (but this is secondary)
                    if portfolio_pe_calculated == 'N/A' and hasattr(st.session_state, 'df_comprehensive') and st.session_state.df_comprehensive is not None:
                        df_comp = st.session_state.df_comprehensive
                        if not df_comp.empty and 'P/E Ratio' in df_comp.columns and '% of Portfolio' in df_comp.columns:
                            # Convert PE Ratio to numeric, replacing 'N/A' with NaN
                            pe_numeric = pd.to_numeric(df_comp['P/E Ratio'], errors='coerce')
                            
                            # Convert % of Portfolio to numeric, handling percentage strings like '24.83%'
                            portfolio_pct_str = df_comp['% of Portfolio'].astype(str)
                            portfolio_pct_numeric = portfolio_pct_str.str.replace('%', '').apply(pd.to_numeric, errors='coerce')
                            
                            # Apply same logic as weighted_average function
                            valid_mask = pe_numeric.notna() & portfolio_pct_numeric.notna()
                            valid_mask = valid_mask & (pe_numeric > 0) & (pe_numeric <= 1000)
                            
                            if valid_mask.sum() > 0:
                                valid_pe_numeric = pe_numeric[valid_mask]
                                valid_portfolio_pct = portfolio_pct_numeric[valid_mask]
                                
                                # Calculate weighted average (weights are already in percentage)
                                weighted_pe = (valid_pe_numeric * valid_portfolio_pct / 100).sum() / (valid_portfolio_pct.sum() / 100)
                                portfolio_pe_calculated = f"{weighted_pe:.2f}"
                            else:
                                pass
                        else:
                            pass
                    else:
                        # TEMPORARY: Check if we have session state PE as fallback
                        session_pe = getattr(st.session_state, 'portfolio_pe', None)
                        if session_pe is not None and not pd.isna(session_pe):
                            portfolio_pe_calculated = f"{session_pe:.2f}"
                        else:
                            pass
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    pass
                portfolio_volatility_calculated = 'N/A'
                portfolio_beta_calculated = 'N/A'
                
                # Get Volatility and Beta from historical portfolio results (last 365 calendar days)
                try:
                    all_results = st.session_state.get('alloc_all_results', {})
                    if active_name and active_name in all_results:
                        portfolio_result = all_results[active_name]
                        if 'no_additions' in portfolio_result:
                            portfolio_values = portfolio_result['no_additions']
                            if len(portfolio_values) >= 60:
                                # Use last 365 calendar days for the volatility/beta window
                                pv = portfolio_values.copy()
                                pv.index = pd.to_datetime(pv.index)
                                start_date = pv.index[-1] - pd.Timedelta(days=365)
                                portfolio_values_1y = pv.loc[start_date:]
                                portfolio_returns = portfolio_values_1y.pct_change().dropna()
                                # Annualize using 252 trading days convention
                                volatility = portfolio_returns.std() * np.sqrt(252) * 100
                                portfolio_volatility_calculated = f"{volatility:.2f}%"
                                
                                # Simple Beta calculation against SPY
                                try:
                                    if 'SPY' in available_data and not available_data['SPY'].empty:
                                        spy_data = available_data['SPY'].copy()
                                        if 'Close' in spy_data.columns and len(spy_data) >= 60:
                                            spy_data.index = pd.to_datetime(spy_data.index)
                                            spy_close = spy_data['Close']
                                            spy_ret_window = spy_close.loc[start_date:]
                                            spy_returns = spy_ret_window.pct_change().dropna()
                                            # Align on common dates
                                            common_idx = portfolio_returns.index.intersection(spy_returns.index)
                                            if len(common_idx) >= 60:
                                                port_ret = portfolio_returns.reindex(common_idx).dropna()
                                                spy_ret = spy_returns.reindex(common_idx).dropna()
                                                
                                                # Simple beta = correlation * (portfolio_vol / market_vol)
                                                correlation = port_ret.corr(spy_ret)
                                                port_vol = port_ret.std()
                                                spy_vol = spy_ret.std()
                                                
                                                if spy_vol > 0 and not np.isnan(correlation):
                                                    beta = correlation * (port_vol / spy_vol)
                                                    portfolio_beta_calculated = f"{beta:.2f}"
                                                else:
                                                    portfolio_beta_calculated = "1.00"
                                except Exception:
                                    pass
                except Exception:
                    pass
                
                portfolio_returns_dict['PE'] = portfolio_pe_calculated
                portfolio_returns_dict['Volatility'] = portfolio_volatility_calculated
                portfolio_returns_dict['Beta'] = portfolio_beta_calculated
                benchmark_data.append(portfolio_returns_dict)
                
                # Calculate returns for all available benchmarks
                for ticker in benchmark_tickers:
                    if ticker not in available_data:
                        continue
                    
                    df = available_data[ticker]
                    if df is None or 'Close' not in df.columns:
                        continue
                    
                    ticker_returns = {'Ticker': ticker}
                    
                    # Get PE ratio for this benchmark ticker from preloaded info
                    ticker_pe = 'N/A'
                    try:
                        info = preloaded_info.get(ticker, {})
                        if info and 'trailingPE' in info and info['trailingPE'] is not None:
                            ticker_pe = f"{info['trailingPE']:.2f}"
                        else:
                            # Debug: Check what info we actually have for this ticker
                            print(f"[BENCHMARK PE DEBUG] {ticker}: trailingPE = {info.get('trailingPE', 'NOT_FOUND')}")
                    except Exception as e:
                        print(f"[BENCHMARK PE DEBUG] {ticker}: Error = {e}")
                        pass
                    
                    ticker_returns['PE'] = ticker_pe
                    
                    # Ensure datetime index
                    df_local = df.copy()
                    df_local.index = pd.to_datetime(df_local.index)
                    for period_name, days in periods.items():
                        try:
                            current_price = df_local['Close'].iloc[-1]
                            past_price = get_value_days_ago(df_local['Close'], days)
                            if past_price is not None and past_price > 0:
                                return_pct = ((current_price - past_price) / past_price) * 100
                                ticker_returns[period_name] = f"{return_pct:+.2f}%"
                            else:
                                ticker_returns[period_name] = 'N/A'
                        except Exception:
                            ticker_returns[period_name] = 'N/A'
                    
                    # Calculate Volatility and Beta for this ticker (last 365 calendar days)
                    ticker_volatility = 'N/A'
                    ticker_beta = 'N/A'
                    try:
                        if len(df) >= 60:
                            # Use last 365 calendar days
                            df_local = df.copy()
                            df_local.index = pd.to_datetime(df_local.index)
                            start_date = df_local.index[-1] - pd.Timedelta(days=365)
                            ticker_window = df_local['Close'].loc[start_date:]
                            ticker_returns_series = ticker_window.pct_change().dropna()
                            if len(ticker_returns_series) >= 60:  # Allow some flexibility
                                # Annualize using 252 for weekdays assets, 365 for always-open (BITCOIN)
                                annualize_days = 365 if ticker == 'BITCOIN' else 252
                                ticker_vol = ticker_returns_series.std() * np.sqrt(annualize_days) * 100
                                ticker_volatility = f"{ticker_vol:.2f}%"
                                
                                if ticker == 'SPY':
                                    ticker_beta = "1.00"
                                elif 'SPY' in available_data and not available_data['SPY'].empty:
                                    spy_data = available_data['SPY'].copy()
                                    if 'Close' in spy_data.columns and len(spy_data) >= 60:
                                        spy_data.index = pd.to_datetime(spy_data.index)
                                        spy_close = spy_data['Close']
                                        spy_ret_window = spy_close.loc[start_date:]
                                        spy_returns = spy_ret_window.pct_change().dropna()
                                        # Align on common dates
                                        common_idx = ticker_returns_series.index.intersection(spy_returns.index)
                                        if len(common_idx) >= 60:
                                            ticker_ret = ticker_returns_series.reindex(common_idx).dropna()
                                            spy_ret = spy_returns.reindex(common_idx).dropna()
                                            
                                            # Simple beta = correlation * (ticker_vol / market_vol)
                                            correlation = ticker_ret.corr(spy_ret)
                                            ticker_vol = ticker_ret.std()
                                            spy_vol = spy_ret.std()
                                            
                                            if spy_vol > 0 and not np.isnan(correlation):
                                                beta = correlation * (ticker_vol / spy_vol)
                                                ticker_beta = f"{beta:.2f}"
                                            else:
                                                ticker_beta = "1.00"
                    except Exception:
                        pass
                    
                    ticker_returns['Volatility'] = ticker_volatility
                    ticker_returns['Beta'] = ticker_beta
                    benchmark_data.append(ticker_returns)
                
                if benchmark_data:
                    df_benchmark = pd.DataFrame(benchmark_data)
                    # Reorder columns to put Ticker first, then PE, then periods, then Volatility and Beta at the end
                    period_cols = [col for col in df_benchmark.columns if col not in ['Ticker', 'PE', 'Volatility', 'Beta']]
                    columns = ['Ticker', 'PE'] + period_cols + ['Volatility', 'Beta']
                    df_benchmark = df_benchmark[columns]
                    return df_benchmark
                
            except Exception as e:
                pass
                return None
        
        # Preload benchmark ticker info BEFORE calculating returns to ensure PE ratios are available immediately
        # NUCLEAR OPTION: Portfolio PE is already calculated and stored in session state, no need to preload portfolio tickers!
        benchmark_tickers_to_preload = ['SPY', 'QQQ', 'SPMO', 'VTI', 'VT', 'SSO', 'QLD', 'BITCOIN']
        preloaded_benchmark_info = get_multiple_tickers_info_batch(benchmark_tickers_to_preload)
        
        benchmark_df = calculate_benchmark_returns(available_data, preloaded_benchmark_info)
        if benchmark_df is not None and not benchmark_df.empty:
            # Style the dataframe
            styled_benchmark = benchmark_df.style
            
            # Apply coloring to each column separately
            for col in benchmark_df.columns:
                if col == 'PE':
                    def style_pe(val):
                        if isinstance(val, str) and val != 'N/A' and not val.endswith('%'):
                            try:
                                pe_val = float(val)
                                if pe_val >= 35:
                                    return 'color: #ff4444; font-weight: bold'  # Red for PE >= 35 (Overvalued)
                                elif pe_val >= 25:
                                    return 'color: #ffaa00; font-weight: bold'  # Orange for PE 25-35 (Expensive)
                                elif pe_val >= 15:
                                    return 'color: #00ff00; font-weight: bold'  # Green for PE 15-25 (Fair Value)
                                else:
                                    return 'color: #00ff00; font-weight: bold'  # Green for PE < 15 (Undervalued)
                            except:
                                pass
                        return ''
                    styled_benchmark = styled_benchmark.applymap(style_pe, subset=[col])
                elif col not in ['Ticker', 'Beta', 'Volatility']:
                    def style_returns(val):
                        if isinstance(val, str) and val.endswith('%'):
                            try:
                                num_val = float(val.replace('%', '').replace('+', ''))
                                if num_val > 0:
                                    return 'color: #00ff00; font-weight: bold'
                                elif num_val < 0:
                                    return 'color: #ff4444; font-weight: bold'
                            except:
                                pass
                        return ''
                    styled_benchmark = styled_benchmark.applymap(style_returns, subset=[col])
            
            # Highlight the PORTFOLIO row in benchmark table
            def highlight_benchmark_portfolio_row(row):
                if row['Ticker'] == 'PORTFOLIO':
                    return ['background-color: #333333; font-weight: bold; border: 2px solid #ffff00' for _ in row]
                return ['' for _ in row]
            
            # Apply row highlighting
            styled_benchmark = styled_benchmark.apply(highlight_benchmark_portfolio_row, axis=1)
            
            # Add custom CSS for uniform column widths
            st.markdown("""
            <style>
            /* Uniform column widths for Benchmark Comparison table */
            .stDataFrame table {
                table-layout: fixed !important;
                width: 100% !important;
            }
            .stDataFrame table th:nth-child(1),
            .stDataFrame table td:nth-child(1) {
                width: 15% !important; /* Ticker column */
            }
            .stDataFrame table th:nth-child(2),
            .stDataFrame table td:nth-child(2) {
                width: 8% !important; /* PE column */
            }
            .stDataFrame table th:nth-child(n+3),
            .stDataFrame table td:nth-child(n+3) {
                width: 11% !important; /* Period columns (7 columns = 77% / 7) */
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(styled_benchmark, )
        else:
            st.info("Benchmark data not available.")
        
        build_table_from_alloc({**today_weights, 'CASH': today_weights.get('CASH', 0)}, None, f"Shares if Rebalanced Today (snapshot)")

    if allocs_for_portfolio:
        st.markdown("**Historical Allocations**")
        # Ensure proper DataFrame structure with explicit column names - FIXED LIKE PAGE 1
        # First, collect all tickers excluding None (EXACTLY LIKE PAGE 1)
        all_tickers = set()
        for date, alloc_dict in allocs_for_portfolio.items():
            for ticker in alloc_dict.keys():
                if ticker is not None:
                    all_tickers.add(ticker)
        
        # Create complete data structure with all tickers for all dates
        complete_data = {}
        for date, alloc_dict in allocs_for_portfolio.items():
            complete_data[date] = {ticker: alloc_dict.get(ticker, 0) for ticker in all_tickers}
        
        allocations_df_raw = pd.DataFrame(complete_data).T
        
        # Fill missing values with 0 for unavailable assets (EXACTLY LIKE PAGE 1)
        allocations_df_raw = allocations_df_raw.fillna(0)
        
        # Sort tickers with CASH always last (EXACTLY LIKE PAGE 1)
        ticker_list = sorted(list(all_tickers))
        if 'CASH' in ticker_list:
            ticker_list.remove('CASH')
            ticker_list.append('CASH')
        
        # Reorder DataFrame columns to match the desired order
        allocations_df_raw = allocations_df_raw[ticker_list]
        
        allocations_df_raw.index.name = "Date"

        def highlight_rows_by_index(s):
            is_even_row = allocations_df_raw.index.get_loc(s.name) % 2 == 0
            bg_color = 'background-color: #0e1117' if is_even_row else 'background-color: #262626'
            return [f'{bg_color}; color: white;'] * len(s)

        styler = allocations_df_raw.mul(100).style.apply(highlight_rows_by_index, axis=1)
        styler.format('{:,.0f}%', na_rep='N/A')
        
        # NUCLEAR OPTION: Inject custom CSS to override Streamlit's stubborn styling
        st.markdown("""
        <style>
        /* NUCLEAR CSS OVERRIDE - BEAT STREAMLIT INTO SUBMISSION */
        .stDataFrame [data-testid="stDataFrame"] div[data-testid="stDataFrame"] table td,
        .stDataFrame [data-testid="stDataFrame"] div[data-testid="stDataFrame"] table th,
        .stDataFrame table td,
        .stDataFrame table th {
            color: #FFFFFF !important;
            font-weight: bold !important;
            text-shadow: 1px 1px 2px black !important;
        }
        
        /* Force ALL text in dataframes to be white */
        .stDataFrame * {
            color: #FFFFFF !important;
        }
        
        /* Override any Streamlit bullshit */
        .stDataFrame [data-testid="stDataFrame"] *,
        .stDataFrame div[data-testid="stDataFrame"] * {
            color: #FFFFFF !important;
            font-weight: bold !important;
        }
        
        /* Target EVERYTHING in the dataframe */
        .stDataFrame table *,
        .stDataFrame div table *,
        .stDataFrame [data-testid="stDataFrame"] table *,
        .stDataFrame [data-testid="stDataFrame"] div table * {
            color: #FFFFFF !important;
            font-weight: bold !important;
        }
        
        /* Target all cells specifically */
        .stDataFrame td,
        .stDataFrame th,
        .stDataFrame table td,
        .stDataFrame table th {
            color: #FFFFFF !important;
            font-weight: bold !important;
        }
        
        /* Target Streamlit's specific elements */
        div[data-testid="stDataFrame"] table td,
        div[data-testid="stDataFrame"] table th,
        div[data-testid="stDataFrame"] div table td,
        div[data-testid="stDataFrame"] div table th {
            color: #FFFFFF !important;
            font-weight: bold !important;
        }
        
        /* Target everything with maximum specificity */
        div[data-testid="stDataFrame"] *,
        div[data-testid="stDataFrame"] div *,
        div[data-testid="stDataFrame"] table *,
        div[data-testid="stDataFrame"] div table * {
            color: #FFFFFF !important;
            font-weight: bold !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(styler, )

    if metrics_for_portfolio:
        st.markdown("---")
        st.markdown("**Rebalancing Metrics & Calculated Weights**")
        metrics_records = []
        for date, tickers_data in metrics_for_portfolio.items():
            for ticker, data in tickers_data.items():
                # Handle None ticker as CASH
                display_ticker = 'CASH' if ticker is None else ticker
                filtered_data = {k: v for k, v in (data or {}).items() if k != 'Composite'}
                
                # Check if momentum is used for this portfolio
                use_momentum = active_portfolio.get('use_momentum', True) if active_portfolio else True
                
                # If momentum is not used, replace Calculated_Weight with target_allocation
                if not use_momentum:
                    if 'target_allocation' in filtered_data:
                        filtered_data['Calculated_Weight'] = filtered_data['target_allocation']
                    else:
                        # If target_allocation is not available, use the entered allocations from active_portfolio
                        ticker_name = display_ticker if display_ticker != 'CASH' else None
                        if ticker_name:
                            # Find the stock in active_portfolio and use its allocation
                            for stock in active_portfolio.get('stocks', []):
                                if stock.get('ticker', '').strip() == ticker_name:
                                    filtered_data['Calculated_Weight'] = stock.get('allocation', 0)
                                    break
                        else:
                            # For CASH, calculate the remaining allocation
                            total_alloc = sum(stock.get('allocation', 0) for stock in active_portfolio.get('stocks', []))
                            filtered_data['Calculated_Weight'] = max(0, 1.0 - total_alloc)
                
                record = {'Date': date, 'Ticker': display_ticker, **filtered_data}
                metrics_records.append(record)
            
            # Ensure CASH line is added if there's non-zero cash in allocations
            if allocs_for_portfolio and date in allocs_for_portfolio:
                cash_alloc = allocs_for_portfolio[date].get('CASH', 0)
                if cash_alloc > 0:
                    # Check if CASH is already in metrics_records for this date
                    cash_exists = any(record['Date'] == date and record['Ticker'] == 'CASH' for record in metrics_records)
                    if not cash_exists:
                        # Add CASH line to metrics
                        # Check if momentum is used to determine which weight to show
                        use_momentum = active_portfolio.get('use_momentum', True) if active_portfolio else True
                        if not use_momentum:
                            # When momentum is not used, calculate CASH allocation from entered allocations
                            total_alloc = sum(stock.get('allocation', 0) for stock in active_portfolio.get('stocks', []))
                            cash_weight = max(0, 1.0 - total_alloc)
                            cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_weight}
                        else:
                            cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_alloc}
                        metrics_records.append(cash_record)

        if metrics_records:
            metrics_df = pd.DataFrame(metrics_records)
            
            # Filter out CASH lines where Calculated_Weight is 0 for the last date
            if 'Calculated_Weight' in metrics_df.columns:
                # Get the last date
                last_date = metrics_df['Date'].max()
                # Remove CASH records where Calculated_Weight is 0 for the last date
                mask = ~((metrics_df['Ticker'] == 'CASH') & (metrics_df['Date'] == last_date) & (metrics_df['Calculated_Weight'] == 0))
                metrics_df = metrics_df[mask].reset_index(drop=True)
            
            if not metrics_df.empty:
                metrics_df.set_index(['Date', 'Ticker'], inplace=True)
                metrics_df_display = metrics_df.copy()
            if 'Momentum' in metrics_df_display.columns:
                metrics_df_display['Momentum'] = metrics_df_display['Momentum'].fillna(0) * 100
            if 'Calculated_Weight' in metrics_df_display.columns:
                metrics_df_display['Calculated_Weight'] = metrics_df_display['Calculated_Weight'].fillna(0) * 100
            if 'Volatility' in metrics_df_display.columns:
                metrics_df_display['Volatility'] = metrics_df_display['Volatility'].fillna(np.nan) * 100

            def color_momentum(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                # Force white color for None, NA, and other non-numeric values
                return 'color: #FFFFFF; font-weight: bold;'
            
            def color_all_columns(val):
                # Force white color for None, NA, and other non-numeric values in ALL columns
                if pd.isna(val) or val == 'None' or val == 'NA' or val == '':
                    return 'color: #FFFFFF; font-weight: bold;'
                if isinstance(val, (int, float)):
                    return ''  # Let default styling handle numeric values
                return 'color: #FFFFFF; font-weight: bold;'  # Force white for any other text
            
            def highlight_metrics_rows(s):
                if s.name[1] == 'CASH':
                    return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                unique_dates = list(metrics_df_display.index.get_level_values('Date').unique())
                is_even = unique_dates.index(s.name[0]) % 2 == 0
                bg_color = 'background-color: #0e1117' if is_even else 'background-color: #262626'
                return [f'{bg_color}; color: white;'] * len(s)

            styler_metrics = metrics_df_display.style.apply(highlight_metrics_rows, axis=1)
            if 'Momentum' in metrics_df_display.columns:
                styler_metrics = styler_metrics.map(color_momentum, subset=['Momentum'])
            # Force white color for None/NA values in ALL columns
            styler_metrics = styler_metrics.map(color_all_columns)
            
            fmt_dict = {}
            if 'Momentum' in metrics_df_display.columns:
                fmt_dict['Momentum'] = '{:,.0f}%'
            if 'Beta' in metrics_df_display.columns:
                fmt_dict['Beta'] = '{:,.2f}'
            if 'Volatility' in metrics_df_display.columns:
                fmt_dict['Volatility'] = '{:,.2f}%'
            if 'Calculated_Weight' in metrics_df_display.columns:
                fmt_dict['Calculated_Weight'] = '{:,.0f}%'
            if fmt_dict:
                styler_metrics = styler_metrics.format(fmt_dict)
            
            # NUCLEAR OPTION: Inject custom CSS to override Streamlit's stubborn styling
            st.markdown("""
            <style>
            /* NUCLEAR CSS OVERRIDE - BEAT STREAMLIT INTO SUBMISSION */
            .stDataFrame [data-testid="stDataFrame"] div[data-testid="stDataFrame"] table td,
            .stDataFrame [data-testid="stDataFrame"] div[data-testid="stDataFrame"] table th,
            .stDataFrame table td,
            .stDataFrame table th {
                color: #FFFFFF !important;
                font-weight: bold !important;
                text-shadow: 1px 1px 2px black !important;
            }
            
            /* Force ALL text in dataframes to be white */
            .stDataFrame * {
                color: #FFFFFF !important;
            }
            
            /* Override any Streamlit bullshit */
            .stDataFrame [data-testid="stDataFrame"] *,
            .stDataFrame div[data-testid="stDataFrame"] * {
                color: #FFFFFF !important;
                font-weight: bold !important;
            }
            
            /* Target EVERYTHING in the dataframe */
            .stDataFrame table *,
            .stDataFrame div table *,
            .stDataFrame [data-testid="stDataFrame"] table *,
            .stDataFrame [data-testid="stDataFrame"] div table * {
                color: #FFFFFF !important;
                font-weight: bold !important;
            }
            
            /* Target all cells specifically */
            .stDataFrame td,
            .stDataFrame th,
            .stDataFrame table td,
            .stDataFrame table th {
                color: #FFFFFF !important;
                font-weight: bold !important;
            }
            
            /* Target Streamlit's specific elements */
            div[data-testid="stDataFrame"] table td,
            div[data-testid="stDataFrame"] table th,
            div[data-testid="stDataFrame"] div table td,
            div[data-testid="stDataFrame"] div table th {
                color: #FFFFFF !important;
                font-weight: bold !important;
            }
            
            /* Target everything with maximum specificity */
            div[data-testid="stDataFrame"] *,
            div[data-testid="stDataFrame"] div *,
            div[data-testid="stDataFrame"] table *,
            div[data-testid="stDataFrame"] div table * {
                color: #FFFFFF !important;
                font-weight: bold !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(styler_metrics, )

    # Allocation pie charts (last rebalance vs current)
    if allocs_for_portfolio:
        try:
            alloc_dates = sorted(list(allocs_for_portfolio.keys()))
            final_date = alloc_dates[-1]
            last_rebal_date = alloc_dates[-2] if len(alloc_dates) > 1 else alloc_dates[-1]
            final_alloc = allocs_for_portfolio.get(final_date, {})
            rebal_alloc = allocs_for_portfolio.get(last_rebal_date, {})

            def prepare_bar_data(d):
                labels = []
                values = []
                for k, v in sorted(d.items(), key=lambda x: (-x[1], x[0])):
                    try:
                        val = float(v) * 100
                        if val > 0:  # Only include tickers with allocation > 0%
                            labels.append(k)
                            values.append(val)
                    except Exception:
                        pass  # Skip invalid values
                return labels, values

            labels_final, vals_final = prepare_bar_data(final_alloc)
            labels_rebal, vals_rebal = prepare_bar_data(rebal_alloc)
            # prepare helpers used by the 'Rebalance Today' UI
            # Prefer the snapshot saved when backtests were run so this UI is static until rerun
            snapshot = st.session_state.get('alloc_snapshot_data', {})
            snapshot_raw = snapshot.get('raw_data')
            snapshot_portfolios = snapshot.get('portfolio_configs')

            # select raw_data and portfolio config from snapshot if available, otherwise fall back to live state
            raw_data = snapshot_raw if snapshot_raw is not None else st.session_state.get('alloc_raw_data', {})
            # find snapshot portfolio config by name if present
            snapshot_cfg = None
            if snapshot_portfolios:
                try:
                    snapshot_cfg = next((c for c in snapshot_portfolios if c.get('name') == active_name), None)
                except Exception:
                    snapshot_cfg = None
            portfolio_cfg_for_today = snapshot_cfg if snapshot_cfg is not None else active_portfolio

            try:
                portfolio_value = float(portfolio_cfg_for_today.get('initial_value', 0) or 0)
            except Exception:
                portfolio_value = portfolio_cfg_for_today.get('initial_value', 0) or 0
            


            def _price_on_or_before(df, target_date):
                try:
                    idx = df.index[df.index <= pd.to_datetime(target_date)]
                    if len(idx) == 0:
                        return None
                    return float(df.loc[idx[-1], 'Close'])
                except Exception:
                    return None

            def build_table_from_alloc(alloc_dict, price_date, label):
                rows = []
                for tk in sorted(alloc_dict.keys()):
                    alloc_pct = float(alloc_dict.get(tk, 0))
                    if tk == 'CASH':
                        price = None
                        shares = 0
                        total_val = portfolio_value * alloc_pct
                    else:
                        df = raw_data.get(tk)
                        price = None
                        if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                            if price_date is None:
                                # use latest price
                                try:
                                    price = float(df['Close'].iloc[-1])
                                except Exception:
                                    price = None
                            else:
                                price = _price_on_or_before(df, price_date)
                        try:
                            if price and price > 0:
                                allocation_value = portfolio_value * alloc_pct
                                # allow fractional shares shown to 1 decimal place
                                shares = round(allocation_value / price, 1)
                                total_val = shares * price
                            else:
                                shares = 0.0
                                total_val = portfolio_value * alloc_pct
                        except Exception:
                            shares = 0
                            total_val = portfolio_value * alloc_pct

                    pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                    rows.append({
                        'Ticker': tk,
                        'Allocation %': alloc_pct * 100,
                        'Price ($)': price if price is not None else float('nan'),
                        'Shares': shares,
                        'Total Value ($)': total_val,
                        '% of Portfolio': pct_of_port,
                    })

                df_table = pd.DataFrame(rows).set_index('Ticker')
                # Decide whether to show CASH row: hide if Total Value is zero or Shares zero/NaN
                df_display = df_table.copy()
                show_cash = False
                if 'CASH' in df_display.index:
                    cash_val = None
                    if 'Total Value ($)' in df_display.columns:
                        cash_val = df_display.at['CASH', 'Total Value ($)']
                    elif 'Shares' in df_display.columns:
                        cash_val = df_display.at['CASH', 'Shares']
                    try:
                        show_cash = bool(cash_val and not pd.isna(cash_val) and cash_val != 0)
                    except Exception:
                        show_cash = False
                    if not show_cash:
                        df_display = df_display.drop('CASH')
                
                # Add total row
                total_alloc_pct = df_display['Allocation %'].sum()
                total_value = df_display['Total Value ($)'].sum()
                total_port_pct = df_display['% of Portfolio'].sum()
                
                total_row = pd.DataFrame({
                    'Allocation %': [total_alloc_pct],
                    'Price ($)': [float('nan')],
                    'Shares': [float('nan')],
                    'Total Value ($)': [total_value],
                    '% of Portfolio': [total_port_pct]
                }, index=['TOTAL'])
                
                df_display = pd.concat([df_display, total_row])

                # formatting for display
                fmt = {
                    'Allocation %': '{:,.1f}%',
                    'Price ($)': '${:,.2f}',
                    'Shares': '{:,.1f}',
                    'Total Value ($)': '${:,.2f}',
                    '% of Portfolio': '{:,.2f}%'
                }
                try:
                    st.markdown(f"**{label}**")
                    sty = df_display.style.format(fmt)
                    if 'CASH' in df_table.index and show_cash:
                        def _highlight_cash_row(s):
                            if s.name == 'CASH':
                                return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                            return [''] * len(s)
                        sty = sty.apply(_highlight_cash_row, axis=1)
                    
                    # Highlight TOTAL row
                    def _highlight_total_row(s):
                        if s.name == 'TOTAL':
                            return ['background-color: #1f4e79; color: white; font-weight: bold;' for _ in s]
                        return [''] * len(s)
                    sty = sty.apply(_highlight_total_row, axis=1)
                    
                    st.dataframe(sty, )
                except Exception:
                    st.dataframe(df_display, )
            
            # Add Returns Table BEFORE the pie charts
            st.markdown("### 📈 **Returns Summary**")
            
            def calculate_returns_table():
                """Calculate returns for different periods"""
                try:
                    # Get raw data
                    snapshot = st.session_state.get('alloc_snapshot_data', {})
                    raw_data = snapshot.get('raw_data') if snapshot and snapshot.get('raw_data') is not None else st.session_state.get('alloc_raw_data', {})
                    
                    if not raw_data:
                        return None
                    
                    today = pd.Timestamp.now().date()
                    returns_data = []
                    
                    # Get all tickers from current allocation
                    current_tickers = list(final_alloc.keys())
                    
                    for ticker in current_tickers:
                        if ticker in raw_data and not raw_data[ticker].empty:
                            df = raw_data[ticker].copy()
                            if 'Close' not in df.columns:
                                continue
                            
                            # Calculate different period returns using calendar-day lookbacks (same as Benchmark Comparison)
                            periods = {
                                '1W': 7,
                                '1M': 30,
                                '3M': 90,
                                '6M': 180,
                                '1Y': 365
                            }
                            
                            def _get_value_days_ago(series, days):
                                """Return value at or before last_date - days from a datetime-indexed Series."""
                                if series is None or len(series) == 0:
                                    return None
                                last_date = pd.to_datetime(series.index[-1])
                                target_date = last_date - pd.Timedelta(days=days)
                                # Ensure datetime index
                                series.index = pd.to_datetime(series.index)
                                prior = series.loc[:target_date]
                                if len(prior) == 0:
                                    return series.iloc[0]
                                return prior.iloc[-1]
                            
                            current_price = df['Close'].iloc[-1]
                            ticker_returns = {'Ticker': ticker}
                            
                            # Ensure datetime index
                            df.index = pd.to_datetime(df.index)
                            for period_name, days in periods.items():
                                try:
                                    past_val = _get_value_days_ago(df['Close'], days)
                                    if past_val is not None and past_val > 0:
                                        return_pct = ((current_price - past_val) / past_val) * 100
                                        ticker_returns[period_name] = f"{return_pct:+.2f}%"
                                    else:
                                        ticker_returns[period_name] = 'N/A'
                                except Exception:
                                    ticker_returns[period_name] = 'N/A'
                            
                            returns_data.append(ticker_returns)
                    
                    if returns_data:
                        df_returns = pd.DataFrame(returns_data)
                        # Sort by ticker name
                        df_returns = df_returns.sort_values('Ticker').reset_index(drop=True)
                        
                        # Add weighted portfolio return row - use same backtest data as Benchmark Comparison
                        weighted_row = {'Ticker': 'PORTFOLIO HISTORICAL'}
                        
                        # Use backtest results directly for portfolio returns (same as Benchmark Comparison)
                        try:
                            active_name = active_portfolio.get('name') if active_portfolio else None
                            all_results = st.session_state.get('alloc_all_results', {})
                            
                            if active_name in all_results:
                                portfolio_result = all_results[active_name]
                                if 'no_additions' in portfolio_result:
                                    portfolio_values = portfolio_result['no_additions']
                                    
                                    for period_name, days in periods.items():
                                        try:
                                            # Ensure we have enough data points
                                            if len(portfolio_values) < days + 1:
                                                weighted_row[period_name] = 'N/A'
                                                continue
                                            
                                            # Get current and past values safely
                                            current_value = portfolio_values.iloc[-1]
                                            past_value = portfolio_values.iloc[-(days + 1)]
                                            
                                            if past_value > 0:
                                                return_pct = ((current_value - past_value) / past_value) * 100
                                                weighted_row[period_name] = f"{return_pct:+.2f}%"
                                            else:
                                                weighted_row[period_name] = 'N/A'
                                                
                                        except (IndexError, KeyError):
                                            weighted_row[period_name] = 'N/A'
                                else:
                                    # Fallback to weighted calculation if no backtest data
                                    for period_name, days in periods.items():
                                        try:
                                            weighted_return = 0.0
                                            valid_weights = 0.0
                                            
                                            for _, row in df_returns.iterrows():
                                                ticker = row['Ticker']
                                                return_str = row[period_name]
                                                
                                                if return_str != 'N/A' and ticker in final_alloc:
                                                    try:
                                                        # Parse return percentage
                                                        return_pct = float(return_str.replace('%', '').replace('+', ''))
                                                        # Get allocation weight
                                                        weight = final_alloc[ticker]
                                                        # Add weighted return
                                                        weighted_return += return_pct * weight
                                                        valid_weights += weight
                                                    except (ValueError, KeyError):
                                                        continue
                                            
                                            if valid_weights > 0:
                                                # Normalize by actual weights used
                                                final_weighted_return = weighted_return / valid_weights
                                                weighted_row[period_name] = f"{final_weighted_return:+.2f}%"
                                            else:
                                                weighted_row[period_name] = 'N/A'
                                                
                                        except Exception:
                                            weighted_row[period_name] = 'N/A'
                            else:
                                # Fallback: all N/A if no portfolio data
                                for period_name in periods.keys():
                                    weighted_row[period_name] = 'N/A'
                        except Exception as e:
                            print(f"[RETURNS SUMMARY DEBUG] Error getting backtest results: {e}")
                            # Fallback to weighted calculation
                            for period_name, days in periods.items():
                                try:
                                    weighted_return = 0.0
                                    valid_weights = 0.0
                                    
                                    for _, row in df_returns.iterrows():
                                        ticker = row['Ticker']
                                        return_str = row[period_name]
                                        
                                        if return_str != 'N/A' and ticker in final_alloc:
                                            try:
                                                # Parse return percentage
                                                return_pct = float(return_str.replace('%', '').replace('+', ''))
                                                # Get allocation weight
                                                weight = final_alloc[ticker]
                                                # Add weighted return
                                                weighted_return += return_pct * weight
                                                valid_weights += weight
                                            except (ValueError, KeyError):
                                                continue
                                    
                                    if valid_weights > 0:
                                        # Normalize by actual weights used
                                        final_weighted_return = weighted_return / valid_weights
                                        weighted_row[period_name] = f"{final_weighted_return:+.2f}%"
                                    else:
                                        weighted_row[period_name] = 'N/A'
                                        
                                except Exception:
                                    weighted_row[period_name] = 'N/A'
                        
                        # Add weighted row at the end
                        df_returns = pd.concat([df_returns, pd.DataFrame([weighted_row])], ignore_index=True)
                        return df_returns
                    
                except Exception as e:
                    print(f"[RETURNS DEBUG] Error calculating returns: {e}")
                    return None
                
                return None
            
            returns_df = calculate_returns_table()
            if returns_df is not None and not returns_df.empty:
                # Style the dataframe
                def style_returns(val):
                    if isinstance(val, str) and val.endswith('%'):
                        try:
                            num_val = float(val.replace('%', '').replace('+', ''))
                            if num_val > 0:
                                return 'color: #00ff00; font-weight: bold'  # Green for positive
                            elif num_val < 0:
                                return 'color: #ff4444; font-weight: bold'  # Red for negative
                        except:
                            pass
                    return ''
                
                # Apply styling
                styled_returns = returns_df.style.applymap(style_returns)
                
                # Highlight the PORTFOLIO row
                def highlight_portfolio_row(row):
                    if row['Ticker'] == 'PORTFOLIO':
                        return ['background-color: #333333; font-weight: bold; border: 2px solid #ffff00' for _ in row]
                    return ['' for _ in row]
                
                # Apply row highlighting
                styled_returns = styled_returns.apply(highlight_portfolio_row, axis=1)
                
                # Display the table
                st.dataframe(styled_returns, )
            else:
                st.info("Returns data not available. Please run a backtest first.")
            
            st.markdown("---")
            
            # Render small pies for Last Rebalance and Current Allocation
            try:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Target Allocation at Last Rebalance ({last_rebal_date.date()})**")
                    fig_rebal_small = go.Figure(data=[go.Pie(
                        labels=labels_rebal,
                        values=vals_rebal,
                        hole=0.35
                    )])
                    fig_rebal_small.update_traces(textinfo='percent+label')
                    fig_rebal_small.update_layout(template='plotly_dark', margin=dict(t=10))
                    st.plotly_chart(fig_rebal_small, key=f"alloc_rebal_small_{active_name}")
                with col2:
                    st.markdown(f"**Portfolio Evolution (Current Allocation)**")
                    fig_today_small = go.Figure(data=[go.Pie(
                        labels=labels_final,
                        values=vals_final,
                        hole=0.35
                    )])
                    fig_today_small.update_traces(textinfo='percent+label')
                    fig_today_small.update_layout(template='plotly_dark', margin=dict(t=10))
                    st.plotly_chart(fig_today_small, key=f"alloc_today_small_{active_name}")
            except Exception:
                # If plotting fails, continue and still render the tables below
                pass

            # Last rebalance table (use last_rebal_date)
            build_table_from_alloc(rebal_alloc, last_rebal_date, f"Target Allocation at Last Rebalance ({last_rebal_date.date()})")
            # Current / Today table (use final_date's latest available prices as of now)
            build_table_from_alloc(final_alloc, None, f"Portfolio Evolution (Current Allocation)")
        except Exception as e:
            print(f"[ALLOC PLOT DEBUG] Failed to render allocation plots for {active_name}: {e}")

    # Add PDF generation button at the very end
    st.markdown("---")
    st.markdown("### 📄 Generate PDF Report")
    
    # Optional custom PDF report name
    custom_report_name = st.text_input(
        "📝 Custom Report Name (optional):", 
        value="",
        placeholder="e.g., Portfolio Allocation Analysis, Asset Distribution Q4, Sector Breakdown Study",
        help="Leave empty to use automatic naming: 'Allocations_Report_[timestamp].pdf'",
        key="allocations_custom_report_name"
    )
    
    if st.button("Generate PDF Report", type="primary", key="alloc_pdf_btn_2", use_container_width=True):
        try:
            success = generate_allocations_pdf(custom_report_name)
            if success:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Generate filename based on custom name or default
                if custom_report_name.strip():
                    clean_name = custom_report_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
                    filename = f"{clean_name}_{timestamp}.pdf"
                else:
                    filename = f"Allocations_Report_{timestamp}.pdf"
                
                st.success("✅ PDF Report Generated Successfully!")
                st.download_button(
                    label="📥 Download PDF Report",
                    data=st.session_state.get('pdf_buffer', b''),
                    file_name=filename,
                    mime="application/pdf",
                )
            else:
                st.error("❌ Failed to generate PDF report")
        except Exception as e:
            st.error(f"❌ Error generating PDF: {str(e)}")
            st.exception(e)
