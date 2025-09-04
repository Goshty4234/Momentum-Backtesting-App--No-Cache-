import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
import contextlib
from datetime import datetime, timedelta, date
import warnings
import os
import plotly.io as pio
warnings.filterwarnings('ignore')

# =============================================================================
# PERFORMANCE OPTIMIZATION: NO CACHING VERSION
# =============================================================================

def get_ticker_data(ticker_symbol, period="max", auto_adjust=False):
    """Get ticker data without caching (NO_CACHE version)
    
    Args:
        ticker_symbol: Stock ticker symbol
        period: Data period
        auto_adjust: Auto-adjust setting
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=period, auto_adjust=auto_adjust)[["Close", "Dividends"]]
        return hist
    except Exception:
        return pd.DataFrame()

# Matplotlib configuration for high-quality PDF generation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# PDF Generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors as reportlab_colors

def plotly_to_matplotlib_figure(plotly_fig, title="", width_inches=8, height_inches=6):
    """
    Convert a Plotly figure to a matplotlib figure for PDF generation
    """
    try:
        # Extract data from Plotly figure
        fig_data = plotly_fig.data
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        
        # Set title with wrapping for long titles
        if title:
            # Use textwrap for proper word-based wrapping
            import textwrap
            wrapped_title = textwrap.fill(title, width=40, break_long_words=True, break_on_hyphens=False)
            ax.set_title(wrapped_title, fontsize=14, fontweight='bold', pad=20)
        
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
        # Don't add legend here - it will be added separately below the plot
        # Extract legend information for separate placement
        if ax.get_legend_handles_labels()[0]:
            handles, labels = ax.get_legend_handles_labels()
            colors = [handle.get_color() if hasattr(handle, 'get_color') else 'black' for handle in handles]
            fig.legend_info = [{'label': label, 'color': color} for label, color in zip(labels, colors)]
            # Remove legend from plot since we'll add it separately
            ax.legend().remove()
        
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
                ha='center', va='center', transform=ax.transAxes)
        return fig


def create_legend_figure(legend_info, title="Legend", width_inches=10, height_inches=2):
    """
    Create a separate matplotlib figure for the legend to be placed below the main plot
    """
    try:
        if not legend_info:
            return None
        
        # Process labels to handle long portfolio names
        processed_labels = []
        max_label_length = 0
        
        for item in legend_info:
            label = item['label']
            # If label is very long, wrap it intelligently
            if len(label) > 40:
                # Split long labels at spaces or special characters
                words = label.split()
                if len(words) > 3:
                    # For very long names, create multiple lines
                    if len(words) <= 6:
                        # Split in the middle
                        mid = len(words) // 2
                        wrapped_label = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                    else:
                        # Split into thirds for extremely long names
                        third = len(words) // 3
                        wrapped_label = '\n'.join([
                            ' '.join(words[:third]),
                            ' '.join(words[third:2*third]),
                            ' '.join(words[2*third:])
                        ])
                else:
                    wrapped_label = label
            else:
                wrapped_label = label
            
            processed_labels.append(wrapped_label)
            # Calculate max height needed for wrapped labels
            lines = wrapped_label.count('\n') + 1
            max_label_length = max(max_label_length, lines)
        
        # Calculate dynamic height based on number of items and label complexity
        num_items = len(legend_info)
        base_height = 2.5  # Increased from 2.0 to 2.5 for larger text
        height_per_item = max(0.8, max_label_length * 0.25)  # Increased from 0.6/0.2 to 0.8/0.25 for larger text
        legend_height = max(base_height, min(6.0, num_items * height_per_item))  # Increased max from 5.0 to 6.0
        
        fig, ax = plt.subplots(figsize=(width_inches, legend_height))
        ax.axis('off')
        
        # Create legend handles
        handles = []
        
        for item in legend_info:
            # Create a line handle for the legend
            line = plt.Line2D([], [], color=item['color'], linewidth=3)
            handles.append(line)
        
        # Determine optimal number of columns based on content
        if num_items <= 3:
            ncol = num_items
        elif num_items <= 6:
            ncol = min(3, num_items)
        else:
            ncol = min(4, num_items)  # Max 4 columns for many items
        
        # Create legend with specific positioning and better text handling
        legend = ax.legend(handles, processed_labels, 
                           loc='center',
                           ncol=ncol,
                           frameon=True,
                           fancybox=True,
                           shadow=True,
                           fontsize=16,  # Increased from 12 to 16 for better readability
                           columnspacing=2.0,
                           labelspacing=1.2)  # Increased from 1.0 to 1.2 for better spacing
        
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=18, fontweight='bold', pad=15)  # Increased from 14 to 18
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(width_inches, 1))
        ax.text(0.5, 0.5, f'Error creating legend: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig


def create_paginated_legends(legend_info, title="Legend", width_inches=10, max_items_per_page=25):
    """
    Create multiple legend figures if the legend is very long, splitting across pages
    """
    if not legend_info:
        return []
    
    legends = []
    total_items = len(legend_info)
    
    # If legend is short enough, create single legend
    if total_items <= max_items_per_page:
        single_legend = create_legend_figure(legend_info, title, width_inches)
        if single_legend:
            legends.append(single_legend)
        return legends
    
    # Split legend into multiple pages
    num_pages = (total_items + max_items_per_page - 1) // max_items_per_page
    
    for page in range(num_pages):
        start_idx = page * max_items_per_page
        end_idx = min((page + 1) * max_items_per_page, total_items)
        
        page_legend_info = legend_info[start_idx:end_idx]
        page_title = f"{title} (Page {page + 1} of {num_pages})"
        
        # Calculate height for this page
        num_items = len(page_legend_info)
        base_height = 2.5  # Increased from 2.0 to 2.5 for larger text
        height_per_item = 0.8  # Increased from 0.6 to 0.8 for larger text
        page_height = max(base_height, min(6.0, num_items * height_per_item))  # Increased max from 5.0 to 6.0
        
        page_legend = create_legend_figure(page_legend_info, page_title, width_inches, page_height)
        if page_legend:
            legends.append(page_legend)
    
    return legends

def create_matplotlib_table(data, headers, title="", width_inches=10, height_inches=4):
    """
    Create a matplotlib table for PDF generation
    """
    try:
        # Ensure data is properly formatted
        if not data or not headers:
            raise ValueError("Data or headers are empty")
        
        # Convert data to strings and ensure proper format
        formatted_data = []
        for row in data:
            formatted_row = []
            for cell in row:
                if cell is None:
                    formatted_row.append('')
                else:
                    formatted_row.append(str(cell))
            formatted_data.append(formatted_row)
        
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=formatted_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternating rows
        for i in range(1, len(formatted_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        # Add title with wrapping for long titles
        if title:
            # Use textwrap for proper word-based wrapping
            import textwrap
            wrapped_title = textwrap.fill(title, width=40, break_long_words=True, break_on_hyphens=False)
            ax.set_title(wrapped_title, fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.text(0.5, 0.5, f'Error creating table: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig

def generate_simple_pdf_report(custom_name=""):
    """
    Generate a simple PDF report with exactly 4 sections using existing Streamlit data
    """
    try:
        # Show progress bar for PDF generation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Add proper PDF metadata
        if custom_name.strip():
            title = f"Multi Backtest Report - {custom_name.strip()}"
            subject = f"Portfolio Analysis Report: {custom_name.strip()}"
        else:
            title = "Multi Backtest Report"
            subject = "Portfolio Analysis and Performance Report"
        
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            title=title,
            author="Portfolio Backtest System",
            subject=subject,
            creator="Multi Backtest Application"
        )
        story = []
        
        # Update progress
        progress_bar.progress(10)
        status_text.text("📄 Initializing PDF document...")
        
        # Get styles
        styles = getSampleStyleSheet()
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=reportlab_colors.Color(0.2, 0.4, 0.6)
        )
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=reportlab_colors.Color(0.3, 0.5, 0.7)
        )
        
        # Update progress
        progress_bar.progress(20)
        status_text.text("📊 Adding portfolio configurations...")
        
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
            main_title = f"Multi Backtest Report - {custom_name.strip()}"
            subtitle = f"Investment Portfolio Analysis: {custom_name.strip()}"
        else:
            main_title = "Multi-Portfolio Backtest Report"
            subtitle = "Comprehensive Investment Portfolio Analysis"
        
        story.append(Paragraph(main_title, title_style))
        story.append(Paragraph(subtitle, subtitle_style))
        
        # Document metadata is set in SimpleDocTemplate creation above
        
        # Report metadata
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated on: {current_time}", styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Get backtest period from data if available
        if 'multi_backtest_snapshot_data' in st.session_state:
            snapshot = st.session_state.multi_backtest_snapshot_data
            raw_data = snapshot.get('raw_data', {})
            if raw_data:
                # Get first and last dates from any available data
                all_dates = []
                for ticker_data in raw_data.values():
                    if isinstance(ticker_data, pd.DataFrame) and 'Close' in ticker_data.columns:
                        all_dates.extend(ticker_data.index.tolist())
                
                if all_dates:
                    start_date = min(all_dates)
                    end_date = max(all_dates)
                    total_days = (end_date - start_date).days
                    
                    story.append(Paragraph(f"Backtest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", styles['Normal']))
                    story.append(Paragraph(f"Total Days Analyzed: {total_days:,}", styles['Normal']))
                    story.append(Spacer(1, 20))
        
        # Table of contents first (correct order)
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
            "Performance Charts & Analysis", 
            "Final Performance Statistics",
            "Portfolio Allocations & Rebalancing Timers"
        ]
        
        for i, point in enumerate(toc_points, 1):
            story.append(Paragraph(f"{i}. {point}", styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Report overview (after TOC, correct order)
        overview_style = ParagraphStyle(
            'Overview',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=reportlab_colors.Color(0.3, 0.5, 0.7)
        )
        
        story.append(Paragraph("Report Overview", overview_style))
        story.append(Paragraph("This report provides comprehensive analysis of investment portfolios, including:", styles['Normal']))
        
        # Overview bullet points (non-personal, clear descriptions)
        overview_points = [
            "Detailed portfolio configurations with all parameters and strategies",
            "Performance analysis with value comparison and drawdown charts",
            "Comprehensive performance statistics and risk metrics",
            "Current allocations and rebalancing countdown timers"
        ]
        
        for point in overview_points:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        
        story.append(PageBreak())
        
        # SECTION 1: Portfolio Configurations & Parameters
        story.append(Paragraph("1. Portfolio Configurations & Parameters", heading_style))
        story.append(Spacer(1, 20))
        
        # Get portfolio configs from session state
        portfolio_configs = st.session_state.get('multi_backtest_portfolio_configs', [])
        
        for i, config in enumerate(portfolio_configs):
            # Add page break for all portfolios except the first one
            if i > 0:
                story.append(PageBreak())
            
            story.append(Paragraph(f"Portfolio: {config.get('name', 'Unknown')}", subheading_style))
            story.append(Spacer(1, 10))
            
            # Create configuration table with all parameters
            config_data = [
                ['Parameter', 'Value', 'Description'],
                ['Initial Value', f"${config.get('initial_value', 0):,.2f}", 'Starting portfolio value'],
                ['Added Amount', f"${config.get('added_amount', 0):,.2f}", 'Regular contribution amount'],
                ['Added Frequency', config.get('added_frequency', 'N/A'), 'How often contributions are made'],
                ['Rebalancing Frequency', config.get('rebalancing_frequency', 'N/A'), 'How often portfolio is rebalanced'],
                ['Benchmark', config.get('benchmark_ticker', 'N/A'), 'Performance comparison index'],
                ['Use Momentum', 'Yes' if config.get('use_momentum', False) else 'No', 'Whether momentum strategy is enabled'],
                ['Momentum Strategy', config.get('momentum_strategy', 'N/A'), 'Type of momentum calculation'],
                ['Negative Momentum Strategy', config.get('negative_momentum_strategy', 'N/A'), 'How to handle negative momentum'],
                ['Use Relative Momentum', 'Yes' if config.get('use_relative_momentum', False) else 'No', 'Whether to use relative momentum'],
                ['Equal if All Negative', 'Yes' if config.get('equal_if_all_negative', False) else 'No', 'Equal weight when all momentum is negative'],
                ['Calculate Beta', 'Yes' if config.get('calc_beta', False) else 'No', 'Include beta in momentum weighting'],
                ['Calculate Volatility', 'Yes' if config.get('calc_volatility', False) else 'No', 'Include volatility in momentum weighting'],
                ['Start Strategy', config.get('start_with', 'N/A'), 'Initial allocation strategy'],
                                 ['First Rebalance Strategy', 
                  "rebalancing date" if config.get('first_rebalance_strategy', 'rebalancing_date') == 'rebalancing_date' else "momentum window complete", 
                  'Initial rebalancing approach'],
                ['Collect Dividends as Cash', 'Yes' if config.get('collect_dividends_as_cash', False) else 'No', 'Dividend handling method'],
                ['Beta Lookback', f"{config.get('beta_window_days', 0)} days", 'Days for beta calculation'],
                ['Beta Exclude', f"{config.get('exclude_days_beta', 0)} days", 'Days excluded from beta calculation'],
                ['Volatility Lookback', f"{config.get('vol_window_days', 0)} days", 'Days for volatility calculation'],
                ['Volatility Exclude', f"{config.get('exclude_days_vol', 0)} days", 'Days excluded from volatility calculation'],
                ['Minimal Threshold', f"{config.get('minimal_threshold_percent', 2.0):.1f}%" if config.get('use_minimal_threshold', False) else 'Disabled', 'Minimum allocation percentage threshold']
            ]
            
            # Add momentum windows if they exist
            momentum_windows = config.get('momentum_windows', [])
            if momentum_windows:
                for i, window in enumerate(momentum_windows, 1):
                    lookback = window.get('lookback', 0)
                    weight = window.get('weight', 0)
                    config_data.append([
                        f'Momentum Window {i}',
                        f"{lookback} days, {weight:.2f}",
                        f"Lookback: {lookback} days, Weight: {weight:.2f}"
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
            
            story.append(config_table)
            story.append(PageBreak())
            # Show ticker allocations table, but hide Allocation % column if momentum is enabled
            if not active_portfolio.get('use_momentum', True):
                story.append(Paragraph("Initial Ticker Allocations (Entered by User):", styles['Heading3']))
                story.append(Paragraph("Note: These are the initial allocations entered by the user, not rebalanced allocations.", styles['Normal']))
                
                # Create full table with Allocation % column for non-momentum strategies
                stocks_data = [['Ticker', 'Allocation %', 'Include Dividends']]
                for stock in config.get('stocks', []):
                    stocks_data.append([
                        stock['ticker'],
                        f"{stock['allocation']*100:.1f}%",
                        "✓" if stock['include_dividends'] else "✗"
                    ])
                
                stocks_table = Table(stocks_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
                stocks_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                    ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                    ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
                ]))
                
                story.append(stocks_table)
                story.append(Spacer(1, 15))
            else:
                story.append(Paragraph("Initial Ticker Allocations:", styles['Heading3']))
                story.append(Paragraph("Note: Momentum strategy is enabled - ticker allocations are calculated dynamically based on momentum scores.", styles['Normal']))
                
                # Create modified table without Allocation % column for momentum strategies
                stocks_data_momentum = [['Ticker', 'Include Dividends']]
                for stock in config.get('stocks', []):
                    stocks_data_momentum.append([
                        stock['ticker'],
                        "✓" if stock['include_dividends'] else "✗"
                    ])
                
                stocks_table_momentum = Table(stocks_data_momentum, colWidths=[2.25*inch, 2.25*inch])
                stocks_table_momentum.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                    ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                    ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98))
                ]))
                
                story.append(stocks_table_momentum)
                story.append(Spacer(1, 15))
        
        # Update progress
        progress_bar.progress(40)
        status_text.text("📈 Adding performance charts...")
        
        # SECTION 2: Portfolio Value and Drawdown Comparison Plots
        story.append(PageBreak())
        story.append(Paragraph("2. Portfolio Value and Drawdown Comparison", heading_style))
        story.append(Spacer(1, 20))
        
        # Get the EXISTING Plotly figures from session state - these are the literal plots from your UI
        if 'fig1' in st.session_state:
            # Convert the existing Plotly figure to matplotlib for PDF
            try:
                fig1 = st.session_state.fig1
                # Convert Plotly figure to matplotlib
                mpl_fig = plotly_to_matplotlib_figure(fig1, title="Portfolio Value Comparison", width_inches=10, height_inches=6)
                
                # Save matplotlib figure to buffer
                img_buffer = io.BytesIO()
                mpl_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                plt.close(mpl_fig)  # Close to free memory
                
                # Add to PDF
                story.append(Image(img_buffer, width=7.5*inch, height=4.5*inch))  # Full page width
                story.append(Spacer(1, 15))
                
                # Add legend below the plot if available
                if hasattr(mpl_fig, 'legend_info') and mpl_fig.legend_info:
                    try:
                        legend_figures = create_paginated_legends(mpl_fig.legend_info, "Portfolio Legend", width_inches=10)
                        for legend_fig in legend_figures:
                            legend_buffer = io.BytesIO()
                            legend_fig.savefig(legend_buffer, format='png', dpi=300, bbox_inches='tight')
                            legend_buffer.seek(0)
                            plt.close(legend_fig)
                            
                            # Add legend to PDF
                            story.append(Image(legend_buffer, width=7.5*inch, height=2*inch))
                            story.append(Spacer(1, 10))
                    except Exception as e:
                        story.append(Paragraph(f"Error creating legend: {str(e)}", styles['Normal']))
            except Exception as e:
                story.append(Paragraph(f"Error converting performance plot: {str(e)}", styles['Normal']))
        
        # Add Max Drawdown plot
        if 'fig2' in st.session_state:
            try:
                fig2 = st.session_state.fig2
                # Convert Plotly figure to matplotlib
                mpl_fig = plotly_to_matplotlib_figure(fig2, title="Portfolio Drawdown Comparison", width_inches=10, height_inches=6)
                
                # Save matplotlib figure to buffer
                img_buffer = io.BytesIO()
                mpl_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                plt.close(mpl_fig)  # Close to free memory
                
                # Add to PDF
                story.append(Image(img_buffer, width=7.5*inch, height=4.5*inch))  # Full page width
                story.append(Spacer(1, 15))
                
                # Add legend below the plot if available
                if hasattr(mpl_fig, 'legend_info') and mpl_fig.legend_info:
                    try:
                        legend_figures = create_paginated_legends(mpl_fig.legend_info, "Portfolio Legend", width_inches=10)
                        for legend_fig in legend_figures:
                            legend_buffer = io.BytesIO()
                            legend_fig.savefig(legend_buffer, format='png', dpi=300, bbox_inches='tight')
                            legend_buffer.seek(0)
                            plt.close(legend_fig)
                            
                            # Add legend to PDF
                            story.append(Image(legend_buffer, width=7.5*inch, height=2*inch))
                            story.append(Spacer(1, 10))
                    except Exception as e:
                        story.append(Paragraph(f"Error creating legend: {str(e)}", styles['Normal']))
            except Exception as e:
                story.append(Paragraph(f"Error converting drawdown plot: {str(e)}", styles['Normal']))
        
        # Update progress
        progress_bar.progress(60)
        status_text.text("📋 Adding performance statistics...")
        
        # SECTION 3: Final Performance Statistics Table
        story.append(PageBreak())
        story.append(Paragraph("3. Final Performance Statistics", heading_style))
        story.append(Spacer(1, 15))
        
        # GUARANTEED statistics table creation - use multiple data sources
        table_created = False
        
        # Method 1: NUKE APPROACH - Extract from fig_stats with proper data handling
        if 'fig_stats' in st.session_state and not table_created:
            try:
                fig_stats = st.session_state.fig_stats
                if hasattr(fig_stats, 'data') and fig_stats.data:
                    for trace in fig_stats.data:
                        if trace.type == 'table':
                            # Get headers
                            if hasattr(trace, 'header') and trace.header and hasattr(trace.header, 'values'):
                                headers = trace.header.values
                            else:
                                headers = ['Portfolio', 'CAGR (%)', 'Max Drawdown (%)', 'Volatility (%)', 'Sharpe Ratio', 'Sortino Ratio']
                            
                            # Get cell data
                            if hasattr(trace, 'cells') and trace.cells and hasattr(trace.cells, 'values'):
                                cell_data = trace.cells.values
                                if cell_data and len(cell_data) > 0:
                                    # Convert to proper table format with header wrapping
                                    num_rows = len(cell_data[0]) if cell_data[0] else 0
                                    table_rows = []
                                    
                                    # Improved header wrapping with better line breaks and dynamic sizing
                                    wrapped_headers = []
                                    common_words = ['Portfolio', 'Volatility', 'Drawdown', 'Sharpe', 'Sortino', 'Ulcer', 'Index', 'Return', 'Value', 'Money', 'Added', 'Contributions']
                                    
                                    for header in headers:
                                        if len(header) > 8:  # More aggressive wrapping for better readability
                                            # Split on spaces and create multi-line header
                                            words = header.split()
                                            if len(words) > 1:
                                                # Smart splitting: try to balance lines
                                                if len(words) == 2:
                                                    wrapped_header = '\n'.join(words)
                                                elif len(words) == 3:
                                                    wrapped_header = '\n'.join([words[0], ' '.join(words[1:])])
                                                elif len(words) == 4:
                                                    wrapped_header = '\n'.join([' '.join(words[:2]), ' '.join(words[2:])])
                                                else:
                                                    # For longer headers, split more aggressively
                                                    mid = len(words) // 2
                                                    wrapped_header = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                            else:
                                                # Single long word - split more aggressively
                                                if header not in common_words and len(header) > 10:
                                                    mid = len(header) // 2
                                                    wrapped_header = header[:mid] + '\n' + header[mid:]
                                                else:
                                                    wrapped_header = header
                                        else:
                                            wrapped_header = header
                                        wrapped_headers.append(wrapped_header)
                                    
                                    for row_idx in range(num_rows):
                                        row = []
                                        for col_idx in range(len(cell_data)):
                                            if col_idx < len(cell_data) and row_idx < len(cell_data[col_idx]):
                                                value = cell_data[col_idx][row_idx]
                                                # Wrap long portfolio names in the first column
                                                if col_idx == 0 and len(str(value)) > 25:
                                                    # Split long portfolio names at spaces with balanced line breaks
                                                    words = str(value).split()
                                                    if len(words) > 5:
                                                        # For very long names, create 2-3 lines maximum
                                                        if len(words) <= 8:
                                                            # 2 lines: split in the middle
                                                            mid = len(words) // 2
                                                            wrapped_value = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                                        else:
                                                            # 3 lines: split into thirds for extremely long names
                                                            third = len(words) // 3
                                                            wrapped_value = '\n'.join([
                                                                ' '.join(words[:third]),
                                                                ' '.join(words[third:2*third]),
                                                                ' '.join(words[2*third:])
                                                            ])
                                                    elif len(words) > 3:
                                                        # Split in the middle for medium names
                                                        mid = len(words) // 2
                                                        wrapped_value = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                                    else:
                                                        wrapped_value = str(value)
                                                else:
                                                    wrapped_value = str(value) if value is not None else ''
                                                row.append(wrapped_value)
                                            else:
                                                row.append('')
                                        table_rows.append(row)
                                    
                                    # Create table with smart column widths for statistics table - WIDER TABLE WITH MONETARY COLUMNS
                                    page_width = 8.2*inch  # Increased from 7.5 to 8.2 inches for maximum width usage
                                    
                                    # Optimized column width distribution for statistics table - WITH WIDER MONETARY COLUMNS
                                    if len(headers) > 8:  # If we have many columns, use optimized widths
                                        # Portfolio column: increased from 1.4 to 2.1 inches for better text wrapping
                                        # Performance metrics get more space for better readability
                                        portfolio_width = 2.1*inch
                                        remaining_width = page_width - portfolio_width
                                        
                                        # Create custom column widths with wider monetary columns
                                        col_widths = [portfolio_width]
                                        for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                            header_lower = header.lower()
                                            # Give extra width to monetary value columns
                                            if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                                col_widths.append(1.6 * (remaining_width / (len(headers) - 1)))  # 60% wider for monetary columns
                                            else:
                                                col_widths.append(remaining_width / (len(headers) - 1))
                                        
                                        # Ensure total width equals page_width
                                        total_allocated = sum(col_widths)
                                        if total_allocated > page_width:
                                            # Scale down proportionally
                                            scale_factor = page_width / total_allocated
                                            col_widths = [w * scale_factor for w in col_widths]
                                            
                                    elif len(headers) > 6:  # Medium number of columns
                                        # Portfolio column: 2.3 inches for medium tables
                                        portfolio_width = 2.3*inch
                                        remaining_width = page_width - portfolio_width
                                        
                                        # Create custom column widths with wider monetary columns
                                        col_widths = [portfolio_width]
                                        for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                            header_lower = header.lower()
                                            # Give extra width to monetary value columns
                                            if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                                col_widths.append(1.7 * (remaining_width / (len(headers) - 1)))  # 70% wider for monetary columns
                                            else:
                                                col_widths.append(remaining_width / (len(headers) - 1))
                                        
                                        # Ensure total width equals page_width
                                        total_allocated = sum(col_widths)
                                        if total_allocated > page_width:
                                            # Scale down proportionally
                                            scale_factor = page_width / total_allocated
                                            col_widths = [w * scale_factor for w in col_widths]
                                            
                                    else:
                                        # Few columns: Portfolio gets 2.0 inches, others share remaining space
                                        portfolio_width = 2.0*inch
                                        remaining_width = page_width - portfolio_width
                                        
                                        # Create custom column widths with wider monetary columns
                                        col_widths = [portfolio_width]
                                        for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                            header_lower = header.lower()
                                            # Give extra width to monetary value columns
                                            if any(word in header_lower for word in ['value', 'portfolio', 'total']):
                                                col_widths.append(1.7 * (remaining_width / (len(headers) - 1)))  # 70% wider for monetary columns
                                            else:
                                                col_widths.append(remaining_width / (len(headers) - 1))
                                        
                                        # Ensure total width equals page_width
                                        total_allocated = sum(col_widths)
                                        if total_allocated > page_width:
                                            # Scale down proportionally
                                            scale_factor = page_width / total_allocated
                                            col_widths = [w * scale_factor for w in col_widths]
                                    
                                    stats_table = Table([wrapped_headers] + table_rows, colWidths=col_widths)
                                    # Dynamic font sizing based on number of columns and header complexity
                                    num_columns = len(headers)
                                    max_header_length = max(len(header) for header in headers)
                                    
                                    # More sophisticated font sizing - SLIGHTLY LARGER FOR BETTER READABILITY
                                    if num_columns > 14:
                                        font_size = 5  # Slightly increased from 4
                                    elif num_columns > 12:
                                        font_size = 6  # Slightly increased from 5
                                    elif num_columns > 10:
                                        font_size = 7  # Slightly increased from 6
                                    elif num_columns > 8:
                                        font_size = 8  # Slightly increased from 7
                                    else:
                                        font_size = 9  # Slightly increased from 8
                                    
                                    # Adjust for very long headers - moderate reduction
                                    if max_header_length > 20:
                                        font_size = max(4, font_size - 1)  # Reduce font size moderately
                                    
                                    stats_table.setStyle(TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                        ('FONTSIZE', (0, 0), (-1, 0), font_size),  # Font size for headers
                                        ('FONTSIZE', (0, 1), (-1, -1), font_size + 2),  # Slightly larger font for data rows
                                        ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                                        ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                        ('LEFTPADDING', (0, 0), (-1, -1), 1),  # Reduced padding to maximize table width usage
                                        ('RIGHTPADDING', (0, 0), (-1, -1), 1),  # Reduced padding to maximize table width usage
                                        ('TOPPADDING', (0, 0), (-1, 0), 4),  # Increased padding for header row for better title visibility
                                        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),  # Increased padding for header row for better title visibility
                                        ('TOPPADDING', (0, 1), (-1, -1), 2),  # Padding for data rows
                                        ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
                                        ('WORDWRAP', (0, 0), (-1, -1), True)
                                    ]))
                                story.append(stats_table)
                                story.append(Spacer(1, 15))
                                table_created = True
                                break
            except Exception as e:
                pass
        
        # Method 2: Try to get from snapshot data
        if not table_created and 'multi_backtest_snapshot_data' in st.session_state:
            try:
                snapshot = st.session_state.multi_backtest_snapshot_data
                all_results = snapshot.get('all_results', {})
                
                if all_results:
                    table_data = []
                    headers = ['Portfolio', 'CAGR (%)', 'Max Drawdown (%)', 'Volatility (%)', 'Sharpe Ratio', 'Sortino Ratio']
                    
                    # Wrap long headers to multiple lines - but don't split common words
                    wrapped_headers = []
                    common_words = ['Portfolio', 'Volatility', 'Drawdown', 'Sharpe', 'Sortino', 'Ulcer', 'Index', 'Return', 'Value', 'Money', 'Added', 'Contributions']
                    
                    for header in headers:
                        if len(header) > 12:  # Only wrap very long headers
                            # Split on spaces and create multi-line header
                            words = header.split()
                            if len(words) > 1:
                                # Try to split in the middle, but avoid splitting common words
                                mid = len(words) // 2
                                wrapped_header = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                            else:
                                # Single long word - only split if it's not a common word
                                if header not in common_words:
                                    mid = len(header) // 2
                                    wrapped_header = header[:mid] + '\n' + header[mid:]
                                else:
                                    wrapped_header = header
                        else:
                            wrapped_header = header
                        wrapped_headers.append(wrapped_header)
                    
                    for portfolio_name, result in all_results.items():
                        if isinstance(result, dict) and 'metrics' in result:
                            metrics = result['metrics']
                            # Wrap long portfolio names with balanced line breaks
                            if len(portfolio_name) > 25:
                                words = portfolio_name.split()
                                if len(words) > 5:
                                    # For very long names, create 2-3 lines maximum
                                    if len(words) <= 8:
                                        # 2 lines: split in the middle
                                        mid = len(words) // 2
                                        wrapped_name = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                    else:
                                        # 3 lines: split into thirds for extremely long names
                                        third = len(words) // 3
                                        wrapped_name = '\n'.join([
                                            ' '.join(words[:third]),
                                            ' '.join(words[third:2*third]),
                                            ' '.join(words[2*third:])
                                        ])
                                elif len(words) > 3:
                                    # Split in the middle for medium names
                                    mid = len(words) // 2
                                    wrapped_name = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                else:
                                    wrapped_name = portfolio_name
                            else:
                                wrapped_name = portfolio_name
                            
                            row = [
                                wrapped_name,
                                f"{metrics.get('cagr', 0):.2f}",
                                f"{metrics.get('max_drawdown', 0):.2f}",
                                f"{metrics.get('volatility', 0):.2f}",
                                f"{metrics.get('sharpe_ratio', 0):.2f}",
                                f"{metrics.get('sortino_ratio', 0):.2f}"
                            ]
                            table_data.append(row)
                    
                    if table_data:
                        # Create table with smart formatting for statistics - WIDER TABLE WITH MONETARY COLUMNS
                        page_width = 8.2*inch  # Increased from 7.5 to 8.2 inches for maximum width usage
                        
                        # Optimized column width distribution for statistics table - WITH WIDER MONETARY COLUMNS
                        if len(headers) > 8:  # If we have many columns, use optimized widths
                            # Portfolio column: increased from 1.4 to 2.1 inches for better text wrapping
                            # Performance metrics get more space for better readability
                            portfolio_width = 2.1*inch
                            remaining_width = page_width - portfolio_width
                            
                            # Create custom column widths with wider monetary columns
                            col_widths = [portfolio_width]
                            for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                header_lower = header.lower()
                                # Give extra width to monetary value columns
                                if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                    col_widths.append(1.5 * (remaining_width / (len(headers) - 1)))  # 50% wider for monetary columns
                                else:
                                    col_widths.append(remaining_width / (len(headers) - 1))
                            
                            # Ensure total width equals page_width
                            total_allocated = sum(col_widths)
                            if total_allocated > page_width:
                                # Scale down proportionally
                                scale_factor = page_width / total_allocated
                                col_widths = [w * scale_factor for w in col_widths]
                                
                        elif len(headers) > 6:  # Medium number of columns
                            # Portfolio column: 2.3 inches for medium tables
                            portfolio_width = 2.3*inch
                            remaining_width = page_width - portfolio_width
                            
                            # Create custom column widths with wider monetary columns
                            col_widths = [portfolio_width]
                            for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                header_lower = header.lower()
                                # Give extra width to monetary value columns
                                if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                    col_widths.append(1.6 * (remaining_width / (len(headers) - 1)))  # 60% wider for monetary columns
                                else:
                                    col_widths.append(remaining_width / (len(headers) - 1))
                            
                            # Ensure total width equals page_width
                            total_allocated = sum(col_widths)
                            if total_allocated > page_width:
                                # Scale down proportionally
                                scale_factor = page_width / total_allocated
                                col_widths = [w * scale_factor for w in col_widths]
                                
                        else:
                            # Few columns: Portfolio gets 2.0 inches, others share remaining space
                            portfolio_width = 2.0*inch
                            remaining_width = page_width - portfolio_width
                            
                            # Create custom column widths with wider monetary columns
                            col_widths = [portfolio_width]
                            for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                header_lower = header.lower()
                                # Give extra width to monetary value columns
                                if any(word in header_lower for word in ['value', 'portfolio', 'total']):
                                    col_widths.append(1.7 * (remaining_width / (len(headers) - 1)))  # 70% wider for monetary columns
                                else:
                                    col_widths.append(remaining_width / (len(headers) - 1))
                            
                            # Ensure total width equals page_width
                            total_allocated = sum(col_widths)
                            if total_allocated > page_width:
                                # Scale down proportionally
                                scale_factor = page_width / total_allocated
                                col_widths = [w * scale_factor for w in col_widths]
                        
                        stats_table = Table([wrapped_headers] + table_data, colWidths=col_widths)
                        
                        # Smart font size based on number of columns - SLIGHTLY LARGER FOR BETTER READABILITY
                        font_size = 4 if len(headers) > 14 else 5 if len(headers) > 12 else 6 if len(headers) > 10 else 7 if len(headers) > 8 else 8
                        
                        stats_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), font_size),  # Font size for headers
                                ('FONTSIZE', (0, 1), (-1, -1), font_size + 2),  # Slightly larger font for data rows
                            ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 0),  # Zero padding for maximum space
                            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                            ('TOPPADDING', (0, 0), (-1, 0), 1),  # Minimal padding for header row
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 1),
                            ('TOPPADDING', (0, 1), (-1, -1), 0.5),  # Minimal padding for data rows
                            ('BOTTOMPADDING', (0, 1), (-1, -1), 0.5),
                            ('WORDWRAP', (0, 0), (-1, -1), True)
                        ]))
                        story.append(stats_table)
                        story.append(Spacer(1, 15))
                        table_created = True
            except Exception as e:
                pass
        
        # Method 3: Fallback - create simple table from any available data
        if not table_created:
            try:
                # Try to get any available portfolio data
                available_data = []
                if 'multi_backtest_snapshot_data' in st.session_state:
                    snapshot = st.session_state.multi_backtest_snapshot_data
                    portfolio_configs = snapshot.get('portfolio_configs', [])
                    if portfolio_configs:
                        headers = ['Portfolio', 'Status']
                        for config in portfolio_configs:
                            name = config.get('name', 'Unknown')
                            # Wrap long portfolio names with balanced line breaks for consistency
                            if len(name) > 25:
                                words = name.split()
                                if len(words) > 5:
                                    # For very long names, create 2-3 lines maximum
                                    if len(words) <= 8:
                                        # 2 lines: split in the middle
                                        mid = len(words) // 2
                                        wrapped_name = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                    else:
                                        # 3 lines: split into thirds for extremely long names
                                        third = len(words) // 3
                                        wrapped_name = '\n'.join([
                                            ' '.join(words[:third]),
                                            ' '.join(words[third:2*third]),
                                            ' '.join(words[2*third:])
                                        ])
                                elif len(words) > 3:
                                    # Split in the middle for medium names
                                    mid = len(words) // 2
                                    wrapped_name = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                else:
                                    wrapped_name = name
                            else:
                                wrapped_name = name
                            available_data.append([wrapped_name, 'Data Available'])
                
                if available_data:
                    fallback_table = Table([headers] + available_data, colWidths=[2.2*inch, 2.8*inch])
                    fallback_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                        ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black)
                    ]))
                    story.append(fallback_table)
                    story.append(Spacer(1, 15))
                    table_created = True
            except Exception as e:
                pass
        
        if not table_created:
            story.append(Paragraph("Statistics data not available. Please run the backtest first.", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # SECTION 3.1: Top 5 Best and Worst Performing Portfolios
        # Extract data from the Final Performance Statistics table that was just created
        if table_created and 'fig_stats' in st.session_state:
            try:
                all_results = st.session_state.multi_all_results
                
                fig_stats = st.session_state.fig_stats
                if hasattr(fig_stats, 'data') and fig_stats.data:
                    for trace in fig_stats.data:
                        if trace.type == 'table':
                            # Get headers and data from the existing table
                            if hasattr(trace, 'header') and trace.header and hasattr(trace.header, 'values'):
                                headers = trace.header.values
                            else:
                                headers = ['Portfolio', 'CAGR (%)', 'Max Drawdown (%)', 'Volatility (%)', 'Sharpe Ratio', 'Sortino Ratio']
                            
                            if hasattr(trace, 'cells') and trace.cells and hasattr(trace.cells, 'values'):
                                cell_data = trace.cells.values
                                if cell_data and len(cell_data) > 0 and len(cell_data[0]) > 0:
                                    # Convert to list of rows for easier processing
                                    num_rows = len(cell_data[0])
                                    table_rows = []
                                    for row_idx in range(num_rows):
                                        row = []
                                        for col_idx in range(len(cell_data)):
                                            if col_idx < len(cell_data) and row_idx < len(cell_data[col_idx]):
                                                value = cell_data[col_idx][row_idx]
                                                row.append(str(value) if value is not None else '')
                                            else:
                                                row.append('')
                                        table_rows.append(row)
                                    
                                    if len(table_rows) > 0:
                                        # Sort by Final Portfolio Value (column 1) to get best and worst performers
                                        # Convert Final Portfolio Value to float for sorting (remove $ and commas)
                                        def get_final_value(row):
                                            try:
                                                value_str = str(row[1]).replace('$', '').replace(',', '')
                                                return float(value_str)
                                            except:
                                                return 0.0
                                        
                                        # Sort by Final Portfolio Value descending (best first)
                                        sorted_rows = sorted(table_rows, key=get_final_value, reverse=True)
                                        
                                        # Get top 10 best and worst
                                        num_portfolios = len(sorted_rows)
                                        top_10_best = sorted_rows[:min(10, num_portfolios)]
                                        
                                        # For worst performers, get the actual worst (lowest final values)
                                        # Sort by final value ascending to get worst first, then take top 10
                                        worst_sorted = sorted(table_rows, key=get_final_value, reverse=False)
                                        top_10_worst = worst_sorted[:min(10, num_portfolios)]
                    
                    # Add page break before the new tables
                    story.append(PageBreak())
                    
                    # Top 10 Best Performing Portfolios
                    story.append(Paragraph("3.1. Top 10 Best Performing Portfolios by Final Value", heading_style))
                    story.append(Spacer(1, 10))
                    
                    if len(top_10_best) > 0:
                        # Create table data with headers - EXACT SAME TEXT WRAPPING AS FINAL PERFORMANCE STATISTICS
                        # Wrap headers for better display
                        wrapped_headers = []
                        for header in headers:
                            if len(header) > 15:  # Wrap long headers
                                words = header.split()
                                if len(words) > 1:
                                    if len(words) == 2:
                                        wrapped_header = '\n'.join(words)
                                    elif len(words) == 3:
                                        wrapped_header = '\n'.join([words[0], ' '.join(words[1:])])
                                    elif len(words) == 4:
                                        wrapped_header = '\n'.join([' '.join(words[:2]), ' '.join(words[2:])])
                                    else:
                                        mid = len(words) // 2
                                        wrapped_header = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                else:
                                    if len(header) > 10:
                                        mid = len(header) // 2
                                        wrapped_header = header[:mid] + '\n' + header[mid:]
                                    else:
                                        wrapped_header = header
                            else:
                                wrapped_header = header
                            wrapped_headers.append(wrapped_header)
                        
                        # Wrap portfolio names in the first column for best performers
                        wrapped_best_rows = []
                        for row in top_10_best:
                            wrapped_row = row.copy()
                            if len(str(row[0])) > 25:  # Wrap long portfolio names
                                words = str(row[0]).split()
                                if len(words) > 5:
                                    if len(words) <= 8:
                                        mid = len(words) // 2
                                        wrapped_row[0] = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                    else:
                                        third = len(words) // 3
                                        wrapped_row[0] = '\n'.join([
                                            ' '.join(words[:third]),
                                            ' '.join(words[third:2*third]),
                                            ' '.join(words[2*third:])
                                        ])
                                elif len(words) > 3:
                                    mid = len(words) // 2
                                    wrapped_row[0] = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                            wrapped_best_rows.append(wrapped_row)
                        
                        # Create table data with wrapped headers and rows
                        best_table_data = [wrapped_headers] + wrapped_best_rows
                        
                        # EXACT SAME COLUMN WIDTH LOGIC AS FINAL PERFORMANCE STATISTICS TABLE
                        page_width = 8.2*inch  # Same as Final Performance Statistics
                        
                        # Optimized column width distribution - EXACT SAME LOGIC
                        if len(headers) > 8:  # If we have many columns, use optimized widths
                            portfolio_width = 2.1*inch
                            remaining_width = page_width - portfolio_width
                            
                            col_widths = [portfolio_width]
                            for i, header in enumerate(headers[1:], 1):  # Skip portfolio column
                                header_lower = header.lower()
                                if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                    col_widths.append(1.6 * (remaining_width / (len(headers) - 1)))
                                else:
                                    col_widths.append(remaining_width / (len(headers) - 1))
                            
                            total_allocated = sum(col_widths)
                            if total_allocated > page_width:
                                scale_factor = page_width / total_allocated
                                col_widths = [w * scale_factor for w in col_widths]
                                
                        elif len(headers) > 6:  # Medium number of columns
                            portfolio_width = 2.3*inch
                            remaining_width = page_width - portfolio_width
                            
                            col_widths = [portfolio_width]
                            for i, header in enumerate(headers[1:], 1):
                                header_lower = header.lower()
                                if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                    col_widths.append(1.7 * (remaining_width / (len(headers) - 1)))
                                else:
                                    col_widths.append(remaining_width / (len(headers) - 1))
                            
                            total_allocated = sum(col_widths)
                            if total_allocated > page_width:
                                scale_factor = page_width / total_allocated
                                col_widths = [w * scale_factor for w in col_widths]
                                
                        else:  # Few columns
                            portfolio_width = 2.0*inch
                            remaining_width = page_width - portfolio_width
                            
                            col_widths = [portfolio_width]
                            for i, header in enumerate(headers[1:], 1):
                                header_lower = header.lower()
                                if any(word in header_lower for word in ['value', 'portfolio', 'money', 'total']):
                                    col_widths.append(1.7 * (remaining_width / (len(headers) - 1)))
                                else:
                                    col_widths.append(remaining_width / (len(headers) - 1))
                            
                            total_allocated = sum(col_widths)
                            if total_allocated > page_width:
                                scale_factor = page_width / total_allocated
                                col_widths = [w * scale_factor for w in col_widths]
                        
                        # EXACT SAME TABLE CREATION AND STYLING AS FINAL PERFORMANCE STATISTICS
                        best_table = Table(best_table_data, colWidths=col_widths)
                        
                        # Dynamic font sizing - EXACT SAME LOGIC
                        num_columns = len(headers)
                        max_header_length = max(len(header) for header in headers)
                        
                        if num_columns > 14:
                            font_size = 5
                        elif num_columns > 12:
                            font_size = 6
                        elif num_columns > 10:
                            font_size = 7
                        elif num_columns > 8:
                            font_size = 8
                        else:
                            font_size = 9
                        
                        if max_header_length > 20:
                            font_size = max(4, font_size - 1)
                        
                        best_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.3, 0.5, 0.7)),
                            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), font_size),
                            ('FONTSIZE', (0, 1), (-1, -1), font_size + 2),
                            ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 1),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 1),
                            ('TOPPADDING', (0, 0), (-1, 0), 4),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                            ('TOPPADDING', (0, 1), (-1, -1), 2),
                            ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
                            ('WORDWRAP', (0, 0), (-1, -1), True)
                        ]))
                        story.append(best_table)
                        story.append(Spacer(1, 15))
                    
                    # Top 10 Worst Performing Portfolios
                    story.append(PageBreak())
                    story.append(Paragraph("3.2. Top 10 Worst Performing Portfolios by Final Value", heading_style))
                    story.append(Spacer(1, 10))
                    
                    if len(top_10_worst) > 0:
                        # Create table data with headers - EXACT SAME TEXT WRAPPING AS FINAL PERFORMANCE STATISTICS
                        # Wrap portfolio names in the first column for worst performers
                        wrapped_worst_rows = []
                        for row in top_10_worst:
                            wrapped_row = row.copy()
                            if len(str(row[0])) > 25:  # Wrap long portfolio names
                                words = str(row[0]).split()
                                if len(words) > 5:
                                    if len(words) <= 8:
                                        mid = len(words) // 2
                                        wrapped_row[0] = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                                    else:
                                        third = len(words) // 3
                                        wrapped_row[0] = '\n'.join([
                                            ' '.join(words[:third]),
                                            ' '.join(words[third:2*third]),
                                            ' '.join(words[2*third:])
                                        ])
                                elif len(words) > 3:
                                    mid = len(words) // 2
                                    wrapped_row[0] = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                            wrapped_worst_rows.append(wrapped_row)
                        
                        worst_table_data = [wrapped_headers] + wrapped_worst_rows
                        
                        # EXACT SAME TABLE CREATION AND STYLING AS FINAL PERFORMANCE STATISTICS
                        worst_table = Table(worst_table_data, colWidths=col_widths)
                        worst_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.Color(0.7, 0.3, 0.3)),  # Red header for worst
                            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), font_size),
                            ('FONTSIZE', (0, 1), (-1, -1), font_size + 2),
                            ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
                            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.Color(0.98, 0.98, 0.98)),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 1),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 1),
                            ('TOPPADDING', (0, 0), (-1, 0), 4),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                            ('TOPPADDING', (0, 1), (-1, -1), 2),
                            ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
                            ('WORDWRAP', (0, 0), (-1, -1), True)
                        ]))
                        story.append(worst_table)
                        story.append(Spacer(1, 15))
                    
            except Exception as e:
                story.append(Paragraph(f"Error creating top performers tables: {str(e)}", styles['Normal']))
                story.append(Spacer(1, 15))
        
        # Update progress
        progress_bar.progress(80)
        status_text.text("🎯 Adding allocation charts and timers...")
        
        # SECTION 4: Target Allocation if Rebalanced Today
        story.append(PageBreak())
        current_date_str = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"4. Target Allocation if Rebalanced Today ({current_date_str})", heading_style))
        story.append(Spacer(1, 10))
        
        # Get the allocation data from your existing UI - fetch the existing allocation data
        if 'multi_backtest_snapshot_data' in st.session_state:
            snapshot = st.session_state.multi_backtest_snapshot_data
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
                    story.append(Spacer(1, 10))
                    
                    # Create pie chart for this portfolio (since we need ALL portfolios, not just the selected one)
                    try:
                        # Create labels and values for the plot
                        labels_today = [k for k, v in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
                        vals_today = [float(today_weights[k]) * 100 for k in labels_today]
                        
                        if labels_today and vals_today:
                            # Create matplotlib pie chart (same format as pages 1, 2, 3)
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
                            
                            # Add to PDF - increased pie chart size for better visibility
                            story.append(Image(target_img_buffer, width=5.5*inch, height=5.5*inch))
                            
                            # Add minimal spacing after pie chart before timer section
                            story.append(Spacer(1, 5))
                            
                            # Add Next Rebalance Timer information - simple text display
                            story.append(Paragraph(f"Next Rebalance Timer - {portfolio_name}", subheading_style))
                            story.append(Spacer(1, 5))
                            
                            # Try to get timer information from session state
                            timer_table_key = f"timer_table_{portfolio_name}"
                            if timer_table_key in st.session_state:
                                timer_fig = st.session_state[timer_table_key]
                                if timer_fig and hasattr(timer_fig, 'data') and timer_fig.data:
                                    # Extract timer information from the figure data
                                    for trace in timer_fig.data:
                                        if trace.type == 'table' and hasattr(trace, 'cells') and trace.cells:
                                            cell_values = trace.cells.values
                                            if cell_values and len(cell_values) >= 2:
                                                # Format the timer information
                                                timer_info = []
                                                for i in range(len(cell_values[0])):
                                                    if i < len(cell_values[0]) and i < len(cell_values[1]):
                                                        param = cell_values[0][i]
                                                        value = cell_values[1][i]
                                                        timer_info.append(f"{param}: {value}")
                                                
                                                if timer_info:
                                                    for info in timer_info:
                                                        story.append(Paragraph(info, styles['Normal']))
                                                else:
                                                    story.append(Paragraph("Next rebalance information not available", styles['Normal']))
                                                break
                                    else:
                                        story.append(Paragraph("Next rebalance information not available", styles['Normal']))
                                else:
                                    story.append(Paragraph("Next rebalance information not available", styles['Normal']))
                            else:
                                story.append(Paragraph("Next rebalance information not available", styles['Normal']))
                            
                            # Add page break so Allocation Details starts on a new page
                            story.append(PageBreak())
                            
                            # Now add the allocation table on a new page
                            story.append(Paragraph(f"Allocation Details for {portfolio_name}", subheading_style))
                            story.append(Spacer(1, 3))
                            
                            # NUKE APPROACH: Rebuild allocation table from scratch with correct final portfolio values
                            alloc_table_key = f"alloc_table_{portfolio_name}"
                            table_created = False
                            
                            # First, try to get the FINAL portfolio value from backtest results for PDF generation
                            pdf_portfolio_value = 10000  # Default fallback
                            if 'multi_all_results' in st.session_state and st.session_state.multi_all_results:
                                portfolio_results = st.session_state.multi_all_results.get(portfolio_name)
                                if portfolio_results:
                                    try:
                                        if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                            final_value = portfolio_results['with_additions'].iloc[-1]
                                            if not pd.isna(final_value) and final_value > 0:
                                                pdf_portfolio_value = float(final_value)
                                        elif isinstance(portfolio_results, dict) and 'no_additions' in portfolio_results:
                                            final_value = portfolio_results['no_additions'].iloc[-1]
                                            if not pd.isna(final_value) and final_value > 0:
                                                pdf_portfolio_value = float(final_value)
                                        elif isinstance(portfolio_results, pd.Series):
                                            latest_value = portfolio_results.iloc[-1]
                                            if not pd.isna(latest_value) and latest_value > 0:
                                                pdf_portfolio_value = float(latest_value)
                                    except (IndexError, ValueError, TypeError):
                                        pass  # Keep default value
                            
                            if alloc_table_key in st.session_state:
                                try:
                                    fig_alloc = st.session_state[alloc_table_key]
                                    
                                    # Method 1: Extract from Plotly figure data structure but recalculate with correct final portfolio value
                                    if hasattr(fig_alloc, 'data') and fig_alloc.data:
                                        for trace in fig_alloc.data:
                                            if trace.type == 'table':
                                                # Get headers
                                                if hasattr(trace, 'header') and trace.header and hasattr(trace.header, 'values'):
                                                    headers = trace.header.values
                                                else:
                                                    headers = ['Asset', 'Allocation %', 'Price ($)', 'Shares', 'Total Value ($)', '% of Portfolio']
                                                
                                                # Get cell data and recalculate with correct final portfolio value
                                                if hasattr(trace, 'cells') and trace.cells and hasattr(trace.cells, 'values'):
                                                    cell_data = trace.cells.values
                                                    if cell_data and len(cell_data) > 0:
                                                        # Recalculate table with correct final portfolio value
                                                        table_rows = []
                                                        
                                                        # Get raw data for price calculations
                                                        raw_data = {}
                                                        if 'multi_backtest_snapshot_data' in st.session_state:
                                                            snapshot = st.session_state.multi_backtest_snapshot_data
                                                            raw_data = snapshot.get('raw_data', {})
                                                        
                                                        # Recalculate each row with correct final portfolio value
                                                        for row_idx in range(len(cell_data[0])):
                                                            asset = cell_data[0][row_idx] if row_idx < len(cell_data[0]) else ''
                                                            if asset and asset != 'TOTAL':
                                                                # Get allocation percentage from stored data
                                                                alloc_pct_str = cell_data[1][row_idx] if row_idx < len(cell_data[1]) else '0%'
                                                                alloc_pct = float(alloc_pct_str.rstrip('%')) / 100.0
                                                                
                                                                # Calculate with correct final portfolio value
                                                                allocation_value = pdf_portfolio_value * alloc_pct
                                                                
                                                                # Get current price
                                                                current_price = None
                                                                shares = 0.0
                                                                if asset != 'CASH' and asset in raw_data:
                                                                    df = raw_data[asset]
                                                                    if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                                        try:
                                                                            current_price = float(df['Close'].iloc[-1])
                                                                            if current_price and current_price > 0:
                                                                                shares = round(allocation_value / current_price, 1)
                                                                        except Exception:
                                                                            current_price = None
                                                                
                                                                total_val = shares * current_price if current_price and shares > 0 else allocation_value
                                                                pct_of_port = (total_val / pdf_portfolio_value * 100) if pdf_portfolio_value > 0 else 0
                                                                
                                                                table_rows.append([
                                                                    asset,
                                                                    f"{alloc_pct * 100:.2f}%",
                                                                    f"{current_price:.2f}" if current_price else "N/A",
                                                                    f"{shares:.1f}",
                                                                    f"${total_val:,.2f}",
                                                                    f"{pct_of_port:.2f}%"
                                                                ])
                                                        
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

                                                        # Create table with proper formatting
                                                        page_width = 7.5*inch
                                                        col_widths = [page_width/len(headers)] * len(headers)
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
                                                        table_created = True
                                                        break
                                except Exception as e:
                                    pass
                            
                            # Method 2: Create table from today_weights directly if stored table not available
                            if not table_created:
                                try:
                                    if today_weights:
                                        headers = ['Asset', 'Allocation %']
                                        table_rows = []
                                        
                                        for asset, weight in today_weights.items():
                                            if float(weight) > 0:
                                                table_rows.append([asset, f"{float(weight)*100:.2f}%"])
                                        
                                        if table_rows:
                                            # Calculate total values for summary row
                                            total_alloc_pct = sum(float(row[1].rstrip('%')) for row in table_rows)

                                            # Add total row
                                            total_row = [
                                                'TOTAL',
                                                f"{total_alloc_pct:.2f}%"
                                            ]

                                            page_width = 7.5*inch
                                            col_widths = [page_width/len(headers)] * len(headers)
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
                                            story.append(Spacer(1, 3))
                                            table_created = True
                                        else:
                                            story.append(Paragraph("No allocation data available", styles['Normal']))
                                    else:
                                        story.append(Paragraph("No allocation data available", styles['Normal']))
                                except Exception as e2:
                                    story.append(Paragraph(f"Error creating allocation table: {str(e2)}", styles['Normal']))
                            else:
                                # Fallback: recalculate allocation table with correct final portfolio value
                                try:
                                    # Get raw data for price calculations
                                    raw_data = {}
                                    if 'multi_backtest_snapshot_data' in st.session_state:
                                        snapshot = st.session_state.multi_backtest_snapshot_data
                                        raw_data = snapshot.get('raw_data', {})
                                    
                                    # Create allocation table data with correct final portfolio value
                                    headers = ['Asset', 'Allocation %', 'Price ($)', 'Shares', 'Total Value ($)', '% of Portfolio']
                                    table_rows = []
                                    
                                    for asset, weight in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])):
                                        if float(weight) > 0:
                                            alloc_pct = float(weight) * 100
                                            allocation_value = pdf_portfolio_value * float(weight)
                                            
                                            # Get current price
                                            current_price = None
                                            shares = 0.0
                                            if asset != 'CASH' and asset in raw_data:
                                                df = raw_data[asset]
                                                if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                    try:
                                                        current_price = float(df['Close'].iloc[-1])
                                                        if current_price and current_price > 0:
                                                            shares = round(allocation_value / current_price, 1)
                                                    except Exception:
                                                        current_price = None
                                            
                                            total_val = shares * current_price if current_price and shares > 0 else allocation_value
                                            pct_of_port = (total_val / pdf_portfolio_value * 100) if pdf_portfolio_value > 0 else 0
                                            
                                            table_rows.append([
                                                asset,
                                                f"{alloc_pct:.2f}%",
                                                f"{current_price:.2f}" if current_price else "N/A",
                                                f"{shares:.1f}",
                                                f"${total_val:,.2f}",
                                                f"{pct_of_port:.2f}%"
                                            ])
                                    
                                    # Add total row
                                    total_alloc_pct = sum(float(row[1].rstrip('%')) for row in table_rows)
                                    total_value = sum(float(row[4].replace('$', '').replace(',', '')) for row in table_rows)
                                    total_port_pct = sum(float(row[5].rstrip('%')) for row in table_rows)
                                    
                                    total_row = [
                                        'TOTAL',
                                        f"{total_alloc_pct:.2f}%",
                                        '',
                                        '',
                                        f"${total_value:,.2f}",
                                        f"{total_port_pct:.2f}%"
                                    ]
                                    
                                    # Create table with optimized column widths
                                    page_width = 7.5*inch
                                    col_widths = [1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch]
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
                                except Exception as e3:
                                    # Final fallback: simple text representation
                                    story.append(Paragraph("Target Allocation if Rebalanced Today:", styles['Heading4']))
                                    for asset, weight in today_weights.items():
                                        if float(weight) > 0:
                                            story.append(Paragraph(f"{asset}: {float(weight)*100:.1f}%", styles['Normal']))
                            
                            story.append(Spacer(1, 3))
                        else:
                            story.append(Paragraph(f"No allocation data available for {portfolio_name}", styles['Normal']))
                    except Exception as e:
                        story.append(Paragraph(f"Error creating pie chart for {portfolio_name}: {str(e)}", styles['Normal']))
        else:
            story.append(Paragraph("Allocation data not available. Please run the backtest first.", styles['Normal']))
            story.append(Spacer(1, 5))
        
        # Update progress
        progress_bar.progress(95)
        status_text.text("🔨 Building PDF document...")
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("✅ PDF generation complete! Downloading...")
        
        return buffer
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

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

# Initialize page-specific session state for Multi-Backtest page
if 'multi_backtest_page_initialized' not in st.session_state:
    st.session_state.multi_backtest_page_initialized = True
    # Initialize multi-backtest specific session state
    st.session_state.multi_backtest_portfolio_configs = [
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
            'start_with': 'first',
            'use_momentum': False,
            'use_relative_momentum': False,
            'equal_if_all_negative': False,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [],
            'calc_beta': False,
            'calc_volatility': False,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
            'use_minimal_threshold': False,
            'minimal_threshold_percent': 2.0,
        },
        # 2) Momentum-based portfolio using SPY, QQQ, GLD, TLT
        {
            'name': 'Momentum-Based Portfolio',
            'stocks': [
                {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
            ],
            'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
            'added_amount': 10000,
            'added_frequency': 'Annually',
            'rebalancing_frequency': 'Annually',
            'start_date_user': None,
            'end_date_user': None,
            'start_with': 'first',
            'use_momentum': True,
            'use_relative_momentum': True,
            'equal_if_all_negative': True,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [
                {'lookback': 365, 'exclude': 30, 'weight': 0.5},
                {'lookback': 180, 'exclude': 30, 'weight': 0.3},
                {'lookback': 120, 'exclude': 30, 'weight': 0.2},
            ],
            'calc_beta': True,
            'calc_volatility': True,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
            'use_minimal_threshold': False,
            'minimal_threshold_percent': 2.0,
        },
        # 3) Equal weight (No Momentum) using the same tickers
        {
            'name': 'Equal Weight Portfolio (No Momentum)',
            'stocks': [
                {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
            ],
            'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
            'added_amount': 10000,
            'added_frequency': 'Annually',
            'rebalancing_frequency': 'Annually',
            'start_date_user': None,
            'end_date_user': None,
            'start_with': 'first',
            'use_momentum': False,
            'use_relative_momentum': False,
            'equal_if_all_negative': False,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [],
            'calc_beta': False,
            'calc_volatility': False,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
            'use_minimal_threshold': False,
            'minimal_threshold_percent': 2.0,
        },
    ]
    st.session_state.multi_backtest_active_portfolio_index = 0
    st.session_state.multi_backtest_rerun_flag = False
    # Clean up any existing portfolio configs to remove unused settings
if 'multi_backtest_portfolio_configs' in st.session_state:
    for config in st.session_state.multi_backtest_portfolio_configs:
        config.pop('use_relative_momentum', None)
        config.pop('equal_if_all_negative', None)

st.set_page_config(layout="wide", page_title="Multi-Portfolio Analysis", page_icon="📈")

# Handle imported values from JSON - MUST BE AT THE VERY BEGINNING
if "_import_start_with" in st.session_state:
    st.session_state["multi_backtest_start_with"] = st.session_state.pop("_import_start_with")
    st.session_state["multi_backtest_start_with_radio"] = st.session_state["multi_backtest_start_with"]
if "_import_first_rebalance_strategy" in st.session_state:
    st.session_state["multi_backtest_first_rebalance_strategy"] = st.session_state.pop("_import_first_rebalance_strategy")
    st.session_state["multi_backtest_first_rebalance_strategy_radio"] = st.session_state["multi_backtest_first_rebalance_strategy"]
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

# Handle rerun flag for smooth UI updates - must be at the very top
if st.session_state.get('multi_backtest_rerun_flag', False):
    st.session_state.multi_backtest_rerun_flag = False
    st.rerun()

# Place rerun logic after first portfolio input widget
active_portfolio = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index] if 'multi_backtest_portfolio_configs' in st.session_state and 'multi_backtest_active_portfolio_index' in st.session_state else None

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
# Backtest_Engine.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io
import contextlib
import json
from datetime import datetime, timedelta, time
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

st.set_page_config(layout="wide", page_title="Multi-Portfolio Analysis")

st.title("Multi-Portfolio Backtest")
st.markdown("Use the forms below to configure and run backtests for multiple portfolios.")

# Portfolio name is handled in the main UI below

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
        'start_with': 'first',
    'use_momentum': False,
        'use_relative_momentum': False,
        'equal_if_all_negative': False,
        'momentum_windows': [],
        'calc_beta': False,
        'calc_volatility': False,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
    },
    # 2) Momentum-based portfolio using SPY, QQQ, GLD, TLT
    {
        'name': 'Momentum-Based Portfolio',
        'stocks': [
            {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
        ],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 10000,
        'added_frequency': 'Annually',
        'rebalancing_frequency': 'Annually',
        'start_date_user': None,
        'end_date_user': None,
        'start_with': 'first',
        'use_momentum': True,
        'momentum_strategy': 'Classic',
        'negative_momentum_strategy': 'Cash',
        'momentum_windows': [
            {'lookback': 365, 'exclude': 30, 'weight': 0.5},
            {'lookback': 180, 'exclude': 30, 'weight': 0.3},
            {'lookback': 120, 'exclude': 30, 'weight': 0.2},
        ],
        'calc_beta': True,
        'calc_volatility': True,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
    },
    # 3) Equal weight (No Momentum) using the same tickers
    {
        'name': 'Equal Weight Portfolio (No Momentum)',
        'stocks': [
            {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
        ],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 10000,
        'added_frequency': 'Annually',
        'rebalancing_frequency': 'Annually',
        'start_date_user': None,
        'end_date_user': None,
        'start_with': 'first',
        'use_momentum': False,
        'momentum_windows': [],
        'calc_beta': False,
        'calc_volatility': False,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
    },
]

# -----------------------
# Helper functions
# -----------------------
def get_trading_days(start_date, end_date):
    return pd.bdate_range(start=start_date, end=end_date)

def get_dates_by_freq(freq, start, end, market_days):
    market_days = sorted(market_days)
    if freq == "market_day":
        return set(market_days)
    elif freq == "calendar_day":
        return set(pd.date_range(start=start, end=end, freq='D'))
    elif freq == "Weekly":
        base = pd.date_range(start=start, end=end, freq='W-MON')
    elif freq == "Biweekly":
        base = pd.date_range(start=start, end=end, freq='2W-MON')
    elif freq == "Monthly":
        base = pd.date_range(start=start, end=end, freq='MS')
    elif freq == "Quarterly":
        base = pd.date_range(start=start, end=end, freq='3MS')
    elif freq == "Semiannually":
        # First day of Jan and Jul each year
        semi = []
        for y in range(start.year, end.year + 1):
            for m in [1, 7]:
                semi.append(pd.Timestamp(year=y, month=m, day=1))
        base = pd.DatetimeIndex(semi)
    elif freq == "Annually" or freq == "year":
        base = pd.date_range(start=start, end=end, freq='YS')
    elif freq == "Never" or freq == "none" or freq is None:
        return set()
    elif freq == "Buy & Hold" or freq == "Buy & Hold (Target)":
        # Buy & Hold options don't have specific rebalancing dates - they rebalance immediately when cash is available
        return set()
    else:
        raise ValueError(f"Unknown frequency: {freq}")

    dates = []
    for d in base:
        idx = np.searchsorted(market_days, d, side='right')
        if idx > 0 and market_days[idx-1] >= d:
            dates.append(market_days[idx-1])
        elif idx < len(market_days):
            dates.append(market_days[idx])
    return set(dates)

def calculate_cagr(values, dates):
    if len(values) < 2:
        return np.nan
    start_val = values[0]
    end_val = values[-1]
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0 or start_val <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1

def calculate_max_drawdown(values):
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / np.where(peak == 0, 1, peak)
    return np.nanmin(drawdowns), drawdowns

def calculate_volatility(returns):
    # Annualized volatility - same as Backtest_Engine.py
    return returns.std() * np.sqrt(365) if len(returns) > 1 else np.nan

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

# FIXED: Correct Sortino Ratio calculation - EXACTLY like Backtest_Engine.py
def calculate_sortino(returns, risk_free_rate):
    """Calculates the Sortino ratio."""
    # Create a constant risk-free rate series aligned with returns
    daily_rf_rate = risk_free_rate / 252
    rf_series = pd.Series(daily_rf_rate, index=returns.index)
    
    aligned_returns, aligned_rf = returns.align(rf_series, join='inner')
    if aligned_returns.empty:
        return np.nan
        
    downside_returns = aligned_returns[aligned_returns < aligned_rf]
    if downside_returns.empty or downside_returns.std() == 0:
        # If no downside returns, Sortino is infinite or undefined.
        # We can return nan or a very high value. nan is safer.
        return np.nan
    
    downside_std = downside_returns.std()
    
    return (aligned_returns.mean() - aligned_rf.mean()) / downside_std * np.sqrt(365)

# -----------------------
# Timer function for next rebalance date
# -----------------------
def calculate_next_rebalance_date(rebalancing_frequency, last_rebalance_date):
    """
    Calculate the next rebalance date based on rebalancing frequency and last rebalance date.
    Excludes today and yesterday as mentioned in the requirements.
    """
    if rebalancing_frequency == 'none':
        return None, None, None
    
    # Convert to datetime if it's a pandas Timestamp
    if hasattr(last_rebalance_date, 'to_pydatetime'):
        last_rebalance_date = last_rebalance_date.to_pydatetime()
    
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    # If no last rebalance date, use yesterday as base
    if not last_rebalance_date:
        base_date = yesterday
    else:
        # If last rebalance was today or yesterday, use the day before yesterday as base
        if last_rebalance_date.date() >= yesterday:
            base_date = yesterday - timedelta(days=1)
        else:
            base_date = last_rebalance_date.date()
    
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
        # Add one month - handle month overflow safely
        try:
            if base_date.month == 12:
                next_date = base_date.replace(year=base_date.year + 1, month=1)
            else:
                next_date = base_date.replace(month=base_date.month + 1)
        except ValueError:
            # Handle invalid day for target month (e.g., day 31 in February)
            next_date = base_date.replace(month=base_date.month + 1, day=1)
            # Try to find a valid day in the target month
            while True:
                try:
                    next_date = next_date.replace(day=base_date.day)
                    break
                except ValueError:
                    next_date = next_date.replace(day=next_date.day - 1)
                    if next_date.day == 1:
                        break
    elif rebalancing_frequency == '3months':
        # Add three months - handle month overflow safely
        try:
            new_month = base_date.month + 3
            new_year = base_date.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            next_date = base_date.replace(year=new_year, month=new_month)
        except ValueError:
            # Handle invalid day for target month
            new_month = base_date.month + 3
            new_year = base_date.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            next_date = base_date.replace(year=new_year, month=new_month, day=1)
            # Try to find a valid day in the target month
            while True:
                try:
                    next_date = next_date.replace(day=base_date.day)
                    break
                except ValueError:
                    next_date = next_date.replace(day=next_date.day - 1)
                    if next_date.day == 1:
                        break
    elif rebalancing_frequency == '6months':
        # Add six months - handle month overflow safely
        try:
            new_month = base_date.month + 6
            new_year = base_date.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            next_date = base_date.replace(year=new_year, month=new_month)
        except ValueError:
            # Handle invalid day for target month
            new_month = base_date.month + 6
            new_year = base_date.year + (new_month - 1) % 12
            new_month = ((new_month - 1) % 12) + 1
            next_date = base_date.replace(year=new_year, month=new_month, day=1)
            # Try to find a valid day in the target month
            while True:
                try:
                    next_date = next_date.replace(day=base_date.day)
                    break
                except ValueError:
                    next_date = next_date.replace(day=next_date.day - 1)
                    if next_date.day == 1:
                        break
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
            if next_date.month == 12:
                next_date = next_date.replace(year=next_date.year + 1, month=1)
            else:
                next_date = next_date.replace(month=next_date.month + 1)
            # Handle day overflow
            try:
                next_date = next_date.replace(day=min(next_date.day, 28))
            except ValueError:
                next_date = next_date.replace(day=1)
        elif rebalancing_frequency == '3months':
            # Add three months safely
            new_month = next_date.month + 3
            new_year = next_date.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            next_date = next_date.replace(year=new_year, month=new_month, day=1)
        elif rebalancing_frequency == '6months':
            # Add six months safely
            new_month = next_date.month + 6
            new_year = next_date.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            next_date = next_date.replace(year=new_year, month=new_month, day=1)
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

# FIXED: Correct Ulcer Index calculation - EXACTLY like Backtest_Engine.py
def calculate_ulcer_index(series):
    """Calculates the Ulcer Index (average squared percent drawdown, then sqrt)."""
    if series.empty:
        return np.nan
    peak = series.expanding(min_periods=1).max()
    drawdown = (series - peak) / peak * 100  # percent drawdown
    drawdown_sq = drawdown ** 2
    return np.sqrt(drawdown_sq.mean())

# FIXED: Correct Sharpe ratio calculation - EXACTLY like Backtest_Engine.py
def calculate_sharpe(returns, risk_free_rate):
    """Calculates the Sharpe ratio."""
    # Create a constant risk-free rate series aligned with returns
    daily_rf_rate = risk_free_rate / 252
    rf_series = pd.Series(daily_rf_rate, index=returns.index)
    
    aligned_returns, aligned_rf = returns.align(rf_series, join='inner')
    if aligned_returns.empty:
        return np.nan
    
    excess_returns = aligned_returns - aligned_rf
    if excess_returns.std() == 0:
        return np.nan
        
    return excess_returns.mean() / excess_returns.std() * np.sqrt(365)

# FIXED: Correct UPI calculation - EXACTLY like Backtest_Engine.py
def calculate_upi(cagr, ulcer_index):
    """Calculates the Ulcer Performance Index (UPI = CAGR / Ulcer Index, both as decimals)."""
    if ulcer_index is None or pd.isna(ulcer_index) or ulcer_index == 0:
        return np.nan
    return cagr / (ulcer_index / 100)

def calculate_total_money_added(config, start_date, end_date):
    """Calculate total money added to portfolio (initial + periodic additions)"""
    if start_date is None or end_date is None:
        return "N/A"
    
    # Initial investment
    initial_value = config.get('initial_value', 0)
    
    # Calculate periodic additions
    added_amount = config.get('added_amount', 0)
    added_frequency = config.get('added_frequency', 'None')
    
    if added_frequency in ['None', 'Never', 'none', None] or added_amount == 0:
        return initial_value
    
    # Get dates when additions were made
    dates_added = get_dates_by_freq(added_frequency, start_date, end_date, pd.date_range(start_date, end_date, freq='D'))
    
    # Count additions (excluding the first date which is initial investment)
    num_additions = len([d for d in dates_added if d != start_date])
    total_additions = num_additions * added_amount
    
    return initial_value + total_additions

# -----------------------
# Single-backtest core (adapted from your code, robust)
# -----------------------
def single_backtest(config, sim_index, reindexed_data):
    stocks_list = config['stocks']
    tickers = [s['ticker'] for s in stocks_list if s['ticker']]
    # Filter tickers to those present in reindexed_data to avoid KeyErrors for invalid tickers
    available_tickers = [t for t in tickers if t in reindexed_data]
    if len(available_tickers) < len(tickers):
        missing = set(tickers) - set(available_tickers)
    tickers = available_tickers
    # Recompute allocations and include_dividends to only include valid tickers
    # Handle duplicate tickers by summing their allocations
    allocations = {}
    include_dividends = {}
    for s in stocks_list:
        if s.get('ticker') and s.get('ticker') in tickers:
            ticker = s['ticker']
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
    benchmark_ticker = config['benchmark_ticker']
    initial_value = config.get('initial_value', 0)
    added_amount = config.get('added_amount', 0)
    added_frequency = config.get('added_frequency', 'none')
    rebalancing_frequency = config.get('rebalancing_frequency', 'none')
    use_momentum = config.get('use_momentum', True)
    momentum_windows = config.get('momentum_windows', [])
    calc_beta = config.get('calc_beta', False)
    calc_volatility = config.get('calc_volatility', False)
    beta_window_days = config.get('beta_window_days', 365)
    exclude_days_beta = config.get('exclude_days_beta', 30)
    vol_window_days = config.get('vol_window_days', 365)
    exclude_days_vol = config.get('exclude_days_vol', 30)
    current_data = {t: reindexed_data[t] for t in tickers + [benchmark_ticker] if t in reindexed_data}
    dates_added = get_dates_by_freq(added_frequency, sim_index[0], sim_index[-1], sim_index)
    
    # Get regular rebalancing dates
    dates_rebal = sorted(get_dates_by_freq(rebalancing_frequency, sim_index[0], sim_index[-1], sim_index))
    
    # Handle first rebalance strategy - replace first rebalance date if needed
    first_rebalance_strategy = st.session_state.get('multi_backtest_first_rebalance_strategy', 'rebalancing_date')
    if first_rebalance_strategy == "momentum_window_complete" and use_momentum and momentum_windows:
        try:
            # Calculate when momentum window completes
            window_sizes = [int(w.get('lookback', 0)) for w in momentum_windows if w is not None]
            max_window_days = max(window_sizes) if window_sizes else 0
            momentum_completion_date = sim_index[0] + pd.Timedelta(days=max_window_days)
            
            # Find the closest trading day to momentum completion
            momentum_completion_trading_day = sim_index[sim_index >= momentum_completion_date][0] if len(sim_index[sim_index >= momentum_completion_date]) > 0 else sim_index[-1]
            
            # Replace the first rebalancing date with momentum completion date
            if len(dates_rebal) > 0:
                # Remove the first rebalancing date and add momentum completion date
                dates_rebal = dates_rebal[1:] if len(dates_rebal) > 1 else []
                dates_rebal.insert(0, momentum_completion_trading_day)
                dates_rebal = sorted(dates_rebal)
        except Exception:
            pass  # Fall back to regular rebalancing dates

    # Dictionaries to store historical data for new tables
    historical_allocations = {}
    historical_metrics = {}

    def calculate_momentum(date, current_assets, momentum_windows):
        cumulative_returns, valid_assets = {}, []
        filtered_windows = [w for w in momentum_windows if w["weight"] > 0]
        # Normalize weights so they sum to 1 (same as app.py)
        total_weight = sum(w["weight"] for w in filtered_windows)
        if total_weight == 0:
            normalized_weights = [0 for _ in filtered_windows]
        else:
            normalized_weights = [w["weight"] / total_weight for w in filtered_windows]
        start_dates_config = {t: reindexed_data[t].first_valid_index() for t in tickers if t in reindexed_data}
        for t in current_assets:
            is_valid, asset_returns = True, 0.0
            for idx, window in enumerate(filtered_windows):
                lookback, exclude = window["lookback"], window["exclude"]
                weight = normalized_weights[idx]
                start_mom = date - pd.Timedelta(days=lookback)
                end_mom = date - pd.Timedelta(days=exclude)
                if start_dates_config.get(t, pd.Timestamp.max) > start_mom:
                    is_valid = False; break
                df_t = current_data[t]
                price_start_index = df_t.index.asof(start_mom)
                price_end_index = df_t.index.asof(end_mom)
                if pd.isna(price_start_index) or pd.isna(price_end_index):
                    is_valid = False; break
                price_start = df_t.loc[price_start_index, "Close"]
                price_end = df_t.loc[price_end_index, "Close"]
                if pd.isna(price_start) or pd.isna(price_end) or price_start == 0:
                    is_valid = False; break
                ret = (price_end - price_start) / price_start
                asset_returns += ret * weight
            if is_valid:
                cumulative_returns[t] = asset_returns
                valid_assets.append(t)
        return cumulative_returns, valid_assets

    def calculate_momentum_weights(returns, valid_assets, date, momentum_strategy='Classic', negative_momentum_strategy='Cash'):
        # Mirror approach used in allocations/app.py: compute weights from raw momentum
        # (Classic or Relative) and then optionally post-filter by inverse volatility
        # and inverse absolute beta (multiplicative), then renormalize. This avoids
        # dividing by beta directly which flips signs when beta is negative.
        if not valid_assets:
            return {}, {}
        # Keep only non-nan momentum values
        rets = {t: returns.get(t, np.nan) for t in valid_assets}
        rets = {t: rets[t] for t in rets if not pd.isna(rets[t])}
        if not rets:
            return {}, {}

        metrics = {t: {} for t in rets.keys()}

        # compute beta and volatility metrics when requested
        beta_vals = {}
        vol_vals = {}
        df_bench = current_data.get(benchmark_ticker)
        if calc_beta:
            start_beta = date - pd.Timedelta(days=beta_window_days)
            end_beta = date - pd.Timedelta(days=exclude_days_beta)
        if calc_volatility:
            start_vol = date - pd.Timedelta(days=vol_window_days)
            end_vol = date - pd.Timedelta(days=exclude_days_vol)

        for t in list(rets.keys()):
            df_t = current_data.get(t)
            if calc_beta and df_bench is not None and isinstance(df_t, pd.DataFrame):
                mask_beta = (df_t.index >= start_beta) & (df_t.index <= end_beta)
                returns_t_beta = df_t.loc[mask_beta, 'Price_change']
                mask_bench_beta = (df_bench.index >= start_beta) & (df_bench.index <= end_beta)
                returns_bench_beta = df_bench.loc[mask_bench_beta, 'Price_change']
                if len(returns_t_beta) < 2 or len(returns_bench_beta) < 2:
                    beta_vals[t] = np.nan
                else:
                    variance = np.var(returns_bench_beta)
                    beta_vals[t] = (np.cov(returns_t_beta, returns_bench_beta)[0,1] / variance) if variance > 0 else np.nan
                metrics[t]['Beta'] = beta_vals[t]
            if calc_volatility and isinstance(df_t, pd.DataFrame):
                mask_vol = (df_t.index >= start_vol) & (df_t.index <= end_vol)
                returns_t_vol = df_t.loc[mask_vol, 'Price_change']
                if len(returns_t_vol) < 2:
                    vol_vals[t] = np.nan
                else:
                    vol_vals[t] = returns_t_vol.std() * np.sqrt(365)
                metrics[t]['Volatility'] = vol_vals[t]

        # attach raw momentum
        for t in rets:
            metrics[t]['Momentum'] = rets[t]

        # Build initial weights from raw momentum (Classic or Relative)
        weights = {}
        rets_keys = list(rets.keys())
        all_negative = all(rets[t] <= 0 for t in rets_keys)
        relative_mode = isinstance(momentum_strategy, str) and momentum_strategy.lower().startswith('relat')

        if all_negative:
            if negative_momentum_strategy == 'Cash':
                weights = {t: 0.0 for t in rets_keys}
            elif negative_momentum_strategy == 'Equal weight':
                weights = {t: 1.0 / len(rets_keys) for t in rets_keys}
            else:  # Relative momentum
                min_score = min(rets[t] for t in rets_keys)
                offset = -min_score + 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
        else:
            if relative_mode:
                min_score = min(rets[t] for t in rets_keys)
                offset = -min_score + 0.01 if min_score < 0 else 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
            else:
                positive_scores = {t: rets[t] for t in rets_keys if rets[t] > 0}
                if positive_scores:
                    ssum = sum(positive_scores.values())
                    weights = {t: (positive_scores.get(t, 0.0) / ssum) for t in rets_keys}
                else:
                    weights = {t: 0.0 for t in rets_keys}

        # Post-filtering: multiply weights by inverse vol and inverse |beta| when requested
        if (calc_volatility or calc_beta) and weights:
            filter_scores = {}
            for t in weights:
                score = 1.0
                if calc_volatility:
                    v = metrics.get(t, {}).get('Volatility', np.nan)
                    if not pd.isna(v) and v > 0:
                        score *= 1.0 / v
                if calc_beta:
                    b = metrics.get(t, {}).get('Beta', np.nan)
                    if not pd.isna(b) and b != 0:
                        score *= 1.0 / abs(b)
                filter_scores[t] = score

            filtered = {t: weights.get(t, 0.0) * filter_scores.get(t, 1.0) for t in weights}
            ssum = sum(filtered.values())
            # If filtering removes all weight (sum==0), fall back to unfiltered weights
            if ssum > 0:
                weights = {t: filtered[t] / ssum for t in filtered}

        # Apply minimal threshold filter if enabled
        use_threshold = config.get('use_minimal_threshold', False)
        threshold_percent = config.get('minimal_threshold_percent', 2.0)
        
        if use_threshold and weights:
            threshold_decimal = threshold_percent / 100.0
            
            # STEP 1: Complete rebalancing simulation is already done above
            # (momentum calculation, beta/volatility filtering, etc. - weights are the "simulation" result)
            
            # STEP 2: Check which stocks are below threshold in the simulation
            filtered_weights = {}
            for ticker, weight in weights.items():
                if weight >= threshold_decimal:
                    # Keep stocks above or equal to threshold (remove stocks below threshold)
                    filtered_weights[ticker] = weight
            
            # STEP 3: Do the actual rebalancing with only the remaining stocks
            if filtered_weights:
                total_weight = sum(filtered_weights.values())
                if total_weight > 0:
                    # Normalize remaining stocks to sum to 1.0
                    weights = {ticker: weight / total_weight for ticker, weight in filtered_weights.items()}
                else:
                    weights = {}
            else:
                # If no stocks meet threshold, keep original weights
                weights = weights

        # Attach calculated weights to metrics and return
        for t in weights:
            metrics[t]['Calculated_Weight'] = weights.get(t, 0.0)

        # Debug print when beta/vol are used
        if calc_beta or calc_volatility:
            try:
                for t in rets_keys:
                    pass
            except Exception:
                pass

        return weights, metrics
        # --- MODIFIED LOGIC END ---

    values = {t: [0.0] for t in tickers}
    unallocated_cash = [0.0]
    unreinvested_cash = [0.0]
    portfolio_no_additions = [initial_value]
    
    # Initial allocation and metric storage
    if not use_momentum:
        current_allocations = {t: allocations.get(t,0) for t in tickers}
        
        # Apply minimal threshold filter for non-momentum strategies
        use_threshold = config.get('use_minimal_threshold', False)
        threshold_percent = config.get('minimal_threshold_percent', 2.0)
        
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
    else:
        returns, valid_assets = calculate_momentum(sim_index[0], set(tickers), momentum_windows)
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
                if "Dividends" in df.columns:
                    # If dividend is not on a trading day, roll forward to next available trading day
                    if date in df.index:
                        div = df.loc[date, "Dividends"]
                    else:
                        # Find next trading day in index after 'date'
                        future_dates = df.index[df.index > date]
                        if len(future_dates) > 0:
                            div = df.loc[future_dates[0], "Dividends"]
                var = df.loc[date, "Price_change"] if date in df.index else 0.0
                if include_dividends.get(t, False):
                    # Check if dividends should be collected as cash instead of reinvested
                    collect_as_cash = config.get('collect_dividends_as_cash', False)
                    if collect_as_cash and div > 0:
                        # Calculate dividend cash and add to unreinvested cash
                        nb_shares = val_prev / price_prev if price_prev > 0 else 0
                        dividend_cash = nb_shares * div
                        total_unreinvested_dividends += dividend_cash
                        # Don't include dividend in rate of return
                        rate_of_return = var
                        val_new = val_prev * (1 + rate_of_return)
                    else:
                        # Reinvest dividends (original behavior)
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
            if "Dividends" in df.columns:
                if date in df.index:
                    div = df.loc[date, "Dividends"]
                else:
                    future_dates = df.index[df.index > date]
                    if len(future_dates) > 0:
                        div = df.loc[future_dates[0], "Dividends"]
            var = df.loc[date, "Price_change"] if date in df.index else 0.0
            if include_dividends.get(t, False):
                # Check if dividends should be collected as cash instead of reinvested
                collect_as_cash = config.get('collect_dividends_as_cash', False)
                if collect_as_cash and div > 0:
                    # Calculate dividend cash and add to unreinvested cash
                    nb_shares = val_prev / price_prev if price_prev > 0 else 0
                    dividend_cash = nb_shares * div
                    total_unreinvested_dividends += dividend_cash
                    # Don't include dividend in rate of return
                    rate_of_return = var
                    val_new = val_prev * (1 + rate_of_return)
                else:
                    # Reinvest dividends (original behavior)
                    rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                    val_new = val_prev * (1 + rate_of_return)
            else:
                val_new = val_prev * (1 + var)
            values[t].append(val_new)
        unallocated_cash.append(unallocated_cash[-1])
        if date in dates_added:
            unallocated_cash[-1] += added_amount
        unreinvested_cash.append(unreinvested_cash[-1] + total_unreinvested_dividends)
        portfolio_no_additions.append(portfolio_no_additions[-1] * daily_growth_factor)
        
        current_total = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
        
        # Check if we should rebalance
        should_rebalance = False
        if date in dates_rebal and set(tickers):
            should_rebalance = True
        elif rebalancing_frequency in ["Buy & Hold", "Buy & Hold (Target)"] and set(tickers):
            # Buy & Hold: rebalance whenever there's cash available
            total_cash = unallocated_cash[-1] + unreinvested_cash[-1]
            if total_cash > 0:
                should_rebalance = True
        
        if should_rebalance:
            if use_momentum:
                returns, valid_assets = calculate_momentum(date, set(tickers), momentum_windows)
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
                        for t in tickers:
                            values[t][-1] = current_total * weights.get(t, 0)
                        unreinvested_cash[-1] = 0
                        unallocated_cash[-1] = 0
            else:
                # Apply threshold filter for non-momentum strategies during rebalancing
                use_threshold = config.get('use_minimal_threshold', False)
                threshold_percent = config.get('minimal_threshold_percent', 2.0)
                
                if use_threshold:
                    threshold_decimal = threshold_percent / 100.0
                    
                    # First: Filter out stocks below threshold
                    filtered_allocations = {}
                    for t in tickers:
                        allocation = allocations.get(t, 0)
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
                        # If no stocks meet threshold, use original allocations
                        rebalance_allocations = {t: allocations.get(t, 0) for t in tickers}
                else:
                    # Use original allocations if threshold filter is disabled
                    rebalance_allocations = {t: allocations.get(t, 0) for t in tickers}
                
                sum_alloc = sum(rebalance_allocations.values())
                if sum_alloc > 0:
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

    # Store last allocation
    last_date = sim_index[-1]
    last_total = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
    if last_total > 0:
        historical_allocations[last_date] = {t: values[t][-1] / last_total for t in tickers}
        historical_allocations[last_date]['CASH'] = unallocated_cash[-1] / last_total if last_total > 0 else 0
    else:
        historical_allocations[last_date] = {t: 0 for t in tickers}
        historical_allocations[last_date]['CASH'] = 0
    
    # Store last metrics: always add a last-rebalance snapshot so the UI has a metrics row
    # If momentum is used, compute metrics; otherwise build metrics from the last allocation snapshot
    if use_momentum:
        returns, valid_assets = calculate_momentum(last_date, set(tickers), momentum_windows)
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
# PAGE-SCOPED SESSION STATE INITIALIZATION - MULTI-BACKTEST PAGE
# -----------------------
# Ensure complete independence from other pages by using page-specific session keys
if 'multi_backtest_page_initialized' not in st.session_state:
    st.session_state.multi_backtest_page_initialized = True
    # Clear any shared session state that might interfere with other pages
    keys_to_clear = [
        # Main app keys
        'main_portfolio_configs', 'main_active_portfolio_index', 'main_rerun_flag',
        'main_all_results', 'main_all_allocations', 'main_all_metrics',
        'main_drawdowns', 'main_stats_df', 'main_years_data', 'main_portfolio_map',
        'main_backtest_ran', 'main_raw_data', 'main_running', 'main_run_requested',
        'main_pending_backtest_params', 'main_tickers', 'main_allocs', 'main_divs',
        'main_use_momentum', 'main_mom_windows', 'main_use_beta', 'main_use_vol',
        'main_initial_value_input_decimals', 'main_initial_value_input_int',
        'main_added_amount_input_decimals', 'main_added_amount_input_int',
        'main_start_date', 'main_end_date', 'main_use_custom_dates',
        'main_momentum_strategy', 'main_negative_momentum_strategy',
        'main_beta_window_days', 'main_beta_exclude_days', 'main_vol_window_days', 'main_vol_exclude_days',
        # Allocations page keys
        'alloc_portfolio_configs', 'alloc_active_portfolio_index', 'alloc_rerun_flag',
        'alloc_all_results', 'alloc_all_allocations', 'alloc_all_metrics',
        'alloc_paste_json_text', 'allocations_page_initialized',
        # Any other potential shared keys
        'multi_all_results', 'multi_all_allocations', 'multi_all_metrics',
        'all_drawdowns', 'stats_df_display', 'all_years', 'portfolio_key_map',
        'multi_backtest_ran', 'raw_data'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# Main App Logic
# -----------------------

if 'multi_backtest_portfolio_configs' not in st.session_state:
    st.session_state.multi_backtest_portfolio_configs = default_configs
if 'multi_backtest_active_portfolio_index' not in st.session_state:
    st.session_state.multi_backtest_active_portfolio_index = 0

# Ensure all portfolios have threshold settings
for portfolio in st.session_state.multi_backtest_portfolio_configs:
    if 'use_minimal_threshold' not in portfolio:
        portfolio['use_minimal_threshold'] = False
    if 'minimal_threshold_percent' not in portfolio:
        portfolio['minimal_threshold_percent'] = 2.0

if 'multi_backtest_paste_json_text' not in st.session_state:
    st.session_state.multi_backtest_paste_json_text = ""
if 'multi_backtest_rerun_flag' not in st.session_state:
    st.session_state.multi_backtest_rerun_flag = False

def ensure_unique_portfolio_name(proposed_name, existing_portfolios):
    """
    Ensures a portfolio name is unique by adding (1), (2), etc. if needed.
    This happens BEFORE the portfolio is added to prevent crashes.
    """
    existing_names = [p.get('name', '') for p in existing_portfolios]
    
    # If name is unique, return as-is
    if proposed_name not in existing_names:
        return proposed_name
    
    # Find the next available number
    counter = 1
    while f"{proposed_name} ({counter})" in existing_names:
        counter += 1
    
    return f"{proposed_name} ({counter})"

def add_portfolio_to_configs(portfolio):
    """
    CENTRAL function to add ANY portfolio to the configs.
    ALL portfolio additions MUST go through this function.
    Automatically ensures unique names no matter the source.
    """
    # Ensure unique name IMMEDIATELY
    unique_name = ensure_unique_portfolio_name(
        portfolio.get('name', 'Unnamed Portfolio'), 
        st.session_state.multi_backtest_portfolio_configs
    )
    portfolio['name'] = unique_name
    
    # Add to configs
    st.session_state.multi_backtest_portfolio_configs.append(portfolio)
    return portfolio

def ensure_all_portfolio_names_unique():
    """
    NUCLEAR OPTION: Ensures ALL existing portfolios have unique names.
    Call this at startup or after any bulk operations.
    """
    if 'multi_backtest_portfolio_configs' not in st.session_state:
        return
    
    configs = st.session_state.multi_backtest_portfolio_configs
    seen_names = set()
    
    for i, portfolio in enumerate(configs):
        original_name = portfolio.get('name', f'Unnamed Portfolio {i+1}')
        
        # If this name is already seen, make it unique
        if original_name in seen_names:
            counter = 1
            while f"{original_name} ({counter})" in seen_names:
                counter += 1
            portfolio['name'] = f"{original_name} ({counter})"
        
        seen_names.add(portfolio['name'])

# NUCLEAR OPTION: Ensure all portfolio names are unique at startup
# Called AFTER all functions are defined
ensure_all_portfolio_names_unique()

# CONTINUOUS MONITORING: Check for duplicates on every UI render
# This catches ANY duplicate that appears through ANY unknown method
def continuous_duplicate_check():
    """
    GOD-PROOF duplicate checking - runs on every UI render.
    Even if GOD spawns a duplicate portfolio, this catches it.
    """
    if 'multi_backtest_portfolio_configs' not in st.session_state:
        return
    
    configs = st.session_state.multi_backtest_portfolio_configs
    names = [p.get('name', '') for p in configs]
    
    # Check if there are any duplicates
    if len(names) != len(set(names)):
        # Duplicates found! Fix them immediately
        ensure_all_portfolio_names_unique()

# Call continuous check on every render
continuous_duplicate_check()

def add_portfolio_callback():
    new_portfolio = default_configs[1].copy()
    proposed_name = f"New Portfolio {len(st.session_state.multi_backtest_portfolio_configs) + 1}"
    new_portfolio['name'] = proposed_name
    
    # Use central function - automatically ensures unique name
    add_portfolio_to_configs(new_portfolio)
    st.session_state.multi_backtest_active_portfolio_index = len(st.session_state.multi_backtest_portfolio_configs) - 1
    st.session_state.multi_backtest_rerun_flag = True

def remove_portfolio_callback():
    if len(st.session_state.multi_backtest_portfolio_configs) > 1:
        st.session_state.multi_backtest_portfolio_configs.pop(st.session_state.multi_backtest_active_portfolio_index)
        st.session_state.multi_backtest_active_portfolio_index = max(0, st.session_state.multi_backtest_active_portfolio_index - 1)
        st.session_state.multi_backtest_rerun_flag = True

def bulk_delete_portfolios_callback(portfolio_names_to_delete):
    """Delete multiple portfolios at once"""
    if len(st.session_state.multi_backtest_portfolio_configs) <= 1:
        return  # Don't delete the last portfolio
    
    # Get indices of portfolios to delete
    indices_to_delete = []
    for name in portfolio_names_to_delete:
        for i, cfg in enumerate(st.session_state.multi_backtest_portfolio_configs):
            if cfg['name'] == name:
                indices_to_delete.append(i)
                break
    
    # Sort indices in descending order to avoid index shifting issues
    indices_to_delete.sort(reverse=True)
    
    # Delete portfolios
    deleted_count = 0
    for idx in indices_to_delete:
        if len(st.session_state.multi_backtest_portfolio_configs) > 1:
            st.session_state.multi_backtest_portfolio_configs.pop(idx)
            deleted_count += 1
    
    # Clear all checkboxes after deletion
    st.session_state.multi_backtest_portfolio_checkboxes = {}
    
    # Update active portfolio index if necessary
    if st.session_state.multi_backtest_active_portfolio_index >= len(st.session_state.multi_backtest_portfolio_configs):
        st.session_state.multi_backtest_active_portfolio_index = max(0, len(st.session_state.multi_backtest_portfolio_configs) - 1)
    
    # Set success message
    st.session_state.multi_backtest_bulk_delete_success = f"Successfully deleted {deleted_count} portfolio(s)!"
    
    st.session_state.multi_backtest_rerun_flag = True

def add_stock_callback():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks'].append({'ticker': '', 'allocation': 0.0, 'include_dividends': True})
    # Removed rerun flag - no need to refresh entire page for adding a stock

def remove_stock_callback(ticker):
    """Immediate stock removal callback - OPTIMIZED NO REFRESH"""
    try:
        active_portfolio = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]
        stocks = active_portfolio['stocks']
        
        # Find and remove the stock with matching ticker
        for i, stock in enumerate(stocks):
            if stock['ticker'] == ticker:
                stocks.pop(i)
                # If this was the last stock, add an empty one
                if len(stocks) == 0:
                    stocks.append({'ticker': '', 'allocation': 0.0, 'include_dividends': True})
                # Removed rerun flag - no need to refresh entire page for removing a stock
                break
    except (IndexError, KeyError):
        pass

def normalize_stock_allocations_callback():
    if 'multi_backtest_portfolio_configs' not in st.session_state or 'multi_backtest_active_portfolio_index' not in st.session_state:
        return
    stocks = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks']
    valid_stocks = [s for s in stocks if s['ticker']]
    total_alloc = sum(s['allocation'] for s in valid_stocks)
    if total_alloc > 0:
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] /= total_alloc
                alloc_key = f"multi_backtest_alloc_input_{st.session_state.multi_backtest_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = int(s['allocation'] * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"multi_backtest_alloc_input_{st.session_state.multi_backtest_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks'] = stocks
    st.session_state.multi_backtest_rerun_flag = True

def equal_stock_allocation_callback():
    if 'multi_backtest_portfolio_configs' not in st.session_state or 'multi_backtest_portfolio_configs' not in st.session_state or 'multi_backtest_active_portfolio_index' not in st.session_state:
        return
    stocks = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks']
    valid_stocks = [s for s in stocks if s['ticker']]
    if valid_stocks:
        equal_weight = 1.0 / len(valid_stocks)
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] = equal_weight
                alloc_key = f"multi_backtest_alloc_input_{st.session_state.multi_backtest_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = int(equal_weight * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"multi_backtest_alloc_input_{st.session_state.multi_backtest_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks'] = stocks
    st.session_state.multi_backtest_rerun_flag = True
    
def reset_portfolio_callback():
    current_name = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['name']
    default_cfg_found = next((cfg for cfg in default_configs if cfg['name'] == current_name), None)
    if default_cfg_found is None:
        default_cfg_found = default_configs[1].copy()
        default_cfg_found['name'] = current_name
    # Clear any saved momentum settings when resetting
    if 'saved_momentum_settings' in default_cfg_found:
        del default_cfg_found['saved_momentum_settings']
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index] = default_cfg_found
    st.session_state.multi_backtest_rerun_flag = True

def reset_stock_selection_callback():
    current_name = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['name']
    default_cfg_found = next((cfg for cfg in default_configs if cfg['name'] == current_name), None)
    if default_cfg_found is None:
        default_cfg_found = default_configs[1].copy()
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks'] = default_cfg_found['stocks']
    st.session_state.multi_backtest_rerun_flag = True

def reset_momentum_windows_callback():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['momentum_windows'] = [
        {"lookback": 365, "exclude": 30, "weight": 0.5},
        {"lookback": 180, "exclude": 30, "weight": 0.3},
        {"lookback": 120, "exclude": 30, "weight": 0.2},
    ]
    st.session_state.multi_backtest_rerun_flag = True

def reset_beta_callback():
    # Reset beta lookback/exclude to defaults and enable beta calculation
    idx = st.session_state.multi_backtest_active_portfolio_index
    st.session_state.multi_backtest_portfolio_configs[idx]['beta_window_days'] = 365
    st.session_state.multi_backtest_portfolio_configs[idx]['exclude_days_beta'] = 30
    # Ensure checkbox state reflects enabled
    st.session_state.multi_backtest_portfolio_configs[idx]['calc_beta'] = True
    st.session_state['multi_backtest_active_calc_beta'] = True
    # Update UI widget values to reflect reset
    st.session_state['multi_backtest_active_beta_window'] = 365
    st.session_state['multi_backtest_active_beta_exclude'] = 30
    # Trigger rerun to update UI
    st.session_state.multi_backtest_rerun_flag = True

def reset_vol_callback():
    # Reset volatility lookback/exclude to defaults and enable volatility calculation
    idx = st.session_state.multi_backtest_active_portfolio_index
    st.session_state.multi_backtest_portfolio_configs[idx]['vol_window_days'] = 365
    st.session_state.multi_backtest_portfolio_configs[idx]['exclude_days_vol'] = 30
    st.session_state.multi_backtest_portfolio_configs[idx]['calc_volatility'] = True
    st.session_state['multi_backtest_active_calc_vol'] = True
    # Update UI widget values to reflect reset
    st.session_state['multi_backtest_active_vol_window'] = 365
    st.session_state['multi_backtest_active_vol_exclude'] = 30
    # Trigger rerun to update UI
    st.session_state.multi_backtest_rerun_flag = True

def sync_cashflow_from_first_portfolio_callback():
    """Sync initial value, added amount, and added frequency from first portfolio to all others"""
    try:
        if len(st.session_state.multi_backtest_portfolio_configs) > 1:
            first_portfolio = st.session_state.multi_backtest_portfolio_configs[0]
            
            # Get values from first portfolio
            initial_value = first_portfolio.get('initial_value', 10000)
            added_amount = first_portfolio.get('added_amount', 1000)
            added_frequency = first_portfolio.get('added_frequency', 'Monthly')
            
            # Update all other portfolios (skip those excluded from sync)
            updated_count = 0
            for i in range(1, len(st.session_state.multi_backtest_portfolio_configs)):
                portfolio = st.session_state.multi_backtest_portfolio_configs[i]
                if not portfolio.get('exclude_from_cashflow_sync', False):
                    # Only update if values are actually different
                    if (portfolio.get('initial_value') != initial_value or 
                        portfolio.get('added_amount') != added_amount or 
                        portfolio.get('added_frequency') != added_frequency):
                        portfolio['initial_value'] = initial_value
                        portfolio['added_amount'] = added_amount
                        portfolio['added_frequency'] = added_frequency
                        updated_count += 1
            
            # Only update UI and rerun if something actually changed
            if updated_count > 0:
                # Only update UI widgets if the current portfolio is NOT excluded from cash flow sync
                current_portfolio = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]
                if not current_portfolio.get('exclude_from_cashflow_sync', False):
                    # Update UI widget session states to reflect the changes
                    st.session_state['multi_backtest_active_initial'] = initial_value
                    st.session_state['multi_backtest_active_added_amount'] = added_amount
                    st.session_state['multi_backtest_active_add_freq'] = added_frequency
                
                # Store success message in session state instead of showing it at top
                st.session_state['multi_backtest_cashflow_sync_message'] = f"✅ Successfully synced cashflow settings to {updated_count} portfolio(s)"
                st.session_state['multi_backtest_cashflow_sync_message_type'] = 'success'
                
                # Force immediate rerun to show changes
                st.session_state.multi_backtest_rerun_flag = True
            else:
                # Store info message in session state
                st.session_state['multi_backtest_cashflow_sync_message'] = "ℹ️ No portfolios were updated (all were excluded or already had matching values)"
                st.session_state['multi_backtest_cashflow_sync_message_type'] = 'info'
    except Exception as e:
        # Store error message in session state
        st.session_state['multi_backtest_cashflow_sync_message'] = f"❌ Error during cash flow sync: {str(e)}"
        st.session_state['multi_backtest_cashflow_sync_message_type'] = 'error'

def sync_rebalancing_from_first_portfolio_callback():
    """Sync rebalancing frequency from first portfolio to all others"""
    try:
        if len(st.session_state.multi_backtest_portfolio_configs) > 1:
            first_portfolio = st.session_state.multi_backtest_portfolio_configs[0]
            
            # Get rebalancing frequency from first portfolio
            rebalancing_frequency = first_portfolio.get('rebalancing_frequency', 'Monthly')
            
            # Update all other portfolios (skip those excluded from sync)
            updated_count = 0
            for i in range(1, len(st.session_state.multi_backtest_portfolio_configs)):
                portfolio = st.session_state.multi_backtest_portfolio_configs[i]
                if not portfolio.get('exclude_from_rebalancing_sync', False):
                    # Only update if value is actually different
                    if portfolio.get('rebalancing_frequency') != rebalancing_frequency:
                        portfolio['rebalancing_frequency'] = rebalancing_frequency
                        updated_count += 1
            
            # Only update UI and rerun if something actually changed
            if updated_count > 0:
                # Only update UI widgets if the current portfolio is NOT excluded from rebalancing sync
                current_portfolio = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]
                if not current_portfolio.get('exclude_from_rebalancing_sync', False):
                    # Update UI widget session state to reflect the change
                    st.session_state['multi_backtest_active_rebal_freq'] = rebalancing_frequency
                
                # Store success message in session state instead of showing it at top
                st.session_state['multi_backtest_rebalancing_sync_message'] = f"✅ Successfully synced rebalancing frequency to {updated_count} portfolio(s)"
                st.session_state['multi_backtest_rebalancing_sync_message_type'] = 'success'
                
                # Force immediate rerun to show changes
                st.session_state.multi_backtest_rerun_flag = True
            else:
                # Store info message in session state
                st.session_state['multi_backtest_rebalancing_sync_message'] = "ℹ️ No portfolios were updated (all were excluded or already had matching values)"
                st.session_state['multi_backtest_rebalancing_sync_message_type'] = 'info'
    except Exception as e:
        # Store error message in session state
        st.session_state['multi_backtest_rebalancing_sync_message'] = f"❌ Error during rebalancing sync: {str(e)}"
        st.session_state['multi_backtest_rebalancing_sync_message_type'] = 'error'

def add_momentum_window_callback():
    # Append a new momentum window with modest defaults
    idx = st.session_state.multi_backtest_active_portfolio_index
    cfg = st.session_state.multi_backtest_portfolio_configs[idx]
    if 'momentum_windows' not in cfg:
        cfg['momentum_windows'] = []
    # default new window
    cfg['momentum_windows'].append({"lookback": 90, "exclude": 30, "weight": 0.1})
    # Don't trigger immediate re-run for better performance
    # st.session_state.multi_backtest_rerun_flag = True
    st.session_state.multi_backtest_portfolio_configs[idx] = cfg
    # Don't trigger immediate re-run for better performance
    # st.session_state.multi_backtest_rerun_flag = True

def remove_momentum_window_callback():
    idx = st.session_state.multi_backtest_active_portfolio_index
    cfg = st.session_state.multi_backtest_portfolio_configs[idx]
    if 'momentum_windows' in cfg and cfg['momentum_windows']:
        cfg['momentum_windows'].pop()
        st.session_state.multi_backtest_portfolio_configs[idx] = cfg
        # Don't trigger immediate re-run for better performance
        # st.session_state.multi_backtest_rerun_flag = True

def normalize_momentum_weights_callback():
    if 'multi_backtest_portfolio_configs' not in st.session_state or 'multi_backtest_active_portfolio_index' not in st.session_state:
        return
    active_portfolio = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]
    total_weight = sum(w['weight'] for w in active_portfolio['momentum_windows'])
    if total_weight > 0:
        for idx, w in enumerate(active_portfolio['momentum_windows']):
            w['weight'] /= total_weight
            weight_key = f"multi_backtest_weight_input_active_{idx}"
            # Sanitize weight to prevent StreamlitValueAboveMaxError
            weight = w['weight']
            if isinstance(weight, (int, float)):
                # Convert decimal to percentage, ensuring it's within bounds
                weight_percentage = max(0.0, min(weight * 100.0, 100.0))
            else:
                # Invalid weight, set to default
                weight_percentage = 10.0
            st.session_state[weight_key] = int(weight_percentage)
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['momentum_windows'] = active_portfolio['momentum_windows']
    st.session_state.multi_backtest_rerun_flag = True

def paste_json_callback():
    try:
        # Use the SAME parsing logic as successful PDF extraction
        raw_text = st.session_state.multi_backtest_paste_json_text
        
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
            json_data['minimal_threshold_percent'] = 2.0
        
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
                        
                        stock = {
                            'ticker': tickers[i].strip(),
                            'allocation': allocation,
                            'include_dividends': bool(divs[i]) if i < len(divs) and divs[i] is not None else True
                        }
                        stocks.append(stock)
            
            # Debug output
            st.info(f"Converted {len(stocks)} stocks from legacy format: {[s['ticker'] for s in stocks]}")
        
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
        
        # Map frequency values from app.py format to Multi-Backtest format
        def map_frequency(freq):
            if freq is None:
                return 'Never'
            freq_map = {
                'Never': 'Never',
                'Buy & Hold': 'Buy & Hold',
                'Buy & Hold (Target)': 'Buy & Hold (Target)',
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
        
        # Multi-Backtest page specific: ensure all required fields are present
        # and ignore fields that are specific to other pages
        multi_backtest_config = {
            'name': json_data.get('name', 'New Portfolio'),
            'stocks': stocks,
            'benchmark_ticker': json_data.get('benchmark_ticker', '^GSPC'),
            'initial_value': json_data.get('initial_value', 10000),
            'added_amount': json_data.get('added_amount', 1000),
            'added_frequency': map_frequency(json_data.get('added_frequency', 'Monthly')),
            'rebalancing_frequency': map_frequency(json_data.get('rebalancing_frequency', 'Monthly')),
            'start_date_user': parse_date_from_json(json_data.get('start_date_user')),
            'end_date_user': parse_date_from_json(json_data.get('end_date_user')),
            'start_with': json_data.get('start_with', 'first'),
            'use_momentum': json_data.get('use_momentum', True),
            'momentum_strategy': momentum_strategy,
            'negative_momentum_strategy': negative_momentum_strategy,
            'momentum_windows': momentum_windows,
            'use_minimal_threshold': json_data.get('use_minimal_threshold', False),
            'minimal_threshold_percent': json_data.get('minimal_threshold_percent', 2.0),
            'calc_beta': json_data.get('calc_beta', True),
            'calc_volatility': json_data.get('calc_volatility', True),
            'beta_window_days': json_data.get('beta_window_days', 365),
            'exclude_days_beta': json_data.get('exclude_days_beta', 30),
            'vol_window_days': json_data.get('vol_window_days', 365),
            'exclude_days_vol': json_data.get('exclude_days_vol', 30),
            'collect_dividends_as_cash': json_data.get('collect_dividends_as_cash', False),
            # Preserve sync exclusion settings from imported JSON
            'exclude_from_cashflow_sync': json_data.get('exclude_from_cashflow_sync', False),
            'exclude_from_rebalancing_sync': json_data.get('exclude_from_rebalancing_sync', False),
        }
        
        st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index] = multi_backtest_config
        
        # Handle global start_with setting from imported JSON
        if 'start_with' in json_data:
            # Handle start_with value mapping from other pages
            start_with = json_data['start_with']
            if start_with == 'first':
                start_with = 'oldest'  # Map 'first' to 'oldest' (closest equivalent)
            elif start_with not in ['all', 'oldest']:
                start_with = 'all'  # Default fallback
            st.session_state['multi_backtest_start_with'] = start_with
            # Update the radio button widget key
            st.session_state['multi_backtest_start_with_radio'] = start_with
        
        # Handle first rebalance strategy from imported JSON
        if 'first_rebalance_strategy' in json_data:
            st.session_state['multi_backtest_first_rebalance_strategy'] = json_data['first_rebalance_strategy']
            # Update the radio button widget key
            st.session_state['multi_backtest_first_rebalance_strategy_radio'] = json_data['first_rebalance_strategy']
        
        st.success("Portfolio configuration updated from JSON (Multi-Backtest page).")
        st.info(f"Final stocks list: {[s['ticker'] for s in multi_backtest_config['stocks']]}")
        st.info(f"Final momentum windows: {multi_backtest_config['momentum_windows']}")
        st.info(f"Final use_momentum: {multi_backtest_config['use_momentum']}")
        st.info(f"Sync exclusions - Cash Flow: {multi_backtest_config.get('exclude_from_cashflow_sync', False)}, Rebalancing: {multi_backtest_config.get('exclude_from_rebalancing_sync', False)}")
        
        # Sync date widgets with the updated portfolio
        sync_date_widgets_with_portfolio()
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check the text and try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    st.session_state.multi_backtest_rerun_flag = True

def update_active_portfolio_index():
    # Use safe accessors to avoid AttributeError when keys are not yet set
    selected_name = st.session_state.get('multi_backtest_portfolio_selector', None)
    portfolio_configs = st.session_state.get('multi_backtest_portfolio_configs', [])
    portfolio_names = [cfg.get('name', '') for cfg in portfolio_configs]
    
    if selected_name and selected_name in portfolio_names:
        new_index = portfolio_names.index(selected_name)
        st.session_state.multi_backtest_active_portfolio_index = new_index
    else:
        # default to first portfolio if selector is missing or value not found
        st.session_state.multi_backtest_active_portfolio_index = 0 if portfolio_names else None
    
    # Additional safety check - ensure index is always valid
    if (st.session_state.multi_backtest_active_portfolio_index is not None and 
        st.session_state.multi_backtest_active_portfolio_index >= len(portfolio_names)):
        st.session_state.multi_backtest_active_portfolio_index = max(0, len(portfolio_names) - 1) if portfolio_names else None
    
    # Sync date widgets with the new portfolio
    sync_date_widgets_with_portfolio()
    
    # NUCLEAR SYNC: FORCE momentum widgets to sync with the new portfolio
    if portfolio_configs and st.session_state.multi_backtest_active_portfolio_index is not None:
        active_portfolio = portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]
        
        # NUCLEAR APPROACH: FORCE all momentum session state widgets to sync
        st.session_state['multi_backtest_active_use_momentum'] = active_portfolio.get('use_momentum', False)
        st.session_state['multi_backtest_active_momentum_strategy'] = active_portfolio.get('momentum_strategy', 'Classic')
        st.session_state['multi_backtest_active_negative_momentum_strategy'] = active_portfolio.get('negative_momentum_strategy', 'Cash')
        st.session_state['multi_backtest_active_calc_beta'] = active_portfolio.get('calc_beta', True)
        st.session_state['multi_backtest_active_calc_vol'] = active_portfolio.get('calc_volatility', True)
        st.session_state['multi_backtest_active_beta_window'] = active_portfolio.get('beta_window_days', 365)
        st.session_state['multi_backtest_active_beta_exclude'] = active_portfolio.get('exclude_days_beta', 30)
        st.session_state['multi_backtest_active_vol_window'] = active_portfolio.get('vol_window_days', 365)
        st.session_state['multi_backtest_active_vol_exclude'] = active_portfolio.get('exclude_days_vol', 30)
        
        # Sync expander state (same pattern as other portfolio parameters)
        st.session_state['multi_backtest_active_variant_expanded'] = active_portfolio.get('variant_expander_expanded', False)
        st.session_state['multi_backtest_active_use_threshold'] = active_portfolio.get('use_minimal_threshold', False)
        st.session_state['multi_backtest_active_threshold_percent'] = active_portfolio.get('minimal_threshold_percent', 2.0)
        
        # NUCLEAR: If portfolio has momentum enabled but no windows, FORCE create them
        if active_portfolio.get('use_momentum', False) and not active_portfolio.get('momentum_windows'):
            active_portfolio['momentum_windows'] = [
                {"lookback": 365, "exclude": 30, "weight": 0.5},
                {"lookback": 180, "exclude": 30, "weight": 0.3},
                {"lookback": 120, "exclude": 30, "weight": 0.2},
            ]
            print(f"NUCLEAR: FORCED momentum windows for portfolio {active_portfolio.get('name', 'Unknown')}")
        
        # NUCLEAR: Ensure threshold settings exist
        if 'use_minimal_threshold' not in active_portfolio:
            active_portfolio['use_minimal_threshold'] = False
        if 'minimal_threshold_percent' not in active_portfolio:
            active_portfolio['minimal_threshold_percent'] = 2.0
        
        print(f"NUCLEAR: Synced momentum widgets for portfolio {active_portfolio.get('name', 'Unknown')}, use_momentum={active_portfolio.get('use_momentum', False)}, windows_count={len(active_portfolio.get('momentum_windows', []))}")
        
        # RESET variant generator checkboxes when switching portfolios
        # This prevents stale checkbox states from previous portfolio selections
        variant_generator_keys = [
            "multi_use_momentum_vary",
            # Rebalance frequency checkboxes
            "multi_rebalance_never", "multi_rebalance_buyhold", "multi_rebalance_buyhold_target",
            "multi_rebalance_weekly", "multi_rebalance_biweekly", "multi_rebalance_monthly",
            "multi_rebalance_quarterly", "multi_rebalance_semiannually", "multi_rebalance_annually",
            # Momentum variant checkboxes
            "multi_momentum_classic", "multi_momentum_relative",
            "multi_negative_cash", "multi_negative_equal", "multi_negative_relative", 
            "multi_beta_yes", "multi_beta_no", "multi_vol_yes", "multi_vol_no"
        ]
        for key in variant_generator_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    st.session_state.multi_backtest_rerun_flag = True

def update_name():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['name'] = st.session_state.multi_backtest_active_name

def update_initial():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['initial_value'] = st.session_state.multi_backtest_active_initial

def update_added_amount():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['added_amount'] = st.session_state.multi_backtest_active_added_amount

def update_add_freq():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['added_frequency'] = st.session_state.multi_backtest_active_add_freq

def update_rebal_freq():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['rebalancing_frequency'] = st.session_state.multi_backtest_active_rebal_freq

def update_benchmark():
    # Convert benchmark ticker to uppercase
    upper_benchmark = st.session_state.multi_backtest_active_benchmark.upper()
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['benchmark_ticker'] = upper_benchmark
    # Update the widget to show uppercase value
    st.session_state.multi_backtest_active_benchmark = upper_benchmark

def update_use_momentum():
    current_val = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['use_momentum']
    new_val = st.session_state.multi_backtest_active_use_momentum
    
    if current_val != new_val:
        portfolio = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]
        
        if new_val:
            # Enabling momentum - restore saved settings or use defaults
            if 'saved_momentum_settings' in portfolio:
                # Restore previously saved momentum settings
                saved_settings = portfolio['saved_momentum_settings']
                portfolio['momentum_windows'] = saved_settings.get('momentum_windows', [
                    {"lookback": 365, "exclude": 30, "weight": 0.5},
                    {"lookback": 180, "exclude": 30, "weight": 0.3},
                    {"lookback": 120, "exclude": 30, "weight": 0.2},
                ])
                portfolio['momentum_strategy'] = saved_settings.get('momentum_strategy', 'Classic')
                portfolio['negative_momentum_strategy'] = saved_settings.get('negative_momentum_strategy', 'Cash')
                portfolio['calc_beta'] = saved_settings.get('calc_beta', True)
                portfolio['calc_volatility'] = saved_settings.get('calc_volatility', True)
                portfolio['beta_window_days'] = saved_settings.get('beta_window_days', 365)
                portfolio['exclude_days_beta'] = saved_settings.get('exclude_days_beta', 30)
                portfolio['vol_window_days'] = saved_settings.get('vol_window_days', 365)
                portfolio['exclude_days_vol'] = saved_settings.get('exclude_days_vol', 30)
                
                # Update UI widgets to reflect restored values
                st.session_state['multi_backtest_active_momentum_strategy'] = portfolio['momentum_strategy']
                st.session_state['multi_backtest_active_negative_momentum_strategy'] = portfolio['negative_momentum_strategy']
                st.session_state['multi_backtest_active_calc_beta'] = portfolio['calc_beta']
                st.session_state['multi_backtest_active_calc_vol'] = portfolio['calc_volatility']
                st.session_state['multi_backtest_active_beta_window'] = portfolio['beta_window_days']
                st.session_state['multi_backtest_active_beta_exclude'] = portfolio['exclude_days_beta']
                st.session_state['multi_backtest_active_vol_window'] = portfolio['vol_window_days']
                st.session_state['multi_backtest_active_vol_exclude'] = portfolio['exclude_days_vol']
            else:
                # SMART NUCLEAR: No saved settings, create defaults only if no windows exist
                if not portfolio.get('momentum_windows'):
                    portfolio['momentum_windows'] = [
                        {"lookback": 365, "exclude": 30, "weight": 0.5},
                        {"lookback": 180, "exclude": 30, "weight": 0.3},
                        {"lookback": 120, "exclude": 30, "weight": 0.2},
                    ]
                    print("SMART NUCLEAR: Added default momentum windows (had none)")
                else:
                    print(f"SMART NUCLEAR: Preserved existing momentum windows (had {len(portfolio['momentum_windows'])} windows)")
                # Set default momentum settings only if not already set
                portfolio['momentum_strategy'] = portfolio.get('momentum_strategy', 'Classic')
                portfolio['negative_momentum_strategy'] = portfolio.get('negative_momentum_strategy', 'Cash')
                portfolio['calc_beta'] = portfolio.get('calc_beta', True)
                portfolio['calc_volatility'] = portfolio.get('calc_volatility', True)
        else:
            # Disabling momentum - save current settings before clearing
            saved_settings = {
                'momentum_windows': portfolio.get('momentum_windows', []),
                'momentum_strategy': portfolio.get('momentum_strategy', 'Classic'),
                'negative_momentum_strategy': portfolio.get('negative_momentum_strategy', 'Cash'),
                'calc_beta': portfolio.get('calc_beta', True),
                'calc_volatility': portfolio.get('calc_volatility', True),
                'beta_window_days': portfolio.get('beta_window_days', 365),
                'exclude_days_beta': portfolio.get('exclude_days_beta', 30),
                'vol_window_days': portfolio.get('vol_window_days', 365),
                'exclude_days_vol': portfolio.get('exclude_days_vol', 30),
            }
            portfolio['saved_momentum_settings'] = saved_settings
            # Don't clear momentum_windows - preserve them for variant generation
        
        portfolio['use_momentum'] = new_val
        st.session_state.multi_backtest_rerun_flag = True



def update_calc_beta():
    portfolio_index = st.session_state.multi_backtest_active_portfolio_index
    active_portfolio = st.session_state.multi_backtest_portfolio_configs[portfolio_index]
    current_val = active_portfolio.get('calc_beta', False)
    new_val = st.session_state.multi_backtest_active_calc_beta
    
    if current_val != new_val:
        if new_val:
            # Enabling beta - restore saved settings or use defaults
            if 'saved_beta_settings' in active_portfolio:
                # Restore previously saved beta settings
                saved_settings = active_portfolio['saved_beta_settings']
                active_portfolio['beta_window_days'] = saved_settings.get('beta_window_days', 365)
                active_portfolio['exclude_days_beta'] = saved_settings.get('exclude_days_beta', 30)
                
                # Update UI widgets to reflect restored values
                st.session_state['multi_backtest_active_beta_window'] = active_portfolio['beta_window_days']
                st.session_state['multi_backtest_active_beta_exclude'] = active_portfolio['exclude_days_beta']
            else:
                # No saved settings, use current portfolio values or defaults
                beta_window = active_portfolio.get('beta_window_days', 365)
                beta_exclude = active_portfolio.get('exclude_days_beta', 30)
                active_portfolio['beta_window_days'] = beta_window
                active_portfolio['exclude_days_beta'] = beta_exclude
                st.session_state['multi_backtest_active_beta_window'] = beta_window
                st.session_state['multi_backtest_active_beta_exclude'] = beta_exclude
        else:
            # Disabling beta - save current values to BOTH saved settings AND main portfolio
            beta_window = st.session_state.get('multi_backtest_active_beta_window', active_portfolio.get('beta_window_days', 365))
            beta_exclude = st.session_state.get('multi_backtest_active_beta_exclude', active_portfolio.get('exclude_days_beta', 30))
            
            # Save to main portfolio keys (so variants inherit them)
            active_portfolio['beta_window_days'] = beta_window
            active_portfolio['exclude_days_beta'] = beta_exclude
            
            # Also save to saved_settings (for restore later)
            saved_settings = {
                'beta_window_days': beta_window,
                'exclude_days_beta': beta_exclude,
            }
            active_portfolio['saved_beta_settings'] = saved_settings
        
        active_portfolio['calc_beta'] = new_val
        st.session_state.multi_backtest_rerun_flag = True

def update_beta_window():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['beta_window_days'] = st.session_state.multi_backtest_active_beta_window

def update_beta_exclude():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['exclude_days_beta'] = st.session_state.multi_backtest_active_beta_exclude

def update_calc_vol():
    portfolio_index = st.session_state.multi_backtest_active_portfolio_index
    active_portfolio = st.session_state.multi_backtest_portfolio_configs[portfolio_index]
    current_val = active_portfolio.get('calc_volatility', False)
    new_val = st.session_state.multi_backtest_active_calc_vol
    
    if current_val != new_val:
        if new_val:
            # Enabling volatility - restore saved settings or use defaults
            if 'saved_vol_settings' in active_portfolio:
                # Restore previously saved volatility settings
                saved_settings = active_portfolio['saved_vol_settings']
                active_portfolio['vol_window_days'] = saved_settings.get('vol_window_days', 365)
                active_portfolio['exclude_days_vol'] = saved_settings.get('exclude_days_vol', 30)
                
                # Update UI widgets to reflect restored values
                st.session_state['multi_backtest_active_vol_window'] = active_portfolio['vol_window_days']
                st.session_state['multi_backtest_active_vol_exclude'] = active_portfolio['exclude_days_vol']
            else:
                # No saved settings, use current portfolio values or defaults
                vol_window = active_portfolio.get('vol_window_days', 365)
                vol_exclude = active_portfolio.get('exclude_days_vol', 30)
                active_portfolio['vol_window_days'] = vol_window
                active_portfolio['exclude_days_vol'] = vol_exclude
                st.session_state['multi_backtest_active_vol_window'] = vol_window
                st.session_state['multi_backtest_active_vol_exclude'] = vol_exclude
        else:
            # Disabling volatility - save current values to BOTH saved settings AND main portfolio
            vol_window = st.session_state.get('multi_backtest_active_vol_window', active_portfolio.get('vol_window_days', 365))
            vol_exclude = st.session_state.get('multi_backtest_active_vol_exclude', active_portfolio.get('exclude_days_vol', 30))
            
            # Save to main portfolio keys (so variants inherit them)
            active_portfolio['vol_window_days'] = vol_window
            active_portfolio['exclude_days_vol'] = vol_exclude
            
            # Also save to saved_settings (for restore later)
            saved_settings = {
                'vol_window_days': vol_window,
                'exclude_days_vol': vol_exclude,
            }
            active_portfolio['saved_vol_settings'] = saved_settings
        
        active_portfolio['calc_volatility'] = new_val
        st.session_state.multi_backtest_rerun_flag = True

def update_vol_window():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['vol_window_days'] = st.session_state.multi_backtest_active_vol_window

def update_vol_exclude():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['exclude_days_vol'] = st.session_state.multi_backtest_active_vol_exclude

def update_use_threshold():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['use_minimal_threshold'] = st.session_state.multi_backtest_active_use_threshold

def update_threshold_percent():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['minimal_threshold_percent'] = st.session_state.multi_backtest_active_threshold_percent

def update_start_with():
    st.session_state.multi_backtest_start_with = st.session_state.multi_backtest_start_with_radio

def update_first_rebalance_strategy():
    st.session_state.multi_backtest_first_rebalance_strategy = st.session_state.multi_backtest_first_rebalance_strategy_radio

def update_collect_dividends_as_cash():
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['collect_dividends_as_cash'] = st.session_state.multi_backtest_active_collect_dividends_as_cash

def clear_dates_callback():
    """Clear the date inputs and reset to None"""
    st.session_state.multi_backtest_start_date = None
    st.session_state.multi_backtest_end_date = date.today()
    st.session_state.multi_backtest_use_custom_dates = False
    # Also clear from the portfolio config
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['start_date_user'] = None
    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['end_date_user'] = None

def update_sync_exclusion(sync_type):
    """Update sync exclusion settings when checkboxes change"""
    try:
        portfolio = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]
        
        if sync_type == 'cashflow':
            key = f"multi_backtest_exclude_cashflow_sync_{st.session_state.multi_backtest_active_portfolio_index}"
            if key in st.session_state:
                portfolio['exclude_from_cashflow_sync'] = st.session_state[key]
        elif sync_type == 'rebalancing':
            key = f"multi_backtest_exclude_rebalancing_sync_{st.session_state.multi_backtest_active_portfolio_index}"
            if key in st.session_state:
                portfolio['exclude_from_rebalancing_sync'] = st.session_state[key]
        
        # Force immediate update to session state
        st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index] = portfolio
        st.session_state.multi_backtest_rerun_flag = True
    except Exception:
        pass

def parse_date_from_json(date_value):
    """Parse date from JSON string format back to date object"""
    if date_value is None:
        return None
    if isinstance(date_value, date):
        return date_value
    if isinstance(date_value, str):
        try:
            return datetime.strptime(date_value, '%Y-%m-%d').date()
        except ValueError:
            try:
                # Try parsing as ISO format
                return datetime.fromisoformat(date_value).date()
            except ValueError:
                return None
    return None

def sync_date_widgets_with_portfolio():
    """Sync date widgets with current portfolio configuration"""
    from datetime import date
    if st.session_state.multi_backtest_active_portfolio_index is not None:
        portfolio = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]
        
        # Sync start date
        portfolio_start_date = portfolio.get('start_date_user')
        if portfolio_start_date is not None:
            st.session_state["multi_backtest_start_date"] = portfolio_start_date
        else:
            st.session_state["multi_backtest_start_date"] = date(2010, 1, 1)
        
        # Sync end date
        portfolio_end_date = portfolio.get('end_date_user')
        if portfolio_end_date is not None:
            st.session_state["multi_backtest_end_date"] = portfolio_end_date
        else:
            st.session_state["multi_backtest_end_date"] = date.today()
        
        # Sync custom dates checkbox
        has_custom_dates = portfolio_start_date is not None or portfolio_end_date is not None
        st.session_state["multi_backtest_use_custom_dates"] = has_custom_dates

# Sidebar for portfolio selection
st.sidebar.title("Manage Portfolios")
portfolio_names = [cfg['name'] for cfg in st.session_state.multi_backtest_portfolio_configs]

# Ensure the active portfolio index is valid
if (st.session_state.multi_backtest_active_portfolio_index is None or 
    st.session_state.multi_backtest_active_portfolio_index >= len(portfolio_names) or
    st.session_state.multi_backtest_active_portfolio_index < 0):
    st.session_state.multi_backtest_active_portfolio_index = 0 if portfolio_names else None

# Use the current portfolio name as the default selection to make it more reliable
current_portfolio_name = None
if (st.session_state.multi_backtest_active_portfolio_index is not None and 
    st.session_state.multi_backtest_active_portfolio_index < len(portfolio_names)):
    current_portfolio_name = portfolio_names[st.session_state.multi_backtest_active_portfolio_index]

selected_portfolio_name = st.sidebar.selectbox(
    "Select Portfolio",
    options=portfolio_names,
    index=st.session_state.multi_backtest_active_portfolio_index,
    key="multi_backtest_portfolio_selector",
    on_change=update_active_portfolio_index
)

active_portfolio = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]

if st.sidebar.button("Add New Portfolio", on_click=add_portfolio_callback):
    pass

# Individual portfolio removal (original functionality)
if len(st.session_state.multi_backtest_portfolio_configs) > 1:
    if st.sidebar.button("Remove Selected Portfolio", on_click=remove_portfolio_callback):
        pass

# Reset selected portfolio button
if st.sidebar.button("Reset Selected Portfolio", on_click=reset_portfolio_callback):
    pass

# NEW: Enhanced bulk portfolio management dropdown
if len(st.session_state.multi_backtest_portfolio_configs) > 1:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Bulk Portfolio Management")
    
    # Initialize session state for selected portfolios
    if "multi_backtest_portfolio_checkboxes" not in st.session_state:
        st.session_state.multi_backtest_portfolio_checkboxes = {}
    
    # Enhanced dropdown with built-in selection controls
    with st.sidebar.expander("📋 Manage Multiple Portfolios", expanded=False):
        st.caption(f"Total portfolios: {len(portfolio_names)}")
        
        # Create checkboxes for each portfolio
        st.markdown("**Select portfolios to delete:**")
        
        # Quick selection buttons at the top
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("✅ Select All", key="multi_backtest_select_all_portfolios", 
                        help="Select all portfolios for deletion", use_container_width=True):
                for name in portfolio_names:
                    st.session_state.multi_backtest_portfolio_checkboxes[name] = True
                st.rerun()
        
        with col2:
            if st.button("❌ Clear All", key="multi_backtest_clear_all_portfolios", 
                        help="Clear all portfolio selections", use_container_width=True):
                st.session_state.multi_backtest_portfolio_checkboxes = {}
                st.rerun()
        
        with col3:
            if st.button("🔄 Refresh", key="multi_backtest_refresh_selections", 
                        help="Refresh the selection list", use_container_width=True):
                st.rerun()
        
        # Portfolio checkboxes with scrollable container
        st.markdown("---")
        
        # Create a scrollable container for many portfolios
        with st.container():
            # Limit height and add scrollbar for many portfolios
            st.markdown("""
            <style>
            .portfolio-checkboxes {
                max-height: 300px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Portfolio checkboxes with individual callback functions
            for i, portfolio_name in enumerate(portfolio_names):
                # Initialize checkbox state if not exists
                if portfolio_name not in st.session_state.multi_backtest_portfolio_checkboxes:
                    st.session_state.multi_backtest_portfolio_checkboxes[portfolio_name] = False
                
                # Create a unique callback function for each portfolio
                def create_portfolio_callback(portfolio_name):
                    def callback():
                        # Toggle the current state
                        current_state = st.session_state.multi_backtest_portfolio_checkboxes.get(portfolio_name, False)
                        st.session_state.multi_backtest_portfolio_checkboxes[portfolio_name] = not current_state
                    return callback
                
                # Create checkbox for each portfolio with callback
                checkbox_key = f"multi_backtest_portfolio_checkbox_{hash(portfolio_name)}"
                is_checked = st.checkbox(
                    f"🗑️ {portfolio_name}",
                    value=st.session_state.multi_backtest_portfolio_checkboxes[portfolio_name],
                    key=checkbox_key,
                    help=f"Select {portfolio_name} for deletion",
                    on_change=create_portfolio_callback(portfolio_name)
                )
        
        # Get selected portfolios from checkboxes
        selected_portfolios_for_deletion = [
            name for name, checked in st.session_state.multi_backtest_portfolio_checkboxes.items() 
            if checked
        ]
        
        # Show success message if portfolios were deleted
        if "multi_backtest_bulk_delete_success" in st.session_state and st.session_state.multi_backtest_bulk_delete_success:
            st.success(st.session_state.multi_backtest_bulk_delete_success)
            # Clear the success message after showing it
            del st.session_state.multi_backtest_bulk_delete_success
        
        # Show selection summary
        if selected_portfolios_for_deletion:
            st.info(f"📊 Selected: {len(selected_portfolios_for_deletion)} portfolio(s)")
            st.caption(f"Selected: {', '.join(selected_portfolios_for_deletion[:3])}{'...' if len(selected_portfolios_for_deletion) > 3 else ''}")
            
            # Bulk delete button with confirmation
            confirm_deletion = st.checkbox(
                f"🗑️ Confirm deletion of {len(selected_portfolios_for_deletion)} portfolio(s)",
                key="multi_backtest_confirm_bulk_deletion",
                help="Check this box to enable the delete button"
            )
            
            if confirm_deletion:
                if st.button("🚨 DELETE SELECTED PORTFOLIOS", 
                           type="secondary",
                           help=f"Delete {len(selected_portfolios_for_deletion)} selected portfolio(s)",
                           on_click=bulk_delete_portfolios_callback,
                           args=(selected_portfolios_for_deletion,),
                           use_container_width=True):
                    pass
        else:
            st.caption("No portfolios selected for deletion")

# Start with option
st.sidebar.markdown("---")
st.sidebar.subheader("Data Options")
if "multi_backtest_start_with_radio" not in st.session_state:
    st.session_state["multi_backtest_start_with_radio"] = st.session_state.get("multi_backtest_start_with", "all")
st.sidebar.radio(
    "How to handle assets with different start dates?",
    ["all", "oldest"],
    format_func=lambda x: "Start when ALL assets are available" if x == "all" else "Start with OLDEST asset",
    help="""
    **All:** Starts the backtest when all selected assets are available.
    **Oldest:** Starts at the oldest date of any asset and adds assets as they become available.
    """,
    key="multi_backtest_start_with_radio",
    on_change=update_start_with
)

# First rebalance strategy option
if "multi_backtest_first_rebalance_strategy_radio" not in st.session_state:
    st.session_state["multi_backtest_first_rebalance_strategy_radio"] = st.session_state.get("multi_backtest_first_rebalance_strategy", "rebalancing_date")
st.sidebar.radio(
    "When should the first rebalancing occur?",
    ["rebalancing_date", "momentum_window_complete"],
    format_func=lambda x: "First rebalance on rebalancing date" if x == "rebalancing_date" else "First rebalance when momentum window complete",
    help="""
    **First rebalance on rebalancing date:** Start rebalancing immediately when possible.
    **First rebalance when momentum window complete:** Wait for the largest momentum window to complete before first rebalance.
    """,
    key="multi_backtest_first_rebalance_strategy_radio",
    on_change=update_first_rebalance_strategy
)

# Date range options
st.sidebar.markdown("---")
st.sidebar.subheader("Date Range Options")
use_custom_dates = st.sidebar.checkbox("Use custom date range", key="multi_backtest_use_custom_dates", help="Enable to set custom start and end dates for the backtest")

if use_custom_dates:
    col_start_date, col_end_date, col_clear_dates = st.sidebar.columns([1, 1, 1])
    with col_start_date:
        # Initialize widget key with session state value
        if "multi_backtest_start_date" not in st.session_state:
            st.session_state["multi_backtest_start_date"] = date(2010, 1, 1)
        # Let Streamlit manage the session state automatically
        start_date = st.date_input("Start Date", min_value=date(1900, 1, 1), key="multi_backtest_start_date")
        # Update portfolio config when date changes
        if start_date != st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index].get('start_date_user'):
            st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['start_date_user'] = start_date
    
    with col_end_date:
        # Initialize widget key with session state value
        if "multi_backtest_end_date" not in st.session_state:
            st.session_state["multi_backtest_end_date"] = date.today()
        # Let Streamlit manage the session state automatically
        end_date = st.date_input("End Date", key="multi_backtest_end_date")
        # Update portfolio config when date changes
        if end_date != st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index].get('end_date_user'):
            st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['end_date_user'] = end_date
    
    with col_clear_dates:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer for alignment
        st.button("Clear Dates", on_click=clear_dates_callback)
else:
    st.session_state["multi_backtest_start_date"] = None
    st.session_state["multi_backtest_end_date"] = None

st.header(f"Editing Portfolio: {active_portfolio['name']}")
# Ensure session-state key exists before creating widgets to avoid duplicate-default warnings
if "multi_backtest_active_name" not in st.session_state:
    st.session_state["multi_backtest_active_name"] = active_portfolio['name']
active_portfolio['name'] = st.text_input("Portfolio Name", key="multi_backtest_active_name", on_change=update_name)

# Portfolio Variant Generator - Multi-Select with Custom Options
st.markdown("---")  # Add separator

# NUCLEAR APPROACH: Portfolio-specific expander with forced refresh
portfolio_index = st.session_state.multi_backtest_active_portfolio_index

# Store expander state in portfolio config  
if 'variant_expander_expanded' not in active_portfolio:
    active_portfolio['variant_expander_expanded'] = False

# NUCLEAR: Force expander to refresh by clearing its widget state when portfolio changes
last_portfolio_key = "multi_backtest_last_portfolio_for_variants"
if st.session_state.get(last_portfolio_key) != portfolio_index:
    # Portfolio changed - clear all variant-related widget states
    keys_to_clear = [k for k in st.session_state.keys() if 'variant' in k.lower() and 'multi' in k]
    for key in keys_to_clear:
        if key != last_portfolio_key:  # Don't clear the tracker itself
            del st.session_state[key]
    st.session_state[last_portfolio_key] = portfolio_index

# Use the beautiful expander with portfolio state
current_state = active_portfolio.get('variant_expander_expanded', False)

# NUCLEAR: Use a unique key that includes portfolio info to force recreation
unique_expander_key = f"variants_exp_p{portfolio_index}_v{hash(str(active_portfolio.get('name', '')))}"

with st.expander("🔧 Generate Portfolio Variants", expanded=current_state):
    # Show current pin status and provide pin/unpin controls
    col_status, col_pin, col_unpin = st.columns([2, 1, 1])
    
    with col_status:
        if current_state:
            st.info("📌 **Status: EXPANDED & PINNED** for this portfolio")
        else:
            st.info("📌 **Status: COLLAPSED** for this portfolio")
    
    with col_pin:
        if not current_state:
            if st.button("📌 Pin Expanded", key=f"pin_expanded_{portfolio_index}", type="primary"):
                active_portfolio['variant_expander_expanded'] = True
                st.success("✅ Expander state PINNED for this portfolio!")
                st.rerun()
    
    with col_unpin:
        if current_state:
            if st.button("🔓 Unpin", key=f"unpin_expanded_{portfolio_index}", type="secondary"):
                active_portfolio['variant_expander_expanded'] = False
                st.success("🔓 Expander state UNPINNED for this portfolio!")
                st.rerun()

    st.markdown("**Select parameters to vary and customize their values:**")
    
    # Add explanatory text about how it works and naming
    st.info("""
    **📚 How Portfolio Variants Work:**
    
    This tool generates multiple portfolio variants by combining your selected options. Each variant will be a complete copy of your current portfolio with the specified changes.
    
    **🏷️ Portfolio Naming Convention:**
    - **Format**: `Portfolio Name (Rebalancing Frequency - Momentum Strategy : When momentum not all negative and When momentum all negative - Include Beta in weighting - Include Volatility in weighting)`
    - **Examples**:
      - `My Portfolio (Quarterly - Momentum : Classic and Cash - Beta - Volatility)`
      - `My Portfolio (Monthly - Momentum : Relative and Equal Weight - Beta)`
      - `My Portfolio (Quarterly - No Momentum)`
    
    **💡 Tips**: 
    - Select at least one rebalancing frequency
    - If "Use Momentum" is unchecked, momentum options are hidden
    - Beta and Volatility only appear when enabled
    """)

    # Add checkbox to keep current portfolio
    keep_current_portfolio = st.checkbox(
        "✅ Keep Current Portfolio", 
        value=True, 
        key="multi_backtest_keep_current_portfolio",
        help="When checked, the current portfolio (including benchmark) will be kept. When unchecked, only the generated variants will be created."
    )
    
    # Add explanatory note about what happens when unchecked
    if not keep_current_portfolio:
        st.info("⚠️ **Note:** When unchecked, the current portfolio will be **removed** after generating variants. Only the variants will remain in your portfolio list.")
    
    st.markdown("---")  # Add separator before variant parameters

    variant_params = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rebalance Frequency (section title - not a checkbox!)
        st.markdown("**Rebalance Frequency:**")
        rebalance_options = []
        if st.checkbox("Never", key="multi_rebalance_never"):
            rebalance_options.append("Never")
        if st.checkbox("Buy & Hold", key="multi_rebalance_buyhold"):
            rebalance_options.append("Buy & Hold")
        if st.checkbox("Buy & Hold (Target)", key="multi_rebalance_buyhold_target"):
            rebalance_options.append("Buy & Hold (Target)")
        if st.checkbox("Weekly", key="multi_rebalance_weekly"):
            rebalance_options.append("Weekly")
        if st.checkbox("Biweekly", key="multi_rebalance_biweekly"):
            rebalance_options.append("Biweekly")
        if st.checkbox("Monthly", key="multi_rebalance_monthly"):
            rebalance_options.append("Monthly")
        if st.checkbox("Quarterly", value=True, key="multi_rebalance_quarterly"):
            rebalance_options.append("Quarterly")
        if st.checkbox("Semiannually", key="multi_rebalance_semiannually"):
            rebalance_options.append("Semiannually")
        if st.checkbox("Annually", key="multi_rebalance_annually"):
            rebalance_options.append("Annually")
        
        # Validation: At least one rebalance frequency must be selected
        if rebalance_options:
            variant_params["rebalance_frequency"] = rebalance_options
        else:
            st.error("⚠️ **At least one Rebalance Frequency must be selected!**")
    
    with col2:
        # Use Momentum (simple checkbox - just enables momentum options, doesn't create variants)
        # Reset checkbox to default if it doesn't exist in session state (fresh portfolio selection)
        if "multi_use_momentum_vary" not in st.session_state:
            st.session_state["multi_use_momentum_vary"] = False
        use_momentum_vary = st.checkbox("Use Momentum", key="multi_use_momentum_vary")
    
    # Show momentum options ONLY if user checked "Use Momentum" 
    # (regardless of current portfolio's momentum status)
    if use_momentum_vary:
        st.markdown("---")
        col_mom_left, col_mom_right = st.columns(2)
        
        with col_mom_left:
            # Momentum Strategy Section
            st.markdown("**Momentum strategy when NOT all negative:**")
            momentum_options = []
            if st.checkbox("Classic momentum", value=True, key="multi_momentum_classic"):
                momentum_options.append("Classic")
            if st.checkbox("Relative momentum", key="multi_momentum_relative"):
                momentum_options.append("Relative Momentum")
            
            # Validation and storage for momentum strategy
            if momentum_options:
                variant_params["momentum_strategy"] = momentum_options
            else:
                st.error("⚠️ **At least one momentum strategy must be selected!**")
            
            st.markdown("---")
            
            # Negative Strategy Section  
            st.markdown("**Strategy when ALL momentum scores are negative:**")
            negative_options = []
            if st.checkbox("Cash", value=True, key="multi_negative_cash"):
                negative_options.append("Cash")
            if st.checkbox("Equal weight", key="multi_negative_equal"):
                negative_options.append("Equal weight")
            if st.checkbox("Relative momentum", key="multi_negative_relative"):
                negative_options.append("Relative momentum")
            
            # Validation and storage for negative strategy
            if negative_options:
                variant_params["negative_strategy"] = negative_options
            else:
                st.error("⚠️ **At least one negative strategy must be selected!**")
        
        with col_mom_right:
            # Beta in momentum weighting (section title - not a checkbox!)
            st.markdown("**Include Beta in momentum weighting:**")
            beta_options = []
            if st.checkbox("With Beta", value=True, key="multi_beta_yes"):
                beta_options.append(True)
            if st.checkbox("Without Beta", key="multi_beta_no"):
                beta_options.append(False)
            
            # Validation: At least one beta option must be selected
            if beta_options:
                variant_params["include_beta"] = beta_options
            else:
                st.error("⚠️ **At least one Beta option must be selected!**")
            
            st.markdown("---")
            
            # Volatility in momentum weighting (section title - not a checkbox!)
            st.markdown("**Include Volatility in momentum weighting:**")
            vol_options = []
            if st.checkbox("With Volatility", value=True, key="multi_vol_yes"):
                vol_options.append(True)
            if st.checkbox("Without Volatility", key="multi_vol_no"):
                vol_options.append(False)
            
            # Validation: At least one volatility option must be selected
            if vol_options:
                variant_params["include_volatility"] = vol_options
            else:
                st.error("⚠️ **At least one Volatility option must be selected!**")
    else:
        st.info("💡 **Momentum-related options** (Momentum Strategy, Negative Strategy, Beta, Volatility) are only available when momentum is enabled in the current portfolio or when varying 'Use Momentum' to include enabled variants.")
    
    # Minimal Threshold Filter Section - Only show when momentum is enabled
    if use_momentum_vary:
        st.markdown("---")
        st.markdown("**Minimal Threshold Filter:**")
        
        # Initialize session state for threshold filters if not exists
        if f"threshold_filters_{portfolio_index}" not in st.session_state:
            st.session_state[f"threshold_filters_{portfolio_index}"] = []
        
        # Checkboxes for enable/disable
        col_thresh_left, col_thresh_right = st.columns(2)
        
        with col_thresh_left:
            disable_threshold = st.checkbox(
                "Disable Minimal Threshold Filter", 
                value=True, 
                key=f"disable_threshold_{portfolio_index}",
                help="Keeps the minimal threshold filter disabled"
            )
        
        with col_thresh_right:
            enable_threshold = st.checkbox(
                "Enable Minimal Threshold Filter", 
                value=False, 
                key=f"enable_threshold_{portfolio_index}",
                help="Enables the minimal threshold filter with customizable values"
            )
        
        # Validation: At least one must be selected
        if not disable_threshold and not enable_threshold:
            st.error("⚠️ **At least one Minimal Threshold Filter option must be selected!**")
        
        # Build threshold options list
        threshold_options = []
        
        # If disable is selected, add None to options
        if disable_threshold:
            threshold_options.append(None)
        
        # If enable is selected, show threshold input options
        if enable_threshold:
            st.markdown("**Threshold Values:**")
            
            # Add new threshold button
            if st.button("➕ Add Threshold Value", key=f"add_threshold_{portfolio_index}"):
                st.session_state[f"threshold_filters_{portfolio_index}"].append(2.0)
                st.rerun()
            
            # Display existing threshold inputs
            for i, threshold in enumerate(st.session_state[f"threshold_filters_{portfolio_index}"]):
                col_input, col_remove = st.columns([3, 1])
                
                with col_input:
                    threshold_value = st.number_input(
                        f"Threshold {i+1} (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=threshold,
                        step=0.1,
                        format="%.2f",
                        key=f"threshold_input_{portfolio_index}_{i}",
                        help="Minimum threshold percentage for portfolio allocation"
                    )
                    threshold_options.append(threshold_value)
                
                with col_remove:
                    if st.button("🗑️", key=f"remove_threshold_{portfolio_index}_{i}", help="Remove this threshold"):
                        st.session_state[f"threshold_filters_{portfolio_index}"].pop(i)
                        st.rerun()
            
            # If no thresholds exist, add a default one
            if not st.session_state[f"threshold_filters_{portfolio_index}"]:
                st.session_state[f"threshold_filters_{portfolio_index}"].append(2.0)
                st.rerun()
        
        # Store threshold options in variant params if any are selected
        if threshold_options:
            variant_params["minimal_threshold"] = threshold_options
        elif not disable_threshold and not enable_threshold:
            st.error("⚠️ **At least one threshold value must be provided when Enable is selected!**")
    else:
        # When momentum is not enabled, add None as default
        variant_params["minimal_threshold"] = [None]
    
    # Calculate total combinations
    total_variants = 1
    for param_values in variant_params.values():
        total_variants *= len(param_values)
    
    if variant_params:
        st.info(f"🎯 **{total_variants} variants** will be generated")
        
        # Validation: Check for required parameters
        validation_errors = []
        
        # Check if rebalance frequency is missing (always required)
        if "rebalance_frequency" not in variant_params:
            validation_errors.append("⚠️ Select at least one **Rebalance Frequency**")
        
        # If momentum is enabled (user checked "Use Momentum"), we need ALL momentum parameters
        if use_momentum_vary:
            # Check if momentum strategies are missing (they're always required when momentum enabled)
            if "momentum_strategy" not in variant_params:
                validation_errors.append("⚠️ Select at least one **Momentum Strategy** when momentum is enabled")
            
            # Check if negative strategies are missing (they're always required when momentum enabled)
            if "negative_strategy" not in variant_params:
                validation_errors.append("⚠️ Select at least one **Negative Strategy** when momentum is enabled")
            
            # Check if beta options are missing (they're always required when momentum enabled)
            if "include_beta" not in variant_params:
                validation_errors.append("⚠️ Select at least one **Beta option** when momentum is enabled")
            
            # Check if volatility options are missing (they're always required when momentum enabled)
            if "include_volatility" not in variant_params:
                validation_errors.append("⚠️ Select at least one **Volatility option** when momentum is enabled")
        
        # Check if minimal threshold filter is missing (only required when momentum is enabled)
        if use_momentum_vary and "minimal_threshold" not in variant_params:
            validation_errors.append("⚠️ Select at least one **Minimal Threshold Filter** option when momentum is enabled")
        
        # Show validation errors
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            st.warning("🚫 **Cannot generate variants** - Fix the errors above first")
        else:
            # All validations passed - show generate button
            if st.button(f"✨ Generate {total_variants} Portfolio Variants", type="primary"):
                # Define the function locally to avoid import issues
                def generate_portfolio_variants(base_portfolio, variant_params, base_name):
                    """
                    Generate multiple portfolio variants based on the base portfolio and variant parameters.
                    
                    Args:
                        base_portfolio (dict): The base portfolio configuration
                        variant_params (dict): Dictionary containing variant parameters and their possible values
                        base_name (str): The base name to use for variant naming
                        
                    Returns:
                        list: List of portfolio variant configurations
                    """
                    variants = []
                    
                    # Get all possible values for each parameter
                    param_values = {}
                    for param, values in variant_params.items():
                        if isinstance(values, list):
                            param_values[param] = values
                        else:
                            param_values[param] = [values]
                    
                    # Generate all combinations
                    from itertools import product
                    
                    # Get the parameter names and their possible values
                    param_names = list(param_values.keys())
                    param_value_lists = [param_values[param] for param in param_names]
                    
                    # Generate all combinations
                    combinations = list(product(*param_value_lists))
                    
                    # Create a variant for each combination
                    for i, combination in enumerate(combinations):
                        # Create a deep copy of the base portfolio
                        variant = base_portfolio.copy()
                        
                        # Update the variant with the new parameter values
                        for j, param in enumerate(param_names):
                            value = combination[j]
                            
                            # Map UI parameter names to actual portfolio configuration fields
                            if param == "rebalance_frequency":
                                variant["rebalancing_frequency"] = value
                            elif param == "momentum_strategy":
                                variant["momentum_strategy"] = value
                            elif param == "negative_strategy":
                                variant["negative_momentum_strategy"] = value
                            elif param == "include_beta":
                                variant["calc_beta"] = value
                            elif param == "include_volatility":
                                variant["calc_volatility"] = value
                            elif param == "minimal_threshold":
                                if value is not None:
                                    variant["use_minimal_threshold"] = True
                                    variant["minimal_threshold_percent"] = value
                                else:
                                    variant["use_minimal_threshold"] = False
                                    variant["minimal_threshold_percent"] = 2.0  # Default value
                            else:
                                # For any other parameters, use the original name
                                variant[param] = value
                        
                        # Generate a unique name for the variant
                        variant_name_parts = []
                        for param in param_names:
                            if param in variant:
                                value = variant[param]
                                if isinstance(value, bool):
                                    variant_name_parts.append(f"{param}_{'ON' if value else 'OFF'}")
                                elif isinstance(value, (int, float)):
                                    variant_name_parts.append(f"{param}_{value}")
                                else:
                                    variant_name_parts.append(f"{param}_{str(value)}")
                        
                        # Create variant name
                        if variant_name_parts:
                            variant['name'] = f"{base_portfolio.get('name', 'Portfolio')}_Variant_{i+1}_{'_'.join(variant_name_parts)}"
                        else:
                            variant['name'] = f"{base_portfolio.get('name', 'Portfolio')}_Variant_{i+1}"
                        
                        # Ensure unique name by adding suffix if needed
                        base_name = variant['name']
                        counter = 1
                        while any(v.get('name') == variant['name'] for v in variants):
                            variant['name'] = f"{base_name}_{counter}"
                            counter += 1
                        
                        variants.append(variant)
                    
                    return variants
                
                import copy
                
                base_portfolio = copy.deepcopy(active_portfolio)
                base_name = base_portfolio['name']
                
                # SMART NUCLEAR: Handle momentum based on "Use Momentum" checkbox
                if use_momentum_vary:
                    base_portfolio['use_momentum'] = True
                    # Only add default momentum windows if base portfolio had none
                    if not base_portfolio.get('momentum_windows'):
                        base_portfolio['momentum_windows'] = [
                            {"lookback": 365, "exclude": 30, "weight": 0.5},
                            {"lookback": 180, "exclude": 30, "weight": 0.3},
                            {"lookback": 120, "exclude": 30, "weight": 0.2},
                        ]
                        base_portfolio['momentum_strategy'] = base_portfolio.get('momentum_strategy', 'Classic')
                        base_portfolio['negative_momentum_strategy'] = base_portfolio.get('negative_momentum_strategy', 'Cash')
                        base_portfolio['calc_beta'] = base_portfolio.get('calc_beta', True)
                        base_portfolio['calc_volatility'] = base_portfolio.get('calc_volatility', True)
                        print("SMART NUCLEAR: Added default momentum settings to base portfolio (had none)")
                    else:
                        print(f"SMART NUCLEAR: Preserved existing momentum windows on base portfolio (had {len(base_portfolio['momentum_windows'])} windows)")
                else:
                    # User unchecked "Use Momentum" - disable momentum for variants
                    base_portfolio['use_momentum'] = False
                    print("SMART NUCLEAR: Disabled momentum for variants (Use Momentum unchecked)")
                
                variants = generate_portfolio_variants(base_portfolio, variant_params, base_name)
                
                # CUSTOM NAMING: Override the generated names with clearer, more readable names
                for variant in variants:
                    # Create a much clearer name format
                    clear_name_parts = []
                    
                    # Rebalancing frequency (compact - just the frequency word)
                    if 'rebalancing_frequency' in variant:
                        freq = variant['rebalancing_frequency']
                        clear_name_parts.append(freq)  # Just "Quarterly", "Monthly", etc.
                    
                    # Add dash separator
                    clear_name_parts.append("-")
                    
                    # Momentum status and strategy
                    if variant.get('use_momentum', False):
                        clear_name_parts.append("Momentum :")
                        
                        # Momentum strategy
                        if variant.get('momentum_strategy') == 'Classic':
                            clear_name_parts.append("Classic")
                        elif variant.get('momentum_strategy') == 'Relative Momentum':
                            clear_name_parts.append("Relative")
                        
                        # Negative strategy with "and" connector
                        if variant.get('negative_momentum_strategy') == 'Cash':
                            clear_name_parts.append("and Cash")
                        elif variant.get('negative_momentum_strategy') == 'Equal weight':
                            clear_name_parts.append("and Equal Weight")
                        elif variant.get('negative_momentum_strategy') == 'Relative momentum':
                            clear_name_parts.append("and Relative")
                        
                        # Beta and Volatility (only show when True, omit when False)
                        if variant.get('calc_beta', False):
                            clear_name_parts.append("- Beta")
                        if variant.get('calc_volatility', False):
                            clear_name_parts.append("- Volatility")
                    else:
                        clear_name_parts.append("No Momentum")
                        # Stop here - no beta/volatility for non-momentum portfolios
                    
                    # Add threshold information (only show when enabled)
                    if variant.get('use_minimal_threshold', False):
                        threshold_percent = variant.get('minimal_threshold_percent', 2.0)
                        clear_name_parts.append(f"- Min {threshold_percent:.2f}%")
                    
                    # Create the new clear name
                    clear_name = f"{base_name} ({' '.join(clear_name_parts)})"
                    variant['name'] = clear_name
                
                # Handle current portfolio based on user choice - use exact same logic as Remove Selected Portfolio
                if not keep_current_portfolio:
                    if len(st.session_state.multi_backtest_portfolio_configs) > 1:
                        # Use exact same logic as remove_portfolio_callback
                        st.session_state.multi_backtest_portfolio_configs.pop(st.session_state.multi_backtest_active_portfolio_index)
                        st.session_state.multi_backtest_active_portfolio_index = max(0, st.session_state.multi_backtest_active_portfolio_index - 1)
                        
                        # CRITICAL: Force a proper portfolio switch to update all UI widgets
                        # This ensures the portfolio name text box and other widgets show the new portfolio's data
                        st.session_state.multi_backtest_rerun_flag = True
                        
                        st.success("🗑️ Removed original portfolio - Active portfolio updated")
                    else:
                        # Only one portfolio - can't remove it
                        st.warning("⚠️ Cannot remove the only portfolio. Keeping original portfolio.")
                        keep_current_portfolio = True
                
                # Add variants to portfolio list with unique names
                for variant in variants:
                    # Use central function - automatically ensures unique name
                    add_portfolio_to_configs(variant)
                
                # Show appropriate success message based on user choice
                if keep_current_portfolio:
                    st.success(f"🎉 Generated {len(variants)} variants of '{base_name}'! Original portfolio kept.")
                    st.info(f"📊 Total portfolios: {len(st.session_state.multi_backtest_portfolio_configs)}")
                else:
                    st.success(f"🎉 Generated {len(variants)} variants of '{base_name}'! Original portfolio removed.")
                    st.info(f"📊 Total portfolios: {len(st.session_state.multi_backtest_portfolio_configs)}")
                
                st.rerun()
    else:
        st.warning("⚠️ Select at least one parameter to vary")

col_left, col_right = st.columns([1, 1])
with col_left:
    if "multi_backtest_active_initial" not in st.session_state:
        st.session_state["multi_backtest_active_initial"] = int(active_portfolio['initial_value'])
    st.number_input("Initial Value ($)", min_value=0, step=1000, format="%d", key="multi_backtest_active_initial", on_change=update_initial, help="Starting cash", )
with col_right:
    if "multi_backtest_active_added_amount" not in st.session_state:
        st.session_state["multi_backtest_active_added_amount"] = int(active_portfolio['added_amount'])
    st.number_input("Added Amount ($)", min_value=0, step=1000, format="%d", key="multi_backtest_active_added_amount", on_change=update_added_amount, help="Amount added at each Added Frequency")

# Swap positions: show Rebalancing Frequency first, then Added Frequency.
# Use two equal-width columns and make selectboxes use the container width so they match visually.
col_freq_rebal, col_freq_add = st.columns([1, 1])
freq_options = ["Never", "Buy & Hold", "Buy & Hold (Target)", "Weekly", "Biweekly", "Monthly", "Quarterly", "Semiannually", "Annually"]
with col_freq_rebal:
    if "multi_backtest_active_rebal_freq" not in st.session_state:
        st.session_state["multi_backtest_active_rebal_freq"] = active_portfolio['rebalancing_frequency']
    st.selectbox("Rebalancing Frequency", freq_options, key="multi_backtest_active_rebal_freq", on_change=update_rebal_freq, help="How often the portfolio is rebalanced. 'Buy & Hold' reinvests cash immediately using current proportions. 'Buy & Hold (Target)' reinvests cash immediately using target allocations. Cash from dividends (if 'Collect Dividends as Cash' is enabled) will be available for rebalancing.", )
with col_freq_add:
    if "multi_backtest_active_add_freq" not in st.session_state:
        st.session_state["multi_backtest_active_add_freq"] = active_portfolio['added_frequency']
    st.selectbox("Added Frequency", freq_options, key="multi_backtest_active_add_freq", on_change=update_add_freq, help="How often cash is added to the portfolio. 'Buy & Hold' reinvests cash immediately using current proportions. 'Buy & Hold (Target)' reinvests cash immediately using target allocations.")

# Dividend handling option
st.session_state["multi_backtest_active_collect_dividends_as_cash"] = active_portfolio.get('collect_dividends_as_cash', False)
st.checkbox(
    "Collect Dividends as Cash", 
    key="multi_backtest_active_collect_dividends_as_cash",
    help="When enabled, dividends are collected as cash instead of being automatically reinvested in the stock. This cash will be available for rebalancing.",
    on_change=update_collect_dividends_as_cash
)

with st.expander("Rebalancing and Added Frequency Explained", expanded=False):
    st.markdown("""
    **Added Frequency** is the frequency at which cash is added to the portfolio.
    
    **Rebalancing Frequency** is the frequency at which the portfolio is rebalanced to the specified allocations. It is also at this date that any additional cash from the `Added Frequency` is invested into the portfolio.
    
    **Buy & Hold Options:**
    - **Buy & Hold**: When cash is available (from additions or dividends), it's immediately reinvested using the current portfolio proportions
    - **Buy & Hold (Target)**: When cash is available (from additions or dividends), it's immediately reinvested using the target allocations
    
    **Collect Dividends as Cash**: When enabled, dividends are collected as cash instead of being automatically reinvested. This cash becomes available for rebalancing.
    
    *Keeping a Rebalancing Frequency to "Never" will mean no additional cash is invested, even if you have an `Added Frequency` specified.*
    """)

# Sync buttons
if len(st.session_state.multi_backtest_portfolio_configs) > 1:
    if st.button("Sync ALL Portfolios Cashflow from First Portfolio", on_click=sync_cashflow_from_first_portfolio_callback, use_container_width=True):
        pass
    if st.button("Sync ALL Portfolios Rebalancing Frequency from First Portfolio", on_click=sync_rebalancing_from_first_portfolio_callback, use_container_width=True):
        pass
    
    # Display sync messages locally below the buttons
    if 'multi_backtest_cashflow_sync_message' in st.session_state and st.session_state['multi_backtest_cashflow_sync_message']:
        message = st.session_state['multi_backtest_cashflow_sync_message']
        message_type = st.session_state.get('multi_backtest_cashflow_sync_message_type', 'info')
        
        if message_type == 'success':
            st.success(message)
        elif message_type == 'error':
            st.error(message)
        else:
            st.info(message)
        
        # Clear the message after displaying it
        del st.session_state['multi_backtest_cashflow_sync_message']
        del st.session_state['multi_backtest_cashflow_sync_message_type']
    
    if 'multi_backtest_rebalancing_sync_message' in st.session_state and st.session_state['multi_backtest_rebalancing_sync_message']:
        message = st.session_state['multi_backtest_rebalancing_sync_message']
        message_type = st.session_state.get('multi_backtest_rebalancing_sync_message_type', 'info')
        
        if message_type == 'success':
            st.success(message)
        elif message_type == 'error':
            st.error(message)
        else:
            st.info(message)
        
        # Clear the message after displaying it
        del st.session_state['multi_backtest_rebalancing_sync_message']
        del st.session_state['multi_backtest_rebalancing_sync_message_type']

# Sync exclusion options (only show if there are multiple portfolios and not for the first portfolio)
if len(st.session_state.multi_backtest_portfolio_configs) > 1 and st.session_state.multi_backtest_active_portfolio_index > 0:
    st.markdown("**🔄 Sync Exclusion Options:**")
    col_sync1, col_sync2 = st.columns(2)
    
    with col_sync1:
        # Initialize sync exclusion settings if not present (but preserve imported values)
        if 'exclude_from_cashflow_sync' not in active_portfolio:
            active_portfolio['exclude_from_cashflow_sync'] = False
        if 'exclude_from_rebalancing_sync' not in active_portfolio:
            active_portfolio['exclude_from_rebalancing_sync'] = False
        
        # Rebalancing sync exclusion - use direct portfolio value to avoid caching issues
        exclude_rebalancing = st.checkbox(
            "Exclude from Rebalancing Sync", 
            value=active_portfolio['exclude_from_rebalancing_sync'],
            key=f"multi_backtest_exclude_rebalancing_sync_{st.session_state.multi_backtest_active_portfolio_index}",
            help="When checked, this portfolio will not be affected by 'Sync ALL Portfolios Rebalancing' button",
            on_change=lambda: update_sync_exclusion('rebalancing')
        )
        
        # Update portfolio config when checkbox changes
        if exclude_rebalancing != active_portfolio['exclude_from_rebalancing_sync']:
            active_portfolio['exclude_from_rebalancing_sync'] = exclude_rebalancing
            # Force immediate update to session state
            st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index] = active_portfolio
            st.session_state.multi_backtest_rerun_flag = True
    
    with col_sync2:
        # Cash flow sync exclusion - use direct portfolio value to avoid caching issues
        exclude_cashflow = st.checkbox(
            "Exclude from Cash Flow Sync", 
            value=active_portfolio['exclude_from_cashflow_sync'],
            key=f"multi_backtest_exclude_cashflow_sync_{st.session_state.multi_backtest_active_portfolio_index}",
            help="When checked, this portfolio will not be affected by 'Sync ALL Portfolios Cashflow' button",
            on_change=lambda: update_sync_exclusion('cashflow')
        )
        
        # Update portfolio config when checkbox changes
        if exclude_cashflow != active_portfolio['exclude_from_cashflow_sync']:
            active_portfolio['exclude_from_cashflow_sync'] = exclude_cashflow
            # Force immediate update to session state
            st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index] = active_portfolio
            st.session_state.multi_backtest_rerun_flag = True

if "multi_backtest_active_benchmark" not in st.session_state:
    st.session_state["multi_backtest_active_benchmark"] = active_portfolio['benchmark_ticker']
st.text_input("Benchmark Ticker (default: ^GSPC, used for beta calculation)", key="multi_backtest_active_benchmark", on_change=update_benchmark)

st.subheader("Stocks")
col_stock_buttons = st.columns([0.3, 0.3, 0.3, 0.1])
with col_stock_buttons[0]:
    if st.button("Normalize Tickers %", on_click=normalize_stock_allocations_callback, use_container_width=True):
        pass
with col_stock_buttons[1]:
    if st.button("Equal Allocation %", on_click=equal_stock_allocation_callback, use_container_width=True):
        pass
with col_stock_buttons[2]:
    if st.button("Reset Tickers", on_click=reset_stock_selection_callback, use_container_width=True):
        pass

# Calculate live total ticker allocation
valid_tickers = [s for s in st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks'] if s['ticker']]
total_ticker_allocation = sum(s['allocation'] for s in valid_tickers)

use_mom_flag = st.session_state.get('multi_backtest_active_use_momentum', active_portfolio.get('use_momentum', True))
if use_mom_flag:
    st.info("Ticker allocations are not used directly for Momentum strategies.")
else:
    if abs(total_ticker_allocation - 1.0) > 0.001:
        st.warning(f"Total ticker allocation is {total_ticker_allocation*100:.2f}%, not 100%. Click 'Normalize' to fix.")
    else:
        st.success(f"Total ticker allocation is {total_ticker_allocation*100:.2f}%.")

def update_stock_allocation(index):
    try:
        key = f"multi_backtest_alloc_input_{st.session_state.multi_backtest_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks'][index]['allocation'] = float(val) / 100.0
    except Exception:
        # Ignore transient errors (e.g., active_portfolio_index changed); UI will reflect state on next render
        return


def update_stock_ticker(index):
    try:
        key = f"multi_backtest_ticker_{st.session_state.multi_backtest_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            # key not yet initialized (race condition). Skip update; the widget's key will be present on next rerender.
            return
        
        # Convert the input value to uppercase
        upper_val = val.upper()

        # Update the portfolio configuration with the uppercase value
        st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks'][index]['ticker'] = upper_val
        
        # Update the text box's state to show the uppercase value
        st.session_state[key] = upper_val
    except Exception:
        # Defensive: if portfolio index or structure changed, skip silently
        return


def update_stock_dividends(index):
    try:
        key = f"multi_backtest_div_{st.session_state.multi_backtest_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks'][index]['include_dividends'] = bool(val)
    except Exception:
        return

# Update active_portfolio
active_portfolio = st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]

for i in range(len(active_portfolio['stocks'])):
    stock = active_portfolio['stocks'][i]
    col_t, col_a, col_d, col_b = st.columns([0.2, 0.2, 0.3, 0.15])
    with col_t:
        ticker_key = f"multi_backtest_ticker_{st.session_state.multi_backtest_active_portfolio_index}_{i}"
        if ticker_key not in st.session_state:
            st.session_state[ticker_key] = stock['ticker']
        st.text_input("Ticker", key=ticker_key, label_visibility="visible", on_change=update_stock_ticker, args=(i,))
    with col_a:
        use_mom = st.session_state.get('multi_backtest_active_use_momentum', active_portfolio.get('use_momentum', True))
        if not use_mom:
            alloc_key = f"multi_backtest_alloc_input_{st.session_state.multi_backtest_active_portfolio_index}_{i}"
            if alloc_key not in st.session_state:
                st.session_state[alloc_key] = int(stock['allocation'] * 100)
            st.number_input("Allocation %", min_value=0, step=1, format="%d", key=alloc_key, label_visibility="visible", on_change=update_stock_allocation, args=(i,))
            if st.session_state[alloc_key] != int(stock['allocation'] * 100):
                st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks'][i]['allocation'] = st.session_state[alloc_key] / 100.0
        else:
            st.write("")
    with col_d:
        div_key = f"multi_backtest_div_{st.session_state.multi_backtest_active_portfolio_index}_{i}"
        if div_key not in st.session_state:
            st.session_state[div_key] = stock['include_dividends']
        st.checkbox("Include Dividends", key=div_key)
        if st.session_state[div_key] != stock['include_dividends']:
            st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['stocks'][i]['include_dividends'] = st.session_state[div_key]
    with col_b:
        st.write("")
        if st.button("Remove", key=f"multi_backtest_rem_stock_{st.session_state.multi_backtest_active_portfolio_index}_{i}_{stock['ticker']}_{id(stock)}", on_click=remove_stock_callback, args=(stock['ticker'],)):
            pass

if st.button("Add Ticker", on_click=add_stock_callback):
    pass

# Bulk ticker input section - FIXED VERSION
with st.expander("📝 Bulk Ticker Input", expanded=False):
    st.markdown("**Enter multiple tickers separated by spaces or commas:**")
    
    # Initialize bulk ticker input in session state
    if 'multi_backtest_bulk_tickers' not in st.session_state:
        st.session_state.multi_backtest_bulk_tickers = ""
    
    # Auto-populate bulk ticker input with current tickers
    portfolio_index = st.session_state.multi_backtest_active_portfolio_index
    current_tickers = [stock['ticker'] for stock in st.session_state.multi_backtest_portfolio_configs[portfolio_index]['stocks'] if stock['ticker']]
    if current_tickers:
        current_ticker_string = ' '.join(current_tickers)
        if st.session_state.multi_backtest_bulk_tickers != current_ticker_string:
            st.session_state.multi_backtest_bulk_tickers = current_ticker_string
    
    # Text area for bulk ticker input
    bulk_tickers = st.text_area(
        "Tickers (e.g., SPY QQQ GLD TLT or SPY,QQQ,GLD,TLT)",
        value=st.session_state.multi_backtest_bulk_tickers,
        key="multi_backtest_bulk_ticker_input",
        height=100,
        help="Enter ticker symbols separated by spaces or commas. Click 'Fill Tickers' to replace tickers (keeps existing allocations)."
    )
    
    if st.button("Fill Tickers", key="multi_backtest_fill_tickers_btn"):
        if bulk_tickers.strip():
            # Parse tickers (split by comma or space)
            ticker_list = []
            for ticker in bulk_tickers.replace(',', ' ').split():
                ticker = ticker.strip().upper()
                if ticker:
                    ticker_list.append(ticker)
            
            if ticker_list:
                portfolio_index = st.session_state.multi_backtest_active_portfolio_index
                current_stocks = st.session_state.multi_backtest_portfolio_configs[portfolio_index]['stocks'].copy()
                
                # Replace tickers - new ones get 0% allocation
                new_stocks = []
                
                for i, ticker in enumerate(ticker_list):
                    if i < len(current_stocks):
                        # Use existing allocation if available
                        new_stocks.append({
                            'ticker': ticker,
                            'allocation': current_stocks[i]['allocation'],
                            'include_dividends': current_stocks[i]['include_dividends']
                        })
                    else:
                        # New tickers get 0% allocation
                        new_stocks.append({
                            'ticker': ticker,
                            'allocation': 0.0,
                            'include_dividends': True
                        })
                
                # Update the portfolio with new stocks
                st.session_state.multi_backtest_portfolio_configs[portfolio_index]['stocks'] = new_stocks
                
                # Update the active_portfolio reference to match session state
                active_portfolio['stocks'] = new_stocks
                
                # Clear any existing session state keys for individual ticker inputs to force refresh
                for key in list(st.session_state.keys()):
                    if key.startswith(f"multi_backtest_ticker_{portfolio_index}_") or key.startswith(f"multi_backtest_alloc_{portfolio_index}_"):
                        del st.session_state[key]
                
                st.success(f"✅ Replaced tickers with: {', '.join(ticker_list)}")
                st.info("💡 **Note:** Existing allocations preserved. Adjust allocations manually if needed.")
                
                # Force immediate rerun
                st.rerun()
            else:
                st.error("❌ No valid tickers found. Please enter ticker symbols separated by spaces or commas.")
        else:
            st.error("❌ Please enter ticker symbols.")


st.subheader("Strategy")
if "multi_backtest_active_use_momentum" not in st.session_state:
    st.session_state["multi_backtest_active_use_momentum"] = active_portfolio['use_momentum']
st.checkbox("Use Momentum Strategy", key="multi_backtest_active_use_momentum", on_change=update_use_momentum, help="Enables momentum-based weighting of stocks.")

if st.session_state.get('multi_backtest_active_use_momentum', active_portfolio.get('use_momentum', True)):
    st.markdown("---")
    col_mom_options, col_beta_vol = st.columns(2)
    with col_mom_options:
        st.markdown("**Momentum Strategy Options**")
        momentum_strategy = st.selectbox(
            "Momentum strategy when NOT all negative:",
            ["Classic", "Relative Momentum"],
            index=["Classic", "Relative Momentum"].index(active_portfolio.get('momentum_strategy', 'Classic')),
            key=f"multi_backtest_momentum_strategy_{st.session_state.multi_backtest_active_portfolio_index}"
        )
        negative_momentum_strategy = st.selectbox(
            "Strategy when ALL momentum scores are negative:",
            ["Cash", "Equal weight", "Relative momentum"],
            index=["Cash", "Equal weight", "Relative momentum"].index(active_portfolio.get('negative_momentum_strategy', 'Cash')),
            key=f"multi_backtest_negative_momentum_strategy_{st.session_state.multi_backtest_active_portfolio_index}"
        )
        active_portfolio['momentum_strategy'] = momentum_strategy
        active_portfolio['negative_momentum_strategy'] = negative_momentum_strategy
        st.markdown("💡 **Note:** These options control how weights are assigned based on momentum scores.")

    with col_beta_vol:
        if "multi_backtest_active_calc_beta" not in st.session_state:
            st.session_state["multi_backtest_active_calc_beta"] = active_portfolio['calc_beta']
        st.checkbox("Include Beta in momentum weighting", key="multi_backtest_active_calc_beta", on_change=update_calc_beta, help="Incorporates a stock's Beta (volatility relative to the benchmark) into its momentum score.")
        # Reset Beta button
        if st.button("Reset Beta", key=f"multi_backtest_reset_beta_btn_{st.session_state.multi_backtest_active_portfolio_index}", on_click=reset_beta_callback):
            pass
        if st.session_state.get('multi_backtest_active_calc_beta', False):
            # Always ensure widgets have the correct values when beta is enabled
            # Check for saved settings first, then use portfolio values, then defaults
            if 'saved_beta_settings' in active_portfolio:
                saved_settings = active_portfolio['saved_beta_settings']
                st.session_state["multi_backtest_active_beta_window"] = saved_settings.get('beta_window_days', 365)
                st.session_state["multi_backtest_active_beta_exclude"] = saved_settings.get('exclude_days_beta', 30)
            else:
                st.session_state["multi_backtest_active_beta_window"] = active_portfolio.get('beta_window_days', 365)
                st.session_state["multi_backtest_active_beta_exclude"] = active_portfolio.get('exclude_days_beta', 30)
            st.number_input("Beta Lookback (days)", min_value=1, key="multi_backtest_active_beta_window", on_change=update_beta_window)
            st.number_input("Beta Exclude (days)", min_value=0, key="multi_backtest_active_beta_exclude", on_change=update_beta_exclude)
        if "multi_backtest_active_calc_vol" not in st.session_state:
            st.session_state["multi_backtest_active_calc_vol"] = active_portfolio['calc_volatility']
        st.checkbox("Include Volatility in momentum weighting", key="multi_backtest_active_calc_vol", on_change=update_calc_vol, help="Incorporates a stock's volatility (standard deviation of returns) into its momentum score.")
        # Reset Volatility button
        if st.button("Reset Volatility", key=f"multi_backtest_reset_vol_btn_{st.session_state.multi_backtest_active_portfolio_index}", on_click=reset_vol_callback):
            pass
        if st.session_state.get('multi_backtest_active_calc_vol', False):
            # Always ensure widgets have the correct values when volatility is enabled
            # Check for saved settings first, then use portfolio values, then defaults
            if 'saved_vol_settings' in active_portfolio:
                saved_settings = active_portfolio['saved_vol_settings']
                st.session_state["multi_backtest_active_vol_window"] = saved_settings.get('vol_window_days', 365)
                st.session_state["multi_backtest_active_vol_exclude"] = saved_settings.get('exclude_days_vol', 30)
            else:
                st.session_state["multi_backtest_active_vol_window"] = active_portfolio.get('vol_window_days', 365)
                st.session_state["multi_backtest_active_vol_exclude"] = active_portfolio.get('exclude_days_vol', 30)
            st.number_input("Volatility Lookback (days)", min_value=1, key="multi_backtest_active_vol_window", on_change=update_vol_window)
            st.number_input("Volatility Exclude (days)", min_value=0, key="multi_backtest_active_vol_exclude", on_change=update_vol_exclude)
    
    # Minimal Threshold Filter Section
    st.markdown("---")
    st.subheader("Minimal Threshold Filter")
    
    # Initialize threshold settings if not present
    if "multi_backtest_active_use_threshold" not in st.session_state:
        st.session_state["multi_backtest_active_use_threshold"] = active_portfolio.get('use_minimal_threshold', False)
    if "multi_backtest_active_threshold_percent" not in st.session_state:
        st.session_state["multi_backtest_active_threshold_percent"] = active_portfolio.get('minimal_threshold_percent', 2.0)
    
    st.checkbox(
        "Enable Minimal Threshold Filter", 
        key="multi_backtest_active_use_threshold", 
        on_change=update_use_threshold,
        help="Exclude stocks with allocations below the threshold percentage and normalize remaining allocations to 100%"
    )
    
    if st.session_state.get("multi_backtest_active_use_threshold", False):
        st.number_input(
            "Minimal Threshold (%)", 
            min_value=0.1, 
            max_value=50.0, 
            value=2.0, 
            step=0.1,
            key="multi_backtest_active_threshold_percent", 
            on_change=update_threshold_percent,
            help="Stocks with allocations below this percentage will be excluded and their weight redistributed to remaining stocks"
        )
    
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
        st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['momentum_windows'][index]['lookback'] = st.session_state[f"multi_backtest_lookback_active_{index}"]

    def update_momentum_exclude(index):
        st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['momentum_windows'][index]['exclude'] = st.session_state[f"multi_backtest_exclude_active_{index}"]
    
    def update_momentum_weight(index):
        st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['momentum_windows'][index]['weight'] = st.session_state[f"multi_backtest_weight_input_active_{index}"] / 100.0

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

    for j in range(len(active_portfolio['momentum_windows'])):
        with st.container():
            col_mw1, col_mw2, col_mw3 = st.columns(3)
            lookback_key = f"multi_backtest_lookback_active_{j}"
            exclude_key = f"multi_backtest_exclude_active_{j}"
            weight_key = f"multi_backtest_weight_input_active_{j}"
            if lookback_key not in st.session_state:
                # Convert lookback to integer to match min_value type
                st.session_state[lookback_key] = int(active_portfolio['momentum_windows'][j]['lookback'])
            if exclude_key not in st.session_state:
                # Convert exclude to integer to match min_value type
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
                st.number_input(f"Lookback {j+1}", min_value=1, key=lookback_key, label_visibility="collapsed")
                if st.session_state[lookback_key] != active_portfolio['momentum_windows'][j]['lookback']:
                    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['momentum_windows'][j]['lookback'] = st.session_state[lookback_key]
            with col_mw2:
                st.number_input(f"Exclude {j+1}", min_value=0, key=exclude_key, label_visibility="collapsed")
                if st.session_state[exclude_key] != active_portfolio['momentum_windows'][j]['exclude']:
                    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['momentum_windows'][j]['exclude'] = st.session_state[exclude_key]
            with col_mw3:
                st.number_input(f"Weight {j+1}", min_value=0, max_value=100, step=1, format="%d", key=weight_key, label_visibility="collapsed")
                # Update the portfolio weight when the widget value changes
                if st.session_state[weight_key] != int(active_portfolio['momentum_windows'][j]['weight'] * 100.0):
                    st.session_state.multi_backtest_portfolio_configs[st.session_state.multi_backtest_active_portfolio_index]['momentum_windows'][j]['weight'] = st.session_state[weight_key] / 100.0
else:
    # Don't clear momentum_windows - they should persist when momentum is disabled
    # so they're available when momentum is re-enabled or for variant generation
    pass

with st.expander("JSON Configuration (Copy & Paste)", expanded=False):
    # Clean portfolio config for export by removing unused settings
    cleaned_config = active_portfolio.copy()
    cleaned_config.pop('use_relative_momentum', None)
    cleaned_config.pop('equal_if_all_negative', None)
    # Update global settings from session state
    cleaned_config['start_with'] = st.session_state.get('multi_backtest_start_with', 'all')
    cleaned_config['first_rebalance_strategy'] = st.session_state.get('multi_backtest_first_rebalance_strategy', 'rebalancing_date')
    
    # Convert date objects to strings for JSON serialization
    if cleaned_config.get('start_date_user') is not None:
        cleaned_config['start_date_user'] = cleaned_config['start_date_user'].isoformat() if hasattr(cleaned_config['start_date_user'], 'isoformat') else str(cleaned_config['start_date_user'])
    if cleaned_config.get('end_date_user') is not None:
        cleaned_config['end_date_user'] = cleaned_config['end_date_user'].isoformat() if hasattr(cleaned_config['end_date_user'], 'isoformat') else str(cleaned_config['end_date_user'])
    
    config_json = json.dumps(cleaned_config, indent=4)
    st.code(config_json, language='json')
    # Fixed JSON copy button
    import streamlit.components.v1 as components
    copy_html = f"""
    <button onclick='navigator.clipboard.writeText({json.dumps(config_json)});' style='margin-bottom:10px;'>Copy to Clipboard</button>
    """
    components.html(copy_html, height=40)
    
    # Add PDF download button for individual portfolio JSON
    def generate_individual_json_pdf(custom_name=""):
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
            title = f"Multi Backtest - {custom_name.strip()} - JSON Configuration"
            subject = f"JSON Configuration: {custom_name.strip()}"
        else:
            title = f"Multi Backtest - {portfolio_name} - JSON Configuration"
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
            creator="Multi Backtest Application"
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
        placeholder=f"e.g., {active_portfolio.get('name', 'Portfolio')} Configuration, Custom Setup Analysis",
        help="Leave empty to use automatic naming based on portfolio name",
        key="multi_individual_custom_pdf_name"
    )
    
    if st.button("📄 Download JSON as PDF", help="Download a PDF containing the JSON configuration for easy copying", key="multi_individual_json_pdf_btn"):
        try:
            pdf_data = generate_individual_json_pdf(custom_individual_pdf_name)
            
            # Generate filename based on custom name or default
            if custom_individual_pdf_name.strip():
                clean_name = custom_individual_pdf_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
                filename = f"{clean_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            else:
                filename = f"multi_portfolio_{active_portfolio.get('name', 'portfolio').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            st.download_button(
                label="💾 Download Portfolio JSON PDF",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf",
                key="multi_individual_json_pdf_download"
            )
            st.success("PDF generated successfully! Click the download button above.")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
    
    st.text_area("Paste JSON Here to Update Portfolio", key="multi_backtest_paste_json_text", height=200)
    st.button("Update with Pasted JSON", on_click=paste_json_callback)
    
    # Add PDF drag and drop functionality
    st.markdown("**OR** 📎 **Drag & Drop JSON PDF:**")
    
    def extract_json_from_pdf(pdf_file):
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
        key="multi_individual_pdf_upload"
    )
    
    if uploaded_pdf is not None:
        json_data, error = extract_json_from_pdf(uploaded_pdf)
        if json_data:
            # Store the extracted JSON in a different session state key to avoid widget conflicts
            st.session_state["multi_backtest_extracted_json"] = json.dumps(json_data, indent=4)
            st.success(f"✅ Successfully extracted JSON from {uploaded_pdf.name}")
            st.info("👇 Click the button below to load the JSON into the text area.")
            def load_extracted_json():
                st.session_state["multi_backtest_paste_json_text"] = st.session_state["multi_backtest_extracted_json"]
            
            st.button("📋 Load Extracted JSON", key="load_extracted_json", on_click=load_extracted_json)
        else:
            st.error(f"❌ Failed to extract JSON from PDF: {error}")
            st.info("💡 Make sure the PDF contains valid JSON content (generated by this app)")

# Validation constants
_TOTAL_TOL = 1.0
_ALLOC_TOL = 1.0

# Move Run Backtest to the left sidebar to make it conspicuous and separate from config
if st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True):
    
    # Pre-backtest validation check for all portfolios
    configs_to_run = st.session_state.multi_backtest_portfolio_configs
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
                
    if not valid_configs:
        for error in validation_errors:
            st.error(error)
        # Don't run the backtest, but continue showing the UI
        pass
    else:
        # Show standalone popup notification that code is really running
        st.toast("**Code is running!** Starting backtest...", icon="🚀")
        
        progress_bar = st.empty()
        progress_bar.progress(0, text="Initializing multi-portfolio backtest...")
        
        # Get all tickers first
        all_tickers = sorted(list(set(s['ticker'] for cfg in st.session_state.multi_backtest_portfolio_configs for s in cfg['stocks'] if s['ticker']) | set(cfg['benchmark_ticker'] for cfg in st.session_state.multi_backtest_portfolio_configs if 'benchmark_ticker' in cfg)))
        all_tickers = [t for t in all_tickers if t]
        
        # BULLETPROOF VALIDATION: Check for empty ticker list first
        if not all_tickers:
            st.error("❌ **No valid tickers found!** Please add at least one ticker to your portfolios before running the backtest.")
            progress_bar.empty()
            st.session_state.multi_all_results = None
            st.session_state.multi_all_allocations = None
            st.session_state.multi_all_metrics = None
            st.stop()
        
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            data = {}
            invalid_tickers = []
            for i, t in enumerate(all_tickers):
                try:
                    progress_text = f"Downloading data for {t} ({i+1}/{len(all_tickers)})..."
                    progress_bar.progress((i + 1) / (len(all_tickers) + 1), text=progress_text)
                    hist = get_ticker_data(t, period="max", auto_adjust=False)
                    if hist.empty:
                        invalid_tickers.append(t)
                        continue
                    
                    # Force tz-naive for hist (like Backtest_Engine.py)
                    hist = hist.copy()
                    hist.index = hist.index.tz_localize(None)
                    
                    hist["Price_change"] = hist["Close"].pct_change(fill_method=None).fillna(0)
                    data[t] = hist
                except Exception as e:
                    invalid_tickers.append(t)
            # Display invalid ticker warnings in Streamlit UI
            if invalid_tickers:
                # Separate portfolio tickers from benchmark tickers
                portfolio_tickers = set(s['ticker'] for cfg in st.session_state.multi_backtest_portfolio_configs for s in cfg['stocks'] if s['ticker'])
                benchmark_tickers = set(cfg.get('benchmark_ticker') for cfg in st.session_state.multi_backtest_portfolio_configs if 'benchmark_ticker' in cfg)
                
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
                st.session_state.multi_all_results = None
                st.session_state.multi_all_allocations = None
                st.session_state.multi_all_metrics = None
                st.stop()
            else:
                # Persist raw downloaded price data so later recomputations can access benchmark series
                st.session_state.multi_backtest_raw_data = data
                # Determine common date range for all portfolios
                common_start = max(df.first_valid_index() for df in data.values())
                common_end = min(df.last_valid_index() for df in data.values())
                
                # Get all portfolio tickers (excluding benchmarks)
                all_portfolio_tickers = set()
                for cfg in st.session_state.multi_backtest_portfolio_configs:
                    portfolio_tickers = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                    all_portfolio_tickers.update(portfolio_tickers)
                
                # Check for non-USD tickers and display currency warning
                check_currency_warning(list(all_portfolio_tickers))
                
                # Determine final start date based on global start_with setting
                # Filter to only valid tickers that exist in data
                valid_portfolio_tickers = [t for t in all_portfolio_tickers if t in data]
                
                if not valid_portfolio_tickers:
                    st.error("❌ **No valid tickers found!** None of your portfolio tickers have data available. Please check your ticker symbols and try again.")
                    progress_bar.empty()
                    st.session_state.multi_all_results = None
                    st.session_state.multi_all_allocations = None
                    st.session_state.multi_all_metrics = None
                    st.stop()
                
                global_start_with = st.session_state.get('multi_backtest_start_with', 'all')
                if global_start_with == 'all':
                    final_start = max(data[t].first_valid_index() for t in valid_portfolio_tickers)
                else:  # global_start_with == 'oldest'
                    # For 'oldest', we need to find the portfolio that starts the LATEST
                    # (has the most recent earliest asset), then use that portfolio's earliest asset
                    portfolio_earliest_dates = {}
                    for cfg in st.session_state.multi_backtest_portfolio_configs:
                        portfolio_tickers = [stock['ticker'] for stock in cfg.get('stocks', []) if stock['ticker']]
                        valid_portfolio_tickers_for_cfg = [t for t in portfolio_tickers if t in data]
                        if valid_portfolio_tickers_for_cfg:
                            # Find the earliest asset in this portfolio
                            portfolio_earliest = min(data[t].first_valid_index() for t in valid_portfolio_tickers_for_cfg)
                            portfolio_earliest_dates[cfg['name']] = portfolio_earliest
                    
                    if portfolio_earliest_dates:
                        # Find the portfolio with the LATEST earliest asset
                        latest_starting_portfolio = max(portfolio_earliest_dates.items(), key=lambda x: x[1])
                        final_start = latest_starting_portfolio[1]
                    else:
                        # Fallback to original logic
                        final_start = min(data[t].first_valid_index() for t in valid_portfolio_tickers)
                
                # Apply user date constraints if any
                for cfg in st.session_state.multi_backtest_portfolio_configs:
                    if cfg.get('start_date_user'):
                        user_start = pd.to_datetime(cfg['start_date_user'])
                        final_start = max(final_start, user_start)
                    if cfg.get('end_date_user'):
                        user_end = pd.to_datetime(cfg['end_date_user'])
                        common_end = min(common_end, user_end)
                
                if final_start > common_end:
                    st.error(f"Start date {final_start.date()} is after end date {common_end.date()}. Cannot proceed.")
                    st.stop()
                
                # Create simulation index for the entire period
                simulation_index = pd.date_range(start=final_start, end=common_end, freq='D')
                
                # Reindex all data to the simulation period (only valid tickers)
                data_reindexed = {}
                for t in all_tickers:
                    if t in data:  # Only process tickers that have data
                        df = data[t].reindex(simulation_index)
                        df["Close"] = df["Close"].ffill()
                        df["Dividends"] = df["Dividends"].fillna(0)
                        df["Price_change"] = df["Close"].pct_change(fill_method=None).fillna(0)
                        data_reindexed[t] = df
                
                progress_bar.progress(1.0, text="Executing multi-portfolio backtest analysis...")
                
                # =============================================================================
                # SIMPLE, FAST, AND RELIABLE PORTFOLIO PROCESSING (NO CACHE VERSION)
                # =============================================================================
                
                # Initialize results storage
                all_results = {}
                all_drawdowns = {}
                all_stats = {}
                all_allocations = {}
                all_metrics = {}
                portfolio_key_map = {}
                successful_portfolios = 0
                failed_portfolios = []
                
                st.info(f"🚀 **Processing {len(st.session_state.multi_backtest_portfolio_configs)} portfolios with enhanced reliability (NO CACHE)...**")
                
                # Process portfolios one by one with robust error handling
                for i, cfg in enumerate(st.session_state.multi_backtest_portfolio_configs, start=1):
                    try:
                        # Update progress
                        progress_percent = i / len(st.session_state.multi_backtest_portfolio_configs)
                        progress_bar.progress(progress_percent, text=f"Processing portfolio {i}/{len(st.session_state.multi_backtest_portfolio_configs)}: {cfg.get('name', f'Portfolio {i}')}")
                        
                        name = cfg.get('name', f'Portfolio {i}')
                        
                        # Ensure unique key for storage
                        base_name = name
                        unique_name = base_name
                        suffix = 1
                        while unique_name in all_results or unique_name in all_allocations:
                            unique_name = f"{base_name} ({suffix})"
                            suffix += 1
                        
                        # Run single backtest for this portfolio
                        total_series, total_series_no_additions, historical_allocations, historical_metrics = single_backtest(cfg, simulation_index, data_reindexed)
                        
                        if total_series is not None and len(total_series) > 0:
                            # Compute today_weights_map
                            today_weights_map = {}
                            try:
                                alloc_dates = sorted(list(historical_allocations.keys()))
                                if alloc_dates:
                                    final_d = alloc_dates[-1]
                                    metrics_local = historical_metrics
                                    
                                    # Check if momentum is used for this portfolio
                                    use_momentum = cfg.get('use_momentum', True)
                                    
                                    if final_d in metrics_local:
                                        if use_momentum:
                                            # Extract Calculated_Weight if present (momentum-based)
                                            weights = {t: v.get('Calculated_Weight', 0) for t, v in metrics_local[final_d].items()}
                                            # Normalize (ensure sums to 1 excluding CASH)
                                            sumw = sum(w for k, w in weights.items() if k != 'CASH')
                                            if sumw > 0:
                                                norm = {k: (w / sumw) if k != 'CASH' else weights.get('CASH', 0) for k, w in weights.items()}
                                            else:
                                                norm = weights
                                            today_weights_map = norm
                                        else:
                                            # Use user-defined allocations from portfolio config
                                            today_weights_map = {}
                                            for stock in cfg.get('stocks', []):
                                                ticker = stock.get('ticker', '').strip()
                                                if ticker:
                                                    today_weights_map[ticker] = stock.get('allocation', 0)
                                            # Add CASH if needed
                                            total_alloc = sum(today_weights_map.values())
                                            if total_alloc < 1.0:
                                                today_weights_map['CASH'] = 1.0 - total_alloc
                                            else:
                                                today_weights_map['CASH'] = 0
                                    else:
                                        # Fallback: use allocation snapshot at final date
                                        final_alloc = historical_allocations.get(final_d, {})
                                        noncash = {k: v for k, v in final_alloc.items() if k != 'CASH'}
                                        s = sum(noncash.values())
                                        if s > 0:
                                            norm = {k: (v / s) for k, v in noncash.items()}
                                            norm['CASH'] = final_alloc.get('CASH', 0)
                                        else:
                                            norm = final_alloc
                                        today_weights_map = norm
                            except Exception as e:
                                # If computation fails, use user-defined allocations as fallback
                                today_weights_map = {}
                                for stock in cfg.get('stocks', []):
                                    ticker = stock.get('ticker', '').strip()
                                    if ticker:
                                        today_weights_map[ticker] = stock.get('allocation', 0)
                                # Add CASH if needed
                                total_alloc = sum(today_weights_map.values())
                                if total_alloc < 1.0:
                                    today_weights_map['CASH'] = 1.0 - total_alloc
                                else:
                                    today_weights_map['CASH'] = 0
                            
                            # Store results in simplified format
                            all_results[unique_name] = {
                                'no_additions': total_series_no_additions,
                                'with_additions': total_series,
                                'today_weights_map': today_weights_map
                            }
                            all_allocations[unique_name] = historical_allocations
                            all_metrics[unique_name] = historical_metrics
                            
                            # Remember mapping from portfolio index (0-based) to unique key
                            portfolio_key_map[i-1] = unique_name
                            
                            successful_portfolios += 1
                            
                            # Memory cleanup every 20 portfolios
                            if successful_portfolios % 20 == 0:
                                import gc
                                gc.collect()
                                
                        else:
                            failed_portfolios.append((name, "Empty results from backtest"))
                            st.warning(f"⚠️ Portfolio {name} failed: Empty results from backtest")
                            
                    except Exception as e:
                        failed_portfolios.append((cfg.get('name', f'Portfolio {i}'), str(e)))
                        st.warning(f"⚠️ Portfolio {cfg.get('name', f'Portfolio {i}')} failed: {str(e)}")
                        continue
                
                # Final progress update
                progress_bar.progress(1.0, text="Portfolio processing completed!")
                
                # Show results summary
                if successful_portfolios > 0:
                    st.success(f"🎉 **Successfully processed {successful_portfolios}/{len(st.session_state.multi_backtest_portfolio_configs)} portfolios!**")
                    if failed_portfolios:
                        st.warning(f"⚠️ **{len(failed_portfolios)} portfolios failed** - check warnings above for details")
                else:
                    st.error("❌ **No portfolios were processed successfully!** Please check your configuration.")
                    st.stop()
                
                # Memory cleanup
                import gc
                gc.collect()
                progress_bar.empty()
                
                # --- CASH FLOW LOGIC FOR MWRR (stored for later calculation) ---
                # Track cash flows as pandas Series indexed by date
                cash_flows = pd.Series(0.0, index=total_series.index)
                # Initial investment: negative cash flow on first date
                if len(total_series.index) > 0:
                    cash_flows.iloc[0] = -cfg.get('initial_value', 0)
                # Periodic additions: negative cash flow on their respective dates
                dates_added = get_dates_by_freq(cfg.get('added_frequency'), total_series.index[0], total_series.index[-1], total_series.index)
                for d in dates_added:
                    if d in cash_flows.index and d != cash_flows.index[0]:
                        cash_flows.loc[d] -= cfg.get('added_amount', 0)
                # Final value: positive cash flow on last date for MWRR
                if len(total_series.index) > 0:
                    cash_flows.iloc[-1] += total_series.iloc[-1]
                
                # Store cash flows and portfolio values for MWRR calculation after all portfolios are processed
                all_results[unique_name]['cash_flows'] = cash_flows
                all_results[unique_name]['portfolio_values'] = total_series
                
                # Get benchmark returns for stats calculation
                benchmark_returns = None
                if cfg['benchmark_ticker'] and cfg['benchmark_ticker'] in data_reindexed:
                    benchmark_returns = data_reindexed[cfg['benchmark_ticker']]['Price_change']
                    # Ensure benchmark_returns is a pandas Series aligned to total_series
                    if benchmark_returns is not None:
                        benchmark_returns = pd.Series(benchmark_returns, index=total_series.index).dropna()
                    
                    # Calculate statistics (excluding MWRR for now)
                    # Use total_series_no_additions for all stats except MWRR
                    stats_values = total_series_no_additions.values
                    stats_dates = total_series_no_additions.index
                    stats_returns = pd.Series(stats_values, index=stats_dates).pct_change().fillna(0)
                    cagr = calculate_cagr(stats_values, stats_dates)
                    max_dd, drawdowns = calculate_max_drawdown(stats_values)
                    vol = calculate_volatility(stats_returns)
                    
                    # Use 2% annual risk-free rate (same as Backtest_Engine.py default)
                    risk_free_rate = 0.02
                    sharpe = calculate_sharpe(stats_returns, risk_free_rate)
                    sortino = calculate_sortino(stats_returns, risk_free_rate)
                    ulcer = calculate_ulcer_index(pd.Series(stats_values, index=stats_dates))
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
                            common_idx2 = pr.index.intersection(br.index)
                            if len(common_idx2) >= 2 and br.loc[common_idx2].var() != 0:
                                cov = pr.loc[common_idx2].cov(br.loc[common_idx2])
                                var = br.loc[common_idx2].var()
                                beta = cov / var
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
                        if stat_type == "MWRR":
                            # MWRR is already a percentage from calculate_mwrr, don't scale it
                            v = val
                        else:
                            v = scale_pct(val)
                        # Clamp ranges for each stat type
                        if stat_type in ["CAGR", "Volatility", "Total Return"]:
                            if v < 0 or v > 100:
                                return "N/A"
                        elif stat_type == "MWRR":
                            # MWRR can be negative or exceed 100%, so don't clamp it
                            pass
                        elif stat_type == "MaxDrawdown":
                            if v < -100 or v > 0:
                                return "N/A"
                        return f"{v:.2f}%" if stat_type in ["CAGR", "MaxDrawdown", "Volatility", "MWRR", "Total Return"] else f"{v:.3f}" if isinstance(v, float) else v

                    # Calculate total return (no additions)
                    total_return = None
                    if len(stats_values) > 0:
                        initial_val = stats_values[0]
                        final_val = stats_values[-1]
                        if initial_val > 0:
                            total_return = (final_val / initial_val - 1)  # Return as decimal, not percentage

                    stats = {
                        "Total Return": clamp_stat(total_return, "Total Return"),
                    "CAGR": clamp_stat(cagr, "CAGR"),
                    "MaxDrawdown": clamp_stat(max_dd, "MaxDrawdown"),
                    "Volatility": clamp_stat(vol, "Volatility"),
                    "Sharpe": clamp_stat(sharpe / 100 if isinstance(sharpe, (int, float)) and pd.notna(sharpe) else sharpe, "Sharpe"),
                    "Sortino": clamp_stat(sortino / 100 if isinstance(sortino, (int, float)) and pd.notna(sortino) else sortino, "Sortino"),
                    "UlcerIndex": clamp_stat(ulcer, "UlcerIndex"),
                    "UPI": clamp_stat(upi / 100 if isinstance(upi, (int, float)) and pd.notna(upi) else upi, "UPI"),
                    "Beta": clamp_stat(beta / 100 if isinstance(beta, (int, float)) and pd.notna(beta) else beta, "Beta"),
                    # MWRR will be calculated after all portfolios are processed
                }
                all_stats[unique_name] = stats
                all_drawdowns[unique_name] = pd.Series(drawdowns, index=stats_dates)
            progress_bar.progress(100, text="Multi-portfolio backtest analysis complete!")
            progress_bar.empty()
            
            # --- CALCULATE MWRR FOR ALL PORTFOLIOS AFTER LOOP COMPLETES ---
            for unique_name, results in all_results.items():
                if 'cash_flows' in results and 'portfolio_values' in results:
                    cash_flows = results['cash_flows']
                    portfolio_values = results['portfolio_values']
                    # Calculate MWRR with complete cash flow series
                    mwrr = calculate_mwrr(portfolio_values, cash_flows, portfolio_values.index)
                    # Add MWRR to the stats
                    if unique_name in all_stats:
                        all_stats[unique_name]["MWRR"] = mwrr
                    # Clean up temporary data
                    del results['cash_flows']
                    del results['portfolio_values']
                else:
                    if unique_name in all_stats:
                        all_stats[unique_name]["MWRR"] = np.nan
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
                stats_df_display['Total Return'] = stats_df_display['Total Return'].apply(lambda x: fmt_pct(x))
                stats_df_display['CAGR'] = stats_df_display['CAGR'].apply(lambda x: fmt_pct(x))
                stats_df_display['Max Drawdown'] = stats_df_display['Max Drawdown'].apply(lambda x: fmt_pct(x))
                stats_df_display['Volatility'] = stats_df_display['Volatility'].apply(lambda x: fmt_pct(x))
                # Ensure MWRR is the last column, Beta immediately before it, Total Return at the very end
                if 'Beta' in stats_df_display.columns and 'MWRR' in stats_df_display.columns and 'Total Return' in stats_df_display.columns:
                    cols = list(stats_df_display.columns)
                    # Remove Beta, MWRR, and Total Return
                    beta_col = cols.pop(cols.index('Beta'))
                    mwrr_col = cols.pop(cols.index('MWRR'))
                    total_return_col = cols.pop(cols.index('Total Return'))
                    # Insert Beta before MWRR, then Total Return at the very end
                    cols.append(beta_col)
                    cols.append(mwrr_col)
                    cols.append(total_return_col)
                    stats_df_display = stats_df_display[cols]
                # MWRR is already a percentage from calculate_mwrr, format it properly
                if 'MWRR' in stats_df_display.columns:
                    stats_df_display['MWRR'] = stats_df_display['MWRR'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) and pd.notna(x) else "N/A")
                stats_df_display['Sharpe'] = stats_df_display['Sharpe'].apply(lambda x: fmt_num(x))
                stats_df_display['Sortino'] = stats_df_display['Sortino'].apply(lambda x: fmt_num(x))
                stats_df_display['Ulcer Index'] = stats_df_display['Ulcer Index'].apply(lambda x: fmt_num(x))
                stats_df_display['UPI'] = stats_df_display['UPI'].apply(lambda x: fmt_num(x))
                if 'Beta' in stats_df_display.columns:
                    stats_df_display['Beta'] = stats_df_display['Beta'].apply(lambda x: fmt_num(x))
            else:
                pass
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
            
            
            for y in years:
                row_items = [f"{y}"]
                for nm in names:
                    ser = all_years[nm]
                    ser_year = ser[ser.index.year == y]
                    
                    # Corrected logic for yearly performance calculation
                    start_val_for_year = None
                    if y == min(years):
                        config_for_name = next((c for c in st.session_state.multi_backtest_portfolio_configs if c['name'] == nm), None)
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
    
            # console output captured previously is no longer shown on the page
            # Create today_weights_map for all portfolios
            today_weights_map = {}
            for unique_name, results in all_results.items():
                if isinstance(results, dict):
                    if 'today_weights_map' in results:
                        today_weights_map[unique_name] = results['today_weights_map']
            
            # Get last rebalance dates for all portfolios from actual allocation data
            last_rebalance_dates = {}
            for portfolio_name in st.session_state.multi_backtest_portfolio_configs:
                portfolio_name = portfolio_name.get('name', 'Unknown')
                
                # Get the actual allocation data for this portfolio
                if portfolio_name in all_allocations:
                    allocs_for_portfolio = all_allocations[portfolio_name]
                    if allocs_for_portfolio:
                        # Get sorted allocation dates (same logic as real code)
                        alloc_dates = sorted(list(allocs_for_portfolio.keys()))
                        if len(alloc_dates) > 1:
                            # Use second to last date (same as real timer code)
                            last_rebalance_dates[portfolio_name] = alloc_dates[-2]
                        elif len(alloc_dates) == 1:
                            # Use the only available date
                            last_rebalance_dates[portfolio_name] = alloc_dates[-1]
                        else:
                            # No allocation data available
                            last_rebalance_dates[portfolio_name] = None
                    else:
                        last_rebalance_dates[portfolio_name] = None
                else:
                    last_rebalance_dates[portfolio_name] = None
            
            st.session_state.multi_backtest_snapshot_data = {
                'raw_data': data_reindexed,
                'portfolio_configs': st.session_state.multi_backtest_portfolio_configs,
                'all_allocations': all_allocations,
                'all_metrics': all_metrics,
                'today_weights_map': today_weights_map,
                'last_rebalance_dates': last_rebalance_dates
            }
            
            # Create allocation tables for ALL portfolios automatically for PDF export
            try:
                raw_data = st.session_state.get('multi_backtest_raw_data', {})
                
                for portfolio_name, today_weights in today_weights_map.items():
                    if today_weights:
                        # Get portfolio configuration for calculations
                        portfolio_configs = st.session_state.multi_backtest_portfolio_configs
                        portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == portfolio_name), None)
                        
                        if portfolio_cfg:
                            # Get portfolio value
                            portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)
                            
                            # Get current portfolio value from backtest results
                            if 'multi_all_results' in st.session_state and st.session_state.multi_all_results:
                                portfolio_results = st.session_state.multi_all_results.get(portfolio_name)
                                if portfolio_results:
                                    if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                        final_value = portfolio_results['with_additions'].iloc[-1]
                                        if not pd.isna(final_value) and final_value > 0:
                                            portfolio_value = float(final_value)
                                    elif isinstance(portfolio_results, dict) and 'no_additions' in portfolio_results:
                                        final_value = portfolio_results['no_additions'].iloc[-1]
                                        if not pd.isna(final_value) and final_value > 0:
                                            portfolio_value = float(final_value)
                                    elif isinstance(portfolio_results, pd.Series):
                                        latest_value = portfolio_results.iloc[-1]
                                        if not pd.isna(latest_value) and latest_value > 0:
                                            portfolio_value = float(latest_value)
                            
                            # Create allocation table data
                            rows = []
                            for tk in sorted(today_weights.keys()):
                                alloc_pct = float(today_weights.get(tk, 0))
                                if tk == 'CASH':
                                    price = None
                                    shares = 0
                                    total_val = portfolio_value * alloc_pct
                                else:
                                    df = raw_data.get(tk)
                                    price = None
                                    if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                        try:
                                            price = float(df['Close'].iloc[-1])
                                        except Exception:
                                            price = None
                                    try:
                                        if price and price > 0:
                                            allocation_value = portfolio_value * alloc_pct
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
                                    'Allocation %': round(alloc_pct * 100, 2),
                                    'Price ($)': round(price, 2) if price is not None else float('nan'),
                                    'Shares': round(shares, 2),
                                    'Total Value ($)': round(total_val, 2),
                                    '% of Portfolio': round(pct_of_port, 2),
                                })

                            df_table = pd.DataFrame(rows).set_index('Ticker')
                            df_display = df_table.copy()
                            
                            # Remove CASH if it has zero value
                            if 'CASH' in df_display.index:
                                cash_val = df_display.at['CASH', 'Total Value ($)']
                                if not (cash_val and not pd.isna(cash_val) and cash_val != 0):
                                    df_display = df_display.drop('CASH')
                            
                            # Create Plotly table figure with ticker column included
                            df_display_with_ticker = df_display.reset_index()
                            
                            # Format the data to ensure 2 decimal places for display
                            formatted_values = []
                            for col in df_display_with_ticker.columns:
                                if col in ['Price ($)', 'Total Value ($)', '% of Portfolio']:
                                    # Format monetary and percentage values to 2 decimal places
                                    formatted_values.append([f"{df_display_with_ticker[col][i]:.2f}" if pd.notna(df_display_with_ticker[col][i]) else "" for i in range(len(df_display_with_ticker))])
                                elif col == 'Shares':
                                    # Format shares to 1 decimal place
                                    formatted_values.append([f"{df_display_with_ticker[col][i]:.1f}" if pd.notna(df_display_with_ticker[col][i]) else "" for i in range(len(df_display_with_ticker))])
                                elif col == 'Allocation %':
                                    # Format allocation to 2 decimal places
                                    formatted_values.append([f"{df_display_with_ticker[col][i]:.2f}" if pd.notna(df_display_with_ticker[col][i]) else "" for i in range(len(df_display_with_ticker))])
                                else:
                                    # Keep other columns as is
                                    formatted_values.append([str(df_display_with_ticker[col][i]) if pd.notna(df_display_with_ticker[col][i]) else "" for i in range(len(df_display_with_ticker))])
                            
                            fig_alloc_table = go.Figure(data=[go.Table(
                                header=dict(values=list(df_display_with_ticker.columns),
                                           fill_color='paleturquoise',
                                           align='left',
                                           font=dict(size=12)),
                                cells=dict(values=formatted_values,
                                          fill_color='lavender',
                                          align='left',
                                          font=dict(size=11))
                            )])
                            fig_alloc_table.update_layout(
                                title=f"Target Allocation if Rebalanced Today - {portfolio_name}",
                                margin=dict(t=30, b=10, l=10, r=10),
                                height=400
                            )
                            table_key = f"alloc_table_{portfolio_name}"
                            st.session_state[table_key] = fig_alloc_table
            except Exception as e:
                pass
            
            # Create timer tables for ALL portfolios automatically for PDF export
            try:
                snapshot = st.session_state.get('multi_backtest_snapshot_data', {})
                last_rebalance_dates = snapshot.get('last_rebalance_dates', {})
                
                
                for portfolio_cfg in st.session_state.multi_backtest_portfolio_configs:
                    portfolio_name = portfolio_cfg.get('name', 'Unknown')
                    
                    # Get rebalancing frequency for this portfolio
                    rebal_freq = portfolio_cfg.get('rebalancing_frequency', 'none')
                    rebal_freq = rebal_freq.lower()
                    
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
                    rebal_freq = frequency_mapping.get(rebal_freq, rebal_freq)
                    
                    # Get last rebalance date for this portfolio
                    last_rebal_date = last_rebalance_dates.get(portfolio_name)
                    
                    if rebal_freq != 'none':
                        # Ensure last_rebal_date is a naive datetime object if it exists
                        if last_rebal_date and isinstance(last_rebal_date, str):
                            last_rebal_date = pd.to_datetime(last_rebal_date)
                        if last_rebal_date and hasattr(last_rebal_date, 'tzinfo') and last_rebal_date.tzinfo is not None:
                            last_rebal_date = last_rebal_date.replace(tzinfo=None)
                        
                        # Calculate next rebalance for this portfolio (works even with None last_rebal_date)
                        next_date_port, time_until_port, next_rebalance_datetime_port = calculate_next_rebalance_date(
                            rebal_freq, last_rebal_date
                        )
                        
                        
                        if next_date_port and time_until_port:
                            # Create timer data for this portfolio
                            timer_data_port = [
                                ['Time Until Next Rebalance', format_time_until(time_until_port)],
                                ['Target Rebalance Date', next_date_port.strftime("%B %d, %Y")],
                                ['Rebalancing Frequency', rebal_freq.replace('_', ' ').title()]
                            ]
                            
                            # Create timer table figure for this portfolio
                            fig_timer_port = go.Figure(data=[go.Table(
                                header=dict(
                                    values=['Parameter', 'Value'],
                                    fill_color='#2E86AB',
                                    align='center',
                                    font=dict(color='white', size=16, family='Arial Black')
                                ),
                                cells=dict(
                                    values=[[row[0] for row in timer_data_port], [row[1] for row in timer_data_port]],
                                    fill_color=[['#F8F9FA', '#FFFFFF'] * 2, ['#F8F9FA', '#FFFFFF'] * 2],
                                    align='center',
                                    font=dict(color='black', size=14, family='Arial'),
                                    height=40
                                )
                            )])
                            
                            fig_timer_port.update_layout(
                                title=dict(
                                    text=f"⏰ Next Rebalance Timer - {portfolio_name}",
                                    x=0.5,
                                    font=dict(size=18, color='#2E86AB', family='Arial Black')
                                ),
                                width=700,
                                height=250,
                                margin=dict(l=20, r=20, t=60, b=20)
                            )
                            
                            # Store in session state for PDF export
                            st.session_state[f'timer_table_{portfolio_name}'] = fig_timer_port
                        else:
                            pass
                    else:
                        pass
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            st.session_state.multi_all_results = all_results
            st.session_state.multi_backtest_all_drawdowns = all_drawdowns
            if 'stats_df_display' in locals():
                st.session_state.multi_backtest_stats_df_display = stats_df_display
            st.session_state.multi_backtest_all_years = all_years
            st.session_state.multi_all_allocations = all_allocations
            st.session_state.multi_all_metrics = all_metrics
            # Save portfolio index -> unique key mapping so UI selectors can reference results reliably
            st.session_state.multi_backtest_portfolio_key_map = portfolio_key_map
            st.session_state.multi_backtest_ran = True

# Sidebar JSON export/import for ALL portfolios
def paste_all_json_callback():
    txt = st.session_state.get('multi_backtest_paste_all_json_text', '')
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
                if 'use_minimal_threshold' not in portfolio:
                    portfolio['use_minimal_threshold'] = False
                if 'minimal_threshold_percent' not in portfolio:
                    portfolio['minimal_threshold_percent'] = 2.0
        
        if isinstance(obj, list):
            # Clear widget keys to force re-initialization
            widget_keys_to_clear = [
                "multi_backtest_active_name", "multi_backtest_active_initial", 
                "multi_backtest_active_added_amount", "multi_backtest_active_rebal_freq",
                "multi_backtest_active_add_freq", "multi_backtest_active_benchmark",
                "multi_backtest_active_use_momentum", "multi_backtest_active_collect_dividends_as_cash",
                "multi_backtest_start_with_radio", "multi_backtest_first_rebalance_strategy_radio"
            ]
            for key in widget_keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            # Process each portfolio configuration for Multi-Backtest page
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
                                
                                stock = {
                                    'ticker': tickers[i].strip(),
                                    'allocation': allocation,
                                    'include_dividends': bool(divs[i]) if i < len(divs) and divs[i] is not None else True
                                }
                                stocks.append(stock)
                
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
                
                # Map frequency values from app.py format to Multi-Backtest format
                def map_frequency(freq):
                    if freq is None:
                        return 'Never'
                    freq_map = {
                        'Never': 'Never',
                        'Buy & Hold': 'Buy & Hold',
                        'Buy & Hold (Target)': 'Buy & Hold (Target)',
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
                
                # Multi-Backtest page specific: ensure all required fields are present
                # and ignore fields that are specific to other pages
                multi_backtest_config = {
                    'name': cfg.get('name', 'New Portfolio'),
                    'stocks': stocks,
                    'benchmark_ticker': cfg.get('benchmark_ticker', '^GSPC'),
                    'initial_value': cfg.get('initial_value', 10000),
                                          'added_amount': cfg.get('added_amount', 1000),
                      'added_frequency': map_frequency(cfg.get('added_frequency', 'Monthly')),
                      'rebalancing_frequency': map_frequency(cfg.get('rebalancing_frequency', 'Monthly')),
                      'start_date_user': parse_date_from_json(cfg.get('start_date_user')),
                      'end_date_user': parse_date_from_json(cfg.get('end_date_user')),
                      'start_with': cfg.get('start_with', 'all'),
                      'first_rebalance_strategy': cfg.get('first_rebalance_strategy', 'rebalancing_date'),
                      'use_momentum': cfg.get('use_momentum', True),
                    'momentum_strategy': momentum_strategy,
                    'negative_momentum_strategy': negative_momentum_strategy,
                    'momentum_windows': momentum_windows,
                    'use_minimal_threshold': cfg.get('use_minimal_threshold', False),
                    'minimal_threshold_percent': cfg.get('minimal_threshold_percent', 2.0),
                    'calc_beta': cfg.get('calc_beta', True),
                    'calc_volatility': cfg.get('calc_volatility', True),
                    'beta_window_days': cfg.get('beta_window_days', 365),
                    'exclude_days_beta': cfg.get('exclude_days_beta', 30),
                    'vol_window_days': cfg.get('vol_window_days', 365),
                    'exclude_days_vol': cfg.get('exclude_days_vol', 30),
                    'collect_dividends_as_cash': cfg.get('collect_dividends_as_cash', False),
                    # Preserve sync exclusion settings from imported JSON
                    'exclude_from_cashflow_sync': cfg.get('exclude_from_cashflow_sync', False),
                    'exclude_from_rebalancing_sync': cfg.get('exclude_from_rebalancing_sync', False),
                    # Note: Ignoring Backtest Engine specific fields like 'portfolio_drag_pct', 'use_custom_dates', etc.
                }
                processed_configs.append(multi_backtest_config)
            
            st.session_state.multi_backtest_portfolio_configs = processed_configs
            
            # Handle global start_with setting from imported JSON
            if processed_configs and 'start_with' in processed_configs[0]:
                # Handle start_with value mapping from other pages
                start_with = processed_configs[0]['start_with']
                if start_with == 'first':
                    start_with = 'oldest'  # Map 'first' to 'oldest' (closest equivalent)
                elif start_with not in ['all', 'oldest']:
                    start_with = 'all'  # Default fallback
                st.session_state['_import_start_with'] = start_with
            
            # Handle global first_rebalance_strategy setting from imported JSON
            if processed_configs and 'first_rebalance_strategy' in processed_configs[0]:
                st.session_state['_import_first_rebalance_strategy'] = processed_configs[0]['first_rebalance_strategy']
            
            # Reset active selection and derived mappings so the UI reflects the new configs
            if processed_configs:
                st.session_state.multi_backtest_active_portfolio_index = 0
                st.session_state.multi_backtest_portfolio_selector = processed_configs[0].get('name', '')
                # Mirror several active_* widget defaults so the UI selectboxes/inputs update
                st.session_state['multi_backtest_active_name'] = processed_configs[0].get('name', '')
                st.session_state['multi_backtest_active_initial'] = int(processed_configs[0].get('initial_value', 0) or 0)
                st.session_state['multi_backtest_active_added_amount'] = int(processed_configs[0].get('added_amount', 0) or 0)
                st.session_state['multi_backtest_active_rebal_freq'] = processed_configs[0].get('rebalancing_frequency', 'none')
                st.session_state['multi_backtest_active_add_freq'] = processed_configs[0].get('added_frequency', 'none')
                st.session_state['multi_backtest_active_benchmark'] = processed_configs[0].get('benchmark_ticker', '')
                st.session_state['multi_backtest_active_use_momentum'] = bool(processed_configs[0].get('use_momentum', True))
                st.session_state['multi_backtest_active_collect_dividends_as_cash'] = bool(processed_configs[0].get('collect_dividends_as_cash', False))
            else:
                st.session_state.multi_backtest_active_portfolio_index = None
                st.session_state.multi_backtest_portfolio_selector = ''
            st.session_state.multi_backtest_portfolio_key_map = {}
            st.session_state.multi_backtest_ran = False
            st.success('All portfolio configurations updated from JSON (Multi-Backtest page).')
            # Debug: Show final momentum windows for first portfolio
            if processed_configs:
                st.info(f"Final momentum windows for first portfolio: {processed_configs[0]['momentum_windows']}")
                st.info(f"Final use_momentum for first portfolio: {processed_configs[0]['use_momentum']}")
                st.info(f"Sync exclusions for first portfolio - Cash Flow: {processed_configs[0].get('exclude_from_cashflow_sync', False)}, Rebalancing: {processed_configs[0].get('exclude_from_rebalancing_sync', False)}")
            # Sync date widgets with the updated portfolio
            sync_date_widgets_with_portfolio()
            
            # Force a rerun so widgets rebuild with the new configs
            try:
                st.experimental_rerun()
            except Exception:
                # In some environments experimental rerun may raise; setting a rerun flag is a fallback
                st.session_state.multi_backtest_rerun_flag = True
        else:
            st.error('JSON must be a list of portfolio configurations.')
    except Exception as e:
        st.error(f'Failed to parse JSON: {e}')


with st.sidebar.expander('All Portfolios JSON (Export / Import)', expanded=False):
    # Clean portfolio configs for export by removing unused settings
    def clean_portfolio_configs_for_export(configs):
        cleaned_configs = []
        for config in configs:
            cleaned_config = config.copy()
            # Remove unused settings that were cleaned up
            cleaned_config.pop('use_relative_momentum', None)
            cleaned_config.pop('equal_if_all_negative', None)
            # Update global settings from session state
            cleaned_config['start_with'] = st.session_state.get('multi_backtest_start_with', 'all')
            cleaned_config['first_rebalance_strategy'] = st.session_state.get('multi_backtest_first_rebalance_strategy', 'rebalancing_date')
            
            # Ensure threshold settings are included (read from current config)
            cleaned_config['use_minimal_threshold'] = config.get('use_minimal_threshold', False)
            cleaned_config['minimal_threshold_percent'] = config.get('minimal_threshold_percent', 2.0)
            
            # Convert date objects to strings for JSON serialization
            if cleaned_config.get('start_date_user') is not None:
                cleaned_config['start_date_user'] = cleaned_config['start_date_user'].isoformat() if hasattr(cleaned_config['start_date_user'], 'isoformat') else str(cleaned_config['start_date_user'])
            if cleaned_config.get('end_date_user') is not None:
                cleaned_config['end_date_user'] = cleaned_config['end_date_user'].isoformat() if hasattr(cleaned_config['end_date_user'], 'isoformat') else str(cleaned_config['end_date_user'])
            
            cleaned_configs.append(cleaned_config)
        return cleaned_configs
    
    cleaned_configs = clean_portfolio_configs_for_export(st.session_state.get('multi_backtest_portfolio_configs', []))
    all_json = json.dumps(cleaned_configs, indent=2)
    st.code(all_json, language='json')
    import streamlit.components.v1 as components
    copy_html_all = f"""
    <button onclick='navigator.clipboard.writeText({json.dumps(all_json)});' style='margin-bottom:10px;'>Copy All Configs to Clipboard</button>
    """
    components.html(copy_html_all, height=40)
    
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
        portfolio_count = len(st.session_state.get('multi_backtest_portfolio_configs', []))
        
        # Use custom name if provided, otherwise use default
        if custom_name.strip():
            title = f"Multi Backtest - {custom_name.strip()} - JSON Configuration"
            subject = f"JSON Configuration for Multi Backtest: {custom_name.strip()} ({portfolio_count} portfolios)"
        else:
            title = f"Multi Backtest - All Portfolios ({portfolio_count}) - JSON Configuration"
            subject = f"JSON Configuration for {portfolio_count} Multi Backtest Portfolios"
        
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
            creator="Multi Backtest Application"
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
        json_lines = all_json.split('\n')
        for line in json_lines:
            story.append(Preformatted(line, json_style))
        
        # Build PDF
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    # Optional custom PDF name
    custom_pdf_name = st.text_input(
        "📝 Custom PDF Name (optional):", 
        value="",
        placeholder="e.g., Tech vs Conservative Comparison, Q4 2024 Analysis, etc.",
        help="Leave empty to use automatic naming: 'Multi Backtest - All Portfolios (X) - JSON Configuration'",
        key="multi_backtest_custom_pdf_name"
    )
    
    if st.button("📄 Download JSON as PDF", help="Download a PDF containing the JSON configuration for easy copying", key="multi_backtest_json_pdf_btn"):
        try:
            pdf_data = generate_json_pdf(custom_pdf_name)
            
            # Generate filename based on custom name or default
            if custom_pdf_name.strip():
                clean_name = custom_pdf_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
                filename = f"{clean_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            else:
                filename = f"multi_backtest_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            st.download_button(
                label="💾 Download Multi Backtest JSON PDF",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf",
                key="multi_backtest_json_pdf_download"
            )
            st.success("PDF generated successfully! Click the download button above.")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
    
    st.text_area('Paste JSON Here to Replace All Portfolios', key='multi_backtest_paste_all_json_text', height=240)
    st.button('Update All Portfolios from JSON', on_click=paste_all_json_callback)
    
    # Add PDF drag and drop functionality for all portfolios
    st.markdown("**OR** 📎 **Drag & Drop JSON PDF:**")
    
    def extract_json_from_pdf_all(pdf_file):
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
    
    uploaded_pdf_all = st.file_uploader(
        "Drop your All Portfolios JSON PDF here", 
        type=['pdf'], 
        help="Upload a JSON PDF file containing all portfolio configurations",
        key="multi_all_pdf_upload"
    )
    
    if uploaded_pdf_all is not None:
        json_data, error = extract_json_from_pdf_all(uploaded_pdf_all)
        if json_data:
            # Store the extracted JSON in a different session state key to avoid widget conflicts
            st.session_state["multi_backtest_extracted_json_all"] = json.dumps(json_data, indent=2)
            st.success(f"✅ Successfully extracted JSON from {uploaded_pdf_all.name}")
            st.info("👇 Click the button below to load the JSON into the text area.")
            def load_extracted_json_all():
                st.session_state["multi_backtest_paste_all_json_text"] = st.session_state["multi_backtest_extracted_json_all"]
            
            st.button("📋 Load Extracted JSON", key="load_extracted_json_all", on_click=load_extracted_json_all)
        else:
            st.error(f"❌ Failed to extract JSON from PDF: {error}")
            st.info("💡 Make sure the PDF contains valid JSON content (generated by this app)")

if 'multi_backtest_ran' in st.session_state and st.session_state.multi_backtest_ran:
    if 'multi_all_results' in st.session_state and st.session_state.multi_all_results:
        # Use the no-additions series for all display and calculations
        first_date = min(series['no_additions'].index.min() for series in st.session_state.multi_all_results.values())
        last_date = max(series['no_additions'].index.max() for series in st.session_state.multi_all_results.values())
        st.subheader(f"Results for Backtest Period: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")

        fig1 = go.Figure()
        for name, series_dict in st.session_state.multi_all_results.items():
            # Plot the series that includes added cash (with_additions) for comparison
            series_to_plot = series_dict['with_additions'] if isinstance(series_dict, dict) and 'with_additions' in series_dict else series_dict
            # Convert timestamp index to proper datetime for plotting - ensure it's actually datetime format
            if hasattr(series_to_plot.index, 'to_pydatetime'):
                x_dates = series_to_plot.index.to_pydatetime()
            else:
                x_dates = pd.to_datetime(series_to_plot.index)
            fig1.add_trace(go.Scatter(x=x_dates, y=series_to_plot.values, mode='lines', name=name))
        fig1.update_layout(
            title="Backtest Comparison — Portfolio Value (with cash additions)",
            xaxis_title="Date",
            legend_title="Portfolios",
            hovermode="x unified",
            template="plotly_dark",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            # No width/height restrictions - let them be responsive like other plots
            xaxis=dict(
                type='date',  # Explicitly set as date type
                tickformat="%Y-%m-%d",  # Proper date format
                tickmode="auto",
                nticks=10,  # Reasonable number of ticks
                tickangle=45,  # Angle labels for better readability
                automargin=True,  # Ensure labels fit
                range=None  # Let Plotly auto-range to ensure perfect alignment
            ),
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=80, r=80, t=120, b=80),  # EXACT same margins as Strategy Comparison Chart 1
            height=600,  # Taller height to prevent crushing
            yaxis=dict(
                title="Portfolio Value ($)", 
                title_standoff=20,
                side="left",
                position=0.0  # Force left alignment for perfect positioning
            )
        )
        st.plotly_chart(fig1, use_container_width=True, key="multi_performance_chart")
        # Store in session state for PDF export
        st.session_state.fig1 = fig1

        fig2 = go.Figure()
        for name, series_dict in st.session_state.multi_all_results.items():
            # Use the no-additions series for drawdown calculation (pure portfolio performance)
            series_to_plot = series_dict['no_additions'] if isinstance(series_dict, dict) and 'no_additions' in series_dict else series_dict
            
            # Calculate drawdown for this series
            values = series_to_plot.values
            peak = np.maximum.accumulate(values)
            drawdowns = (values - peak) / np.where(peak == 0, 1, peak) * 100  # Convert to percentage
            
            # Convert timestamp index to proper datetime for plotting - ensure it's actually datetime format
            if hasattr(series_to_plot.index, 'to_pydatetime'):
                x_dates = series_to_plot.index.to_pydatetime()
            else:
                x_dates = pd.to_datetime(series_to_plot.index)
            fig2.add_trace(go.Scatter(x=x_dates, y=drawdowns, mode='lines', name=name))
        fig2.update_layout(
            title="Backtest Comparison (Max Drawdown)",
            xaxis_title="Date",
            legend_title="Portfolios",
            hovermode="x unified",
            template="plotly_dark",
            # No width/height restrictions - let them be responsive like other plots
            xaxis=dict(
                type='date',  # Explicitly set as date type
                tickformat="%Y-%m-%d",  # Proper date format
                tickmode="auto",
                nticks=10,  # Reasonable number of ticks
                tickangle=45,  # Angle labels for better readability
                automargin=True,  # Ensure labels fit
                range=None  # Let Plotly auto-range to ensure perfect alignment
            ),
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=80, r=80, t=120, b=80),  # EXACT same margins as Strategy Comparison
            height=600,  # Taller height to prevent crushing
            yaxis=dict(
                title="Drawdown (%)", 
                title_standoff=20,
                side="left",
                position=0.0  # Force left alignment for perfect positioning
            )
        )
        st.plotly_chart(fig2, use_container_width=True, key="multi_drawdown_chart")
        # Store in session state for PDF export
        st.session_state.fig2 = fig2

        # --- Variation summary chart: compares total return, CAGR, volatility and max drawdown across portfolios ---
        try:
            def get_no_additions_series(obj):
                return obj['no_additions'] if isinstance(obj, dict) and 'no_additions' in obj else obj if isinstance(obj, pd.Series) else None

            metrics_summary = {}
            for name, series_obj in st.session_state.multi_all_results.items():
                ser_no = get_no_additions_series(series_obj)
                if ser_no is None or len(ser_no) < 2:
                    continue
                vals = ser_no.values
                dates = ser_no.index
                # Total return over the period
                try:
                    total_return = (vals[-1] / vals[0] - 1) * 100 if vals[0] and not np.isnan(vals[0]) else np.nan
                except Exception:
                    total_return = np.nan

                # CAGR, volatility, max drawdown (convert to percent for display)
                try:
                    cagr = calculate_cagr(vals, dates)
                except Exception:
                    cagr = np.nan
                try:
                    returns = pd.Series(vals, index=dates).pct_change().fillna(0)
                    vol = calculate_volatility(returns)
                except Exception:
                    vol = np.nan
                try:
                    max_dd, _ = calculate_max_drawdown(vals)
                except Exception:
                    max_dd = np.nan

                metrics_summary[name] = {
                    'Total Return': total_return,
                    'CAGR': (cagr * 100) if isinstance(cagr, (int, float)) and not np.isnan(cagr) else np.nan,
                    'Volatility': (vol * 100) if isinstance(vol, (int, float)) and not np.isnan(vol) else np.nan,
                    'Max Drawdown': (max_dd * 100) if isinstance(max_dd, (int, float)) and not np.isnan(max_dd) else np.nan,
                }

            if metrics_summary:
                df_metrics = pd.DataFrame(metrics_summary).T
                # Ensure numeric columns
                for c in df_metrics.columns:
                    df_metrics[c] = pd.to_numeric(df_metrics[c], errors='coerce')

                # Create grouped bar chart
                fig_metrics = go.Figure()
                metric_order = ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown']
                colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
                for i, metric in enumerate(metric_order):
                    if metric in df_metrics.columns:
                        y_values = df_metrics[metric].values
                        
                        # Transform values for symmetric log-like display
                        # Positive values: log scale, Negative values: -log(abs(value))
                        transformed_values = []
                        for val in y_values:
                            if pd.isna(val):
                                transformed_values.append(np.nan)
                            elif val > 0:
                                transformed_values.append(np.log10(val + 1))  # +1 to handle 0
                            elif val < 0:
                                transformed_values.append(-np.log10(abs(val) + 1))  # Negative log for negative values
                            else:  # val == 0
                                transformed_values.append(0)
                        
                        fig_metrics.add_trace(go.Bar(
                            x=df_metrics.index,
                            y=transformed_values,
                            name=metric,
                            marker_color=colors[i % len(colors)],
                            text=[f"{v:.2f}%" if not pd.isna(v) else 'N/A' for v in y_values],
                            textposition='auto',
                            showlegend=True
                        ))

                fig_metrics.update_layout(
                    title='Portfolio Variation Summary (percent)',
                    barmode='group',
                    template='plotly_dark',
                    yaxis=dict(
                        title='Percent (Log Scale)', 
                        ticksuffix='%',
                        showticklabels=False,  # Remove Y-axis tick labels
                        # Linear scale since we transformed the values manually
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=10)
                    ),
                    height=520,
                    margin=dict(l=60, r=40, t=80, b=120),
                )

                st.plotly_chart(fig_metrics, use_container_width=True, key="multi_metrics_chart")
        except Exception as e:
            pass

        # --- Monthly returns heatmap: rows = portfolios, columns = Year-Month, values = monthly % change ---
        try:
            # Build a DataFrame of monthly returns for each portfolio
            monthly_returns = {}
            for name, series_obj in st.session_state.multi_all_results.items():
                ser_no = series_obj['no_additions'] if isinstance(series_obj, dict) and 'no_additions' in series_obj else series_obj if isinstance(series_obj, pd.Series) else None
                if ser_no is None or len(ser_no) < 2:
                    continue
                # Resample to month-end and compute percent change
                try:
                    # Use month-end resample with 'ME' alias to avoid FutureWarning; keep as DatetimeIndex
                    ser_month = ser_no.resample('ME').last()
                    pct_month = ser_month.pct_change().dropna() * 100
                    # label months as 'YYYY-MM' using DatetimeIndex to avoid PeriodArray conversion
                    pct_month.index = pct_month.index.strftime('%Y-%m')
                    monthly_returns[name] = pct_month
                except Exception:
                    continue

            if monthly_returns:
                # Align indexes (months) across portfolios
                all_months = sorted(list({m for ser in monthly_returns.values() for m in ser.index}))
                heat_data = pd.DataFrame(index=list(monthly_returns.keys()), columns=all_months)
                for name, ser in monthly_returns.items():
                    for m, v in ser.items():
                        heat_data.at[name, m] = v
                heat_data = heat_data.astype(float)

                # Create heatmap with Plotly
                fig_heat = go.Figure(data=go.Heatmap(
                    z=heat_data.values,
                    x=heat_data.columns.astype(str),
                    y=heat_data.index.astype(str),
                    colorscale='RdYlGn',
                    colorbar=dict(title='Monthly %'),
                    hovertemplate='Portfolio: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
                ))
                fig_heat.update_layout(
                    title='Monthly Returns Heatmap (rows=portfolios, columns=year-month)',
                    xaxis_nticks=20,
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_heat, use_container_width=True, key="multi_heatmap_chart")
        except Exception as e:
            pass

        # Recompute Final Performance Statistics from stored results to ensure they use the no-additions series
        if 'multi_all_results' in st.session_state and st.session_state.multi_all_results:
            # Helper to extract no-additions series whether stored as dict or Series
            def get_no_additions(series_or_dict):
                return series_or_dict['no_additions'] if isinstance(series_or_dict, dict) and 'no_additions' in series_or_dict else series_or_dict

            # MWRR will be calculated fresh for each portfolio in the recomputation loop

            recomputed_stats = {}

            def scale_pct(val):
                if val is None or (isinstance(val, (int, float)) and np.isnan(val)):
                    return np.nan
                if isinstance(val, str):
                    return val  # Return strings as-is (like "N/A")
                if isinstance(val, (int, float)) and -1.5 < val < 1.5:
                    return val * 100
                return val

            def clamp_stat(val, stat_type):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A"
                v = scale_pct(val)
                
                # If scale_pct returned a string (like "N/A"), return it as-is
                if isinstance(v, str):
                    return v
                
                # Apply specific scaling for Total Return before clamping
                if stat_type == "Total Return":
                    v = v * 100
                
                # Clamping logic - separate Total Return from other percentage stats
                if stat_type in ["CAGR", "Volatility", "MWRR"]:
                    if isinstance(v, (int, float)) and (v < 0 or v > 100):
                        return "N/A"
                elif stat_type == "Total Return":
                    if isinstance(v, (int, float)) and v < 0:  # Only check for negative values
                        return "N/A"
                elif stat_type == "MaxDrawdown":
                    if isinstance(v, (int, float)) and (v < -100 or v > 0):
                        return "N/A"
                
                return f"{v:.2f}%" if stat_type in ["CAGR", "MaxDrawdown", "Volatility", "MWRR", "Total Return"] else f"{v:.3f}" if isinstance(v, float) else v

            for name, series_obj in st.session_state.multi_all_results.items():
                ser_noadd = get_no_additions(series_obj)
                if ser_noadd is None or len(ser_noadd) < 2:
                    recomputed_stats[name] = {
                        "Total Return": "N/A",
                        "CAGR": "N/A",
                        "MaxDrawdown": "N/A",
                        "Volatility": "N/A",
                        "Sharpe": "N/A",
                        "Sortino": "N/A",
                        "UlcerIndex": "N/A",
                        "UPI": "N/A",
                        "Beta": "N/A",
                        "MWRR": "N/A",
                        # Final values with and without additions (if available)
                        "Final Value (with)": (series_obj['with_additions'].iloc[-1] if isinstance(series_obj, dict) and 'with_additions' in series_obj and len(series_obj['with_additions'])>0 else "N/A"),
                        "Final Value (no_additions)": (ser_noadd.iloc[-1] if isinstance(ser_noadd, pd.Series) and len(ser_noadd)>0 else "N/A")
                    }
                    continue

                stats_values = ser_noadd.values
                stats_dates = ser_noadd.index
                stats_returns = pd.Series(stats_values, index=stats_dates).pct_change().fillna(0)
                
                # Calculate total return (no additions)
                total_return = None
                if len(stats_values) > 0:
                    initial_val = stats_values[0]
                    final_val = stats_values[-1]
                    if initial_val > 0:
                        total_return = (final_val / initial_val - 1)  # Return as decimal, not percentage
                
                cagr = calculate_cagr(stats_values, stats_dates)
                max_dd, drawdowns = calculate_max_drawdown(stats_values)
                vol = calculate_volatility(stats_returns)
                
                # Use 2% annual risk-free rate (same as Backtest_Engine.py default)
                risk_free_rate = 0.02
                sharpe = calculate_sharpe(stats_returns, risk_free_rate)
                sortino = calculate_sortino(stats_returns, risk_free_rate)
                ulcer = calculate_ulcer_index(pd.Series(stats_values, index=stats_dates))
                upi = calculate_upi(cagr, ulcer)
                # Compute Beta based on the no-additions portfolio returns and the portfolio's benchmark (if available)
                beta = np.nan
                # Find the portfolio config to get benchmark ticker
                cfg_for_name = next((c for c in st.session_state.multi_backtest_portfolio_configs if c['name'] == name), None)
                if cfg_for_name:
                    bench_ticker = cfg_for_name.get('benchmark_ticker')
                    raw_data = st.session_state.get('multi_backtest_raw_data')
                    if bench_ticker and raw_data and bench_ticker in raw_data:
                        # get benchmark price_change series aligned to ser_noadd index
                        try:
                            bench_df = raw_data[bench_ticker].reindex(ser_noadd.index)
                            if 'Price_change' in bench_df.columns:
                                bench_returns = bench_df['Price_change'].fillna(0)
                            else:
                                bench_returns = bench_df['Close'].pct_change().fillna(0)

                            portfolio_returns = pd.Series(stats_values, index=stats_dates).pct_change().fillna(0)
                            common_idx = portfolio_returns.index.intersection(bench_returns.index)
                            if len(common_idx) >= 2:
                                pr = portfolio_returns.reindex(common_idx).dropna()
                                br = bench_returns.reindex(common_idx).dropna()
                                common_idx2 = pr.index.intersection(br.index)
                                if len(common_idx2) >= 2 and br.loc[common_idx2].var() != 0:
                                    cov = pr.loc[common_idx2].cov(br.loc[common_idx2])
                                    var = br.loc[common_idx2].var()
                                    beta = cov / var
                        except Exception as e:
                            pass
                
                # Calculate MWRR for this portfolio using the complete cash flow series
                mwrr_val = np.nan  # Use NaN instead of "N/A" string
                if isinstance(series_obj, dict) and 'with_additions' in series_obj:
                    portfolio_values = series_obj['with_additions']
                    # Reconstruct cash flows for this portfolio
                    cfg_for_name = next((c for c in st.session_state.multi_backtest_portfolio_configs if c['name'] == name), None)
                    if cfg_for_name and len(portfolio_values) > 0:
                        cash_flows = pd.Series(0.0, index=portfolio_values.index)
                        # Initial investment: negative cash flow on first date
                        cash_flows.iloc[0] = -cfg_for_name.get('initial_value', 0)
                        # Periodic additions: negative cash flow on their respective dates
                        dates_added = get_dates_by_freq(cfg_for_name.get('added_frequency'), portfolio_values.index[0], portfolio_values.index[-1], portfolio_values.index)
                        for d in dates_added:
                            if d in cash_flows.index and d != cash_flows.index[0]:
                                cash_flows.loc[d] -= cfg_for_name.get('added_amount', 0)
                        # Final value: positive cash flow on last date for MWRR
                        cash_flows.iloc[-1] += portfolio_values.iloc[-1]
                        # Calculate MWRR
                        mwrr = calculate_mwrr(portfolio_values, cash_flows, portfolio_values.index)
                        mwrr_val = mwrr

                # Calculate total money added for this portfolio
                total_money_added = np.nan  # Use NaN instead of "N/A" string
                cfg_for_name = next((c for c in st.session_state.multi_backtest_portfolio_configs if c['name'] == name), None)
                if cfg_for_name and isinstance(ser_noadd, pd.Series) and len(ser_noadd) > 0:
                    total_money_added = calculate_total_money_added(cfg_for_name, ser_noadd.index[0], ser_noadd.index[-1])
                
                # Calculate total return based on total money contributed
                total_return_contributed = np.nan  # Use NaN instead of "N/A" string
                if isinstance(series_obj, dict) and 'with_additions' in series_obj and len(series_obj['with_additions']) > 0:
                    final_value_with_additions = series_obj['with_additions'].iloc[-1]
                    if isinstance(total_money_added, (int, float)) and total_money_added > 0:
                        total_return_contributed = (final_value_with_additions / total_money_added - 1)  # Return as decimal

                recomputed_stats[name] = {
                    "Total Return": clamp_stat(total_return, "Total Return"),
                    "Total Return (Contributed)": clamp_stat(total_return_contributed, "Total Return"),
                    "CAGR": clamp_stat(cagr, "CAGR"),
                    "MaxDrawdown": clamp_stat(max_dd, "MaxDrawdown"),
                    "Volatility": clamp_stat(vol, "Volatility"),
                    "Sharpe": clamp_stat(sharpe / 100 if isinstance(sharpe, (int, float)) and pd.notna(sharpe) else sharpe, "Sharpe"),
                    "Sortino": clamp_stat(sortino / 100 if isinstance(sortino, (int, float)) and pd.notna(sortino) else sortino, "Sortino"),
                    "UlcerIndex": clamp_stat(ulcer, "UlcerIndex"),
                    "UPI": clamp_stat(upi / 100 if isinstance(upi, (int, float)) and pd.notna(upi) else upi, "UPI"),
                    "Beta": clamp_stat(beta / 100 if isinstance(beta, (int, float)) and pd.notna(beta) else beta, "Beta"),
                    "MWRR": mwrr_val,
                    # Final values with and without additions
                    "Final Value (with)": (series_obj['with_additions'].iloc[-1] if isinstance(series_obj, dict) and 'with_additions' in series_obj and len(series_obj['with_additions'])>0 else np.nan),
                    "Final Value (no_additions)": (ser_noadd.iloc[-1] if isinstance(ser_noadd, pd.Series) and len(ser_noadd)>0 else np.nan),
                    "Total Money Added": total_money_added
                }

            stats_df_display = pd.DataFrame(recomputed_stats).T
            # Move final value columns to the front and format them as currency
            cols = list(stats_df_display.columns)
            fv_with = 'Final Value (with)'
            fv_no = 'Final Value (no_additions)'
            front = [c for c in [fv_with, fv_no] if c in cols]
            for c in front:
                cols.remove(c)
            cols = front + cols
            stats_df_display = stats_df_display[cols]
            # Rename and format columns to be more descriptive
            stats_df_display.rename(columns={
                'MaxDrawdown': 'Max Drawdown', 
                'UlcerIndex': 'Ulcer Index',
                'Final Value (with)': 'Final Portfolio Value',
                'Final Value (no_additions)': 'Final Value (No Contributions)',
                'Total Return (Contributed)': 'Total Return (All Money)'
            }, inplace=True)
            # Ensure ordering: Beta then MWRR at end, Total Return columns and Total Money Added at the very end
            cols = list(stats_df_display.columns)
            if 'Beta' in cols and 'MWRR' in cols and 'Total Return' in cols and 'Total Return (All Money)' in cols and 'Total Money Added' in cols:
                cols.remove('Beta'); cols.remove('MWRR'); cols.remove('Total Return'); cols.remove('Total Return (All Money)'); cols.remove('Total Money Added')
                cols.extend(['Beta','MWRR','Total Return','Total Return (All Money)','Total Money Added'])
                stats_df_display = stats_df_display[cols]

            # Display start and end dates next to the title
            col_title, col_dates = st.columns([2, 1])
            with col_title:
                st.subheader("Final Performance Statistics")
            with col_dates:
                if 'multi_all_results' in st.session_state and st.session_state.multi_all_results:
                    # Get the first portfolio's dates (they should all be the same)
                    first_portfolio = next(iter(st.session_state.multi_all_results.values()))
                    if isinstance(first_portfolio, dict) and 'no_additions' in first_portfolio:
                        series = first_portfolio['no_additions']
                        if hasattr(series, 'index') and len(series.index) > 0:
                            start_date = series.index[0].strftime('%Y-%m-%d')
                            end_date = series.index[-1].strftime('%Y-%m-%d')
                            st.markdown(f"**📅 Period:** {start_date} to {end_date}")
                        else:
                            st.markdown("**📅 Period:** N/A")
                    else:
                        st.markdown("**📅 Period:** N/A")
                else:
                    st.markdown("**📅 Period:** N/A")
            # Format currency for final value columns if present
            fmt_map_display = {}
            if 'Final Portfolio Value' in stats_df_display.columns:
                fmt_map_display['Final Portfolio Value'] = '${:,.2f}'
            if 'Final Value (No Contributions)' in stats_df_display.columns:
                fmt_map_display['Final Value (No Contributions)'] = '${:,.2f}'
            if 'Total Money Added' in stats_df_display.columns:
                fmt_map_display['Total Money Added'] = '${:,.2f}'
            # Format MWRR as percentage - but only if it contains numeric data
            if 'MWRR' in stats_df_display.columns:
                # Check if MWRR column has any non-N/A values
                mwrr_has_numeric = False
                for value in stats_df_display['MWRR']:
                    if pd.notna(value) and value != 'N/A' and value != '':
                        try:
                            float(value)
                            mwrr_has_numeric = True
                            break
                        except (ValueError, TypeError):
                            pass
                if mwrr_has_numeric:
                    fmt_map_display['MWRR'] = '{:.2f}%'
            
            # Create tooltips for each column
            tooltip_data = {
                'Total Return': 'Return based on initial investment only. Formula: (Final Value / Initial Investment) - 1',
                'Total Return (All Money)': 'Return based on all money contributed. Formula: (Final Portfolio Value / Total Money Added) - 1',
                'CAGR': 'Compound Annual Growth Rate. Average annual return over the entire period.',
                'Max Drawdown': 'Largest peak-to-trough decline. Shows the worst loss from a peak.',
                'Volatility': 'Standard deviation of returns. Measures price variability.',
                'Sharpe': 'Excess return per unit of total volatility. >1 good, >2 very good, >3 excellent.',
                'Sortino': 'Excess return per unit of downside volatility. >1 good, >2 very good, >3 excellent.',
                'Ulcer Index': 'Average depth of drawdowns. <5 excellent, 5-10 moderate, >10 high.',
                'UPI': 'Ulcer Performance Index. Excess return relative to Ulcer Index. >1 good, >2 very good, >3 excellent.',
                'Beta': 'Portfolio volatility relative to benchmark. <1 less volatile, >1 more volatile than market.',
                'MWRR': 'Money-Weighted Rate of Return. Accounts for timing and size of cash flows.',
                'Final Portfolio Value': 'Final value including all contributions and investment returns.',
                'Final Value (No Contributions)': 'Final value excluding additional contributions (only initial investment).',
                'Total Money Added': 'Total amount of money contributed (initial + periodic additions).'
            }
            
            # Clean the dataframe to handle problematic values before styling
            stats_df_clean = stats_df_display.copy()
            
            # Replace problematic values that can't be formatted
            for col in stats_df_clean.columns:
                for idx in stats_df_clean.index:
                    value = stats_df_clean.loc[idx, col]
                    if pd.isna(value) or value is None or value == 'N/A' or value == '':
                        stats_df_clean.loc[idx, col] = 'N/A'
                    elif isinstance(value, str) and value.strip() == '':
                        stats_df_clean.loc[idx, col] = 'N/A'
            
            # Create a safe formatting map that only includes numeric columns
            safe_fmt_map = {}
            has_problematic_data = False
            
            # Check if any column has problematic data
            for col in stats_df_clean.columns:
                for value in stats_df_clean[col]:
                    if pd.isna(value) or value == 'N/A' or value == '':
                        has_problematic_data = True
                        break
                if has_problematic_data:
                    break
            
            # Only apply formatting if no problematic data
            if not has_problematic_data:
                for col, fmt in fmt_map_display.items():
                    if col in stats_df_clean.columns:
                        # Check if the column contains numeric data
                        numeric_count = 0
                        total_count = 0
                        for value in stats_df_clean[col]:
                            if pd.notna(value) and value != 'N/A' and value != '':
                                total_count += 1
                                try:
                                    float(value)
                                    numeric_count += 1
                                except (ValueError, TypeError):
                                    pass
                        
                        # Only apply formatting if most values are numeric
                        if total_count > 0 and numeric_count / total_count > 0.5:
                            safe_fmt_map[col] = fmt
            
            # Add tooltips to the dataframe
            if safe_fmt_map and not has_problematic_data:
                try:
                    styled_df = stats_df_clean.style.format(safe_fmt_map)
                except Exception as e:
                    styled_df = stats_df_clean
            else:
                # Skip styling entirely if there's problematic data
                styled_df = stats_df_clean
            
            # Add tooltips using HTML
            tooltip_html = "<div style='background-color: #1e1e1e; color: white; padding: 10px; border-radius: 5px; font-size: 12px;'>"
            tooltip_html += "<b>Column Definitions:</b><br><br>"
            for col, tooltip in tooltip_data.items():
                if col in stats_df_clean.columns:
                    tooltip_html += f"<b>{col}:</b> {tooltip}<br><br>"
            tooltip_html += "</div>"
            
            # Display tooltip info
            with st.expander("ℹ️ Column Definitions", expanded=False):
                st.markdown(tooltip_html, unsafe_allow_html=True)
            
            # Display the dataframe with multiple fallback options
            try:
                st.dataframe(styled_df, use_container_width=True)
            except Exception as e1:
                try:
                    st.dataframe(stats_df_clean, use_container_width=True)
                except Exception as e2:
                    # Last resort: convert all values to strings
                    stats_df_strings = stats_df_clean.astype(str)
                    st.dataframe(stats_df_strings, use_container_width=True)
            
            # Store the statistics table as a Plotly figure for PDF export
            try:
                import plotly.graph_objects as go
                # Create a Plotly table from the stats DataFrame with MUCH better formatting
                # Format the values to remove excessive decimals with proper error handling
                formatted_values = []
                for col in stats_df_display.columns:
                    col_values = []
                    for name in stats_df_display.index:
                        value = stats_df_display.loc[name, col]
                        
                        # Handle NaN, None, and string values
                        if pd.isna(value) or value is None or value == 'N/A' or value == '':
                            col_values.append('N/A')
                            continue
                        
                        try:
                            if 'Portfolio Value' in col or 'Final Value' in col:
                                # Portfolio values: full numbers only (no decimals)
                                col_values.append(f"${float(value):,.0f}")
                            elif 'MWRR' in col:
                                # MWRR: max 2 decimal places
                                col_values.append(f"{float(value):.2f}%")
                            elif 'Total Money Added' in col:
                                # Money added: full numbers only
                                col_values.append(f"${float(value):,.0f}")
                            else:
                                # For other numeric columns, try to format as float
                                try:
                                    float_val = float(value)
                                    if 'Ratio' in col or 'Index' in col:
                                        col_values.append(f"{float_val:.3f}")
                                    elif 'Drawdown' in col or 'Volatility' in col or 'CAGR' in col:
                                        col_values.append(f"{float_val:.2f}%")
                                    else:
                                        col_values.append(f"{float_val:.2f}")
                                except (ValueError, TypeError):
                                    # If conversion fails, keep original value
                                    col_values.append(str(value))
                        except (ValueError, TypeError):
                            # If any conversion fails, use original value
                            col_values.append(str(value))
                    
                    formatted_values.append(col_values)
                
                fig_stats = go.Figure(data=[go.Table(
                    header=dict(
                        values=['Portfolio'] + list(stats_df_display.columns),
                        fill_color='rgb(51, 102, 153)',
                        align='center',
                        font=dict(color='white', size=14, family='Arial Black')  # Bigger, bolder font
                    ),
                    cells=dict(
                        values=[stats_df_display.index] + formatted_values,
                        fill_color='rgb(242, 242, 242)',
                        align='center',
                        font=dict(color='black', size=12, family='Arial'),  # Bigger font
                        height=35  # Taller cells for better readability
                    ),
                    columnwidth=[0.15] + [0.85/len(stats_df_display.columns)] * len(stats_df_display.columns)  # Portfolio column wider, others equal
                )])
                fig_stats.update_layout(
                    title="Final Performance Statistics",
                    title_x=0.5,
                    width=2000,  # Much wider to fit all columns
                    height=600,   # Taller for better spacing
                    margin=dict(l=20, r=20, t=50, b=20)  # Better margins
                )
                st.session_state.fig_stats = fig_stats
            except Exception as e:
                pass

        # Portfolio Configuration Comparison Table
        st.subheader("Portfolio Configuration Comparison")
        
        # Create configuration comparison dataframe
        config_data = {}
        for cfg in st.session_state.multi_backtest_portfolio_configs:
            portfolio_name = cfg.get('name', 'Unknown')
            
            # Extract configuration details
            config_data[portfolio_name] = {
                'Initial Investment': f"${cfg.get('initial_value', 0):,.2f}",
                'Added Amount': f"${cfg.get('added_amount', 0):,.2f}",
                'Added Frequency': cfg.get('added_frequency', 'None'),
                'Rebalancing Frequency': cfg.get('rebalancing_frequency', 'None'),
                'Use Momentum': 'Yes' if cfg.get('use_momentum', False) else 'No',
                'Momentum Strategy': cfg.get('momentum_strategy', 'N/A'),
                'Negative Momentum Strategy': cfg.get('negative_momentum_strategy', 'N/A'),
                'Number of Stocks': len(cfg.get('stocks', [])),
                'Stocks': ', '.join([s.get('ticker', '') for s in cfg.get('stocks', [])]),
                'Benchmark': cfg.get('benchmark_ticker', 'N/A'),
                'Momentum Windows': str(cfg.get('momentum_windows', [])),
                'Beta Enabled': 'Yes' if cfg.get('use_beta', False) else 'No',
                'Volatility Enabled': 'Yes' if cfg.get('use_vol', False) else 'No',
                'Beta Window': f"{cfg.get('beta_window_days', 0)} days" if cfg.get('use_beta', False) else 'N/A',
                'Volatility Window': f"{cfg.get('vol_window_days', 0)} days" if cfg.get('use_vol', False) else 'N/A',
                'Beta Exclude Days': f"{cfg.get('beta_exclude_days', 0)} days" if cfg.get('use_beta', False) else 'N/A',
                'Volatility Exclude Days': f"{cfg.get('vol_exclude_days', 0)} days" if cfg.get('use_vol', False) else 'N/A',
                'Minimal Threshold': f"{cfg.get('minimal_threshold_percent', 2.0):.1f}%" if cfg.get('use_minimal_threshold', False) else 'Disabled'
            }
        
        config_df = pd.DataFrame(config_data).T
        
        # Format the configuration table
        st.dataframe(config_df, use_container_width=True)

        st.subheader("Yearly Performance (Interactive Table)")
        all_years = st.session_state.multi_backtest_all_years
        years = sorted(list(set(y.year for ser in all_years.values() for y in ser.index)))
        # Order portfolio columns according to the portfolio_configs order so new portfolios are added to the right
        names = [cfg['name'] for cfg in st.session_state.multi_backtest_portfolio_configs if cfg.get('name') in all_years]

        # Corrected yearly table creation
        df_yearly_pct_data = {}
        df_yearly_final_data = {}
        for name in names:
            pct_list = []
            final_list = []
            # with-additions yearly series (used for final values)
            ser_with = all_years.get(name) if isinstance(all_years, dict) else None
            # no-additions yearly series (used for percent-change to avoid skew)
            ser_noadd = None
            try:
                series_obj = st.session_state.multi_all_results.get(name)
                if isinstance(series_obj, dict) and 'no_additions' in series_obj:
                    ser_noadd = series_obj['no_additions'].resample('YE').last()
                elif isinstance(series_obj, pd.Series):
                    ser_noadd = series_obj.resample('YE').last()
            except Exception:
                ser_noadd = None

            for y in years:
                # get year slices
                ser_year_with = ser_with[ser_with.index.year == y] if ser_with is not None else pd.Series()
                ser_year_no = ser_noadd[ser_noadd.index.year == y] if ser_noadd is not None else pd.Series()

                start_val_for_year = None
                if y == min(years):
                    config_for_name = next((c for c in st.session_state.multi_backtest_portfolio_configs if c['name'] == name), None)
                    if config_for_name:
                        initial_val_of_config = config_for_name['initial_value']
                        if initial_val_of_config > 0:
                            start_val_for_year = initial_val_of_config
                else:
                    prev_year = y - 1
                    # Use no-additions previous year end as the start value for pct change
                    prev_ser_year_no = ser_noadd[ser_noadd.index.year == prev_year] if ser_noadd is not None else pd.Series()
                    if not prev_ser_year_no.empty:
                        start_val_for_year = prev_ser_year_no.iloc[-1]

                # Percent change computed from no-additions series
                if not ser_year_no.empty and start_val_for_year is not None:
                    end_val_no = ser_year_no.iloc[-1]
                    if start_val_for_year > 0:
                        pct_change = (end_val_no - start_val_for_year) / start_val_for_year * 100
                    else:
                        pct_change = np.nan
                else:
                    pct_change = np.nan

                # Final value displayed from with-additions series (if available)
                if not ser_year_with.empty:
                    final_value = ser_year_with.iloc[-1]
                else:
                    final_value = np.nan

                pct_list.append(pct_change)
                final_list.append(final_value)

            df_yearly_pct_data[f'{name} % Change'] = pct_list
            df_yearly_final_data[f'{name} Final Value'] = final_list

        df_yearly_pct = pd.DataFrame(df_yearly_pct_data, index=years)
        df_yearly_final = pd.DataFrame(df_yearly_final_data, index=years)
        # Build combined dataframe but preserve the desired column order (selected portfolio first)
        temp_combined = pd.concat([df_yearly_pct, df_yearly_final], axis=1)
        ordered_cols = []
        for nm in names:
            pct_col = f'{nm} % Change'
            val_col = f'{nm} Final Value'
            if pct_col in temp_combined.columns:
                ordered_cols.append(pct_col)
            if val_col in temp_combined.columns:
                ordered_cols.append(val_col)
        # Fallback: if nothing matched, use whatever columns exist
        if not ordered_cols:
            combined_df = temp_combined
        else:
            combined_df = temp_combined[ordered_cols]

        def color_gradient_stock(val):
            if isinstance(val, (int, float)):
                if val > 50:
                    return 'background-color: #004d00'
                elif val > 20:
                    return 'background-color: #1e8449'
                elif val > 5:
                    return 'background-color: #388e3c'
                elif val > 0:
                    return 'background-color: #66bb6a'
                elif val < -50:
                    return 'background-color: #7b0000'
                elif val < -20:
                    return 'background-color: #b22222'
                elif val < -5:
                    return 'background-color: #d32f2f'
                elif val < 0:
                    return 'background-color: #ef5350'
            return ''
        
        # Ensure columns and index are unique (pandas Styler requires unique labels)
        if combined_df.columns.duplicated().any():
            cols = list(combined_df.columns)
            seen = {}
            new_cols = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c} ({seen[c]})")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            combined_df.columns = new_cols

        if combined_df.index.duplicated().any():
            idx = list(map(str, combined_df.index))
            seen_idx = {}
            new_idx = []
            for v in idx:
                if v in seen_idx:
                    seen_idx[v] += 1
                    new_idx.append(f"{v} ({seen_idx[v]})")
                else:
                    seen_idx[v] = 0
                    new_idx.append(v)
            combined_df.index = new_idx

        # Recompute percent and final value column lists after any renaming
        pct_cols = [col for col in combined_df.columns if '% Change' in col]
        final_val_cols = [col for col in combined_df.columns if 'Final Value' in col]

        # Coerce percent columns to numeric so formatting applies correctly
        for col in pct_cols:
            if col in combined_df.columns:
                try:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                except TypeError:
                    # Unexpected column type (not Series/array). Try to coerce via pd.Series or fall back to NaN.
                    try:
                        combined_df[col] = pd.to_numeric(pd.Series(combined_df[col]), errors='coerce')
                    except Exception:
                        combined_df[col] = np.nan

        # Create combined format mapping: percent columns get '%' suffix, final value columns get currency
        fmt_map = {col: '{:,.2f}%' for col in pct_cols if col in combined_df.columns}
        fmt_map.update({col: '${:,.2f}' for col in final_val_cols if col in combined_df.columns})

        styler = combined_df.style
        # Color percent cells with a gradient and then apply formatting in one call
        if pct_cols:
            try:
                # Styler.map is the supported replacement for applymap
                styler = styler.map(color_gradient_stock, subset=pct_cols)
            except Exception:
                # If map still fails (edge cases), skip coloring to avoid breaking the page
                pass
        if fmt_map:
            styler = styler.format(fmt_map, na_rep='N/A')

        st.dataframe(styler, use_container_width=True, hide_index=False)

        # Monthly Performance Table
        st.subheader("Monthly Performance (Interactive Table)")
        # Use the original results data for monthly calculation, not the yearly resampled data
        all_results = st.session_state.multi_all_results
        # Get all available months from the original data
        all_months_data = {}
        for name, results in all_results.items():
            if isinstance(results, dict) and 'with_additions' in results:
                all_months_data[name] = results['with_additions']
            elif isinstance(results, pd.Series):
                all_months_data[name] = results
        
        # Extract all unique year-month combinations from the original data
        months = set()
        for ser in all_months_data.values():
            if not ser.empty:
                for date in ser.index:
                    months.add((date.year, date.month))
        months = sorted(list(months))
        
        # Order portfolio columns according to the portfolio_configs order so new portfolios are added to the right
        names = [cfg['name'] for cfg in st.session_state.multi_backtest_portfolio_configs if cfg.get('name') in all_months_data]

        # Monthly table creation
        df_monthly_pct_data = {}
        df_monthly_final_data = {}
        for name in names:
            pct_list = []
            final_list = []
            # with-additions monthly series (used for final values)
            ser_with = all_months_data.get(name) if isinstance(all_months_data, dict) else None
            # no-additions monthly series (used for percent-change to avoid skew)
            ser_noadd = None
            try:
                series_obj = st.session_state.multi_all_results.get(name)
                if isinstance(series_obj, dict) and 'no_additions' in series_obj:
                    ser_noadd = series_obj['no_additions'].resample('M').last()
                elif isinstance(series_obj, pd.Series):
                    ser_noadd = series_obj.resample('M').last()
            except Exception:
                ser_noadd = None

            for y, m in months:
                # get month slices
                ser_month_with = ser_with[(ser_with.index.year == y) & (ser_with.index.month == m)] if ser_with is not None else pd.Series()
                ser_month_no = ser_noadd[(ser_noadd.index.year == y) & (ser_noadd.index.month == m)] if ser_noadd is not None else pd.Series()

                start_val_for_month = None
                if (y, m) == min(months):
                    config_for_name = next((c for c in st.session_state.multi_backtest_portfolio_configs if c['name'] == name), None)
                    if config_for_name:
                        initial_val_of_config = config_for_name['initial_value']
                        if initial_val_of_config > 0:
                            start_val_for_month = initial_val_of_config
                else:
                    # Find previous month
                    prev_month_idx = months.index((y, m)) - 1
                    if prev_month_idx >= 0:
                        prev_y, prev_m = months[prev_month_idx]
                        # Use no-additions previous month end as the start value for pct change
                        prev_ser_month_no = ser_noadd[(ser_noadd.index.year == prev_y) & (ser_noadd.index.month == prev_m)] if ser_noadd is not None else pd.Series()
                        if not prev_ser_month_no.empty:
                            start_val_for_month = prev_ser_month_no.iloc[-1]

                # Percent change computed from no-additions series
                if not ser_month_no.empty and start_val_for_month is not None:
                    end_val_no = ser_month_no.iloc[-1]
                    if start_val_for_month > 0:
                        pct_change = (end_val_no - start_val_for_month) / start_val_for_month * 100
                    else:
                        pct_change = np.nan
                else:
                    pct_change = np.nan

                # Final value displayed from with-additions series (if available)
                if not ser_month_with.empty:
                    final_value = ser_month_with.iloc[-1]
                else:
                    final_value = np.nan

                pct_list.append(pct_change)
                final_list.append(final_value)

            df_monthly_pct_data[f'{name} % Change'] = pct_list
            df_monthly_final_data[f'{name} Final Value'] = final_list

        df_monthly_pct = pd.DataFrame(df_monthly_pct_data, index=[f"{y}-{m:02d}" for y, m in months])
        df_monthly_final = pd.DataFrame(df_monthly_final_data, index=[f"{y}-{m:02d}" for y, m in months])
        # Build combined dataframe but preserve the desired column order (selected portfolio first)
        temp_combined_monthly = pd.concat([df_monthly_pct, df_monthly_final], axis=1)
        ordered_cols_monthly = []
        for nm in names:
            pct_col = f'{nm} % Change'
            val_col = f'{nm} Final Value'
            if pct_col in temp_combined_monthly.columns:
                ordered_cols_monthly.append(pct_col)
            if val_col in temp_combined_monthly.columns:
                ordered_cols_monthly.append(val_col)
        # Fallback: if nothing matched, use whatever columns exist
        if not ordered_cols_monthly:
            combined_df_monthly = temp_combined_monthly
        else:
            combined_df_monthly = temp_combined_monthly[ordered_cols_monthly]

        # Ensure columns and index are unique (pandas Styler requires unique labels)
        if combined_df_monthly.columns.duplicated().any():
            cols = list(combined_df_monthly.columns)
            seen = {}
            new_cols = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c} ({seen[c]})")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            combined_df_monthly.columns = new_cols

        if combined_df_monthly.index.duplicated().any():
            idx = list(map(str, combined_df_monthly.index))
            seen_idx = {}
            new_idx = []
            for v in idx:
                if v in seen_idx:
                    seen_idx[v] += 1
                    new_idx.append(f"{v} ({seen_idx[v]})")
                else:
                    seen_idx[v] = 0
                    new_idx.append(v)
            combined_df_monthly.index = new_idx

        # Recompute percent and final value column lists after any renaming
        pct_cols_monthly = [col for col in combined_df_monthly.columns if '% Change' in col]
        final_val_cols_monthly = [col for col in combined_df_monthly.columns if 'Final Value' in col]

        # Coerce percent columns to numeric so formatting applies correctly
        for col in pct_cols_monthly:
            if col in combined_df_monthly.columns:
                try:
                    combined_df_monthly[col] = pd.to_numeric(combined_df_monthly[col], errors='coerce')
                except TypeError:
                    # Unexpected column type (not Series/array). Try to coerce via pd.Series or fall back to NaN.
                    try:
                        combined_df_monthly[col] = pd.to_numeric(pd.Series(combined_df_monthly[col]), errors='coerce')
                    except Exception:
                        combined_df_monthly[col] = np.nan

        # Create combined format mapping: percent columns get '%' suffix, final value columns get currency
        fmt_map_monthly = {col: '{:,.2f}%' for col in pct_cols_monthly if col in combined_df_monthly.columns}
        fmt_map_monthly.update({col: '${:,.2f}' for col in final_val_cols_monthly if col in combined_df_monthly.columns})

        styler_monthly = combined_df_monthly.style
        # Color percent cells with a gradient and then apply formatting in one call
        if pct_cols_monthly:
            try:
                # Styler.map is the supported replacement for applymap
                styler_monthly = styler_monthly.map(color_gradient_stock, subset=pct_cols_monthly)
            except Exception:
                # If map still fails (edge cases), skip coloring to avoid breaking the page
                pass
        if fmt_map_monthly:
            styler_monthly = styler_monthly.format(fmt_map_monthly, na_rep='N/A')

        st.dataframe(styler_monthly, use_container_width=True, hide_index=False)

        st.markdown("**Detailed Portfolio Information**")
        # Make the selector visually prominent
        st.markdown(
            "<div style='background:#0b1221;padding:12px;border-radius:8px;margin-bottom:8px;'>"
            "<div style='font-size:16px;font-weight:700;color:#ffffff;margin-bottom:6px;'>Select a portfolio for detailed view</div>"
            "</div>", unsafe_allow_html=True)

        # NUCLEAR APPROACH: Store selection by portfolio name, not display index
        portfolio_configs = st.session_state.get('multi_backtest_portfolio_configs', [])
        
        # Get all available portfolio names
        available_portfolio_names = [cfg.get('name', 'Portfolio') for cfg in portfolio_configs]
        extra_names = [n for n in st.session_state.get('multi_all_results', {}).keys() if n not in available_portfolio_names]
        all_portfolio_names = available_portfolio_names + extra_names
        
        # Initialize persistent selection by name
        if "multi_backtest_selected_portfolio_name" not in st.session_state:
            st.session_state["multi_backtest_selected_portfolio_name"] = all_portfolio_names[0] if all_portfolio_names else "No portfolios"
        
        # Ensure the selected name is still valid
        if st.session_state["multi_backtest_selected_portfolio_name"] not in all_portfolio_names and all_portfolio_names:
            st.session_state["multi_backtest_selected_portfolio_name"] = all_portfolio_names[0]
        
        # Create display options with index prefixes for uniqueness
        display_options = [f"{i} - {name}" for i, name in enumerate(all_portfolio_names)]
        
        # Find the current selection index
        current_selection_index = 0
        if st.session_state["multi_backtest_selected_portfolio_name"] in all_portfolio_names:
            current_selection_index = all_portfolio_names.index(st.session_state["multi_backtest_selected_portfolio_name"])
        
        # Place the selectbox in its own column to make it larger/centered
        # Build a prominent action row: selector + colored 'View' button
        left_col, mid_col, right_col = st.columns([1, 3, 1])
        with mid_col:
            st.markdown("<div style='display:flex; gap:8px; align-items:center;'>", unsafe_allow_html=True)
            def update_selected_portfolio():
                selected_display = st.session_state.get("multi_backtest_detail_portfolio_selector")
                if selected_display:
                    try:
                        prefix, rest = selected_display.split(' - ', 1)
                        if prefix.startswith('extra_'):
                            # extra entries use the rest as the name
                            st.session_state["multi_backtest_selected_portfolio_name"] = rest
                        else:
                            idx = int(prefix)
                            st.session_state["multi_backtest_selected_portfolio_name"] = all_portfolio_names[idx]
                    except Exception:
                        st.session_state["multi_backtest_selected_portfolio_name"] = selected_display

            selected_display = st.selectbox(
                "Select portfolio for details", 
                options=display_options, 
                index=current_selection_index,
                key="multi_backtest_detail_portfolio_selector", 
                help='Choose which portfolio to inspect in detail', 
                label_visibility='collapsed',
                on_change=update_selected_portfolio
            )
            # Add a prominent view button with a professional color
            view_clicked = st.button("View Details", key='view_details_btn')
            st.markdown("</div>", unsafe_allow_html=True)

        # Map display label back to actual portfolio name
        selected_portfolio_detail = st.session_state["multi_backtest_selected_portfolio_name"]

        if selected_portfolio_detail:
            # Highlight the selected portfolio and optionally expand details when the View button is used
            st.markdown(f"<div style='padding:8px 12px;background:#04293a;border-radius:6px;margin-top:8px;'><strong style='color:#bde0fe;'>Showing details for:</strong> <span style='font-size:16px;color:#ffffff;margin-left:8px;'>{selected_portfolio_detail}</span></div>", unsafe_allow_html=True)
            if view_clicked:
                # No-op here; the detail panels below will render based on selected_portfolio_detail. Keep a small indicator
                st.success(f"Loaded details for {selected_portfolio_detail}")
            # Table 1: Historical Allocations
            if selected_portfolio_detail in st.session_state.multi_all_allocations:
                st.markdown("---")
                st.markdown(f"**Historical Allocations for {selected_portfolio_detail}**")
                # Ensure proper DataFrame structure with explicit column names
                allocations_df_raw = pd.DataFrame(st.session_state.multi_all_allocations[selected_portfolio_detail]).T
                
                # Handle case where only CASH exists - ensure column name is preserved
                if allocations_df_raw.empty or (len(allocations_df_raw.columns) == 1 and allocations_df_raw.columns[0] is None):
                    # Reconstruct DataFrame with proper column names
                    processed_data = {}
                    for date, alloc_dict in st.session_state.multi_all_allocations[selected_portfolio_detail].items():
                        processed_data[date] = {}
                        for ticker, value in alloc_dict.items():
                            if ticker is None:
                                processed_data[date]['CASH'] = value
                            else:
                                processed_data[date][ticker] = value
                    allocations_df_raw = pd.DataFrame(processed_data).T
                
                allocations_df_raw.index.name = "Date"
                
                # Corrected styling logic for alternating row colors
                def highlight_rows_by_index(s):
                    is_even_row = allocations_df_raw.index.get_loc(s.name) % 2 == 0
                    bg_color = 'background-color: #0e1117' if is_even_row else 'background-color: #262626'
                    return [f'{bg_color}; color: white;'] * len(s)

                styler = allocations_df_raw.mul(100).style.apply(highlight_rows_by_index, axis=1)
                styler.format('{:,.0f}%', na_rep='N/A')
                st.dataframe(styler, use_container_width=True)


            # Table 2: Momentum Metrics and Calculated Weights
            if selected_portfolio_detail in st.session_state.multi_all_metrics:
                st.markdown("---")
                st.markdown(f"**Momentum Metrics and Calculated Weights for {selected_portfolio_detail}**")

                metrics_records = []
                for date, tickers_data in st.session_state.multi_all_metrics[selected_portfolio_detail].items():
                    # Add all asset lines
                    asset_weights = []
                    for ticker, data in tickers_data.items():
                        # Handle None ticker as CASH
                        display_ticker = 'CASH' if ticker is None else ticker
                        if display_ticker != 'CASH':
                            asset_weights.append(data.get('Calculated_Weight', 0))
                        # Filter out any internal-only keys (e.g., 'Composite') so they don't show in the UI
                        filtered_data = {k: v for k, v in (data or {}).items() if k != 'Composite'}
                        
                        # Check if momentum is used for this portfolio
                        portfolio_configs = st.session_state.get('multi_backtest_portfolio_configs', [])
                        portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                        use_momentum = portfolio_cfg.get('use_momentum', True) if portfolio_cfg else True
                        
                        # If momentum is not used, replace Calculated_Weight with target_allocation
                        if not use_momentum:
                            if 'target_allocation' in filtered_data:
                                filtered_data['Calculated_Weight'] = filtered_data['target_allocation']
                            else:
                                # If target_allocation is not available, use the entered allocations from portfolio_cfg
                                ticker_name = display_ticker if display_ticker != 'CASH' else None
                                if ticker_name and portfolio_cfg:
                                    # Find the stock in portfolio_cfg and use its allocation
                                    for stock in portfolio_cfg.get('stocks', []):
                                        if stock.get('ticker', '').strip() == ticker_name:
                                            filtered_data['Calculated_Weight'] = stock.get('allocation', 0)
                                            break
                                elif display_ticker == 'CASH' and portfolio_cfg:
                                    # For CASH, calculate the remaining allocation
                                    total_alloc = sum(stock.get('allocation', 0) for stock in portfolio_cfg.get('stocks', []))
                                    filtered_data['Calculated_Weight'] = max(0, 1.0 - total_alloc)
                        
                        record = {'Date': date, 'Ticker': display_ticker, **filtered_data}
                        metrics_records.append(record)
                    
                    # Ensure CASH line is added if there's non-zero cash in allocations
                    allocs_for_portfolio = st.session_state.multi_all_allocations.get(selected_portfolio_detail) if 'multi_all_allocations' in st.session_state else None
                    if allocs_for_portfolio and date in allocs_for_portfolio:
                        cash_alloc = allocs_for_portfolio[date].get('CASH', 0)
                        if cash_alloc > 0:
                            # Check if CASH is already in metrics_records for this date
                            cash_exists = any(record['Date'] == date and record['Ticker'] == 'CASH' for record in metrics_records)
                            if not cash_exists:
                                # Add CASH line to metrics
                                # Check if momentum is used to determine which weight to show
                                portfolio_configs = st.session_state.get('multi_backtest_portfolio_configs', [])
                                portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                use_momentum = portfolio_cfg.get('use_momentum', True) if portfolio_cfg else True
                                
                                if not use_momentum:
                                    # When momentum is not used, calculate CASH allocation from entered allocations
                                    total_alloc = sum(stock.get('allocation', 0) for stock in portfolio_cfg.get('stocks', []))
                                    cash_weight = max(0, 1.0 - total_alloc)
                                    cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_weight}
                                else:
                                    cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_alloc}
                                metrics_records.append(cash_record)
                    
                    # Add CASH line if fully allocated to cash (100%) or all asset weights are 0% (fallback logic)
                    cash_line_needed = False
                    if 'CASH' in tickers_data or None in tickers_data:
                        cash_data = tickers_data.get('CASH', tickers_data.get(None, {}))
                        cash_weight = cash_data.get('Calculated_Weight', 0)
                        if abs(cash_weight - 1.0) < 1e-6:  # 100% in decimal
                            cash_line_needed = True
                    if all(w == 0 for w in asset_weights) and asset_weights:
                        cash_line_needed = True
                    if cash_line_needed and 'CASH' not in [r['Ticker'] for r in metrics_records if r['Date'] == date]:
                        # If no explicit CASH data, create a default line
                        cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': 1.0}
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
                        # Ensure unique index by adding a counter if needed
                        if metrics_df.duplicated(subset=['Date', 'Ticker']).any():
                            # Add a counter to make indices unique
                            metrics_df['Counter'] = metrics_df.groupby(['Date', 'Ticker']).cumcount()
                            metrics_df['Ticker_Unique'] = metrics_df['Ticker'] + metrics_df['Counter'].astype(str)
                            metrics_df.set_index(['Date', 'Ticker_Unique'], inplace=True)
                        else:
                            metrics_df.set_index(['Date', 'Ticker'], inplace=True)
                        
                    metrics_df_display = metrics_df.copy()

                    # Ensure Momentum column exists and normalize to percent when present
                    if 'Momentum' in metrics_df_display.columns:
                        metrics_df_display['Momentum'] = metrics_df_display['Momentum'].fillna(0) * 100
                    else:
                        metrics_df_display['Momentum'] = np.nan

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
                        date_str = s.name[0]
                        ticker_str = s.name[1]
                        # If this is the CASH row, use dark green background
                        if 'CASH' in ticker_str:
                            return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                        # Otherwise, alternate row colors by date with WHITE TEXT
                        unique_dates = list(metrics_df_display.index.get_level_values(0).unique())
                        is_even = unique_dates.index(date_str) % 2 == 0
                        bg_color = 'background-color: #0e1117' if is_even else 'background-color: #262626'
                        return [f'{bg_color}; color: white;'] * len(s)

                    # Format Calculated_Weight as a percentage if present
                    if 'Calculated_Weight' in metrics_df_display.columns:
                        metrics_df_display['Calculated_Weight'] = metrics_df_display['Calculated_Weight'].fillna(0) * 100
                    # Convert Volatility from decimal (e.g., 0.20) to percent (20.0)
                    if 'Volatility' in metrics_df_display.columns:
                        metrics_df_display['Volatility'] = metrics_df_display['Volatility'].fillna(np.nan) * 100

                    # Corrected styling logic for alternating row colors and momentum color
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

                    st.dataframe(styler_metrics, use_container_width=True)

                                        # --- Allocation plots: Final allocation and last rebalance allocation ---
                    allocs_for_portfolio = st.session_state.multi_all_allocations.get(selected_portfolio_detail) if 'multi_all_allocations' in st.session_state else None
                    if allocs_for_portfolio:
                        try:
                            # Sort allocation dates
                            alloc_dates = sorted(list(allocs_for_portfolio.keys()))
                            if len(alloc_dates) == 0:
                                st.info("No allocation history available to plot.")
                            else:
                                final_date = alloc_dates[-1]
                                last_rebal_date = alloc_dates[-2] if len(alloc_dates) > 1 else alloc_dates[-1]

                                final_alloc = allocs_for_portfolio.get(final_date, {})
                                rebal_alloc = allocs_for_portfolio.get(last_rebal_date, {})

                                # Helper to prepare bar data
                                def prepare_bar_data(d):
                                    labels = []
                                    values = []
                                    for k, v in sorted(d.items(), key=lambda x: (-x[1], x[0])):
                                        labels.append(k)
                                        try:
                                            values.append(float(v) * 100)
                                        except Exception:
                                            values.append(0.0)
                                    return labels, values

                                labels_final, vals_final = prepare_bar_data(final_alloc)
                                labels_rebal, vals_rebal = prepare_bar_data(rebal_alloc)

                                # Add timer for next rebalance date
                                try:
                                    # Get the last rebalance date from allocation history
                                    if len(alloc_dates) > 1:
                                        last_rebal_date_for_timer = alloc_dates[-2]  # Second to last date (excluding today/yesterday)
                                    else:
                                        last_rebal_date_for_timer = alloc_dates[-1] if alloc_dates else None
                                    
                                    # Get rebalancing frequency from portfolio config
                                    portfolio_configs = st.session_state.get('multi_backtest_portfolio_configs', [])
                                    portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                    rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none') if portfolio_cfg else 'none'
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
                                    
                                    if last_rebal_date_for_timer and rebalancing_frequency != 'none':
                                        # Ensure last_rebal_date_for_timer is a naive datetime object
                                        if isinstance(last_rebal_date_for_timer, str):
                                            last_rebal_date_for_timer = pd.to_datetime(last_rebal_date_for_timer)
                                        if hasattr(last_rebal_date_for_timer, 'tzinfo') and last_rebal_date_for_timer.tzinfo is not None:
                                            last_rebal_date_for_timer = last_rebal_date_for_timer.replace(tzinfo=None)
                                        next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                                            rebalancing_frequency, last_rebal_date_for_timer
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
                                                if hasattr(last_rebal_date_for_timer, 'to_pydatetime'):
                                                    last_rebal_datetime = last_rebal_date_for_timer.to_pydatetime()
                                                else:
                                                    last_rebal_datetime = last_rebal_date_for_timer
                                                
                                                total_period = (next_rebalance_datetime - last_rebal_datetime).total_seconds()
                                                elapsed_period = (datetime.now() - last_rebal_datetime).total_seconds()
                                                progress = min(max(elapsed_period / total_period, 0), 1)
                                                
                                                st.progress(progress, text=f"Progress to next rebalance: {progress:.1%}")
                                            
                                            # Create and store timer table figure for PDF export
                                            try:
                                                timer_data = [
                                                    ['Time Until Next Rebalance', format_time_until(time_until)],
                                                    ['Target Rebalance Date', next_date.strftime("%B %d, %Y")],
                                                    ['Rebalancing Frequency', rebalancing_frequency.replace('_', ' ').title()]
                                                ]
                                                
                                                fig_timer = go.Figure(data=[go.Table(
                                                    header=dict(
                                                        values=['Parameter', 'Value'],
                                                        fill_color='#2E86AB',
                                                        align='center',
                                                        font=dict(color='white', size=16, family='Arial Black')
                                                    ),
                                                    cells=dict(
                                                        values=[[row[0] for row in timer_data], [row[1] for row in timer_data]],
                                                        fill_color=[['#F8F9FA', '#FFFFFF'] * 2, ['#F8F9FA', '#FFFFFF'] * 2],
                                                        align='center',
                                                        font=dict(color='black', size=14, family='Arial'),
                                                        height=40
                                                    )
                                                )])
                                                
                                                fig_timer.update_layout(
                                                    title=dict(
                                                        text="⏰ Next Rebalance Timer",
                                                        x=0.5,
                                                        font=dict(size=18, color='#2E86AB', family='Arial Black')
                                                    ),
                                                    width=700,
                                                    height=250,
                                                    margin=dict(l=20, r=20, t=60, b=20)
                                                )
                                                
                                                # Store in session state for PDF export
                                                st.session_state[f'timer_table_{selected_portfolio_detail}'] = fig_timer
                                            except Exception as e:
                                                pass  # Silently ignore timer table creation errors
                                            
                                            # Also create timer tables for ALL portfolios for PDF export
                                            try:
                                                # Get all portfolio configs
                                                all_portfolio_configs = st.session_state.get('multi_backtest_portfolio_configs', [])
                                                snapshot = st.session_state.get('multi_backtest_snapshot_data', {})
                                                last_rebalance_dates = snapshot.get('last_rebalance_dates', {})
                                                
                                                for portfolio_cfg in all_portfolio_configs:
                                                    portfolio_name = portfolio_cfg.get('name', 'Unknown')
                                                    
                                                    # Get rebalancing frequency for this portfolio
                                                    rebal_freq = portfolio_cfg.get('rebalancing_frequency', 'none')
                                                    rebal_freq = rebal_freq.lower()
                                                    rebal_freq = frequency_mapping.get(rebal_freq, rebal_freq)
                                                    
                                                    # Get last rebalance date for this portfolio
                                                    last_rebal_date = last_rebalance_dates.get(portfolio_name)
                                                    
                                                    if last_rebal_date and rebal_freq != 'none':
                                                        # Ensure last_rebal_date is a naive datetime object
                                                        if isinstance(last_rebal_date, str):
                                                            last_rebal_date = pd.to_datetime(last_rebal_date)
                                                        if hasattr(last_rebal_date, 'tzinfo') and last_rebal_date.tzinfo is not None:
                                                            last_rebal_date = last_rebal_date.replace(tzinfo=None)
                                                        
                                                        # Calculate next rebalance for this portfolio
                                                        next_date_port, time_until_port, next_rebalance_datetime_port = calculate_next_rebalance_date(
                                                            rebal_freq, last_rebal_date
                                                        )
                                                        
                                                        if next_date_port and time_until_port:
                                                            # Create timer data for this portfolio
                                                            timer_data_port = [
                                                                ['Time Until Next Rebalance', format_time_until(time_until_port)],
                                                                ['Target Rebalance Date', next_date_port.strftime("%B %d, %Y")],
                                                                ['Rebalancing Frequency', rebal_freq.replace('_', ' ').title()]
                                                            ]
                                                            
                                                            # Create timer table figure for this portfolio
                                                            fig_timer_port = go.Figure(data=[go.Table(
                                                                header=dict(
                                                                    values=['Parameter', 'Value'],
                                                                    fill_color='#2E86AB',
                                                                    align='center',
                                                                    font=dict(color='white', size=16, family='Arial Black')
                                                                ),
                                                                cells=dict(
                                                                    values=[[row[0] for row in timer_data_port], [row[1] for row in timer_data_port]],
                                                                    fill_color=[['#F8F9FA', '#FFFFFF'] * 2, ['#F8F9FA', '#FFFFFF'] * 2],
                                                                    align='center',
                                                                    font=dict(color='black', size=14, family='Arial'),
                                                                    height=40
                                                                )
                                                            )])
                                                            
                                                            fig_timer_port.update_layout(
                                                                title=dict(
                                                                    text=f"⏰ Next Rebalance Timer - {portfolio_name}",
                                                                    x=0.5,
                                                                    font=dict(size=18, color='#2E86AB', family='Arial Black')
                                                                ),
                                                                width=700,
                                                                height=250,
                                                                margin=dict(l=20, r=20, t=60, b=20)
                                                            )
                                                            
                                                            # Store in session state for PDF export
                                                            st.session_state[f'timer_table_{portfolio_name}'] = fig_timer_port
                                            except Exception as e:
                                                pass  # Silently ignore timer table creation errors
                                except Exception as e:
                                    pass  # Silently ignore timer calculation errors

                                # Main "Rebalance as of today" plot and table - this should be the main rebalancing representation
                                st.markdown("---")
                                st.markdown(f"**🔄 Rebalance as of Today ({pd.Timestamp.now().strftime('%Y-%m-%d')})**")
                                
                                # Get momentum-based calculated weights for today's rebalancing from stored snapshot
                                today_weights = {}
                                
                                # Get the stored today_weights_map from snapshot data
                                snapshot = st.session_state.get('multi_backtest_snapshot_data', {})
                                today_weights_map = snapshot.get('today_weights_map', {}) if snapshot else {}
                                
                                if selected_portfolio_detail in today_weights_map:
                                    today_weights = today_weights_map.get(selected_portfolio_detail, {})
                                else:
                                    # Fallback to current allocation if no stored weights found
                                    today_weights = final_alloc
                                
                                # Create labels and values for the plot
                                labels_today = [k for k, v in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
                                vals_today = [float(today_weights[k]) * 100 for k in labels_today]
                                
                                # Create a larger plot for the main rebalancing representation
                                st.markdown(f"**Target Allocation if Rebalanced Today**")
                                fig_today = go.Figure()
                                fig_today.add_trace(go.Pie(labels=labels_today, values=vals_today, hole=0.3))
                                fig_today.update_traces(textinfo='percent+label')
                                fig_today.update_layout(
                                    template='plotly_dark', 
                                    margin=dict(t=30),
                                    height=600,  # Make it even bigger
                                    showlegend=True
                                )
                                st.plotly_chart(fig_today, use_container_width=True, key=f"multi_today_{selected_portfolio_detail}")
                                # Store in session state for PDF export
                                st.session_state[f'pie_chart_{selected_portfolio_detail}'] = fig_today
                                
                                # Table moved under the plot
                                # Add the "Rebalance as of today" table
                                try:
                                        # Get portfolio configuration for calculations
                                        portfolio_configs = st.session_state.get('multi_backtest_portfolio_configs', [])
                                        portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                        
                                        if portfolio_cfg:
                                            # Use current portfolio value from backtest results instead of initial value
                                            portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)  # fallback to initial value
                                            
                                            # Get current portfolio value from backtest results
                                            if 'multi_all_results' in st.session_state and st.session_state.multi_all_results:
                                                portfolio_results = st.session_state.multi_all_results.get(selected_portfolio_detail)
                                                if portfolio_results:
                                                    # Use the Final Value (with additions) for Multi Backtest - total portfolio value including all cash additions and compounding
                                                    if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                                        # Get the final value from the with_additions series (includes all cash additions and compounding)
                                                        final_value = portfolio_results['with_additions'].iloc[-1]
                                                        if not pd.isna(final_value) and final_value > 0:
                                                            portfolio_value = float(final_value)
                                                    elif isinstance(portfolio_results, dict) and 'no_additions' in portfolio_results:
                                                        # Fallback to no_additions if with_additions not available
                                                        final_value = portfolio_results['no_additions'].iloc[-1]
                                                        if not pd.isna(final_value) and final_value > 0:
                                                            portfolio_value = float(final_value)
                                                    elif isinstance(portfolio_results, pd.Series):
                                                        # Get the latest value from the series
                                                        latest_value = portfolio_results.iloc[-1]
                                                        if not pd.isna(latest_value) and latest_value > 0:
                                                            portfolio_value = float(latest_value)
                                            
                                            # Get raw data for price calculations
                                            raw_data = st.session_state.get('multi_backtest_raw_data', {})
                                            
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
                                                    
                                                    st.dataframe(sty, use_container_width=True)
                                                except Exception:
                                                    st.dataframe(df_display, use_container_width=True)
                                            
                                            # "Rebalance as of today" table (use momentum-based calculated weights)
                                            build_table_from_alloc(today_weights, None, f"Target Allocation if Rebalanced Today")
                                            
                                            # Store the table for PDF export AFTER the function call
                                            # Create a Plotly table figure for PDF export (EXACT same approach as fig_stats)
                                            try:
                                                # Get the DataFrame that was just created by build_table_from_alloc
                                                # We need to recreate it here since it's not returned by the function
                                                rows = []
                                                for tk in sorted(today_weights.keys()):
                                                    alloc_pct = float(today_weights.get(tk, 0))
                                                    if tk == 'CASH':
                                                        price = None
                                                        shares = 0
                                                        total_val = portfolio_value * alloc_pct
                                                    else:
                                                        df = raw_data.get(tk)
                                                        price = None
                                                        if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                            try:
                                                                price = float(df['Close'].iloc[-1])
                                                            except Exception:
                                                                price = None
                                                        try:
                                                            if price and price > 0:
                                                                allocation_value = portfolio_value * alloc_pct
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
                                                df_display = df_table.copy()
                                                
                                                # Remove CASH if it has zero value
                                                if 'CASH' in df_display.index:
                                                    cash_val = df_display.at['CASH', 'Total Value ($)']
                                                    if not (cash_val and not pd.isna(cash_val) and cash_val != 0):
                                                        df_display = df_display.drop('CASH')
                                                
                                                # Create Plotly table figure with ticker column included
                                                # Reset index to include ticker as a column
                                                df_display_with_ticker = df_display.reset_index()
                                                
                                                # Format the data to ensure 2 decimal places for display (same as PDF tables)
                                                formatted_values = []
                                                for col in df_display_with_ticker.columns:
                                                    if col in ['Price ($)', 'Total Value ($)', '% of Portfolio']:
                                                        # Format monetary and percentage values to 2 decimal places
                                                        formatted_values.append([f"{df_display_with_ticker[col][i]:.2f}" if pd.notna(df_display_with_ticker[col][i]) else "" for i in range(len(df_display_with_ticker))])
                                                    elif col == 'Shares':
                                                        # Format shares to 1 decimal place
                                                        formatted_values.append([f"{df_display_with_ticker[col][i]:.1f}" if pd.notna(df_display_with_ticker[col][i]) else "" for i in range(len(df_display_with_ticker))])
                                                    elif col == 'Allocation %':
                                                        # Format allocation to 2 decimal places
                                                        formatted_values.append([f"{df_display_with_ticker[col][i]:.2f}" if pd.notna(df_display_with_ticker[col][i]) else "" for i in range(len(df_display_with_ticker))])
                                                    else:
                                                        # Keep other columns as is
                                                        formatted_values.append([str(df_display_with_ticker[col][i]) if pd.notna(df_display_with_ticker[col][i]) else "" for i in range(len(df_display_with_ticker))])
                                                
                                                fig_alloc_table = go.Figure(data=[go.Table(
                                                    header=dict(values=list(df_display_with_ticker.columns),
                                                               fill_color='paleturquoise',
                                                               align='left',
                                                               font=dict(size=12)),
                                                    cells=dict(values=formatted_values,
                                                              fill_color='lavender',
                                                              align='left',
                                                              font=dict(size=11))
                                                )])
                                                fig_alloc_table.update_layout(
                                                    title=f"Target Allocation if Rebalanced Today - {selected_portfolio_detail}",
                                                    margin=dict(t=30, b=10, l=10, r=10),
                                                    height=400
                                                )
                                                table_key = f"alloc_table_{selected_portfolio_detail}"
                                                st.session_state[table_key] = fig_alloc_table
                                            except Exception as e:
                                                pass

                                except Exception as e:
                                    pass
                                
                                # Other rebalancing plots (smaller, placed after the main one)
                                st.markdown("---")
                                st.markdown("**📊 Historical Rebalancing Comparison**")
                                
                                col_plot1, col_plot2 = st.columns(2)
                                with col_plot1:
                                    st.markdown(f"**Last Rebalance Allocation (as of {last_rebal_date.date()})**")
                                    fig_rebal = go.Figure()
                                    fig_rebal.add_trace(go.Pie(labels=labels_rebal, values=vals_rebal, hole=0.3))
                                    fig_rebal.update_traces(textinfo='percent+label')
                                    fig_rebal.update_layout(template='plotly_dark', margin=dict(t=30), height=400)
                                    st.plotly_chart(fig_rebal, use_container_width=True, key=f"multi_rebal_{selected_portfolio_detail}")
                                with col_plot2:
                                    st.markdown(f"**Current Allocation (as of {final_date.date()})**")
                                    fig_final = go.Figure()
                                    fig_final.add_trace(go.Pie(labels=labels_final, values=vals_final, hole=0.3))
                                    fig_final.update_traces(textinfo='percent+label')
                                    fig_final.update_layout(template='plotly_dark', margin=dict(t=30), height=400)
                                    st.plotly_chart(fig_final, use_container_width=True, key=f"multi_final_{selected_portfolio_detail}")
                                
                                # Add the three allocation tables from Allocations page
                                try:
                                    # Get portfolio configuration for calculations
                                    portfolio_configs = st.session_state.get('multi_backtest_portfolio_configs', [])
                                    portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                    
                                    if portfolio_cfg:
                                        # Use current portfolio value from backtest results instead of initial value
                                        portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)  # fallback to initial value
                                        
                                        # Get current portfolio value from backtest results
                                        if 'multi_all_results' in st.session_state and st.session_state.multi_all_results:
                                            portfolio_results = st.session_state.multi_all_results.get(selected_portfolio_detail)
                                            if portfolio_results:
                                                # Use the Final Value (with additions) for Multi Backtest - total portfolio value including all cash additions and compounding
                                                if isinstance(portfolio_results, dict) and 'with_additions' in portfolio_results:
                                                    # Get the final value from the with_additions series (includes all cash additions and compounding)
                                                    final_value = portfolio_results['with_additions'].iloc[-1]
                                                    if not pd.isna(final_value) and final_value > 0:
                                                        portfolio_value = float(final_value)
                                                elif isinstance(portfolio_results, dict) and 'no_additions' in portfolio_results:
                                                    # Fallback to no_additions if with_additions not available
                                                    final_value = portfolio_results['no_additions'].iloc[-1]
                                                    if not pd.isna(final_value) and final_value > 0:
                                                        portfolio_value = float(final_value)
                                                elif isinstance(portfolio_results, pd.Series):
                                                    # Get the latest value from the series
                                                    latest_value = portfolio_results.iloc[-1]
                                                    if not pd.isna(latest_value) and latest_value > 0:
                                                        portfolio_value = float(latest_value)
                                        
                                        # Get raw data for price calculations
                                        raw_data = st.session_state.get('multi_backtest_raw_data', {})
                                        
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
                                                
                                                st.dataframe(sty, use_container_width=True)
                                                
                                                # Table storage is now handled outside this function
                                                pass
                                                    
                                            except Exception:
                                                st.dataframe(df_display, use_container_width=True)
                                                
                                                # Table storage is now handled outside this function (fallback case)
                                                pass
                                        
                                        # Last rebalance table (use last_rebal_date)
                                        build_table_from_alloc(rebal_alloc, last_rebal_date, f"Target Allocation at Last Rebalance ({last_rebal_date.date()})")
                                        # Current / Today table (use final_date's latest available prices as of now)
                                        build_table_from_alloc(final_alloc, None, f"Portfolio Evolution (Current Allocation)")
                                        
                                except Exception as e:
                                    pass
                                    
                        except Exception as e:
                            pass
                    else:
                        st.info("No allocation history available for this portfolio to show allocation plots.")
                else:
                    # Fallback: show table and plots based on last known allocations so UI stays visible
                    allocs_for_portfolio = st.session_state.multi_all_allocations.get(selected_portfolio_detail) if 'multi_all_allocations' in st.session_state else None
                    if not allocs_for_portfolio:
                        st.info("No allocation or momentum metrics available for this portfolio.")
                    else:
                        alloc_dates = sorted(list(allocs_for_portfolio.keys()))
                        last_date = alloc_dates[-1]
                        last_alloc = allocs_for_portfolio.get(last_date, {})
                        metrics_records_fb = []
                        for ticker, alloc in last_alloc.items():
                            record = {'Date': last_date, 'Ticker': ticker, 'Momentum': np.nan, 'Beta': np.nan, 'Volatility': np.nan, 'Calculated_Weight': alloc}
                            metrics_records_fb.append(record)

                        metrics_df = pd.DataFrame(metrics_records_fb)
                        metrics_df.set_index(['Date', 'Ticker'], inplace=True)
                        metrics_df_display = metrics_df.copy()
                        if 'Momentum' in metrics_df_display.columns:
                            metrics_df_display['Momentum'] = metrics_df_display['Momentum'].fillna(0) * 100
                        if 'Calculated_Weight' in metrics_df_display.columns:
                            metrics_df_display['Calculated_Weight'] = metrics_df_display['Calculated_Weight'].fillna(0) * 100
                            metrics_df_display['Volatility'] = metrics_df_display['Volatility'].fillna(np.nan) * 100

                        def color_momentum(val):
                            if isinstance(val, (int, float)):
                                color = 'green' if val > 0 else 'red'
                                return f'color: {color}'
                            return ''

                        def highlight_metrics_rows(s):
                            date_str = s.name[0]
                            if s.name[1] == 'CASH':
                                return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                            unique_dates = list(metrics_df_display.index.get_level_values('Date').unique())
                            is_even = unique_dates.index(date_str) % 2 == 0
                            bg_color = 'background-color: #0e1117' if is_even else 'background-color: #262626'
                            return [bg_color] * len(s)

                        fmt_map = {}
                        if 'Momentum' in metrics_df_display.columns:
                            fmt_map['Momentum'] = '{:,.0f}%'
                        if 'Beta' in metrics_df_display.columns:
                            fmt_map['Beta'] = '{:,.2f}'
                        if 'Volatility' in metrics_df_display.columns:
                            fmt_map['Volatility'] = '{:,.2f}%'
                        if 'Calculated_Weight' in metrics_df_display.columns:
                            fmt_map['Calculated_Weight'] = '{:,.0f}%'

                        styler_metrics = metrics_df_display.style.apply(highlight_metrics_rows, axis=1)
                        if 'Momentum' in metrics_df_display.columns:
                            styler_metrics = styler_metrics.map(color_momentum, subset=['Momentum'])
                        if fmt_map:
                            styler_metrics = styler_metrics.format(fmt_map)
                        st.dataframe(styler_metrics, use_container_width=True)

                        # Also show allocation plots for the last allocation snapshot
                        try:
                            final_date = last_date
                            final_alloc = last_alloc
                            labels_final = list(final_alloc.keys())
                            vals_final = [float(final_alloc[k]) * 100 for k in labels_final]
                            col_plot1, col_plot2 = st.columns(2)
                            with col_plot1:
                                st.markdown(f"**Last Rebalance Allocation (as of {final_date.date()})**")
                                fig_rebal = go.Figure()
                                fig_rebal.add_trace(go.Pie(labels=labels_final, values=vals_final, hole=0.3))
                                fig_rebal.update_traces(textinfo='percent+label')
                                fig_rebal.update_layout(template='plotly_dark', margin=dict(t=30))
                                st.plotly_chart(fig_rebal, use_container_width=True, key=f"multi_rebal_fallback_{selected_portfolio_detail}")
                            with col_plot2:
                                st.markdown(f"**Current Allocation (as of {final_date.date()})**")
                                fig_final = go.Figure()
                                fig_final.add_trace(go.Pie(labels=labels_final, values=vals_final, hole=0.3))
                                fig_final.update_traces(textinfo='percent+label')
                                fig_final.update_layout(template='plotly_dark', margin=dict(t=30))
                                st.plotly_chart(fig_final, use_container_width=True, key=f"multi_final_fallback_{selected_portfolio_detail}")
                        except Exception as e:
                            pass

        else:
            st.info("Configuration is ready. Press 'Run Backtest' to see results.")
    
    # Console log UI removed
    
    # PDF Export Section
    st.markdown("---")
    st.subheader("📄 PDF Export")
    
    # Optional custom PDF report name
    custom_report_name = st.text_input(
        "📝 Custom Report Name (optional):", 
        value="",
        placeholder="e.g., Tech Stocks Q4 Analysis, Conservative vs Aggressive, Monthly Performance Review",
        help="Leave empty to use automatic naming: 'Multi_Backtest_Report_[timestamp].pdf'",
        key="multi_backtest_custom_report_name"
    )
    
    if st.button("Generate PDF Report", type="primary", use_container_width=True):
        try:
            pdf_buffer = generate_simple_pdf_report(custom_report_name)
            if pdf_buffer:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Generate filename based on custom name or default
                if custom_report_name.strip():
                    clean_name = custom_report_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
                    filename = f"{clean_name}_{timestamp}.pdf"
                else:
                    filename = f"Multi_Backtest_Report_{timestamp}.pdf"
                
                st.success("✅ PDF Report Generated Successfully!")
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("❌ Failed to generate PDF report")
        except Exception as e:
            st.error(f"❌ Error generating PDF: {str(e)}")
            st.exception(e)
