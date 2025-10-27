import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_vix_data():
    """Download VIX data with caching"""
    try:
        vix = yf.Ticker("^VIX")
        data = vix.history(period="max")
        return data
    except Exception as e:
        st.error(f"Error downloading VIX data: {e}")
        return None

def calculate_vix_statistics(data, threshold):
    """Calculate statistics for periods when VIX was below threshold"""
    if data is None or data.empty:
        return {}
    
    # Create boolean mask for VIX below threshold
    below_threshold = data['Close'] < threshold
    
    # Calculate basic stats
    total_days = len(data)
    below_threshold_days = below_threshold.sum()
    above_threshold_days = total_days - below_threshold_days
    
    # Calculate percentage
    below_percentage = (below_threshold_days / total_days) * 100
    
    # Find consecutive periods below threshold
    # Create a DataFrame to work with groups
    below_threshold_df = pd.DataFrame({'below_threshold': below_threshold})
    below_threshold_df['group'] = (below_threshold_df['below_threshold'] != below_threshold_df['below_threshold'].shift()).cumsum()
    consecutive_periods = below_threshold_df[below_threshold_df['below_threshold']].groupby('group').size()
    
    # VIX statistics
    vix_stats = {
        'total_days': total_days,
        'below_threshold_days': below_threshold_days,
        'above_threshold_days': above_threshold_days,
        'below_percentage': below_percentage,
        'consecutive_periods_count': len(consecutive_periods),
        'max_consecutive_days': consecutive_periods.max() if len(consecutive_periods) > 0 else 0,
        'avg_consecutive_days': consecutive_periods.mean() if len(consecutive_periods) > 0 else 0,
        'current_vix': data['Close'].iloc[-1],
        'min_vix': data['Close'].min(),
        'max_vix': data['Close'].max(),
        'avg_vix': data['Close'].mean(),
        'median_vix': data['Close'].median()
    }
    
    return vix_stats

def create_vix_chart(data, threshold):
    """Create interactive VIX chart with threshold highlighting"""
    if data is None or data.empty:
        return None
    
    fig = go.Figure()
    
    # Create boolean mask for VIX below threshold
    below_threshold = data['Close'] < threshold
    
    # Add VIX line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='VIX',
        line=dict(color='blue', width=2),
        hovertemplate='<b>%{x}</b><br>VIX: %{y:.2f}<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_trace(go.Scatter(
        x=[data.index[0], data.index[-1]],
        y=[threshold, threshold],
        mode='lines',
        name=f'Threshold ({threshold})',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate=f'Threshold: {threshold}<extra></extra>'
    ))
    
    # Highlight periods below threshold
    below_data = data[below_threshold]
    if not below_data.empty:
        fig.add_trace(go.Scatter(
            x=below_data.index,
            y=below_data['Close'],
            mode='markers',
            name=f'VIX < {threshold}',
            marker=dict(color='green', size=6, opacity=0.7),
            hovertemplate='<b>%{x}</b><br>VIX: %{y:.2f} (Below threshold)<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"VIX Analysis - Threshold: {threshold}",
        xaxis_title="Date",
        yaxis_title="VIX Level",
        hovermode='x',
        height=600,
        showlegend=True,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showspikes=True,
            spikecolor="orange",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1
        ),
        yaxis=dict(
            title="VIX Level",
            tickformat=".1f",
            showspikes=True,
            spikecolor="orange",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1
        )
    )
    
    return fig

def create_consecutive_periods_chart(data, threshold):
    """Create chart showing consecutive periods below threshold"""
    if data is None or data.empty:
        return None
    
    below_threshold = data['Close'] < threshold
    below_threshold_df = pd.DataFrame({'below_threshold': below_threshold})
    below_threshold_df['group'] = (below_threshold_df['below_threshold'] != below_threshold_df['below_threshold'].shift()).cumsum()
    
    # Get consecutive periods
    consecutive_periods = []
    for group_id, group_data in below_threshold_df[below_threshold_df['below_threshold']].groupby('group'):
        if len(group_data) > 0:
            start_date = group_data.index[0]
            end_date = group_data.index[-1]
            consecutive_periods.append({
                'start': start_date,
                'end': end_date,
                'duration': len(group_data),
                'min_vix': data.loc[group_data.index, 'Close'].min(),
                'max_vix': data.loc[group_data.index, 'Close'].max(),
                'avg_vix': data.loc[group_data.index, 'Close'].mean()
            })
    
    if not consecutive_periods:
        return None, []
    
    # Create chart
    fig = go.Figure()
    
    for i, period in enumerate(consecutive_periods):
        fig.add_trace(go.Scatter(
            x=[period['start'], period['end']],
            y=[i, i],
            mode='lines+markers',
            name=f"Period {i+1}",
            line=dict(width=8),
            marker=dict(size=10),
            hovertemplate=f'<b>Period {i+1}</b><br>' +
                         f'Start: {period["start"].strftime("%Y-%m-%d")}<br>' +
                         f'End: {period["end"].strftime("%Y-%m-%d")}<br>' +
                         f'Duration: {period["duration"]} days<br>' +
                         f'Min VIX: {period["min_vix"]:.2f}<br>' +
                         f'Max VIX: {period["max_vix"]:.2f}<br>' +
                         f'Avg VIX: {period["avg_vix"]:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Consecutive Periods with VIX < {threshold}",
        xaxis_title="Date",
        yaxis_title="Period Number",
        height=400,
        showlegend=False,
        template="plotly_white",
        hovermode='x',
        xaxis=dict(
            showspikes=True,
            spikecolor="orange",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1
        ),
        yaxis=dict(
            showspikes=True,
            spikecolor="orange",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1
        )
    )
    
    return fig, consecutive_periods

def calculate_rolling_minimum(data, days):
    """Calculate rolling minimum VIX over specified number of days"""
    if data is None or data.empty:
        return None
    
    # Calculate rolling minimum
    rolling_min = data['Close'].rolling(window=days, min_periods=1).min()
    
    # Get current rolling minimum
    current_rolling_min = rolling_min.iloc[-1]
    
    # Find the lowest rolling minimum in history
    lowest_rolling_min = rolling_min.min()
    
    # Find when this lowest occurred
    lowest_date = rolling_min.idxmin()
    
    return {
        'current_rolling_min': current_rolling_min,
        'lowest_rolling_min': lowest_rolling_min,
        'lowest_date': lowest_date,
        'days': days
    }

def calculate_yearly_vix_stats(data, threshold):
    """Calculate yearly statistics for VIX below threshold"""
    if data is None or data.empty:
        return pd.DataFrame()
    
    # Add year column
    data_with_year = data.copy()
    data_with_year['Year'] = data_with_year.index.year
    
    # Group by year and calculate statistics
    yearly_stats = []
    
    for year in sorted(data_with_year['Year'].unique()):
        year_data = data_with_year[data_with_year['Year'] == year]
        
        # Calculate stats for this year
        total_days = len(year_data)
        below_threshold = year_data['Close'] < threshold
        below_days = below_threshold.sum()
        below_percentage = (below_days / total_days) * 100 if total_days > 0 else 0
        
        # Find consecutive periods for this year
        below_threshold_df = pd.DataFrame({'below_threshold': below_threshold})
        below_threshold_df['group'] = (below_threshold_df['below_threshold'] != below_threshold_df['below_threshold'].shift()).cumsum()
        consecutive_periods = below_threshold_df[below_threshold_df['below_threshold']].groupby('group').size()
        
        max_consecutive = consecutive_periods.max() if len(consecutive_periods) > 0 else 0
        num_periods = len(consecutive_periods)
        
        # VIX stats for this year
        min_vix = year_data['Close'].min()
        max_vix = year_data['Close'].max()
        avg_vix = year_data['Close'].mean()
        
        yearly_stats.append({
            'Year': year,
            'Total Days': total_days,
            'Days Below Threshold': below_days,
            'Below %': f"{below_percentage:.1f}%",
            'Max Consecutive Days': max_consecutive,
            'Number of Periods': num_periods,
            'Min VIX': f"{min_vix:.2f}",
            'Max VIX': f"{max_vix:.2f}",
            'Avg VIX': f"{avg_vix:.2f}",
            'Below Threshold': 'Yes' if below_days > 0 else 'No'
        })
    
    return pd.DataFrame(yearly_stats)

st.set_page_config(page_title="VIX Analysis", page_icon="📈", layout="wide")

st.title("📈 VIX Analysis - Threshold Analysis")

# Sidebar for controls
st.sidebar.header("Controls")

# VIX threshold selection
threshold = st.sidebar.slider(
    "VIX Threshold",
    min_value=5.0,
    max_value=50.0,
    value=15.0,
    step=0.5,
    help="Select the VIX threshold to analyze"
)

# Additional options
show_consecutive_periods = st.sidebar.checkbox("Show Consecutive Periods", value=True)
show_statistics = st.sidebar.checkbox("Show Detailed Statistics", value=True)


# Get VIX data
with st.spinner("Loading VIX data..."):
    vix_data = get_vix_data()

if vix_data is not None and not vix_data.empty:
    # Calculate statistics
    stats = calculate_vix_statistics(vix_data, threshold)
    
    # Display key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Current VIX",
            f"{stats['current_vix']:.2f}",
            delta=f"{stats['current_vix'] - threshold:.2f} vs threshold"
        )
    
    with col2:
        st.metric(
            "Days Below Threshold",
            f"{stats['below_threshold_days']:,}",
            delta=f"{stats['below_percentage']:.1f}% of total"
        )
    
    with col3:
        st.metric(
            "Max Consecutive Days",
            f"{stats['max_consecutive_days']:,}",
            delta=f"{stats['consecutive_periods_count']} periods"
        )
    
    with col4:
        st.metric(
            "Average VIX",
            f"{stats['avg_vix']:.2f}",
            delta=f"{stats['current_vix'] - stats['avg_vix']:.2f} vs current"
        )
    
    with col5:
        st.metric(
            "Median VIX",
            f"{stats['median_vix']:.2f}",
            delta=f"{stats['current_vix'] - stats['median_vix']:.2f} vs current"
        )
    
    # Add explanation dropdown
    with st.expander("Explication des métriques"):
        st.markdown("""
        **📊 Current VIX (VIX Actuel)**
        - Valeur actuelle du VIX (Volatility Index)
        - Le delta montre la différence avec ton seuil
        - VIX élevé = volatilité élevée, VIX bas = marché calme
        
        **📅 Days Below Threshold (Jours Sous le Seuil)**
        - Nombre total de jours où le VIX était sous ton seuil
        - Le pourcentage indique la proportion du temps historique
        - Plus le % est élevé, plus le marché a été calme historiquement
        
        **🔗 Max Consecutive Days (Jours Consécutifs Max)**
        - La plus longue période ininterrompue sous le seuil
        - Le delta montre le nombre total de périodes distinctes
        - Indique la durée maximale des "périodes de calme"
        
        **📈 Average VIX (VIX Moyen)**
        - VIX moyen sur toute la période historique
        - Le delta montre la différence avec le VIX actuel
        - Positif = VIX actuel au-dessus de la moyenne historique
        
        **📊 Median VIX (VIX Médiane)**
        - VIX médiane (plus robuste aux valeurs extrêmes)
        - Le delta montre la différence avec le VIX actuel
        - Positif = VIX actuel au-dessus de la médiane historique
        
        **💡 Interprétation:**
        - VIX < 15 = Marché très calme (rare)
        - VIX 15-25 = Volatilité normale
        - VIX > 25 = Marché stressé/volatil
        - VIX > 40 = Crise/panique
        """)
    
    # Create and display main chart
    fig = create_vix_chart(vix_data, threshold)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Show consecutive periods if requested
    if show_consecutive_periods:
        st.subheader("📅 Consecutive Periods Below Threshold")
        consecutive_fig, consecutive_periods = create_consecutive_periods_chart(vix_data, threshold)
        
        if consecutive_fig:
            st.plotly_chart(consecutive_fig, use_container_width=True)
            
            # Display periods table in dropdown
            if consecutive_periods:
                with st.expander("📋 Period Details"):
                    periods_df = pd.DataFrame(consecutive_periods)
                    periods_df['start'] = periods_df['start'].dt.strftime('%Y-%m-%d')
                    periods_df['end'] = periods_df['end'].dt.strftime('%Y-%m-%d')
                    periods_df = periods_df.rename(columns={
                        'start': 'Start Date',
                        'end': 'End Date',
                        'duration': 'Duration (Days)',
                        'min_vix': 'Min VIX',
                        'max_vix': 'Max VIX',
                        'avg_vix': 'Avg VIX'
                    })
                    st.dataframe(periods_df, use_container_width=True)
    
    # Show detailed statistics if requested
    if show_statistics:
        with st.expander("📈 Detailed Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Period Analysis:**")
                st.write(f"• Total trading days: {stats['total_days']:,}")
                st.write(f"• Days below threshold: {stats['below_threshold_days']:,} ({stats['below_percentage']:.1f}%)")
                st.write(f"• Days above threshold: {stats['above_threshold_days']:,} ({100-stats['below_percentage']:.1f}%)")
                st.write(f"• Number of consecutive periods: {stats['consecutive_periods_count']}")
                st.write(f"• Maximum consecutive days: {stats['max_consecutive_days']:,}")
                st.write(f"• Average consecutive days: {stats['avg_consecutive_days']:.1f}")
            
            with col2:
                st.markdown("**VIX Statistics:**")
                st.write(f"• Current VIX: {stats['current_vix']:.2f}")
                st.write(f"• Minimum VIX: {stats['min_vix']:.2f}")
                st.write(f"• Maximum VIX: {stats['max_vix']:.2f}")
                st.write(f"• Average VIX: {stats['avg_vix']:.2f}")
                st.write(f"• Median VIX: {stats['median_vix']:.2f}")
                st.write(f"• VIX volatility: {vix_data['Close'].std():.2f}")
    
    # Show yearly analysis
    st.subheader("📅 Yearly VIX Analysis")
    yearly_stats_df = calculate_yearly_vix_stats(vix_data, threshold)
    
    if not yearly_stats_df.empty:
        # Add some styling to highlight years with VIX below threshold
        def highlight_below_threshold(row):
            if row['Below Threshold'] == 'Yes':
                return ['background-color: #228B22'] * len(row)  # Forest Green
            else:
                return [''] * len(row)
        
        # Display the table with styling
        st.dataframe(
            yearly_stats_df.style.apply(highlight_below_threshold, axis=1),
            use_container_width=True,
            height=600
        )
        
        # Show summary
        years_below = len(yearly_stats_df[yearly_stats_df['Below Threshold'] == 'Yes'])
        total_years = len(yearly_stats_df)
        st.write(f"**Summary:** {years_below} out of {total_years} years had VIX below threshold at least once")
    
    # VIX Minimum by Year Coverage Analysis
    st.subheader("📊 VIX Minimum Coverage by Years")
    
    # Calculate VIX minimum for each year
    vix_data['Year'] = vix_data.index.year
    yearly_min_vix = vix_data.groupby('Year')['Close'].min().reset_index()
    yearly_min_vix.columns = ['Year', 'Min_VIX']
    
    # Calculate coverage for different VIX levels
    vix_levels = []
    for vix_level in range(5, 51, 1):  # From 5 to 50 VIX
        years_covered = len(yearly_min_vix[yearly_min_vix['Min_VIX'] <= vix_level])
        total_years = len(yearly_min_vix)
        coverage_percentage = (years_covered / total_years) * 100
        
        vix_levels.append({
            'VIX_Level': vix_level,
            'Years_Covered': years_covered,
            'Total_Years': total_years,
            'Coverage_Percentage': coverage_percentage
        })
    
    coverage_df = pd.DataFrame(vix_levels)
    
    # Find minimum VIX needed for specific coverage percentages
    coverage_analysis = []
    target_coverages = [100, 95, 90, 85, 80, 75, 70, 50]
    
    for target in target_coverages:
        # Find the minimum VIX level that covers at least target% of years
        covered_data = coverage_df[coverage_df['Coverage_Percentage'] >= target]
        if not covered_data.empty:
            min_vix_needed = covered_data['VIX_Level'].min()
            actual_coverage = covered_data[covered_data['VIX_Level'] == min_vix_needed]['Coverage_Percentage'].iloc[0]
            years_covered = covered_data[covered_data['VIX_Level'] == min_vix_needed]['Years_Covered'].iloc[0]
            total_years = covered_data[covered_data['VIX_Level'] == min_vix_needed]['Total_Years'].iloc[0]
            
            # Calculate number of periods for this VIX level
            below_threshold = vix_data['Close'] <= min_vix_needed
            below_threshold_df = pd.DataFrame({'below_threshold': below_threshold})
            below_threshold_df['group'] = (below_threshold_df['below_threshold'] != below_threshold_df['below_threshold'].shift()).cumsum()
            consecutive_periods = below_threshold_df[below_threshold_df['below_threshold']].groupby('group').size()
            num_periods = len(consecutive_periods)
            
            coverage_analysis.append({
                'Coverage_Needed': f'{target}%',
                'Min_VIX_Required': min_vix_needed,
                'Actual_Coverage': f'{actual_coverage:.1f}%',
                'Years_Covered': years_covered,
                'Total_Years': total_years,
                'Number_of_Periods': num_periods
            })
    
    coverage_analysis_df = pd.DataFrame(coverage_analysis)
    
    # Display results
    st.dataframe(coverage_analysis_df, use_container_width=True)
    
    # Create visualization
    fig_coverage = go.Figure()
    
    fig_coverage.add_trace(go.Scatter(
        x=coverage_df['VIX_Level'],
        y=coverage_df['Coverage_Percentage'],
        mode='lines',
        name='Year Coverage',
        line=dict(color='blue', width=2),
        hovertemplate='VIX Level: %{x}<br>Coverage: %{y:.1f}%<br>Years Covered: %{customdata}<extra></extra>',
        customdata=coverage_df['Years_Covered']
    ))
    
    # Add threshold line
    fig_coverage.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Threshold: {threshold}",
        annotation_position="top"
    )
    
    # Add horizontal lines for key coverage levels
    for target in [100, 90, 80, 70, 50]:
        fig_coverage.add_hline(
            y=target,
            line_dash="dot",
            line_color="red",
            annotation_text=f"{target}%",
            annotation_position="right"
        )
    
    fig_coverage.update_layout(
        title="VIX Level Coverage Across All Years",
        xaxis_title="VIX Level",
        yaxis_title="Percentage of Years Covered",
        height=500,
        template="plotly_white",
        hovermode='x',
        xaxis=dict(
            showspikes=True,
            spikecolor="orange",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1
        ),
        yaxis=dict(
            tickformat=".0f",
            ticksuffix="%",
            showspikes=True,
            spikecolor="orange",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1
        )
    )
    
    st.plotly_chart(fig_coverage, use_container_width=True)
    
    # Add explanation
    with st.expander("Explication de l'analyse de couverture"):
        st.markdown("""
        **📊 VIX Minimum Coverage by Years**
        
        Cette analyse répond à ta question : **"Quel VIX minimum utiliser pour couvrir X% des années ?"**
        
        **💡 Interprétation:**
        - **Coverage Needed** : Le pourcentage d'années que tu veux couvrir (100%, 90%, etc.)
        - **Min VIX Required** : Le VIX minimum nécessaire pour couvrir ce pourcentage
        - **Actual Coverage** : Le pourcentage réellement obtenu avec ce VIX
        - **Years Covered** : Le nombre d'années couvertes
        - **Number of Periods** : Le nombre de périodes distinctes où le VIX était sous ce niveau
        
        **🎯 Utilisation pratique:**
        - **Pour 100% des années** : Utilise le VIX minimum requis (ex: VIX 20)
        - **Pour 90% des années** : Utilise le VIX minimum requis (ex: VIX 18)
        - **Pour tes options** : Tu sais exactement quel VIX utiliser selon ton niveau de confiance
        
        **📈 Exemple:** 
        - Pour couvrir 90% des années, tu dois utiliser VIX 18
        - Avec VIX 18, tu obtiens 94.4% de couverture (même plus que tes 90% !)
        
        **📊 Graphique:** Montre la courbe - plus tu veux couvrir d'années, plus le VIX minimum requis est élevé.
        """)
    
    # Data summary
    with st.expander("📊 Data Summary"):
        st.write(f"**Period:** {vix_data.index[0].strftime('%Y-%m-%d')} to {vix_data.index[-1].strftime('%Y-%m-%d')}")
        st.write(f"**Total Days:** {len(vix_data):,}")
        st.write(f"**Data Source:** Yahoo Finance (^VIX)")
        st.write(f"**Cache:** 2 hours TTL")
        
        # Show recent data
        st.write("**Last 10 days:**")
        recent_data = vix_data[['Close']].tail(10)
        recent_data['Date'] = recent_data.index.strftime('%Y-%m-%d')
        recent_data = recent_data[['Date', 'Close']].reset_index(drop=True)
        st.dataframe(recent_data, use_container_width=True)

else:
    st.error("❌ Could not load VIX data. Please check your internet connection and try again.")

# Footer
st.markdown("---")
st.markdown("💡 **Tips:**")
st.markdown("- Adjust the threshold slider to analyze different VIX levels")
st.markdown("- Green dots highlight periods when VIX was below your threshold")
st.markdown("- Use consecutive periods chart to identify low volatility regimes")
st.markdown("- Data is cached for 2 hours for faster analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style="
    text-align: center; 
    color: #666; 
    margin: 2rem 0; 
    padding: 1rem; 
    font-size: 0.9rem;
    font-weight: 500;
">
    Made by Nicolas Cool
</div>
""", unsafe_allow_html=True)
