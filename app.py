import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import os
from etl import NigeriaDataETL
import matplotlib.colors as mcolors

# Set page config
st.set_page_config(
    page_title="Snapstats",
    page_icon="🗺️",
    layout="wide"
)


def load_data():
    """Load and return the processed GeoJSON data"""
    data_dir = 'data'
    processed_dir = os.path.join(data_dir, 'processed')
    processed_file = os.path.join(processed_dir, 'processed_nigeria_states.geojson')
    
    try:
        # Run ETL if data doesn't exist
        if not os.path.exists(processed_file):
            with st.spinner('Fetching and processing data...'):
                etl = NigeriaDataETL()
                success = etl.run_pipeline()
                if not success:
                    st.error("Failed to fetch and process data")
                    return None
        # Load and return the data
        return gpd.read_file(processed_file)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def assign_zone_colors(gdf):
    """Assign colors to states based on their zones"""
    # Define colors for each zone
    zone_colors = {
        'North Central': '#2E8B57',  # Sea Green
        'North East': '#4169E1',     # Royal Blue
        'North West': '#DC143C',     # Crimson
        'South East': '#FF8C00',     # Dark Orange
        'South South': '#9932CC',    # Dark Orchid
        'South West': '#FFD700'      # Gold
    }
    
    # Assign colors to each state based on zone
    gdf['zone_color'] = gdf['Zone'].map(zone_colors)
    
    # Handle any missing zones with a default color
    gdf['zone_color'] = gdf['zone_color'].fillna('#808080')  # Gray for missing data
    
    return gdf


def create_static_map(gdf):
    """Create a static matplotlib map focused on Nigeria with zone-based coloring"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot states with zone-based colors
    gdf.plot(
        ax=ax,
        color=gdf['zone_color'],
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add country boundary outline
    country_boundary = gdf.union_all().boundary
    gpd.GeoSeries(country_boundary).plot(
        ax=ax,
        color='red',
        linewidth=2.5
    )
    
    # Add state labels
    for idx, row in gdf.iterrows():
        ax.text(
            row.geometry.centroid.x,
            row.geometry.centroid.y,
            row['shapeName'],
            fontsize=8,
            ha='center',
            weight='bold'
        )
    
    # Isolate to Nigeria bounds
    minx, miny, maxx, maxy = gdf.total_bounds
    ax.set_xlim(minx - 0.5, maxx + 0.5)
    ax.set_ylim(miny - 0.5, maxy + 0.5)
    
    # Add legend for zones
    zone_colors = {
        'North Central': '#2E8B57',
        'North East': '#4169E1',
        'North West': '#DC143C',
        'South East': '#FF8C00',
        'South South': '#9932CC',
        'South West': '#FFD700'
    }
    
    # Create legend
    legend_elements = []
    for zone, color in zone_colors.items():
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', label=zone))
    
    ax.legend(handles=legend_elements, loc='upper right', title='Geopolitical Zones')
    
    # Hide axis frame
    ax.set_title('Nigeria States Map - By Geopolitical Zones', fontsize=16, weight='bold')
    ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return fig





def create_interactive_map(gdf):
    """Create an interactive folium map focused on Nigeria with zone-based coloring"""
    # Calculate center point using bounds
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create the base map with transparent background
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles=None,
        control_scale=False,
        zoom_control=False,
        scrollWheelZoom=False,
        dragging=False,
        doubleClickZoom=False,
        touchZoom=False
    )
    
    # Add custom CSS to make background transparent and hide Leaflet elements
    custom_css = """
        <style>
        .folium-map {
            background-color: transparent !important;
        }
        .leaflet-control-attribution {
            display: none !important;
        }
        .leaflet-container {
            border: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        </style>
        """
    m.get_root().html.add_child(folium.Element(custom_css))
    
    # Convert GeoDataFrame to GeoJSON string
    geo_data = gdf.to_json()
    
    # Add GeoJSON layer with zone-based styling and enhanced tooltips
    folium.GeoJson(
        data=geo_data,
        name='Nigeria States',
        style_function=lambda feature: {
            'fillColor': feature['properties']['zone_color'],
            'color': '#000000',
            'weight': 1.5,
            'fillOpacity': 0.7
        },
        highlight_function=lambda x: {
            'weight': 2,
            'fillOpacity': 0.9
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['shapeName', 'Zone', 'Typical_Parties', 'Major_Tribes', 'area_km2'],
            aliases=['🏛️ State:', '🗺️ Zone:', '🏛️ Parties:', '👥 Major Tribes:', '📏 Area (km²):'],
            localize=True,
            sticky=True,
            style="""
                    background-color: #f8f9fa;
                    border: 2px solid #007bff;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size: 12px;
                    padding: 12px;
                    max-width: 220px;
                    word-wrap: break-word;
                    white-space: normal;
                    overflow-wrap: break-word;
                    color: #333;
                    line-height: 1.3;
                """
        ),
        popup=folium.GeoJsonPopup(
            fields=['shapeName', 'Zone', 'Typical_Parties', 'Major_Tribes', 'Minority_Tribes', 'area_km2'],
            aliases=['🏛️ State:', '🗺️ Zone:', '🏛️ Political Parties:', '👥 Major Tribes:', '👥 Minority Tribes:', '📏 Area (km²):'],
            localize=True,
            style="""
                    background-color: #f8f9fa;
                    border: 2px solid #007bff;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size: 13px;
                    padding: 15px;
                    max-width: 320px;
                    word-wrap: break-word;
                    white-space: normal;
                    overflow-wrap: break-word;
                    hyphens: auto;
                    color: #333;
                    line-height: 1.4;
                """
        )
    ).add_to(m)
    
    # Fit map to Nigeria bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m


def create_zone_analysis(gdf):
    """Create comprehensive zone-based analysis and statistics"""
    st.subheader("📊 Enhanced Zone Analysis")
    
    # Zone statistics
    zone_stats = gdf.groupby('Zone').agg({
        'shapeName': 'count',
        'area_km2': 'sum'
    }).rename(columns={'shapeName': 'State_Count', 'area_km2': 'Total_Area'})
    
    # Calculate average area per zone
    zone_stats['Avg_Area'] = zone_stats['Total_Area'] / zone_stats['State_Count']
    
    # Executive Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Zones", len(zone_stats))
    
    with col2:
        st.metric("Average States/Zone", f"{zone_stats['State_Count'].mean():.1f}")
    
    with col3:
        largest_zone = zone_stats['Total_Area'].idxmax()
        st.metric("Largest Zone", largest_zone)
    
    with col4:
        smallest_zone = zone_stats['Total_Area'].idxmin()
        st.metric("Smallest Zone", smallest_zone)
    
    # Visual Analytics Section
    st.subheader("📈 Zone Comparison Charts")
    
    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Bar chart for states per zone
        st.write("**States per Zone**")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        zones = zone_stats.index
        states_count = zone_stats['State_Count']
        colors = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC', '#FFD700']
        
        bars = ax1.bar(zones, states_count, color=colors[:len(zones)])
        ax1.set_title('Number of States per Zone', fontsize=14, weight='bold')
        ax1.set_ylabel('Number of States')
        ax1.set_xlabel('Geopolitical Zones')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig1)
    
    with chart_col2:
        # Pie chart for area distribution
        st.write("**Area Distribution by Zone**")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Create pie chart
        wedges, texts, autotexts = ax2.pie(
            zone_stats['Total_Area'], 
            labels=zone_stats.index,
            colors=colors[:len(zones)],
            autopct='%1.1f%%',
            startangle=90
        )
        
        # Beautify the pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        ax2.set_title('Area Distribution by Zone', fontsize=14, weight='bold')
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Zone Comparison Table
    st.subheader("📋 Detailed Zone Statistics")
    display_zone_stats = zone_stats.copy()
    display_zone_stats['Total_Area_Formatted'] = display_zone_stats['Total_Area'].apply(lambda x: f"{x:,.0f} km²")
    display_zone_stats['Avg_Area_Formatted'] = display_zone_stats['Avg_Area'].apply(lambda x: f"{x:,.0f} km²")
    display_zone_stats['Area_Percentage'] = (display_zone_stats['Total_Area'] / display_zone_stats['Total_Area'].sum() * 100).apply(lambda x: f"{x:.1f}%")
    
    final_display = display_zone_stats[['State_Count', 'Total_Area_Formatted', 'Avg_Area_Formatted', 'Area_Percentage']].rename(columns={
        'State_Count': 'States',
        'Total_Area_Formatted': 'Total Area',
        'Avg_Area_Formatted': 'Average Area',
        'Area_Percentage': 'Area %'
    })
    
    st.dataframe(final_display, use_container_width=True)
    
    # Zone Deep Dive Section
    st.subheader("🔍 Zone Deep Dive")
    
    # Zone selector
    selected_zone = st.selectbox("Select a zone for detailed analysis:", zone_stats.index)
    
    if selected_zone:
        zone_data = gdf[gdf['Zone'] == selected_zone]
        
        # Zone profile metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("States in Zone", len(zone_data))
            st.metric("Total Area", f"{zone_data['area_km2'].sum():,.0f} km²")
        
        with col2:
            # Most common party in zone
            party_counts = {}
            for parties in zone_data['Typical_Parties'].dropna():
                for party in parties.split(','):
                    party = party.strip()
                    party_counts[party] = party_counts.get(party, 0) + 1
            
            if party_counts:
                dominant_party = max(party_counts, key=party_counts.get)
                st.metric("Dominant Party", dominant_party)
        
        with col3:
            # Most common tribes in zone
            tribe_counts = {}
            for tribes in zone_data['Major_Tribes'].dropna():
                for tribe in tribes.split(','):
                    tribe = tribe.strip()
                    tribe_counts[tribe] = tribe_counts.get(tribe, 0) + 1
            
            if tribe_counts:
                dominant_tribe = max(tribe_counts, key=tribe_counts.get)
                st.metric("Most Common Tribe", dominant_tribe)
        
        # States in selected zone
        st.write(f"**States in {selected_zone}:**")
        zone_states = zone_data[['shapeName', 'area_km2', 'Typical_Parties', 'Major_Tribes']].copy()
        zone_states['area_km2'] = zone_states['area_km2'].apply(lambda x: f"{x:,.0f} km²")
        zone_states = zone_states.rename(columns={
            'shapeName': 'State',
            'area_km2': 'Area',
            'Typical_Parties': 'Parties',
            'Major_Tribes': 'Major Tribes'
        })
        
        st.dataframe(zone_states, use_container_width=True)
    
    # Executive Summary
    st.subheader("📄 Executive Summary")
    
    summary_text = f"""
    **Nigeria Geopolitical Zones Analysis:**
    
    • **Total Zones**: {len(zone_stats)} geopolitical zones covering all 37 states
    • **Largest Zone**: {largest_zone} with {zone_stats.loc[largest_zone, 'Total_Area']:,.0f} km²
    • **Smallest Zone**: {smallest_zone} with {zone_stats.loc[smallest_zone, 'Total_Area']:,.0f} km²
    • **Most States**: {zone_stats['State_Count'].idxmax()} has {zone_stats['State_Count'].max()} states
    • **Fewest States**: {zone_stats['State_Count'].idxmin()} has {zone_stats['State_Count'].min()} states
    • **Average Zone Size**: {zone_stats['Total_Area'].mean():,.0f} km²
    """
    
    st.markdown(summary_text)


def create_tribal_analysis(gdf):
    """Create enhanced tribal analysis with visual statistics"""
    st.subheader("👥 Enhanced Tribal Analysis")
    
    # Load tribes data for additional insights
    tribes_file = os.path.join('data', 'raw', 'tribes.csv')
    if os.path.exists(tribes_file):
        tribes_data = pd.read_csv(tribes_file)
        
        # Executive metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Ethnic Groups", len(tribes_data))
        
        with col2:
            largest_tribe = tribes_data.loc[tribes_data['Estimated_Population_Millions'].idxmax(), 'Ethnic_Group']
            st.metric("Largest Tribe", largest_tribe)
        
        with col3:
            total_pop = tribes_data['Estimated_Population_Millions'].sum()
            st.metric("Total Population", f"{total_pop:.1f}M")
        
        with col4:
            most_widespread = tribes_data.loc[tribes_data['Main_States'].str.count(',').idxmax(), 'Ethnic_Group']
            st.metric("Most Widespread", most_widespread)
        
        # Visual Analytics
        st.subheader("📊 Tribal Population Charts")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Bar chart for top tribes
            st.write("**Top 8 Tribes by Population**")
            top_8_tribes = tribes_data.nlargest(8, 'Estimated_Population_Millions')
            
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            bars = ax1.bar(top_8_tribes['Ethnic_Group'], top_8_tribes['Estimated_Population_Millions'], 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'])
            
            ax1.set_title('Top Tribes by Population (Millions)', fontsize=14, weight='bold')
            ax1.set_ylabel('Population (Millions)')
            ax1.set_xlabel('Ethnic Groups')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}M', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with chart_col2:
            # Pie chart for population percentage
            st.write("**Population Distribution (%)**")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Get top 6 tribes and group others
            top_6 = tribes_data.nlargest(6, 'Percentage')
            others_percentage = 100 - top_6['Percentage'].sum()
            
            labels = list(top_6['Ethnic_Group']) + ['Others']
            sizes = list(top_6['Percentage']) + [others_percentage]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#E0E0E0']
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
            
            ax2.set_title('Nigeria Population by Ethnic Group', fontsize=14, weight='bold')
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Top tribes table
        st.subheader("📋 Detailed Tribal Statistics")
        top_tribes = tribes_data.nlargest(8, 'Estimated_Population_Millions')[['Ethnic_Group', 'Estimated_Population_Millions', 'Percentage', 'Main_Zones']]
        top_tribes['Population'] = top_tribes['Estimated_Population_Millions'].apply(lambda x: f"{x:.1f}M")
        top_tribes['Percentage_Formatted'] = top_tribes['Percentage'].apply(lambda x: f"{x:.1f}%")
        top_tribes = top_tribes.rename(columns={
            'Ethnic_Group': 'Tribe',
            'Main_Zones': 'Primary Zones'
        })
        
        st.dataframe(top_tribes[['Tribe', 'Population', 'Percentage_Formatted', 'Primary Zones']], use_container_width=True)
    
    # State tribal diversity analysis
    st.subheader("🏛️ State Tribal Diversity Analysis")
    
    # Count tribes per state
    gdf['tribe_count'] = gdf['Major_Tribes'].str.count(',') + 1
    gdf['tribe_count'] = gdf['tribe_count'].fillna(1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_diverse = gdf.loc[gdf['tribe_count'].idxmax()]
        st.metric("Most Diverse State", f"{most_diverse['shapeName']}")
        st.caption(f"{int(most_diverse['tribe_count'])} major tribes")
    
    with col2:
        least_diverse = gdf.loc[gdf['tribe_count'].idxmin()]
        st.metric("Least Diverse State", f"{least_diverse['shapeName']}")
        st.caption(f"{int(least_diverse['tribe_count'])} major tribes")
    
    with col3:
        avg_diversity = gdf['tribe_count'].mean()
        st.metric("Average Diversity", f"{avg_diversity:.1f}")
        st.caption("tribes per state")
    
    # Diversity distribution chart
    st.write("**Tribal Diversity Distribution**")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    diversity_counts = gdf['tribe_count'].value_counts().sort_index()
    bars = ax3.bar(diversity_counts.index, diversity_counts.values, color='#6C5CE7')
    
    ax3.set_title('Number of States by Tribal Diversity Level', fontsize=14, weight='bold')
    ax3.set_xlabel('Number of Major Tribes per State')
    ax3.set_ylabel('Number of States')
    ax3.set_xticks(diversity_counts.index)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig3)


def create_political_analysis(gdf):
    """Create enhanced political party analysis with visual charts"""
    st.subheader("🏛️ Enhanced Political Analysis")
    
    # Party distribution analysis
    party_counts = {}
    for parties in gdf['Typical_Parties'].dropna():
        for party in parties.split(','):
            party = party.strip()
            party_counts[party] = party_counts.get(party, 0) + 1
    
    if party_counts:
        # Executive metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_parties = len(party_counts)
            st.metric("Total Parties", total_parties)
        
        with col2:
            dominant_party = max(party_counts, key=party_counts.get)
            st.metric("Dominant Party", dominant_party)
        
        with col3:
            dominant_count = party_counts[dominant_party]
            st.metric("States Controlled", f"{dominant_count}/37")
        
        with col4:
            dominance_pct = (dominant_count / len(gdf)) * 100
            st.metric("Dominance %", f"{dominance_pct:.1f}%")
        
        # Visual Analytics
        st.subheader("📊 Political Party Charts")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Bar chart for party distribution
            st.write("**Party Distribution Across States**")
            party_df = pd.DataFrame(list(party_counts.items()), columns=['Party', 'States'])
            party_df = party_df.sort_values('States', ascending=False)
            
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
            bars = ax1.bar(party_df['Party'], party_df['States'], color=colors[:len(party_df)])
            
            ax1.set_title('States per Political Party', fontsize=14, weight='bold')
            ax1.set_ylabel('Number of States')
            ax1.set_xlabel('Political Parties')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with chart_col2:
            # Pie chart for party control percentage
            st.write("**Political Control Distribution**")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            party_percentages = [(count / len(gdf)) * 100 for count in party_df['States']]
            
            wedges, texts, autotexts = ax2.pie(
                party_percentages,
                labels=party_df['Party'],
                colors=colors[:len(party_df)],
                autopct='%1.1f%%',
                startangle=90
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
            
            ax2.set_title('Political Party Control (%)', fontsize=14, weight='bold')
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Detailed party breakdown
        st.subheader("📋 Detailed Party Statistics")
        party_df['Percentage'] = (party_df['States'] / len(gdf) * 100).apply(lambda x: f"{x:.1f}%")
        party_df['Control_Level'] = party_df['States'].apply(lambda x: 
            'Dominant' if x >= 10 else 'Moderate' if x >= 5 else 'Limited')
        
        st.dataframe(party_df, use_container_width=True)
    
    # Zone-party analysis
    st.subheader("🗺️ Zone-Party Political Landscape")
    
    # Create zone-party matrix
    zone_party_matrix = {}
    for zone in gdf['Zone'].unique():
        zone_data = gdf[gdf['Zone'] == zone]
        zone_parties = {}
        
        for parties in zone_data['Typical_Parties'].dropna():
            for party in parties.split(','):
                party = party.strip()
                zone_parties[party] = zone_parties.get(party, 0) + 1
        
        zone_party_matrix[zone] = zone_parties
    
    # Convert to DataFrame for visualization
    zone_party_df = pd.DataFrame(zone_party_matrix).fillna(0).astype(int)
    
    if not zone_party_df.empty:
        st.write("**Party Presence by Zone (Heatmap)**")
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        
        # Create heatmap
        import numpy as np
        im = ax3.imshow(zone_party_df.values, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax3.set_xticks(np.arange(len(zone_party_df.columns)))
        ax3.set_yticks(np.arange(len(zone_party_df.index)))
        ax3.set_xticklabels(zone_party_df.columns)
        ax3.set_yticklabels(zone_party_df.index)
        
        # Add text annotations
        for i in range(len(zone_party_df.index)):
            for j in range(len(zone_party_df.columns)):
                text = ax3.text(j, i, zone_party_df.iloc[i, j],
                               ha="center", va="center", color="black", weight="bold")
        
        ax3.set_title('Party Presence Across Zones', fontsize=14, weight='bold')
        ax3.set_xlabel('Geopolitical Zones')
        ax3.set_ylabel('Political Parties')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        st.pyplot(fig3)
    
    # Zone-party correlation table
    st.subheader("📊 Zone-Party Detailed Breakdown")
    zone_party_list = []
    for zone in gdf['Zone'].unique():
        zone_data = gdf[gdf['Zone'] == zone]
        for _, row in zone_data.iterrows():
            if pd.notna(row['Typical_Parties']):
                zone_party_list.append({
                    'Zone': zone,
                    'State': row['shapeName'],
                    'Parties': row['Typical_Parties'],
                    'Major_Tribes': row['Major_Tribes']
                })
    
    zone_party_detailed = pd.DataFrame(zone_party_list)
    if not zone_party_detailed.empty:
        st.dataframe(zone_party_detailed, use_container_width=True)
    
    # Political insights summary
    st.subheader("📄 Political Insights Summary")
    
    if party_counts:
        top_3_parties = sorted(party_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        insights = f"""
        **Nigeria Political Landscape Analysis:**
        
        • **Party Diversity**: {len(party_counts)} political parties active across 37 states
        • **Leading Party**: {top_3_parties[0][0]} controls {top_3_parties[0][1]} states ({(top_3_parties[0][1]/37)*100:.1f}%)
        • **Runner-up**: {top_3_parties[1][0]} controls {top_3_parties[1][1]} states ({(top_3_parties[1][1]/37)*100:.1f}%)
        • **Third Place**: {top_3_parties[2][0]} controls {top_3_parties[2][1]} states ({(top_3_parties[2][1]/37)*100:.1f}%)
        • **Political Competition**: {'High' if len(party_counts) >= 4 else 'Moderate' if len(party_counts) >= 3 else 'Limited'} party competition
        • **Most Contested**: Multi-party presence indicates active political competition
        """
        
        st.markdown(insights)


def main():
    # Custom CSS for better styling and centering
    st.markdown("""
            <style>
            .stApp {
                max-width: 100%;
                padding: 0;
                background: transparent !important;
            }
            iframe {
                width: 100% !important;
                border: none !important;
                height: 600px !important;
                background: transparent !important;
                padding: 0 !important;
                margin: 0 auto !important;
                display: block !important;
            }
            [data-testid="stAppViewContainer"] {
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                flex-direction: column !important;
            }
            .element-container {
                background: transparent !important;
                padding: 0 !important;
                margin: 0 auto !important;
                display: flex !important;
                justify-content: center !important;
                max-width: 800px !important;
            }
            .main .block-container {
                padding: 0 !important;
                margin: 0 auto !important;
                max-width: 100% !important;
            }
            .stDeployButton {
                display: none !important;
            }
            </style>
        """, unsafe_allow_html=True)
    
    # Main content
    # Load data with progress indicator
    with st.spinner("Loading data..."):
        gdf = load_data()
        if gdf is None:
            st.error("Failed to load data. Please try again later.")
            st.stop()
    
    # Assign zone-based colors
    gdf = assign_zone_colors(gdf)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌍 Interactive Map", 
        "📊 Static Map", 
        "📋 Data Table",
        "🗺️ Zone Analysis",
        "👥 Tribal Analysis", 
        "🏛️ Political Analysis"
    ])
    
    with tab1:
        st.header("Interactive Map - By Geopolitical Zones")
        st.info("🖱️ Hover over states for quick info • 🖱️ Click on states for detailed popups with comprehensive data")
        
        # Display interactive map
        interactive_map = create_interactive_map(gdf)
        st_folium(
            interactive_map,
            width="100%",
            height=600,
            returned_objects=[],
            key="nigeria_map"
        )
    
    with tab2:
        st.header("Static Map - By Geopolitical Zones")
        st.info("A static visualization with zone-based coloring and state labels")
        static_map = create_static_map(gdf)
        st.pyplot(static_map, use_container_width=True)
    
    with tab3:
        st.header("States Data Analysis")
        
        # Add summary statistics
        col1, col2 = st.columns(2)
        
        # Convert area to numeric for calculations
        numeric_area = pd.to_numeric(gdf['area_km2'], errors='coerce')
        
        with col1:
            total_states = len(gdf)
            total_area = numeric_area.sum()
            st.metric("Total States", total_states)
            st.metric("Total Area (km²)", f"{total_area:,.2f}")
        
        with col2:
            avg_area = numeric_area.mean()
            largest_state_idx = numeric_area.idxmax()
            largest_state = gdf.loc[largest_state_idx, 'shapeName'] if pd.notna(largest_state_idx) else "N/A"
            st.metric("Average State Area (km²)", f"{avg_area:,.2f}")
            st.metric("Largest State", largest_state)
        
        st.divider()
        
        # Display the full data table
        st.subheader("Complete Dataset")
        display_data = gdf.drop('geometry', axis=1).copy()
        
        # Ensure area_km2 is numeric and format it
        display_data['area_km2'] = pd.to_numeric(display_data['area_km2'], errors='coerce')
        display_data['area_km2'] = display_data['area_km2'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(display_data, hide_index=True, use_container_width=True)
    
    with tab4:
        create_zone_analysis(gdf)
    
    with tab5:
        create_tribal_analysis(gdf)
    
    with tab6:
        create_political_analysis(gdf)


if __name__ == "__main__":
    main()