import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import os
from etl import NigeriaDataETL

# Set page config
st.set_page_config(
    page_title="Nigeria States Visualization",
    page_icon="🗺️",
    layout="wide"
)

def load_data():
    """Load and return the processed GeoJSON data"""
    data_dir = 'data'
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    processed_file = os.path.join(data_dir, 'processed_nigeria_states.geojson')
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

def create_static_map(gdf):
    """Create a static matplotlib map focused on Nigeria"""
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot states with unique colors and clear boundaries
    gdf.plot(
        ax=ax,
        column='shapeName',
        cmap='tab20',
        legend=False,
        edgecolor='black',
        linewidth=1.5
    )
    # Add country boundary outline
    country_boundary = gdf.union_all().boundary  # Using union_all() instead of unary_union
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
            ha='center'
        )
    # Isolate to Nigeria bounds (remove extra space)
    minx, miny, maxx, maxy = gdf.total_bounds
    ax.set_xlim(minx - 0.5, maxx + 0.5)  # Small buffer; adjust as needed
    ax.set_ylim(miny - 0.5, maxy + 0.5)
    # Hide axis frame/square
    ax.set_title('Nigeria States Map')
    ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig

def create_interactive_map(gdf):
    """Create an interactive folium map focused on Nigeria"""
    # Calculate center point using bounds
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    # Create the base map with transparent background
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles=None,  # No basemap tiles
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
    # Add GeoJSON layer with styling and tool-tips
    folium.GeoJson(
        data=geo_data,
        name='Nigeria States',
        style_function=lambda x: {
            'fillColor': '#3388ff',
            'color': '#000000',
            'weight': 1.5,
            'fillOpacity': 0.6
        },
        highlight_function=lambda x: {
            'weight': 2,
            'fillOpacity': 0.8
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['shapeName', 'area_km2'],
            aliases=['State:', 'Area (km²):'],
            localize=True,
            sticky=True,
            style="""
                    background-color: white;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                    font-family: Arial;
                    font-size: 12px;
                    padding: 10px;
                """
        )
    ).add_to(m)
    # Fit map to Nigeria bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m

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
                border: none !important;
                height: 600px !important;  /* Adjust height as needed */
                background: transparent !important;
                padding: 0 !important;
                margin: 0 auto !important;
                display: block !important;
            }
            /* Center the map container */
            [data-testid="stAppViewContainer"] {
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                flex-direction: column !important;
            }
            .element-container {
                background: transparent !important;
                padding: 0 !important;
                margin: 0 auto !important;  /* Center the element container */
                display: flex !important;
                justify-content: center !important;
                max-width: 800px !important;  /* Match your desired map width */
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

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["🌍 Interactive Map", "📊 Static Map", "📋 Data Table"])
    
    with tab1:
        # Display interactive map (remove width:100% from iframe CSS to allow fixed width centering)
        interactive_map = create_interactive_map(gdf)
        st_folium(
            interactive_map,
            width=800,  # Fixed width for the map; adjust as needed
            height=600,
            returned_objects=[],
            key="nigeria_map"
        )
    
    with tab2:
        st.header("Static Map")
        st.info("A static visualization with state labels and clear boundaries")
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
        
        st.dataframe(
            display_data,
            hide_index=True,
            use_container_width=True
        )

if __name__ == "__main__":
    main()