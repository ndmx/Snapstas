import geopandas as gpd
import pandas as pd
import requests
import os
from datetime import datetime

class NigeriaDataETL:
    def __init__(self):
        self.data_dir = 'data'
        self.geojson_url = 'https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/NGA/ADM1/geoBoundaries-NGA-ADM1.geojson'
        self.ensure_data_directory()

    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_geojson(self):
        """Fetch GeoJSON data and save locally"""
        try:
            response = requests.get(self.geojson_url)
            response.raise_for_status()
            
            filepath = os.path.join(self.data_dir, 'nigeria_states.geojson')
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"Successfully downloaded GeoJSON to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error downloading GeoJSON: {e}")
            return None

    def get_state_areas(self):
        """Returns a dictionary of state names and their areas"""
        return {
            'Niger': 76363,
            'Borno': 70898,
            'Taraba': 54473,
            'Kaduna': 46053,
            'Bauchi': 45893,
            'Yobe': 45502,
            'Zamfara': 39762,
            'Adamawa': 36917,
            'Kwara': 36825,
            'Kebbi': 36800,
            'Benue': 34059,
            'Plateau': 30913,
            'Kogi': 29833,
            'Oyo': 28454,
            'Nasarawa': 27117,
            'Sokoto': 25973,
            'Katsina': 24192,
            'Jigawa': 23154,
            'Cross River': 22590,
            'Kano': 20131,
            'Edo': 19559,
            'Gombe': 18768,
            'Delta': 17698,
            'Ogun': 16762,
            'Ondo': 17500,
            'Rivers': 11077,
            'Bayelsa': 10773,
            'Osun': 9251,
            'Federal Capital Territory': 7315,
            'Enugu': 7161,
            'Akwa Ibom': 7081,
            'Ekiti': 6353,
            'Abia': 6320,
            'Ebonyi': 5670,
            'Imo': 5530,
            'Anambra': 4844,
            'Lagos': 3577
        }

    def process_geojson(self, filepath):
        """Process GeoJSON data and add additional metrics"""
        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError("GeoJSON file not found")

        # Read GeoJSON
        gdf = gpd.read_file(filepath)
        
        # Get accurate state areas
        state_areas = self.get_state_areas()
        
        # Extract state names without 'State' suffix and handle special cases
        gdf['state_name'] = gdf['shapeName'].str.replace(' State', '')
        # Handle special case for Abuja FCT
        gdf.loc[gdf['shapeName'] == 'Abuja Federal Capital Territory', 'state_name'] = 'Federal Capital Territory'
        gdf['area_km2'] = pd.to_numeric(gdf['state_name'].map(state_areas), errors='coerce')
        gdf = gdf.drop('state_name', axis=1)  # Remove temporary column
        
        # Add centroid coordinates
        gdf['centroid_lat'] = gdf.geometry.centroid.y
        gdf['centroid_lon'] = gdf.geometry.centroid.x
        
        # Save processed data
        processed_filepath = os.path.join(self.data_dir, 'processed_nigeria_states.geojson')
        gdf.to_file(processed_filepath, driver='GeoJSON')
        
        # Save a CSV with non-geometric data for easy analysis
        csv_filepath = os.path.join(self.data_dir, 'nigeria_states_data.csv')
        gdf.drop('geometry', axis=1).to_csv(csv_filepath, index=False)
        
        return processed_filepath, csv_filepath

    def run_pipeline(self):
        """Run the complete ETL pipeline"""
        print("Starting ETL pipeline...")
        
        # Record start time
        start_time = datetime.now()
        
        # Fetch data
        geojson_filepath = self.fetch_geojson()
        if not geojson_filepath:
            return False
        
        # Process data
        try:
            processed_filepath, csv_filepath = self.process_geojson(geojson_filepath)
            print(f"Data processing complete. Files saved:")
            print(f"- Processed GeoJSON: {processed_filepath}")
            print(f"- CSV data: {csv_filepath}")
            
            # Record completion time
            duration = datetime.now() - start_time
            print(f"\nETL pipeline completed in {duration}")
            return True
            
        except Exception as e:
            print(f"Error in data processing: {e}")
            return False

if __name__ == "__main__":
    etl = NigeriaDataETL()
    etl.run_pipeline()
