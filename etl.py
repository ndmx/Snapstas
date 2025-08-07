import requests
import geopandas as gpd
import pandas as pd
import os
import json

class NigeriaDataETL:
    def __init__(self):
        self.data_dir = 'data'
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.metadata_dir = os.path.join(self.data_dir, 'metadata')
        
        # Raw data files
        self.raw_geojson_file = os.path.join(self.raw_dir, 'nigeria_states.geojson')
        self.raw_area_file = os.path.join(self.raw_dir, 'nigeria_states_data.csv')
        self.states_demographic_file = os.path.join(self.raw_dir, 'states_zones_parties_tribes.csv')
        self.tribes_file = os.path.join(self.raw_dir, 'tribes.csv')
        
        # Processed data files
        self.processed_file = os.path.join(self.processed_dir, 'processed_nigeria_states.geojson')
        
        # Ensure directories exist
        for directory in [self.raw_dir, self.processed_dir, self.metadata_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
    def fetch_data(self):
        """Fetch Nigeria states GeoJSON data from geoBoundaries"""
        try:
            # URL for Nigeria states data from geoBoundaries
            url = "https://www.geoboundaries.org/api/current/gbOpen/NGA/ADM1/"
            
            print("Fetching Nigeria states data...")
            response = requests.get(url)
            response.raise_for_status()
            
            # Get the download URL from the response
            data = response.json()
            download_url = data['gjDownloadURL']
            
            # Download the actual GeoJSON file
            print("Downloading GeoJSON file...")
            geojson_response = requests.get(download_url)
            geojson_response.raise_for_status()
            
    # Create data directory if it doesn't exist
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            
            # Save the raw data
            with open(self.raw_geojson_file, 'w') as f:
                f.write(geojson_response.text)
            
            print(f"Data saved to {self.raw_geojson_file}")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return False
    
    def load_demographic_data(self):
        """Load demographic CSV data"""
        try:
            print("Loading demographic data...")
            
            # Load states demographic data
            if os.path.exists(self.states_demographic_file):
                states_demo = pd.read_csv(self.states_demographic_file)
                print(f"Loaded states demographic data: {len(states_demo)} states")
            else:
                print(f"Warning: {self.states_demographic_file} not found")
                states_demo = None
            
            # Load tribes data
            if os.path.exists(self.tribes_file):
                tribes_data = pd.read_csv(self.tribes_file)
                print(f"Loaded tribes data: {len(tribes_data)} ethnic groups")
            else:
                print(f"Warning: {self.tribes_file} not found")
                tribes_data = None
            
            return states_demo, tribes_data
            
        except Exception as e:
            print(f"Error loading demographic data: {str(e)}")
            return None, None
    
    def merge_demographic_data(self, gdf, states_demo):
        """Merge demographic data with GeoJSON data"""
        try:
            print("Merging demographic data...")
            
            if states_demo is None:
                print("No demographic data to merge")
                return gdf
            
            # Check for state name mismatches
            geo_states = set(gdf['shapeName'].unique())
            demo_states = set(states_demo['State'].unique())
            
            print(f"GeoJSON states: {len(geo_states)}")
            print(f"Demographic states: {len(demo_states)}")
            
            # Find mismatches
            missing_in_demo = geo_states - demo_states
            missing_in_geo = demo_states - geo_states
            
            if missing_in_demo:
                print(f"States in GeoJSON but missing in demographic data: {missing_in_demo}")
            if missing_in_geo:
                print(f"States in demographic data but missing in GeoJSON: {missing_in_geo}")
            
            # Merge the data
            merged_gdf = gdf.merge(
                states_demo, 
                left_on='shapeName', 
                right_on='State', 
                how='left'
            )
            
            # Check merge results
            merged_count = merged_gdf['Zone'].notna().sum()
            print(f"Successfully merged demographic data for {merged_count} states")
            
            return merged_gdf
            
        except Exception as e:
            print(f"Error merging demographic data: {str(e)}")
            return gdf
    
    def process_data(self):
        """Process the raw data sources and create integrated dataset"""
        try:
            print("Processing data...")
            
            # Load the GeoJSON data
            gdf = gpd.read_file(self.raw_geojson_file)
            
            # Ensure we have the required columns
            if 'shapeName' not in gdf.columns:
                # Try to find alternative column names
                name_columns = [col for col in gdf.columns if 'name' in col.lower() or 'state' in col.lower()]
                if name_columns:
                    gdf['shapeName'] = gdf[name_columns[0]]
                else:
                    gdf['shapeName'] = gdf.index.astype(str)
            
            # Load and merge area data from CSV (preferred source)
            if os.path.exists(self.raw_area_file):
                print("Loading area data from CSV...")
                area_data = pd.read_csv(self.raw_area_file)
                gdf = gdf.merge(
                    area_data[['shapeName', 'area_km2']], 
                    on='shapeName', 
                    how='left'
                )
                print("Successfully merged area data from CSV")
            else:
                print("CSV area data not found, calculating from geometry...")
                # Fallback to geometric calculation
                gdf['area_km2'] = gdf.geometry.area / 1000000  # Convert to km²
            
            # Load and merge demographic data
            states_demo, tribes_data = self.load_demographic_data()
            gdf = self.merge_demographic_data(gdf, states_demo)
            
            # Create metadata file
            self.create_metadata(gdf, tribes_data)
            
            # Save processed data
            gdf.to_file(self.processed_file, driver='GeoJSON')
            print(f"Processed data saved to {self.processed_file}")
            
            return True
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return False
    
    def create_metadata(self, gdf, tribes_data):
        """Create metadata file with data source information"""
        try:
            metadata = {
                "data_sources": {
                    "geojson_source": "geoBoundaries.org - Nigeria ADM1 boundaries",
                    "area_data_source": "nigeria_states_data.csv - Official state areas",
                    "demographic_source": "states_zones_parties_tribes.csv - Political and tribal data",
                    "tribal_source": "tribes.csv - Population statistics"
                },
                "processing_info": {
                    "total_states": int(len(gdf)),
                    "states_with_demographic_data": int(gdf['Zone'].notna().sum() if 'Zone' in gdf.columns else 0),
                    "total_ethnic_groups": int(len(tribes_data) if tribes_data is not None else 0),
                    "area_data_source": "CSV" if os.path.exists(self.raw_area_file) else "Geometric calculation"
                },
                "data_quality": {
                    "missing_zones": int(gdf['Zone'].isna().sum() if 'Zone' in gdf.columns else len(gdf)),
                    "missing_areas": int(gdf['area_km2'].isna().sum() if 'area_km2' in gdf.columns else len(gdf))
                }
            }
            
            metadata_file = os.path.join(self.metadata_dir, 'data_sources.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Metadata saved to {metadata_file}")
            
        except Exception as e:
            print(f"Error creating metadata: {str(e)}")
    
    def run_pipeline(self):
        """Run the complete ETL pipeline"""
        print("Starting ETL pipeline...")
        
        # Step 1: Fetch data
        if not self.fetch_data():
            return False
        
        # Step 2: Process data
        if not self.process_data():
            return False
        
        print("ETL pipeline completed successfully!")
        return True

def main():
    """Main function to run the ETL pipeline"""
    etl = NigeriaDataETL()
    success = etl.run_pipeline()
    
    if success:
        print("ETL pipeline completed successfully!")
    else:
        print("ETL pipeline failed!")

if __name__ == "__main__":
    main()