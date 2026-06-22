# src/hazard_footprint.py
import os
import yaml
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

class HazardFootprintEngine:
    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}.")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.mode = self.config["model_settings"]["resolution_mode"]
        self.r_max = self.config["hazard"]["r_max_km"]
        self.alpha = self.config["hazard"]["alpha_decay"]
        
        self.exposure_path = "./data/portfolio_exposure.csv"
        self.output_dir = f"./data/{self.mode}"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_stochastic_catalog(self) -> pd.DataFrame:
        """
        Loads the event catalog by safely probing for either .parquet or .csv variants,
        and dynamically rectifies tracking column header variants, adapting seamlessly 
        if only landfall coordinates are present.
        """
        parquet_path = f"./data/{self.mode}/stochastic_events.parquet"
        csv_path = f"./data/{self.mode}/stochastic_event_catalog.csv"
        
        if os.path.exists(parquet_path):
            print(f"Loading binary event matrix from: {parquet_path}")
            df = pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            print(f"Loading delimited catalog framework from: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"Could not locate an event catalog in {self.output_dir}. "
                f"Expected 'stochastic_events.parquet' or 'stochastic_event_catalog.csv'."
            )
            
        # Standardize column headers to uniform lowercase to avoid casing mismatches
        df.columns = [c.lower().strip() for c in df.columns]

        # FALLBACK MAPPING: If explicit track points are absent, fall back to landfall anchors
        if 'track_lat' not in df.columns and 'landfall_latitude' in df.columns:
            print(" -> Re-aliasing 'landfall_latitude' to unified 'track_lat' tracking sequence.")
            df['track_lat'] = df['landfall_latitude']
        if 'track_lon' not in df.columns and 'landfall_longitude' in df.columns:
            print(" -> Re-aliasing 'landfall_longitude' to unified 'track_lon' tracking sequence.")
            df['track_lon'] = df['landfall_longitude']

        # Dynamic mapping dictionary to handle structural edge cases
        rename_dict = {}
        for col in df.columns:
            if col in ['track_latitude', 'latitude', 'lat', 'simulated_latitude'] and 'track_lat' not in df.columns:
                rename_dict[col] = 'track_lat'
            elif col in ['track_longitude', 'longitude', 'lon', 'simulated_longitude'] and 'track_lon' not in df.columns:
                rename_dict[col] = 'track_lon'
            elif col in ['id', 'event']:
                rename_dict[col] = 'event_id'
            elif col in ['max_wind', 'intensity', 'wind_speed', 'simulated_max_wind_knots']:
                rename_dict[col] = 'max_wind_knots'
                
        if rename_dict:
            df = df.rename(columns=rename_dict)
            
        # Verification sanity check to catch structural key issues before processing
        required_keys = ['event_id', 'track_lat', 'track_lon', 'max_wind_knots']
        missing = [k for k in required_keys if k not in df.columns]
        if missing:
            raise KeyError(
                f"Catalog is missing mandatory structural keys: {missing}. "
                f"Available headers are: {list(df.columns)}"
            )
            
        return df

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculates distance between coordinate matrices in kilometers."""
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2) ** 2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def run_footprint_generation(self):
        print(f"\n--- Initializing Hazard Footprint Engine [{self.mode.upper()} Mode] ---")
        
        df_events = self.load_stochastic_catalog()
        if not os.path.exists(self.exposure_path):
            raise FileNotFoundError(f"Portfolio exposure catalog missing at {self.exposure_path}.")

        df_exposure = pd.read_csv(self.exposure_path)
        output_file = f"{self.output_dir}/hazard_footprint_matrix.parquet"

        if self.mode == "spatial":
            print("Processing footprint across regional macro spatial anchors...")
            anchors = self.config["hazard"]["exposure_anchors"]
            df_anchors = pd.DataFrame(anchors)
            
            closest_anchors = []
            for _, prop in df_exposure.iterrows():
                distances = self.haversine_distance(prop['latitude'], prop['longitude'], df_anchors['lat'], df_anchors['lon'])
                closest_idx = distances.idxmin()
                closest_anchors.append(df_anchors.loc[closest_idx, 'name'])
            df_exposure['nearest_spatial_anchor'] = closest_anchors
            
            df_exposure.to_csv(self.exposure_path, index=False)
            
            footprint_records = []
            for _, event in tqdm(df_events.iterrows(), total=len(df_events), desc="Spatial Tracking"):
                distances_km = self.haversine_distance(
                    event['track_lat'], event['track_lon'], 
                    df_anchors['lat'], df_anchors['lon']
                ).values
                
                intensities = event['max_wind_knots'] * np.minimum(1.0, (self.r_max / (distances_km + 1e-5)) ** self.alpha)
                
                for idx, anchor_name in enumerate(df_anchors['name']):
                    footprint_records.append({
                        'event_id': str(event['event_id']),
                        'spatial_anchor_name': anchor_name,
                        'wind_speed_knots': intensities[idx]
                    })
                    
            df_footprint = pd.DataFrame(footprint_records)
            df_footprint.to_parquet(output_file, index=False)
            print(f"SUCCESS: Generated {len(df_footprint):,} spatial metrics at: {output_file}")
            
        else: # individual point intercept tracking mode
            print("Processing point-intercept track footprints via memory-safe PyArrow streaming arrays...")
            
            # Define PyArrow Schema explicitly for fast binary disk output
            schema = pa.schema([
                ('event_id', pa.string()),
                ('property_id', pa.string()),
                ('wind_speed_knots', pa.float64())
            ])
            
            # Pre-extract underlying numpy arrays for ultra-fast vectorized calculations
            prop_ids = df_exposure['property_id'].astype(str).values
            prop_lats = df_exposure['latitude'].values
            prop_lons = df_exposure['longitude'].values
            
            total_records_written = 0
            
            # Open disk stream writer
            with pq.ParquetWriter(output_file, schema) as writer:
                # Iterate through events with a dynamic progress bar
                for _, event in tqdm(df_events.iterrows(), total=len(df_events), desc="Individual Intercept Tracking"):
                    # Calculate distances across all properties instantly via vectorized arrays
                    distances_km = self.haversine_distance(
                        event['track_lat'], event['track_lon'], 
                        prop_lats, prop_lons
                    )
                    
                    intensities = event['max_wind_knots'] * np.minimum(1.0, (self.r_max / (distances_km + 1e-5)) ** self.alpha)
                    
                    # Create a complete matrix batch without ANY inner loops
                    batch_df = pd.DataFrame({
                        'event_id': str(event['event_id']),
                        'property_id': prop_ids,
                        'wind_speed_knots': intensities
                    })
                    
                    # Convert to PyArrow Table and append to file instantly
                    table = pa.Table.from_pandas(batch_df, schema=schema)
                    writer.write_table(table)
                    total_records_written += len(batch_df)

            print(f"SUCCESS: Generated {total_records_written:,} individualized tracking rows directly at: {output_file}")

if __name__ == "__main__":
    engine = HazardFootprintEngine()
    engine.run_footprint_generation()