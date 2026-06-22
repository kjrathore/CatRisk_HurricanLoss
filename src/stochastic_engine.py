# src/stochastic_engine.py
import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import weibull_min

class DualModeStochasticEngine:
    def __init__(self, poisson_lambda: float, historical_winds: np.ndarray, config_path: str = "config.yaml", seed: int = 42):
        self.poisson_lambda = poisson_lambda
        np.random.seed(seed)
        
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
            
        self.mode = config_data["model_settings"]["resolution_mode"]
        self.anchors = config_data["hazard"]["exposure_anchors"]
        
        print(f"Initializing Stochastic Engine in [{self.mode.upper()}] mode...")
        self.shape, self.loc, self.scale = weibull_min.fit(historical_winds)

    def generate_stochastic_catalog(self, num_years: int = 10000) -> pd.DataFrame:
        print(f"--- Generating {num_years:,}-Year Event Catalog ---")
        stochastic_events = []
        event_counter = 0
        
        storms_per_year = np.random.poisson(self.poisson_lambda, num_years)
        
        for year_idx in range(1, num_years + 1):
            num_storms = storms_per_year[year_idx - 1]
            
            for _ in range(num_storms):
                event_counter += 1
                simulated_max_wind = np.clip(weibull_min.rvs(self.shape, loc=self.loc, scale=self.scale), 34.0, 165.0)
                target_anchor = np.random.choice(self.anchors)
                
                # Spatial point center base
                lat = target_anchor['lat'] + np.random.normal(0, 0.75)
                lon = target_anchor['lon'] + np.random.normal(0, 0.75)
                
                record = {
                    "stochastic_year": year_idx,
                    "event_id": f"STOCH_{event_counter:07d}",
                    "simulated_max_wind_knots": float(simulated_max_wind)
                }
                
                if self.mode == "individual":
                    record["landfall_latitude"] = float(lat)
                    record["landfall_longitude"] = float(lon)
                    record["track_heading_degrees"] = float(np.random.normal(300.0, 25.0))
                else:
                    record["simulated_latitude"] = float(lat)
                    record["simulated_longitude"] = float(lon)
                    
                stochastic_events.append(record)
                
        df_catalog = pd.DataFrame(stochastic_events)
        
        # Branch directories based on selection parameters
        out_dir = f"./data/{self.mode}"
        os.makedirs(out_dir, exist_ok=True)
        
        output_path = f"{out_dir}/stochastic_event_catalog.csv"
        df_catalog.to_csv(output_path, index=False)
        print(f"Saved catalog smoothly inside structural path: {output_path}")
        return df_catalog

if __name__ == "__main__":
    df_raw = pd.read_csv("./data/hurdat2_clean.csv")
    df_all_peaks = df_raw.sort_values('max_wind_knots').groupby('storm_id').last().reset_index()
    
    total_years_recorded = df_raw['year'].nunique()
    poisson_lambda = df_raw[
        (df_raw['latitude'] >= 24.0) & (df_raw['latitude'] <= 36.0) &
        (df_raw['longitude'] >= -100.0) & (df_raw['longitude'] <= -75.0)
    ].groupby('year')['storm_id'].nunique().sum() / total_years_recorded
    
    engine = DualModeStochasticEngine(poisson_lambda=poisson_lambda, historical_winds=df_all_peaks['max_wind_knots'].values)
    engine.generate_stochastic_catalog(num_years=10000)