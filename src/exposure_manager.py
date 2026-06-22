# src/exposure_manager.py
import os
import pandas as pd
import numpy as np

class ExposureManager:
    def __init__(self, weights_path: str = "./data/noaa_coastal_gdp_weights.csv", seed: int = 101):
        np.random.seed(seed)
        self.weights_path = weights_path
        
        # Hardcoded spatial coordinates for the core economic centroids of our target counties
        # This maps the NOAA 'geoname' fields cleanly to realistic geographic pins
        self.county_coordinates = {
            'Miami-Dade County, FL':    {'lat': 25.7617,  'lon': -80.1918},
            'Hillsborough County, FL':  {'lat': 27.9506,  'lon': -82.4572},
            'Harris County, TX':        {'lat': 29.7604,  'lon': -95.3698},
            'Orleans Parish, LA':       {'lat': 29.9511,  'lon': -90.0715},
            'Charleston County, SC':    {'lat': 32.7765,  'lon': -79.9311},
            'New Hanover County, NC':   {'lat': 34.2104,  'lon': -77.8868}
        }
        
    def _load_noaa_weights(self):
        """Loads and aligns the generated NOAA data weights."""
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Missing baseline weights file: {self.weights_path}. Please run src.download_noaa_exposure first!"
            )
            
        df_weights = pd.read_csv(self.weights_path)
        
        # Align headers dynamically based on our processor's output schema
        name_col = 'geoname' if 'geoname' in df_weights.columns else df_weights.columns[0]
        weight_col = 'gdp_weight'
        
        # Build a dictionary mapping county names to their parsed data weights
        return dict(zip(df_weights[name_col], df_weights[weight_col]))

    def generate_portfolio(self, num_properties: int = 10000) -> pd.DataFrame:
        print("--- Running CatRisk Exposure Manager (NOAA-Driven Allocation) ---")
        
        noaa_weights = self._load_noaa_weights()
        counties = list(noaa_weights.keys())
        weights = list(noaa_weights.values())
        
        # Normalize weights just in case of slight floating-point parsing rounding variances
        weights = [w / sum(weights) for w in weights]
        
        # Sample rows mimicking real-world wealth distribution density
        sampled_counties = np.random.choice(counties, size=num_properties, p=weights)
        
        portfolio_data = []
        for i, county_name in enumerate(sampled_counties):
            prop_id = f"PROP_{i+1:06d}"
            
            # Extract state identifier string dynamically from county name context
            state = county_name.split(', ')[1] if ', ' in county_name else 'US'
            
            # Match county coordinates fallback to a general mean location if not explicitly tracked
            coords = self.county_coordinates.get(county_name, {'lat': 28.0, 'lon': -82.0})
            
            # Inject tight Gaussian jitter around county hubs to represent suburban sprawl distribution
            lat = coords['lat'] + np.random.normal(0, 0.12)
            lon = coords['lon'] + np.random.normal(0, 0.12)
            
            # Assign structural property features
            occupancy = np.random.choice(['Residential', 'Commercial'], p=[0.75, 0.25])
            
            if occupancy == 'Residential':
                construction = np.random.choice(['Wood Frame', 'Masonry'], p=[0.85, 0.15])
                tiv = float(np.random.lognormal(mean=13.1, sigma=0.35)) # Mid-range residential asset log-normal curve
            else:
                construction = np.random.choice(['Steel Frame', 'Reinforced Concrete'], p=[0.35, 0.65])
                tiv = float(np.random.lognormal(mean=16.2, sigma=0.75)) # Right-skewed multi-million commercial asset curve
                
            portfolio_data.append({
                "property_id": prop_id,
                "county": county_name,
                "state": state,
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "occupancy": occupancy,
                "construction": construction,
                "total_insured_value_usd": round(tiv, 2)
            })
            
        df_exposure = pd.DataFrame(portfolio_data)
        total_val = df_exposure['total_insured_value_usd'].sum()
        
        print(f"Portfolio Generation Process complete.")
        print(f"Total Portfolio Exposure Asset Accumulation: ${total_val:,.2f}\n")
        return df_exposure

if __name__ == "__main__":
    manager = ExposureManager()
    exposure_df = manager.generate_portfolio(num_properties=10000)
    
    output_file = "./data/portfolio_exposure.csv"
    exposure_df.to_csv(output_file, index=False)


    # Add this to the bottom of your script to inspect the portfolio breakdown
    print("--- Portfolio Concentration Breakdown by County ---")
    summary = exposure_df.groupby('county').agg(
        property_count=('property_id', 'count'),
        total_tiv_usd=('total_insured_value_usd', 'sum')
    ).reset_index()
    
    summary['tiv_percentage'] = (summary['total_tiv_usd'] / summary['total_tiv_usd'].sum()) * 100
    
    # Format currency for readable terminal output
    summary['total_tiv_usd'] = summary['total_tiv_usd'].map('${:,.2f}'.format)
    summary['tiv_percentage'] = summary['tiv_percentage'].map('{:.2f}%'.format)
    print(summary.to_string(index=False))

    
    print(f"Successfully saved un-hardcoded exposure profile database to {output_file}")