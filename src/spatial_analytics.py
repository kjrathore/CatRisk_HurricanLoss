# src/spatial_analytics.py
import os
import pandas as pd
import numpy as np

class SpatialAnalyticsModule:
    def __init__(self, resolution_degrees: float = 0.05):
        # 0.05 degrees is roughly equal to a 5km x 5km spatial grid block
        self.grid_res = resolution_degrees

    def aggregate_spatial_exposure(self, exposure_path: str, damage_matrix_path: str):
        print("--- Running Macro Spatial Aggregation Engine ---")
        
        # Load backend databases
        df_exposure = pd.read_csv(exposure_path)
        df_damage = pd.read_csv(damage_matrix_path)
        
        print("Binning properties into broader 5km grid blocks...")
        # Create macro spatial grid coordinates by rounding exact lat/lons
        df_exposure['grid_lat'] = np.round(df_exposure['latitude'] / self.grid_res) * self.grid_res
        df_exposure['grid_lon'] = np.round(df_exposure['longitude'] / self.grid_res) * self.grid_res
        
        # 1. Aggregate Wealth Accumulation per spatial grid block
        grid_wealth = df_exposure.groupby(['county', 'grid_lat', 'grid_lon']).agg(
            total_properties=('property_id', 'count'),
            total_value_at_risk_usd=('total_insured_value_usd', 'sum')
        ).reset_index()
        
        # 2. Compute Macro County-Level Risk Profile Metrics
        print("\n--- Compiling Broader County Spatial Risk Metrics ---")
        
        # Calculate average historical hazard impact wind speeds felt across the county space
        county_hazard_summary = df_damage.merge(df_exposure, on='property_id')
        
        county_metrics = county_hazard_summary.groupby('county').agg(
            total_historical_impacts=('event_id', 'count'),
            avg_wind_speed_felt=('local_wind_speed_knots', 'mean'),
            max_wind_speed_felt=('local_wind_speed_knots', 'max'),
            total_ground_damage_usd=('ground_loss_usd', 'sum')
        ).reset_index()
        
        # Calculate what percentage of total regional wealth was damaged overall
        county_tiv_totals = df_exposure.groupby('county')['total_insured_value_usd'].sum().reset_index()
        county_metrics = county_metrics.merge(county_tiv_totals, on='county')
        county_metrics['regional_loss_ratio_pct'] = (county_metrics['total_ground_damage_usd'] / county_metrics['total_insured_value_usd']) * 100
        
        # Format for clean display
        display_df = county_metrics.copy()
        display_df['total_insured_value_usd'] = display_df['total_insured_value_usd'].map('${:,.2f}'.format)
        display_df['total_ground_damage_usd'] = display_df['total_ground_damage_usd'].map('${:,.2f}'.format)
        display_df['avg_wind_speed_felt'] = display_df['avg_wind_speed_felt'].map('{:.1f} kts'.format)
        display_df['regional_loss_ratio_pct'] = display_df['regional_loss_ratio_pct'].map('{:.4f}%'.format)
        
        print(display_df_string := display_df[[
            'county', 'total_properties_count' if 'total_properties_count' in display_df else 'avg_wind_speed_felt', 
            'total_insured_value_usd', 'total_ground_damage_usd', 'regional_loss_ratio_pct'
        ]].to_string(index=False))
        
        return grid_wealth, county_metrics

if __name__ == "__main__":
    analytics = SpatialAnalyticsModule()
    
    grid_df, county_df = analytics.aggregate_spatial_exposure(
        exposure_path="./data/portfolio_exposure.csv",
        damage_matrix_path="./data/portfolio_damage_matrix.csv"
    )
    
    # Save the spatial data layers to disk
    os.makedirs('./data', exist_ok=True)
    grid_df.to_csv("./data/spatial_grid_accumulation.csv", index=False)
    county_df.to_csv("./data/county_risk_metrics.csv", index=False)
    print("\nMacro spatial layers successfully exported to ./data/")