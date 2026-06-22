# src/data_preprocessor.py
import pandas as pd
import numpy as np

class CatRiskPreprocessor:
    def __init__(self, clean_data_path: str):
        self.df = pd.read_csv(clean_data_path)
        # Filter for the modern satellite era to avoid historical detection bias
        self.df_modern = self.df[(self.df['year'] >= 1970) & (self.df['year'] <= 2025)].copy()
        
    def calculate_baseline_metrics(self):
        """
        Calculates the historical baseline frequency and gathers 
        maximum wind speeds per storm to feed the stochastic engine.
        """
        print("--- Running CatRisk Preprocessor (1970-2025 Baselining) ---")
        
        # 1. Calculate Poisson Lambda (Mean storms per year)
        storms_per_year = self.df_modern.groupby('year')['storm_id'].nunique()
        poisson_lambda = storms_per_year.mean()
        
        # 2. Extract the peak max wind speed achieved by each unique storm
        peak_storm_winds = self.df_modern.groupby('storm_id')['max_wind_knots'].max().dropna().values
        
        print(f"Calculated Baseline Lambda (λ): {poisson_lambda:.4f} storms/year")
        print(f"Extracted {len(peak_storm_winds)} historical storm peak intensities for empirical resampling.\n")
        
        return {
            "poisson_lambda": poisson_lambda,
            "peak_wind_distribution": peak_storm_winds
        }

if __name__ == "__main__":
    # Test execution
    preprocessor = CatRiskPreprocessor(clean_data_path="./data/hurdat2_clean.csv")
    metrics = preprocessor.calculate_baseline_metrics()