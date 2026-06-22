# src/download_noaa_exposure.py
import os
import io
import zipfile
import requests
import pandas as pd

# The exact target backend URL for the Coastal Economy data sheet
NOAA_COASTAL_ECONOMY_URL = "https://coast.noaa.gov/htdata/SocioEconomic/CoastalEconomy.zip"
DATA_DIR = "./data"
OUTPUT_CLEAN_PATH = "./data/noaa_coastal_gdp_weights.csv"

def download_coastal_economy_direct():
    """
    Downloads the specific Coastal Economy database zip, extracts the flat 
    county aggregation time-series, and normalizes it for exposure seeding.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("--- Fetching Dedicated NOAA Coastal Economy Aggregations ---")
    print(f"Source: {NOAA_COASTAL_ECONOMY_URL}")
    
    try:
        response = requests.get(NOAA_COASTAL_ECONOMY_URL, timeout=30)
        if response.status_code == 200:
            print("Download successful. Extracting archive contents...")
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Target the exact main database file discovered in your printout
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    print("Structure mismatch. Found files:")
                    print(z.namelist()[:5])
                    return None
                
                # ... inside download_coastal_economy_direct() ...
                target_file = 'CoastalEconomy.csv' if 'CoastalEconomy.csv' in csv_files else csv_files[0]
                print(f"Extracting and reading table: {target_file}")
                
                with z.open(target_file) as f:
                    # Added encoding='latin1' to handle proprietary symbols and smart quotes cleanly
                    df = pd.read_csv(f, encoding='latin1')

                # --- ADD THIS LINE TO SAVE THE FULL RAW FILE TO DISK ---
                RAW_OUTPUT_PATH = os.path.join(DATA_DIR, "CoastalEconomy.csv")
                df.to_csv(RAW_OUTPUT_PATH, index=False)
                print(f"Saved full raw dataset to {RAW_OUTPUT_PATH}")
            
            # Standardize column headers
            df.columns = [col.strip().lower() for col in df.columns]
            return df
        else:
            print(f"Failed to connect. Server returned status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error handling the network payload: {e}")
        return None

def process_county_weights(df):
    """
    Isolates target hazard-exposed markets using correct NOAA CoastalEconomy headers
    and normalizes their total GDP shares to drive property asset accumulation.
    """
    if df is None:
        print("Aborting calculation: No data frame received.")
        return
        
    # Standardize column mapping based on your sample headers
    name_col = 'geoname' if 'geoname' in df.columns else 'geoid'
    gdp_col = 'gdp' if 'gdp' in df.columns else 'realgdp'
    year_col = 'year' if 'year' in df.columns else None
    
    # Target property hubs
    target_hubs = ['Miami-Dade', 'Hillsborough', 'Harris', 'Orleans', 'Charleston', 'New Hanover']
    
    # 1. Filter for the most recent statistical data year in the file
    if year_col and year_col in df.columns:
        latest_year = df[year_col].max()
        df = df[df[year_col] == latest_year]
        print(f"Isolating coastal metrics for operational year: {latest_year}")
        
    # 2. Extract matching records from the geoname column
    # Using a case-insensitive regex match to catch variations like "Miami-Dade County"
    df_filtered = df[df[name_col].str.contains('|'.join(target_hubs), case=False, na=False)].copy()
    
    if df_filtered.empty:
        print("Warning: Filtered DataFrame is empty! Double check target_hubs names vs. geoname entries.")
        print("Sample geoname values in data:")
        print(df[name_col].dropna().unique()[:10])
        return

    # 3. Handle cleaning types on financial values if parsed as strings
    if df_filtered[gdp_col].dtype == object:
        df_filtered[gdp_col] = df_filtered[gdp_col].astype(str).str.replace(',', '').str.replace('$', '')
        df_filtered[gdp_col] = pd.to_numeric(df_filtered[gdp_col], errors='coerce')
    else:
        df_filtered[gdp_col] = pd.to_numeric(df_filtered[gdp_col], errors='coerce')
        
    # Drop any rows where GDP parsing failed
    df_filtered = df_filtered.dropna(subset=[gdp_col])

    # 4. Group and aggregate unique rows
    df_summary = df_filtered.groupby(name_col)[gdp_col].first().reset_index()
    
    # 5. Compute normalized baseline weights
    total_gdp = df_summary[gdp_col].sum()
    if total_gdp > 0:
        df_summary['gdp_weight'] = df_summary[gdp_col] / total_gdp
    else:
        df_summary['gdp_weight'] = 1.0 / len(df_summary)
    
    df_summary.to_csv(OUTPUT_CLEAN_PATH, index=False)
    print(f"\nSuccessfully compiled NOAA Coastal Economy metrics into {OUTPUT_CLEAN_PATH}:")
    print(df_summary[[name_col, 'gdp_weight']])


if __name__ == "__main__":
    raw_df = download_coastal_economy_direct()
    process_county_weights(raw_df)