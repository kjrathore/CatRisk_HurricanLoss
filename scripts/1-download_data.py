import os
import requests
import pandas as pd
import sweetviz as sv
import numpy as np

os.makedirs('./data', exist_ok=True)
os.makedirs('./reports', exist_ok=True)

# Updated direct raw text file URL containing full data up to the latest re-analysis
RAW_HURDAT2_URL = "https://www.aoml.noaa.gov/hrd/hurdat/hurdat2.html"
LOCAL_RAW_PATH = "./data/hurdat2_raw.txt"
PROCESSED_CSV_PATH = "./data/hurdat2_clean.csv"
def parse_lat_lon(coord_str):
    """Converts HURDAT coordinate strings (e.g., '28.0N', '94.8W') to signed floats."""
    coord_str = coord_str.strip().upper()
    if not coord_str:
        return np.nan
    
    # Extract hemisphere identifier
    direction = coord_str[-1]
    try:
        val = float(coord_str[:-1])
        if direction in ['S', 'W']:
            val = -val
        return val
    except ValueError:
        return np.nan

def clean_and_parse_hurdat():
    if not os.path.exists(LOCAL_RAW_PATH):
        print("Downloading raw HURDAT2 data...")
        r = requests.get(RAW_HURDAT2_URL)
        with open(LOCAL_RAW_PATH, 'w', encoding='utf-8') as f:
            f.write(r.text)

    print("Parsing full dataset matching official NOAA metadata guidelines...")
    with open(LOCAL_RAW_PATH, 'r') as f:
        lines = f.readlines()

    flat_data = []
    current_id, current_name = None, None
    
    for line in lines:
        # Split and strip white spaces
        parts = [p.strip() for p in line.split(',') if p.strip()]
        
        # Skip completely empty lines or malformed trailing rows
        if len(parts) < 3:
            continue
            
        # Header Row (Always exactly 3 elements: ID, Name, Number of rows)
        if len(parts) == 3:
            current_id = parts[0]
            current_name = parts[1]
        else:
            # Safe check: An official observation row must have at least 7 items
            if len(parts) < 7:
                continue
                
            date_val = parts[0]
            time_val = parts[1]
            
            # Identify if an optional record identifier (like 'L') is pushing the columns over
            if parts[2] in ['L', 'W', 'P', 'I', 'C', 'G', 'E', 'S', 'R', 'T']:
                record_id = parts[2]
                status_idx = 3
            else:
                record_id = "None"
                status_idx = 2
                
            # Extra guard to ensure the row didn't cut off early
            if len(parts) <= status_idx + 3:
                continue

            status = parts[status_idx]
            lat = parse_lat_lon(parts[status_idx+1])
            lon = parse_lat_lon(parts[status_idx+2])
            
            def safe_int(val):
                try:
                    v = int(val)
                    return np.nan if v == -999 else v
                except ValueError:
                    return np.nan

            max_wind = safe_int(parts[status_idx+3])
            min_press = safe_int(parts[status_idx+4]) if len(parts) > status_idx+4 else np.nan

            flat_data.append({
                "storm_id": current_id,
                "name": current_name,
                "year": int(date_val[0:4]),
                "date": date_val,
                "time": time_val,
                "record_identifier": record_id,
                "status": status,
                "latitude": lat,
                "longitude": lon,
                "max_wind_knots": max_wind,
                "min_pressure_mb": min_press
            })

    df = pd.DataFrame(flat_data)
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    print(f"Standardized clean dataset written to {PROCESSED_CSV_PATH} ({len(df)} tracking points parsed)")
    return df

if __name__ == "__main__":
    clean_and_parse_hurdat()