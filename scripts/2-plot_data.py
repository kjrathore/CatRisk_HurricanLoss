import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import geoplot as gpt
from shapely.geometry import Point
from geodatasets import get_path

os.makedirs('./plots', exist_ok=True)
df = pd.read_csv('./data/hurdat2_clean.csv')

# --- GEOSPATIAL MAP WITH REAL LANDMASSES ---
print("Loading world map geometry and preparing spatial data...")

# 1. Convert our clean DataFrame into a GeoDataFrame
# Guard against missing coordinates before transforming
df_geo = df.dropna(subset=['latitude', 'longitude']).copy()
geometry = [Point(xy) for xy in zip(df_geo['longitude'], df_geo['latitude'])]
gdf = gpd.GeoDataFrame(df_geo, geometry=geometry, crs="EPSG:4326")

# 2. Define Saffir-Simpson Hurricane Wind Scale categories
def assign_category(knots):
    if knots < 34: return 'Tropical Depression (<34 kt)'
    elif knots < 64: return 'Tropical Storm (34-63 kt)'
    elif knots < 83: return 'Category 1 Hurricane (64-82 kt)'
    elif knots < 96: return 'Category 2 Hurricane (83-95 kt)'
    elif knots < 113: return 'Category 3 Hurricane (96-112 kt)'
    elif knots < 130: return 'Category 4 Hurricane (113-129 kt)'
    else: return 'Category 5 Hurricane (≥130 kt)'

# Assigning classic meteorological risk colors (Cool colors for weak, Hot/Bright colors for severe)
category_colors = {
    'Tropical Depression (<34 kt)': '#abd9e9',       # Light Ice Blue
    'Tropical Storm (34-63 kt)': '#4575b4',          # Deep Ocean Blue
    'Category 1 Hurricane (64-82 kt)': '#fecc5c',    # Soft Yellow
    'Category 2 Hurricane (83-95 kt)': '#f98d11',    # Orange
    'Category 3 Hurricane (96-112 kt)': '#e31a1c',   # Bright Red
    'Category 4 Hurricane (113-129 kt)': '#b10026',  # Deep Crimson
    'Category 5 Hurricane (≥130 kt)': '#800026'      # Dark Maroon/Purple-Red
}

gdf['intensity_category'] = gdf['max_wind_knots'].apply(assign_category)
hue_order = [
    'Tropical Depression (<34 kt)', 'Tropical Storm (34-63 kt)',
    'Category 1 Hurricane (64-82 kt)', 'Category 2 Hurricane (83-95 kt)',
    'Category 3 Hurricane (96-112 kt)', 'Category 4 Hurricane (113-129 kt)',
    'Category 5 Hurricane (≥130 kt)'
]

# --- PLOT 1: Historical Frequency Histogram + Rolling Trend Line ---
plt.figure(figsize=(14, 6))

# Group to get unique storm counts per year (bounded 1851-2025)
storms_per_year = df.groupby('year')['storm_id'].nunique().reset_index()
storms_per_year = storms_per_year[(storms_per_year['year'] >= 1851) & (storms_per_year['year'] <= 2025)]

# Calculate a 10-year rolling average trend line to highlight climate/detection patterns
storms_per_year['rolling_trend'] = storms_per_year['storm_id'].rolling(window=10, min_periods=1, center=True).mean()

# Draw the base annual frequencies
sns.barplot(data=storms_per_year, x='year', y='storm_id', color='teal', alpha=0.5, edgecolor='none', label='Annual Count')

# Overlay the line plot trend
# We use a lineplot with a secondary twin axis or just overlay directly since scales match
x_indices = np.arange(len(storms_per_year))
plt.plot(x_indices, storms_per_year['rolling_trend'], color='crimson', linewidth=2.5, label='10-Year Rolling Average')

plt.title('Atlantic Tropical Cyclone Frequency & Historical Trend (1851 - 2025)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Unique Storm Count', fontsize=12)

# Adjust X-tick positions nicely
plt.gca().set_xticks(x_indices[::15])
plt.gca().set_xticklabels(storms_per_year['year'].iloc[::15], rotation=45)

plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./plots/historical_frequency_trend.png', dpi=300)
plt.close()
print("Saved trend-overlay frequency chart to ./plots/historical_frequency_trend.png")


# --- PLOT 2: Geospatial Map on a Flattened Earth Coordinate System ---


# 3. Load built-in world dataset from geopandas to act as our background map
# The direct, modern recipe replacing the deprecated datasets module
print("Automatically fetching and caching lowres naturalearth boundaries...")
world_path = get_path("naturalearth.land") # or "naturalearth.countries"
world = gpd.read_file(world_path)

fig, ax = plt.subplots(figsize=(15, 9))

# Draw your map base using geoplot
gpt.polyplot(
    world, 
    facecolor='#e8ece9', 
    edgecolor='#b0b7bd', 
    linewidth=0.6, 
    ax=ax
)

# Set the background color of the axes to represent ocean water
ax.set_facecolor('#d9e4f5')

# 5. Overlay the storm tracking points using standard seaborn scatter 
# (Since both use the same underlying matplotlib axes and EPSG:4326 coordinates)
sns.scatterplot(
    data=gdf,
    x='longitude', y='latitude',
    hue='intensity_category', hue_order=hue_order,
    palette=category_colors, alpha=0.25, s=10, edgecolor='none', ax=ax
)

# 6. Constrain the view layout specifically to the North Atlantic Basin
ax.set_xlim(-110.0, -10.0)
ax.set_ylim(5.0, 65.0)

ax.set_title('Historical Atlantic Tropical Cyclone Tracks (1851 - 2025)', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.grid(True, linestyle=':', color='gray', alpha=0.4)
ax.legend(title='Storm Intensity Class', loc='lower left', frameon=True, facecolor='white')

plt.tight_layout()
plt.savefig('./plots/geospatial_real_map.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print("Saved real map tracking layout to ./plots/geospatial_real_map.png")