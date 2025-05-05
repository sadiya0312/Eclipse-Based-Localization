import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from math import radians, cos, sin

craters_df = pd.read_csv("cleaned_crater_catalog.csv")

latitudes = np.radians(craters_df['Latitude(degree)'].values) 
longitudes = np.radians(craters_df['Longitude(degree)'].values) 
crater_xyz = np.column_stack((
    np.cos(latitudes) * np.cos(longitudes),
    np.cos(latitudes) * np.sin(longitudes),
    np.sin(latitudes)
))

# Build KD-Tree
tree = KDTree(crater_xyz)

limb_latlon = np.load("limb_latlon_skyfield.npy")
limb_lat_rad = np.radians(limb_latlon[:, 0])
limb_lon_rad = np.radians(limb_latlon[:, 1])

limb_xyz = np.column_stack((
    np.cos(limb_lat_rad) * np.cos(limb_lon_rad),
    np.cos(limb_lat_rad) * np.sin(limb_lon_rad),
    np.sin(limb_lat_rad)
))

# Query nearest crater for each limb point
distances, indices = tree.query(limb_xyz)

matched_craters = craters_df.iloc[indices].copy()
matched_craters['Distance_on_Sphere'] = distances 

# Work in progress here
moon_radius_km = 1737.4
matched_craters['Distance_km'] = matched_craters['Distance_on_Sphere'] * moon_radius_km

matched_craters.to_csv("extracted_limb_craters.csv", index=False)
print("Matching complete! Results saved in extracted_limb_craters.csv")

