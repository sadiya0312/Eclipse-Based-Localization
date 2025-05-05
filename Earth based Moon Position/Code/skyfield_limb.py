import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, Topos
from datetime import datetime
from skyfield.positionlib import Apparent


# Monkey-patch angular_diameter for Apparent objects

def angular_diameter(self):
    diameter_km = getattr(self.target, 'diameter_km', 3474.8)
    distance_km = self.distance().km
    return np.degrees(diameter_km / distance_km)

Apparent.angular_diameter = angular_diameter


# Compute Observer-to-Moon Vector using Skyfield

def get_earth_to_moon_vector(latitude_deg, longitude_deg, elevation_m=0, utc_time="2024-04-08 18:37:00"):
    eph = load('de421.bsp')
    ts = load.timescale()
    dt = datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S")
    t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    observer = Topos(latitude_degrees=latitude_deg, longitude_degrees=longitude_deg, elevation_m=elevation_m)

    earth = eph['earth']
    moon = eph['moon']
    obs = earth + observer

    apparent = obs.at(t).observe(moon).apparent()

    vector_km = apparent.position.km
    unit_vector = vector_km / np.linalg.norm(vector_km)

    moon_angular_diameter_deg = apparent.angular_diameter()

    print("Observer-to-Moon unit vector:", unit_vector)
    print(f"Moon angular diameter: {moon_angular_diameter_deg:.6f} degrees")

    return unit_vector, moon_angular_diameter_deg


# SLDEM Limb Point Extraction

def compute_limb_points(img_file, obs_vec, moon_angular_diameter_deg):
    nLines = 15360
    nSamples = 46080
    maxLat = 60.0
    minLat = -60.0
    minLon = 0.0
    maxLon = 360.0
    offset_km = 1737.4

    print(f"Reading DEM file: {img_file}")
    dem_km = np.fromfile(img_file, dtype='<f4')
    if dem_km.size != nLines * nSamples:
        raise ValueError(f"Unexpected DEM size: {dem_km.size}")
    dem_km = dem_km.reshape((nSamples, nLines)).T

    lat_vec = np.linspace(maxLat, minLat, nLines)
    lon_vec = np.linspace(minLon, maxLon, nSamples)
    LonGrid, LatGrid = np.meshgrid(lon_vec, lat_vec)

    dem_m = dem_km * 1000.0
    offset_m = offset_km * 1000.0
    radius = offset_m + dem_m

    lat_rad = np.deg2rad(LatGrid)
    lon_rad = np.deg2rad(LonGrid)

    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    rMag = np.sqrt(x**2 + y**2 + z**2)
    x_unit, y_unit, z_unit = x/rMag, y/rMag, z/rMag
    dot_vals = x_unit * obs_vec[0] + y_unit * obs_vec[1] + z_unit * obs_vec[2]

    angle_deg = np.degrees(np.arccos(dot_vals))

    limb_angle = moon_angular_diameter_deg / 2
    tolerance_deg = 0.03
    lim_mask = (np.abs(angle_deg - limb_angle) < tolerance_deg)

    print("Limb points found:", np.count_nonzero(lim_mask))
    x_limb, y_limb, z_limb = x[lim_mask], y[lim_mask], z[lim_mask]
    limb_points = np.column_stack((x_limb, y_limb, z_limb))
    np.save("limb_points_skyfield.npy", limb_points)
    print("Saved to limb_points_skyfield.npy")
    
   
    lat_limb = LatGrid[lim_mask]
    lon_limb = LonGrid[lim_mask]
    limb_latlon = np.column_stack((lat_limb, lon_limb))
    np.save("limb_latlon_skyfield.npy", limb_latlon)
    print("Saved limb_latlon_skyfield.npy")

    return limb_points
# Lat/lon defined as per the photograph
if __name__ == "__main__":
    latitude = 39.7936
    longitude = -86.2472
    elevation = 240
    utc_time = "2024-04-08 18:37:00"

    obs_vec, moon_angular_diameter_deg = get_earth_to_moon_vector(latitude, longitude, elevation, utc_time)

    sldem_file = "sldem2015_128_60s_60n_000_360_float.img"
    compute_limb_points(sldem_file, obs_vec, moon_angular_diameter_deg)

