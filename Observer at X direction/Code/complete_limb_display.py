import numpy as np
import matplotlib.pyplot as plt


LIMB_POINTS = None 

def compute_limb_points(img_file):
    """
    Reads a global SLDEM2015 float image file spanning +60 to -60 lat
    and 0 to 360 lon at 128 px/degree, identifies limb points, and stores them
    in a global array LIMB_POINTS as (x, y, z).
    """
    global LIMB_POINTS
    
    nLines   = 15360
    nSamples = 46080
    
    maxLat = 60.0
    minLat = -60.0
    minLon = 0.0
    maxLon = 360.0
    offset_km = 1737.4  #Km
    

    print(f"Reading .img data from: {img_file}")
    dem_km = np.fromfile(img_file, dtype='<f4') 
    expected_size = nLines * nSamples
    if dem_km.size != expected_size:
        raise ValueError(f"Expected {expected_size} pixels, got {dem_km.size}")
    
    dem_km = dem_km.reshape((nSamples, nLines)).T
    print(f"DEM shape after transpose: {dem_km.shape}")
    
    print(f"DEM (km) range: {dem_km.min():.3f} to {dem_km.max():.3f}")
    
    lat_vec = np.linspace(maxLat, minLat, nLines)
    lon_vec = np.linspace(minLon, maxLon, nSamples)
    LonGrid, LatGrid = np.meshgrid(lon_vec, lat_vec)
    
    dem_m = dem_km * 1000.0
    offset_m = offset_km * 1000.0  # 1,737,400 m
    radius = offset_m + dem_m

    lat_rad = np.deg2rad(LatGrid)
    lon_rad = np.deg2rad(LonGrid)
    
    # Planetocentric coordinates
    x_moon = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y_moon = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z_moon = radius * np.sin(lat_rad)
    
    # approximate limb from +X observer viewpoint
    obs_vec = np.array([1.0, 0.0, 0.0])
    obs_unit = obs_vec / np.linalg.norm(obs_vec)
    
    rMag = np.sqrt(x_moon**2 + y_moon**2 + z_moon**2)
    x_unit = x_moon / rMag
    y_unit = y_moon / rMag
    z_unit = z_moon / rMag
    
    dot_vals = x_unit*obs_unit[0] + y_unit*obs_unit[1] + z_unit*obs_unit[2]
    angle_deg = np.degrees(np.arccos(dot_vals))
 
    lim_mask = (angle_deg > 88) & (angle_deg < 92)
    
    count_limb = np.count_nonzero(lim_mask)
    print("Limb points found: {count_limb}")

    x_limb = x_moon[lim_mask]
    y_limb = y_moon[lim_mask]
    z_limb = z_moon[lim_mask]
    
    LIMB_POINTS = np.column_stack((x_limb, y_limb, z_limb))
    print("LIMB_POINTS stored as shape:", LIMB_POINTS.shape)


def display_limb_points_2d(skip=100):
    """
    Displays the global LIMB_POINTS in a simple 2D scatter (X vs Y).
    Ignores Z. Downsamples to manage large data sets.
    """
    global LIMB_POINTS
    if LIMB_POINTS is None or LIMB_POINTS.shape[0] == 0:
        print("No limb points available")
        return
    
    np.save("limb_points.npy", LIMB_POINTS)
    print("Saved LIMB_POINTS to limb_points.npy")
    
    points = LIMB_POINTS[::skip, :]
    
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, s=1, color='r')
    plt.axis('equal')
    plt.title("2D view of Limb Points (X-Y), downsampled by {skip}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()


def display_limb_points_3d(skip=100):
    """
    Displays the global LIMB_POINTS in a 3D scatter plot.
    Downsamples to avoid excessive plotting overhead.
    """
    from mpl_toolkits.mplot3d import Axes3D  
    
    global LIMB_POINTS
    if LIMB_POINTS is None or LIMB_POINTS.shape[0] == 0:
        print("No limb points available. Did you run compute_limb_points() first?")
        return
    
    points = LIMB_POINTS[::skip, :]
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    z_vals = points[:, 2]
    
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_vals, y_vals, z_vals, s=1, c='r')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"3D Limb Points (skip={skip})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    plt.savefig("limb_points.png", dpi=300)
    plt.show()


if __name__ == "__main__":
   
    img_path = r"sldem2015_128_60s_60n_000_360_float.img"
    
    compute_limb_points(img_path)
    display_limb_points_2d(skip=500) 
    display_limb_points_3d(skip=1000)

