import numpy as np
import matplotlib.pyplot as plt

def project_limb_points_to_angles(limb_points):
    """
    Converts 3D SLDEM limb points to angular offsets from the Moon center
    as seen from +X observer viewpoint.

    """
    x = limb_points[:, 0]
    y = limb_points[:, 1]
    z = limb_points[:, 2]

    angle_x_rad = np.arctan2(y, x)
    angle_y_rad = np.arctan2(z, x)

    angle_x_deg = np.degrees(angle_x_rad)
    angle_y_deg = np.degrees(angle_y_rad)

    angles_deg = np.column_stack((angle_x_deg, angle_y_deg))
    return angles_deg

def plot_projected_limb(angles_deg, skip=1, filename="projected_limb_angles.png"):
    """
    Saves a 2D scatter plot of angular limb points (in degrees).

    """
    downsampled = angles_deg[::skip]
    x = downsampled[:, 0]
    y = downsampled[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=1, color='blue')
    plt.title("Limb Points in Angular Offsets (Degrees)")
    plt.xlabel("X Angular Offset (deg)")
    plt.ylabel("Y Angular Offset (deg)")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved angular limb plot to: {filename}")

def convert_angles_to_pixels(angles_deg, center_x, center_y, radius_px, moon_diameter_deg=0.533):
    """
    Converts angular limb offsets (deg) to image pixel coordinates.
    """
    deg_per_pixel = moon_diameter_deg / (2 * radius_px)
    px_x = center_x + angles_deg[:, 0] / deg_per_pixel
    px_y = center_y - angles_deg[:, 1] / deg_per_pixel 
    pixel_coords = np.column_stack((px_x, px_y))
    return pixel_coords

if __name__ == "__main__":
    try:
       
        LIMB_POINTS = np.load("limb_points.npy")

    except FileNotFoundError:
        print("❌ limb_points.npy not found. Run compute_limb_points() first.")
        exit()

    print("Projecting limb points to angular coordinates...")
    limb_angles = project_limb_points_to_angles(LIMB_POINTS)
    print("Angular shape:", limb_angles.shape)

    plot_projected_limb(limb_angles, skip=10)


    center_x = 946       
    center_y = 648
    radius_px = 308      
    moon_diameter_deg = 0.533

    limb_pixels = convert_angles_to_pixels(
        angles_deg=limb_angles,
        center_x=center_x,
        center_y=center_y,
        radius_px=radius_px,
        moon_diameter_deg=moon_diameter_deg
    )

    np.save("projected_limb_pixels.npy", limb_pixels)
    print("✅ Saved limb pixel coordinates to: projected_limb_pixels.npy")


