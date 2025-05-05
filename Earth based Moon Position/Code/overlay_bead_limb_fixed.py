import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
from skyfield.api import load, Topos
from skyfield.positionlib import Apparent

def angular_diameter(self):
    diameter_km = getattr(self.target, 'diameter_km', 3474.8)
    distance_km = self.distance().km
    return np.degrees(diameter_km / distance_km)

Apparent.angular_diameter = angular_diameter

def project_limb_points_to_angles(limb_points, obs_vec):
    obs_vec = obs_vec / np.linalg.norm(obs_vec)
    r_vecs = limb_points
    x_angles = np.arcsin(np.dot(r_vecs, np.cross(obs_vec, [0, 1, 0])) / np.linalg.norm(r_vecs, axis=1))
    y_angles = np.arcsin(np.dot(r_vecs, np.cross(obs_vec, [0, 0, 1])) / np.linalg.norm(r_vecs, axis=1))
    angle_x_deg = np.degrees(x_angles)
    angle_y_deg = np.degrees(y_angles)
    return np.column_stack((angle_x_deg, angle_y_deg))

def convert_angles_to_pixels(angles_deg, center_x, center_y, radius_px, moon_diameter_deg):
    deg_per_pixel = moon_diameter_deg / (2 * radius_px)
    px_x = center_x + angles_deg[:, 0] / deg_per_pixel
    px_y = center_y - angles_deg[:, 1] / deg_per_pixel
    return np.column_stack((px_x, px_y))

def overlay_limb_on_image(image_path, limb_pixel_coords, center_x, center_y, radius_px, output_path="eclipse_with_limb_overlay.png"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_rgb)

    moon_circle = patches.Circle(
        (center_x, center_y), radius_px,
        fill=False, edgecolor='red', linewidth=0.5, alpha=0.6
    )
    ax.add_patch(moon_circle)

    valid = np.isfinite(limb_pixel_coords[:, 0]) & np.isfinite(limb_pixel_coords[:, 1])

    xs = limb_pixel_coords[valid, 0]
    ys = limb_pixel_coords[valid, 1]

    limb_pixels_filtered = np.column_stack((xs, ys))
    np.save("limb_pixels_overlay.npy", limb_pixels_filtered)

    ax.scatter(xs, ys, s=8.0, c='lime', edgecolors='blue', linewidths=0.5, alpha=0.8, label='SLDEM Limb', zorder=10)

    margin = radius_px * 1.4
    ax.set_xlim(center_x - margin, center_x + margin)
    ax.set_ylim(center_y + margin, center_y - margin)

    ax.set_title("Overlay: SLDEM Limb on Eclipse Image", fontsize=14)
    ax.axis('off')
    ax.legend(loc='lower left', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        points = np.load("limb_points_skyfield.npy")
        print(" Loaded rotated SLDEM points.")
    except FileNotFoundError:
        points = np.load("limb_points_skyfield.npy")
        print("Rotated points not found, using original SLDEM points.")
    print("points.shape:", points.shape)
    print(points[:5])

    center_x, center_y, radius_px = np.load("eclipse_geometry.npy")

    ts = load.timescale()
    t = ts.utc(2024, 4, 8, 18, 37, 0)
    eph = load('de421.bsp')
    observer = eph['earth'] + Topos(latitude_degrees=39.7936, longitude_degrees=-86.2472, elevation_m=240)
    moon = eph['moon']

    apparent = observer.at(t).observe(moon).apparent()
    moon_diameter_deg = apparent.angular_diameter()
    print(f"Moon angular diameter: {moon_diameter_deg:.6f} degrees")

    # Compute true observer-to-Moon direction
    obs_vec = apparent.position.km / np.linalg.norm(apparent.position.km)

    # Angular filtering instead of radial filtering
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    unit_vectors = points / norms
    cos_angles = unit_vectors @ obs_vec
    angles_rad = np.arccos(cos_angles)
    print("Angular range of all points (deg):", np.degrees(angles_rad).min(), "to", np.degrees(angles_rad).max())
    angles_deg_full = np.degrees(angles_rad)
    half_diameter_deg = moon_diameter_deg / 2
    tolerance_deg = 0.03
    limb_mask = np.abs(angles_deg_full - half_diameter_deg) < tolerance_deg
    limb_points = points[limb_mask]
    print("Filtered limb_points.shape:", limb_points.shape)

    angles_deg = project_limb_points_to_angles(limb_points, obs_vec)
    limb_pixels = convert_angles_to_pixels(angles_deg, center_x, center_y, radius_px, moon_diameter_deg)

    print("limb_pixels.shape:", limb_pixels.shape)

    eclipse_image = "B2_Speedway_Indiana_April_8th_2024.jpg"
    overlay_limb_on_image(eclipse_image, limb_pixels, center_x, center_y, radius_px)

    # After computing limb_pixels
    #limb_pixels_centered = limb_pixels - np.array([center_x, center_y])
    #limb_pixels_scaled = limb_pixels_centered * (619 / 310)  # scale up ~2x
    #limb_pixels = limb_pixels_scaled + np.array([center_x, center_y])

    #np.save("limb_pixels_overlay.npy", limb_pixels)

