import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Detected_lunar_disk_output.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
center_x, center_y, radius_px = np.load("eclipse_geometry.npy")
num_angles = 360  
num_radii = 200  
theta = np.linspace(0, 2 * np.pi, num_angles)
r = np.linspace(0, num_radii, num_radii)
theta_grid, r_grid = np.meshgrid(theta, r)

# Convert polar coordinates to Cartesian
X = center_x + r_grid * np.cos(theta_grid)
Y = center_y + r_grid * np.sin(theta_grid)

X_int = np.clip(X.astype(int), 0, gray.shape[1] - 1)
Y_int = np.clip(Y.astype(int), 0, gray.shape[0] - 1)
polar_values = gray[Y_int, X_int]

plt.figure(figsize=(10, 6))
plt.imshow(polar_values.T, extent=[0, num_radii, 0, 360], aspect='auto', cmap='inferno')
plt.xlabel("Radius (pixels from center)")
plt.ylabel("Azimuth (degrees)")
plt.title("Radial-Azimuthal Intensity Map")
plt.colorbar(label="Intensity")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("radial_azimuthal_intensity_map.png", dpi=300)
plt.show()

np.save("polar_values.npy", polar_values)
np.save("theta_grid.npy", theta_grid)  # radians
np.save("r_grid.npy", r_grid)


