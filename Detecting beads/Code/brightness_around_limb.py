import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Detected_lunar_disk_output.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

center_x, center_y, radius_px = np.load("eclipse_geometry.npy")

ring_width = 3 

# Angles around the circle
angles_deg = np.arange(0, 360, 0.5)
angles_rad = np.deg2rad(angles_deg)

brightness = []
x_coords = []
y_coords = []

for angle in angles_rad:
    values = []
    for offset in range(-ring_width, ring_width + 1):
        r = radius_px + offset
        x = int(center_x + r * np.cos(angle))
        y = int(center_y + r * np.sin(angle))
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            values.append(gray[y, x])
    if values:
        brightness.append(np.mean(values))
        # Save central (x, y) for bead location
        r = radius_px
        x_coords.append(center_x + r * np.cos(angle))
        y_coords.append(center_y + r * np.sin(angle))
    else:
        brightness.append(np.nan)
        x_coords.append(np.nan)
        y_coords.append(np.nan)

plt.figure(figsize=(10, 4))
plt.plot(angles_deg, brightness, color='lime')
plt.xlabel("Azimuthal angle (degrees)")
plt.ylabel("Mean brightness near limb")
plt.title("Unwrapped Brightness Around Lunar Limb")
plt.grid()
plt.tight_layout()
plt.savefig("unwrapped_limb_brightness.png", dpi=300)
plt.show()

np.save("azimuth_angles_deg.npy", angles_deg)
np.save("limb_brightness.npy", brightness)
np.save("bead_x_coords.npy", x_coords)
np.save("bead_y_coords.npy", y_coords)

