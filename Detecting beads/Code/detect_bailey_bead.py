import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

angles_deg = np.load("azimuth_angles_deg.npy")
brightness = np.load("limb_brightness.npy")
x_coords = np.load("bead_x_coords.npy")
y_coords = np.load("bead_y_coords.npy")

# Detect peaks in brightness (candidate beads)
# Adjust `prominence` to control sensitivity
peaks, properties = find_peaks(brightness, prominence=2.0)

bead_angles = angles_deg[peaks]
bead_brightness = brightness[peaks]
bead_xs = x_coords[peaks]
bead_ys = y_coords[peaks]

print(f"Detected {len(peaks)} potential Baily’s Beads:")
for i, angle in enumerate(bead_angles):
    print(f"  Bead {i+1:2d}: Azimuth = {angle:.2f}°, Brightness = {bead_brightness[i]:.2f}, Pixel = ({bead_xs[i]:.1f}, {bead_ys[i]:.1f})")

plt.figure(figsize=(12, 5))
plt.plot(angles_deg, brightness, label="Limb Brightness")
plt.plot(bead_angles, bead_brightness, "ro", label="Detected Baily’s Beads")
plt.xlabel("Azimuthal Angle (degrees)")
plt.ylabel("Mean Brightness")
plt.title("Detected Baily’s Beads Along the Lunar Limb")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("baileys_beads_detected_plot.png", dpi=300)
plt.show()

np.save("baileys_beads_angles.npy", bead_angles)
np.save("baileys_beads_xy.npy", np.column_stack((bead_xs, bead_ys)))

