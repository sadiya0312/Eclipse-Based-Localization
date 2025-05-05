import numpy as np
import matplotlib.pyplot as plt
import cv2

xy_coords = np.load("baileys_beads_xy.npy")           
all_brightness = np.load("limb_brightness.npy")        
all_angles = np.load("azimuth_angles_deg.npy")         
bead_x = np.load("bead_x_coords.npy")                  
bead_y = np.load("bead_y_coords.npy")                  

all_xy = np.column_stack((bead_x, bead_y))
def find_nearest_index(target_point, all_points, tolerance=1.0):
    dists = np.linalg.norm(all_points - target_point, axis=1)
    min_idx = np.argmin(dists)
    return min_idx if dists[min_idx] < tolerance else -1

matching_indices = []
for pt in xy_coords:
    idx = find_nearest_index(pt, all_xy)
    if idx != -1:
        matching_indices.append(idx)

matching_indices = np.array(matching_indices)
if len(matching_indices) == 0:
    raise ValueError("No matches found within tolerance.")

matched_xy = all_xy[matching_indices]
matched_brightness = all_brightness[matching_indices]
matched_angles = all_angles[matching_indices]

top10_idx = np.argsort(matched_brightness)[-10:][::-1]
top_xy = matched_xy[top10_idx]
top_brightness = matched_brightness[top10_idx]
top_angles = matched_angles[top10_idx]


img = cv2.imread("Detected_lunar_disk_output.png") 
if img is None:
    raise FileNotFoundError("Eclipse image not found.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.scatter(top_xy[:, 0], top_xy[:, 1], c='red', s=50, label="Top 10 Beads")

for i, (x, y, angle, bright) in enumerate(zip(top_xy[:, 0], top_xy[:, 1], top_angles, top_brightness)):
    plt.text(x + 5, y, f"{i+1}", color='yellow', fontsize=8, weight='bold')
    print(f"Bead {i+1}: Azimuth = {angle:.2f}°, Brightness = {bright:.2f}, Pixel = ({x:.1f}, {y:.1f})")

plt.title("Top 10 Baily’s Beads from Precomputed Peaks")
plt.axis('off')
plt.legend()
plt.tight_layout()
plt.savefig("top_10_baileys_beads_matched_overlay.png", dpi=300)
plt.show()


