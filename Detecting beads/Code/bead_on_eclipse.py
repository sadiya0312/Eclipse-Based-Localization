import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread("Detected_lunar_disk_output.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

bead_xy = np.load("baileys_beads_xy.npy")

plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.scatter(bead_xy[:, 0], bead_xy[:, 1], c='red', s=40, label='Detected Beads')
plt.legend()
plt.title("Detected Baily's Beads Overlay")
plt.axis('off')
plt.tight_layout()
plt.savefig("overlay_beads_on_image.png", dpi=300)
plt.show()

