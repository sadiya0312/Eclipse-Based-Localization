import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

crater_csv = "extracted_limb_craters.csv"
limb_pixels_npy = "limb_pixels_overlay.npy"
eclipse_image_jpg = "B2_Speedway_Indiana_April_8th_2024.jpg"
eclipse_overlay="eclipse_with_limb_overlay.png"

matched_craters = pd.read_csv(crater_csv)
limb_pixels = np.load(limb_pixels_npy)
img = cv2.imread(eclipse_image_jpg)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
plt.title("Matched Craters (Bailey's Beads) on Eclipse Image", fontsize=15)

plt.scatter(limb_pixels[:, 0], limb_pixels[:, 1], c='lime', s=8, label="Limb Points", alpha=0.6)

N = 20  
for i in range(0, len(limb_pixels), N):
    x, y = limb_pixels[i]
    # Change 'ID' to 'Lat' -work in progress
    crater_label = f"#{matched_craters.iloc[i]['ID']}"  
    plt.text(x + 4, y - 4, crater_label, fontsize=6, color='Blue', rotation=15)

plt.axis("off")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("craters_on_eclipse.png", dpi=300)
plt.show()

