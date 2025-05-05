import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
img = cv2.imread("eclipse_with_limb_overlay.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

limb_pixels = np.load("limb_pixels_overlay.npy")  

craters_df = pd.read_csv("extracted_limb_craters.csv")

if len(limb_pixels) != len(craters_df):
    print(f"❗ Mismatch: {len(limb_pixels)} limb pixels vs {len(craters_df)} craters")
    min_len = min(len(limb_pixels), len(craters_df))
    limb_pixels = limb_pixels[:min_len]
    craters_df = craters_df.iloc[:min_len].reset_index(drop=True)
craters_df.to_csv("craters_overlay_matched.csv", index=False)
plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
plt.title("Crater Annotations from Skyfield-Proper Limb Overlay", fontsize=15)
N = 20
for i in range(0, len(limb_pixels), N):
    x, y = limb_pixels[i]
    crater_id = craters_df.iloc[i]["ID"]
    plt.text(x + 4, y - 4, str(crater_id), fontsize=6, color='yellow', rotation=15)

plt.axis("off")
plt.tight_layout()
plt.savefig("craters_on_skyfield_limb.png", dpi=300)
plt.show()


