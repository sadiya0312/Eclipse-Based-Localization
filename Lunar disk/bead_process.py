import cv2
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


img = cv2.imread("B2_Speedway_Indiana_April_8th_2024.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

circles = cv2.HoughCircles(
    gray_blur,
    method=cv2.HOUGH_GRADIENT, 
    dp=1.2,                     
    minDist=100,                
    param1=50,                 
    param2=30,                  
    minRadius=300,             
    maxRadius=700               
)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    # If multiple circles are found, pick the one with the largest radius,
    # or the one whose center is near the image center, etc.
    largest_circle = max(circles, key=lambda c: c[2])
    center_x, center_y, radius_px = largest_circle
    print("Detected circle (x,y,radius):", (center_x, center_y,radius_px))
    
    #np.save("eclipse_geometry.npy", np.array([int(center_x), int(center_y), int(radius_px)]))

    np.save("eclipse_geometry.npy", np.array([center_x, center_y, radius_px]))  
    
#Threshhold
# _, thresh = cv2.threshold(gray_blur, 30, 255, cv2.THRESH_BINARY)
#contours, hierarchy = cv2.findContours(
#    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#)
# Find the largest contour by area (likely the Moon’s disk)
#largest_contour = max(contours, key=cv2.contourArea)
#(x_cont, y_cont), radius_fit = cv2.minEnclosingCircle(largest_contour)
#center_x, center_y = int(x_cont), int(y_cont)

img_out = img.copy()
cv2.circle(img_out, (int(center_x), int(center_y)), int(radius_px), (0, 255, 0), 4)
cv2.circle(img_out, (int(center_x), int(center_y)), 6, (0, 0, 255), -1)


cv2.imwrite("Detected_lunar_disk_output.png", img_out)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
plt.title("✔ Correctly Marked Lunar Disk")
plt.axis("off")
plt.show()

