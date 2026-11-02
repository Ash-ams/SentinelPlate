import os
import cv2
import numpy as np

images_dir = "dataset/images"
all_files = [f for f in os.listdir(images_dir) if f.startswith("Cars")]

# Let's inspect resolutions and some images
print(f"Total files: {len(all_files)}")

# Check Cars0
im0 = cv2.imread(os.path.join(images_dir, "Cars0.png"))
print(f"Cars0 shape: {im0.shape if im0 is not None else None}")
