from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import cv2

# Managing source path
root = os.path.dirname(__file__)
raw_path = os.path.join(root, "..", "training/raw_dataset")

# # Getting images
# # Upload raw images
# imgs = []
# valid_images = [".jpg", ".png"]

# for f in os.listdir(raw_path):
#     ext = os.path.splitext(f)[1]
#     if ext.lower() not in valid_images:
#         continue
#     imgs.append(cv2.imread(os.path.join(raw_path,f)))

a = cv2.imread(os.path.join(raw_path, "1.png"))
b = cv2.imread(os.path.join(raw_path, "7.png"))

# Resize images
a_res = cv2.resize(a, (1024, 768), interpolation = cv2.INTER_CUBIC)
b_res = cv2.resize(b, (1024, 768), interpolation = cv2.INTER_CUBIC)

# convert the images to grayscale
a_gray = cv2.cvtColor(a_res, cv2.COLOR_BGR2GRAY)
b_gray = cv2.cvtColor(b_res, cv2.COLOR_BGR2GRAY)

err = np.sum((a_res.astype("float") - b_res.astype("float")) ** 2)
err /= float(a_res.shape[0] * b_res.shape[1])

s = ssim(a_gray, b_gray)

print(err, s)