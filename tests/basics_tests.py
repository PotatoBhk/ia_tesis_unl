from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import cv2

# Managing source path
root = os.path.dirname(__file__)
raw_path = os.path.join(root, "..", "training/raw_dataset")

a = cv2.imread(os.path.join(raw_path, "1.png"))

print(a.shape)

cv2.rectangle(a, (285, 175), (285 + 79, 175 + 264),
                        color=(0, 255, 0), thickness=2)

cv2.imshow("frame", a) 

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break