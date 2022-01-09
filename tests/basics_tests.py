from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import cv2

# Managing source path
root = os.path.dirname(__file__)
raw_path = os.path.join(root, "..", "training/raw_dataset")

tst_dir = os.listdir(raw_path)

print(len(tst_dir))

tst_dir = np.asarray(tst_dir)
splt = np.array_split(tst_dir, 10)
splt = np.asarray(splt, dtype=object)

print(np.shape(splt))

a = cv2.imread(os.path.join(raw_path, "vid-1.png"))

print(a.shape)

cv2.rectangle(a, (285, 175), (285 + 79, 175 + 264),
                        color=(0, 255, 0), thickness=2)

cv2.imshow("frame", a) 

cv2.waitKey(0)
cv2.destroyAllWindows()