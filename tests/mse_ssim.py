from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import cv2
from ctypes import wintypes, windll
from functools import cmp_to_key

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

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

paths = winsort(os.listdir(raw_path))
a = cv2.imread(os.path.join(raw_path, paths[0]))
b = cv2.imread(os.path.join(raw_path, paths[0]))
path_a = os.path.join(raw_path,paths[0])
path_b = os.path.join(raw_path,paths[0])

print(path_a)
print(path_b)

# # Resize images
# a_res = cv2.resize(a, (1024, 768), interpolation = cv2.INTER_CUBIC)
# b_res = cv2.resize(b, (1024, 768), interpolation = cv2.INTER_CUBIC)

# convert the images to grayscale
a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

err = np.sum((a.astype("float") - b.astype("float")) ** 2)
err /= float(a.shape[0] * b.shape[1])

s = ssim(a_gray, b_gray)

print(err, s)