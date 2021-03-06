from turtle import width
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
raw_path = os.path.join(root, "..", "training/result/outs")
paths = winsort(os.listdir(raw_path))
img = cv2.imread(os.path.join(raw_path, paths[20]))
txt = os.path.join(raw_path, paths[20].replace(".png", ".txt"))

print(img.shape[0])

with open(txt, 'r') as f:
    values = f.read().splitlines()
aux = 1
for value in values:
    splitted = value.split()
    print(splitted)
    x_center = float(splitted[1])
    y_center = float(splitted[2])
    width = float(splitted[3])
    height = float(splitted[4])

    x1 = int((x_center - (width/2)) * img.shape[1])
    y1 = int((y_center - (height/2)) * img.shape[0])
    x2 = int((x_center + (width/2)) * img.shape[1])
    y2 = int((y_center + (height/2)) * img.shape[0])

    print(x1,y1,x2,y2)

    cv2.rectangle(img,(x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.putText(img, str(aux), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color=(0, 255, 0), thickness=3)
    aux += 1



resized = cv2.resize(img, (int(img.shape[1]*0.7),int(img.shape[0]*0.7)), interpolation = cv2.INTER_CUBIC)

cv2.imshow("frame", resized) 

cv2.waitKey(0)
cv2.destroyAllWindows()