# import the necessary packages
from ctypes import wintypes, windll
from functools import cmp_to_key
import cv2
import os

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

# Managing source path
root = os.path.dirname(__file__)
raw_path = os.path.join(root, "..", "training/raw_dataset")

paths = winsort(os.listdir(raw_path))

a = cv2.imread(os.path.join(raw_path, paths[2]))
b = cv2.imread(os.path.join(raw_path, paths[3]))
path_a = os.path.join(raw_path,paths[2])
path_b = os.path.join(raw_path,paths[3])

print(path_a)
print(path_b)

a_color = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
b_color = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

hist_a = cv2.calcHist([a_color], [0, 1, 2], None, [8, 8, 8],
	[0, 256, 0, 256, 0, 256])
hist_b = cv2.calcHist([b_color], [0, 1, 2], None, [8, 8, 8],
	[0, 256, 0, 256, 0, 256])

hista = cv2.normalize(hist_a, hist_a).flatten()
histb = cv2.normalize(hist_b, hist_b).flatten()

res = cv2.compareHist(hista, histb,  cv2.HISTCMP_CORREL)

print(res)