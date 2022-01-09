from skimage.metrics import structural_similarity as ssim
from ctypes import wintypes, windll
from functools import cmp_to_key
import numpy as np
import os
import cv2
from tqdm import tqdm

# Managing source path
root = os.path.dirname(__file__)
raw_path = os.path.join(root, "training/raw_dataset")

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

def filter_imgs(imgs, paths):
    # Compare each image with others
    idx = []
    print("Comparando imágenes: ")
    for i in tqdm(range(len(imgs))):
        for j in range((i+1), len(imgs)):
            if(imgs[i].shape == imgs[j].shape):
                # convert the images to grayscale
                a = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
                b = cv2.cvtColor(imgs[j], cv2.COLOR_BGR2GRAY)
                s = ssim(a, b)
                if(s > 0.95):
                    if j not in idx:
                        idx.append(j)
                else:
                    break  
    # Delete equal images
    print("Eliminando imágenes con un porcentaje mayor a 95% de igualdad: ")
    for pos in tqdm(idx):
        os.remove(paths[pos])

# Getting images
# Upload raw images
source = []
source_paths = []
valid_images = [".jpg", ".png"]

print("Buscando y cargando archivos de imagen válido: ...")
for f in tqdm(winsort(os.listdir(raw_path))):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    source.append(cv2.imread(os.path.join(raw_path,f)))
    source_paths.append(os.path.join(raw_path,f))


if len(source) > 2000:
    spltd_imgs = np.asarray(source, dtype=object)
    spltd_imgs = np.array_split(spltd_imgs, 10)
    spltd_paths = np.asarray(source_paths, dtype=object)
    spltd_paths = np.array_split(spltd_paths, 10)
    aux = 1
    for (imgs, paths) in zip(spltd_imgs, spltd_paths):
        print("SERIE ", str(aux), ":")
        filter_imgs(imgs, paths)
        aux += 1
else:
    filter_imgs(source, source_paths)
    
print("Proceso terminado...")