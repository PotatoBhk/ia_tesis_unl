from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import cv2
from tqdm import tqdm

# Managing source path
root = os.path.dirname(__file__)
raw_path = os.path.join(root, "training/raw_dataset")

# Getting images
# Upload raw images
imgs = []
paths = []
valid_images = [".jpg", ".png"]

print("Buscando y cargando archivos de imagen válido: ...")
for f in tqdm(os.listdir(raw_path)):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(cv2.imread(os.path.join(raw_path,f)))
    paths.append(f)

# Compare each image with others
idx = []
print("Comparando imágenes: ")
for i in tqdm(range(len(imgs))):
    for j in tqdm(range((i+1), len(imgs))):
        if(imgs[i].shape == imgs[j].shape):
            # convert the images to grayscale
            a = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(imgs[j], cv2.COLOR_BGR2GRAY)
            if(ssim(a, b) > 0.95):
                if j not in idx:
                    idx.append(j)

# Delete equal images
print("Eliminando imágenes con un porcentaje igual o mayor a 95% \de igualdad: ")
for pos in tqdm(idx):
    os.remove(paths[pos])