from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import cv2
from tqdm import tqdm
from random import seed
from random import random

# Managing source path
root = os.path.dirname(__file__)
raw_path = os.path.join(root, "..", "training/raw_dataset")

# ---------------
a = cv2.imread(os.path.join(raw_path, "1.png"))
print(a.shape)
print("-----------------")
for i in range(10):
    print(i)

# -----------------
print("-------------------")
b = [1, 3, 7, 9, 11, 13, 15, 17, 19, 20]
c = [0, 3, 5]
print(b[1:3])
#------------------
print("-------------------")
print("Buscando y cargando archivos de imagen válido: ...")
d = os.listdir(raw_path)
for path in tqdm(d):
    print(path)

#---------------------------
# print("-------------------")
# with open(os.path.join(raw_path, "readme.txt"), 'w') as f:
#     f.write('readme')

#-------------------------
seed(1)
print("Random numbers: ")
print(random())
print(random())
print(random())
print(random())
print(random())
print(random())