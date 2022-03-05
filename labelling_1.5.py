from detect import yolov4
from tqdm import tqdm
from detect import utils
from random import random, seed
import os
import cv2
import numpy as np
import random_alter

#Loading utils
utls = utils.Utils()

# Managing source path
root = os.path.dirname(__file__)
yolo_sources = utls.join_path(root, "sources/yolo")

# Getting Yolo instance
yolo = yolov4.Yolo(yolo_sources)

# Setting config, weights and names
yolo.set_configFile("pretrained_yolov4.cfg")
yolo.set_weights("pretrained_yolov4.weights")
yolo.initModel()

# Instantiating function for random image alteration
ra = random_alter.ImageAlt()

def get_files(dir):
    paths = []
    valid_images = [".jpg", ".png"]        
    print("Cargando imágenes: ")
    for f in tqdm(os.listdir(dir)):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        paths.append(utls.join_path(dir, f))
    print(len(paths), " imágenes encontradas...")
    return utls.winsort(paths)

# Formattings outs to YoloV4 format
def format_outs(outs, shape, modified):
    text = ""
    #Filter only Person class
    index = np.argwhere(outs[0] == 0)
    index = index[:,0]
    classesId = outs[0][index]
    boxes = outs[2][index]
    #Iter arrays
    for (classId, box) in zip(classesId, boxes):
        if modified == 2:
            y_center = ((box[2]/2) + box[0])/shape[1]
            x_center = (shape[0] - box[1] - (box[3]/2))/shape[0]
            height = box[2]/shape[1]
            width = box[3]/shape[0]
        else:
            x_center = ((box[2]/2) + box[0])/shape[1]
            y_center = ((box[3]/2) + box[1])/shape[0]
            width = box[2]/shape[1]
            height = box[3]/shape[0]
        text += str(classId) + " "
        text += str(x_center) + " "
        text += str(y_center) + " "
        text += str(width) + " "
        text += str(height) + "\n"
    return text

#Setting output folders
result_path = utls.join_path(root, "training/result/outs")
test_path = utls.join_path(root, "training/result/tests")
verification_path = utls.join_path(root, "training/result/img_labeled")

#Getting paths
paths_img = get_files(utls.join_path(root, 'training/raw_dataset'))

rotaded = "Imágenes rotadas: \n"
fix_ilum = "Imágenes con la iluminación alterada: \n"
prefix = "out-a-"
aux = 1
seed(1)
#Iterating over the list of images
for path_img in tqdm(paths_img):
    img = cv2.imread(path_img)
    copy = img.copy()
    outs = yolo.detect(img)
    (img, modified) = ra.random_alter(img)
    annotations = format_outs(outs, copy.shape, modified)
    if modified == 1:
        fix_ilum += prefix + str(aux) + "\n"
    elif modified == 2:
        rotaded += prefix + str(aux) + "\n"    
    yolo.post_process(outs, copy)
    
    #Random number to generate test output
    rnd = random()
    
    if rnd <= 0.1:
        root = test_path
    else:
        root = result_path
        
    fig_name = utls.join_path(root, prefix + str(aux) + ".png")
    fv_name = utls.join_path(verification_path, prefix + str(aux) + ".png")
    cv2.imwrite(fig_name, img)
    cv2.imwrite(fv_name, copy)
    with open(utls.join_path(root, prefix + str(aux) + ".txt"), 'w') as f:
        f.write(annotations)
    aux += 1
    
with open(utls.join_path(result_path, "imgs_modified.txt"), 'w') as f:
    f.write(rotaded + fix_ilum)