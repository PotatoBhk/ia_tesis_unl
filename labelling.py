from detect import yolov4
from tqdm import tqdm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random_alter

# Managing source path
root = os.path.dirname(__file__)
yolo_sources = os.path.join(root, "sources/yolo")

# Getting Yolo instance
yolo = yolov4.Yolo(os.path.realpath(yolo_sources))

# Setting config, weights and names
yolo.set_configFile("pretrained_yolov4.cfg")
yolo.set_weights("pretrained_yolov4.weights")

#Columns for figplot
cols = 10

# Instantiating function for random image alteration
ra = random_alter.ImageAlt()

# Function to load images
def load_imgs(folder):
    # Upload raw images
    imgs = []
    path = os.path.join(root, folder)
    valid_images = [".jpg", ".png"]
    
    print("Cargando imágenes: ")
    for f in tqdm(os.listdir(path)):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(cv2.imread(os.path.join(path,f)))
    print(len(imgs), " imágenes encontradas...")

    return imgs

# Function to generate figures for analysis
def generate_fig(imgs, target, rows):
    #Inicialize variables
    plt.figure()
    f, axs = plt.subplots(rows, cols)
    r, c = (0, 0)
    imgs = np.multiply(imgs, 1/255.0)

    #Add images to figure
    if(rows > 1):
        for img in tqdm(imgs):
            axs[r,c].imshow(img)
            c += 1
            if(c == axs.shape[1]):
                c = 0
                r += 1
        #Delete blank spaces
        while(r < axs.shape[0]):
            if(c < axs.shape[1]):
                plt.delaxes(axs[r,c])
                c += 1
            else:
                c = 0
                r += 1                
    else:
        for img in tqdm(imgs):
            axs[c].imshow(img)
            c += 1
        while(c < len(axs)):
            plt.delaxes(axs[c])
            c += 1
    #Save figure
    path_dataset = os.path.join(root, target)
    f.set_size_inches(35, rows * 2.1)
    plt.savefig(path_dataset, dpi = 250)

# Formattings outs to YoloV4 format
def format_outs(outs, shape, rotated):
    text = ""
    #Filter only Person class
    index = np.argwhere(outs[0] == 0)
    index = index[:,0]
    classesId = outs[0][index]
    boxes = outs[2][index]
    #Iter arrays
    for (classId, box) in zip(classesId, boxes):
        if rotated:
            y_center = ((box[2]/2) + box[0])/shape[1] #TODO (((b - a)/2) + a) / width
            x_center = ((box[3]/2) + box[1])/shape[0]
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

# Save the raw dataset
imgs = load_imgs('training/raw_dataset')
rows = np.ceil(len(imgs) / cols).astype(np.dtype(int))
target = 'training/result/media/fig_rawdataset.png'
print("Generación de figura de dataset sin procesar: ")
generate_fig(imgs, target, rows)

#Detect objects to label in the group of images
results = []
strings = []
yolo.initModel()
print("Generación del formato de etiquetas y alteración aleatoria de imágenes: ")
for img in tqdm(imgs):
    outs = yolo.detect(img)
    (img, rotated) = ra.random_alter(img)
    results.append(img)
    strings.append(format_outs(outs, img.shape, rotated))

# Save the labeled dataset
print("Generación de figura de dataset procesado: ")
target = 'training/result/media/fig_labeled.png'
generate_fig(results, target, rows)

#Save resulting images
print("Guardando los resultados: ")
result_path = os.path.join(root, "training/result/img_labeled")
aux = 1
for (result, string) in tqdm(zip(results, strings)):
    print(string)
    fig_name = os.path.join(result_path, "out-" + str(aux) + ".png")
    cv2.imwrite(fig_name,result)
    with open(os.path.join(result_path, "out-" + str(aux) + ".txt"), 'w') as f:
        f.write(string)
    aux += 1