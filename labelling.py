from detect import yolov4
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def load_imgs(folder):
    # Upload raw images
    imgs = []
    path = os.path.join(root, folder)
    valid_images = [".jpg", ".png"]

    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(cv2.imread(os.path.join(path,f)))
    
    return imgs

def generate_fig(imgs, target, rows):
    #Inicialize variables
    plt.figure()
    f, axs = plt.subplots(rows, cols)

    (r, c) = (0, 0)
    imgs = np.multiply(imgs, 1/255.0)
    
    #Add images to figure
    for img in imgs:
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

    #Save figure
    path_dataset = os.path.join(root, target)
    f.set_size_inches(35, rows * 2.1)
    plt.savefig(path_dataset, dpi = 250)

# Save the raw dataset
imgs = load_imgs('training/raw_dataset')
rows = np.ceil(len(imgs) / cols).astype(np.dtype(int))
target = 'training/result/media/fig_rawdataset.png'
generate_fig(imgs, target, rows)

#Detect objects to label in the group of images
results = []
yolo.initModel()
for img in imgs:
    outs = yolo.detect(img)
    yolo.post_process(outs, img)
    results.append(img)

# Save the labeled dataset
target = 'training/result/media/fig_labeled.png'
generate_fig(results, target, rows)

#Save resulting images
result_path = os.path.join(root, "training/result/img_labeled")
aux = 1
for result in results:
    fig_name = os.path.join(result_path, "out-" + str(aux))
    cv2.imwrite(fig_name,result)
    aux += 1
