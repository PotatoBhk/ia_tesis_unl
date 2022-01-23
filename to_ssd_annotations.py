from tqdm import tqdm
from detect import utils
import xml.etree.cElementTree as ET
import cv2
import os
import numpy as np

#Loading utils
utls = utils.Utils()

# Managing source path
root = os.path.dirname(__file__)
outs = utls.join_path(root, 'training/result/outs')
tests = utls.join_path(root, 'training/result/tests')
annotations_folder = utls.join_path(root, 'training/ssd/Annotations')

# Getting all files from a specific directory
def get_files(dir):
    paths = []
    txts = []   
    valid_images = [".jpg", ".png"]        
    print("Cargando imágenes: ")
    for f in tqdm(os.listdir(dir)):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        paths.append(utls.join_path(dir, f))
        txt = f.replace(ext.lower(), ".txt")
        txts.append(utls.join_path(dir, txt))
    print(len(paths), " imágenes encontradas...")
    return (utls.winsort(paths), utls.winsort(txts))

def convert(directory):    
    #Iterate over the files
    paths, txts = get_files(directory)
    print("Progreso de conversión de anotaciones - Carpeta: ", directory)
    for path, txt in tqdm(zip(paths, txts)):
        #Creating xml's path file
        file = os.path.basename(path)
        xml = file.replace(os.path.splitext(file)[1], ".xml")
        xml = utls.join_path(annotations_folder, xml)
        #Reading data from image
        img = cv2.imread(path)
        #Reading .txt file
        with open(txt, 'r') as f:
            lines = f.read().splitlines()
            
        #Creating xml object
        annotation = ET.Element('annotation')
        folder = ET.SubElement(annotation, 'folder')
        filename = ET.SubElement(annotation, 'filename')
        str_path = ET.SubElement(annotation, 'path')
        source = ET.SubElement(annotation, 'source')
            #<source>
        database = ET.SubElement(source, 'database')
            #</source>
        size = ET.SubElement(annotation, 'size')
            #<size>
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')
            #</size>
        segmented = ET.SubElement(annotation, 'segmented')
        
        #Setting default values
        database.text = "Person"
        segmented.text = "0"    
        depth.text = "3" 
        
        #Generating xml data
        f = os.path.basename(os.path.dirname(path))
        folder.text = f
        filename.text = os.path.basename(path)
        str_path.text = path
        width.text = str(img.shape[1])
        height.text = str(img.shape[0])
        print("Leyendo y generando anotaciones de objeto desde ", os.path.basename(txt))
        for line in tqdm(lines):
            params = line.split()            
            x_center = float(params[1])
            y_center = float(params[2])
            detect_width = float(params[3])
            detect_height = float(params[4])
            #Object tag data       
            object = ET.SubElement(annotation, 'object')
                #<object>
            name = ET.SubElement(object, 'name')
            name.text = "person"
            pose = ET.SubElement(object, 'pose')
            pose.text = "Unspecified"
            truncated = ET.SubElement(object, 'truncated')
            truncated.text = "0"
            difficult = ET.SubElement(object, 'difficult')
            difficult.text = "0"
            bndbox = ET.SubElement(object, 'bndbox')
                    #<bndbox>
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(np.round((x_center - (detect_width/2)) * img.shape[1]).astype(int))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(np.round((y_center - (detect_height/2)) * img.shape[0]).astype(int))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(np.round((x_center + (detect_width/2)) * img.shape[1]).astype(int))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(np.round((y_center + (detect_height/2)) * img.shape[0]).astype(int))
                    #</bndbox>
                #</object> 
        tree = ET.ElementTree(annotation)    
        tree.write(xml)
paths, txts = get_files(outs)

convert(outs)