from detect import yolov4
from tqdm import tqdm
from detect import utils
import numpy as np
import os
import cv2

# Invoke utils
utls = utils.Utils()

# Managing source path
root = os.path.dirname(__file__)
yolo_sources = os.path.join(root, "sources/yolo")

# Getting Yolo instance
yolo = yolov4.Yolo(os.path.realpath(yolo_sources))

# Setting config, weights and names
yolo.set_configFile("pretrained_yolov4.cfg")
yolo.set_weights("pretrained_yolov4.weights")

yolo.initModel()
result_path = "training/"

vid = cv2.VideoCapture(os.path.join(result_path,"videotest.mp4"))
salida = cv2.VideoWriter('training/yolov4-full.mp4',cv2.VideoWriter_fourcc(*'mp4v'),60.0,(1280,720))

while(vid.isOpened()):
    ret, frame = vid.read()
    if(ret == True):
        time = 0
        outs = yolo.detect(frame)
        time += yolo.get_detection_time()
        yolo.post_process(outs, frame)  
        time += yolo.get_postprocess_time()
        fps = 1 / time  
        cv2.putText(frame, "FPS = {}".format(fps), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow('video', frame)
        salida.write(frame)    
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break