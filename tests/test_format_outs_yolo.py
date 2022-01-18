from ctypes import wintypes, windll
from functools import cmp_to_key
import cv2
import numpy as np
import os

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
            y_center = (box[0] + (box[2]/2))/shape[1] #TODO (((b - a)/2) + a) / width
            x_center = (box[1] - (box[3]/2))/shape[0]
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

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

# Managing source path
root = os.path.dirname(__file__)
raw_path = os.path.join(root, "..", "training/raw_dataset")
result_path = os.path.join(root, "..", "training/result/outs")

sources_path = os.path.join(root, "..", "sources/yolo")
yolo = os.path.join(sources_path, "pretrained_yolov4.weights")
cfg = os.path.join(sources_path, "pretrained_yolov4.cfg")
names = os.path.join(sources_path, "coco.names")

paths = winsort(os.listdir(raw_path))
a = cv2.imread(os.path.join(raw_path, paths[0]))
b = a.copy()

with open(names, 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNet(yolo, cfg)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
outs = model.detect(a, confThreshold=0.6, nmsThreshold=0.4)

formatted = format_outs(outs, a.shape, 2)


fig_name = os.path.join(result_path, "out-test.png")



