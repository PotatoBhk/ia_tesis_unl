from imutils.video import VideoStream
from detect import yolov4
from tqdm import tqdm
from detect import utils
import numpy as np
import os
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
        "--stream",
    default = False,
    help="Model definition file."
)

args = parser.parse_args()

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
result_path = os.path.join(root, "training/raw_dataset")

if(args.stream):
    user = "admin"
    password = "@ezakmi1105"
    ip = "192.168.1.35"
    ch = 1

    rtsp_url = "rtsp://{user}:{passw}@{ip}:554/h264/ch{ch}/main/av_stream"
    rtsp_url = rtsp_url.format(user = user, passw = password, ip = ip, ch = ch)

    vid = VideoStream(rtsp_url).start()
    aux = 0 
    while(True):
        frame = vid.read()
        outs = yolo.detect(frame)
        index = np.argwhere(outs[0] == 0)
        if(len(index) > 0):
            # index = index[:,0]
            # a = outs[0][index]
            # b = outs[1][index]
            # c = outs[2][index]
            # outs = (a, b, c)
            # yolo.post_process(outs, frame)
            aux += 1
            fig_name = os.path.join(result_path, "stream-" + str(aux) + ".png")
            cv2.imwrite(fig_name, frame)
else:
    path = os.path.join(root, "training/vids")    
    valid_vids = [".mpg", ".mp4"]  
    aux = 1  
    for f in tqdm(utls.winsort(os.listdir(path))):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_vids:
            continue
        vid = cv2.VideoCapture(os.path.join(path,f))
        if (vid.isOpened() == False):
            continue
        last = np.asarray([]);     
        while(vid.isOpened()):
            ret, frame = vid.read()
            if(ret == True):
                outs = yolo.detect(frame)
                index = np.argwhere(outs[0] == 0)
                if(len(index) > 0):
                    if(utls.equality(last, frame) <= 0.60):
                        last = frame                    
                        aux += 1
                        fig_name = os.path.join(result_path, "vid-1-" + str(aux) + ".png")
                        cv2.imwrite(fig_name, frame)
            else:
                break
        vid.release()
        cv2.destroyAllWindows()
                    
print(args.stream)
