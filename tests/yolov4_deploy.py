import time
import cv2
import os
from imutils.video import VideoStream
from numpy import True_
#Save on encrypted database
user = "admin"
password = "@ezakmi1105"
ip = "192.168.1.35"

# Source address
rtsp_url = "rtsp://{user}:{passw}@{ip}:554/h264/ch{ch}/main/av_stream"

#Processing outputs and writing detections
def post_process(outs, frm):
    start_time = time.time()
    for (classId, score, box) in zip(outs[0], outs[1], outs[2]):
        cv2.rectangle(frm, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0), thickness=2)
        text = '%s: %.2f' % (classes[classId], score)
        cv2.putText(frm, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 255, 0), thickness=3)
    print("Tiempo del post-proceso: %s segundos" % (time.time() - start_time))

video_source = rtsp_url.format(
    user = user, 
    passw = password, 
    ip = ip, 
    ch = 1
)
source = VideoStream(video_source).start()
# Managing source path
root = os.path.dirname(__file__)
raw_path = os.path.join(root, "..", "sources/yolo")
weights = os.path.join(raw_path, "custom-yolov4-tiny.weights")
cfg = os.path.join(raw_path, "custom-yolov4-tiny.cfg")
names = os.path.join(raw_path, "coco.names")
img_path = os.path.join(root, "media/2022-03-07_21-25-39-873142_HiLook-CCTV_camera2.png")

#Gettin class names from file
with open(names, 'r') as f:
    classes = f.read().splitlines()

#Load YOLOv4 network
start_time = time.time()
net = cv2.dnn.readNet(weights, cfg)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(704, 704), swapRB=True)
print("Tiempo de carga del algoritmo: %s segundos" % (time.time() - start_time))


# while True:
#     frm = source.read()
#     outs = model.detect(frm, confThreshold=0.6, nmsThreshold=0.4)
#     post_process(outs, frm)

#     cv2.imshow('frame', frm)
#     if cv2.waitKey(1) == ord('q'):
#         break

# #Running detections
frm = cv2.imread(img_path)
start_time = time.time()
outs = model.detect(frm, confThreshold=0.5, nmsThreshold=0.4)
print("Tiempo de detección: %s segundos" % (time.time() - start_time))
print(len(outs[0]))
post_process(outs, frm)
cv2.imwrite(os.path.join(root,"media/out7.png"), frm)



