import numpy as np
import cv2
import time
import os

#Processing working root path
test_folder = os.path.dirname(__file__)
root = os.path.join(test_folder, '..')
root = os.path.realpath(root)

#Labels of network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

#Load test image
image_path = os.path.join(test_folder, "media/test2.png")

#Load the Caffe model 
prototxt = os.path.join(root, "sources/ssd/deploy.prototxt")
weights = os.path.join(root, "sources/ssd/model.caffemodel")

#General params
threshold = 0.5

frame = cv2.imread(image_path)
frame_resized = cv2.resize(frame,(300,300), interpolation = cv2.INTER_CUBIC)

#Load detection model
start_time = time.time()
net = cv2.dnn.readNetFromCaffe(prototxt, weights)
print("Tiempo de carga del algoritmo: %s segundos" % (time.time() - start_time))

start_time = time.time()
blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
#Set to network the input blob 
net.setInput(blob)
#Prediction of network
detections = net.forward()
print("Tiempo de detección: %s segundos" % (time.time() - start_time))

#POSTPROCESS
start_time = time.time()
index = np.argwhere(detections[0][0][:,2]>threshold)
index = index[:,0]

class_ids = detections[0][0][index,1]
scores = detections[0][0][index,2]
x_points = np.vstack((detections[0][0][index,3], detections[0][0][index,5])).T
y_points = np.vstack((detections[0][0][index,4], detections[0][0][index,6])).T
#Resize and scale points
rows = frame_resized.shape[0]
cols = frame_resized.shape[1] 

heightFactor = frame.shape[0]/rows 
widthFactor = frame.shape[1]/cols

x_points = np.multiply(x_points, (cols * widthFactor)).astype(np.dtype(int))
y_points = np.multiply(y_points, (rows * heightFactor)).astype(np.dtype(int))

for(class_id, score, x, y) in zip(
        class_ids, scores, x_points, y_points):
    cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]),
                        (0, 255, 0), thickness=2)
    label = classNames[class_id] + ": " + str(score)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    y_lb = max(y[0], labelSize[1])
    cv2.rectangle(frame, (x[0], y_lb - labelSize[1]),
                            (x[0] + labelSize[0], y_lb + baseLine),
                            (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (x[0], y_lb),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

print("Tiempo del post-proceso: %s segundos" % (time.time() - start_time))

cv2.imwrite(os.path.join(test_folder, "media/out_ssd.png"), frame)