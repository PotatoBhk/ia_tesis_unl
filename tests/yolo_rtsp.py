import time
import cv2
import imutils
from imutils.video import VideoStream

def detect(frm, model="", config="", conf_threshold=0.6, nms_threshold=0.4):
    net = cv2.dnn.readNet(model, config)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    return model.detect(frm, confThreshold=conf_threshold, nmsThreshold=nms_threshold)


def post_process(outs, frm, names=""):
    with open(names, 'r') as f:
        classes = f.read().splitlines()
    for (classId, score, box) in zip(outs[0], outs[1], outs[2]):
        cv2.rectangle(frm, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0), thickness=2)
        text = '%s: %.2f' % (classes[classId], score)
        cv2.putText(frm, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 255, 0), thickness=3)

if __name__ == '__main__':
    #vid = cv2.VideoCapture("rtsp://admin:@ezakmi1105@192.168.1.35:554/h264/ch1/main/av_stream")
    rtsp_url = "rtsp://admin:@ezakmi1105@192.168.1.35:554/h264/ch1/main/av_stream"
    vid = VideoStream(rtsp_url).start()
    while (True):
        start_time = time.time()
        frame = vid.read()
        if frame is not None:
            outs = detect(frame, './yolo-files/yolov4-tiny.weights',
                                './yolo-files/darknet/cfg/yolov4-tiny.cfg')
            post_process(outs, frame, './yolo-files/darknet/cfg/coco.names')
            cv2.imshow('frame', frame)
            print("--- %s seconds ---" % (time.time() - start_time))
        else:
            print("passed")
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    vid.release()
    cv2.destroyAllWindows()
