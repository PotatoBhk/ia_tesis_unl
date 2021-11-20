from flask import Flask, render_template, Response
import time
import cv2
import numpy as np
import urllib

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


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


def run_model():
    vid = cv2.VideoCapture(1)
    while (True):
        ret, frame = vid.read()

        outs = detect(frame, './yolo-files/yolov4.weights',
                      './yolo-files/darknet/cfg/yolov4.cfg')

        post_process(outs, frame, './yolo-files/darknet/cfg/coco.names')

        res = cv2.imencode('.jpeg', frame)[1].tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + res + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(run_model(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True)

    # vid = cv2.VideoCapture(1)
    # while (True):
    start_time = time.time()
    # ret, frame = vid.read()
    req = urllib.request.urlopen('http://admin:%40ezakmi1105@192.168.1.35:80/ISAPI/Streaming/channels/100/picture')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    frame = cv2.imdecode(arr, -1)  # 'Load it as it is'
    outs = detect(frame, './yolo-files/yolov4-tiny.weights',
                         './yolo-files/darknet/cfg/yolov4-tiny.cfg')
    post_process(outs, frame, './yolo-files/darknet/cfg/coco.names')
    scale_percent = 70  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)
    print("--- %s seconds ---" % (time.time() - start_time))

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # vid.release()
    # cv2.destroyAllWindows()
    #------------------------------------------------------
    # start_time = time.time()
    # frame = cv2.imread('./result/test.jpg')
    #
    # outs = detect(frame, './yolo-files/yolov4.weights',
    #               './yolo-files/darknet/cfg/yolov4.cfg')
    #
    # post_process(outs, frame, './yolo-files/darknet/cfg/coco.names')
    #
    # scale_percent = 70  # percent of original size
    # width = int(frame.shape[1] * scale_percent / 100)
    # height = int(frame.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # cv2.imshow('frame', resized)
    #
    # k = cv2.waitKey(0)
    # if k == 27:  # wait for ESC key to exit
    #     cv2.destroyAllWindows()
    # -----------------------------------------------
    # vid = cv2.VideoCapture(1)
    # while (True):
    #     ret, frame = vid.read()
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # vid.release()
    # cv2.destroyAllWindows()
