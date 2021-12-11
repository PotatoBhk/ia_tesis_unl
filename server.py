from flask import Flask, render_template, Response
from ssd import SSD
from imutils.video import VideoStream
import cv2

app = Flask(__name__)

user = "admin"
password = "@ezakmi1105"
ip = "192.168.1.35"

@app.route('/')
def index():
    return render_template('index.html')

def run_model(ch = 1):
    ssd = SSD()
    ssd.init_model()

    rtsp_url = "rtsp://{user}:{passw}@{ip}:554/h264/ch{ch}/main/av_stream"
    rtsp_url = rtsp_url.format(user = user, passw = password, ip = ip, ch = ch)

    vid = VideoStream(rtsp_url).start()
    while (True):
        frame = vid.read()
        detections = ssd.detect(frame)
        outs = ssd.format_output(detections)
        img = ssd.postprocess(frame,outs)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = cv2.imencode('.jpeg', gray)[1].tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + res + b'\r\n')


@app.route('/channel_a')
def channel_a():
    return Response(run_model(1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/channel_b')
def channel_b():
    return Response(run_model(2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)