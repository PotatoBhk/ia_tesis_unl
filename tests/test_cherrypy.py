import cherrypy
import os, os.path
import cv2
from imutils.video import VideoStream

class Root:

    @cherrypy.expose
    def index(self):
        return open('sources/templates/index.html')

    @cherrypy.expose
    def streaming(self):
        cherrypy.response.headers['Content-Type'] = "image/jpeg"
        rtsp_url = "rtsp://admin:@ezakmi1105@192.168.1.35:554/h264/ch1/main/av_stream"
        vid = VideoStream(rtsp_url).start()
        def streamer():
            while True:
                frame = vid.read()
                res = cv2.imencode('.jpeg', frame)[1].tobytes()
                yield res
        return streamer()

    streaming._cp_config = {'response.stream': True}

if __name__ == '__main__':
    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './sources/templates/'
        }
    }
    cherrypy.quickstart(Root())

