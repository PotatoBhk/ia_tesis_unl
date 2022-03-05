import asyncio
from imutils.video import VideoStream
import cv2
class VideoStreaming():
    #Save on encrypted database
    user = "admin"
    password = "@ezakmi1105"
    ip = "192.168.1.35"
    
    # Source address
    rtsp_url = "rtsp://{user}:{passw}@{ip}:554/h264/ch{ch}/main/av_stream"

    def __init__(self):
        self.video_source = self.rtsp_url.format(
            user = self.user, 
            passw = self.password, 
            ip = self.ip, 
            ch = 1
        )

    async def frames(self):
        source = VideoStream(self.video_source).start()
        while True:
            img = source.read()
            yield cv2.imencode('.jpg', img)[1].tobytes()
            await asyncio.sleep(1/120)

    async def stream(self, rsp):
        async for frame in self.frames():
            await rsp.write(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )