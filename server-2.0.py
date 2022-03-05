from sanic import Sanic
from sanic import response
from video_streaming import VideoStreaming
import asyncio

app = Sanic(__name__)

app.config.WEBSOCKET_MAX_SIZE = 2 ** 20
app.config.WEBSOCKET_MAX_QUEUE = 32
app.config.WEBSOCKET_READ_LIMIT = 2 ** 16
app.config.WEBSOCKET_WRITE_LIMIT = 2 ** 16
app.config.WEBSOCKET_PING_INTERVAL = 20
app.config.WEBSOCKET_PING_TIMEOUT = 20

@app.route('/')
async def index(request):
    return response.html('''<img src="/camera-stream/">''')


@app.route('/camera-stream/')
async def camera_stream(request):
    video_stream = VideoStreaming()
    return response.stream(
        video_stream.stream,
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
    

@app.websocket("/feed")
async def feed(request, ws):
    while True:
        data = "hello!"
        print("Sending: " + data)
        await asyncio.wait({asyncio.create_task(ws.send(data))})
        recp = asyncio.create_task(ws.recv())
        data = await recp
        print("Received: " + data)
        
@app.websocket("/stream")
async def feed(request, ws):
    while True:
        data = "hello!"
        print("Sending: " + data)
        await asyncio.wait({asyncio.create_task(ws.send(data))})
        recp = asyncio.create_task(ws.recv())
        data = await recp
        print("Received: " + data)

if __name__ == '__main__':
    app.run(port=5000)