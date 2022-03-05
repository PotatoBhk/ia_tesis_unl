import asyncio
import websockets
import time

async def hello():
    
    async with websockets.connect("ws://localhost:5000/feed") as websocket:
        await websocket.send("Hello world!")
        data = await websocket.recv()
        print(data)

asyncio.run(hello())