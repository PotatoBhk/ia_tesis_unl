import os, os.path
import cherrypy
from cherrypy.process import plugins
from ws4py.server.cherrypyserver import WebSocketPlugin, WebSocketTool
from ws4py.websocket import WebSocket

bus = cherrypy.engine

class MonitorWebSocketHandler(WebSocket):
    def received_message(self, m):
        self.send(m)

class MonitorWebSocketPlugin(WebSocketPlugin):
    def __init__(self, bus):

        WebSocketPlugin.__init__(self, bus)

        plugins.Monitor(bus, self.broadcastChanges, 0.1).subscribe()

    def broadcastChanges(self):
        bus.publish('websocket-broadcast', "hola")

class Root(object):
    @cherrypy.expose
    def index(self):
        return open('sources/templates/websockets.html')

    @cherrypy.expose
    def ws(self):
        # you can access the class instance through
        handler = cherrypy.request.ws_handler

cherrypy.tools.websocket = WebSocketTool()
bus.websockets = MonitorWebSocketPlugin(bus)

if __name__ == '__main__':
    cherrypy.config.update({'server.socket_port': 9000})
    WebSocketPlugin(cherrypy.engine).subscribe()
    cherrypy.tools.websocket = WebSocketTool()

    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        '/ws': {
            'tools.websocket.on': True,
            'tools.websocket.handler_cls': MonitorWebSocketHandler
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './sources/templates/'
        }
    }
    cherrypy.quickstart(Root(),'/', conf)