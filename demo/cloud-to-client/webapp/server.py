#!/usr/bin/python

# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

import logging
import asyncio
from functools import partial
import os
import sys
from datetime import datetime
import json
import signal
import threading
import time
import tornado.ioloop
import tornado.web
import tornado.websocket
import uuid
import importlib
import xstream
import multiprocessing
from tornado.options import define, options, parse_command_line

define("port", default=8998, help="run web server on this port", type=int)
define("wsport", default=8999, help="run websocket server on this port", type=int)
define("debug", default=True, help="run in debug mode")

logger = logging.getLogger(__name__)
g_runners = {}
g_runnerQs = {}

# use this to launch: https://stackoverflow.com/a/46710785
# this code from here: https://gist.github.com/nvgoldin/30cea3c04ee0796ebd0489aa62bcf00a
def cancel_async_tasks():
    async def shutdown(loop):
      tasks = [task for task in asyncio.Task.all_tasks() if task is not
              asyncio.tasks.Task.current_task()]
      list(map(lambda task: task.cancel(), tasks))
      results = await asyncio.gather(*tasks, return_exceptions=True)
      loop.stop()

    loop = tornado.ioloop.IOLoop.instance()
    asyncio.ensure_future(shutdown(loop))

class IndexHandler(tornado.web.RequestHandler):
    def get(self, *args):
        self.render("index.html", wsport=options.wsport)

class PageHandler(tornado.web.RequestHandler):
    def get(self, *args):
        url = self.request.uri
        url_arg = url.split('/')[-1]
        html = url_arg + ".html"
        self.render(html, wsport=options.wsport)

def _bg_result_callback(infertype, callback_id, xspub, result):
  # stream live result back to client

  #print(result)
  # convert file path to URL
  fname = result['image']
  url = fname.replace('\\', '/')
  url = url.replace('C:/', '/file/')
  objects = []
  if 'objects' in result:
    objects = result['objects']
  elif 'faces' in result:
    objects = result['faces']

  # unpack labels and bboxes

  print("\n_bg_result_callback %s %s" % (infertype, fname))
  msg = {
    'callback_id': callback_id,
    'topic': 'callback',
    'message': json.dumps({'type': infertype,
      'url': url, 
      'objects': objects})
  }
  #print(msg)
  xspub.send_msg('__server__', json.dumps(msg))

def _objdetect_bg(q):
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  signal.signal(signal.SIGTERM, signal.SIG_IGN)
  xspub = xstream.Publisher()

  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models", "yolov5"))
  import yolov5.yolov5_detect as objdetect
  modelPath = os.path.join(os.path.dirname(__file__), "..", "models\\yolov5\\model\\yolov5s6.onnx")
  runners = { }

  backendPriority = ['azure', 'ipu', 'cpu']
  while True:
    [backend, path, callback_id] = q.get()
    fn = partial(_bg_result_callback, 'objdetect', callback_id, xspub)

    while backend in backendPriority:
      try: 
        if backend not in runners:
          runners[backend] = objdetect.Runner(weights=modelPath, ep=backend)
        runner = runners[backend]
        runner.run(path=path, callback=fn)
        break
      except:
        if backend == 'cpu':
          break # last resort
        # try next backend
        backend = backendPriority[backendPriority.index(backend)+1]

def _facedetect_bg(q):
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  signal.signal(signal.SIGTERM, signal.SIG_IGN)
  xspub = xstream.Publisher()

  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models", "retinaface"))
  import retinaface.retinaface_detect as facedetect
  modelPath = os.path.join(os.path.dirname(__file__), "..", "models\\retinaface\\model\\RetinaFace_int.onnx")
  runners = {}

  backendPriority = ['azure', 'ipu', 'cpu']
  while True:
    [backend, path, callback_id] = q.get()
    fn = partial(_bg_result_callback, 'facedetect', callback_id, xspub)

    while backend in backendPriority:
      try:
        if backend not in runners:
          runners[backend] = facedetect.Runner(weights=modelPath, ep=backend)
        runner = runners[backend]
        runner.run(path=path, callback=fn)
        break
      except:
        if backend == 'cpu':
          break # last resort
        # try next backend
        backend = backendPriority[backendPriority.index(backend)+1]

class InferenceHandler(tornado.web.RequestHandler):
  global g_runners
  global g_runnerQs

  def get(self, *args):
    url = self.request.uri
    url = url.split('?')[0]
    url_arg = url.split('/')[-1]
    if url_arg == "objdetect_bg" or url_arg == "facedetect_bg":
      backend = self.get_argument('backend', 'ipu')
      path = self.get_argument('path', None)
      callback_id = self.get_argument('callback_id', None)

      # spawn background process for batch object detection             
      if url_arg not in g_runners:
        target = _objdetect_bg
        if url_arg == 'facedetect_bg':
          target = _facedetect_bg

        q = multiprocessing.Queue()
        g_runnerQs[url_arg] = q

        g_runners[url_arg] = multiprocessing.Process(\
          target=target, args=(q, ))
        g_runners[url_arg].start()

      # send task to runner
      g_runnerQs[url_arg].put([backend, path, callback_id])

class RequestIdGenerator(object):
    def __init__(self):
        self.handler_ids = {}

    def get(self, name='__default__'):
        if name not in self.handler_ids:
            self.handler_ids[name] = 0

        curr_id = self.handler_ids[name]
        self.handler_ids[name] = (curr_id + 1) % 10000 # wraparound
        return curr_id

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    clientConnections = []
    msgQueue = [] # populated by zmq pipe thread, processed by main ioloop

    def __init__(self, *args, **kwargs):
        super(WebSocketHandler, self).__init__(*args, **kwargs)
        print("[WS] websocket ready")

    def open(self):
        self.id = str(uuid.uuid4())
        self.last_send = None

        print("[WS] websocket opened %s" % self.id)
        self.send('id', self.id)
        WebSocketHandler.clientConnections.append(self)

    def on_message(self, messageStr):
        try:
            print('[WS] message received from %s: %s' % (self.id, messageStr))
            message = json.loads(messageStr)
            if message['topic'] == 'update_id':
                origId = message['id']
                self.id = origId # take over original id
        except:
            pass

    def on_close(self):
        print("[WS] websocket closed %s" % self.id)
        WebSocketHandler.clientConnections.remove(self)

    def send(self, topic, msg):
        if not msg:
            return

        now = time.time()
        #if self.last_send and (now - self.last_send) < 0.05:
        #    # don't flood the client with too many messages; drop
        #    return
        self.last_send = now

        try:
            msg_POD = {}
            msg_POD['time'] = datetime.now().isoformat()
            msg_POD['topic'] = topic
            msg_POD['message'] = msg
            self.write_message(json.dumps(msg_POD))
        except Exception as e:
            print(e)

    @staticmethod
    def send_to_client(id, topic, msg):
        try:
            for c in WebSocketHandler.clientConnections:
                if c.id == id:
                    c.send(topic, msg)
        except:
            pass

    @staticmethod
    def broadcast(topic, msg):
        try:
            for c in WebSocketHandler.clientConnections:
                c.send(topic, msg)
        except:
            pass

    @staticmethod
    def process_msgqueue():
      while len(WebSocketHandler.msgQueue):
          msg = WebSocketHandler.msgQueue.pop()
          if msg['topic'] == 'broadcast':
              WebSocketHandler.broadcast(msg['topic'], msg['message'])
          elif msg['topic'] == 'callback' and 'callback_id' in msg:
              WebSocketHandler.send_to_client(\
                  msg['callback_id'], msg['topic'], msg['message'])

    def check_origin(self, origin):
        return True

class ServerWebApplication(tornado.web.Application):
    def __init__(self):
        self.request_id_gen = RequestIdGenerator()
        handlers = self.init_handlers()

        super(ServerWebApplication, self).__init__(
            handlers,
            cookie_secret="COOKIE_SECRET",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=True,
            autoreload=False,
            debug=options.debug
        )

    def init_handlers(self):
        """
        Define the basic REST handlers. These cannot be destroyed.

        Returns:
            List: List of handler tuples for initializing a Tornado web app
        """
        handlers = []
        handlers.append((r"/", IndexHandler))
        handlers.append((r"/page/([^/]+)", PageHandler))
        handlers.append((r"/infer/([^/]+)", InferenceHandler))
        handlers.append((r"/file/(.*)", tornado.web.StaticFileHandler, 
          { 'path': os.path.splitdrive(os.getcwd())[0] + os.sep }))
        return handlers

class ServerApp(object):
    def __init__(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        self.do_exit = False
        parse_command_line()

        # start zmq server for external processes to talk to this server
        self.xserver = xstream.Server()

        # start web server to serve the browser web app
        self.web_app = ServerWebApplication()
        self.web_server = self.web_app.listen(options.port)

        # start websocket server for this server to talk with the browser
        self.ws_app = tornado.web.Application([(r"/", WebSocketHandler)])
        self.ws_server = self.ws_app.listen(options.wsport)

        # connect to __server__ zmq
        self.xspub = xstream.Publisher()
        self.xssub = xstream.Subscribe("__server__", timeout=1)
        self.xspipe = threading.Thread(target=ServerApp.xspipe)
        self.xspipe.start()

        logger.info("Server online @ port %d" % options.port)

    @staticmethod
    def xspipe():
      # subscribe to special "__server__" channel for
      # external processes to send messages to this server
      asyncio.set_event_loop(asyncio.new_event_loop())
      xs = xstream.Subscribe("__server__")
      while True:
        msg_str = xs.get_msg()
        if msg_str is None:
          break

        try:
          msg = json.loads(msg_str)
          WebSocketHandler.msgQueue.append(msg)
        except Exception as e:
          print(str(e))

    def launch(self):
        tornado.ioloop.PeriodicCallback(self.taskrabbit, 500).start()
        loop = tornado.ioloop.IOLoop.instance()
        loop.start()
        loop.close()

    def signal_handler(self, signum, frame):
        logger.info("Shutting down server...")
        self.do_exit = True

    def taskrabbit(self):
      WebSocketHandler.process_msgqueue()

      if self.do_exit:
        global g_runners
        for r in g_runners:
          g_runners[r].kill()
        self.xspub.end("__server__")
        self.xspipe.join()
        self.ws_server.stop()
        self.web_server.stop()
        cancel_async_tasks()
        del self.xserver
        logger.info("Goodbye")
        sys.exit()

def main():
    app = ServerApp()
    app.launch()

if __name__ == "__main__":
    main()
