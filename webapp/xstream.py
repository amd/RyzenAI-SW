#!/usr/bin/python

# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

import json
import numpy as np
import multiprocessing
import os, subprocess
import signal
import time
import zmq
import traceback
import inspect
import threading
import logging

logger = logging.getLogger(__name__)
g_defaultComPort = int(os.getenv('XSTREAM_PORT', 6661)) # this is pub port; sub port is pub port + 1
g_defaultComHost = '127.0.0.1'

class Proxy(object):
  def __init__(self, in_port=g_defaultComPort, out_port=(g_defaultComPort+1)):
    self.zcontext = zmq.Context()

    # Socket facing producers
    self.zproducers = self.zcontext.socket(zmq.XSUB)
    try:
      self.zproducers.bind("tcp://*:%d" % in_port)
    except zmq.ZMQError as e:
      logger.error(e, exc_info=1)
      exit(0)

    # Socket facing consumers
    self.zconsumers = self.zcontext.socket(zmq.XPUB)
    self.zconsumers.bind("tcp://*:%d" % out_port)

    zmq.proxy(self.zproducers, self.zconsumers)

def _run_proxy(in_port, out_port):
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  signal.signal(signal.SIGTERM, signal.SIG_IGN)
  Proxy(in_port, out_port)

class Server(object):
  def __init__(self, in_port=g_defaultComPort, start_proxy=True):
    self.proxy = None

    if start_proxy:
      self.proxy = multiprocessing.Process(target=_run_proxy, args=(in_port, in_port+1))
      self.proxy.start()

    # start dummy publisher to block until system is ready
    # having this publisher here, results in some kind of deadlock in Py3 for
    # some reason. For now, it's commented out
    # Publisher()

  def __del__(self):
    time.sleep(1) # give time for transmissions to finish

    try:
      if self.proxy:
        self.proxy.kill()
    except Exception as e:
      print(str(e))

class Publisher():
  def __init__(self,
    host=g_defaultComHost,
    port=g_defaultComPort,
    use_context = None,
    transport = 'tcp',
    label = 'thread'
  ):
    if use_context:
      self.zcontext = use_context
    else:
      self.zcontext = zmq.Context()
    self.zpub = self.zcontext.socket(zmq.PUB)
    self.thread_id = threading.current_thread().ident
    self.pid = os.getpid()
    if transport == 'tcp':
      self.zpub.connect("tcp://%s:%d" % (host, port))
    elif transport == 'inproc':
      self.zpub.bind("inproc://%s" % label)

    time.sleep(0.5) # give subscribers time to connect before transmitting

  def send_msg(self, channel, msg):
    if isinstance(msg, str):
      msg = msg.encode()
    self.zpub.send_multipart([channel.encode(), msg])

  def end(self, channel):
    self.send_msg(channel, b'')

class Subscribe():
  def __init__(self,
    channel,
    host=g_defaultComHost,
    port=g_defaultComPort+1,
    timeout=-1,
    use_context=None,
    transport = 'tcp',
    label = 'thread'
  ):
    if use_context:
      self.zcontext = use_context
    else:
      self.zcontext = zmq.Context()
    self.zsub = self.zcontext.socket(zmq.SUB)
    if transport == 'tcp':
      self.zsub.connect("tcp://%s:%d" % (host, port))
    elif transport == 'inproc':
      self.zsub.connect("inproc://%s" % label)
    self.zsub.setsockopt(zmq.SUBSCRIBE, channel.encode())
    self.zsub.setsockopt(zmq.LINGER, 0)
    self.thread_id = threading.current_thread().ident
    self.pid = os.getpid()
    if timeout > 0:
      self.zsub.setsockopt(zmq.RCVTIMEO, timeout)
    self.timeout = timeout

  def get_msg(self):
    while True:
      retval = None
      try:
        retval = self.zsub.recv_multipart()
        (channel, msg) = retval
        break
      except zmq.Again as e:
        if self.timeout > 0:
          return None
      except TimeoutError:
        raise
      except Exception as e:
        logger.info(retval)
        logger.exception('xstream get_msg error')
    if msg == b'':
      return None

    return msg
