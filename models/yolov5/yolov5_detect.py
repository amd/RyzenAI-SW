# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import json
import cv2
import yaml
import asyncio
import time
import functools
from pathlib import Path
from itertools import chain

import numpy as np
import onnxruntime as ort

from utils import nms

coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='model/yolov5s6.onnx', help='model path(s)')
    parser.add_argument('--source', type=str, default='images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--instance-count', type=int, default=0, help='Instance count for streaming')
    parser.add_argument('--ep', type=str, default ='cpu',choices = ['cpu','ipu','azure'], help='EP backend selection')
    opt = parser.parse_args()
    return opt

def force_async(fn):
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        future = pool.submit(fn, *args, **kwargs)
        return asyncio.wrap_future(future)  # make it awaitable

    return wrapper

class Runner:

    def create_session(self, model, ep):
        ep_map = {
                'cpu': 'CPUExecutionProvider',
                'ipu': 'VitisAIExecutionProvider',
                'azure': 'AzureExecutionProvider',
                }
        self.ep = ep
        providers = [ep_map[ep]]
        sess_opt = ort.SessionOptions()
        provider_options = [{}]
        if ep == 'ipu':
            vaip_config = Path(__file__).parents[1].resolve() / 'vaip_config.json'
            cache_dir = Path(__file__).parent.resolve()
            cache_key = 'cache'
            provider_options = [{
                'config_file': str(vaip_config),
                'cacheDir': str(cache_dir),
                'cacheKey': cache_key
            }]
        if ep == 'azure':
            for k,v in self.azure_config['session_options'].items():
                sess_opt.add_session_config_entry(f'azure.{k}', str(v))
        self.session = ort.InferenceSession(model, sess_options = sess_opt,
                                            providers = providers,
                                            provider_options = provider_options)

    def __init__(self, weights,
            conf_thres=0.25,
            iou_thres=0.45,
            ep ='cpu',
            ):
        self.ep = ep
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.stride, self.names = 64, coco_names
        self.instance_count = 1
        self.run_opt = ort.RunOptions()

        if ep == 'azure':
            p = Path(__file__).parent / 'azure_config.yaml'
            if not p.is_file():
                raise FileNotFoundError(f'Config file not found for Azure ep: {str(p)}')
            with open(p) as f:
                self.azure_config = yaml.safe_load(f)
                self.instance_count = self.azure_config.get('instance_count', 1)
            self.run_opt.add_run_config_entry('use_azure', '1')
            self.run_opt.add_run_config_entry('azure.auth_key', self.azure_config['auth_key'])

        if ep == 'ipu':
            if 'XLNX_VART_FIRMWARE' not in os.environ:
                xclbin_path = Path(__file__).resolve().parents[2] / 'xclbin' / '1x4.xclbin'
                os.environ['XLNX_VART_FIRMWARE'] = str(xclbin_path)

        self.create_session(weights, ep)
        self.imgsz = self.session.get_inputs()[0].shape[2:]
        self.async_run = force_async(self.session.run)


    def preprocess(self, image_path):
        result = np.ones((1, 3, self.imgsz[0], self.imgsz[1]), dtype=np.uint8) * 114
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        scale = min(self.imgsz[0] / img.shape[0], self.imgsz[1] / img.shape[1])
        newsz = [int(scale * img.shape[0]),int(scale * img.shape[1])]
        padh = (self.imgsz[0] - newsz[0]) // 2
        padw = (self.imgsz[0] - newsz[1]) // 2
        img_new = cv2.resize(img, (newsz[1], newsz[0]), interpolation=cv2.INTER_LINEAR).transpose([2,0,1])[::-1]
        result[0,:,padh:padh+newsz[0], padw:padw+newsz[1]] = img_new
        return result, newsz, [padh, padw]

    async def run_once(self, callback, results):
        while True:
            p = await self.queue.get()
            imgpath = str(p)
            img, newsz, pad = self.preprocess(imgpath)
            result_img = []
            img = img.astype('float32') / 255
            # Inference
            try:
                # outputs = self.session.run(None, {self.session.get_inputs()[0].name: img}, run_options = self.run_opt)
                outputs = await self.async_run(None, {self.session.get_inputs()[0].name: img}, run_options = self.run_opt)
            except ort.capi.onnxruntime_pybind11_state.Fail as e:
                raise RuntimeError("Session inference failed")
            pred = np.array(outputs[0:1])
            pred = nms(pred, self.conf_thres, self.iou_thres)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                if len(det):
                    det[:, [0,2]] = ((det[:, [0,2]] - pad[1]) / newsz[1]).clip(0,1)
                    det[:, [1,3]] = ((det[:, [1,3]] - pad[0]) / newsz[0]).clip(0,1)
                    for *xyxy, conf, cls in reversed(det):
                        xyxy_norm = [float(x) for x in xyxy]
                        result_img.append({ 'label': self.names[int(cls)],
                          'conf': float(conf),
                          'bbox': [xyxy_norm[0], xyxy_norm[1], xyxy_norm[2], xyxy_norm[3]] })
            result = {'image': str(p), 'objects': result_img}
            callback(result)
            results.append(result)
            self.queue.task_done()

    async def _a_run(self, path, callback = lambda x:None, instance_count = 1):
        self.queue = asyncio.Queue()
        source_p = Path(path)
        image_types = ('*.jpg', '*.png', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp')
        image_list = chain(*[source_p.glob(t) for t in image_types])
        results = []
        for p in image_list:
            self.queue.put_nowait(p)

        tasks = [asyncio.create_task(self.run_once(callback, results)) for _ in range(instance_count)]
        queue_complete = asyncio.create_task(self.queue.join())
        await asyncio.wait([queue_complete, *tasks], return_when = asyncio.FIRST_COMPLETED)
        for task in tasks:
            task.cancel()
        if not queue_complete.done():
            queue_complete.cancel()
            for task in tasks:
                if task.done():
                    task.result()

        await asyncio.gather(*tasks, return_exceptions=True)

        return json.dumps({'results': results})

    def run(self, path, callback = lambda x:None, instance_count = 0):
        if not instance_count > 0:
            instance_count = self.instance_count
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._a_run(path, callback, instance_count))

if __name__ == "__main__":
    opt = parse_opt()
    runner = Runner(weights = opt.weights, conf_thres = opt.conf_thres, iou_thres = opt.iou_thres, ep = opt.ep)
    json_str = runner.run(path = opt.source, instance_count = opt.instance_count)
    print(json_str)
    with open("yolov5.json", "w") as json_file:
      json_file.write(json_str)
