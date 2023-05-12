# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
import numpy as np
import json
import cv2
import yaml
import asyncio
import time
import functools
import onnxruntime as ort
from pathlib import Path
from itertools import chain
from myutils.config import cfg_mnet
from myutils.prior_box import PriorBox
from myutils.py_cpu_nms import py_cpu_nms
from myutils.np_box_utils import decode, decode_landm



def parse_args():
    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('--weights', default='./model/RetinaFace_int.onnx',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--source', default='./images',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--imgsz', type=int, default=320, help='input image size (integer)')
    parser.add_argument('--ep', type=str, default='cpu',choices = ['cpu','ipu','azure'], help='EP backend selection')
    parser.add_argument('--instance-count', type=int, default=0, help='instance count for streaming')
    parser.add_argument('--save-img', action='store_true', help='save_img')

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
                imgsz = 320, # HW
                confidence_threshold = 0.02,
                top_k = 5000,
                nms_threshold = 0.4,
                keep_top_k = 750,
                vis_thres = 0.6,
                ep = 'cpu',
                save_img = False,
                ):
        self.cfg = cfg_mnet
        self.imgsz = [imgsz, imgsz]
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres
        self.save_img = save_img
        self.instance_count = 1
        self.run_opt = ort.RunOptions()

        if ep == 'ipu' and 'XLNX_VART_FIRMWARE' not in os.environ:
            xclbin_path = Path(__file__).resolve().parents[2] / 'xclbin' / '1x4.xclbin'
            os.environ['XLNX_VART_FIRMWARE'] = str(xclbin_path)

        if ep == 'azure':
            p = Path(__file__).parent / 'azure_config.yaml'
            if not p.is_file():
                raise FileNotFoundError(f'Config file not found for Azure ep: {str(p)}')
            with open(p) as f:
                self.azure_config = yaml.safe_load(f)
                self.instance_count = self.azure_config.get('instance_count', 1)

            self.run_opt.add_run_config_entry('use_azure', '1')
            self.run_opt.add_run_config_entry('azure.auth_key', self.azure_config['auth_key'])

        self.create_session(weights, ep)
        self.async_run = force_async(self.session.run)

    async def run_once(self, callback, results):
        while True:
            p = await self.queue.get()
            result_img = []
            image_path = str(p)
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            h, w, _ = img_raw.shape
            scale_boxes = (w, h, w, h)
            scale_landms = (w, h, w, h, w, h, w, h, w, h)
 
            img = np.float32(cv2.resize(img_raw, self.imgsz))
            img -= (104, 117, 123)
            img = np.expand_dims(img.transpose(2, 0, 1), 0)

            # loc, conf, landms = session.run(img)  # forward pass
            try:
                # loc, conf, landms = self.session.run([o.name for o in self.session.get_outputs()], {self.session.get_inputs()[0].name: img}, run_options = self.run_opt)
                loc, conf, landms = await self.async_run([o.name for o in self.session.get_outputs()], {self.session.get_inputs()[0].name: img}, run_options = self.run_opt)
            except ort.capi.onnxruntime_pybind11_state.Fail as e:
                raise RuntimeError("Session inference failed")
            priorbox = PriorBox(self.cfg, image_size=self.imgsz)
            priors = priorbox.forward(True)
            boxes = decode(loc.squeeze(0), priors, self.cfg['variance'])
            # boxes = boxes * scale / resize
            scores = conf.squeeze(0)[:, 1]
            # landms = decode_landm(landms.data.squeeze(0), priors, cfg['variance'])
            landms = decode_landm(landms.squeeze(0), priors, self.cfg['variance'])

            # ignore low scores
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:self.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:self.keep_top_k, :]
            landms = landms[:self.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # save result
            for b in dets:
                if b[4] < self.vis_thres:
                    continue
                result_img.append({
                    'conf': b[4],
                    'bbox': list(b[:4]),
                    'landmark': list(b[5:]),
                    })
 
                if self.save_img:
                    text = "{:.4f}".format(b[4])
                    b[:4] *= scale_boxes
                    b[5:] *= scale_landms
                    b = list(map(int, b))
                    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] + 12
                    cv2.putText(img_raw, text, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    # landms
                    cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                    cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                    cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            # save image
            if self.save_img:
                name = "test.jpg"
                cv2.imwrite(name, img_raw)
            result = {'image': str(p), 'faces': result_img}
            callback(result)
            results.append(result)
            self.queue.task_done()

    async def _a_run(self, path = './images', callback = lambda x:None, instance_count = 1):
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

if __name__ == '__main__':
    opt = parse_args()
    runner = Runner(weights = opt.weights, imgsz = opt.imgsz, ep = opt.ep, save_img = opt.save_img)
    json_str = runner.run(path = opt.source, instance_count = opt.instance_count)
    print(json_str)
    with open("retinaface.json", "w") as json_file:
      json_file.write(json_str)
