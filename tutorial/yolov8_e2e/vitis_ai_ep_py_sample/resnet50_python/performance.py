#
# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

import sys
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent
sys.path.append(str(CURRENT_DIR))
import time
import multiprocessing
import os
from multiprocessing import Process
from multiprocessing import shared_memory
import demo.input
import demo.onnx
import demo.utils

image_file_path = CURRENT_DIR / "images" / "orange_0.jpg"
onnx_model_path = CURRENT_DIR.parent.parent / "models" / "resnet50_pt.onnx"
class_file_path = CURRENT_DIR / "models" / "ResNet" / "words.txt"
config_file_path = CURRENT_DIR.parent.parent / "vaip_config.json"

N = 0


def func():
    onnx_session = demo.onnx.OnnxSession(onnx_model_path, config_file_path)
    model_shape = onnx_session.input_shape()
    input_data = demo.input.InputData(image_file_path,
                                      model_shape).preprocess()
    global N
    while True:
        raw_result = onnx_session.run(input_data)
        N += 1


if __name__ == "__main__":

    # tasks count
    proc_count = os.cpu_count()
    print("instance count : ", os.cpu_count())
    # test seconds
    duration = 10

    shm_a = shared_memory.SharedMemory(create=True, size=proc_count)
    processes = []
    for i in range(proc_count):
        p = Process(target=func)
        p.start()
        processes.append(p)

    time.sleep(duration)

    print("qps : ", N / duration, "/s")
