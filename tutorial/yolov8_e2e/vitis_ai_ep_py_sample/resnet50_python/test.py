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

import demo.input
import demo.onnx
import demo.utils
import os

image_file_path = CURRENT_DIR / "images" / "resnet50.jpg"
onnx_model_path = CURRENT_DIR.parent.parent / "models" / "resnet50_pt.onnx"
class_file_path = CURRENT_DIR / "models" / "ResNet" / "words.txt"
config_file_path = CURRENT_DIR.parent.parent / "vaip_config.json"

onnx_session = demo.onnx.OnnxSession(onnx_model_path, str(config_file_path))
model_shape = onnx_session.input_shape()

input_data = demo.input.InputData(image_file_path, model_shape).preprocess()

raw_result = onnx_session.run(input_data)
res_list = demo.utils.softmax(raw_result)
sort_idx = demo.utils.sort_idx(res_list)

with open(class_file_path, "rt") as f:
    classes = f.read().rstrip('\n').split('\n')

print('============ Top 5 labels are: ============================')
for k in sort_idx[:5]:
    print(classes[k], res_list[k])
print('===========================================================')
