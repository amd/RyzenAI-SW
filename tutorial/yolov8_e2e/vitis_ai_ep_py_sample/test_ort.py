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
import onnxruntime
import numpy as np
import sys
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent
config_file_path = CURRENT_DIR.parent / "vaip_config.json"
onnx_model_path = sys.argv[1]

onnx_session = onnxruntime.InferenceSession(
    onnx_model_path,
    providers=["VitisAIExecutionProvider"],
    provider_options=[{
        'config_file': config_file_path
    }])

inputs = onnx_session.get_inputs()

raw_result = onnx_session.run(
    [], {i.name: np.random.random(i.shape).astype('float32')
         for i in inputs})

print('OK')
