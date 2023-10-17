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


class OnnxSession:

    def __init__(self, onnx_model_path, config_file_path):
        self.session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=['VitisAIExecutionProvider'],
            provider_options=[{'config_file': config_file_path}])

    def input_shape(self):
        return self.session.get_inputs()[0].shape

    def run(self, input_data):
        input_name = self.session.get_inputs()[0].name
        raw_result = self.session.run([], {input_name: input_data})
        return raw_result
