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
import argparse

CURRENT_DIR = pathlib.Path(__file__).parent
sys.path.append(str(CURRENT_DIR))

import demo.input
import demo.onnx
import demo.utils
import os

image_file_path = None # CURRENT_DIR / "images" / "camel.jpg"
class_file_path = CURRENT_DIR / "imagenet" / "words.txt"


def main(args):
    parser = argparse.ArgumentParser(
        description="Python script to run classification with ONNX Runtime"
    )
    parser.add_argument(
        "--model", type=str, default=None, required=True, help="Path to model file"
    )
    parser.add_argument(
        "--img", type=str, required=True, help="Input image file path"
    )
    parser.add_argument(
        "--config", type=str, default="vaip_config_merged.json", help="Config file"
    )
    parser.add_argument("--ep", type=str, default="vai", help="Execution Provider")
    parser.add_argument(
        "--iters", type=int, default=1, help="Number of iterations of session run"
    )
    # Parse args
    known_args, unknown_args = parser.parse_known_args(args)

    ## Quant model with unquantized GEMM
    onnx_model_path = known_args.model

    ## Input image path
    image_file_path = known_args.img

    ## Config file
    config_file_path = known_args.config

    ## Log dir
    model_name = onnx_model_path.rsplit("\\")[-1]
    log_dir = "outputs\\"
    # Create log directory
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    if known_args.ep == "vai":
        onnx_session = demo.onnx.OnnxSession(onnx_model_path, str(config_file_path))
        if os.environ["USE_CPU_RUNNER"] == "1":
            log_file = log_dir + model_name + ".vai.cpu"
        else:
            log_file = log_dir + model_name + ".vai.dpu"
    else:
        onnx_session = demo.onnx.OnnxSession(onnx_model_path, None)
        log_file = log_dir + model_name + ".cpu"

    model_shape = onnx_session.input_shape()

    input_data = demo.input.InputData(image_file_path, model_shape).preprocess()

    for i in range(known_args.iters):
        raw_result = onnx_session.run(input_data)
    res_list = demo.utils.softmax(raw_result)
    sort_idx = demo.utils.sort_idx(res_list)

    with open(class_file_path, "rt") as f:
        classes = f.read().rstrip("\n").split("\n")

    print("\n----------------------------------------------------------")
    print("-- Model: {}".format(onnx_model_path))
    print("-- Log file: {}".format(log_file))
    print("----------------------------------------------------------")
    with open(log_file, "w") as lf:
        for k in sort_idx[:5]:
            output = "{} {}".format(classes[k], res_list[k])
            print(output)
            lf.write(output + "\n")
    print("----------------------------------------------------------\n")


if __name__ == "__main__":
    main(sys.argv[1:])
