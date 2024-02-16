#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tools for converting onnxtxt format to onnx."""

import os
import argparse
import onnx

from google.protobuf import text_format


def run_main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",
                        type=str,
                        default="",
                        help="input onnxtxt model file path.")
    parser.add_argument("--output_model",
                        type=str,
                        default="",
                        help="output onnx model file path.")
    FLAGS, uparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.input_model):
        print("Input model file '{}' does not exist!".format(FLAGS.input_model))
        print(
            "Usage: python -m vai_q_onnx.tools.convert_onnxtxt_to_onnx "
            "--input_model INPUT_MODEL_PATH --output_model OUTPUT_MODEL_PATH.")
        exit()

    onnxtxt_str = open(FLAGS.input_model, "rb").read()
    onnx_model = onnx.ModelProto()
    text_format.Parse(onnxtxt_str, onnx_model)
    onnx.save_model(onnx_model, FLAGS.output_model)

    print('Conversion Finished!')
    print('Converted model saved in: {}'.format(FLAGS.output_model))


if __name__ == '__main__':
    run_main()
