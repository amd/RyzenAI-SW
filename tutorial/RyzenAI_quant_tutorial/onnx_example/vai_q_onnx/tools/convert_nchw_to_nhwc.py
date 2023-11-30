#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert the input NCHW model to the NHWC model.

    Example : python -m vai_q_onnx.tools.convert_nchw_to_nhwc --input [INPUT_PATH] --output [OUTPUT_PATH]"

'''

import onnx
import logging
import onnxruntime
from argparse import ArgumentParser

from vai_q_onnx.utils.model_utils import convert_nchw_to_nhwc

logger = logging.getLogger(__name__)


def parse_args():
    usage_str = "python -m vai_q_onnx.tools.convert_nchw_to_nhwc --input [INPUT_PATH] --output [OUTPUT_PATH]"
    parser = ArgumentParser("convert_nchw_to_nhwc", usage=usage_str)
    parser.add_argument("input", type=str, help="input onnx model path")
    parser.add_argument("output", type=str, help="output onnx model path")
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    try:
        ort_session = onnxruntime.InferenceSession(
            args.input, providers=['CPUExecutionProvider'])
    except Exception as e:
        logger.error(
            f"Invalid input model got, please check the input model. Error: \n{e}"
        )
    input_model = onnx.load(args.input)
    output_model = convert_nchw_to_nhwc(input_model)
    onnx.save_model(output_model, args.output)
    print(
        f"Converted the NCHW model {args.input} to the NHWC model {args.output}."
    )
