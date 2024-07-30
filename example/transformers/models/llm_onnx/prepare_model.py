#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import os
import shutil
import sys

from include.llm_onnx_prepare import ONNXLLMConverter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="model name",
        type=str,
        required=False,
        choices=[
            "facebook/opt-125m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "meta-llama/Llama-2-7b-hf",
            "llama-2-7b",
            "Qwen/Qwen1.5-7B-Chat",
            "THUDM/chatglm3-6b",
            "codellama/CodeLlama-7b-hf",
        ],
    )
    parser.add_argument(
        "--groupsize",
        help="group size for blockwise quantization",
        type=int,
        default=128,
        choices=[32, 64, 128],
    )
    parser.add_argument(
        "--output_model_dir", help="output directory path", type=str, required=True
    )

    parser.add_argument(
        "--input_model",
        help="input model path to optimize/quantize",
        type=str,
        required=False,
    )

    # optimize by onnxruntime only
    parser.add_argument(
        "--only_onnxruntime",
        required=False,
        default=False,
        action="store_true",
        help="optimized by onnxruntime only, and no graph fusion in Python",
    )

    # Optimization Level
    parser.add_argument(
        "--opt_level",
        required=False,
        default=0,
        type=int,
        choices=[0, 1, 2, 99],
        help="onnxruntime optimization level. 0 will disable onnxruntime graph optimization. Level 2 and 99 are intended for --only_onnxruntime.",
    )

    parser.add_argument(
        "--export", help="export float model", action="store_true", default=False
    )

    parser.add_argument(
        "--optimize", help="optimize exported model", action="store_true", default=False
    )

    parser.add_argument(
        "--quantize", help="quantize float model", action="store_true", default=False
    )

    args = parser.parse_args()
    print(f"{args}")

    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)

    if not args.export and not args.optimize and not args.quantize:
        print(f"Please select a task to execute")
        sys.exit(1)

    if (args.export or args.optimize) and not args.model_name:
        print(f"Please provide a valid model name to export/optimize")
        sys.exit(1)

    if not args.export and not args.input_model:
        print(f"Please provide a valid model path to optimize/quantize")
        sys.exit(1)

    if args.export or args.optimize or args.quantize:
        # Instantiate the converter
        onnx_converter = ONNXLLMConverter(args.model_name)

    # Convert the model to ONNX
    if args.export:
        export_dir = onnx_converter.export(args.output_model_dir)

    # Optimize the exported model
    if args.optimize and not args.export:
        onnx_converter.optimize(
            args.input_model,
            args.output_model_dir,
            args.opt_level,
            args.only_onnxruntime,
        )

    if args.optimize and args.export:
        onnx_converter.optimize(
            "", args.output_model_dir, args.opt_level, args.only_onnxruntime
        )

    # Perform quantization
    if args.quantize:
        if not args.optimize and not args.export:
            onnx_converter.quantize(
                args.input_model, args.output_model_dir, args.groupsize
            )
        elif args.export and not args.optimize:
            onnx_converter.quantize("", args.output_model_dir, args.groupsize, False)
        else:
            onnx_converter.quantize("", args.output_model_dir, args.groupsize)


if __name__ == "__main__":
    main()
