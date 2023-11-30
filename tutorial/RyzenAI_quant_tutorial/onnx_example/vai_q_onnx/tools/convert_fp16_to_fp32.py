#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert tensor float16 type in the ONNX ModelProto input to tensor float.

:param model: ONNX ModelProto object
:param disable_shape_infer: Type/shape information is needed for conversion to work.
                            Set to True only if the model already has type/shape information for all tensors.
:return: converted ONNX ModelProto object

Examples:

::

    Example 1: Convert ONNX ModelProto object:
    import float16
    new_onnx_model = float16.convert_float16_to_float(onnx_model)

    Example 2: Convert ONNX model binary file:
    import onnx
    import float16
    onnx_model = onnx.load_model('model.onnx')
    new_onnx_model = float16.convert_float16_to_float(onnx_model)
    onnx.save_model(new_onnx_model, 'new_model.onnx')

Use the convert_float16_to_float.py to convert a float16 model to a float32 model:

```
python convert_fp16_to_fp32.py --input $FLOAT_16_ONNX_MODEL_PATH --output $FLOAT_32_ONNX_MODEL_PATH
```

The conversion from float16 models to float32 models may result in 
the generation of unnecessary operations such as casts in the model. 
It is recommended to use onnx-simplifier to remove these redundant nodes.
'''

import onnx
from . import float16
from onnxsim import simplify
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser("float16Converter")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args, _ = parser.parse_known_args()
    return args


def convert(args):

    model = onnx.load(args.input)
    model_fp32 = float16.convert_float16_to_float(model)
    try:
        model_simp, check = simplify(model_fp32)
        assert check, "Simplified ONNX model could not be validated"
    except Exception as e:
        print(f"Fail to Simplify ONNX model because of {e}.")
        model_simp = model_fp32

    onnx.save(model_simp, args.output)
    print(
        f"Convert the float16 model {args.input} to the float32 model {args.output}."
    )


if __name__ == '__main__':
    args = parse_args()
    convert(args)
