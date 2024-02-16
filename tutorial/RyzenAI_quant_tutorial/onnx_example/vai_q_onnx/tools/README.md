# Tools
## Convert a float16 model to a float32 model

Since the vai_q_onnx tool only supports float32 models quantization currently, converting a model from float16 to float32 is required when quantizing a float16 model.

Use the convert_fp16_to_fp32 tool to convert a float16 model to a float32 model:

```
python -m pip install onnxsim
python -m vai_q_onnx.tools.convert_fp16_to_fp32 --input $FLOAT_16_ONNX_MODEL_PATH --output $FLOAT_32_ONNX_MODEL_PATH
```

## Convert a NCHW input model to a NHWC model

Given that some models are designed with an input shape of NCHW instead of NHWC, it's recommended to convert an NCHW input model to NHWC before quantizing a float32 model. 

Use the convert_nchw_to_nhwc tool to convert a NCHW model to a NHWC model:

```
python -m vai_q_onnx.tools.convert_nchw_to_nhwc --input $NCHW_ONNX_MODEL_PATH --output $NHWC_ONNX_MODEL_PATH
```

## Quantize ONNX model using random input 

Given some ONNX models without input for quantization, use random input for the onnx model quantization process.

Use the random_quantize tool to  quantize a onnx model

```
python -m vai_q_onnx.tools.random_quantize --input_model $NCHW_ONNX_MODEL_PATH --quant_model $NHWC_ONNX_MODEL_PATH
```


## License

Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
