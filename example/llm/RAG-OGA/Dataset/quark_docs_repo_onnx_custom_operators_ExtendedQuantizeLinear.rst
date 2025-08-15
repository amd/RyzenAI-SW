ExtendedQuantizeLinear
======================

ExtendedQuantizeLinear - 1
--------------------------

Version
```````
- **name**: ExtendedQuantizeLinear

- **domain**: com.amd.quark

Summary
```````

This operator extends the official ONNX QuantizeLinear operator by adding early support for uint16 and int16 quantization, along with additional support for bfloat16, float16, int32 and uint32. It enables floating-point quantization to deploy models on edge devices and wide-bit quantization to facilitate detailed analysis of accuracy bottlenecks.

The proposed ExtendedQuantizeLinear operator enhances the official QuantizeLinear by supporting additional data types while maintaining backward compatibility with the official one. If you want to try these new data types, you can consider using this operator, but note that you should register our custom operator library to onnxruntime before running the quantized model, and since onnxruntime will not convert the quantized node to a QOperator anymore, there is no additional acceleration effect at runtime.

The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization granularity. The quantization formula is *y = saturate((x / y_scale) + y_zero_point)*.

Saturation is done according to:

uint32: [0, 4294967295]

int32: [−2147483648, 2147483647]

uint16: [0, 65535]

int16: [-32768, 32767]

uint8: [0, 255]

int8: [-128, 127]

float16: [-65504, 65504]

bfloat16: [-3.4e38, 3.4e38]

For *(x / y_scale)*, it rounds to the nearest even for types. Refer to https://en.wikipedia.org/wiki/Rounding for details.

*y_zero_point* and *y* must have the same type. *y_zero_point* is usually not used for quantization to float16 and bfloat16 types, but the quantization formula remains the same for consistency, and the type of the attribute *y_zero_point* still determines the quantization type. *x* and *y_scale* are allowed to have different types. The type of *y_scale* determines the precision of the division operation between *x* and *y_scale*, unless the precision attribute is specified.

There are two supported quantization granularities, determined by the shape of *y_scale*. In all cases, *y_zero_point* must have the same shape as *y_scale*.

Per-tensor/per-layer quantization: *y_scale* is a scalar.

Per-axis/per-channel quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape *(D0, ..., Di, ..., Dn)* and *axis=i*, *y_scale* is a 1-D tensor of length *Di*.

Attributes
``````````

**axis - INT** (default is '1'):

(Optional) The axis of the dequantizing dimension of the input tensor. Used only for per-axis/per-channel quantization. Negative value means counting dimensions from the back. Accepted range is *[-r, r-1]* where *r = rank(input)*. When the rank of the input is 1, per-tensor/per-layer quantization is applied, rendering the axis unnecessary in this scenario.

Inputs
``````

Between 2 and 3 inputs.

- **x** (heterogeneous) - **T1**:

N-D full precision Input tensor to be quantized.

- **y_scale** (heterogeneous) - **T2**:

Scale for doing quantization to get *y*. For per-tensor/per-layer quantization the scale is a scalar, for per-axis/per-channel quantization it is a 1-D Tensor.

- **y_zero_point** (optional, heterogeneous) - **T3**:

Zero point for doing quantization to get *y*. Shape must match *y_scale*. Default is uint8 with zero point of 0 if it’s not specified.

Outputs
```````

- **y** (heterogeneous) - **T3**:

N-D quantized output tensor. It has same shape as input *x*.

Type Constraints
````````````````

- **T1** in ( tensor(float) ):

The type of the input ‘x’.

- **T2** in ( tensor(float) ):

The type of the input ‘y_scale’.

- **T3** in ( tensor(int32), tensor(int16), tensor(int8), tensor(uint32), tensor(uint16), tensor(uint8), tensor(float16), tensor(bfloat16) ):

The type of the input ‘y_zero_point‘ and the output ‘y‘.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
