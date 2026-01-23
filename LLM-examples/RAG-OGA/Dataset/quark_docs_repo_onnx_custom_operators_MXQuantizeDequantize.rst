MXQuantizeDequantize
====================

MXQuantizeDequantize - 1
------------------------

Version
```````
- **name**: MXQuantizeDequantize

- **domain**: com.amd.quark

Summary
```````

Microscaling, also known as OCP (Open Compute Project) MX, assigns a small-scale adjustment to individual exponent within the block: in addition to the shared exponent, each element also has its own micro exponent, meaning that the element has an independent data type. This finer granularity improves precision, as each value can adjust more dynamically to its specific range. The `OCP MX specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_ introduces several concrete formats, including MXFP8(E5M2), MXFP8(E4M3), MXFP6(E3M2), MXFP6(E2M3), MXFP4(E2M1) and MXINT8.

This operator converts floating-point values (typically 32-bit floating-point numbers) into Microscaling values, and then convert them back. It approximates the Quantize-Dequantize process and introduces quantization errors.

.. note::

   Compared with MicroeXponents, Microscaling applies linear scaling per element and is simpler, hardware-friendly, and more stable, making it ideal for production and general-purpose deployment. In contrast, MicroeXponents shares a dynamic exponent within groups, offering better compression and dynamic range at the cost of higher complexity and potential numerical instability. For most applications, especially those targeting standard inference engines or requiring robustness, perhaps Microscaling is the preferred choice.

Attributes
``````````

**element_dtype - STRING** (default is 'int8'):

(Optional) Specify the type of elements, options are 'fp8_e5m2', 'fp8_e4m3', 'fp6_e3m2', 'fp6_e2m3', 'fp4_e2m1', 'int8'.

**axis - INT** (default is '1'):

(Optional) The axis for spliting the input tensor to blocks.

**block_size - INT** (default is '32'):

(Optional) Number of elements in the block.

**rounding_mode - INT** (default is '0'):

(Optional) Rounding mode, 0 for rounding half away from zero, 1 for rounding half upward and 2 for rounding half to even.


Table 1. Configurations of OCP MX data types

+----------------------+------------------+------------------+------------------+------------------+------------------+------------------+
|                      | MXFP8(E5M2)      | MXFP8(E4M3)      | MXFP6(E3M2)      | MXFP6(E2M3)      | MXFP4(E2M1)      | MXINT8           |
+======================+==================+==================+==================+==================+==================+==================+
| element_dtype        | fp8_e5m2         | fp8_e4m3         | fp6_e3m2         | fp6_e2m3         | fp4_e2m1         | int8             | 
+----------------------+------------------+------------------+------------------+------------------+------------------+------------------+
| axis                 | 1                | 1                | 1                | 1                | 1                | 1                |
+----------------------+------------------+------------------+------------------+------------------+------------------+------------------+
| block_size           | 32               | 32               | 32               | 32               | 32               | 32               |
+----------------------+------------------+------------------+------------------+------------------+------------------+------------------+
| rounding_mode        | 2                | 2                | 2                | 2                | 2                | 2                |
+----------------------+------------------+------------------+------------------+------------------+------------------+------------------+


Inputs
``````
- **x** (heterogeneous) - **T**:

N-D input tensor.

Outputs
```````

- **y** (heterogeneous) - **T**:

N-D output tensor. It would have accuracy loss compared to the input tensor *x*.

Type Constraints
````````````````

- **T** in ( tensor(float) ):

Constrain input and output types to float tensors.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
