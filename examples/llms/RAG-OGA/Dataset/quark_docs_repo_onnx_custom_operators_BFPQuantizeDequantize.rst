BFPQuantizeDequantize
=====================

BFPQuantizeDequantize - 1
-------------------------

Version
```````
- **name**: BFPQuantizeDequantize

- **domain**: com.amd.quark

Summary
```````

Block Floating Point (BFP) groups numbers (e.g., tensors, arrays) into blocks, where each block shares a common exponent, and the values in the block are represented with individual mantissas (and the sign bit). This approach offers the performance and speed of 8-bit operations while bringing the precision closer to 16-bit operations.

MicroeXponents (MX) extends the concept of BFP by introducing two levels of exponents: shared exponents for entire blocks and micro exponents for finer-grained sub-blocks. This two-level approach enables more precise scaling of individual elements within a block, reducing quantization error and improving the representational range. The paper https://arxiv.org/abs/2302.08007 introduces three specific formats: MX4, MX6 and MX9, which have different bits of mantissa.

This operator converts floating-point values (typically 32-bit floating-point numbers) into BFP or MX values, then convert them back. It approximates the Quantize-Dequantize process and introduces quantization errors.

.. note::

   In addition to MicroeXponent, there is another technique Microscaling also abbreviated as MX, which has two levels of exponent as well. Unlike MicroeXponent's micro exponents shared over sub-blocks, Microscaling assigns a small-scale adjustment to individual exponents within the block by letting them have an independent data type, such as FP8, FP6 and etc., meaning that each element has its own micro exponent! This finer scaling granularity improves precision, as each value can adjust more dynamically to its specific range. We have implemented Microscaling data types in another custom operator MXQuantizeDequantize, please check out its specification for more details.

Attributes
``````````

**bfp_method - STRING** (default is 'to_bfp'):

(Optional) Specify the type of block floating-point, 'to_bfp' for the vanilla BFP and 'to_bfp_prime' for BFP's variant MicroeXponents.

**axis - INT** (default is '1'):

(Optional) The axis for spliting the input tensor to blocks.

**bit_width - INT** (default is '16'):

(Optional) Bits for the block float-point structure. Default is 16, which corresponds to the commonly used BFP16 that has 8 bits for the shared exponent, 1 bit for sign and 7 bits for mantissa.

**block_size - INT** (default is '8'):

(Optional) Number of elements in the block.

**rounding_mode - INT** (default is '0'):

(Optional) Rounding mode, 0 for rounding half away from zero, 1 for rounding half upward and 2 for rounding half to even.

**sub_block_size - INT** (default is '2'):

(Optional) Size of a sub block, only effective if 'bfp_method' is 'to_bfp_prime'.

**sub_block_shift_bits - INT** (default is '1'):

(Optional) Shift bits of a sub block, only effective if 'bfp_method' is 'to_bfp_prime'.


Table 1. Configurations of commonly used block float-point series data types

+----------------------+----------------+------------------+------------------+------------------+
|                      | BFP16          | MX4              | MX6              | MX9              |
+======================+================+==================+==================+==================+
| bfp_method           | to_bfp         | to_bfp_prime     | to_bfp_prime     | to_bfp_prime     |
+----------------------+----------------+------------------+------------------+------------------+
| axis                 | 1              | 1                | 1                | 1                |
+----------------------+----------------+------------------+------------------+------------------+
| bit_width            | 16             | 11               | 13               | 16               |
+----------------------+----------------+------------------+------------------+------------------+
| block_size           | 8              | 16               | 16               | 16               |
+----------------------+----------------+------------------+------------------+------------------+
| rounding_mode        | 2              | 2                | 2                | 2                |
+----------------------+----------------+------------------+------------------+------------------+
| sub_block_size       | N/A            | 2                | 2                | 2                |
+----------------------+----------------+------------------+------------------+------------------+
| sub_block_shift_bits | N/A            | 1                | 1                | 1                |
+----------------------+----------------+------------------+------------------+------------------+


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
