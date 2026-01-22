ExtendedLSTM
============

ExtendedLSTM - 1
----------------

Version
```````
- **name**: ExtendedLSTM

- **domain**: com.amd.quark

Summary
```````

This is a customized version of the official operator LSTM, it computes an one-layer quantized LSTM with bfloat16.

Attributes
``````````

**direction - STRING** (default is 'forward'):

Specify if the RNN is forward, reverse, or bidirectional. Must be one of forward (default), reverse, or bidirectional, but currently it supports "bidirectional" only.

**hidden_size - INT**:

Number of neurons in the hidden layer.

**input_forget - INT** (default is '0'):

Couple the input and forget gates if 1, Currently it can only be 0.

**layout - INT** (default is '0'):

The shape format of inputs and outputs, Currently it can only be 0.

**x_scale - FLOAT** :

Scale for input *X*. It only supports per-tensor/per-layer quantization, so the scale should be a scalar.

**x_zero_point - INT**:

Zero point for input *X*. Shape must match *x_scale*. It only supports uint16 quantization, so the zero point value should be in the range of [0, 65535].

**w_scale - FLOAT** :

Scale for input *W*. It only supports per-tensor/per-layer quantization, so the scale should be a scalar.

**w_zero_point - INT**:

Zero point for input *W*. Shape must match *w_scale*. It only supports uint16 quantization, so the zero point value should be in the range of [0, 65535].

**r_scale - FLOAT** :

Scale for input *R*. It only supports per-tensor/per-layer quantization, so the scale should be a scalar.

**r_zero_point - INT**:

Zero point for input *R*. Shape must match *r_scale*. It only supports uint16 quantization, so the zero point value should be in the range of [0, 65535].

**b_scale - FLOAT** :

Scale for input *B*. It only supports per-tensor/per-layer quantization, so the scale should be a scalar.

**b_zero_point - INT**:

Zero point for input *B*. Shape must match *b_scale*. It only supports uint16 quantization, so the zero point value should be in the range of [0, 65535].

Inputs
``````

- **X** (heterogeneous) - **T**:

The input sequences packed (and potentially padded) into one 3-D tensor with the shape of *[seq_length, batch_size, input_size]*.

- **W** (heterogeneous) - **T**:

The weight tensor for the gates. Concatenation of *W[iofc]* and *WB[iofc]* (if bidirectional) along dimension 0. The tensor has shape *[num_directions, 4*hidden_size, input_size]*.

- **R** (heterogeneous) - **T**:

The recurrence weight tensor. Concatenation of *R[iofc]* and *RB[iofc]* (if bidirectional) along dimension 0. This tensor has shape *[num_directions, 4*hidden_size, hidden_size]*.

- **B** (optional, heterogeneous) - **T**:

The bias tensor for input gate. Concatenation of *[Wb[iofc]*, *Rb[iofc]]*, and *[WBb[iofc], RBb[iofc]]* (if bidirectional) along dimension 0. This tensor has shape *[num_directions, 8*hidden_size]*. Optional: If not specified - assumed to be 0.

Outputs
```````

- **Y** (optional, heterogeneous) - **T**:

A tensor that concatenates all the intermediate output values of the hidden. It has shape *[seq_length, num_directions, batch_size, hidden_size]*.

Type Constraints
````````````````
- **T** in ( tensor(float) ):

Constrain input and output types to float tensors.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
