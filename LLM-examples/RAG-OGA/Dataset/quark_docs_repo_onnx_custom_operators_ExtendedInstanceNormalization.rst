ExtendedInstanceNormalization
=============================

ExtendedInstanceNormalization - 1
---------------------------------

Version
```````
- **name**: ExtendedInstanceNormalization

- **domain**: com.amd.quark

Summary
```````

This is a customized version of the official operator InstanceNormalization, it carries out instance normalization with bfloat16.

y = scale * (x - mean) / sqrt(variance + epsilon) + B, where mean and variance are computed per instance per channel.

Attributes
``````````

**epsilon - FLOAT** (default is '1e-05'):

The epsilon value to use to avoid division by zero.

Inputs
``````

- **input** (heterogeneous) - **T**:

Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size.

- **scale** (heterogeneous) - **T**:

The input 1-dimensional scale tensor of size C.

- **B** (heterogeneous) - **T**:

The input 1-dimensional bias tensor of size C.

Outputs
```````

- **output** (heterogeneous) - **T**:

The output tensor of the same shape as input.

Type Constraints
````````````````

- **T** in ( tensor(float) ):

Constrain input and output types to float tensors.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
