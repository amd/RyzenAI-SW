.. raw:: html

   <!--
   ## License
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->

Quantizing with Rotation and SmoothQuant
========================================

Introduction
------------

Weight INT8 and Activation INT8 symmetric post-training quantization (W8A8) is one of the most common quantization methods supported by current hardware. It is highly compatible with hardware acceleration, facilitating efficient deployment on various platforms.

The following are the four most common quantization strategies for W8A8:

- Weight INT8 (per tensor) activation INT8 (per tensor) static quantization
- Weight INT8 (per channel) activation INT8 (per tensor) static quantization
- Weight INT8 (per channel) activation INT8 (per tensor) dynamic quantization
- Weight INT8 (per channel) activation INT8 (per token) dynamic quantization

AMD Quark-Torch now offers two pre-optimizations that are suitable for W8A8 quantization:

- Activation/weight smoothing (SmoothQuant). For more details, see :doc:`here <smoothquant>`.
- `Rotation <https://arxiv.org/abs/2405.04532>`_ (R1 in `SpinQuant <https://arxiv.org/abs/2405.16406>`_ with Hadamard matrix)

Sometimes, these two methods are combined by smoothing ``Linear-Linear`` patterns (Smooth_fc_fc) in decoder layers and rotating ``RSMNorm-Linear`` patterns.

Results
-------

In this example, ``meta-llama/Meta-Llama-3.1-8B-Instruct`` is used. All linear layers, excluding lm_head, are quantized using the pre-trained Float16 model (original Float16 model perplexity: 7.2155).

+--------------------------------------------------------------------+--------------------+-------------------+------------------------------------------+
| Quantization Strategy                                              | Smooth(alpha=0.85) | Smooth(alpha=0.5) | Smooth_fc_fc(alpha=0.5) + Rotation       |
+====================================================================+====================+===================+==========================================+
| w_int8_per_tensor_a_int8_per_tensor static quantization            | -                  | 19.42             | **8.58**                                 |
+--------------------------------------------------------------------+--------------------+-------------------+------------------------------------------+
| w_int8_per_channel_a_int8_per_tensor static quantization           | **8.37**           | 15.95             | 8.40                                     |
+--------------------------------------------------------------------+--------------------+-------------------+------------------------------------------+
| w_int8_per_channel_a_int8_per_tensor dynamic quantization          | **9.08**           | 23.35             | 9.22                                     |
+--------------------------------------------------------------------+--------------------+-------------------+------------------------------------------+
| w_int8_per_channel_a_int8_per_token dynamic quantization           | 7.35               | 7.29              | **7.27**                                 |
+--------------------------------------------------------------------+--------------------+-------------------+------------------------------------------+
| w_int8_per_tensor_a_int8_per_tensor_kv_cache_int8_per_tensor       | -                  | 20.51             | **8.58**                                 |
| static quantization                                                |                    |                   |                                          |
+--------------------------------------------------------------------+--------------------+-------------------+------------------------------------------+
| w_int8_per_channel_a_int8_per_tensor_kv_cache_int8_per_tensor      | **8.38**           | 16.87             | 8.42                                     |
| static quantization                                                |                    |                   |                                          |
+--------------------------------------------------------------------+--------------------+-------------------+------------------------------------------+
| w_int8_per_channel_a_int8_per_tensor_kv_cache_int8_per_tensor      | **9.09**           | 23.46             | 9.26                                     |
| dynamic quantization                                               |                    |                   |                                          |
+--------------------------------------------------------------------+--------------------+-------------------+------------------------------------------+
| w_int8_per_channel_a_int8_per_token_kv_cache_int8_per_token        | 7.35               | 7.29              | **7.28**                                 |
| dynamic quantization                                               |                    |                   |                                          |
+--------------------------------------------------------------------+--------------------+-------------------+------------------------------------------+

'-' means perplexity > 30

Expanding with more models
--------------------------

Examples are provided for some typical large language models (LLMs). However, if you want to try these strategies with new models, you may need to follow several steps.

For Smooth, you can set hyperparameters and layer patterns manually with code or a JSON file. In addition, scripts are provided for generating the configuration automatically.

For Rotation, you simply need to enable the rotation option. AMD Quark-Torch supports analyzing the model structure with the ``torch.compile`` graph, helping you identify if there are any pattern layers that could be rotated. This feature is user-friendly.