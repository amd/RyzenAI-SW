.. raw:: html

   <!-- omit in toc -->

Introduction
============

.. note::

    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

This tutorial explains how to use Microexponents (MX) data types for quantization.

Microexponents represent an advancement over Block Floating Point (BFP), aiming to improve the numerical efficiency and flexibility of low-precision computations for artificial intelligence.

Block Floating Point groups numbers (for example, tensors and arrays) into blocks, where each block shares a common exponent, and the values in the block are represented with individual mantissas (and the sign bit). This approach effectively reduces memory usage; however, it is coarse-grained, meaning all numbers within a block are forced to have the same exponent, regardless of their individual value ranges.

To address this issue, Microexponents extend the concept of BFP by introducing two levels of exponents: shared exponents for entire blocks and micro exponents for finer-grained sub-blocks. This dual-level approach enables more precise scaling of individual elements within a block, reducing quantization error and improving the representational range. By allowing sub-blocks to adjust their scaling more accurately, Microexponents strike a balance between the coarse-grained nature of BFP and the fine-grained precision of floating-point formats.

This technique is particularly useful for low-precision computations in modern deep learning models, where maintaining accuracy while optimizing memory and power usage is critical. Furthermore, hardware accelerators that support Microexponents can process data more efficiently while preserving the numerical stability of operations such as matrix multiplications and convolutions.

What is Microexponents Quantization?
------------------------------------

`This paper <https://arxiv.org/abs/2302.08007>`__ introduces several specific formats, including MX4, MX6, and MX9. We have implemented these formats in AMD Quark ONNX quantizer through a custom op named "BFPQuantizeDequantize". This op supports classical BFP and Microexponents both by setting attribute ``bfp_method`` to ``to_bfp`` for BFP or ``to_bfp_prime`` for Microexponents. To select MX4, MX6, and MX9, set the value for the ``bit_width`` attribute according to the following table.

+-------------------+------------------------+
| Formats           | "bit_width" values     |
+===================+========================+
| MX4               | 11                     |
+-------------------+------------------------+
| MX6               | 13                     |
+-------------------+------------------------+
| MX9               | 16                     |
+-------------------+------------------------+

Other parameters should be set as defined in the paper.

How to enable MX9 quantization in AMD Quark for ONNX?
-----------------------------------------------------

Here is a simple example of how to enable Microexponents quantization with
MX9 in AMD Quark for ONNX.

.. code-block:: python

   from quark.onnx import ModelQuantizer, ExtendedQuantType, ExtendedQuantFormat
   from onnxruntime.quantization.calibrate import CalibrationMethod
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                     quant_format=ExtendedQuantFormat.QDQ,
                                     activation_type=ExtendedQuantType.QBFP,
                                     weight_type=ExtendedQuantType.QBFP,
                                     extra_options={
                                       'BFPAttributes': {
                                           'bfp_method': "to_bfp_prime",
                                           'axis': 1,
                                           'bit_width': 16,
                                           'block_size': 16,
                                           'sub_block_size': 2,
                                           'sub_block_shift_bits': 1,
                                           'rounding_mode': 2,
                                       },
                                     })

   config = Config(global_quant_config=quant_config)

   quantizer = ModelQuantizer(config)

   quantizer.quantize_model(input_model_path, output_model_path, data_reader)

*Note* : When inferencing with ONNXRuntime, you need to register the custom operator's shared object file (Linux) or DLL file (Windows) in the ORT session options.

.. code-block:: python

    import onnxruntime
    from quark.onnx import get_library_path

    if 'ROCMExecutionProvider' in onnxruntime.get_available_providers():
        device = 'ROCM'
        providers = ['ROCMExecutionProvider']
    elif 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        device = 'CUDA'
        providers = ['CUDAExecutionProvider']
    else:
        device = 'CPU'
        providers = ['CPUExecutionProvider']

    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path(device))
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)

How to Further Improve the Accuracy of a MX9 Quantized Model?
-------------------------------------------------------------

If you want to further improve the effectiveness of MX9 quantization after applying it, you can use ``fast_finetune`` to enhance the quantization accuracy. Refer to this :doc:`link <accuracy_algorithms/ada>`. This is a simple example code:

.. code-block:: python

   from quark.onnx import ModelQuantizer, ExtendedQuantFormat, ExtendedQuantType
   from onnxruntime.quantization.calibrate import CalibrationMethod
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(
       calibrate_method=CalibrationMethod.MinMax,
       quant_format=ExtendedQuantFormat.QDQ,
       activation_type=ExtendedQuantType.QBFP,
       weight_type=ExtendedQuantType.QBFP,
       include_fast_ft=True,
       extra_options={
          'BFPAttributes': {
                              'bfp_method': "to_bfp_prime",
                              'axis': 1,
                              'bit_width': 16,
                              'block_size': 16,
                              'sub_block_size': 2,
                              'sub_block_shift_bits': 1,
                              'rounding_mode': 2,
                           },
           'FastFinetune': {
                              'DataSize': 100,
                              'FixedSeed': 1705472343,
                              'BatchSize': 2,
                              'NumIterations': 1000,
                              'LearningRate': 0.00001,
                              'OptimAlgorithm': 'adaquant',
                              'OptimDevice': 'cpu',
                              'InferDevice': 'cpu',
                              'EarlyStop': True,
                           },
       }
   )
   config = Config(global_quant_config=quant_config)

.. note::

     You can install onnxruntime-rocm or onnxruntime-gpu instead of onnxruntime to accelerate inference speed. Set ``InferDevice`` to ``hip:0`` or ``cuda:0`` to use the GPU for inference. Additionally, set ``OptimDevice`` to ``hip:0`` or ``cuda:0`` to accelerate the training process of fast finetuning with the GPU.

Examples
--------

An example of quantizing a model using the Microscaling quantization is :doc:`available here <example_quark_onnx_MX>`.

.. raw:: html

   <!-- omit in toc -->

License
-------

Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
