.. raw:: html

   <!-- omit in toc -->

Introduction
============

.. note::  
  
    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

BFloat16 (Brain Floating Point 16) is a floating-point data format used in deep learning to reduce memory usage and computation while maintaining sufficient numerical precision. Unlike other quantization formats like INT8 or FP16, BF16 maintains the same range as FP32 but reduces precision, making it particularly useful for training and inference in neural networks.

AMD accelerators like latest CPU, NPU and GPU devices support BF16 natively, enabling faster matrix operations and reducing latency. In this tutorial, we will explain how to quantize a model into BF16 using AMD Quark.

BF16 quantization in AMD Quark for ONNX
---------------------------------------

Here is a simple example of how to enable BF16 quantization.

.. code:: python

   from quark.onnx import ModelQuantizer, ExtendedQuantType, ExtendedQuantFormat
   from onnxruntime.quantization.calibrate import CalibrationMethod
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                     quant_format=ExtendedQuantFormat.QDQ,
                                     activation_type=ExtendedQuantType.QBFloat16,
                                     weight_type=ExtendedQuantType.QBFloat16,
                                     extra_options={'BF16QDQToCast': True}
                                     )

   config = Config(global_quant_config=quant_config)

   quantizer = ModelQuantizer(config)

   quantizer.quantize_model(input_model_path, output_model_path, data_reader)

The BF16 quantization in the previous example inserts a custom Q/DQ pair for each tensor, which
converts the model weights and activations from FP32 to BF16 directly, just as most frameworks do.

In fact, BF16 has the same range as FP32, but with only 7 bits for the mantissa, it sacrifices
precision. This means small differences between numbers can disappear, which can amplify
numerical instability and cause overflow problems.

To address the overflow issue in BF16 quantization, you can apply calibration and re-scale
weights and activations to better align with dynamic range and utilize the dense numeric
area near zero of BF16. To enable this, set ``WeightScaled`` or ``ActivationScaled``
in extra options if you are seeing overflow issues.

.. code:: python

   quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                     quant_format=ExtendedQuantFormat.QDQ,
                                     activation_type=ExtendedQuantType.QBFloat16,
                                     weight_type=ExtendedQuantType.QBFloat16,
                                     extra_options={
                                         'WeightScaled': True,
                                         'ActivationScaled': True,
                                     }
                                    )

.. note::
    When inference with ONNXRuntime, you need to register the custom OPs so(Linux) or dll(Windows) file in the ORT session options.

.. code:: python

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

How to Further Improve the Accuracy for BF16 Quantization?
----------------------------------------------------------

You can finetune the quantized model to further improve the accuracy of BF16 quantization.
The Fast Finetuning function in AMD Quark for ONNX includes two algorithms: AdaRound and AdaQuant.
There is no explicit rounding in BF16 quantization, so only AdaQuant can be used.

.. code:: python

   quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                     quant_format=ExtendedQuantFormat.QDQ,
                                     activation_type=ExtendedQuantType.QBFloat16,
                                     weight_type=ExtendedQuantType.QBFloat16,
                                     extra_options={
                                         'FastFinetune': {
                                             'NumIterations': 1000,
                                             'LearningRate': 1e-6,
                                             'OptimAlgorithm': 'adaquant',
                                             'OptimDevice': 'cpu',
                                             'InferDevice': 'cpu',
                                         }
                                     }
                                    )

.. raw:: html

   <!-- omit in toc -->

License
-------

Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
