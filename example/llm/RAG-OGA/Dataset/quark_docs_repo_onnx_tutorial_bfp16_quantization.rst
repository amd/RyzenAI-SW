BFP16 (Block floating point) Quantization
=========================================

BFP16 (Block Floating Point 16) quantization is a technique that represents tensors using a block floating-point format, where multiple numbers share a common exponent. This format can provide a balance between dynamic range and precision while using fewer bits than standard floating-point representations. BFP16 quantization aims to reduce the computational complexity and memory footprint of neural networks, making them more efficient for inference on various hardware platforms, particularly those with limited resources.

Key Concepts
------------

1. **Block Floating Point Format**: In BFP16 quantization, data is grouped into blocks, and each block shares a common exponent. This reduces the storage requirements while preserving a sufficient dynamic range for most neural network operations. It differs from standard floating-point formats, which assign an individual exponent to each number.

2. **Dynamic Range and Precision**: By using a shared exponent for each block, BFP16 can achieve a balance between range and precision. It allows for a more flexible representation of values compared to fixed-point formats and can adapt to the magnitude of the data within each block.

3. **Reduced Computation Costs**: BFP16 quantization reduces the number of bits required to represent each tensor element, leading to lower memory usage and faster computations. This is particularly useful for deploying models on devices with limited hardware resources.

4. **Compatibility with Mixed Precision**: BFP16 can be combined with other quantization methods, such as mixed precision quantization, to optimize neural network performance further. This compatibility allows for flexible deployment strategies tailored to specific accuracy and performance requirements.

Benefits of BFP16 Quantization
------------------------------

1. **Improved Efficiency**: BFP16 quantization significantly reduces the
   number of bits needed to represent tensor values, leading to reduced
   memory bandwidth and faster computation times. This makes it ideal
   for resource-constrained environments.

2. **Maintained Accuracy**: By balancing dynamic range and precision,
   BFP16 quantization minimizes the accuracy loss that can occur with
   more aggressive quantization methods.

3. **Hardware Compatibility**: BFP16 is well-supported by modern hardware
   accelerators, making it a flexible and efficient choice for
   large-scale neural network training and deployment.

How to enable BFP16 quantization in AMD Quark for ONNX?
-------------------------------------------------------

Here is a simple example of how to enable BFP16 quantization in AMD Quark
for ONNX.

.. code-block:: python

   from quark.onnx import ModelQuantizer, ExtendedQuantType, ExtendedQuantFormat
   from onnxruntime.quantization.calibrate import CalibrationMethod
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(
       calibrate_method=CalibrationMethod.MinMax,
       quant_format=ExtendedQuantFormat.QDQ,
       activation_type=ExtendedQuantType.QBFP,
       weight_type=ExtendedQuantType.QBFP,
   )
   config = Config(global_quant_config=quant_config)

.. note:: When inferring with ONNX Runtime, we need to register the custom op's so (Linux) or dll (Windows) file in the ORT session options.

.. code-block:: python

    import onnxruntime
    from quark.onnx import get_library_path as vai_lib_path

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
    sess_options.register_custom_ops_library(vai_lib_path(device))
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)

How to further improve the accuracy of a BFP16 quantized model in AMD Quark for ONNX?
-------------------------------------------------------------------------------------

If you want to further improve the effectiveness of BFP16 quantization after applying it, you can use fast_finetune to enhance the quantization accuracy. Please refer to this :doc:`link <accuracy_algorithms/ada>` for more details on how to enable BFP16 Quantization in the configuration of AMD Quark for ONNX. This is a simple example code.

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
           'FastFinetune': {
               'DataSize': 100,
               'FixedSeed': 1705472343,
               'BatchSize': 5,
               'NumIterations': 100,
               'LearningRate': 0.000001,
               'OptimAlgorithm': 'adaquant',
               'OptimDevice': 'cpu',
               'InferDevice': 'cpu',
               'EarlyStop': True,
           }
       }
   )
   config = Config(global_quant_config=quant_config)

.. note:: You can install onnxruntime-gpu instead of onnxruntime to accelerate inference speed. The BFP QuantType only supports fast_finetune with AdaQuant, not AdaRound. Set 'InferDevice' to 'cuda:0' to use the GPU for inference. Additionally, set 'OptimDevice' to 'cuda:0' to accelerate fast_finetune training with the GPU.

Example
-------

An example of quantizing a model using the BFP16 quantization is :doc:`available here <example_quark_onnx_BFP>`.

.. raw:: html

   <!-- omit in toc -->

License
-------

Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
