
Microscaling (MX)
=================

.. note::

    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

In this tutorial, you learn how to use Microscaling (MX) quantization.

MX is an advancement over Block Floating Point (BFP), aiming to improve the numerical efficiency and flexibility of low-precision computations for AI.

BFP groups numbers (for example, tensors, arrays) into blocks, where each block shares a common exponent, and the values in the block are represented with individual mantissas (and the sign bit). This approach is effective for reducing memory usage, but it is coarse-grained, meaning all numbers within a block are forced to have the same exponent, regardless of their individual value ranges.

MX, on the other hand, allows for finer-grained scaling within a block. Instead of forcing all elements in the block to share a single exponent, MX assigns a small-scale adjustment to individual or smaller groups of values within the block. This finer granularity improves precision, as each value or subgroup of values can adjust more dynamically to their specific range, reducing the overall quantization error compared to BFP.

What is MX Quantization?
------------------------

The `OCP MX specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__
introduces several specific MX formats, including MXFP8, MXFP6, MXFP4, and MXINT8. These formats are implemented in the AMD Quark ONNX quantizer through a custom operation named "MXQuantizeDequantize", which has an ``element_dtype`` attribute to set the data type for the elements (while the data type for the shared scale is always E8M0).

+-------------------+------------------------+
| MX Formats        | "element_dtype" values |
+===================+========================+
| MXFP8(E5M2)       | 'fp8_e5m2'             |
+-------------------+------------------------+
| MXFP8(E4M3)       | 'fp8_e4m3'             |
+-------------------+------------------------+
| MXFP6(E3M2)       | 'fp6_e3m2'             |
+-------------------+------------------------+
| MXFP6(E2M3)       | 'fp6_e2m3'             |
+-------------------+------------------------+
| MXFP4(E2M1)       | 'fp4_e2m1'             |
+-------------------+------------------------+
| MXINT8            | 'int8'                 |
+-------------------+------------------------+

If you initialize the quantizer with the MX configuration, it quantizes all the activations and weights using the MXQuantizeDequantize nodes.

How to Enable MX Quantization in AMD Quark for ONNX?
----------------------------------------------------

Here is a simple example of how to enable MX quantization with MXINT8 in AMD Quark for ONNX:

.. code-block:: python

   from quark.onnx import ModelQuantizer, ExtendedQuantType, ExtendedQuantFormat
   from onnxruntime.quantization.calibrate import CalibrationMethod
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                     quant_format=ExtendedQuantFormat.QDQ,
                                     activation_type=ExtendedQuantType.QMX,
                                     weight_type=ExtendedQuantType.QMX,
                                     extra_options={
                                       'MXAttributes': {
                                         'element_dtype': 'int8',
                                         'axis': 1,
                                         'block_size': 32,
                                         'rounding_mode': 2,
                                       },
                                     })

   config = Config(global_quant_config=quant_config)

   quantizer = ModelQuantizer(config)

   quantizer.quantize_model(input_model_path, output_model_path, data_reader)

.. note::

   When inferencing with ONNXRuntime, you need to register the custom operator's shared object file (Linux) or DLL file (Windows) in the ORT session options.

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

How to Further Improve the Accuracy of a MX Quantized Model?
------------------------------------------------------------

If you want to further improve the effectiveness of MX quantization after applying it, you can use ``fast_finetune`` to enhance the quantization accuracy. Refer to this :doc:`link <accuracy_algorithms/ada>`.

Here is a simple example code which is fast finetuning a MXINT8 model:

.. code-block:: python

   from quark.onnx import ModelQuantizer, ExtendedQuantFormat, ExtendedQuantType
   from onnxruntime.quantization.calibrate import CalibrationMethod
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                     quant_format=ExtendedQuantFormat.QDQ,
                                     activation_type=ExtendedQuantType.QMX,
                                     weight_type=ExtendedQuantType.QMX,
                                     include_fast_ft=True,
                                     extra_options={
                                       'MXAttributes': {
                                         'element_dtype': 'int8',
                                         'axis': 1,
                                         'block_size': 32,
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
                                     })
   config = Config(global_quant_config=quant_config)

.. note::

   You can install onnxruntime-rocm or onnxruntime-gpu instead of onnxruntime to accelerate inference speed. Set 'InferDevice' to 'hip:0' or 'cuda:0' to use the GPU for inference. Additionally, set 'OptimDevice' to 'hip:0' or 'cuda:0' to accelerate the training process of fast finetuning with the GPU.

Example
-------

An example of quantizing a model using the Microscaling quantization is :doc:`available here <example_quark_onnx_MX>`.

.. raw:: html

   <!-- omit in toc -->

License
-------

Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
