.. raw:: html

   <!-- omit in toc -->

Mixed Precision
===============

.. note::  
  
    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

As the scale and complexity of AI models continue to grow, optimizing their performance and efficiency becomes a top priority. Quantizing models to mixed precision emerges as a powerful technique, allowing AI practitioners to balance computational speed, memory usage, and model accuracy. This tutorial introduces the characteristics and usage of AMD Quark for ONNX's mixed precision.  
  
What is Mixed Precision Quantization?  
-------------------------------------  
  
Mixed precision quantization involves using different precision levels for different parts of a neural network, such as using 8-bit integers for some layers while retaining higher precision, for example, 16-bit or 32-bit floating point, for others. This approach leverages the fact that not all parts of a model are equally sensitive to quantization. By carefully selecting which parts of the model can tolerate lower precision, you achieve significant computational savings while minimizing the impact on model accuracy.

Benefits of Mixed Precision Quantization
----------------------------------------

1. **Enhanced Efficiency**: By using lower precision where possible, mixed precision quantization significantly reduces computational load and memory usage, leading to faster inference times and lower power consumption.

2. **Maintained Accuracy**: By selectively applying higher precision to sensitive parts of the model, mixed precision quantization minimizes the accuracy loss that typically accompanies uniform quantization.

3. **Flexibility**: Mixed precision quantization is adaptable to various types of neural networks and can be tailored to specific hardware capabilities, making it suitable for a wide range of applications.

Mixed Precision Quantization in AMD Quark for ONNX
--------------------------------------------------

AMD Quark for ONNX is designed to push the boundaries of what is possible with mixed precision. Here is what sets it apart:

1. **Support for All Types of Granularity**

Granularity refers to the level at which precision can be controlled within a model. AMD Quark for ONNX mixed precision supports:

- **Element-wise Granularity**

Element-wise mixed precision allows assigning different numeric precisions to activations and weights at the individual computation level. For example: INT8 Weights for efficient storage and computation and INT16 Activation to preserve dynamic range.

- **Layer-wise Granularity**

Different layers of a neural network can have varying levels of sensitivity to quantization. Layer-wise mixed precision assigns precision levels to layers based on their sensitivity, optimizing both performance and accuracy. For example, INT16 to sensitive layers for high accuracy while INT8 to others for efficient inference.

- **Tensor-wise Granularity**

Tensor-wise mixed precision enables assigning different precisions to individual tensors within a layer. For example, in an INT8 quantized model, specifying any sensitive tensor as INT16.

2. **Support for Various Data Types**

AMD Quark for ONNX mixed precision is not limited to a few integer data types, it supports a wide range of precisions, including but not limited to:

- **More Integer Data Types**

Traditional INT8/UINT8 for significant memory and computation savings, INT16/UINT16 for higher precision and INT32/UINT32 for experimental usage.

- **Half Floating-Point Data Types**

Float16 and BFloat16, the former can be used for iGPU/GPU applications, while the latter can be used for NPU deployment.

- **Block Floating-Point Data Types**

The bit-width for shared exponents and elements can be set arbitrarily. The typical data type is BFP16.

- **Microexponents Data Types**

Supports all the Microexponents data types, including MX4, MX6 and MX9.

- **Microscaling Data Types**

Supports all the Microscaling data types, including MXINT8, MXFP8_E4M3, MXFP8_E5M2, MXFP6_E3M2, MXFP6_E2M3 and MXFP4.

How to Enable Mixed Precision in AMD Quark for ONNX?
----------------------------------------------------

Here, BF16 mixed with BFP16 is used as an example to illustrate how to build configurations for mixed precision quantization.
In fact, you can mix any two other data types equally.

- **Element-wise**

In this configuration, BFP16 is assigned to activations and BFloat16 to weights. Here the BFP16 quantization is
executed by custom operator named "BFPQuantizeDequantize", whose default attributes make it work on BFP16 mode.

.. code-block:: python

   from quark.onnx import ModelQuantizer, CalibrationMethod, ExtendedQuantFormat, ExtendedQuantType
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   # Build the configuration
   quant_config = QuantizationConfig(
       calibrate_method=CalibrationMethod.MinMax,
       quant_format=ExtendedQuantFormat.QDQ,
       activation_type=ExtendedQuantType.QBFP,
       weight_type=ExtendedQuantType.QBFloat16,
   )
   config = Config(global_quant_config=quant_config)

   # Create an ONNX quantizer
   quantizer = ModelQuantizer(config)

   # Quantize the ONNX model. Users need to provide the input model path, output model path,
   # and a data reader for calibration.
   quantizer.quantize_model(input_model_path, output_model_path, data_reader)


You can also assign BFloat16 to activations while BFP16 to weights as follows:

.. code-block:: python

   quant_config = QuantizationConfig(
       calibrate_method=CalibrationMethod.MinMax,
       quant_format=ExtendedQuantFormat.QDQ,
       activation_type=ExtendedQuantType.QBloat16,
       weight_type=ExtendedQuantType.QBFP,
   )

- **Layer-wise**

This is one of the common configurations for deploying models on hardware devices, where the computationally intensive layers are quantized into BFP16 to maintain accuracy while improving computational efficiency, and the remaining layers are quantized into BFloat16.  


.. code-block:: python

   quant_config = QuantizationConfig(
       calibrate_method=CalibrationMethod.MinMax,
       quant_format=ExtendedQuantFormat.QDQ,
       activation_type=ExtendedQuantType.QBloat16,
       weight_type=ExtendedQuantType.QBloat16,
       include_auto_mp=True,
       extra_options={
           "AutoMixprecision": {
               "TargetOpType": ["Conv", "ConvTranspose", "Gemm", "MatMul"],
               "TargetQuantType": ExtendedQuantType.QBFP,
           },
       },
   )

At this point, there are many tensors on the precision boundary whose consumers have different precision from the producers.
Some backend compilers require that two types of quantization nodes exist simultaneously on these tensors, such as inserting
a BFP node for BFP16 and custom QDQ pair for BF16 onto the same tensor. In this case, you can enable the ``DualQuantNodes`` option.

.. code-block:: python

   quant_config = QuantizationConfig(
       calibrate_method=CalibrationMethod.MinMax,
       quant_format=ExtendedQuantFormat.QDQ,
       activation_type=ExtendedQuantType.QBloat16,
       weight_type=ExtendedQuantType.QBloat16,
       include_auto_mp=True,
       extra_options={
           "AutoMixprecision": {
               "TargetOpType": ["Conv", "ConvTranspose", "Gemm", "MatMul"],
               "TargetQuantType": ExtendedQuantType.QBFP,
               "DualQuantNodes": True,
           },
       },
   )

And we can also mix BF16 with MXINT8 as shown below. Please note that for other Microscaling data formats, you need to set MXAttributes
to the parameter "extra_options", see the Microscaling tutorial for details.

.. code-block:: python

   quant_config = QuantizationConfig(
       calibrate_method=CalibrationMethod.MinMax,
       quant_format=ExtendedQuantFormat.QDQ,
       activation_type=ExtendedQuantType.QBloat16,
       weight_type=ExtendedQuantType.QBloat16,
       include_auto_mp=True,
       extra_options={
           "AutoMixprecision": {
               "TargetOpType": ["Conv", "ConvTranspose", "Gemm", "MatMul"],
               "TargetQuantType": ExtendedQuantType.QMX,
           },
       },
   )

- **Tensor-wise**

Certain tensors in a neural network are particularly sensitive to quantization, including weight and activation tensors. Applying
appropriate precision for these sensitive tensors can help maintain model accuracy while reaping the benefits of quantization.
Therefore, after identifying these tensors through sensitivity analysis, you can set the precision separately for these tensors.

.. code-block:: python

   quant_config = QuantizationConfig(
       calibrate_method=CalibrationMethod.MinMax,
       quant_format=ExtendedQuantFormat.QDQ,
       activation_type=ExtendedQuantType.QBFP,
       weight_type=ExtendedQuantType.QBFP,
       specific_tensor_precision=True,
       extra_options={
           # MixedPrecisionTensor is a dictionary in which the key is data type and the value
           # is a list of the names of sensitive tensors.
           "MixedPrecisionTensor": {
               ExtendedQuantType.QBFloat16: ['weight_tensor_name', 'activation_tensor_name'],
           },
       },
   )

You can also assign more data types to more tensors as needed, for example:

.. code-block:: python

   quant_config = QuantizationConfig(
       calibrate_method=CalibrationMethod.MinMax,
       quant_format=ExtendedQuantFormat.QDQ,
       activation_type=ExtendedQuantType.QBFP,
       weight_type=ExtendedQuantType.QBFP,
       specific_tensor_precision=True,
       extra_options={
           # MixedPrecisionTensor is a dictionary in which the key is data type and the value
           # is a list of the names of sensitive tensors.
           "MixedPrecisionTensor": {
               ExtendedQuantType.QBFloat16: ['weight_tensor_name1', 'activation_tensor_name1'],
               ExtendedQuantType.QInt16: ['weight_tensor_name2', 'activation_tensor_name2'],
           },
       },
   )

Automatic Mixed Precision based on Sensitivity Analysis
--------------------------------------------------------

The previous examples are manually specified mixed precision, but in the practical applications automatically identifying sensitive layers and then
applying mixed precision becomes more critical.

AMD Quark for ONNX supports automatic mixed precision as follows:

**Step 1** Sensitivity analysis. This step can involve profiling the model with a new precision settings and measuring the impact on accuracy.

**Step 2** Sort layers by sensitivity. Layers that show significant accuracy degradation when quantized are deemed "sensitive" and are kept at higher
precision. Less sensitive parts can be quantized more aggressively to lower precision without significant impact on overall model performance.

**Step 3** Perform mixed precision operations. Perform layer by layer until reach the accuracy target which is specified by users.

We provide two types of accuracy target: general L2 Norm metric and Top1 metric specific to image classification models. Here is a simple example of
how to use the L2 Norm metric to achieve automatic mixed precision:

.. code-block:: python

   from quark.onnx import ModelQuantizer, CalibrationMethod, QuantType, ExtendedQuantFormat, ExtendedQuantType
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   # Build the configuration
   quant_config = QuantizationConfig(
       calibrate_method=CalibrationMethod.MinMax,
       quant_format=ExtendedQuantFormat.QDQ,
       activation_type=ExtendedQuantType.QInt16,
       weight_type=QuantType.QInt8,
       include_auto_mp=True,
       extra_options={
           'AutoMixprecision': {
               "TargetOpType": ["Conv", "ConvTranspose", "Gemm", "MatMul"],  # The operation types to perform mixed precision
               "ActTargetQuantType": QuantType.QInt8,  # The activation input of insensitive layers will be assign to this precision
               "WeightTargetQuantType": QuantType.QInt8,  # The weight input of insensitive layers will be assign to this precision
               "OutputIndex": 0,  # The index of outputs for evaluating accuracy indicator
               "L2Target": 0.1,  # If L2 is less than this value after assigning a new precision to a certain layer, the process continues
           },
       },
   )
   config = Config(global_quant_config=quant_config)

   # Create an ONNX quantizer
   quantizer = ModelQuantizer(config)

   # Quantize the ONNX model. Users need to provide the input model path, output model path,
   # and a data reader for calibration.
   quantizer.quantize_model(input_model_path, output_model_path, data_reader)

For a detailed example of using Top1 metric for mixed precision, refer to the :doc:`Mixed Precision Example <example_quark_onnx_mixed_precision>`.

.. raw:: html

   <!-- omit in toc -->

License
-------

Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
