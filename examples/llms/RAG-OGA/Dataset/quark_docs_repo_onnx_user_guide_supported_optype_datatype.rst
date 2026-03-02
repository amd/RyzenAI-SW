Supported Data and Op Types
===========================

Supported Data Types
--------------------

Summary Table
~~~~~~~~~~~~~

+------------------------------------------------------------------------------+
| Supported Data Types                                                         |
+==============================================================================+
| Int4 / UInt4                                                                 |
+------------------------------------------------------------------------------+
| Int8 / UInt8                                                                 |
+------------------------------------------------------------------------------+
| Int16 / UInt16                                                               |
+------------------------------------------------------------------------------+
| Int32 / UInt32                                                               |
+------------------------------------------------------------------------------+
| Float16                                                                      |
+------------------------------------------------------------------------------+
| BFloat16                                                                     |
+------------------------------------------------------------------------------+
| BFP16                                                                        |
+------------------------------------------------------------------------------+
| MX4 / MX6 / MX9                                                              |
+------------------------------------------------------------------------------+
| MXFP8(E5M2) / MXFP8(E4M3) / MXFP6(E3M2) / MXFP6(E2M3) / MXFP4(E2M1) / MXINT8 |
+------------------------------------------------------------------------------+

You can see in the table there are many non integer data types that onnxruntime official operators do not support. In order to support these new features, we have developed several custom operators using onnxruntime's custom operation C APIs. Here are these ops and their specifications:

**ExtendedQuantizeLinear** - :doc:`specification <custom_operators/ExtendedQuantizeLinear>`

**ExtendedDequantizeLinear** - :doc:`specification <custom_operators/ExtendedDequantizeLinear>`

**BFPQuantizeDequantize** - :doc:`specification <custom_operators/BFPQuantizeDequantize>`

**MXQuantizeDequantize** - :doc:`specification <custom_operators/MXQuantizeDequantize>`

.. note::

   When installing on Windows, Visual Studio is required. The minimum version of Visual Studio is Visual Studio 2022. During the compilation process, there are two ways to use it:

1. **Use the Developer Command Prompt for Visual Studio**
   When installing Visual Studio, ensure that the Developer Command Prompt for Visual Studio is installed as well. Execute programs in the CMD window of the Developer Command Prompt for Visual Studio.

2. **Manually Add Paths to Environment Variables**
   Visual Studio's ``cl.exe``, ``MSBuild.exe``, and ``link.exe`` will be used. Ensure that the paths are added to the `PATH` environment variable. These programs are located in the Visual Studio installation directory. In the *Edit Environment Variables* window, click **New**, then paste the path to the folder containing ``cl.exe``, ``link.exe``, and ``MSBuild.exe``. Click **OK** on all windows to apply the changes.

1. Quantizing to Other Precisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the INT8/UINT8, the quark.onnx supports quantizing models to other data formats, including INT16/UINT16, INT32/UINT32, Float16 and BFloat16, which can provide better accuracy or be used for experimental purposes. These new data formats are achieved by a customized version of QuantizeLinear and DequantizeLinear named "ExtendedQuantizeLinear" and "ExtendedDequantizeLinear", which expand onnxruntime's UInt8 and Int8 quantization to support UInt16, Int16, UInt32, Int32, Float16, and
BFloat16. This customized Q/DQ was implemented by a custom operations library in quark.onnx using onnxruntime's custom operation C API.

The custom operations library was developed based on Linux and Windows.

To use this feature, the ``quant_format`` should be set to ExtendedQuantFormat.QDQ. You might have noticed that in both the recommended NPU_CNN and NPU_Transformer configurations, the ``quant_format`` is set to QuantFormat.QDQ. NPU targets that support acceleration for models quantized to INT8/UINT8, do not support other precisions.

.. note::

     When the Quant_Type is Int4/UInt4, the onnxruntime version must be 1.19.0 or higher. Only the onnxruntime native "CalibrationMethod" is supported (MinMax, Percentile), and the quant_format is required to be QuantFormat.

1.1 Quantizing Float32 Models to Int16 or Int32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quantizer supports quantizing float32 models to Int16 or Int32 data formats. To enable this, you need to set the ``activation_type`` and ``weight_type`` in the quantize_static API to the new data types. Options are ExtendedQuantType.QInt16/ExtendedQuantType.QUInt16 or ExtendedQuantType.QInt32/ExtendedQuantType.QUInt32.

.. code:: python

   quark.onnx.quantize_static(
       model_input,
       model_output,
       calibration_data_reader,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       quant_format=quark.onnx.ExtendedQuantFormat.QDQ,
       activation_type=quark.onnx.ExtendedQuantType.QInt16,
       weight_type=quark.onnx.ExtendedQuantType.QInt16,
   )

1.2 Quantizing Float32 Models to Float16 or BFloat16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Besides integer data formats, the quantizer also supports quantizing Float32 models to Float16 or BFloat16 data formats. Set the ``activation_type`` and ``weight_type`` to ``ExtendedQuantType.QFloat16`` or ``ExtendedQuantType.QBFloat16``.

.. code:: python

   quark.onnx.quantize_static(
       model_input,
       model_output,
       calibration_data_reader,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       quant_format=quark.onnx.ExtendedQuantFormat.QDQ,
       activation_type=quark.onnx.ExtendedQuantType.QFloat16,
       weight_type=quark.onnx.ExtendedQuantType.QFloat16,
   )

1.3 Quantizing Float32 Models to BFP16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quantizer also supports quantizing Float32 models to BFP16 data formats. The block size can be modified by changing the ``block_size`` parameter in the ``extra_options``. The following is the configuration for BFP16 with a block size of 8.

.. code:: python

   quark.onnx.quantize_static(
       model_input,
       model_output,
       calibration_data_reader,
       calibrate_method=quark.onnx.PowerOfTwoMethod.NonOverflow,
       quant_format=quark.onnx.ExtendedQuantFormat.QDQ,
       activation_type=quark.onnx.ExtendedQuantType.QBFP,
       weight_type=quark.onnx.ExtendedQuantType.QBFP,
       extra_options={
           "BFPAttributes": {
               "bfp_method": "to_bfp",
               "bit_width": 16,
               "block_size": 8,
           }
       },
   )

1.4 Quantizing Float32 Models to MXINT8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quantizer also supports quantizing Float32 models to MXINT8 data formats. The block size can be modified by changing the ``block_size`` parameter in the ``extra_options``. The following is the configuration for MXINT8 with a block size of 32.

.. code:: python

   quark.onnx.quantize_static(
       model_input,
       model_output,
       calibration_data_reader,
       calibrate_method=quark.onnx.PowerOfTwoMethod.NonOverflow,
       quant_format=quark.onnx.ExtendedQuantFormat.QDQ,
       activation_type=quark.onnx.ExtendedQuantType.QMX,
       weight_type=quark.onnx.ExtendedQuantType.QMX,
       extra_options={
           "MXAttributes": {
               "element_dtype": "int8",
               "block_size": 32,
           }
       },
   )

.. note::

     When inference with ONNX Runtime, we need to register the custom op's so(Linux) or dll(Windows) file in the ORT session options.

.. code:: python

    import onnxruntime
    from quark.onnx import get_library_path

    device = 'CPU'
    providers = ['CPUExecutionProvider']

    # Also We can use the GPU configuration:
    # device='ROCM'
    # providers = ['ROCMExecutionProvider']
    # device='CUDA'
    # providers = ['CUDAExecutionProvider']

    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path(device))
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)

1.5 Quantizing Float32 Models to Mixed Data Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quantizer even supports setting the activation and weight to different precisions. For example, activation is Int16 while weight is Int8. This can be used when pure Int8 quantization can not meet accuracy requirements.

.. code:: python

   quark.onnx.quantize_static(
       model_input,
       model_output,
       calibration_data_reader,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       quant_format=quark.onnx.ExtendedQuantFormat.QDQ,
       activation_type=quark.onnx.ExtendedQuantType.QInt16,
       weight_type=QuantType.QInt8,
   )

2. Quantizing Float16 Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For models in Float16, we recommend setting ``convert_fp16_to_fp32`` to True. This first converts your Float16 model to a Float32 model before quantization, reducing redundant nodes such as cast in the model.

.. code:: python

   quark.onnx.quantize_static(
       model_input,
       model_output,
       calibration_data_reader,
       quant_format=QuantFormat.QDQ,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       activation_type=QuantType.QUInt8,
       weight_type=QuantType.QInt8,
       enable_NPU_cnn=True,
       convert_fp16_to_fp32=True,
       extra_options={'ActivationSymmetric':True}
   )

.. note::
    When using ``convert_fp16_to_fp32`` in quark.onnx, it requires onnxsim to simplify the ONNX model. Ensure that onnxsim is installed by using ``python -m pip install onnxsim``.

Supported Op Type
-----------------

.. _quark-onnx-supported-ops:

Summary Table
~~~~~~~~~~~~~

Table: List of Quark ONNX Supported Quantized Ops

+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Supported Ops         | Comments                                                                                                                                                                                                  |
+=======================+===========================================================================================================================================================================================================+
| Add                   |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ArgMax                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| AveragePool           | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| BatchNormalization    | By default, the "optimize_model" parameter will fuse BatchNormalization to Conv/ConvTranspose/Gemm. For standalone BatchNormalization, quantization is supported only for NPU_CNN platforms by converting |
|                       | BatchNormalization to Conv.                                                                                                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Clip                  | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Concat                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Conv                  |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ConvTranspose         |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| DepthToSpace          | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Div                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Erf                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Gather                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Gemm                  |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| GlobalAveragePool     |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| HardSigmoid           | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| InstanceNormalization |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| LayerNormalization    | Supported for opset>=17. Will be quantized only when its input is quantized.                                                                                                                              |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| LeakyRelu             |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| LpNormalization       | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| MatMul                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Min                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Max                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| MaxPool               | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Mul                   |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Pad                   |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| PRelu                 | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ReduceMean            | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Relu                  | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Reshape               | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Resize                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Slice                 | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Sigmoid               |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Softmax               |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| SpaceToDepth          | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Split                 |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Squeeze               | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Sub                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Tanh                  | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Transpose             | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Unsqueeze             | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Where                 |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. toctree::
   :hidden:
   :maxdepth: 1

   ExtendedQuantizeLinear <custom_operators/ExtendedQuantizeLinear.rst>
   ExtendedDequantizeLinear <custom_operators/ExtendedDequantizeLinear.rst>
   ExtendedInstanceNormalization <custom_operators/ExtendedInstanceNormalization.rst>
   ExtendedLSTM <custom_operators/ExtendedLSTM.rst>
   BFPQuantizeDequantize <custom_operators/BFPQuantizeDequantize.rst>
   MXQuantizeDequantize <custom_operators/MXQuantizeDequantize.rst>

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
