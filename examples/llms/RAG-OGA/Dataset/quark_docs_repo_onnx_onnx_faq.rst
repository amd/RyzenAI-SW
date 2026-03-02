Frequently Asked Questions (FAQ)
================================


AMD Quark for ONNX
------------------

Model Issues
~~~~~~~~~~~~

**Issue 1**:

Error of "ValueError:Message onnx.ModelProto exceeds maximum protobuf size of 2GB"

**Solution**:

This error is caused by the input model size exceeding 2GB. Set ``optimize_model=False`` and ``use_external_data_format=True``.

**Issue 2**:

Error of NCHW or NHWC of "index: 1 Got: 244 Expected: 3 index: 2 Got: 3 Expected: 224"

**Solution**:

This error is caused by the calibration data is NCHW and the shape of model input is NHWC. Set ``convert_nchw_to_nhwc=True``. For more detailed information, see :doc:`Tools <tools>`.

Quantization Issues
~~~~~~~~~~~~~~~~~~~

**Issue 1**:

Error of "onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node."

**Solution**:

For networks with an ROI head, such as Mask R-CNN or Faster R-CNN, quantization errors might arise if ROIs are not generated in the network.
Use quark.onnx.PowerOfTwoMethod.MinMSE or quark.onnx.CalibrationMethod.Percentile quantization and perform inference with real data.

Quantization Config Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue 1**:

Does XINT8 refer to INT8? What’s the difference between XINT8 and A8W8?


**Solution**:

XINT8 and A8W8 are both INT8 Quantization. XINT8 and A8W8 are both very common quantization configurations in our Quark ONNX quantizer. A8W8 uses symmetric INT8 activation and weights quantization with float scales. XINT8 uses symmetric INT8 activation and weights quantization with power-of-two scales. A8W8 uses the MinMax method, and XINT8 uses MinMSE to improve quantization precision. XINT8 usually has greater advantages in hardware acceleration. For more detailed information about XINT8, see :doc:`Power-of-Two Scales (XINT8) Quantization <../supported_accelerators/ryzenai/tutorial_xint8_quantize>`. For more details information about A8W8, see :doc:`Float Scales (A8W8 and A16W8) Quantization <../supported_accelerators/ryzenai/tutorial_a8w8_and_a16w8_quantize>`.
.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
