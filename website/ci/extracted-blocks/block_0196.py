# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\quark_quantization\README.mdx:122
from quark.onnx.quantization.config import Config, get_default_config
from quark.onnx.quantization.config.config import QuantizationConfig
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType, QuantFormat

# Use default quantization configuration
quant_config = get_default_config("XINT8")

# Define custom quantization configuration
# quant_config = QuantizationConfig(calibrate_method=PowerOfTwoMethod.MinMSE,
#                                   activation_type=QuantType.QUInt8,
#                                   weight_type=QuantType.QInt8,
#                                   enable_npu_cnn=True,
#                                   extra_options={'ActivationSymmetric': True})

# Defines the quantization configuration for the whole model
config = Config(global_quant_config=quant_config)
print("The configuration of the quantization is {}".format(config))
