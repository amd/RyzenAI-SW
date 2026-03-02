# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\quark_quantization\docs\advanced_quant_readme.mdx:64
INT8_CNN_ACCURATE_CONFIG = QuantizationConfig(calibrate_method=CalibrationMethod.Percentile,
                                              activation_type=QuantType.QUInt8,
                                              weight_type=QuantType.QInt8,
                                              include_fast_ft=True,
                                              extra_options={
                                                  'Percentile': 99.9999,
                                                  'FastFinetune': DEFAULT_ADAROUND_PARAMS
                                              })
config = Config(global_quant_config=INT8_CNN_ACCURATE_CONFIG)
