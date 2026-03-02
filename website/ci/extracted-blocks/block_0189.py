# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\quark_quantization\docs\advanced_quant_readme.mdx:112
INT8_CLE_CONFIG = QuantizationConfig(calibrate_method=PowerOfTwoMethod.MinMSE,
                                    activation_type=QuantType.QUInt8,
                                    weight_type=QuantType.QInt8,
                                    enable_npu_cnn=True,
                                    include_cle=True,
                                    extra_options={'ActivationSymmetric': True})

config = Config(global_quant_config=INT8_CLE_CONFIG)
