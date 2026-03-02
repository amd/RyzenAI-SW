# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\model-quantization.mdx:64
quant_config = QuantizationConfig(calibrate_method=PowerOfTwoMethod.MinMSE,
                                  activation_type=QuantType.QUInt8,
                                  weight_type=QuantType.QInt8,
                                  enable_npu_cnn=True,
                                  extra_options={'ActivationSymmetric': True})
config = Config(global_quant_config=quant_config)
print("The configuration of the quantization is {}".format(config))
