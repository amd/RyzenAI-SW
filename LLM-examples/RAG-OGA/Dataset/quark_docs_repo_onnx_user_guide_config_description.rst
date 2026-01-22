Configuring ONNX Quantization
=============================

Configuration of quantization in ``AMD Quark for ONNX`` is set by Python ``dataclass`` because it is rigorous and can help you avoid typos. We provide a class ``Config`` in ``quark.onnx.quantization.config.config`` for configuration, as demonstrated in the previous example. In ``Config``, you should set certain instances (all instances are optional except ``global_quant_config``):

-  ``global_quant_config``: Global quantization  configuration applied to the entire model.

The ``Config`` should be like:

.. code-block:: python

   from quark.onnx.quantization.config import Config, get_default_config
   config = Config(global_quant_config=...)

We define some default global configurations, including ``XINT8`` and ``U8S8_AAWS``, which can be used like this:

.. code-block:: python

   quant_config = get_default_config("U8S8_AAWS")
   config = Config(global_quant_config=quant_config)

More Quantization Default Configurations
----------------------------------------

AMD Quark for ONNX provides you with default configurations to quickly start model quantization.

-  ``INT8_CNN_DEFAULT``: Perform 8-bit, optimized for CNN quantization.
-  ``INT16_CNN_DEFAULT``: Perform 16-bit, optimized for CNN quantization.
-  ``INT8_TRANSFORMER_DEFAULT``: Perform 8-bit, optimized for transformer quantization.
-  ``INT16_TRANSFORMER_DEFAULT``: Perform 16-bit, optimized for transformer quantization.
-  ``INT8_CNN_ACCURATE``: Perform 8-bit, optimized for CNN quantization. Some advanced algorithms are applied to achieve higher accuracy but consume more time and memory space.
-  ``INT16_CNN_ACCURATE``: Perform 16-bit, optimized for CNN quantization. Some advanced algorithms are applied to achieve higher accuracy but consume more time and memory space.
-  ``INT8_TRANSFORMER_ACCURATE``: Perform 8-bit, optimized for transformer quantization. Some advanced algorithms are applied to achieve higher accuracy but consume more time and memory space.
-  ``INT16_TRANSFORMER_ACCURATE``: Perform 16-bit, optimized for transformer quantization. Some advanced algorithms are applied to achieve higher accuracy but consume more time and memory space.

AMD Quark for ONNX also provides more advanced default configurations to help you quantize models with more options.

-  ``UINT8_DYNAMIC_QUANT``: Perform dynamic activation, uint8 weight quantization.
-  ``XINT8``: Perform uint8 activation, int8 weight, optimized for NPU quantization.
-  ``XINT8_ADAROUND``: Perform uint8 activation, int8 weight, optimized for NPU quantization. The adaround fast finetune applies to preserve quantized accuracy.
-  ``XINT8_ADAQUANT``: Perform uint8 activation, int8 weight, optimized for NPU quantization. The adaquant fast finetune applies to preserve quantized accuracy.
-  ``S8S8_AAWS``: Perform int8 asymmetric activation, int8 symmetric weight quantization.
-  ``S8S8_AAWS_ADAROUND``: Perform int8 asymmetric activation, int8 symmetric weight quantization. The adaround fast finetune applies to preserve quantized accuracy.
-  ``S8S8_AAWS_ADAQUANT``: Perform int8 asymmetric activation, int8 symmetric weight quantization. The adaquant fast finetune applies to preserve quantized accuracy.
-  ``U8S8_AAWS``: Perform uint8 asymmetric activation int8 symmetric weight quantization.
-  ``U8S8_AAWS_ADAROUND``: Perform uint8 asymmetric activation, int8 symmetric weight quantization. The adaround fast finetune applies to preserve quantized accuracy.
-  ``U8S8_AAWS_ADAQUANT``: Perform uint8 asymmetric activation, int8 symmetric weight quantization. The adaquant fast finetune applies to preserve quantized accuracy.
-  ``S16S8_ASWS``: Perform int16 symmetric activation, int8 symmetric weight quantization.
-  ``S16S8_ASWS_ADAROUND``: Perform int16 symmetric activation, int8 symmetric weight quantization. The adaround fast finetune applies to preserve quantized accuracy.
-  ``S16S8_ASWS_ADAQUANT``: Perform int16 symmetric activation, int8 symmetric weight quantization. The adaquant fast finetune applies to preserve quantized accuracy.
-  ``A8W8``: Perform int8 symmetric activation, int8 symmetric weight quantization and optimize for deployment.
-  ``A16W8``: Perform int16 symmetric activation, int8 symmetric weight quantization and optimize for deployment.
-  ``U16S8_AAWS``: Perform uint16 asymmetric activation, int8 symmetric weight quantization.
-  ``U16S8_AAWS_ADAROUND``: Perform uint16 asymmetric activation, int8 symmetric weight quantization. The adaround fast finetune applies to preserve quantized accuracy.
-  ``U16S8_AAWS_ADAQUANT``: Perform uint16 asymmetric activation, int8 symmetric weight quantization. The adaquant fast finetune applies to preserve quantized accuracy.
-  ``BF16``: Perform BFloat16 activation, BFloat16 weight quantization.
-  ``BFP16``: Perform BFP16 activation, BFP16 weight quantization.
-  ``S16S16_MIXED_S8S8``: Perform int16 activation, int16 weight mix-precision quantization.

Customized Configurations
-------------------------

Besides the default configurations in AMD Quark for ONNX, you can also customize the quantization configuration like the following example:

.. toctree::
   :hidden:
   :caption: Advanced AMD Quark Features for PyTorch
   :maxdepth: 1

   Full List of Quantization Config Features <appendix_full_quant_config_features>

.. code-block:: python

   from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(
       quant_format=quark.onnx.QuantFormat.QDQ,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       input_nodes=[],
       output_nodes=[],
       op_types_to_quantize=[],
       per_channel=False,
       reduce_range=False,
       activation_type=quark.onnx.QuantType.QInt8,
       weight_type=quark.onnx.QuantType.QInt8,
       nodes_to_quantize=[],
       nodes_to_exclude=[],
       subgraphs_to_exclude=[],
       optimize_model=True,
       use_dynamic_quant=False,
       use_external_data_format=False,
       execution_providers=['CPUExecutionProvider'],
       enable_npu_cnn=False,
       enable_npu_transformer=False,
       convert_fp16_to_fp32=False,
       convert_nchw_to_nhwc=False,
       include_cle=True,
       include_sq=False,
       extra_options={},)
   config = Config(global_quant_config=quant_config)

   quantizer = ModelQuantizer(config)
   quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)

.. toctree::
   :hidden:
   :maxdepth: 1

   Calibration methods <config/calibration_methods.rst>
   Calibration datasets <config/calibration_datasets.rst>
   Quantization Strategies <config/quantization_strategies.rst>
   Quantization Schemes <config/quantization_schemes.rst>
   Quantization Symmetry <config/quantization_symmetry.rst>

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
