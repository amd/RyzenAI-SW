.. raw:: html

   <!--
   ## License
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->

BFP16 (Block floating point) Quantization
=========================================

Introduction
------------

In this tutorial, you learn how to use the BFP16 data type with AMD Quark.

BFP is short for Block Floating Point. A floating-point number consists of one sign bit, eight exponent bits, and 23 mantissa bits. The main idea of Block Floating Point is a block of numbers sharing one exponent, and the mantissa of each number shifts right accordingly.

This `paper <https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf>`__ introduces an attempt to apply BFP to deep neural networks (DNNs). BFP16 is widely used across the AI industry. The definition of BFP16 in AMD Quark is a block consisting of eight numbers, the shared exponent consisting of eight bits, and the rest of each number consisting of one sign bit and seven mantissa bits.


How to use BFP16 in AMD Quark
-----------------------------


1. Install AMD Quark
~~~~~~~~~~~~~~~~~~~~

Follow the steps in the :doc:`installation guide <../install>` to install AMD Quark.

2. Set the model:
~~~~~~~~~~~~~~~~~

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
   model.eval()
   tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

The model is retrieved from `Hugging Face <https://huggingface.co/>`__ using their `Transformers <https://huggingface.co/docs/transformers/index>`__
library. The ``facebook/opt-125m`` model is used as an example.

3. Set the quantization configuration:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
   from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig
   from quark.torch.quantization.observer.observer import PerBlockBFPObserver
   DEFAULT_BFP16_PER_BLOCK = QuantizationSpec(dtype=Dtype.int8,
                                              symmetric=True,
                                              observer_cls=PerBlockBFPObserver, # for BFP16 the observer_cls is always PerBlockBFPObserver
                                              qscheme=QSchemeType.per_group, # for BFP16 the qscheme is always QSchemeType.per_group
                                              is_dynamic=True, # this controls whether static or dynamic quantization is performed
                                              ch_axis=-1,
                                              scale_type=ScaleType.float,
                                              group_size=8,
                                              round_method=RoundType.half_even)

   DEFAULT_W_BFP16_PER_BLOCK_CONFIG = QuantizationConfig(weight=DEFAULT_BFP16_PER_BLOCK)
   quant_config = Config(global_quant_config=DEFAULT_W_BFP16_PER_BLOCK_CONFIG)

In AMD Quark, the one sign bit and seven mantissa bits are stored as a single ``int8``, so the ``dtype`` is ``Dtype.int8``. The observer class ``PerBlockBFPObserver`` is used for shared exponent calculation.


4. Do quantization
~~~~~~~~~~~~~~~~~~

To perform quantization, initialize a ``ModelQuantizer`` with the ``quant_config`` constructed above and call the method ``quantize_model``:

.. code:: python

   from quark.torch import ModelQuantizer
   from torch.utils.data import DataLoader
   import torch
   calib_dataloader = DataLoader(torch.randint(0, 1000, (1, 64))) # Using random inputs is for demonstration purpose only
   quantizer = ModelQuantizer(quant_config)
   quant_model = quantizer.quantize_model(model, calib_dataloader)

In practice, users should construct meaningful calibration datasets.

How BFP16 works in AMD Quark
----------------------------


Quantizing a floating-point tensor to a BFP16 tensor consists of three main steps:

1. Obtaining the shared exponent.
2. Shifting mantissas right accordingly.
3. Performing rounding on the mantissas.

The maximum exponent in each block is used as the shared exponent. Then, the mantissa of each element is shifted right accordingly. Note that in BFP, the implicit one is included in the mantissa. Finally, rounding is performed and the trailing mantissa bits are removed. Currently, only the rounding method ``half_to_even`` is supported.
