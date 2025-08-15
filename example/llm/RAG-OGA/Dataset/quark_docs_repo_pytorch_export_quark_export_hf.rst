Hugging Face format (safetensors format)
========================================

Hugging Face format (safetensors format) is an optional exporting format for Quark, and the file list of this exporting format is the same as the file list of the original Hugging Face model, with quantization information added to these files. Taking the llama2-7b model as an example, the exported file list and added information are as below:

+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| File name                    | Additional Quantization Information                                                                                 |
+==============================+=====================================================================================================================+
| config.json                  | Original configuration, with quantization configuration added in a ``"quantization_config"`` key                    |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| generation_config.json       | \-                                                                                                                  |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| model*.safetensors           | Quantized checkpoint (weights, scaling factors, zero points)                                                        |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| model.safetensors.index.json | Mapping of weights names to safetensors files, in case the model weights are sharded into multiple files (optional) |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| special_tokens_map.json      | \-                                                                                                                  |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| tokenizer_config.json        | \-                                                                                                                  |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| tokenizer.json               | \-                                                                                                                  |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+

Exporting to Hugging Face format (safetensors format)
-----------------------------------------------------

Here is an example of how to export to Hugging Face format (safetensors format) a Quark model using :py:meth:`.ModelExporter.export_safetensors_model`:

.. code-block:: python

   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
   from quark.torch.export.api import ModelExporter
   from quark.torch.quantization.api import ModelQuantizer
   from quark.torch.quantization.config.config import Int8PerTensorSpec, QuantizationConfig, Config

   from transformers import AutoModelForCausalLM

   quant_spec = Int8PerTensorSpec(
      observer_method="min_max",
      symmetric=True,
      scale_type="float",
      round_method="half_even",
      is_dynamic=False
   ).to_quantization_spec()

   global_quant_config = QuantizationConfig(weight=quant_spec)
   quant_config = Config(global_quant_config=global_quant_config)

   export_config = ExporterConfig(
      json_export_config=JsonExporterConfig(weight_format="real_quantized")
   )

   model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

   quantizer = ModelQuantizer(quant_config)
   quantized_model = quantizer.quantize_model(model, dataloader=None)
   quantized_model = quantizer.freeze(quantized_model)

   model_exporter = ModelExporter(
      config=export_config,
      export_dir="./opt-125m-quantized"
   )
   model_exporter.export_safetensors_model(
      model=quantized_model,
      quant_config=quant_config
   )

By default, :py:meth:`.ModelExporter.export_safetensors_model` exports models with |save_pretrained|_ using a Quark-specific format for the checkpoint and ``"quantization_config"`` key in the ``config.json`` file. This format may not directly be usable by some downstream libraries (AutoAWQ, vLLM).

.. |save_pretrained| replace:: ``model.save_pretrained()``
.. _save_pretrained: https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained

Until downstream libraries support Quark quantized models, one may export models so that the weight checkpoint and ``config.json`` file targets a specific downstream libraries, using ``custom_mode="awq"`` or ``custom_mode="fp8"``. Example:

.. code-block:: python

   # `custom_mode="awq"` would e.g. use `qzeros` instead of `weight_zero_point`, `qweight` instead of `weight` in the checkpoint.
   # Moreover, the `quantization_config` in the `config.json` file is custom, and the full quark `Config` is not serialized.
   model_exporter.export_safetensors_model(
      model,
      quant_config=quant_config,
      custom_mode="awq"
   )

In the ``config.json``, such an export results in using ``"quant_method": "awq"``, that can e.g. be loaded through `AutoAWQ <https://github.com/casper-hansen/AutoAWQ>`__ in `Transformers library <https://huggingface.co/docs/transformers/main/en/quantization/awq#awq>`__.

Loading quantized models saved in Hugging Face format (safetensors format)
--------------------------------------------------------------------------

Quark provides the importing function for HF format export files. In other words, these files can be reloaded into Quark. After reloading, the weights of the quantized operators in the model are stored in the real_quantized format.

Currently, this importing function supports weight-only, static, and dynamic quantization for FP8, INT8/UINT8, FP4, INT4/UINT, AWQ and GPTQ.

Here is an example of how to load a serialized quantized model from a folder containing the model (as ``*.safetensors``) and its artifacts (``config.json``, etc.), using :py:meth:`.ModelImporter.import_model_info`:

.. code-block:: python

   from quark.torch.export.api import ModelImporter
   from transformers import AutoConfig, AutoModelForCausalLM
   import torch

   model_importer = ModelImporter(
      model_info_dir="./opt-125m-quantized",
      saved_format="safetensors"
   )

   # We only need the backbone/architecture of the original model,
   # not its weights, as weights are loaded from the quantized checkpoint.
   config = AutoConfig.from_pretrained("facebook/opt-125m")
   with torch.device("meta"):
      original_model = AutoModelForCausalLM.from_config(config)

   quantized_model = model_importer.import_model_info(original_model)
