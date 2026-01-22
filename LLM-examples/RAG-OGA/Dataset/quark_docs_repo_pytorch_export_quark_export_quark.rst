Quark Format
============

Quark Format Exporting
----------------------

.. note::
   For most use cases with external open-source libraries (Transformers, vLLM, etc.), the serialization format described on this page should not be used, and you should refer to the :doc:`safetensors format <./quark_export_hf>` instead.

Quark format is a proprietary export format for Quark, and the file list of
this exporting format contains the quantized parameters (in a ``model_state_dict.pth`` file) such as weight, scale, and zero point and config.json with quantization  configuration.

Note that this model currently only supports exporting linear parts (which is sufficient for general large language modeling)
For other needs using quark export (e.g., exporting embedding layers, convolutional layers), use `Saving & Loading` below.
In fact, we are gradually migrating the `save and load` functionality to ``ModelExporter`` in `quark format`.

Example of Quark Format Exporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
   from quark.torch.export.api import ModelExporter

   json_export_config = JsonExporterConfig(
      weight_format="real_quantized",
      pack_method="reorder"
   )
   export_config = ExporterConfig(json_export_config=json_export_config)

   exporter = ModelExporter(config=export_config, export_dir="./exported_model_dir")
   exporter.export_quark_model(model, quant_config=quant_config)

By default, ``ModelExporter.export_quark_model`` exports models using a Quark-specific format for the checkpoint and ``quantization_config`` format in the ``config.json`` file.

This format may not directly be usable by some downstream libraries (vLLM) until downstream libraries support Quark quantized models. But it can be loaded and used by quark itself.

This format supports two forms of weight saving, ``fake quantized`` will save the high precision weight after quantization , while ``real quantized`` will save the weights after the real quantization. You can configure this with ``weight_format``.

.. code:: python

   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
   from quark.torch.export.api import ModelExporter

   json_export_config = JsonExporterConfig(weight_format="real_quantized", pack_method="reorder")
   export_config = ExporterConfig(json_export_config=json_export_config)

   exporter = ModelExporter(config=export_config, export_dir=args.output_dir)
   exporter.export_quark_model(model, quant_config=quant_config, custom_mode=args.custom_mode)


Quark Format Importing
----------------------

Models exported using quark format can be imported directly using quark. Models exported using quark format can be imported directly using quark. quark chooses how to load the weights based on the information in the config.

Example of Quark Format Importing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from quark.torch import ModelImporter
   
   importer = ModelImporter(model_info_dir=args.import_model_dir)
   model = importer.import_model_info(model)
