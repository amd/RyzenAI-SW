GGUF Exporting
==============

Currently, only support asymmetric int4 per_group weight-only
quantization, and the group_size must be 32.The models supported include
Llama2-7b, Llama2-13b, Llama2-70b, and Llama3-8b.

Example of GGUF Exporting
-------------------------

.. code:: python

   export_path = "./output_dir"
   from quark.torch import ModelExporter
   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
   export_config = ExporterConfig(json_export_config=JsonExporterConfig())
   exporter = ModelExporter(config=export_config, export_dir=export_path)
   exporter.export_gguf_model(model, tokenizer_path, model_type)

After running the code above successfully, there will be a ``.gguf``
file under export_path, ``./output_dir/llama.gguf`` for example.

.. toctree::
   :hidden:
   :maxdepth: 1

   gguf_llamacpp.rst
