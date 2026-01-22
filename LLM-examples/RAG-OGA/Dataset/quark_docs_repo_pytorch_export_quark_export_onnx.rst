ONNX Exporting
==============

PyTorch provides a function to export the ONNX graph at this
`link <https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export>`__.
Quark supports the export of onnx graph for int4, int8, fp8 , float16 and
bfloat16 quantized models. For int4, int8, and fp8 quantization, the
quantization operators used in onnx graph are
`QuantizerLinear <https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html>`__\ \_\ `DequantizerLinear <https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html>`__
pair. For float16 and bfloat16 quantization, the quantization operators
are the cast_cast pair. Mix quantization of int4/uint4 and int8/uint8 is
not supported currently. In other words, if the model contains both
quantized nodes of uint4/int4 and uint8/int8, this function cannot be
used to export the ONNX graph
Only support weight-only and static quantization for now.

Example of Onnx Exporting
-------------------------

.. code:: python


   export_path = "./output_dir"
   batch_iter = iter(calib_dataloader)
   input_args = next(batch_iter)
   if args.quant_scheme in ["w_int4_per_channel_sym", "w_uint4_per_group_asym", "w_int4_per_group_sym", "w_uint4_a_bfloat16_per_group_asym"]:
       uint4_int4_flag = True
   else:
       uint4_int4_flag = False

   from quark.torch import ModelExporter
   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
   export_config = ExporterConfig(json_export_config=JsonExporterConfig())
   exporter = ModelExporter(config=export_config, export_dir=export_path)
   exporter.export_onnx_model(model, input_args, uint4_int4_flag=uint4_int4_flag)
