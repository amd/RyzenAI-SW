##################
Model Quantization
##################

**Model quantization** is the process of mapping high-precision weights/activations to a lower precision format, such as BF16/INT8, while maintaining model accuracy. This technique enhances the computational and memory efficiency of the model for deployment on NPU devices. It can be applied post-training, allowing existing models to be optimized without the need for retraining.

The Ryzen AI compiler supports input models quantized to either INT8 or BF16 format:

- CNN models: INT8 or BF16
- Transformer models: BF16

Quantization introduces several challenges, primarily revolving around the potential drop in model accuracy. Choosing the right quantization parameters—such as data type, bit-width, scaling factors, and the decision between per-channel or per-tensor quantization—adds layers of complexity to the design process.

*********
AMD Quark
*********

**AMD Quark** is a comprehensive cross-platform deep learning toolkit designed to simplify and enhance the quantization of deep learning models. Supporting both PyTorch and ONNX models, Quark empowers developers to optimize their models for deployment on a wide range of hardware backends, achieving significant performance gains without compromising accuracy.

For more challenging model quantization needs **AMD Quark** supports advanced quantization technique like **Fast Finetuning** that helps recover the lost accuracy of the quantized model. 

Documentation
=============
The complete documentation for AMD Quark for Ryzen AI can be found here: https://quark.docs.amd.com/latest/supported_accelerators/ryzenai/index.html


INT8 Examples
=============
**AMD Quark** provides default configrations that support INT8 quantization configuration. For example, `XINT8` uses symmetric INT8 activation and weights quantization with power-of-two scales using the MinMSE calibration method.
The quantization configuration can be customized using the `QuantizationConfig` class. The following example shows how to set up the quantization configuration for INT8 quantization:

.. code-block::

   quant_config = QuantizationConfig(calibrate_method=PowerOfTwoMethod.MinMSE,
                                     activation_type=QuantType.QUInt8,
                                     weight_type=QuantType.QInt8,
                                     enable_npu_cnn=True,
                                     extra_options={'ActivationSymmetric': True})
   config = Config(global_quant_config=quant_config)
   print("The configuration of the quantization is {}".format(config))

The user can use the `get_default_config('XINT8')` function to get the default configuration for INT8 quantization.

For more details
~~~~~~~~~~~~~~~~
- `AMD Quark Tutorial <https://github.com/amd/RyzenAI-SW/tree/main/tutorial/quark_quantization>`_ for Ryzen AI Deployment
- Running INT8 model on NPU using :doc:`Getting Started Tutorial <getstartex>`
- Advanced quantization techniques `Fast Finetuning and Cross Layer Equalization <https://gitenterprise.xilinx.com/VitisAI/RyzenAI-SW/blob/dev/tutorial/quark_quantization/docs/advanced_quant_readme.md>`_ for INT8 model


BF16 Examples
=============
**AMD Quark** provides default configrations that support BFLOAT16 (BF16) model quantization. For example, BF16 is a 16-bit floating-point format designed to have same exponent size as FP32, allowing a wide dynamic range, but with reduced precision to save memory and speed up computations.
The BFLOAT16 (BF16) model needs to be converted from QDQ nodes to Cast operations to run with VAIML compiler. AMD Quark support this conversion with the configuration option `BF16QDQToCast`.

.. code-block::

   quant_config = get_default_config("BF16")
   quant_config.extra_options["BF16QDQToCast"] = True
   config = Config(global_quant_config=quant_config)
   print("The configuration of the quantization is {}".format(config))

For more details
~~~~~~~~~~~~~~~~
- `Image Classification <https://github.com/amd/RyzenAI-SW/tree/main/example/image_classification>`_ using ResNet50 to run BF16 model on NPU
- `Finetuned DistilBERT for Text Classification <https://github.com/amd/RyzenAI-SW/tree/main/example/DistilBERT_text_classification_bf16>`_ 
- `Text Embedding Model Alibaba-NLP/gte-large-en-v1.5  <https://github.com/amd/RyzenAI-SW/tree/main/example/GTE>`_ 
- Advanced quantization techniques `Fast Finetuning <https://quark.docs.amd.com/latest/supported_accelerators/ryzenai/tutorial_convert_fp32_or_fp16_to_bf16.html>`_ for BF16 models.


..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
