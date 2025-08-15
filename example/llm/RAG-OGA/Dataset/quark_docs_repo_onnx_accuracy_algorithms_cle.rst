Quantizing Using CrossLayerEqualization (CLE)
=============================================

CrossLayerEqualization (CLE) can equalize the weights of consecutive convolution layers, making the model weights easier to perform per-tensor quantization. Experiments show that using the CLE technique can improve the PTQ accuracy of some models, especially for models with depthwise_conv layers, such as Mobilenet. Here is a sample showing how to enable CLE using `quark.onnx`.

.. code-block:: python

    from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
    from quark.onnx.quantization.config.config import Config, QuantizationConfig

    quant_config = QuantizationConfig(
        quant_format=QuantFormat.QDQ,
        calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        enable_npu_cnn=True,
        include_cle=True,
        extra_options={
            'ActivationSymmetric': True,
            'ReplaceClip6Relu': True,
            'CLESteps': 1,
            'CLEScaleAppendBias': True,
        },
    )
    config = Config(global_quant_config=quant_config)

    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)

Arguments
---------

- **include_cle**: (Boolean) This parameter is a flag that determines whether to optimize the models using CrossLayerEqualization; it can improve the accuracy of some models. The default is True.

- **extra_options**: (Dictionary or None) Contains key-value pairs for various options in different cases. Options related to CLE are:

  - **ReplaceClip6Relu**: (Boolean) If True, Replace Clip(0,6) with Relu in the model. The default value is False.

  - **CLESteps**: (Int) Specifies the steps for CrossLayerEqualization execution when include_cle is set to true. The default is 1. When set to -1, adaptive CrossLayerEqualization steps are conducted. The default value is 1.

  - **CLEScaleAppendBias**: (Boolean) Whether the bias is included when calculating the scale of the weights. The default value is True.

Example
=======

.. note::

   For information on accessing AMD Quark ONNX examples, refer to :doc:`Accessing ONNX Examples <../onnx_examples>`.
   This example and the relevant files are available at ``/onnx/accuracy_improvement/cle``

This :doc:`example <../example_quark_onnx_cle>` demonstrates quantizing a resnet152 model using the AMD Quark ONNX quantizer.
