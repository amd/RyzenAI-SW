SmoothQuant (SQ)
================

SmoothQuant (SQ) is another technique used to improve PTQ accuracy. It smooths the outliers of the activation so that it loses as little precision as possible during quantization. Experiments show that using the SQ technique can improve the PTQ accuracy of some models, especially for models with a large number of outliers in the activation. Here is a sample showing how to enable SQ using `quark.onnx`.

.. code-block:: python

    from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
    from quark.onnx.quantization.config.config import Config, QuantizationConfig

    quant_config = QuantizationConfig(
        quant_format=QuantFormat.QDQ,
        calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        enable_npu_cnn=True,
        include_sq=True,
        extra_options={
            'ActivationSymmetric': True,
            'SmoothAlpha': 0.5,
        },
    )
    config = Config(global_quant_config=quant_config)

    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)

Arguments
---------

- **include_sq**: (Boolean) This parameter is a flag that determines whether to optimize the models using SmoothQuant; it can improve the accuracy of some models. The default is False.

- **extra_options**: (Dictionary or None) Contains key-value pairs for various options in different cases. Options related to SQ are:

  - **SmoothAlpha**: (Float) This parameter controls how much difficulty we want to migrate from activation to weights. The default value is 0.5.

Example
-------

.. note::

   For information on accessing AMD Quark ONNX examples, refer to :doc:`Accessing ONNX Examples <../onnx_examples>`.
   This example and the relevant files are available at ``/onnx/accuracy_improvement/smooth_quant``

This :doc:`example <../example_quark_onnx_smoothquant>` demonstrates quantizing an opt-125m model using the AMD Quark ONNX quantizer. It also shows how to use the Smooth Quant algorithm.
