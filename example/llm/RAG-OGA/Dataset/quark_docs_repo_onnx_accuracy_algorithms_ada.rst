Quantization Using AdaQuant and AdaRound
========================================

.. note::

    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

.. note::

   For information on accessing AMD Quark ONNX examples, refer to :doc:`Accessing ONNX Examples <../onnx_examples>`.
   These examples and the relevant files are available at ``/onnx/accuracy_improvement/adaquant`` and ``/onnx/accuracy_improvement/adaround``.

Fast Finetune
-------------

Fast finetune improves the quantized model's accuracy by training the output of each layer as close as possible to the floating-point model. It includes two practical algorithms: "AdaRound" and "AdaQuant". Applying fast finetune might achieve better accuracy for some models but takes much longer time than normal PTQ. It is disabled by default to save quantization time but can be turned on if you encounter accuracy issues. If this feature is enabled, `quark.onnx` will require the PyTorch package.

.. code-block:: python

    from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
    from quark.onnx.quantization.config.config import Config, QuantizationConfig

    quant_config = QuantizationConfig(
        quant_format=QuantFormat.QDQ,
        calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        enable_npu_cnn=True,
        include_fast_ft=True,
        extra_options={
            'ActivationSymmetric': True,
            'FastFinetune': {
                'OptimAlgorithm': 'adaround',
                'OptimDevice': 'cpu',
                'BatchSize': 1,
                'NumIterations': 1000,
                'LearningRate': 0.1,
            },
        },
    )
    config = Config(global_quant_config=quant_config)

    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)

Arguments
~~~~~~~~~

- **include_fast_ft**: (Boolean) This parameter is a flag that determines whether to optimize the models using Fast Finetune. Set to True to enable fast finetune (default is False).

- **extra_options**: (Dictionary or None) Contains key-value pairs for various options in different cases. Fast finetune related options are packaged within `extra_options` as a member whose key is "FastFinetune" and values are:

  - **OptimAlgorithm**: (String) The specified algorithm for fast finetune. Optional values are "adaround" and "adaquant". "Adaround" adjusts the weight's rounding function, which is relatively stable and might converge faster, while "adaquant" trains the weight directly, potentially offering greater improvement. The default value is "adaround".

  - **OptimDevice**: (String) Specifies the compute device used for PyTorch model training during fast finetuning. Optional values are "cpu" and "cuda:0". The default value is "cpu".

  - **BatchSize**: (Int) Batch size for finetuning. A larger batch size might result in better accuracy but longer training time. The default value is 1.

  - **NumIterations**: (Int) The number of iterations for finetuning. More iterations can lead to better accuracy but also longer training time. The default value is 1000.

  - **LearningRate**: (Float) Learning rate for finetuning. It significantly impacts the improvement of fast finetune, and experimenting with different learning rates might yield better results for your model. The default value is 0.1.

AdaRound
~~~~~~~~

**AdaRound**, short for "Adaptive Rounding," is a post-training quantization technique that aims to minimize the accuracy drop typically associated with quantization. Unlike standard rounding methods, which can be too rigid and cause significant deviations from the original model's behavior, AdaRound uses an adaptive approach to determine the optimal rounding of weights. Here is the `link <https://arxiv.org/abs/2004.10568>`__ to the paper.

AdaQuant
~~~~~~~~

**AdaQuant**, short for "Adaptive Quantization," is an advanced quantization technique designed to minimize the accuracy loss typically associated with post-training quantization. Unlike traditional static quantization methods, which apply uniform quantization across all layers and weights, AdaQuant dynamically adapts the quantization parameters based on the characteristics of the model and its data. Here is the `link <https://arxiv.org/abs/1712.01048>`__ to the paper.

Benefits of AdaRound and AdaQuant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Improved Accuracy**: By minimizing the quantization error, AdaRound helps preserve the model's accuracy closer to its original state. By dynamically adjusting quantization parameters, AdaQuant helps retain a higher level of model accuracy compared to traditional quantization methods.
2. **Flexibility**: AdaRound and AdaQuant can be applied to various layers and types of neural networks, making it a versatile tool for different quantization needs.
3. **Post-Training Application**: AdaRound does not require retraining the model from scratch. It can be applied after the model has been trained, making it a convenient choice for deploying pre-trained models in resource-constrained environments.
4. **Efficiency**: AdaQuant enables the deployment of high-performance models in resource-constrained environments, such as mobile and edge devices, without the need for extensive retraining.

Upgrades of AdaRound / AdaQuant in AMD Quark for ONNX
-----------------------------------------------------

Comparing with the original algorithm, AdaRound in AMD Quark for ONNX is modified and upgraded to be more flexible.

1. **Unified Framework**: These two algorithms were integrated into a unified framework named as "fast finetune".
2. **Quantization Aware Finetuning**: Only the weight and bias (optional) will be updated, the scales and zero points are fixed, which ensures that all the quantizing information and the structure of the quantized model keep unchanged after finetuning.
3. **Flexibility**: AdaRound in Quark for ONNX is compatible with many more graph patterns-matching.
4. **More Advanced Options**

   - **Early Stop**: If the average loss of the current batch iterations decreases compared to the previous batch of iterations, the training of the layer will stop early. It will accelerate the finetuning process.
   - **Selective Update**: If the end-to-end accuracy does not improve after training a certain layer, discard the finetuning result of that layer.
   - **Adjust Learning Rate**: Besides the overall learning rate, you could set up a scheme to adjust learning rate layer-wise. For example, apply a larger learning rate on the layer that has a bigger loss.

How to Enable AdaRound / AdaQuant in AMD Quark?
-----------------------------------------------

AdaRound and AdaQuant are provided as options of optimal algorithms for fast finetune.

Here is a simple example showing how to enable default AdaRound and AdaQuant configuration.

.. code:: python

   from quark.onnx.quantization.config import Config, QuantizationConfig, get_default_config
   # Config of default AdaRound
   quant_config = get_default_config("S8S8_AAWS_ADAROUND")
   config = Config(global_quant_config=quant_config)
   # Config of default AdaQuant
   quant_config = get_default_config("S8S8_AAWS_ADAQUANT")
   config = Config(global_quant_config=quant_config)

Examples
--------

AdaRound
~~~~~~~~

This :doc:`example <../example_quark_onnx_adaround>` demonstrates quantizing a mobilenetv2_050.lamb_in1k model using the AMD Quark ONNX quantizer.

AdaQuant
~~~~~~~~

This :doc:`example <../example_quark_onnx_adaquant>` demonstrates quantizing a mobilenetv2_050.lamb_in1k model using the AMD Quark ONNX quantizer.
