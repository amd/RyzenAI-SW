Accelerate with GPUs
====================

This guide provides detailed instructions on how to use ROCm and CUDA to accelerate models on GPUs. It covers the configuration steps for calibration, fast finetuning, and BFP16 models inference.

Environment Setup
-----------------

- **ONNX Runtime with ROCm**:
  For AMD GPUs, refer to the `AMD - ROCm | ONNX Runtime documentation <https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html>`_ for installation and setup instructions.

- **ONNX Runtime with CUDA**:
  For NVIDIA GPUs, refer to the `NVIDIA - CUDA | ONNX Runtime documentation <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>`_ for installation and setup instructions.

Calibration
-----------

In the quantization workflow, calibration adjusts the model's weights and activation values based on a small amount of input data to improve quantization accuracy. When using AMD GPUs, you might accelerate the calibration process with `ROCMExecutionProvider`, and also you can use `CUDAExecutionProvider` for NVIDIA GPUs. The following is an example configuration:

.. code-block:: python

    from quark.onnx.quantization.config.config import Config, QuantizationConfig

    quant_config = QuantizationConfig(
        calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
        quant_format=quark.onnx.QuantFormat.QDQ,
        execution_providers=['ROCMExecutionProvider']
    )

    config = Config(global_quant_config=quant_config)

.. note::
   By setting `execution_providers=['ROCMExecutionProvider']`, the calibration process is configured to run on the GPU for faster execution. Please check if GPUs are available beforehand.

Fast Finetune
-------------

AMD Quark for ONNX offers a fast finetuning feature that improves model accuracy after post-training quantization (PTQ). By adjusting the relevant parameters, you can ensure that both the PyTorch training phase and ONNX inference phase utilize GPU acceleration.

Here is an example configuration for the `adaround` optimization algorithm:

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
                'OptimDevice': "cuda:0",  # Use GPU 0 in PyTorch training
                'InferDevice': 'cuda:0',  # Use GPU 0 for ONNX inference
                'BatchSize': 1,
                'NumIterations': 1000,
                'LearningRate': 0.1,
            }
        }
    )

    config = Config(global_quant_config=quant_config)
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)

.. note::
   - `OptimDevice: "cuda:0"` indicates that GPU (supports AMD and NVIDIA GPUs) acceleration is used during PyTorch training.
   - `InferDevice: 'cuda:0'` indicates that GPU acceleration is used during ONNX inference via the `ROCMExecutionProvider` or `CUDAExecutionProvider`.

Inference
---------

For quantized model's inference, you can also use `ROCMExecutionProvider` or `CUDAExecutionProvider` to enable GPU acceleration. Below is an example that demonstrates how to use AMD GPUs to accelerate ONNX inference:


.. code-block:: python

    import onnxruntime as ort
    from quark.onnx import get_library_path

    so = ort.SessionOptions()
    so.register_custom_ops_library(get_library_path('ROCM'))
    session = ort.InferenceSession("quantized_model.onnx", so, providers=['ROCMExecutionProvider'])
    print("Execution provider:", session.get_providers())  # Ensure 'ROCMExecutionProvider' is present

    output = session.run(None, {"input": input_data})

.. note::
   If the `session.get_providers()` output includes `ROCMExecutionProvider`, the inference process is running on the GPU for acceleration.
