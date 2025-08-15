AMD Quark for PyTorch
=====================

The :doc:`Getting started with AMD Quark <../basic_usage>` guide provides a general overview of the quantization process, irrespective of specific hardware or deep learning frameworks. This page details the features supported by the Quark PyTorch Quantizer and explains how to use it to quantize PyTorch models.

Basic Example
-------------

This example shows a basic use case on how to quantize the ``opt-125m`` model with the ``int8`` data type for ``symmetric`` ``per tensor`` ``weight-only`` quantization. We are following the :ref:`basic quantization steps from the Getting Started page <basic-usage-quantization-steps>`.

1. Load the original floating-point model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will use `Transformers <https://huggingface.co/docs/transformers/en/index>`_, from Hugging Face, to fetch the model.

.. code-block:: bash

   pip install transformers

We start by specifying the model we want to quantize. For this PyTorch example, we instantiate the model through Hugging Face API:

.. code-block:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
   model.eval()
   tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

2. (Optional) Define the data loader for calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The requirements of data loader are divided into two categories:

**DataLoader not required**

* Weight-only quantization (if advanced algorithms like AWQ are not used).
* Weight and activation dynamic quantization (if advanced algorithms like AWQ are not used).
* Advanced algorithms: Rotation.

**DataLoader required**

* Weight and activation static quantization.
* Advanced algorithms: SmoothQuant, AWQ and GPTQ.

.. code-block:: python

   from torch.utils.data import DataLoader
   text = "Hello, how are you?"
   tokenized_outputs = tokenizer(text, return_tensors="pt")
   calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

Refer to :doc:`Adding Calibration Datasets <calibration_datasets>` to learn more about how to use calibration datasets efficiently.

3. Set the quantization configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark for PyTorch provides a granular API to handle diverse quantization scenarios, and it also offers streamlined APIs for common use cases. The example below demonstrates the granular API approach.

.. code-block:: python

   from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
   from quark.torch.quantization.config.config import Config, QuantizationConfig
   from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
   from quark.torch.quantization import Int8PerTensorSpec
   DEFAULT_INT8_PER_TENSOR_SYM_SPEC = Int8PerTensorSpec(observer_method="min_max",
                                         symmetric=True,
                                         scale_type="float",
                                         round_method="half_even",
                                         is_dynamic=False).to_quantization_spec()

   DEFAULT_W_INT8_PER_TENSOR_CONFIG = QuantizationConfig(weight=DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
   quant_config = Config(global_quant_config=DEFAULT_W_INT8_PER_TENSOR_CONFIG)

4. Quantize the model
~~~~~~~~~~~~~~~~~~~~~

Once the model, input data, and quantization configuration are ready, quantizing the model is straightforward, as shown below:

.. code-block:: python

   from quark.torch import ModelQuantizer
   quantizer = ModelQuantizer(quant_config)
   quant_model = quantizer.quantize_model(model, calib_dataloader)

5. (Optional) Export the quantized model to other formats for deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exporting a model is only needed when users want to deploy models in another Deep Learning framework, such as ONNX, Hugging Face safetensors. To export a quantized model, users need to freeze the quantized model.

.. code-block:: python

    freezed_quantized_model = quantizer.freeze(quant_model)
    from quark.torch import ModelExporter

    # Generate dummy input
    for data in calib_dataloader:
        input_args = data
        break

    quant_model = quant_model.to('cuda')
    input_args = input_args.to('cuda')
    exporter = ModelExporter('export_path')
    exporter.export_onnx_model(quant_model, input_args)

If the code runs successfully, the terminal displays `[QUARK-INFO]: Model quantization has been completed.`

Further reading
---------------

* Quantized models can be evaluated to compare its performance with the original model. Learn more on :doc:`Model Evaluation <example_quark_torch_llm_eval>`.
* For more detailed information, see the section on :ref:`Advanced AMD Quark Features for PyTorch <advanced-quark-features-pytorch>`.
