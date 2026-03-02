Quantizing LLMs for ONNX Runtime GenAI
======================================

This document provides examples of quantizing large language models (LLMs) to **UINT4** using the **AWQ algorithm** via the Quark API, and exporting them to ONNX format using the **ONNX Runtime Gen AI Model Builder**.

`ONNX Runtime GenAI <https://github.com/microsoft/onnxruntime-genai>`__ offers an end-to-end pipeline for working with ONNX models, including inference using ONNX Runtime, logits processing, search and sampling, and key-value (KV) cache management. For detailed documentation, visit the `ONNX Runtime Gen AI Documentation <https://onnxruntime.ai/docs/genai>`_. The tool includes a `Model Builder <https://onnxruntime.ai/docs/genai/howto/build-model.html>`_ that facilitates exporting models to the ONNX format.

.. note::

    For large models, it is recommended to run this workflow on a data center GPU, as it is memory-intensive. Most laptop and desktop GPUs will not have enough memory to process larger parameter models efficiently. If you haven't done so already, and have a GPU, we suggest you create a fresh Quark environment with a PyTorch ROCm or CUDA install for your platform https://pytorch.org/get-started/locally/.

Preparation
-----------

Model Preparation
~~~~~~~~~~~~~~~~~

To use **Llama2 models**, download the HF Llama2 checkpoint. Access to these checkpoints requires a permission request to Meta. For more information, refer to the `Llama2 page on Hugging Face <https://huggingface.co>`_. Once permission is granted, download the checkpoint and save it to the ``<llama checkpoint folder>``.

Installation
~~~~~~~~~~~~

We will use a script from the ``examples/`` directory for Quark. This directory is found in the Quark ``.zip`` that can be downloaded at `📥amd_quark.zip release_version <https://www.xilinx.com/bin/public/openDownload?filename=amd_quark-@version@.zip>`__.

The ``quantize_quark.py`` script, that we will use, is found in a sub-directory of ``examples/``. It requires some additional dependencies be installed:

.. code-block:: bash

    cd examples/torch/language_modeling/llm_ptq/
    pip install -r requirements.txt

Quark UINT4 Quantization with AWQ
---------------------------------

**Quantization Configuration**: AWQ / Group 128 / Asymmetric / FP16 activations

Use the following command to quantize the model:

.. code-block:: bash

    python3 quantize_quark.py --model_dir <llama checkpoint folder> \
                              --output_dir <quantized safetensor output dir> \
                              --quant_scheme w_uint4_per_group_asym \
                              --num_calib_data 128 \
                              --quant_algo awq \
                              --dataset pileval_for_awq_benchmark \
                              --seq_len 512 \
                              --model_export hf_format \
                              --data_type float16 \
                              --custom_mode awq

This will generate a directory containing the safe tensors at the specified ``<quantized safetensor output dir>``.

.. note::

    To include the ``lm_head`` layer in the quantization process, add the ``--exclude_layers`` flag. This overrides the default behavior of excluding the ``lm_head`` layer.

.. note::

    To quantize the model for BF16 activations, use the ``--data_type bfloat16`` flag.

.. note::

    To specify a group size other than 128, such as 32, use the ``--group_size 32`` flag.


(Optional) Quark UINT4 Quantization with Different Group Sizes per Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark supports quantizing layers with different group sizes, providing finer-grained control over the quantization process. This allows you to better balance performance and accuracy.
For example, to quantize the model with 32 group size for lm_head, while 128 group size for the rest, use the following command:

.. code-block:: bash

    python3 quantize_quark.py --model_dir <llama checkpoint folder> \
                              --output_dir <quantized safetensor output dir> \
                              --quant_scheme w_uint4_per_group_asym \
                              --num_calib_data 128 \
                              --quant_algo awq \
                              --dataset pileval_for_awq_benchmark \
                              --seq_len 512 \
                              --model_export hf_format \
                              --data_type float16 \
                              --exclude_layers \
                              --group_size 128 \
                              --group_size_per_layer lm_head 32

.. note::

    This is an advanced feature that is **not supported** by the standard AWQ model format. 
    As a result, the quantized model is stored in the Quark model format, which does **not** require 
    the ``--custom_mode awq`` argument.

    Support for the Quark model format in **ONNX Runtime GenAI** is coming soon in v0.7 release.
    For early access you can try this `feature branch <https://github.com/shobrienDMA/onnxruntime-genai/tree/shobrien/per_layer_support>`_.

Exporting Using ONNX Runtime Gen AI Model Builder
-------------------------------------------------

Install the ONNX Runtime Gen AI tool package using ``pip``:

.. code-block:: bash

    pip install onnxruntime-genai

To export the quantized model to ONNX format, run the following command:

.. code-block:: bash

    python3 -m onnxruntime_genai.models.builder \
            -i <quantized safetensor output dir> \
            -o <quantized onnx model output dir> \
            -p int4 \
            -e dml

.. note::

    The activation data type of the ONNX model depends on the combination of the ``-p`` (precision) and ``-e`` (execution provider) flags. For example:

    - Using ``-p int4 -e dml`` will generate an ONNX model with float16 activations prepared for the DirectML execution provider for hybrid (NPU + iGPU) flow.
    - To generate an ONNX model with float32 activations for NPU flow, use the ``-p int4 -e cpu`` flag.

