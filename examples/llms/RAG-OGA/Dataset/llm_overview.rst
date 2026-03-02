########
Overview
########

************************************
OGA-based Flow with Hybrid Execution
************************************

Ryzen AI Software supports deploying quantized 4-bit LLMs on Ryzen AI 300-series PCs. This solution uses a hybrid execution mode, which leverages both the NPU and integrated GPU (iGPU), and is built on the OnnxRuntime GenAI (OGA) framework. 

Hybrid execution mode optimally partitions the model such that different operations are scheduled on NPU vs. iGPU. This minimizes time-to-first-token (TTFT) in the prefill-phase and maximizes token generation (tokens per second, TPS) in the decode phase.

OGA is a multi-vendor generative AI framework from Microsoft that provides a convenient LLM interface for execution backends such as Ryzen AI. 

Supported Configurations
========================

- Only Ryzen AI 300-series Strix Point (STX) and Krackan Point (KRK) processors support OGA-based hybrid execution.
- Developers with Ryzen AI 7000- and 8000-series processors can get started using the CPU-based examples linked in the :ref:`featured-llms` table.
- Windows 11 is the required operating system.


*******************************
Development Interfaces
*******************************

The Ryzen AI LLM software stack is available through three development interfaces, each suited for specific use cases as outlined in the sections below. All three interfaces are built on top of native OnnxRuntime GenAI (OGA) libraries, as shown in the :ref:`software-stack-table` diagram below. 

The high-level Python APIs, as well as the Server Interface, also leverage the Lemonade SDK, which is multi-vendor open-source software that provides everything necessary for quickly getting started with LLMs on OGA.

A key benefit of both OGA and Lemonade is that software developed against their interfaces is portable to many other execution backends.

.. _software-stack-table:

.. flat-table:: Ryzen AI Software Stack
   :header-rows: 1
   :class: center-table

   * - Your Python Application
     - Your LLM Stack
     - Your Native Application
   * - `Lemonade Python API* <#high-level-python-sdk>`_
     - `Lemonade Server Interface* <#server-interface-rest-api>`_
     - :rspan:`1` `OGA C++ Headers <../hybrid_oga.html>`_
   * - :cspan:`1` `OGA Python API* <https://onnxruntime.ai/docs/genai/api/python.html>`_
   * - :cspan:`2` `Custom AMD OnnxRuntime GenAI (OGA) <https://github.com/microsoft/onnxruntime-genai>`_
   * - :cspan:`2` `AMD Ryzen AI Driver and Hardware <https://www.amd.com/en/products/processors/consumer/ryzen-ai.html>`_

\* indicates open-source software (OSS).

High-Level Python SDK
=====================

The high-level Python SDK, Lemonade, allows you to get started using PyPI installation in approximately 5 minutes.

This SDK allows you to:

- Experiment with models in hybrid execution mode on Ryzen AI hardware.
- Validate inference speed and task performance.
- Integrate with Python apps using a high-level API.

To get started in Python, follow these instructions: :doc:`high_level_python`.


Server Interface (REST API)
===========================

The Server Interface provides a convenient means to integrate with applications that:

- Already support an LLM server interface, such as the Ollama server or OpenAI API.
- Are written in any language (C++, C#, Javascript, etc.) that supports REST APIs.
- Benefits from process isolation for the LLM backend.

To get started with the server interface, follow these instructions: :doc:`server_interface`.

For example applications that have been tested with Lemonade Server, see the `Lemonade Server Examples <https://github.com/lemonade-sdk/lemonade/tree/main/docs/server/apps>`_.


OGA APIs for C++ Libraries and Python
=====================================

Native C++ libraries for OGA are available to give full customizability for deployment into native applications.

The Python bindings for OGA also provide a customizable interface for Python development.

To get started with the OGA APIs, follow these instructions: :doc:`../hybrid_oga`.


.. _featured-llms:

*******************************
Featured LLMs
*******************************

The following tables contain a curated list of LLMs that have been validated on Ryzen AI hybrid execution mode. The hybrid examples are built on top of OnnxRuntime GenAI (OGA).

The comprehensive set of pre-optimized models for hybrid execution used in these examples are available in the `AMD hybrid collection on Hugging Face <https://huggingface.co/collections/amd/ryzenai-14-llm-hybrid-models-67da31231bba0f733750a99c>`_. It is also possible to run fine-tuned versions of the models listed (for example, fine-tuned versions of Llama2 or Llama3). For instructions on how to prepare a fine-tuned OGA model for hybrid execution, refer to :doc:`../oga_model_prepare`.

.. _ryzen-ai-oga-featured-llms:

.. flat-table:: Ryzen AI OGA Featured LLMs
   :header-rows: 2
   :class: llm-table

   * - 
     - :cspan:`1` CPU Baseline (HF bfloat16)
     - :cspan:`3` Ryzen AI Hybrid (OGA int4)
   * - Model
     - Example
     - Validation
     - Example
     - TTFT Speedup
     - Tokens/S Speedup
     - Validation
     
   * - `DeepSeek-R1-Distill-Qwen-7B <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`_
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/cpu/DeepSeek_R1_Distill_Qwen_7B.md>`__
     - 🟢
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/hybrid/DeepSeek_R1_Distill_Qwen_7B.md>`__
     - 3.4x
     - 8.4x
     - 🟢
   * - `DeepSeek-R1-Distill-Llama-8B <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B>`_
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/cpu/DeepSeek_R1_Distill_Llama_8B.md>`__
     - 🟢
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/hybrid/DeepSeek_R1_Distill_Llama_8B.md>`__
     - 4.2x
     - 7.6x
     - 🟢
   * - `Llama-3.2-1B-Instruct <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`_
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/cpu/Llama_3_2_1B_Instruct.md>`__
     - 🟢
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/hybrid/Llama_3_2_1B_Instruct.md>`__
     - 1.9x
     - 5.1x
     - 🟢
   * - `Llama-3.2-3B-Instruct <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>`_
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/cpu/Llama_3_2_3B_Instruct.md>`__
     - 🟢
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/hybrid/Llama_3_2_3B_Instruct.md>`__
     - 2.8x
     - 8.1x
     - 🟢
   * - `Phi-3-mini-4k-instruct <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>`_
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/cpu/Phi_3_mini_4k_instruct.md>`__
     - 🟢
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/hybrid/Phi_3_mini_4k_instruct.md>`__
     - 3.6x
     - 7.8x
     - 🟢
   * - `Qwen1.5-7B-Chat <https://huggingface.co/Qwen/Qwen1.5-7B-Chat>`_
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/cpu/Qwen1_5_7B_Chat.md>`__
     - 🟢
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/hybrid/Qwen1_5_7B_Chat.md>`__
     - 4.0x
     - 7.3x
     - 🟢
   * - `Mistral-7B-Instruct-v0.3 <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>`_
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/cpu/Mistral_7B_Instruct_v0_3.md>`__
     - 🟢
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/hybrid/Mistral_7B_Instruct_v0_3.md>`__
     - 5.0x
     - 8.1x
     - 🟢
   * - `Llama-3.1-8B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/cpu/Llama_3_1_8B_Instruct.md>`__
     - 🟢
     - `Link <https://github.com/amd/RyzenAI-SW/blob/main/example/llm/lemonade/hybrid/Llama_3_1_8B_Instruct.md>`__
     - 3.9x
     - 8.9x
     - 🟢

The :ref:`ryzen-ai-oga-featured-llms` table was compiled using validation, benchmarking, and accuracy metrics as measured by the `ONNX TurnkeyML v6.1.0 <https://pypi.org/project/turnkeyml/6.1.0/>`_ ``lemonade`` commands in each example link. After this table was created, the Lemonade SDK moved to the new location found `here <https://github.com/lemonade-sdk/lemonade>`_.

Data collection details:

* All validation, performance, and accuracy metrics are collected on the same system configuration:
  
  * System: HP OmniBook Ultra Laptop 14z
  * Processor: AMD Ryzen AI 9 HX 375 W/ Radeon 890M
  * Memory: 32GB of RAM

* The Hugging Face ``transformers`` framework is used as the baseline implementation for speedup and accuracy comparisons. 

  * The baseline checkpoint is the original ``safetensors`` Hugging Face checkpoint linked in each table row, in the ``bfloat16`` data type.
  
* All speedup numbers are the measured performance of the model with input sequence length (ISL) of ``1024`` and output sequence length (OSL) of ``64``, on the specified backend, divided by the measured performance of the baseline.
* We assign the 🟢 validation score based on this criteria: all commands in the example guide ran successfully.


**************************************
OGA-based Flow with NPU-only Execution
**************************************

The primary OGA-based flow for LLMs employs a hybrid execution mode which leverages both the NPU and iGPU. AMD also provides support for an OGA-based flow where the iGPU is not solicited and where the compute-intensive operations are exclusively offloaded to the NPU.

The OGA-based NPU-only execution mode is supported on STX and KRK platforms.

To get started with the OGA-based NPU-only execution mode, follow these instructions :doc:`../npu_oga`.




..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
