:orphan:

######################
OGA NPU Execution Mode
######################

Ryzen AI Software supports deploying LLMs on Ryzen AI PCs using the native ONNX Runtime Generate (OGA) C++ or Python API. The OGA API is the lowest-level API available for building LLM applications on a Ryzen AI PC. This documentation covers the NPU execution mode for LLMs, which utilizes only the NPU.  

**Note**: Refer to :doc:`hybrid_oga` for Hybrid NPU + GPU execution mode.


************************
Supported Configurations
************************

The Ryzen AI OGA flow supports Strix and Krackan Point processors. Phoenix (PHX) and Hawk (HPT) processors are not supported.


************
Requirements
************
- Install NPU Drivers and Ryzen AI MSI installer according to the :doc:`inst` 
- Install Git for Windows (needed to download models from HF): https://git-scm.com/downloads


********************
Pre-optimized Models
********************

AMD provides a set of pre-optimized LLMs ready to be deployed with Ryzen AI Software and the supporting runtime for NPU execution. These models can be found on Hugging Face in the following collection:

- https://huggingface.co/amd/Phi-3-mini-4k-instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix
- https://huggingface.co/amd/Phi-3.5-mini-instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix
- https://huggingface.co/amd/Mistral-7B-Instruct-v0.3-awq-g128-int4-asym-bf16-onnx-ryzen-strix
- https://huggingface.co/amd/Qwen1.5-7B-Chat-awq-g128-int4-asym-bf16-onnx-ryzen-strix
- https://huggingface.co/amd/chatglm3-6b-awq-g128-int4-asym-bf16-onnx-ryzen-strix
- https://huggingface.co/amd/Llama-2-7b-hf-awq-g128-int4-asym-bf16-onnx-ryzen-strix
- https://huggingface.co/amd/Llama2-7b-chat-awq-g128-int4-asym-bf16-onnx-ryzen-strix
- https://huggingface.co/amd/Llama-3-8B-awq-g128-int4-asym-bf16-onnx-ryzen-strix
- https://huggingface.co/amd/Llama-3.1-8B-awq-g128-int4-asym-bf16-onnx-ryzen-strix
- https://huggingface.co/amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix
- https://huggingface.co/amd/Llama-3.2-3B-Instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix
- https://huggingface.co/amd/DeepSeek-R1-Distill-Llama-8B-awq-g128-int4-asym-bf16-onnx-ryzen-strix  
- https://huggingface.co/amd/DeepSeek-R1-Distill-Qwen-1.5B-awq-g128-int4-asym-bf16-onnx-ryzen-strix 
- https://huggingface.co/amd/DeepSeek-R1-Distill-Qwen-7B-awq-g128-int4-asym-bf16-onnx-ryzen-strix   
- https://huggingface.co/amd/AMD-OLMo-1B-SFT-DPO-awq-g128-int4-asym-bf16-onnx-ryzen-strix

The steps for deploying the pre-optimized models using C++ and python are described in the following sections.

***************************
NPU Execution of OGA Models
***************************

Setup
=====

Activate the Ryzen AI 1.4 Conda environment:

.. code-block:: 
    
    conda activate ryzen-ai-1.4.0

Create a folder to run the LLMs from, and copy the required files:

.. code-block::

  mkdir npu_run
  cd npu_run
  xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\npu-llm\exe" .\libs
  xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\npu-llm\libs\vaip_llm.json" libs
  xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\npu-llm\onnxruntime-genai.dll" libs
  xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\onnxruntime_vitis_ai_custom_ops.dll" libs
  xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\onnxruntime_providers_shared.dll" libs
  xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\onnxruntime_vitisai_ep.dll" libs
  xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\dyn_dispatch_core.dll" libs
  xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\onnxruntime_providers_vitisai.dll" libs
  xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\transaction.dll" libs
  xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\onnxruntime.dll" libs
  xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\xclbin.dll" libs


Download Models from HuggingFace
================================

Download the desired models from the list of pre-optimized models on Hugging Face:

.. code-block:: 
    
     # Make sure you have git-lfs installed (https://git-lfs.com) 
     git lfs install  
     git clone <link to hf model> 

For example, for Llama-2-7b:

.. code-block:: 

     git lfs install  
     git clone https://huggingface.co/amd/Llama-2-7b-hf-awq-g128-int4-asym-bf16-onnx-ryzen-strix


**NOTE**: Ensure the models are cloned in the ``npu_run`` folder.


Enabling Performance Mode (Optional)
====================================

To run the LLMs in the best performance mode, follow these steps:

- Go to ``Windows`` → ``Settings`` → ``System`` → ``Power`` and set the power mode to Best Performance.
- Execute the following commands in the terminal:

.. code-block::

   cd C:\Windows\System32\AMD
   xrt-smi configure --pmode performance



Sample C++ Programs 
===================

The ``run_llm.exe`` test application provides a simple interface to run LLMs. The source code for this application can also be used a reference for how to integrate LLMs using the native OGA C++ APIs. 

It supports the following command line options:: 

    -m: model path
    -f: prompt file
    -n: max new tokens
    -c: use chat template
    -t: input prompt token length
    -l: max length to be set in search options
    -h: help


Example usage:

.. code-block::

   .\libs\run_llm.exe -m "Llama-2-7b-hf-awq-g128-int4-asym-bf16-onnx-ryzen-strix" -f "Llama-2-7b-hf-awq-g128-int4-asym-bf16-onnx-ryzen-strix\prompts.txt" -t "1024" -n 20 

|

The ``model_benchmark.exe`` program can be used to profile the execution of LLMs and report various metrics. It supports the following command line options:: 

    -i,--input_folder <path>
      Path to the ONNX model directory to benchmark, compatible with onnxruntime-genai.
    -l,--prompt_length <numbers separated by commas>
      List of number of tokens in the prompt to use.
    -p,--prompt_file <filename>
      Name of prompt file (txt) expected in the input model directory.
    -g,--generation_length <number>
      Number of tokens to generate. Default: 128
    -r,--repetitions <number>
      Number of times to repeat the benchmark. Default: 5
    -w,--warmup <number>
      Number of warmup runs before benchmarking. Default: 1
    -t,--cpu_util_time_interval <number in ms>
      Sampling time interval for peak cpu utilization calculation, in milliseconds. Default: 250
    -v,--verbose
      Show more informational output.
    -h,--help
      Show this help message and exit.


For example, for Llama-2-7b:

.. code-block::
   
   .\libs\model_benchmark.exe -i "Llama-2-7b-hf-awq-g128-int4-asym-bf16-onnx-ryzen-strix" -g 20 -p "Llama-2-7b-hf-awq-g128-int4-asym-bf16-onnx-ryzen-strix\prompts.txt" -l "2048,1024,512,256,128" 

|

**NOTE**: The C++ source code for the ``run_llm.exe`` and ``model_benchmark.exe`` executables can be found in the ``%RYZEN_AI_INSTALLATION_PATH%\npu-llm\cpp`` folder. This source code can be modified and recompiled using the commands below.

.. code-block::

   :: Copy project files
   xcopy /E /I "%RYZEN_AI_INSTALLATION_PATH%\npu-llm\cpp" .\sources

   :: Build project
   cd sources
   cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
   cmake --build build --config Release

   :: Copy executables in the "libs" folder 
   xcopy /I build\Release .\libs

   :: Copy runtime dependencies in the "libs" folder
   xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\npu-llm\libs\vaip_llm.json" libs
   xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\npu-llm\onnxruntime-genai.dll" libs
   xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\onnxruntime_vitis_ai_custom_ops.dll" libs
   xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\onnxruntime_providers_shared.dll" libs
   xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\onnxruntime_vitisai_ep.dll" libs
   xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\dyn_dispatch_core.dll" libs
   xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\onnxruntime_providers_vitisai.dll" libs
   xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\transaction.dll" libs
   xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\onnxruntime.dll" libs
   xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\voe\xclbin.dll" libs

Sample Python Scripts
=====================

In the model directory, open the ``genai_config.json`` file located in the folder of the downloaded model. Update the value of the "custom_ops_library" key with the path to the ``onnxruntime_vitis_ai_custom_ops.dll``, located in the ``npu_run\libs`` folder:

.. code-block::
  
      "session_options": {
                ...
                "custom_ops_library": "libs\\onnxruntime_vitis_ai_custom_ops.dll",
                ...
      }

To run LLMs other than ChatGLM, use the following command:

.. code-block:: 

     python "%RYZEN_AI_INSTALLATION_PATH%\hybrid-llm\examples\python\llama3\run_model.py" --model_dir <model folder>  

To run ChatGLM, use the following command:

.. code-block:: 

     pip install transformers==4.44.0  
     python "%RYZEN_AI_INSTALLATION_PATH%\hybrid-llm\examples\python\chatglm\model-generate-chatglm3.py" -m <model folder>  

For example, for Llama-2-7b:

.. code-block::
   
   python "%RYZEN_AI_INSTALLATION_PATH%\hybrid-llm\examples\python\llama3\run_model.py" --model_dir Llama-2-7b-hf-awq-g128-int4-asym-bf16-onnx-ryzen-strix


 
***********************
Using Fine-Tuned Models
***********************

It is also possible to run fine-tuned versions of the pre-optimized OGA models. 

To do this, the fine-tuned models must first be prepared for execution with the OGA NPU-only flow. For instructions on how to do this, refer to the page about :doc:`oga_model_prepare`.

Once a fine-tuned model has been prepared for NPU-only execution, it can be deployed by following the steps described above in this page.
