####################
Preparing OGA Models
####################

This section describes the process for preparing LLMs for deployment on a Ryzen AI PC using the hybrid or NPU-only execution mode. Currently, the flow supports only fine-tuned versions of the models already supported (as listed in :doc:`hybrid_oga` page). For example, fine-tuned versions of Llama2 or Llama3 can be used. However, different model families with architectures not supported by the hybrid flow cannot be used.

Preparing a LLM for deployment on a Ryzen AI PC involves 2 steps:

1. **Quantization**: The pretrained model is quantized to reduce memory footprint and better map to compute resources in the hardware accelerators
2. **Postprocessing**: During the postprocessing the model is exported to OGA followed by NPU-only or Hybrid execution mode specific postprocess to obtain the final deployable model.

************
Quantization
************

Prerequisites
=============
Linux machine with AMD (e.g., AMD Instinct MI Series) or Nvidia GPUs

Setup
=====

1. Create and activate Conda Environment 

.. code-block::

    conda create --name <conda_env_name> python=3.11
    conda activate <conda_env_name>

2. If Using AMD GPUs, update PyTorch to use ROCm 

.. code-block:: 
  
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
     python -c "import torch; print(torch.cuda.is_available())" # Must return `True`

3. Download :download:`AMD Quark 0.8 <https://www.xilinx.com/bin/public/openDownload?filename=amd_quark-0.8.zip>` and unzip the archive

4. Install Quark: 

.. code-block::

     cd <extracted amd quark 0.8>
     pip install amd_quark-0.8+<>.whl

5. Install other dependencies

.. code-block::

   pip install datasets
   pip install transformers
   pip install accelerate
   pip install evaluate


Some models may require a specific version of ``transformers``. For example, ChatGLM3 requires version 4.44.0.   

Generate Quantized Model
========================

Use following command to run Quantization. In a GPU equipped Linux machine the quantization can take about 30-60 minutes. 

.. code-block::

     cd examples/torch/language_modeling/llm_ptq/
     
     python quantize_quark.py \
          --model_dir "meta-llama/Llama-2-7b-chat-hf"  \
          --output_dir <quantized safetensor output dir>  \
          --quant_scheme w_uint4_per_group_asym \
          --num_calib_data 128 \
          --quant_algo awq \
          --dataset pileval_for_awq_benchmark \
          --model_export hf_format \
          --data_type <datatype> \
          --exclude_layers


- To generate OGA model for NPU only execution mode use ``--datatype float32``
- To generate OGA model for Hybrid execution mode use ``--datatype float16``
- For a BF16 pretrained model, you can use ``--data_type bfloat16``.

The quantized model is generated in the <quantized safetensor output dir> folder.

**************
Postprocessing
**************

Copy the quantized model to the Windows PC with Ryzen AI installed, activate the Ryzen AI Conda environment, and execute ``model_generate`` command to generate the final model.

Generate the final model for Hybrid execution mode:

.. code-block::

   conda activate ryzen-ai-<version>

   model_generate --hybrid <output_dir> <quantized_model_path>  

 
Generate the final model for NPU execution mode:

.. code-block::

   conda activate ryzen-ai-<version>

   model_generate --npu <output_dir> <quantized_model_path>  

..
  ------------

  #####################################
  License
  #####################################

  Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
