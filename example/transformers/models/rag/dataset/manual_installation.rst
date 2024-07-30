###################
Manual Installation
###################

The main :doc:`inst` page shows a one-step installation process that checks the prerequisite and installs Vitis AI ONNX quantizer, ONNX Runtime, and Vitis AI execution provider.

This page explains how to install each component manually. 

.. note::

   Make sure to follow the installation steps in the order explained below.

********************
Download the Package
********************

Download the :download:`ryzen-ai-sw-1.1.zip <https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=ryzen-ai-sw-1.1.zip>` Ryzen AI Software installation package and extract it. 


**************************
Create a Conda Environment
**************************

The Ryzen AI Software requires using a conda environment (Anaconda or Miniconda) for the installation process. 

Start a conda prompt. In the conda prompt, create and activate an environment for the rest of the installation process. 

.. code-block:: 

  conda create --name <name> python=3.9
  conda activate <name> 


.. _install-onnx-quantizer:

******************************
Install the Vitis AI Quantizer
******************************

The :doc:`Vitis AI Quantizer for ONNX <vai_quant/vai_q_onnx>` supports a post-training quantization method that works on models saved in the ONNX format. 

Install the Vitis AI Quantizer for ONNX as follows:

.. code-block:: shell

   cd ryzen-ai-sw-1.1\ryzen-ai-sw-1.1
   pip install vai_q_onnx-1.16.0+69bc4f2-py2.py3-none-any.whl

To install other quantization tools (Vitis AI PyTorch/TensorFlow 2/TensorFlow Quantization or Olive Quantization), refer to the :doc:`alternate_quantization_setup` page. 


************************
Install the ONNX Runtime
************************

.. code-block::
   
   pip install onnxruntime 


***************************************
Install the Vitis AI Execution Provider
***************************************

.. code-block:: 

     cd ryzen-ai-sw-1.1\ryzen-ai-sw-1.1\voe-4.0-win_amd64
     pip install voe-0.1.0-cp39-cp39-win_amd64.whl
     pip install onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl
     python installer.py
