
################
Other Quantizers 
################


The Ryzen AI software provides support for these additional quantizers:

1. **Vitis AI Quantizer for PyTorch/TensorFlow/TensorFlow 2**: If you require Quantization Aware Training using the original model training framework, you can use Vitis AI PyTorch/Tensorflow 2/Tensorflow quantizer.
2. **Olive Quantizer**: If you are already familiar with the Olive framework and using it for other model transformations, quantization through Olive is also an option. 


.. _install-pt-tf:


******************************************************
Vitis AI Quantizer for PyTorch/TensorFlow/TensorFlow 2
******************************************************

The Vitis AI Quantizer, integrated as a component of either PyTorch, TensorFlow, or TensorFlow 2, is distributed through framework-specific Docker containers.

The Vitis AI Docker containers can be installed on Ubuntu 20.04, CentOS 7.8, 7.9, 8.1, and RHEL 8.3, 8.4. For developers working on Windows 11, WSL can be used to install the Vitis AI Docker containers.

Standard Containers
===================

Multiple versions of the Docker container are available, each tailored to specific frameworks. Follow the Docker download and running instructions as per the following links:

.. list-table:: 
   :widths: 30 70 
   :header-rows: 1

   * - Framework
     - Docker location
   * - PyTorch
     - https://hub.docker.com/r/amdih/ryzen-ai-pytorch
   * - TensorFlow 2
     - https://hub.docker.com/r/amdih/ryzen-ai-tensorflow2
   * - TensorFlow 1
     - https://hub.docker.com/r/amdih/ryzen-ai-tensorflow 


GPU-Accelerated Quantization Containers
=======================================

The standard Vitis AI Docker containers do not support GPU-accelerated quantization. To create a container with GPU-accelerated quantization enabled, download the following archive and follow the instructions in the README file.

`Download and build GPU Docker containers <https://account.amd.com/en/forms/downloads/xef.html?filename=ipu-rel-3.5.0-276.tar.gz>`_

.. note::
   In this documentation, **"NPU"** is used in descriptions, while **"IPU"** is retained in the tool's language, code, screenshots, and commands. This intentional 
   distinction aligns with existing tool references and does not affect functionality. Avoid making replacements in the code.

.. _install-olive:

***************
Olive Quantizer
***************


Microsoft Olive framework supports quantization with Vitis AI ONNX Quantization. The Olive Quantizer can be installed by following these steps:

1. Install the Olive Quantizer as follows:

.. code-block::

    pip install olive-ai[cpu]


2. The current version of the Olive Quantizer is not compatible with the latest version of the pydantic library. To make it compatible, downgrade the pydantic version using the following command:


.. code-block::

    pip install pydantic==1.10.9


For additional information regarding the installation of Olive, refer to the `Microsoft Olive Documentation <https://microsoft.github.io/Olive/getstarted/installation.html>`_


..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
