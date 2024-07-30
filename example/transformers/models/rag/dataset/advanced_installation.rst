.. _advanced_installation.rst:

#####################
Advanced Installation
#####################

This section dives deeper into specific aspects of the installation process, including quantization options and detailed NPU binary configuration.


.. _advanced-quantization:

Quantization
~~~~~~~~~~~~

**Vitis AI PyTorch/TensorFlow 2/TensorFlow Quantization**

The Vitis AI PyTorch and TensorFlow Quantizer, which is part of the Vitis AI toolchain, require the installation of a Docker container on the host server.

The Vitis AI Docker container can be installed on Ubuntu 20.04, CentOS 7.8, 7.9, 8.1, and RHEL 8.3, 8.4. The developers working on Windows 11 can use WSL for installing Vitis AI Docker.

Multiple versions of the Docker container are available, each tailored to specific frameworks. Follow the Docker download and running instructions as per the following links:

.. list-table:: 
   :widths: 25 25 
   :header-rows: 1

   * - Framework
     - Docker location
   * - PyTorch
     - https://hub.docker.com/r/amdih/ryzen-ai-pytorch
   * - TensorFlow 2
     - https://hub.docker.com/r/amdih/ryzen-ai-tensorflow2
   * - TensorFlow 1
     - https://hub.docker.com/r/amdih/ryzen-ai-tensorflow 


The above Docker containers do not have GPU-accelerated quantization support. If you like to leverage GPU for the quantization process, you can download and build GPU Docker containers. The following TAR file has README that you can follow to build and run GPU dockers.  


https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=ryzen-ai-gpudockerfiles-3.5.0-130.tar.gz


**Olive Quantization**


Microsoft Olive framework supports quantization with Vitis AI ONNX Quantization. If you're interested in exploring Olive Quantization as an advanced quantization method, you can follow the steps below:

1. Install Olive Quantization by running the following command:


.. code-block::

    pip install olive-ai[cpu]


2. The current Olive flow is not compatible with the latest pydantic version. To make it compatible, downgrade the pydantic version using the following command:


.. code-block::

   pip install pydantic==1.10.9


For additional information regarding the Olive installation, refer to the [Microsoft documentation](https://microsoft.github.io/Olive/getstarted/installation.html).


NPU Binary selection
~~~~~~~~~~~~~~~~~~~~

The NPU binaries are located inside the Vitis AI Execution Provider package. Selecting an NPU binary is a required step everytime the application is run from a new terminal. Ryzen AI Software platform provides a couple of NPU binaries using different configurations on the NPU device. 

**NPU binary 1x4.xclbin**: An AI stream using 1x4.xclbin use an NPU configuration that provides up to 2 TOPS performance. Most real-time application (video conferencing use cases) performance requirements can be met using this configuration. In the current Ryzen AI software platform, up to four such AI streams can be run in parallel on the NPU without any visible loss of performance.


**NPU binary 5x4.xclbin**: For more advanced use cases or larger models, NPU binary 5x4.xclbin can be used which uses a larger configuration to provide up to 10 TOPs performance. In the current version of the release, 5x4.xclbin does not support multiple concurrent AI streams, and can only be used by a single application. 


The procedure for selecting a specific binary using environment variables is as follows:

Selecting the 1x4.xclbin NPU binary:

.. code-block::

   set XLNX_VART_FIRMWARE=C:\path\to\1x4.xclbin


Selecting the 5x4.xclbin NPU binary:

.. code-block::

   set XLNX_VART_FIRMWARE=C:\path\to\5x4.xclbin
   set XLNX_TARGET_NAME="AMD_AIE2_5x4_Overlay"

Note: To select the 5x4.xclbin as the NPU binary, the additional XLNX_TARGET_NAME environment variable is required. 


..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
