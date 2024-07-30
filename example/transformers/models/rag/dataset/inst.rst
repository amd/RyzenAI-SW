#########################
Installation Instructions
#########################

************************
Supported Configurations
************************

The Ryzen AI Software supports the following processors running Windows 11

- AMD Ryzen™ 7940HS, 7840HS, 7640HS, 7840U, 7640U.
- AMD Ryzen™ 8640U, 8640HS, 8645H, 8840U, 8840HS, 8845H, 8945H. 

.. note::
   In this documentation, "NPU" is used in descriptions, while "IPU" is retained in the tool's language, code, screenshots, and commands. This intentional 
   distinction aligns with existing tool references and does not affect functionality. Avoid making replacements in the code.

******************
Prepare the System
******************

Download the :download:`NPU Driver <https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=ipu_stack_rel_silicon_prod_1.1.zip>` and install it by following these steps:


1. Extract the downloaded zip file.
2. Open a terminal in administrator mode and execute the ``.\amd_install_kipudrv.bat`` bat file.

Ensure that the NPU driver is installed from ``Device Manager`` -> ``System Devices`` -> ``AMD IPU Device`` as shown in the following image.

.. image:: images/ipu_driver_1.1.png
   :align: center
   :width: 400 px

|
|

To enable the development and deployment of applications leveraging the NPU, you must have the following software installed on the system.


.. list-table:: 
   :widths: 25 25 
   :header-rows: 1

   * - Dependencies
     - Version Requirement
   * - Visual Studio
     - 2019
   * - cmake
     - version >= 3.26
   * - python
     - version >= 3.9 
   * - Anaconda or Miniconda
     - Latest version


|
|

.. _install-bundled:

*****************************
Install the Ryzen AI Software
*****************************

Before installing the Ryzen AI Software, ensure that all the prerequisites outlined previously have been met and that the Windows PATH variable is properly set for each component. For example, Anaconda requires following paths to be set in the PATH variable ``path\to\anaconda3\``, ``path\to\anaconda3\Scripts\``, ``path\to\anaconda3\Lib\bin\``. The PATH variable should be set through the *Environment Variables* window of the *System Properties*. 

Download the :download:`ryzen-ai-sw-1.1.zip <https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=ryzen-ai-sw-1.1.zip>` Ryzen AI Software installation package and extract it. 

Open an Anaconda or Windows command prompt in the extracted folder and run the installation script as shown below. Make sure to enter "Y" when prompted to accept the EULA. 

.. code:: 

    .\install.bat

The ``install.bat`` script does the following: 

- Creates a conda environment 
- Installs the :doc:`vai_quant/vai_q_onnx`
- Installs the `ONNX Runtime <https://onnxruntime.ai/>`_
- Installs the :doc:`Vitis AI Execution Provider <modelrun>`
- Configures the environment to use the throughput profile of the NPU
- Prints the name of the conda environment before exiting 

The default Ryzen AI Software packages are now installed in the conda environment created by the installer. You can start using the Ryzen AI Software by activating the conda environment created by the installer (the name of the environment is printed during the installation process). 

**IMPORTANT:** The Ryzen AI Software installation folder (where the zip file was extracted) contains various files required at runtime by the inference session. These files include the NPU binaries (:file:`*.xclbin`) and the default runtime configuration file (:file:`vaip_config.json`) for the Vitis AI Execution Provider. Because of this, the installation folder should not be deleted and should be kept in a convenient location. Refer to the :doc:`runtime_setup` page for more details about setting up the environment before running an inference session on the NPU.


.. rubric:: Customizing the Installation

- To specify the name of the conda work environment created by the installer, run the script as follows:

.. code::

   .\install.bat -env <env name>

- Instead of the automated installation process, you can install each component manually by following the instructions on the :doc:`manual_installation` page.

- To use your existing conda environment with the Ryzen AI software, follow the :doc:`manual_installation` instructions and manually install the Vitis AI ONNX Quantizer, the ONNX Runtime, and the Vitis AI Execution Provider, without creating a new conda environment.

- If you need to install the Vitis AI PyTorch/TensorFlow Quantizer or the Microsoft Olive Quantizer, refer to the :doc:`alternate_quantization_setup` page. 


|
|

*********************
Test the Installation
*********************

The ``ryzen-ai-sw-1.1`` package contains a test to verify that the Ryzen AI software is correctly installed. This installation test can be found in the ``quicktest`` folder.

- Activate the conda environment:

.. code-block::

   conda activate <env_name>

- Run the test: 

.. code-block::

   cd ryzen-ai-sw-1.1\quicktest
   python quicktest.py


- The test runs a simple CNN model. On a successful run, you will see an output similar to the one shown below. This indicates that the model is running on NPU and the installation of the Ryzen AI Software was successful:

.. code-block::
  
   [Vitis AI EP] No. of Operators :   CPU     2    IPU   398  99.50%
   [Vitis AI EP] No. of Subgraphs :   CPU     1    IPU     1 Actually running on IPU     1
   ...
   Test Passed
   ...

..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
