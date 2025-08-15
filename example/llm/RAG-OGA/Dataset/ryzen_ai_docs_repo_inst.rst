.. include:: /icons.txt

#########################
Installation Instructions
#########################



*************
Prerequisites
*************

The Ryzen AI Software supports AMD processors with a Neural Processing Unit (NPU). Consult the release notes for the full list of :ref:`supported configurations <supported-configurations>`. 

The following dependencies must be present on the system before installing the Ryzen AI Software:

.. list-table:: 
   :widths: 25 25 
   :header-rows: 1

   * - Dependencies
     - Version Requirement
   * - Windows 11
     - build >= 22621.3527
   * - Visual Studio
     - 2022
   * - cmake
     - version >= 3.26
   * - Anaconda or Miniconda
     - Latest version

|

|warning| **IMPORTANT**: 

- Visual Studio 2022 Community: ensure that "Desktop Development with C++" is installed

- Anaconda or Miniconda: ensure that the following path is set in the System PATH variable: ``path\to\anaconda3\Scripts`` or ``path\to\miniconda3\Scripts`` (The System PATH variable should be set in the *System Variables* section of the *Environment Variables* window). 

|

.. _install-driver:

*******************
Install NPU Drivers
*******************

- Download the NPU driver installation package :download:`NPU Driver <https://account.amd.com/en/forms/downloads/amd-end-user-license-xef.html?filename=NPU_RAI1.4_GA_257_WHQL.zip>`

- Install the NPU drivers by following these steps:

  - Extract the downloaded ``NPU_RAI1.4_GA_257_WHQL.zip`` zip file.
  - Open a terminal in administrator mode and execute the ``.\npu_sw_installer.exe`` exe file.

- Ensure that NPU MCDM driver (Version:32.0.203.257, Date:3/12/2025) is correctly installed by opening ``Device Manager`` -> ``Neural processors`` -> ``NPU Compute Accelerator Device``.


.. _install-bundled:

*************************
Install Ryzen AI Software
*************************

- Download the RyzenAI Software installer :download:`ryzen-ai-1.4.0.exe <https://account.amd.com/en/forms/downloads/amd-end-user-license-xef.html?filename=ryzen-ai-1.4.0.exe>`.

- Launch the MSI installer and follow the instructions on the installation wizard:

  - Accept the terms of the Licence agreement
  - Provide the destination folder for Ryzen AI installation (default: ``C:\Program Files\RyzenAI\1.4.0``)
  - Specify the name for the conda environment (default: ``ryzen-ai-1.4.0``)


The Ryzen AI Software packages are now installed in the conda environment created by the installer. 


.. _quicktest:


*********************
Test the Installation
*********************

The Ryzen AI Software installation folder contains test to verify that the software is correctly installed. This installation test can be found in the ``quicktest`` subfolder.

- Open a Conda command prompt (search for "Anaconda Prompt" in the Windows start menu)

- Activate the Conda environment created by the Ryzen AI installer:

.. code-block::

   conda activate <env_name>

- Run the test:

.. code-block::

   cd %RYZEN_AI_INSTALLATION_PATH%/quicktest
   python quicktest.py


- The quicktest.py script sets up the environment and runs a simple CNN model. On a successful run, you will see an output similar to the one shown below. This indicates that the model is running on the NPU and that the installation of the Ryzen AI Software was successful:

.. code-block::

   [Vitis AI EP] No. of Operators :   CPU     2    NPU   398
   [Vitis AI EP] No. of Subgraphs :   NPU     1 Actually running on NPU     1
   ...
   Test Passed
   ...


.. note::

    The full path to the Ryzen AI Software installation folder is stored in the ``RYZEN_AI_INSTALLATION_PATH`` environment variable.


**************************
Other Installation Options
**************************

Linux Installer
~~~~~~~~~~~~~~~

Compiling BF16 models requires more processing power than compiling INT8 models. If a larger BF16 model cannot be compiled on a Windows machine due to hardware limitations (e.g., insufficient RAM), an alternative Linux-based compilation flow is supported. More details can be found here: :doc:`rai_linux`



Lightweight Installer
~~~~~~~~~~~~~~~~~~~~~

A lightweight installer is available with reduced features. It cannot be used for compiling BF16 models but fully supports compiling and running INT8 models and running LLM models.

- Download the RyzenAI Software Runtime MSI installer :download:`ryzen-ai-rt-1.4.0.msi <https://account.amd.com/en/forms/downloads/amd-end-user-license-xef.html?filename=ryzen-ai-rt-1.4.0.msi>`.

- Launch the MSI installer and follow the instructions on the installation wizard:

  - Accept the terms of the Licence agreement
  - Provide the destination folder for Ryzen AI installation
  - Specify the name for the conda environment



..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
