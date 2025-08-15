:orphan:

##########################
Ryzen AI Software on Linux
##########################

This guide provides instructions for using Ryzen AI 1.4 on Linux for model compilation and followed by running inference on Windows.

*************
Prerequisites
*************
The following are the recommended system configuration for RyzenAI Linux installer

.. list-table:: 
   :widths: 25 25 
   :header-rows: 1

   * - Dependencies
     - Version Requirement
   * - Ubuntu
     - 22.04
   * - RAM
     - 32GB or Higher
   * - CPU cores
     - >= 8 
   * - Python
     - 3.10 or Higher


*************************
Installation Instructions
*************************

- Download the Ryzen AI Software Linux installer: :download:`ryzen_ai-1.4.0.tgz <https://account.amd.com/en/forms/downloads/amd-end-user-license-xef.html?filename=ryzen_ai-1.4.0.tgz>`.

- Extract the .tgz using the following command: 

.. code-block::

    tar -xvzf ryzen_ai-1.4.0.tgz

- Run the installer with default settings. This will prompt to read and agree to the EULA:

.. code-block::

    cd ryzen_ai-1.4.0
    ./install_ryzen_ai_1_4.sh 

- After reading the EULA, re-run the installer with options to agree to the EULA and create a Python virtual environment:

.. code-block::

    ./install_ryzen_ai_1_4.sh -a yes -p <PATH TO VENV> -l

- Activate the virtual environment to start using the Ryzen AI Software:  

.. code-block::

   source <PATH TO VENV>/bin/activate


******************
Usage Instructions
******************

The process for model compilation on Linux is similar to that on Windows. Refer to the instructions provided in the :doc:`modelrun` page for complete details.

Once the model has been successfully compiled on your Linux machine, proceed to copy the entire working directory to a Windows machine that operates on an STX-based system.

Prior to running the compiled model on the Windows machine, ensure that all required prerequisites are satisfied as listed in the :doc:`inst` page.
