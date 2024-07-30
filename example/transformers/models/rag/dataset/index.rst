##########################
Ryzen AI Software   
##########################

AMD Ryzen™ AI Software enables developers to take full advantage of AMD XDNA™ architecture integrated in select AMD Ryzen AI processors. Ryzen AI software intelligently optimizes AI tasks and workloads, freeing-up CPU and GPU resources, providing optimal performance at lower power.

**Bring your own model**: Ryzen AI software lets developers take machine learning models trained in PyTorch or TensorFlow and deploy them on laptops powered by Ryzen AI processors using ONNX Runtime.

**Use optimized library functions**: The Ryzen AI Library provides ready-made functions optimized for Ryzen AI processors so that developers can integrate these functions in their applications with no machine learning experience required.

|

.. image:: images/landing_new.png
   :scale: 75%
   :align: center

|
|


.. toctree::
   :maxdepth: 1
   :caption: Release Notes

   relnotes.rst


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   inst.rst
   runtime_setup.rst
   devflow.rst
   examples.rst

.. toctree::
   :maxdepth: 1
   :caption: Using Your CNN Model

   modelcompat.rst
   modelport.rst
   modelrun.rst

.. toctree::
   :maxdepth: 1
   :caption: Ryzen AI Library

   Quick Start Guide <ryzen_ai_libraries.rst>

.. toctree::
   :maxdepth: 1
   :caption: Additional Topics

   Model Zoo <https://huggingface.co/models?other=RyzenAI>
   manual_installation.rst
   alternate_quantization_setup.rst 
   onnx_e2e.rst
   early_access_features.rst


..
  ------------
  #####################################
  License
  #####################################
  
  Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
