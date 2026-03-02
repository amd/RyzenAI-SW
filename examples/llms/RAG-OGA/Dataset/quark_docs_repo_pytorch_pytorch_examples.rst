Accessing PyTorch Examples
==========================

You can get the example code after downloading and unzipping ``amd_quark.zip`` (refer to :doc:`Installation Guide <../install>`).
The example folder is in amd_quark.zip.

   Directory Structure of the ZIP File:

   ::

         + quark.zip
            + amd_quark.whl
            + examples
               + torch    # HERE ARE THE PYTORCH EXAMPLES
                  + language_modeling
                  + diffusers
                  + ...
               + onnx
                  + image_classification
                  + language_models
                  + ...
            + ...

.. toctree::
   :caption: PyTorch Examples in Quark for This Release
   :maxdepth: 1

   Diffusion Model Quantization <example_quark_torch_diffusers>
   AMD Quark Extension for Brevitas Integration <example_quark_torch_brevitas>
   Integration with AMD Pytorch-light (APL) <example_quark_torch_pytorch_light>
   Language Model Pruning <example_quark_torch_llm_pruning>
   Language Model PTQ <example_quark_torch_llm_ptq>
   Language Model QAT <example_quark_torch_llm_qat>
   Language Model Evaluation <example_quark_torch_llm_eval>
   Vision Model Quantization using FX Graph Mode <example_quark_torch_vision>
