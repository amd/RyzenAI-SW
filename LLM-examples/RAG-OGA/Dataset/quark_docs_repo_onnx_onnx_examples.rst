Accessing ONNX Examples
=======================

Users can get the example code after downloading and unzipping ``amd_quark.zip`` (referring to :doc:`Installation Guide <../install>`).
The example folder is in amd_quark.zip.

   Directory Structure of the ZIP File:

   ::

         + amd_quark.zip
            + amd_quark.whl
            + examples    # HERE IS THE EXAMPLES
               + torch
                  + language_modeling
                  + diffusers
                  + ...
               + onnx # HERE ARE THE ONNX EXAMPLES
                  + image_classification
                  + language_models
                  + ...
            + ...

ONNX Examples in AMD Quark for This Release
-------------------------------------------

.. toctree::
   :caption: Improving Model Accuracy
   :maxdepth: 1

   Block Floating Point (BFP) <example_quark_onnx_BFP>
   MX Formats <example_quark_onnx_MX>
   Fast Finetune AdaRound <example_quark_onnx_adaround>
   Fast Finetune AdaQuant <example_quark_onnx_adaquant>
   Cross-Layer Equalization (CLE) <example_quark_onnx_cle>
   Layer-wise Percentile <example_quark_onnx_layerwise_percentile>
   GPTQ <example_quark_onnx_gptq>
   Mixed Precision <example_quark_onnx_mixed_precision>
   Smooth Quant <example_quark_onnx_smoothquant>
   QuaRot <example_quark_onnx_quarot>
   Auto-Search for General Yolov3 ONNX Model Quantization <example_quark_onnx_auto_search>
   Auto-Search for Ryzen AI Yolo-nas ONNX Model Quantization <example_quark_onnx_ryzenai_yolonas>
   Auto-Search for Ryzen AI Resnet50 ONNX Model Quantization <example_ryzenai_autosearch_resnet50>
   Auto-Search for Ryzen AI Yolov3 ONNX Quantization with Custom Evalutor <example_quark_onnx_ryzenai_yolov3_custom_evaluator>

.. toctree::
   :caption: Dynamic Quantization
   :maxdepth: 1

   Quantizing an Llama-2-7b Model <example_quark_onnx_dynamic_quantization_llama2>
   Quantizing an OPT-125M Model <example_quark_onnx_dynamic_quantization_opt>

.. toctree::
   :caption: Image Classification
   :maxdepth: 1

   Quantizing a ResNet50-v1-12 Model <example_quark_onnx_image_classification>

.. toctree::
   :caption: Language Models
   :maxdepth: 1

   Quantizing an OPT-125M Model <example_quark_onnx_language_models>

.. toctree::
   :caption: Weights-Only Quantization
   :maxdepth: 1

   Quantizing an Llama-2-7b Model Using the ONNX MatMulNBits <example_quark_onnx_weights_only_quant_int4_matmul_nbits_llama2>
   Quantizating Llama-2-7b model using MatMulNBits <example_quark_onnx_weights_only_quant_int8_qdq_llama2>

.. toctree::
   :caption: Crypto Mode
   :maxdepth: 1

   Quantizing a ResNet50 model in crypto mode <example_quark_onnx_crypto_mode>

.. _ryzenai_onnx_examples:
.. toctree::
   :caption: Ryzen AI Quantization
   :maxdepth: 1

   Best Practice for Quantizing an Image Classification Model <image_classification_example_quark_onnx_ryzen_ai_best_practice>
   Best Practice for Quantizing an Object Detection Model  <object_detection_example_quark_onnx_ryzen_ai_best_practice>

.. toctree::
   :caption: Hugging Face TIMM Models
   :maxdepth: 1

   Hugging Face TIMM Quantization <hugging_face_timm_quantization>

.. toctree::
   :caption: Yolo_nas and Yolox Models
   :maxdepth: 1

   Yolo_nas and Yolox Quantization <example_quark_onnx_yolo_quantization>
