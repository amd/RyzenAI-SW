##################
Model Quantization 
##################


Quantization is the process of converting model weights and activation values from floating-point to lower-precision integer representations. Quantized models are more power-efficient, utilize less memory, and offer better performance. Ryzen AI requires INT8 quantization for inference. 

***************************
Vitis AI Quantizer for ONNX
***************************

**Vitis AI Quantizer for ONNX**: Provides an easy-to-use Post Training Quantization (PTQ) flow on the pre-trained model saved in the ONNX format. It generates a quantized ONNX model ready to be deployed with the Ryzen AI Software.

This is the recommended quantization flow for CNN-based models. 

For more details, refer to the :doc:`vai_quant/vai_q_onnx` section of this documentation.


************************
Other Quantization Flows
************************

The Ryzen AI Software supports other quantization tools that can be used in specific situations: 

**Vitis AI Quantizer for PyTorch**: Allows quantizing models through the PyTorch framework. This flow supports both post-training quantization (PTQ) and quantization-aware training (QAT) to improve model accuracy. For more details, refer to the :doc:`vai_quant/pt` section of this documentation.

**Vitis AI Quantizer for TensorFlow**: Allows quantizing models through the TensorFlow framework. This flow supports both post-training quantization (PTQ) and quantization-aware training (QAT) to improve model accuracy. For more details, refer to the :doc:`vai_quant/tf2` section of this documentation.

**Vitis AI Quantizer for Olive**: The Microsoft Olive framework provides a plugin allowing to use the Vitis AI Quantizer. Developers familiar with the Olive framework may use this flow to quantize their models. For more detail, refer to the :doc:`olive_quant` section of this documentation.


.. toctree::
   :hidden:
   :maxdepth: 1

   ./vai_quant/vai_q_onnx
   ./vai_quant/pt
   ./vai_quant/tf2
   ./olive_quant
   
..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
