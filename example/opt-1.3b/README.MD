# Running OPT on the IPU

OPT (Open Pre-trained Transformer) class of language models are released by Meta and are available through [Huggingface](https://huggingface.co/facebook/opt-1.3b).

We use eager-mode execution with quantized Linear nodes in the OPT-1.3B model. We demonstrate this using two flows: 


#### 1. **ONNX Runtime Flow**:  [ONNX Runtime (ORT) Quantization - Executed using ORT Execution Provider](opt-onnx)

This flow demonstrates the out-of-the-box solution provided by Vitis AI Execution Provider (EP) to run   transformer models on different target devices (IPU/CPU). We provide sample scripts to download a pretrained Huggingface transformer model, quantize it using ORT dynamic quantizer, and execute it on the target using the EP. This flow is presented in opt-onnx/.

#### 2. **PyTorch Flow**: [PyTorch Dynamic Quantization - Executed using PyTorch extension with custom C++ operators](opt-pytorch)

This flow attempts to provide a developer framework to implement custom PyTorch operators and execute them on the IPU. The example provided creates a custom QLinear operator to accelerate linear nodes in the model. The custom operator implementation, required third party libraries, and test cases are present in opt-pytorch/. We also provide sample scripts that download the pretrained tranformer model, quantize it using PyTorch Dynamic Quantization, replace Linear nodes with the custom implementation, and execute the model on IPU/CPU. 

