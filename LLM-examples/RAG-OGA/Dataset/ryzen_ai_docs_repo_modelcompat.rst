.. include:: icons.txt

###################
Model Compatibility
###################

The Ryzen AI Software supports deploying quantized model saved in the ONNX format.

Currently, the NPU supports a subset of the ONNX operators. At runtime, the ONNX graph is automatically partitioned into multiple subgraphs by the Vitis AI ONNX Execution Provider (VAI EP). The subgraph(s) containing operators supported by the NPU are executed on the NPU. The remaining subgraph(s) are executed on the CPU. This graph partitioning and deployment technique across CPU and NPU is fully automated by the VAI EP and is totally transparent to the end-user.

|memo| **NOTE**: Models with ONNX opset 17 are recommended. If your model uses a different opset version, consider converting it using the `ONNX Version Converter <https://github.com/onnx/onnx/blob/main/docs/VersionConverter.md>`_


The Ryzen AI compiler supports input models quantized to either INT8 or BF16 format:

- CNN models: INT8 or BF16
- Transformer models: BF16

BF16 models (CNN or Transformer) require processing power in terms of core count and memory, depending on model size. If a larger model cannot be compiled on a Windows machine due to hardware limitations (e.g., insufficient RAM), an alternative Linux-based compilation flow is supported. More details can be found here: <link>.

The list of the ONNX operators currently supported by the NPU is as follows:

<TBD>

..
  ------------

  #####################################
  License
  #####################################

  Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
