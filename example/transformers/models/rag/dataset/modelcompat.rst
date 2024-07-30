###################
Model Compatibility
###################

The Ryzen AI Software supports deploying quantized model saved in the ONNX format. 

Currently, the NPU supports a subset of the ONNX operators. At runtime, the ONNX graph is automatically partitioned into multiple subgraphs by the Vitis AI ONNX Execution Provider (VAI EP). The subgraph(s) containing operators supported by the NPU are executed on the NPU. The remaining subgraph(s) are executed on the CPU. This graph partitioning and deployment technique across CPU and NPU is fully automated by the VAI EP and is totally transparent to the end-user.

The list of the ONNX operators currently supported by the NPU is as follows:

- Abs
- Add
- And
- Argmax
- Argmin
- Average pool_2D
- Channel Shuffle
- Clip
- Concat
- Convolution
- ConvTranspose
- Depthwise_Convolution
- Div
- Elu
- Equal
- Exp
- Fully-Connected
- Gemm
- GlobalAveragePool
- Greater
- GreaterOrEqual
- Gstiling
- Hard-Sigmoid
- Hard-Swish
- Identity
- LeakyRelu
- Less
- LessOrEqual
- MatMul
- Max
- Min
- MaxPool
- Mul
- Neg
- Not
- Or
- Pad: constant or symmetric
- Pixel-Shuffle
- Pixel-Unshuffle
- Prelu
- ReduceMax
- ReduceMin
- ReduceMean
- ReduceSum
- Relu
- Reshape
- Resize
- Slice
- Softmax
- SpaceToDepth
- Sqrt
- Squeeze
- Strided-Slice
- Sub
- Tanh
- Upsample

..
  ------------

  #####################################
  License
  #####################################

  Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
