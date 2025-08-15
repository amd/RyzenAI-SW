Quantization Schemes
====================

AMD Quark for ONNX is capable of handling ``per tensor`` and ``per channel``
quantization, supporting both symmetric and asymmetric methods.

-  **Per Tensor Quantization** means quantizing the tensor with one
   scalar. The scaling factor is a scalar.

-  **Per Channel Quantization** means that for each dimension, typically the channel dimension of a tensor, you quantize the values in the tensor with different quantization parameters. The scaling factor is a 1-D tensor with the length of the quantization axis. For the input tensor with shape ``(D0, ..., Di, ..., Dn)`` and ``ch_axis=i``, the scaling factor is a 1-D tensor of length ``Di``.

