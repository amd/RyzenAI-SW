Quantization Schemes
====================

AMD Quark for PyTorch is capable of handling ``per tensor``, ``per channel``
, and ``per group`` quantization, supporting both symmetric and asymmetric
methods.

-  **Per Tensor Quantization** means quantizing the tensor with one
   scalar. The scaling factor is a scalar.

-  **Per Channel Quantization** means that for each dimension, typically
   the channel dimension of a tensor, the values in the tensor are
   quantized with different quantization parameters. The scaling factor
   is a 1-D tensor, with the length of the quantization axis. For the
   input tensor with shape ``(D0, ..., Di, ..., Dn)`` and ``ch_axis=i``,
   the scaling factor is a 1-D tensor of length ``Di``.

-  **Per Group Quantization** means dividing the tensor into smaller
   blocks that are independently quantized. The scaling factor has the
   same dimension with the input tensor. For the input tensor with shape
   ``(D0, ..., Di, ..., Dn)``, ``ch_axis=i``, and ``group_size=m``,
   the scaling factor has the shape of ``(D0, ..., Di/m, ..., Dn)``.
