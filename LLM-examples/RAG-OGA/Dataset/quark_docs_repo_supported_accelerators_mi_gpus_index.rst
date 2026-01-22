Quark for AMD Instinct Accelerators
===================================

Depending on the GPU to be used, different quantization schemes may or may not have accelerated support in the underlying hardware.

On all GPUs supported by PyTorch, quantized models can be evaluated using fake quantization (quantize-dequantize), effectively using a higher widely supported precision for compute (e.g.,  ``float16``).

.. note::

    As an example, AMD Instinct MI300 supports ``float8`` compute, which means that linear layers quantized in ``float8`` for both the activation and weights may use ``float8 @ float8 -> float16`` computation.

    On the other hand, Instinct MI210 and Instinct MI250 GPUs (CDNA2 architecture) do not support ``float8`` computations, and only ``QDQ`` can be used for this specific ``dtype`` and hardware.

Below are some references on how you can leverage Quark to seamlessly run accelerated quantized models on AMD Instinct GPUs:

.. toctree::
   :caption: Resources
   :maxdepth: 1

   FP8 (OCP fp8_e4m3) Quantization & Json_SafeTensors_Export with KV Cache <../../pytorch/example_quark_torch_llm_ptq>
   Evaluation of Quantized Models <../../pytorch/example_quark_torch_llm_eval_perplexity>
