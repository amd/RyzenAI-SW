.. include:: /icons.txt

#############
Release Notes
#############

.. _supported-configurations:

************************
Supported Configurations
************************

Ryzen AI 1.4 Software supports AMD processors codenamed Phoenix, Hawk Point, Strix, Strix Halo, and Krackan Point. These processors can be found in the following Ryzen series:

- Ryzen 200 Series
- Ryzen 7000 Series, Ryzen PRO 7000 Series
- Ryzen 8000 Series, Ryzen PRO 8000 Series
- Ryzen AI 300 Series, Ryzen AI PRO Series, Ryzen AI Max 300 Series

For a complete list of supported devices, refer to the `processor specifications <https://www.amd.com/en/products/specifications/processors.html>`_ page (look for the "AMD Ryzen AI" column towards the right side of the table, and select "Available" from the pull-down menu).

The rest of this document will refer to Phoenix as PHX, Hawk Point as HPT, Strix and Strix Halo as STX, and Krackan Point as KRK.


*************************
Model Compatibility Table
*************************

The following table lists which types of models are supported on what hardware platforms.

.. list-table::
   :header-rows: 1

   * - Model Type
     - PHX/HPT
     - STX/KRK
   * - CNN INT8
     - |checkmark|
     - |checkmark|
   * - CNN BF16
     -
     - |checkmark|
   * - NLP BF16
     -
     - |checkmark|
   * - LLM (OGA)
     -
     - |checkmark|


***********
Version 1.4
***********

- New Features:

  - `New architecture support for Ryzen AI 300 series processors <https://www.amd.com/en/products/software/ryzen-ai-software.html#tabs-2733982b05-item-7720bb7a69-tab>`_
  - Unified support for LLMs, INT8, and BF16 models in a single release package
  - Public release for compilation of BF16 CNN and NLP models on Windows
  - `Public release of the LLM Hybrid OGA flow <https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html>`_
  - `LLM building flow for finetuned LLM <https://ryzenai.docs.amd.com/en/latest/oga_model_prepare.html>`_
  - Support for up to 16 hardware contexts on Ryzen AI 300 series processors
  - Vitis AI EP now supports the ONNX Runtime EP context cache feature (for custom handling of pre-compiled models)
  - Ryzen AI environment variables converted to VitisAI EP session options
  - Improved exception handling and fallback to CPU

- `New Hybrid execution mode LLMs <https://huggingface.co/collections/amd/ryzenai-14-llm-hybrid-models-67da31231bba0f733750a99c>`_:

  - DeepSeek-R1-Distill-Llama-8B
  - DeepSeek-R1-Distill-Qwen-1.5B
  - DeepSeek-R1-Distill-Qwen-7B
  - Gemma2-2B
  - Qwen2-1.5B
  - Qwen2-7B
  - AMD-OLMO-1B-SFT-DPO
  - Mistral-7B-Instruct-v0.1
  - Mistral-7B-Instruct-v0.2
  - Mistral-7B-v0.3
  - Llama3.1-8B-Instruct
  - Codellama-7B-Instruct

- :doc:`New BF16 model examples <examples>`:

  - Image classification
  - Finetuned DistilBERT for text classification
  - Text embedding model Alibaba-NLP/gte-large-en-v1.5

- New Tools:

  - `Lemonade SDK <https://github.com/onnx/turnkeyml/blob/main/docs/lemonade/README.md>`_ 

    - `Lemonade Server <https://github.com/onnx/turnkeyml/blob/main/docs/lemonade/README.md#serving>`_: A server interface that uses the standard Open AI API, allowing applications in any language to integrate with Lemonade Server for local LLM deployment and compatibility with existing Open AI apps.
    - `Lemonade Python API <https://github.com/onnx/turnkeyml/blob/main/docs/lemonade/README.md#api>`_: Offers High-Level API for easy integration of Lemonade LLMs into Python applications and Low-Level API for custom experiments with specific checkpoints, devices, and tools. 
    - `Lemonade Command Line <https://github.com/onnx/turnkeyml/blob/main/docs/lemonade/getting_started.md#cli-commands>`_ Interface easily benchmark, measure accuracy, prompt or gather memory usage of your LLM. 
  - `TurnkeyML <https://github.com/onnx/turnkeyml>`_ – Open-source tool that includes low-code APIs for general ONNX workflows. 
  - `Digest AI <https://github.com/onnx/digestai>`_ – A Model Ingestion and Analysis Tool in collaboration with the Linux Foundation. 
  - `GAIA <https://github.com/amd/gaia/tree/main>`_ – An open-source application designed for the quick setup and execution of generative AI applications on local PC hardware. 

- Quark-torch:

  - Added ROUGE and METEOR evaluation metrics for LLMs
  - Support for evaluating ONNX models exported using OGA
  - Support for offline evaluation (evaluation without generation) for LLMs
  - Support for Hugging Face integration
  - Support for Gemma2 quantization using the OGA flow
  - Support for Llama-3.2 quantization with FP8 (weights, activation, and KV-cache) for the vision and language components

- Quark-onnx:

  - Support compatibility with ONNX Runtime version 1.20.0, and 1.20.1
  - Support for microexponents (MX) data types, including MX4, MX6, and MX9
  - Support for BF16 data type for VAIML
  - Support for excluding pre and post-processing from quantization
  - Support for mixed precision with any data type
  - Support for Quarot rotation R1 algorithm
  - Support for microexponents and microscaling AdaQuant
  - Support for an auto-search algorithm to automatically find the best accuracy quantized model
  - Added tools for evaluating L2, PSNR, VMAF, and cosine

- ONNX Runtime EP:

  - Support for Chinese characters in the ``filename/cache_dir/cache_key/xclbin``
  - Support for ``int4/uint4`` data type
  - Support for configurable failure handling: CPU fallback or exception
  - Update for encrypt/decrypt feature

- Known Issues:

  - Microsoft Windows Insider Program (WIP) users may see warnings or need to restart when running all applications concurrently. 
  
    - NPU driver and workloads will continue to work.

  - Context creation may appear to be limited when some application do not close contexts quickly.


***********
Version 1.3
***********

- New Features:

  - Initial release of the Quark quantizer
  - Support for mixed precision data types
  - Compatibility with Copilot+ applications

- Improved support for :doc:`LLMs using OGA <llm/overview>`

- New EoU Tools:

  - CNN profiling tool for VAI-ML flow
  - Idle detection and suspension of contexts
  - Rebalance feature for AIE hardware resource optimization

- NPU and Compiler:

  - New Op Support:

    - MAC
    - QResize Bilinear
    - LUT Q-Power
    - Expand
    - Q-Hsoftmax
    - A16 Q-Pad
    - Q-Reduce-Mean along H/W dimension
    - A16 Q-Global-AvgPool
    - A16 Padding with non-zero values
    - A16 Q-Sqrt
    - Support for XINT8/XINT16 MatMul and A16W16/A8W8 Q-MatMul

  - Performance Improvements:

    - Q-Conv, Q-Pool, Q-Add, Q-Mul, Q-InstanceNorm
    - Enhanced QDQ support for a range of operations
    - Enhanced the tiling algorithm
    - Improved graph-level optimization with extra transpose removal
    - Enhanced AT/MT fusion
    - Optimized memory usage and compile time
    - Improved compilation messages

- Quark for PyTorch:

  - Model Support:

    - Examples of LLM PTQ, such as Llama3.2 and Llama3.2-Vision models
    - Example of YOLO-NAS detection model PTQ/QAT
    - Example of SDXL v1.0 with weight INT8 activation INT8

  - PyTorch Quantizer Enhancements:

    - Partial model quantization by user configuration under FX mode
    - Quantization of ConvTranspose2d in Eager Mode and FX mode
    - Advanced Quantization Algorithms with auto-generated configurations
    - Optimized Configuration with DataTypeSpec for ease of use
    - Accelerated in-place replacement under Eager Mode
    - Loading configuration from file of algorithms and pre-optimizations

- Quark for ONNX:

  - New Features:

    - Compatibility with ONNX Runtime version 1.18, 1.19
    - Support for int4, uint4, Microscaling data types
    - Quantization for arbitrary specified operators
    - Quantization type alignment of element-wise operators for mixed precision
    - ONNX graph cleaning
    - Int32 bias quantization

  - ONNX Quantizer Enhancements:

    - Fast fine-tuning support for the MatMul operator, BFP data type, and GPU acceleration
    - Improved ONNX quantization of LLM models
    - Optimized quantization of FP16 models
    - Custom operator compilation process
    - Default parameters for auto mixed precision
    - Optimized Ryzen AI workflow by aligning with hardware constraints of the NPU

- ONNX Runtime EP:

  - Support for ONNX Runtime EP shared libraries
  - Python dependency removal
  - Memory optimization during the compile phase
  - Pattern API enhancement with multiple outputs and commutable arguments support

- Known Issues:

  - Extended compile time for some models with BF16/BFP16 data types
  - LLM models with 4K sequence length may revert to CPU execution
  - Accuracy drop in some Transformer models using BF16/BFP16 data types, requiring Quark intervention

***********
Version 1.2
***********

- New features:

  - Support added for Strix Point NPUs
  - Support added for integrated GPU
  - Smart installer for Ryzen AI 1.2
  - NPU DPM based on power slider

- New model support:

  - `LLM flow support <https://ryzenai.docs.amd.com/en/latest/llm_flow.html>`_ for multiple models in both PyTorch and ONNX flow (optimized model support will be released asynchronously)
  - SDXL-T with limited performance optimization

- New EoU tools:

  - `AI Analyzer <https://ryzenai.docs.amd.com/en/latest/ai_analyzer.html>`_ : Analysis and visualization of model compilation and inference profiling
  - Platform/NPU inspection and management tool (`xrt-smi <https://ryzenai.docs.amd.com/en/latest/xrt_smi.html>`_)
  - `Onnx Benchmarking tool <https://github.com/amd/RyzenAI-SW/tree/main/onnx-benchmark>`_

- New Demos:

  - NPU-GPU multi-model pipeline application `demo <https://github.com/amd/RyzenAI-SW/tree/main/demo/NPU-GPU-Pipeline>`_

- NPU and Compiler

  - New device support: Strix Nx4 and 4x4 Overlay
  - New Op support:

    - InstanceNorm
    - Silu
    - Floating scale quantization operators (INT8, INT16)
  - Support new rounding mode (Round to even)
  - Performance Improvement:

    - Reduced the model compilation time
    - Improved instruction loading
    - Improved synchronization in large overlay
    - Enhanced strided_slice performance
    - Enhanced convolution MT fusion
    - Enhanced convolution AT fusion
    - Enhanced data movement op performance
- ONNX Quantizer updates

  - Improved usability with various features and tools, including weights-only quantization, graph optimization, dynamic shape fixing, and format transformations.
  - Improved the accuracy of quantized models through automatic mixed precision and enhanced AdaRound and AdaQuant techniques.
  - Enhanced support for the BFP data type, including more attributes and shape inference capability.
  - Optimized the NPU workflow by aligning with the hardware constraints of the NPU.
  - Supported compilation for Windows and Linux.
  - Bugfix:

    - Fixed the problem where per-channel quantization is not compatible with onnxruntime 1.17.
    - Fixed the bug of CLE when conv with groups.
    - Fixed the bug of bias correction.
- Pytorch Quantizer updates

  - Tiny value quantization protection.
  - Higher onnx version support in quantized model exporting.
  - Relu6 hardware constrains support.
  - Support of mean operation with keepdim=True.
- Resolved issues:

  - NPU SW stack will fail to initialize when the system is out of memory. This could impact camera functionality when Microsoft Effect Pack is enabled.
  - If Microsoft Effects Pack is overloaded with other 4+ applications that use NPU to do inference, then camera functionality can be impacted. Can be fixed with a reboot. This will be fixed in the next release.

***********
Version 1.1
***********

- New model support:

  - Llama 2 7B with w4abf16 (3-bit and 4-bit) quantization (Beta)
  - Whisper base (EA access)

- New EoU tools:

  - CNN Benchmarking tool on RyzenAI-SW Repo
  - Platform/NPU inspection and management tool

Quantizer
=========

- ONNX Quantizer:

  - Improved usability with various features and tools, including diverse parameter configurations, graph optimization, shape fixing, and format transformations.
  - Improved quantization accuracy through the implementation of experimental algorithmic improvements, including AdaRound and AdaQuant.
  - Optimized the NPU workflow by distinguishing between different targets and aligning with the hardware constraints of the NPU.
  - Introduced new utilities for model conversion.

- PyTorch Quantizer:

  - Mixed data type quantization enhancement and bug fix.
  - Corner bug fixes for add, sub, and conv1d operations.
  - Tool for converting the S8S8 model to the U8S8 model.
  - Tool for converting the customized Q/DQ to onnxruntime contributed Q/DQ with the "microsoft" domain.
  - Tool for fixing a dynamic shapes model to fixed shape model.

- Bug fixes

  - Fix for incorrect logging when simulating the LeakyRelu alpha value.
  - Fix for useless initializers not being cleaned up during optimization.
  - Fix for external data cannot be found when using use_external_data_format.
  - Fix for custom Ops cannot be registered due to GLIBC version mismatch

NPU and Compiler
================

- New op support:

  - Support Channel-wie Prelu.
  - Gstiling with reverse = false.
- Fixed issues:

  - Fixed Transpose-convolution and concat optimization issues.
  - Fixed Conv stride 3 corner case hang issue.
- Performance improvement:

  - Updated Conv 1x1 stride 2x2 optimization.
  - Enhanced Conv 7x7 performance.
  - Improved padding performance.
  - Enhanced convolution MT fusion.
  - Improved the performance for NCHW layout model.
  - Enhanced the performance for eltwise-like op.
  - Enhanced Conv and eltwise AT fusion.
  - Improved the output convolution/transpose-convolution’s performance.
  - Enhanced the logging message for EoU.


ONNX Runtime EP
===============

- End-2-End Application support on NPU

  - Enhanced existing support: Provided high-level APIs to enable seamless incorporation of pre/post-processing operations into the model to run on NPU
  - Two examples (resnet50 and yolov8) published to demonstrate the usage of these APIs to run end-to-end models on the NPU
- Bug fixes for ONNXRT EP to support customers’ models

Misc
====

- Contains mitigation for the following CVEs: CVE-2024-21974, CVE-2024-21975, CVE-2024-21976

*************
Version 1.0.1
*************

- Minor fix for Single click installation without given env name.
- Perform improvement in the NPU driver.
- Bug fix in elementwise subtraction in the compiler.
- Runtime stability fixes for minor corner cases.
- Quantizer update to resolve performance drop with default settings.

***********
Version 1.0
***********
Quantizer
=========

- ONNX Quantizer

  - Support for ONNXRuntime 1.16.
  - Support for the Cross-Layer-Equalization (CLE) algorithm in quantization, which can balance the weights of consecutive Conv nodes to make it more quantize-friendly in per-tensor quantization.
  - Support for mixed precision quantization including UINT16/INT16/UINT32/INT32/FLOAT16/BFLOAT16, and support asymmetric quantization for BFLOAT16.
  - Support for the MinMSE method for INT16/UINT16/INT32/UINT32 quantization.
  - Support for quantization using the INT16 scale.
  - Support for unsigned ReLU in symmetric activation configuration.
  - Support for converting Float16 to Float32 during quantization.
  - Support for converting NCHW model to NHWC model during quantization.
  - Support for two more modes for MinMSE for better accuracy. The "All" mode computes the scales with all batches while the "MostCommon" mode computes the scale for each batch and uses the most common scales.
  - Support for the quantization of more operations:

    - PReLU, Sub, Max, DepthToSpace, SpaceToDepth, Slice, InstanceNormalization, and LpNormalization.
    - Non-4D ReduceMean.
    - Leakyrelu with arbitrary alpha.
    - Split by converting it to Slice.

  - Support for op fusing of InstanceNormalization and L2Normalization in NPU workflow.
  - Support for converting Clip to ReLU when the minimal value is 0.
  - Updated shift_bias, shift_read, and shift_write constraints in the NPU workflow and added an option "IPULimitationCheck" to disable it.
  - Support for disabling the op fusing of Conv + LeakyReLU/PReLU in the NPU workflow.
  - Support for logging for quantization configurations and summary information.
  - Support for removing initializer from input to support models converted from old version pytorch where weights are stored as inputs.
  - Added a recommended configuration for the IPU_Transformer platform.
  - New utilities:

    - Tool for converting the float16 model to the float32 model.
    - Tool for converting the NCHW model to the NHWC model.
    - Tool for quantized models with random input.

  - Three examples for quantization models from Timm, Torchvision, and ONNXRuntime modelzoo respectively.
  - Bugfixes:

    - Fix a bug that weights are quantized with the "NonOverflow" method when using the "MinMSE" method.

- Pytorch Quantizer

  - Support of some operations quantization in quantizer: inplace div, inplace sub
  - Log and document enhancement to emphasize fast-finetune
  - Timm models quantization script example
  - Bug fix for operators: clamp and prelu
  - QAT Support quantization of operations with multiple outputs
  - QAT EOU enhancements: significantly reduces the need for network modifications
  - QAT ONNX exporting enhancements: support more configurations
  - New QAT examples

- TF2 Quantizer

  - Support for Tensorflow 2.11 and 2.12.
  - Support for the 'tf.linalg.matmul' operator.
  - Updated shift_bias constraints for NPU workflow.
  - Support for dumping models containing operations with multiple outputs.
  - Added an example of a sequential model.
  - Bugfixes:

    - Fix a bug that Hardsigmoid and Hardswish are not mapped to DPU without Batch Normalization.
    - Fix a bug when both align_pool and align_concat are used simultaneously.
    - Fix a bug in the sequential model when a layer has multiple consumers.

- TF1 Quantizer

  - Update shift_bias constraints for NPU workflow.
  - Bugfixes:

    - Fix a bug in fast_finetune when the 'input_node' and 'quant_node' are inconsistent.
    - Fix a bug that AddV2 op identified as BiasAdd.
    - Fix a bug when the data type of the concat op is not float.
    - Fix a bug in split_large_kernel_pool when the stride is not equal to 1.

ONNXRuntime Execution Provider
==============================

- Support new OPs, such as PRelu, ReduceSum, LpNormlization, DepthToSpace(DCR).
- Increase the percentage of model operators performed on the NPU.
- Fixed some issues causing model operators allocation to CPU.
- Improved report summary
- Support the encryption of the VOE cache
- End-2-End Application support on NPU

  - Enable running pre/post/custom ops on NPU, utilizing ONNX feature of E2E extensions.
  - Two examples published for yolov8 and resnet50, in which preprocessing custom op is added and runs on NPU.

- Performance: latency improves by up to 18% and power savings by up to 35% by additionally running preprocessing on NPU apart from inference.
- Multiple NPU overlays support

  - VOE configuration that supports both CNN-centric and GEMM-centric NPU overlays.
  - Increases number of ops that run on NPU, especially for models which have both GEMM and CNN ops.
  - Examples published for use with some of the vision transformer models.

NPU and Compiler
==============================

- New operators support

  - Global average pooling with large spatial dimensions
  - Single Activation (no fusion with conv2d, e.g. relu/single alpha PRelu)

- Operator support enhancement

  - Enlarge the width dimension support range for depthwise-conv2d
  - Support more generic broadcast for element-wise like operator
  - Support output channel not aligned with 4B GStiling
  - Support Mul and LeakyRelu fusion
  - Concatenation’s redundant input elimination
  - Channel Augmentation for conv2d (3x3, stride=2)

- Performance optimization

  - PDI partition refine to reduce the overhead for PDI swap
  - Enabled cost model for some specific models

- Fixed asynchronous error in multiple thread scenario
- Fixed known issue on tanh and transpose-conv2d hang issue

Known Issues
==============================

- Support for multiple applications is limited to up to eight
- Windows Studio Effects should be disabled when using the Latency profile. To disable Windows Studio Effects, open **Settings > Bluetooth & devices > Camera**, select your primary camera, and then disable all camera effects.



***********
Version 0.9
***********

Quantizer
=========

- Pytorch Quantizer

  - Dict input/output support for model forward function
  - Keywords argument support for model forward function
  - Matmul subroutine quantization support
  - Support of some operations in quantizer: softmax, div, exp, clamp
  - Support quantization of some non-standard conv2d.


- ONNX Quantizer

  - Add support for Float16 and BFloat16 quantization.
  - Add C++ kernels for customized QuantizeLinear and DequantizeLinaer operations.
  - Support saving quantizer version info to the quantized models' producer field.
  - Support conversion of ReduceMean to AvgPool in NPU workflow.
  - Support conversion of BatchNorm to Conv in NPU workflow.
  - Support optimization of large kernel GlobalAvgPool and AvgPool operations in NPU workflow.
  - Supports hardware constraints check and adjustment of Gemm, Add, and Mul operations in NPU workflow.
  - Supports quantization for LayerNormalization, HardSigmoid, Erf, Div, and Tanh for NPU.

ONNXRuntime Execution Provider
==============================

- Support new OPs, such as Conv1d, LayerNorm, Clip, Abs, Unsqueeze, ConvTranspose.
- Support pad and depad based on NPU subgraph’s inputs and outputs.
- Support for U8S8 models quantized by ONNX quantizer.
- Improve report summary tools.

NPU and Compiler
================

- Supported exp/tanh/channel-shuffle/pixel-unshuffle/space2depth
- Performance uplift of xint8 output softmax
- Improve the partition messages for CPU/DPU
- Improve the validation check for some operators
- Accelerate the speed of compiling large models
- Fix the elew/pool/dwc/reshape mismatch issue and fix the stride_slice hang issue
- Fix str_w != str_h issue in Conv


LLM
===

- Smoothquant for OPT1.3b, 2.7b, 6.7b, 13b models.
- Huggingface Optimum ORT Quantizer for ONNX and Pytorch dynamic quantizer for Pytorch
- Enabled Flash attention v2 for larger prompts as a custom torch.nn.Module
- Enabled all CPU ops in bfloat16 or float32 with Pytorch
- int32 accumulator in AIE (previously int16)
- DynamicQuantLinear op support in ONNX
- Support different compute primitives for prefill/prompt and token phases
- Zero copy of weights shared between different op primitives
- Model saving after quantization and loading at runtime for both Pytorch and ONNX
- Enabled profiling prefill/prompt and token time using local copy of OPT Model with additional timer instrumentation
- Added demo mode script with greedy, stochastic and contrastive search options

ASR
===
- Support Whipser-tiny
- All GEMMs offloaded to AIE
- Improved compile time
- Improved WER

Known issues
============

- Flow control OPs including "Loop", "If", "Reduce" not supported by VOE
- Resizing OP in ONNX opset 10 or lower is not supported by VOE
- Tensorflow 2.x quantizer supports models within tf.keras.model only
- Running quantizer docker in WSL on Ryzen AI laptops may encounter OOM (Out-of-memory) issue
- Running multiple concurrent models using temporal sharing on the 5x4 binary is not supported
- Only batch sizes of 1 are supported
- Only models with the pretrained weights setting = TRUE should be imported
- Launching multiple processes on 4 1x4 binaries can cause hangs, especially when models have many sub-graphs

|
|

***********
Version 0.8
***********

Quantizer
=========

- Pytorch Quantizer

  - Pytorch 1.13 and 2.0 support
  - Mixed precision quantization support, supporting float32/float16/bfloat16/intx mixed quantization
  - Support of bit-wise accuracy cross check between quantizer and ONNX-runtime
  - Split and chunk operators were automatically converted to slicing
  - Add support for BFP data type quantization
  - Support of some operations in quantizer: where, less, less_equal, greater, greater_equal, not, and, or, eq, maximum, minimum, sqrt, Elu, Reduction_min, argmin
  - QAT supports training on multiple GPUs
  - QAT supports operations with multiple inputs or outputs

- ONNX Quantizer

  - Provided Python wheel file for installation
  - Support OnnxRuntime 1.15
  - Supports setting input shapes of random data reader
  - Supports random data reader in the dump model function
  - Supports saving the S8S8 model in U8S8 format for NPU
  - Supports simulation of Sigmoid, Swish, Softmax, AvgPool, GlobalAvgPool, ReduceMean and LeakyRelu for NPU
  - Supports node fusions for NPU

ONNXRuntime Execution Provider 
==============================

- Supports for U8S8 quantized ONNX models
- Improve the function of falling back to CPU EP
- Improve AIE plugin framework

  - Supports LLM Demo
  - Supports Gemm ASR
  - Supports E2E AIE acceleration for Pre/Post ops
  - Improve the easy-of-use for partition and  deployment
- Supports  models containing subgraphs
- Supports report summary about OP assignment
- Supports report summary about DPU subgraphs falling back to CPU
- Improve log printing and troubleshooting tools.
- Upstreamed to ONNX Runtime Github repo for any data type support and bug fix

NPU and Compiler
================

- Extended the support range of some operators

  - Larger input size: conv2d, dwc
  - Padding mode: pad
  - Broadcast: add
  - Variant dimension (non-NHWC shape): reshape, transpose, add
- Support new operators, e.g. reducemax(min/sum/avg), argmax(min)
- Enhanced multi-level fusion
- Performance enhancement for some operators
- Add quantization information validation
- Improvement in device partition

  - User friendly message
  - Target-dependency check

Demos
=====

- New Demos link: https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=transformers_2308.zip

  - LLM demo with OPT-1.3B/2.7B/6.7B
  - Automatic speech recognition demo with Whisper-tiny

Known issues
============
- Flow control OPs including "Loop", "If", "Reduce" not supported by VOE
- Resize OP in ONNX opset 10 or lower not supported by VOE
- Tensorflow 2.x quantizer supports models within tf.keras.model only
- Running quantizer docker in WSL on Ryzen AI laptops may encounter OOM (Out-of-memory) issue
- Run multiple concurrent models by temporal sharing on the Performance optimized overlay (5x4.xclbin) is not supported
- Support batch size 1 only for NPU


|
|

***********
Version 0.7
***********

Quantizer
=========

- Docker Containers

  - Provided CPU dockers for Pytorch, Tensorflow 1.x, and Tensorflow 2.x quantizer
  - Provided GPU Docker files to build GPU dockers

- Pytorch Quantizer

  - Supports multiple output conversion to slicing
  - Enhanced transpose OP optimization
  - Inspector support new IP targets for NPU

- ONNX Quantizer

  - Provided Python wheel file for installation
  - Supports quantizing ONNX models for NPU as a plugin for the ONNX Runtime native quantizer
  - Supports power-of-two quantization with both QDQ and QOP format
  - Supports Non-overflow and Min-MSE quantization methods
  - Supports various quantization configurations in power-of-two quantization in both QDQ and QOP format.

    - Supports signed and unsigned configurations.
    - Supports symmetry and asymmetry configurations.
    - Supports per-tensor and per-channel configurations.
  - Supports bias quantization using int8 datatype for NPU.
  - Supports quantization parameters (scale) refinement for NPU.
  - Supports excluding certain operations from quantization for NPU.
  - Supports ONNX models larger than 2GB.
  - Supports using CUDAExecutionProvider for calibration in quantization
  - Open source and upstreamed to Microsoft Olive Github repo

- TensorFlow 2.x Quantizer

  - Added support for exporting the quantized model ONNX format.
  - Added support for the keras.layers.Activation('leaky_relu')

- TensorFlow 1.x Quantizer

  - Added support for folding Reshape and ResizeNearestNeighbor operators.
  - Added support for splitting Avgpool and Maxpool with large kernel sizes into smaller kernel sizes.
  - Added support for quantizing Sum, StridedSlice, and Maximum operators.
  - Added support for setting the input shape of the model, which is useful in deploying models with undefined input shapes.
  - Add support for setting the opset version in exporting ONNX format

ONNX Runtime Execution Provider
===============================

- Vitis ONNX Runtime Execution Provider (VOE)

  - Supports ONNX Opset version 18, ONNX Runtime 1.16.0, and ONNX version 1.13
  - Supports both C++ and Python APIs(Python version 3)
  - Supports deploy model with other EPs
  - Supports falling back to CPU EP
  - Open source and upstreamed to ONNX Runtime Github repo
  - Compiler

    - Multiple Level op fusion
    - Supports the  same muti-output operator like chunk split
    - Supports split big pooling to small pooling
    - Supports 2-channel writeback feature for Hard-Sigmoid and Depthwise-Convolution
    - Supports 1-channel GStiling
    - Explicit pad-fix in CPU subgraph for 4-byte alignment
    - Tuning the performance for multiple models

NPU
===

- Two configurations

  - Power Optimized Overlay

    - Suitable for smaller AI models (1x4.xclbin)
    - Supports spatial sharing, up to 4 concurrent AI workloads

  - Performance Optimized Overlay (5x4.xclbin)

    - Suitable for larger AI models

Known issues
============
- Flow control OPs including "Loop", "If", "Reduce" are not supported by VOE
- Resize OP in ONNX opset 10 or lower not supported by VOE
- Tensorflow 2.x quantizer supports models within tf.keras.model only
- Running quantizer docker in WSL on Ryzen AI laptops may encounter OOM (Out-of-memory) issue
- Run multiple concurrent models by temporal sharing on the Performance optimized overlay (5x4.xclbin) is not supported




..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
