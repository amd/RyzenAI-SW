###################
Release Information
###################

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
