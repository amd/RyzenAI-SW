###########################
Vitis AI Quantizer for ONNX 
###########################

********
Overview
********

The AMD-Xilinx Vitis AI Quantizer for ONNX models. It supports various configuration and functions to quantize models targeting for deployment on IPU_CNN, IPU_Transformer and CPU. It is customized based on `Quantization Tool <https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/quantization>`_ in ONNX Runtime.

The Vitis AI Quantizer for ONNX supports Post Training Quantization. This static quantization method first runs the model using a set of inputs called calibration data. During these runs, the flow computes the quantization parameters for each activation. These quantization parameters are written as constants to the quantized model and used for all inputs. The quantization tool supports the following calibration methods: MinMax, Entropy and Percentile, and MinMSE.

.. note::
   In this documentation, **"NPU"** is used in descriptions, while **"IPU"** is retained in the tool's language, code, screenshots, and commands. This intentional 
   distinction aligns with existing tool references and does not affect functionality. Avoid making replacements in the code.

************
Installation
************

If you have prepared your working environment using the :ref:`automatic installation script <install-bundled>`, the Vitis AI Quantizer for ONNX is already installed. 

Otherwise, ensure that the Vitis AI Quantizer for ONNX is correctly installed by following the :ref:`installation instructions <install-onnx-quantizer>`.
 
  
******************
Running vai_q_onnx
******************
  
Quantization in ONNX refers to the linear quantization of an ONNX model. The ``vai_q_onnx`` tool is as a plugin for the ONNX Runtime. It offers powerful post-training quantization (PTQ) functions to quantize machine learning models. Post-training quantization (PTQ) is a technique to convert a pre-trained float model into a quantized model with little degradation in model accuracy. A representative dataset is needed to run a few batches of inference on the float model to obtain the distributions of the activations, which is also called quantized calibration.

Use the following steps to run PTQ with vai_q_onnx.


1. Preparing the Float Model and Calibration Set 
================================================

Before running ``vai_q_onnx``, prepare the float model and calibration set, including these files:

- float model: Floating-point models in ONNX format.
- calibration dataset: A subset of the training dataset or validation dataset to represent the input data distribution; usually 100 to 1000 images are enough.

**Exporting PyTorch Models to ONNX**

For PyTorch models, it is recommended to use the TorchScript-based onnx exporter for exporting ONNX models. Please refer to the `PyTorch documentation for guidance <https://pytorch.org/docs/stable/onnx_torchscript.html#torchscript-based-onnx-exporte>`_

Tips:

- Before exporting, please perform model.eval().
- Models with opset 17 are recommended.
- For CNN's on NPU platform, dynamic input shapes are currently not supported and only a batch size of 1 is allowed. Please ensure that the shape of input is a fixed value, and the batch dimension is set to 1.

Example code:

.. code-block::
   
   torch.onnx.export(
      model,
      input,
      model_output_path,
      opset_version=13,
      input_names=['input'],
      output_names=['output'],
   )


.. note::
   * **Opset Versions**:The ONNX models must be opset 10 or higher (recommended setting 13) to be quantized by Vitis AI ONNX Quantizer. Models with opset < 10 must be reconverted to ONNX from their original framework using opset 10 or above. Alternatively, you can refer to the usage of the version converter for `ONNX Version Converter <https://github.com/onnx/onnx/blob/main/docs/VersionConverter.md>`_
   
   * **Large Models > 2GB**: Due to the 2GB file size limit of Protobuf, for ONNX models exceeding 2GB, additional data will be stored separately. Please ensure that the .onnx file and the data file are placed in the same directory. Also, please set the use_external_data_format parameter to True for large models when quantizing.


2. (Recommended) Pre-processing on the Float Model
==================================================

.. note:: 
   ONNX model optimization cannot output a model size greater than 2GB. For models larger than 2GB, the optimization step must be skipped.

Pre-processing transforms a float model to prepare it for quantization. It consists of the following three optional steps:

- Symbolic shape inference: It is best-suited for transformer models.
- Model Optimization: This step uses the ONNX Runtime native library to rewrite the computation graph, including merging computation nodes, and eliminating redundancies to improve runtime efficiency.
- ONNX shape inference.

The goal of these steps is to improve the quantization quality. The ONNX Runtime quantization tool works best when the tensor’s shape is known. Both symbolic shape inference and ONNX shape inference help figure out tensor shapes. Symbolic shape inference works best with transformer-based models, and ONNX shape inference works with other models.

Model optimization performs certain operator fusion that makes the quantization tool’s job easier. For instance, a Convolution operator followed by BatchNormalization can be fused into one during the optimization, which can be quantized very efficiently.

Pre-processing API is in the Python module ``onnxruntime.quantization.shape_inference``, function ``quant_pre_process()``.

.. code-block::

   from onnxruntime.quantization import shape_inference

   shape_inference.quant_pre_process(
      input_model_path: str,
      output_model_path: str,
      skip_optimization: bool = False,
      skip_onnx_shape: bool = False,
      skip_symbolic_shape: bool = False,
      auto_merge: bool = False,
      int_max: int = 2**31 - 1,
      guess_output_rank: bool = False,
      verbose: int = 0,
      save_as_external_data: bool = False,
      all_tensors_to_one_file: bool = False,
      external_data_location: str = "./",
      external_data_size_threshold: int = 1024,)


**Arguments**

* **input_model_path**: (String) Specifies the file path of the input model that is to be pre-processed for quantization.

* **output_model_path**: (String) Specifies the file path to save the pre-processed model.

* **skip_optimization**: (Boolean) Indicates whether to skip the model optimization step. If set to True, model optimization is skipped, which may cause ONNX shape inference failure for some models. The default value is False.

* **skip_onnx_shape**: (Boolean) Indicates whether to skip the ONNX shape inference step. The symbolic shape inference is most effective with transformer-based models. Skipping all shape inferences may reduce the effectiveness of quantization, as a tensor with an unknown shape cannot be quantized. The default value is False.

* **skip_symbolic_shape**: (Boolean) Indicates whether to skip the symbolic shape inference step. Symbolic shape inference is most effective with transformer-based models. Skipping all shape inferences may reduce the effectiveness of quantization, as a tensor with an unknown shape cannot be quantized. The default value is False.

* **auto_merge**: (Boolean) Determines whether to automatically merge symbolic dimensions when a conflict occurs during symbolic shape inference. The default value is False.

* **int_max**: (Integer) Specifies the maximum integer value that is to be considered as boundless for operations like slice during symbolic shape inference. The default value is 2**31 - 1.

* **guess_output_rank**: (Boolean) Indicates whether to guess the output rank to be the same as input 0 for unknown operations. The default value is False.

* **verbose**: (Integer) Controls the level of detailed information logged during inference. 

  - 0 turns off logging (default)
  - 1 logs warnings
  - 3 logs detailed information. 
  
* **save_as_external_data**: (Boolean) Determines whether to save the ONNX model to external data. The default value is False.

* **all_tensors_to_one_file**: (Boolean) Indicates whether to save all the external data to one file. The default value is False.

* **external_data_location**: (String) Specifies the file location where the external file is saved. The default value is "./".

* **external_data_size_threshold**: (Integer) Specifies the size threshold for external data. The default value is 1024.


3. Quantizing Using the vai_q_onnx API
======================================

The static quantization method first runs the model using a set of inputs called calibration data. During these runs, the quantization parameters for each activation are computed. These quantization parameters are written as constants to the quantized model and used for all inputs. Vai_q_onnx quantization tool has expanded calibration methods to power-of-2 scale/float scale quantization methods. Float scale quantization methods include MinMax, Entropy, and Percentile. Power-of-2 scale quantization methods include MinMax and MinMSE.

.. code-block::

  vai_q_onnx.quantize_static(
   model_input,
   model_output,
   calibration_data_reader,
   quant_format=vai_q_onnx.QuantFormat.QDQ,
   calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
   input_nodes=[],
   output_nodes=[],
   op_types_to_quantize=[],
   random_data_reader_input_shape=[],
   per_channel=False,
   reduce_range=False,
   activation_type=vai_q_onnx.QuantType.QInt8,
   weight_type=vai_q_onnx.QuantType.QInt8,
   nodes_to_quantize=None,
   nodes_to_exclude=None,
   optimize_model=True,
   use_external_data_format=False,
   execution_providers=['CPUExecutionProvider'],
   enable_dpu=False,
   convert_fp16_to_fp32=False,
   convert_nchw_to_nhwc=False,
   inclue_cle=False,
   extra_options={},)


**Arguments**

* **model_input**: (String) This parameter specifies the file path of the model that is to be quantized.
* **model_output**: (String) This parameter specifies the file path where the quantized model will be saved.
* **calibration_data_reader**: (Object or None) This parameter is a calibration data reader that enumerates the calibration data and generates inputs for the original model. If you wish to use random data for a quick test, you can set calibration_data_reader to None.
* **quant_format**: (String) This parameter is used to specify the quantization format of the model. It has the following options:

  -  ``vai_q_onnx.QuantFormat.QOperator``: This option quantizes the model directly using quantized operators.
  -  ``vai_q_onnx.QuantFormat.QDQ``: This option quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor. It supports 8-bit quantization only.
  -  ``vai_q_onnx.VitisQuantFormat.QDQ``: This option quantizes the model by inserting VitisQuantizeLinear/VitisDequantizeLinear into the tensor. It supports a wider range of bit-widths and precisions.
  -  ``vai_q_onnx.VitisQuantFormat.FixNeuron``: (Experimental) This option quantizes the model by inserting FixNeuron (a combination of QuantizeLinear and DeQuantizeLinear) into the tensor. This quant format is currently experimental and should not be used for actual deployment.

* **calibrate_method**: (String) The method used in calibration, default to ``vai_q_onnx.PowerOfTwoMethod.MinMSE``.

  - For CNNs running on the NPU, power-of-two methods should be used, options are:

    - ``vai_q_onnx.PowerOfTwoMethod.NonOverflow``: This method get the power-of-two quantize parameters for each tensor to make sure min/max values not overflow.
    - ``vai_q_onnx.PowerOfTwoMethod.MinMSE``: This method get the power-of-two quantize parameters for each tensor to minimize the mean-square-loss of quantized values and float values. This takes longer time but usually gets better accuracy.

  - For Transformers running on the NPU, or for CNNs running on the CPU, float scale methods should be used, options are:

    -  ``vai_q_onnx.CalibrationMethod.MinMax``: This method obtains the quantization parameters based on the minimum and maximum values of each tensor.
    -  ``vai_q_onnx.CalibrationMethod.Entropy``: This method determines the quantization parameters by considering the entropy algorithm of each tensor's distribution.
    -  ``vai_q_onnx.CalibrationMethod.Percentile``: This method calculates quantization parameters using percentiles of the tensor values.

* **input_nodes**: (List of Strings) This parameter is a list of the names of the starting nodes to be quantized. Nodes in the model before these nodes will not be quantized. For example, this argument can be used to skip some pre-processing nodes or stop the first node from being quantized. The default value is an empty list ([]).
* **output_nodes**: (List of Strings) This parameter is a list of the names of the end nodes to be quantized. Nodes in the model after these nodes will not be quantized. For example, this argument can be used to skip some post-processing nodes or stop the last node from being quantized. The default value is an empty list ([]).
* **op_types_to_quantize**: (List of Strings or None) If specified, only operators of the given types will be quantized (e.g., ['Conv'] to only quantize Convolutional layers). By default, all supported operators will be quantized.
* **random_data_reader_input_shape**: (List or Tuple of Int) If dynamic axes of inputs require specific value, users should provide its shapes when using internal random data reader (That is, set calibration_data_reader to None). The basic format of shape for single input is list (Int) or tuple (Int) and all dimensions should have concrete values (batch dimensions can be set to 1). For example, random_data_reader_input_shape=[1, 3, 224, 224] or random_data_reader_input_shape=(1, 3, 224, 224) for single input. If the model has multiple inputs, it can be fed in list (shape) format, where the list order is the same as the onnxruntime got inputs. For example, random_data_reader_input_shape=[[1, 1, 224, 224], [1, 2, 224, 224]] for 2 inputs. Moreover, it is possible to use dict {name : shape} to specify a certain input, for example, random_data_reader_input_shape={"image" : [1, 3, 224, 224]} for the input named "image". The default value is an empty list ([]).
* **per_channel**: (Boolean) Determines whether weights should be quantized per channel. The default value is False. For DPU/NPU devices, this must be set to False as they currently do not support per-channel quantization.
* **reduce_range**: (Boolean) If True, quantizes weights with 7-bits. The default value is False. For DPU/NPU devices, this must be set to False as they currently do not support reduced range quantization.
* **activation_type**: (QuantType) Specifies the quantization data type for activations, options please refer to Table 1. The default is ``vai_q_onnx.QuantType.QInt8``.
* **weight_type**: (QuantType) Specifies the quantization data type for weights, options please refer to Table 1. The default is ``vai_q_onnx.QuantType.QInt8``. For NPU devices, this must be set to ``QuantType.QInt8``.
* **nodes_to_quantize**: (List of Strings or None) If specified, only the nodes in this list are quantized. The list should contain the names of the nodes, for example, ['Conv__224', 'Conv__252']. The default value is an empty list ([]).
* **nodes_to_exclude**: (List of Strings or None) If specified, the nodes in this list will be excluded from quantization. The default value is an empty list ([]).
* **optimize_model**: (Boolean) If True, optimizes the model before quantization. The default value is True.
* **use_external_data_format**: (Boolean) This option is used for large size (>2GB) model. The model proto and data will be stored in separate files. The default is False.
* **execution_providers**: (List of Strings) This parameter defines the execution providers that will be used by ONNX Runtime to do calibration for the specified model. The default value ``CPUExecutionProvider`` implies that the model will be computed using the CPU as the execution provider. You can also set this to other execution providers supported by ONNX Runtime such as ``CUDAExecutionProvider`` for GPU-based computation, if they are available in your environment. The default is ['CPUExecutionProvider'].
* **enable_dpu**: (Boolean) This parameter is a flag that determines whether to generate a quantized model that is suitable for the DPU/NPU. If set to True, the quantization process will consider the specific limitations and requirements of the DPU/NPU, thus creating a model that is optimized for DPU/NPU computations. The default is False.
* **convert_fp16_to_fp32**: (Boolean) This parameter controls whether to convert the input model from float16 to float32 before quantization. For float16 models, it is recommended to set this parameter to True. The default value is False.
* **convert_nchw_to_nhwc**: (Boolean) This parameter controls whether to convert the input NCHW model to input NHWC model before quantization. For input NCHW models, it is recommended to set this parameter to True. The default value is False.
* **include_cle**: (Boolean) This parameter is a flag that determines whether to optimize the models using CrossLayerEqualization; it can improve the accuracy of some models. The default is False.
* **extra_options**: (Dictionary or None) Contains key-value pairs for various options in different cases. Current used:

  - **ActivationSymmetric**: (Boolean) If True, symmetrize calibration data for activations. The default is False.
  - **WeightSymmetric**: (Boolean) If True, symmetrize calibration data for weights. The default is True.
  - **UseUnsignedReLU**: (Boolean) If True, the output tensor of ReLU and Clip, whose min is 0, will be forced to be asymmetric. The default is False.
  - **QuantizeBias**: (Boolean) If True, quantize the Bias as a normal weights. The default is True. For DPU/NPU devices, this must be set to True.
  - **RemoveInputInit**: (Boolean) If True, initializer in graph inputs will be removed because it will not be treated as constant value/weight. This may prevent some of the graph optimizations, like const folding. The default is True.
  - **EnableSubgraph**: (Boolean) If True, the subgraph will be quantized. The default is False. More support for this feature is planned in the future.
  - **ForceQuantizeNoInputCheck**: (Boolean) If True, latent operators such as maxpool and transpose will always quantize their inputs, generating quantized outputs even if their inputs have not been quantized. The default behavior can be overridden for specific nodes using nodes_to_exclude.
  - **MatMulConstBOnly**: (Boolean) If True, only MatMul operations with a constant 'B' will be quantized. The default is False.
  - **AddQDQPairToWeight**: (Boolean) If True, both QuantizeLinear and DeQuantizeLinear nodes are inserted for weight, maintaining its floating-point format. The default is False, which quantizes floating-point weight and feeds it solely to an inserted DeQuantizeLinear node. In the PowerOfTwoMethod calibration method, this setting will also be effective for the bias.
  - **OpTypesToExcludeOutputQuantization**: (List of Strings or None) If specified, the output of operators with these types will not be quantized. The default is an empty list.
  - **DedicatedQDQPair**: (Boolean) If True, an identical and dedicated QDQ pair is created for each node. The default is False, allowing multiple nodes to share a single QDQ pair as their inputs.
  - **QDQOpTypePerChannelSupportToAxis**: (Dictionary) Sets the channel axis for specific operator types (e.g., {'MatMul': 1}). This is only effective when per-channel quantization is supported and per_channel is True. If a specific operator type supports per-channel quantization but no channel axis is explicitly specified, the default channel axis will be used. For DPU/NPU devices, this must be set to {} as per-channel quantization is currently unsupported. The default is an empty dict ({}).
  - **UseQDQVitisCustomOps**: (Boolean) If True, The UInt8 and Int8 quantization will be executed by the custom operations library, otherwise by the library of onnxruntime extensions. The default is True, only valid in vai_q_onnx.VitisQuantFormat.QDQ.
  - **CalibTensorRangeSymmetric**: (Boolean) If True, the final range of the tensor during calibration will be symmetrically set around the central point "0". The default is False. In PowerOfTwoMethod calibration method, the default is True.
  - **CalibMovingAverage**: (Boolean) If True, the moving average of the minimum and maximum values will be computed when the calibration method selected is MinMax. The default is False. In PowerOfTwoMethod calibration method, this should be set to False.
  - **CalibMovingAverageConstant**: (Float) Specifies the constant smoothing factor to use when computing the moving average of the minimum and maximum values. The default is 0.01. This is only effective when the calibration method selected is MinMax and CalibMovingAverage is set to True. In PowerOfTwoMethod calibration method, this option is unsupported.
  - **RandomDataReaderInputDataRange**: (Dict or None) Specifies the data range for each inputs if used random data reader (calibration_data_reader is None). Currently, if set to None then the random value will be 0 or 1 for all inputs, otherwise range [-128,127] for unsigned int, range [0,255] for signed int and range [0,1] for other float inputs. The default is None.
  - **Int16Scale**: (Boolean) If True, the float scale will be replaced by the closest value corresponding to M and 2**N, where the range of M and 2**N is within the representation range of int16 and uint16. The default is False.
  - **MinMSEMode**: (String) When using ``vai_q_onnx.PowerOfTwoMethod.MinMSE``, you can specify the method for calculating minmse. By default, minmse is calculated using all calibration data. Alternatively, you can set the mode to "MostCommon", where minmse is calculated for each batch separately and take the most common value. The default setting is 'All'.
  - **ConvertBNToConv**: (Boolean) If True, the BatchNormalization operation will be converted to Conv operation when enable_dpu is True. The default is True.
  - **ConvertReduceMeanToGlobalAvgPool**: (Boolean) If True, the Reduce Mean operation will be converted to Global Average Pooling operation when enable_dpu is True. The default is True.
  - **SplitLargeKernelPool**: (Boolean) If True, the large kernel Global Average Pooling operation will be split into multiple Average Pooling operation when enable_dpu is True. The default is True.
  - **ConvertSplitToSlice**: (Boolean) If True, the Split operation will be converted to Slice operation when enable_dpu is True. The default is True.
  - **FuseInstanceNorm**: (Boolean) If True, the split instance norm operation will be fused to InstanceNorm operation when enable_dpu is True. The default is False.
  - **FuseL2Norm**: (Boolean) If True, a set of L2norm operations will be fused to L2Norm operation when enable_dpu is True. The default is False.
  - **ConvertClipToRelu**: (Boolean) If True, the Clip operations that has a min value of 0 will be converted to ReLU operations. The default is False.
  - **SimulateDPU**: (Boolean) If True, a simulation transformation that replaces some operations with an approximate implementation will be applied for DPU when enable_dpu is True. The default is True.
  - **ConvertLeakyReluToDPUVersion**: (Boolean) If True, the Leaky Relu operation will be converted to DPU version when SimulateDPU is True. The default is True.
  - **ConvertSigmoidToHardSigmoid**: (Boolean) If True, the Sigmoid operation will be converted to Hard Sigmoid operation when SimulateDPU is True. The default is True.
  - **ConvertHardSigmoidToDPUVersion**: (Boolean) If True, the Hard Sigmoid operation will be converted to DPU version when SimulateDPU is True. The default is True.
  - **ConvertAvgPoolToDPUVersion**: (Boolean) If True, the global or kernel-based Average Pooling operation will be converted to DPU version when SimulateDPU is True. The default is True.
  - **ConvertReduceMeanToDPUVersion**: (Boolean) If True, the ReduceMean operation will be converted to DPU version when SimulateDPU is True. The default is True.
  - **ConvertSoftmaxToDPUVersion**: (Boolean) If True, the Softmax operation will be converted to DPU version when SimulateDPU is True. The default is False.
  - **SimulateDPU**: (Boolean) If True, a simulation transformation that replaces some operations with an approximate implementation will be applied for DPU when enable_dpu is True. The default is True.
  - **IPULimitationCheck**: (Boolean) If True, the quantization scale will be adjust due to the limitation of DPU/NPU. The default is True.
  - **AdjustShiftCut**: (Boolean) If True, adjust the shift cut of nodes when IPULimitationCheck is True. The default is True.
  - **AdjustShiftBias**: (Boolean) If True, adjust the shift bias of nodes when IPULimitationCheck is True. The default is True.
  - **AdjustShiftRead**: (Boolean) If True, adjust the shift read of nodes when IPULimitationCheck is True. The default is True.
  - **AdjustShiftWrite**: (Boolean) If True, adjust the shift write of nodes when IPULimitationCheck is True. The default is True.
  - **AdjustHardSigmoid**: (Boolean) If True, adjust the pos of hard sigmoid nodes when IPULimitationCheck is True. The default is True.
  - **AdjustShiftSwish**: (Boolean) If True, adjust the shift swish when IPULimitationCheck is True. The default is True.
  - **AlignConcat**: (Boolean) If True, adjust the quantization pos of concat when IPULimitationCheck is True. The default is True.
  - **AlignPool**: (Boolean) If True, adjust the quantization pos of pooling when IPULimitationCheck is True. The default is True.
  - **ReplaceClip6Relu**: (Boolean) If True, Replace Clip(0,6) with Relu in the model. The default is False.
  - **CLESteps**: (Int) Specifies the steps for CrossLayerEqualization execution when include_cle is set to true, The default is 1, When set to -1, an adaptive CrossLayerEqualization will be conducted. The default is 1.
  - **CLETotalLayerDiffThreshold**: (Float) Specifies The threshold represents the sum of mean transformations of CrossLayerEqualization transformations across all layers when utilizing CrossLayerEqualization. The default is 2e-7.
  - **CLEScaleAppendBias**: (Boolean) Whether the bias be included when calculating the scale of the weights, The default is True.
  - **RemoveQDQConvLeakyRelu**: (Boolean) If True, the QDQ between Conv and LeakyRelu will be removed for DPU when enable_dpu is True. The default is False.
  - **RemoveQDQConvPRelu**: (Boolean) If True, the QDQ between Conv and PRelu will be removed for DPU when enable_dpu is True. The default is False.


.. list-table:: Table 1. Quantize Types can be selected in Quantize Formats
   :widths: 25 25 50
   :header-rows: 1

   * - quant_format
     - quant_type
     - comments
   * - QuantFormat.QDQ
     - QuantType.QUInt8 
       QuantType.QInt8
     - Implemented by native QuantizeLinear/DequantizeLinear
   * - vai_q_onnx.VitisQuantFormat.QDQ
     - QuantType.QUInt8 
       QuantType.QInt8 
       vai_q_onnx.VitisQuantType.QUInt16
       vai_q_onnx.VitisQuantType.QInt16 
       vai_q_onnx.VitisQuantType.QUInt32
       vai_q_onnx.VitisQuantType.QInt32
       vai_q_onnx.VitisQuantType.QFloat16 
       vai_q_onnx.VitisQuantType.QBFloat16
     - Implemented by customized VitisQuantizeLinear/VitisDequantizeLinear

.. note:: 
   For pure UInt8 or Int8 quantization, we recommend setting quant_format to QuantFormat.QDQ as it uses native QuantizeLinear/DequantizeLinear operations which 
   may have better compatibility and performance.

.. note::
   In this documentation, **"NPU"** is used in descriptions, while **"IPU"** is retained in the tool's language, code, screenshots, and commands. This intentional 
   distinction aligns with existing tool references and does not affect functionality. Avoid making replacements in the code.

**************************
Recommended Configurations
**************************

CNNs on NPU  
===========

The recommended quantization configuration for CNN models to be deployed on the NPU is as follows:

.. code-block::

   from onnxruntime.quantization import QuantFormat, QuantType 
   import vai_q_onnx

   vai_q_onnx.quantize_static(
      model_input,
      model_output,
      calibration_data_reader,
      quant_format=vai_q_onnx.QuantFormat.QDQ,
      calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
      activation_type=vai_q_onnx.QuantType.QUInt8,
      weight_type=vai_q_onnx.QuantType.QInt8,
      enable_dpu=True,
      extra_options={'ActivationSymmetric':True}
   )



.. note::
   
   By default, Conv + LeakyRelu/PRelu fusion is turned off in the current version. You can try to enable this feature to get better performance if the model 
   contains LeakyRelu or PRelu. This default behavior may change in future versions. Here is the example configuration:

   .. code-block::

       extra_options={"ActivationSymmetric":True, 'RemoveQDQConvLeakyRelu':True, 'RemoveQDQConvPRelu':True}

Transformers on NPU
===================

The recommended quantization configuration for Transformer models to be deployed on the NPU is as follows:

.. code-block::

   import vai_q_onnx

   vai_q_onnx.quantize_static(
      model_input,
      model_output,
      calibration_data_reader,
      quant_format=vai_q_onnx.QuantFormat.QDQ,
      calibrate_method=vai_q_onnx.CalibrationMethod.MinMax,
      activation_type=vai_q_onnx.QuantType.QInt8,
      weight_type=vai_q_onnx.QuantType.QInt8,
   )


CNNs on CPU  
===========

The recommended quantization configuration for CNN models to be deployed on the CPU is as follows:

.. code-block::

   import vai_q_onnx

   vai_q_onnx.quantize_static(
      model_input,
      model_output,
      calibration_data_reader,
      quant_format=vai_q_onnx.QuantFormat.QDQ,
      calibrate_method=vai_q_onnx.CalibrationMethod.MinMax,
      activation_type=vai_q_onnx.QuantType.QUInt8,
      weight_type=vai_q_onnx.QuantType.QInt8
   )


******************************
Quantizing to Other Precisions
******************************


.. note::
   The current release of the Vitis AI Execution Provider ingests quantized ONNX models with INT8/UINT8 data types only. No support is provided for direct 
   deployment of models with other precisions, including FP32.


In addition to the INT8/UINT8, the VAI_Q_ONNX API supports quantizing models to other data formats, including INT16/UINT16, INT32/UINT32, Float16 and BFloat16, which can provide better accuracy or be used for experimental purposes. These new data formats are achieved by a customized version of QuantizeLinear and DequantizeLinear named "VitisQuantizeLinear" and "VitisDequantizeLinear", which expands onnxruntime's UInt8 and Int8 quantization to support UInt16, Int16, UInt32, Int32, Float16 and BFloat16. This customized Q/DQ was implemented by a custom operations library in VAI_Q_ONNX using onnxruntime's custom operation C API.

The custom operations library was developed based on Linux and does not currently support compilation on Windows. If you want to run the quantized model that has the custom Q/DQ on Windows, it is recommended to switch to WSL as a workaround.

To use this feature, the ```quant_format``` should be set to VitisQuantFormat.QDQ. The ```quant_format``` is set to ```QuantFormat.QDQ``` for accelerating both CNN's and transformers on the NPU target. 



1. Quantizing Float32 Models to Int16 or Int32 
==============================================


The quantizer supports quantizing float32 models to Int16 and Int32 data formats. To enable this, you need to set the "activation_type" and "weight_type" in the quantize_static API to the new data types. Options are ```VitisQuantType.QInt16/VitisQuantType.QUInt16``` for Int16, and ```VitisQuantType.QInt32/VitisQuantType.QUInt32``` for Int32.

.. code-block::

   vai_q_onnx.quantize_static(
      model_input,
      model_output,
      calibration_data_reader,
      calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
      quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
      activation_type=vai_q_onnx.VitisQuantType.QInt16,
      weight_type=vai_q_onnx.VitisQuantType.QInt16,
   )


2. Quantizing Float32 Models to Float16 or BFloat16
===================================================


Besides integer data formats, the quantizer also supports quantizing float32 models to float16 and bfloat16 data formats, by setting the "activation_type" and "weight_type" to ```VitisQuantType.QFloat16``` or ```VitisQuantType.QBFloat16``` respectively.

.. code-block::

   vai_q_onnx.quantize_static(
      model_input,
      model_output,
      calibration_data_reader,
      calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
      quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
      activation_type=vai_q_onnx.VitisQuantType.QFloat16,
      weight_type=vai_q_onnx.VitisQuantType.QFloat16,
   )


3. Quantizing Float32 Models to Mixed Data Formats
==================================================


The quantizer supports setting the activation and weight to different precisions. For example, activation is Int16 while weight is set to Int8. This can be used when pure Int8 quantization does not meet the accuracy requirements.

.. code-block::
      
   vai_q_onnx.quantize_static(
      model_input,
      model_output,
      calibration_data_reader,
      calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
      quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
      activation_type=vai_q_onnx.VitisQuantType.QInt16,
      weight_type=QuantType.QInt8,
   )

*************************
Quantizing Float16 Models
*************************


For models in float16, it is recommended to set "convert_fp16_to_fp32" to True. This will first convert your float16 model to a float32 model before quantization, reducing redundant nodes such as cast in the model.

.. code-block::
      
   vai_q_onnx.quantize_static(
      model_input,
      model_output,
      calibration_data_reader,
      quant_format=QuantFormat.QDQ,
      calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
      activation_type=QuantType.QUInt8,
      weight_type=QuantType.QInt8,
      enable_dpu=True,
      convert_fp16_to_fp32=True,
      extra_options={'ActivationSymmetric':True}
   )

*******************************************
Converting NCHW Models to NHWC and Quantize
*******************************************


NHWC input shape typically yields better acceleration performance compared to NCHW on NPU. VAI_Q_ONNX facilitates the conversion of NCHW input models to NHWC input models by setting "convert_nchw_to_nhwc" to True. Please note that the conversion steps will be skipped if the model is already NHWC or has non-convertable input shapes.

.. code-block::
      
   vai_q_onnx.quantize_static(
      model_input,
      model_output,
      calibration_data_reader,
      quant_format=QuantFormat.QDQ,
      calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
      activation_type=QuantType.QUInt8,
      weight_type=QuantType.QInt8,
      enable_dpu=True,
      extra_options={'ActivationSymmetric':True},
      convert_nchw_to_nhwc=True,
   )

*****************************************
Quantizing Using Cross Layer Equalization
*****************************************

Cross Layer Equalization (CLE) is a technique used to improve PTQ accuracy. It can equalize the weights of consecutive convolution layers, making the model weights easier to perform per-tensor quantization. Experiments show that using CLE technique can improve the PTQ accuracy of some models, especially for models with depthwise_conv layers, such as MobileNet. Here is an example showing how to enable CLE using VAI_Q_ONNX.

.. code-block::
      
   vai_q_onnx.quantize_static(
      model_input,
      model_output,
      calibration_data_reader,
      quant_format=QuantFormat.QDQ,
      calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
      activation_type=QuantType.QUInt8,
      weight_type=QuantType.QInt8,
      enable_dpu=True,
      include_cle=True,
      extra_options={
         'ActivationSymmetric':True,
         'ReplaceClip6Relu': True,
         'CLESteps': 1,
         'CLEScaleAppendBias': True,
         },
   )

**Arguments**

* **include_cle**: (Boolean) This parameter is a flag that determines whether to optimize the models using CrossLayerEqualization; it can improve the accuracy of some models. The default is False.

* **extra_options**: (Dictionary or None) Contains key-value pairs for various options in different cases. Options related to CLE are:

  -  **ReplaceClip6Relu**: (Boolean) If True, Replace Clip(0,6) with Relu in the model. The default value is False.
  -  **CLESteps**: (Int) Specifies the steps for CrossLayerEqualization execution when include_cle is set to true, The default is 1, When set to -1, an adaptive CrossLayerEqualization steps will be conducted. The default value is 1.  
  -  **CLEScaleAppendBias**: (Boolean) Whether the bias be included when calculating the scale of the weights, The default value is True.
  

*****
Tools
*****

Vitis AI ONNX quantizer includes a few built-in utility tools for model conversation. 

The list of available tools can be viewed as below

.. code-block::

   (conda_env)dir %CONDA_PREFIX%\lib\site-packages\vai_q_onnx\tools\

or 

.. code-block:: 

   (conda_env)python
   >>> help ('vai_q_onnx.tools')


Currently available utility tools

- convert_customqdq_to_qdq
- convert_dynamic_to_fixed
- convert_fp16_to_fp32
- convert_nchw_to_nhwc
- convert_onnx_to_onnxtxt
- convert_onnxtxt_to_onnx
- convert_qdq_to_qop
- convert_s8s8_to_u8s8
- random_quantize
- remove_qdq


..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
