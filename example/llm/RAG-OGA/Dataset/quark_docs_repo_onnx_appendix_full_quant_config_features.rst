Full List of Quantization Configuration Features
================================================

Overview
--------

It's very simple to quantize a model using the ONNX quantizer of Quark, only a few straightforward Python statements:

.. code:: python

    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, QuantizationConfig

    quant_config = QuantizationConfig()

    config = Config(global_quant_config=quant_config)
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(model_input, model_output, calibration_data_reader)


As shown in the code, just create a quantization configuration and use it to initialize a quantizer, and then call the quantizer's *quantize_model()* API, which has 3 main parameters:
*  **model_input**: (String or ModelProto) This parameter specifies the file path of the model that is to be quantized. When a file path cannot be specified, the loaded ModelProto can also be passed in directly.
*  **model_output**: (Optional String) This parameter specifies the file path where the quantized model will be saved. You can leave it unspecified (it will default to None), and the ModelProto format quantized model will be returned by the API.
*  **calibration_data_reader**: (Optional Object) This parameter is a calibration data reader that enumerates the calibration data and generates inputs for the original model. You can leave it unspecified (it will default to None), and simply enable *UseRandomData* in extra options of quantization configuration to use random data for calibration.

The next section will provide a detailed list of all parameters in the quantization configuration.

Quantization Configuration
--------------------------

.. code:: python

    quant_config = QuantizationConfig(
       calibrate_method = quark.onnx.CalibrationMethod.MinMax,
       quant_format = quark.onnx.QuantFormat.QDQ,
       activation_type = quark.onnx.QuantType.QInt8,
       weight_type = quark.onnx.QuantType.QInt8,
       input_nodes: List[str] = [],
       output_nodes: List[str] = [],
       op_types_to_quantize: List[str] = [],
       nodes_to_quantize: List[str] = [],
       extra_op_types_to_quantize: List[str] = [],
       nodes_to_exclude: List[str] = [],
       subgraphs_to_exclude: List[Tuple[List[str]]] = [],
       specific_tensor_precision: bool = False,
       execution_providers: List[str] = ['CPUExecutionProvider'],
       per_channel: bool = False,
       reduce_range: bool = False,
       optimize_model: bool = True,
       use_dynamic_quant: bool = False,
       use_external_data_format: bool = False,
       convert_fp16_to_fp32: bool = False,
       convert_nchw_to_nhwc: bool = False,
       include_sq: bool = False,
       include_rotation: bool = False,
       include_cle: bool = True,
       include_auto_mp: bool = False,
       include_fast_ft: bool = False,
       enable_npu_cnn: bool = False,
       enable_npu_transformer: bool = False,
       debug_mode: bool = False,
       crypto_mode: bool = False,
       print_summary: bool = True,
       ignore_warnings: bool = True,
       log_severity_level: int = 1,
       extra_options: Dict[str, Any] = {},
    )

*  **calibrate_method**: (String) The method used in calibration, default to quark.onnx.CalibrationMethod.MinMax.

   For NPU_CNN platforms, power-of-two methods should be used, options are:

   -  quark.onnx.PowerOfTwoMethod.NonOverflow: This method get the power-of-two quantize parameters for each tensor to make sure min/max values not overflow.
   -  quark.onnx.PowerOfTwoMethod.MinMSE: This method get the power-of-two quantize parameters for each tensor to minimize the mean-square-loss of quantized values and float values. This takes longer time but usually gets better accuracy.

   For NPU_Transformer or CPU platforms, float scale methods should be used, options are:

   -  quark.onnx.CalibrationMethod.MinMax: This method obtains the quantization parameters based on the minimum and maximum values of each tensor.
   -  quark.onnx.CalibrationMethod.Entropy: This method determines the quantization parameters by considering the entropy algorithm of each tensor's distribution.
   -  quark.onnx.CalibrationMethod.Percentile: This method calculates quantization parameters using percentiles of the tensor values.
   -  quark.onnx.LayerWiseMethod.LayerWisePercentile: This method calculates quantization parameters using different percentiles for different layers according to minimize mean average error or mean square error loss value.

*  **quant_format**: (String) This parameter is used to specify the quantization format of the model. It has the following options:

   -  quark.onnx.QuantFormat.QOperator: This option quantizes the model directly using quantized operators.
   -  quark.onnx.QuantFormat.QDQ: This option quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor. It supports 16-bit/8-bit/4-bit quantization.
   -  quark.onnx.ExtendedQuantFormat.QDQ: This option quantizes the model by inserting our customized QuantizeLinear/DequantizeLinear or BFPQuantizeDequantize/MXQuantizeDequantize into the tensor, which support a wider range of bit-widths and precisions.

*  **activation_type**: (QuantType) Specifies the quantization data type for activations, options can be found in the table below. The default is quark.onnx.QuantType.QInt8.
*  **weight_type**: (QuantType) Specifies the quantization data type for weights, options can be found in the table below. The default is quark.onnx.QuantType.QInt8. For NPU devices, this must be set to QuantType.QInt8.

*  **input_nodes**: (List of Strings) This parameter is a list of the
   names of the starting nodes to be quantized. Nodes in the model
   before these nodes will not be quantized. For example, this argument
   can be used to skip some pre-processing nodes or stop the first node
   from being quantized. The default value is an empty list ([]).
*  **output_nodes**: (List of Strings) This parameter is a list of the
   names of the end nodes to be quantized. Nodes in the model after
   these nodes will not be quantized. For example, this argument can be
   used to skip some post-processing nodes or stop the last node from
   being quantized. The default value is an empty list ([]).
*  **op_types_to_quantize**: (List of Strings or None) If specified,
   only operators of the given types will be quantized (e.g., ['Conv']
   to only quantize Convolutional layers). By default, all supported
   operators will be quantized.
*  **nodes_to_quantize**:(List of Strings or None) If specified, only
   the nodes in this list are quantized. The list should contain the
   names of the nodes, for example, ['Conv\__224', 'Conv\__252']. The
   default value is an empty list ([]).
*  **extra_op_types_to_quantize**: (List of Strings or None) If specified,
   the given operator types will be included as additional targets for
   quantization, expanding the set of operators to be quantized without
   replacing the existing configuration (e.g., ['Gemm'] to include Gemm
   layers in addition to the currently specified types). By default, no
   extra operator types will be added for quantization.
*  **nodes_to_exclude**:(List of Strings or None) If specified, the nodes 
   in this list will be excluded from quantization. The elements in this list
   can be either regular expression patterns with .\* or exact node names. 
   For instance, to exclude all nodes whose names start with /layer0/, you can 
   include a pattern like ^/layer0/.* in the list. The default value is an empty
   list ([]).
*  **subgraphs_to_exclude**:(List or None) If specified, the
   nodes in these subgraphs will be excluded from quantization. For example,
   you can use [(["Conv1"], ["Conv2"]), (["Relu9", "MatMul10"])] if you do
   not want to quantize nodes between "Conv1" and "Conv2" and nodes between
   "Relu9" and "MatMul10", as well as these start and end nodes themselves.
   If the subgraph is complex with multiple start nodes and multiple end nodes,
   you can use [([start_node1, start_node2], [end_node1, end_node2, end_node3])].
   The default value is an empty list ([]).
*  **specific_tensor_precision**: (Boolean) This parameter is a flag
   that determines whether to use tensor-level mixed precision, this is
   an experimental feature. The default is False.
*  **execution_providers**: (List of Strings) This parameter defines the
   execution providers that will be used by ONNX Runtime to do
   calibration for the specified model. The default value
   'CPUExecutionProvider' implies that the model will be computed using
   the CPU as the execution provider. You can also set this to other
   execution providers supported by ONNX Runtime such as
   'ROCMExecutionProvider' and 'CUDAExecutionProvider' for GPU-based computation,
   if they are available in your environment. The default is
   ['CPUExecutionProvider'].
*  **per_channel**: (Boolean) Determines whether weights should be
   quantized per channel. The default value is False. For DPU/NPU
   devices, this must be set to False as they currently do not support
   per-channel quantization.
*  **reduce_range**: (Boolean) If True, quantizes weights with 7-bits.
   The default value is False. For DPU/NPU devices, this must be set to
   False as they currently do not support reduced range quantization.
*  **optimize_model**:(Boolean) If True, optimizes the model before
   quantization. Model optimization performs certain operator fusion
   that makes quantization tool's job easier. For instance, a
   Conv/ConvTranspose/Gemm operator followed by BatchNormalization can
   be fused into one during the optimization, which can be quantized
   very efficiently. The default value is True.
*  **use_dynamic_quant**: (Boolean) This flag determines whether to apply
   dynamic quantization to the model. If True, dynamic quantization is used;
   if False, static quantization is applied. The default is False.
*  **use_external_data_format**: (Boolean) This option is used for large
   size (>2GB) model. The model proto and data will be stored in
   separate files. The default is False.
*  **convert_fp16_to_fp32**: (Boolean) This parameter controls whether
   to convert the input model from float16 to float32 before
   quantization. For float16 models, it is recommended to set this
   parameter to True. The default value is False. When using
   convert_fp16_to_fp32 in AMD Quark for ONNX, it requires onnxsim to
   simplify the ONNX model. Please make sure that onnxsim is installed
   by using 'python -m pip install onnxsim'.
*  **convert_nchw_to_nhwc**: (Boolean) This parameter controls whether
   to convert the input NCHW model to input NHWC model before
   quantization. For input NCHW models, it is recommended to set this
   parameter to True. The default value is False.
*  **include_sq**: (Boolean) This parameter is a flag that determines
   whether to optimize the models using SmoothQuant; it can improve the
   accuracy of transformer-based models like Llama. The default is False.
*  **include_rotation**: (Boolean) This parameter is a flag that determines whether
   to optimize the models using QuaRot. It can improve the accuracy of LLMs like
   Llama. RConfigPath must be given if include_rotation is True. The default is False.
*  **include_cle**: (Boolean) This parameter is a flag that determines
   whether to optimize the models using CrossLayerEqualization; it can
   improve the accuracy of some models. The default is True.
*  **include_auto_mp**: (Boolean) If True, the auto mixed precision will be turned on.
   The default is False.
*  **include_fast_ft**: (Boolean) This parameter is a flag that
   determines whether to use adaround or adaquant algorithm for
   finetuning, this is an experimental feature. The default is False.
*  **enable_npu_cnn**: (Boolean) This parameter is a flag that
   determines whether to generate a quantized model that is suitable for
   the DPU/NPU. If set to True, the quantization process will consider
   the specific limitations and requirements of the DPU/NPU, thus
   creating a model that is optimized for DPU/NPU computations. This
   parameter primarily addresses the optimization of CNN based models
   for deployment on DPU/NPU. The default is False. **Note**: In the
   previous versions, "enable_npu_cnn" was named "enable_dpu".
   "enable_dpu" will be deprecated in future releases, please use
   "enable_npu_cnn" instead.
*  **enable_npu_transformer**: (Boolean) This parameter is a flag that
   determines whether to generate a quantized model that is suitable for
   the NPU. If set to True, the quantization process will consider the
   specific limitations and requirements of the NPU, thus creating a
   model that is optimized for NPU computations. This parameter
   primarily addresses the optimization of transformer models for
   deployment on NPU. The default is False.
*  **debug_mode**: (Boolean) Flag to enable debug mode. In this mode,
   all debugging message will be printed. Default is False.
*  **crypto_mode**: (Boolean) Flag to enable crypto mode. In this mode,
   all message will be blocked, and all intermediate data related to the
   model will not be saved to disk. In addition, the input model to the
   *quantize_model* API should be a ModelProto object. Please that it
   only supports <2GB ModelProto object. Default is False.
*  **print_summary**: (Boolean) Flag to print summary of quantization. Default is True.
*  **ignore_warnings**: (Boolean) Flag to suppress the warnings globally. Default is True.
*  **log_severity_level**: (Int) This parameter is used to select the
   severity level of screen printing logs. Its value ranges from 0 to 4: 0 for DEBUG,
   1 for INFO, 2 for WARNING, 3 for ERROR and 4 for CRITICAL or FATAL. Default value is 1,
   which means printing all messages including INFO, WARNING, ERROR and etc by default.
*  **extra_options**: (Dictionary or None) Contains key-value pairs for
   various options in different cases. Current used:

   -  **ActivationSymmetric**: (Boolean) If True, symmetrize calibration
      data for activations. The default is False.
   -  **WeightSymmetric**: (Boolean) If True, symmetrize calibration
      data for weights. The default is True.
   -  **ActivationScaled**: (Boolean) If True, all activations will be scaled to the exact numeric range.
      The default is True for integer data type quantization and False for BFloat16 and Float16, which means
      by default the BFloat16/Float16 quantization will cast float32 tensors to BFloat16/Float16 directly.
   -  **WeightScaled**: (Boolean) If True, all weights will be scaled to the exact numeric range.
      The default is True for integer data type quantization and False for BFloat16 and Float16, which means
      by default the BFloat16/Float16 quantization will cast float32 tensors to BFloat16/Float16 directly.
   -  **QuantizeFP16**: (Boolean) If True, the data type of the input model should be float16. It only takes effect when onnxruntime version is 1.18 or above. The default is False.
   -  **UseFP32Scale**: (Boolean) If True, the scale of the quantized model is converted from float16 to float32 when the quantization is done. It only takes effect only if QuantizeFP16 is True. It must be False when UseMatMulNBits is True. The default is True.
   -  **UseUnsignedReLU**: (Boolean) If True, the output tensor of ReLU
      and Clip, whose min is 0, will be forced to be asymmetric. The
      default is False.
   -  **QuantizeBias**: (Boolean) If True, quantize the Bias as a normal
      weights. The default is True. For DPU/NPU devices, this must be
      set to True.
   -  **Int32Bias**: (Boolean) If True, bias will be quantized in int32
      data type; if false, it will have the same data type as weight. The
      default is False when enable_npu_cnn is True. Otherwise the
      default is True.
   -  **Int16Bias**: (Boolean) If True, bias will be quantized in int16
      data type; The default is False. **Note**: 1. ONNXRuntime only supports
      Int16 Bias inference when the opset version is 21 or higher, so please 
      ensure that the input model's opset version is 21 or higher. 2. It is 
      recommended to use this together with ADAROUND or ADAQUANT; otherwise, 
      the quantized model with Int16 bias may suffer from poor accuracy.
   -  **RemoveInputInit**: (Boolean) If True, initializer in graph
      inputs will be removed because it will not be treated as constant
      value/weight. This may prevent some of the graph optimizations,
      like const folding. The default is True.
   -  **SimplifyModel**: (Boolean) If True, The input model will be
      simplified using the onnxsim tool. The default is True.
   -  **EnableSubgraph**: (Boolean) If True, the subgraph will be
      quantized. The default is False. More support for this feature is
      planned in the future.
   -  **ForceQuantizeNoInputCheck**: (Boolean) If True, latent operators
      such as maxpool and transpose will always quantize their inputs,
      generating quantized outputs even if their inputs have not been
      quantized. The default behavior can be overridden for specific
      nodes using nodes_to_exclude.
   -  **MatMulConstBOnly**: (Boolean) If True, only MatMul operations
      with a constant 'B' will be quantized. The default is False for
      static mode and True for dynmaic mode.
   -  **AddQDQPairToWeight**: (Boolean) If True, both QuantizeLinear and
      DeQuantizeLinear nodes are inserted for weight, maintaining its
      floating-point format. The default is False, which quantizes
      floating-point weight and feeds it solely to an inserted
      DeQuantizeLinear node. In the PowerOfTwoMethod calibration method,
      this setting will also be effective for the bias.
   -  **OpTypesToExcludeOutputQuantization**: (List of Strings or None)
      If specified, the output of operators with these types will not be
      quantized. The default is an empty list.
   -  **DedicatedQDQPair**: (Boolean) If True, an identical and
      dedicated QDQ pair is created for each node. The default is False,
      allowing multiple nodes to share a single QDQ pair as their
      inputs.
   -  **QDQOpTypePerChannelSupportToAxis**: (Dictionary) Sets the
      channel axis for specific operator types (e.g., {'MatMul': 1}).
      This is only effective when per-channel quantization is supported
      and per_channel is True. If a specific operator type supports
      per-channel quantization but no channel axis is explicitly
      specified, the default channel axis will be used. For DPU/NPU
      devices, this must be set to {} as per-channel quantization is
      currently unsupported. The default is an empty dict ({}).
   -  **CalibTensorRangeSymmetric**: (Boolean) If True, the final range
      of the tensor during calibration will be symmetrically set around
      the central point "0". The default is False. In PowerOfTwoMethod
      calibration method, the default is True.
   -  **CalibMovingAverage**: (Boolean) If True, the moving average of
      the minimum and maximum values will be computed when the
      calibration method selected is MinMax. The default is False. In
      PowerOfTwoMethod calibration method, this should be set to False.
   -  **CalibMovingAverageConstant**: (Float) Specifies the constant
      smoothing factor to use when computing the moving average of the
      minimum and maximum values. The default is 0.01. This is only
      effective when the calibration method selected is MinMax and
      CalibMovingAverage is set to True. In PowerOfTwoMethod calibration
      method, this option is unsupported.
   -  **Percentile**: (Float) If the calibration method is set to
      'quark.onnx.CalibrationMethod.Percentile,' then this parameter can
      be set to the percentage for percentile. The default is 99.999.
   -  **LWPMetric**: (String) If the calibration method is set to
      'quark.onnx.LayerWiseMethod.LayerWisePercentile,' then this parameter can
      be set to select the metric to judge the percentile value. The default is mae.
   -  **ActivationBitWidth**: (Int) If the calibration method is set to
      'quark.onnx.LayerWiseMethod.LayerWisePercentile', then this parameter can
      be set to calculate the quantize/dequantize error. The default is 8.
   -  **PercentileCandidates**: (List) If the calibration method is set to
      'quark.onnx.LayerWiseMethod.LayerWisePercentile' then this parameter can
      be set to the percentage for percentiles. The default is [99.99, 99.999, 99.9999].
   -  **UseRandomData**: (Boolean) Required to be true when the
      RandomDataReader is needed. The default value is false.
   -  **RandomDataReaderInputShape**: (Dict) It is required to use
      dict {name : shape} to specify a certain input. For example,
      RandomDataReaderInputShape={"image" : [1, 3, 224, 224]} for the
      input named "image". The default value is an empty dict {}.
   -  **RandomDataReaderInputDataRange**: (Dict or None) Specifies the
      data range for each inputs if used random data reader
      (calibration_data_reader is None). Currently, if set to None then
      the random value will be 0 or 1 for all inputs, otherwise range
      [-128,127] for unsigned int, range [0,255] for signed int and
      range [0,1] for other float inputs. The default is None.
   -  **Int16Scale**: (Boolean) If True, the float scale will be
      replaced by the closest value corresponding to M and 2\ **N, where
      the range of M and 2**\ N is within the representation range of
      int16 and uint16. The default is False.
   -  **MinMSEMode**: (String) When using
      quark.onnx.PowerOfTwoMethod.MinMSE, you can specify the method for
      calculating minmse. By default, minmse is calculated using all
      calibration data. Alternatively, you can set the mode to
      "MostCommon", where minmse is calculated for each batch separately
      and take the most common value. The default setting is 'All'.
   -  **ConvertOpsetVersion**: (Int or None) Specifies the target opset version for the ONNX model.
      If set, the model's opset version will be updated accordingly. The default is None.
   -  **ConvertBNToConv**: (Boolean) If True, the BatchNormalization
      operation will be converted to Conv operation. The default is True
      when enable_npu_cnn is True.
   -  **ConvertReduceMeanToGlobalAvgPool**: (Boolean) If True, the
      Reduce Mean operation will be converted to Global Average Pooling
      operation. The default is True when enable_npu_cnn is True.
   -  **SplitLargeKernelPool**: (Boolean) If True, the large kernel
      Global Average Pooling operation will be split into multiple
      Average Pooling operation. The default is True when enable_npu_cnn
      is True.
   -  **ConvertSplitToSlice**: (Boolean) If True, the Split operation
      will be converted to Slice operation. The default is True when
      enable_npu_cnn is True.
   -  **FuseInstanceNorm**: (Boolean) If True, the split instance norm
      operation will be fused to InstanceNorm operation. The default is
      True.
   -  **FuseL2Norm**: (Boolean) If True, a set of L2norm operations will
      be fused to L2Norm operation. The default is True.
   -  **FuseGelu**: (Boolean) If True, a set of Gelu operations will
      be fused to Gelu operation. The default is True.
   -  **FuseLayerNorm**: (Boolean) If True, a set of LayerNorm
      operations will be fused to LayerNorm operation. The default is
      True.
   -  **ConvertClipToRelu**: (Boolean) If True, the Clip operations that
      has a min value of 0 will be converted to ReLU operations. The
      default is True when enable_npu_cnn is True.
   -  **SimulateDPU**: (Boolean) If True, a simulation transformation
      that replaces some operations with an approximate implementation
      will be applied for DPU when enable_npu_cnn is True. The default
      is True.
   -  **ConvertLeakyReluToDPUVersion**: (Boolean) If True, the Leaky
      Relu operation will be converted to DPU version when SimulateDPU
      is True. The default is True.
   -  **ConvertSigmoidToHardSigmoid**: (Boolean) If True, the Sigmoid
      operation will be converted to Hard Sigmoid operation when
      SimulateDPU is True. The default is True.
   -  **ConvertHardSigmoidToDPUVersion**: (Boolean) If True, the Hard
      Sigmoid operation will be converted to DPU version when
      SimulateDPU is True. The default is True.
   -  **ConvertAvgPoolToDPUVersion**: (Boolean) If True, the global or
      kernel-based Average Pooling operation will be converted to DPU
      version when SimulateDPU is True. The default is True.
   -  **ConvertClipToDPUVersion**: (Boolean) If True, the Clip operation
      will be converted to DPU version when SimulateDPU is True. The
      default is False.
   -  **ConvertReduceMeanToDPUVersion**: (Boolean) If True, the
      ReduceMean operation will be converted to DPU version when
      SimulateDPU is True. The default is True.
   -  **ConvertSoftmaxToDPUVersion**: (Boolean) If True, the Softmax
      operation will be converted to DPU version when SimulateDPU is
      True. The default is False.
   -  **NPULimitationCheck**: (Boolean) If True, the quantization position
      will be adjust due to the limitation of DPU/NPU. The default is
      True.
   -  **MaxLoopNum**: (Int) The quantizer adjusts or aligns the quantization
      position through loops, this option is used to set the maximum number of loops.
      The default value is 5.
   -  **AdjustShiftCut**: (Boolean) If True, adjust the shift cut of
      nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustShiftBias**: (Boolean) If True, adjust the shift bias of
      nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustShiftRead**: (Boolean) If True, adjust the shift read of
      nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustShiftWrite**: (Boolean) If True, adjust the shift write of
      nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustHardSigmoid**: (Boolean) If True, adjust the position of hard
      sigmoid nodes when NPULimitationCheck is True. The default is
      True.
   -  **AdjustShiftSwish**: (Boolean) If True, adjust the shift swish
      when NPULimitationCheck is True. The default is True.
   -  **AlignConcat**: (Boolean) If True, adjust the quantization position of
      concat when NPULimitationCheck is True. The default is True,
      when the power-of-two scale is used, otherwise it's False.
   -  **AlignPool**: (Boolean) If True, adjust the quantization position of
      pooling when NPULimitationCheck is True. The default is True,
      when the power-of-two scale is used, otherwise it's False.
   -  **AlignPad**: (Boolean) If True, adjust the quantization position of
      pad when NPULimitationCheck is True. The default is True,
      when the power-of-two scale is used, otherwise it's False.
   -  **AlignSlice**: (Boolean) If True, adjust the quantization position of
      slice when NPULimitationCheck is True. The default is True,
      when the power-of-two scale is used, otherwise it's False.
   -  **AlignTranspose**: (Boolean) If True, adjust the quantization position of
      transpose when NPULimitationCheck is True. The default is False.
   -  **AlignReshape**: (Boolean) If True, adjust the quantization position of
      reshape when NPULimitationCheck is True. The default is False.
   -  **AdjustBiasScale**: (Boolean) If True, adjust the bias scale equal to activation scale
      multiply by weights scale. The default is True.
   -  **BFPAttributes**: (Dictionary) A parameter used to specify the
      attributes for BFP quantization nodes.

      -  **bfp_method**: (String) BFP method. The options are "to_bfp“ and "to_bfp_prime",
         corresponding to classic BFP and BFP with micro exponents, respectively.
         The default is 'to_bfp'.
      -  **axis**: (Int) The axis for splitting the input tensor into blocks. The default is 1
         but can be modified by the quantizer according to the tensor's shape.
      -  **bit_width**: (Int) Bits for the block floating point. For BFP16,
         this parameter should be 16, which consists of three parts: 8 bits shared exponent,
         1 bit sign and 7 bits mantissa. The default is 16.
      -  **block_size**: (Int) Size of block. The default is 8.
      -  **sub_block_size**: (Int) Size of sub-block, only effective when bfp_method is "to_bfp_prime”.
         The default is 2.
      -  **sub_block_shift_bits**: (Int) Bits for the micro exponents of a sub block, only effective
         when bfp_method is "to_bfp_prime”. The default is 1.
      -  **rounding_mode**: (Int) Rounding mode, 0 for rounding half away from zero, 1 for rounding half
         upward and 2 for rounding half to even. The default is 0.
      -  **convert_to_bfloat_before_bfp**: (Int) If set to 1, convert the input tensor to BFloat16
         before converting to BFP. The default is 0.
      -  **use_compiler_version_cpu_kernel**: (Int) If set to 1, use a customized cpu kernel.
         The default is 0.

   *  **MXAttributes**: (Dictionary) A parameter used to specify the
      attributes for MX quantization nodes.

      -  **element_dtype**: (String) Element data type. The options are "fp8_e5m2", "fp8_e4m3",
         "fp6_e3m2", "fp6_e2m3", "fp4_e2m1" and "int8". The default is "int8".
      -  **axis**: (Int) The axis for splitting the input tensor into blocks. The default is 1
         but can be modified by the quantizer according to the tensor's shape.
      -  **block_size**: (Int) Size of block. The default is 32.
      -  **rounding_mode**: (Int) Rounding mode, 0 for rounding half away from zero, 1 for rounding half
         upward and 2 for rounding half to even. The default is 0.

   *  **ReplaceClip6Relu**: (Boolean) If True, Replace Clip(0,6) with
      Relu in the model. The default is False.
   *  **CLESteps**: (Int) Specifies the steps for CrossLayerEqualization
      execution when include_cle is set to true, The default is 1, When
      set to -1, an adaptive CrossLayerEqualization will be conducted.
      The default is 1.
   *  **CLETotalLayerDiffThreshold**: (Float) Specifies The threshold
      represents the sum of mean transformations of
      CrossLayerEqualization transformations across all layers when
      utilizing CrossLayerEqualization. The default is 2e-7.
   *  **CLEScaleAppendBias**: (Boolean) Whether the bias be included
      when calculating the scale of the weights, The default is True.
   *  **CopySharedInit**: (List or None) Specifies the node op_types to run 
      duplicating initializer in the model for separate quantization use across 
      different nodes, e.g. ['Conv', 'Gemm', 'Mul'] input, only shared initializer 
      in these nodes will be duplicated. None means that skip this conversion 
      while empty list means that run this for all op_types included in the 
      given model, default is None.
   *  **CopyBiasInit**: (List or None) Specifies the node operation types to run 
      duplicating bias initializer in the model for separate quantization use across 
      different nodes, e.g. ['Conv', 'Gemm', 'Mul'] input, only shared bias initializer 
      in these nodes will be duplicated. None means that skip this conversion 
      while empty list means that run this for all operation types included in the 
      given model. The default is an empty list when using quantization with float scale 
      like A8W8 and A16W8. The default is None otherwise.
   *  **FastFinetune**: (Dictionary) A parameter used to specify the
      settings for fast finetune.

      -  **OptimAlgorithm**: (String) The specified algorithm for fast finetune. Optional values are "adaround" and "adaquant". The
         "adaround" adjusts the weights rounding function, which is
         relatively stable and might converge faster. The "adaquant" trains
         the weight (and bias optional) directly, so might have a greater
         improvement if the parameters, especially the learning rate and
         batch size, are optimal. The default value is "adaround".
      -  **OptimDevice**: (String) Specifies the compute device used for
         PyTorch model training during fast finetuning. Optional values
         are "cpu", and "cuda:0". The default value is "cpu".
      -  **InferDevice**: (String) Specifies the compute device used for
         ONNX model inference during fast finetuning. Optional values are
         "cpu" and "cuda:0". The default value is "cpu".
      -  **FixedSeed**: (Int) Seed for random data generator, that makes
         the fast finetuned results could be reproduced.
      -  **DataSize**: (Int) Specifies the size of the data used for
         finetuning. Its recommended setting the batch size of the data to
         1 in the data reader to ensure counting the size accurately. It
         uses all the data from the data reader by default.
      -  **BatchSize**: (Int) Batch size for finetuning. The larger batch
         size, usually the better accuracy but the longer training time.
         The default value is 1.
      -  **NumBatches**: (Int) The mini-batches in a iteration. It should
         always be 1. The default value is 1.
      -  **NumIterations**: (Int) The Iterations for finetuning. The more
         iterations, the better accuracy but the longer training time. The
         default value is 1000.
      -  **LearningRate**: (Float) Learning rate of finetuning for all
         layers. It has a significant impact on the accuracy improvement,
         you need to try some learning rates to get a better result for
         your model. The default value is 0.1 for AdaRound and 0.00001 for
         AdaQuant.
      -  **EarlyStop**: (Bool) If average loss of a certain number of
         iterations decreases comparing with the previous one, the training
         of the layer will stop early. It will accelerate the finetuning
         process and avoid overfitting. The default value is False.
      -  **LRAdjust**: (Tuple) Besides the overall learning rate, users
         could set up a scheme to adjust learning rate further according to
         the mean square error (MSE) between the quantized module and
         original float module. Its a tuple contains two members, the
         first one is a threshold of the MSE and the second one is the new
         learning rate. For example, setting as (1.0, 0.2) means using a
         new learning rate 0.2 for the layer whose MSE is bigger than 1.0.
      -  **TargetOpType**: (List) The target operation types to finetune.
         The default value is [Conv, ConvTranspose, Gemm, MatMul,
         InstanceNormalization]. The MatMul node must have one and only one
         set of weights.
      -  **SelectiveUpdate**: (Bool) If the end-to-end accuracy does not
         improve after finetuned a certain layer, discard the optimized
         weight (and bias) of the layer. The default value is False.
      -  **UpdateBias**: (Bool) Specifies whether to update bias
         parameters during fine-tuning. Its only available for AdaQuant.
         The default value is False.
      -  **OutputQDQ**: (Bool) Specifies whether include the output
         tensors QDQ pair of the compute nodes for finetuning. The default
         value is False.
      -  **DropRatio**: (Float) Specifies the ratio to drop the input
         data from the float module. It ranges from 0 to 1, 0 represents
         the input data is from the float module fully, 1 represents all
         from quantized module. The default value is 0.5.
      -  **LogPeriod**: (Int) Indicate how many iterations to print the
         log once. The default value is NumIterations/10.

   *  **SmoothAlpha**: (Float) This parameter control how much
      difficulty we want to migrate from activation to weights, The
      default value is 0.5.
   *  **RMatrixDim**: (Int) Specifies the dimension for constructing
      rotation matrix. The default value is 4096.
   *  **UseRandomHad**: (Boolean) If True, the rotation matrix will be 
      generated by the random Hadamard scheme. The default is False.
   *  **RConfigPath**: (String) Set the path for rotation config file.
      This is necessary when using QuaRot. The default is "".
   *  **RemoveQDQConvClip**: (Boolean) If True, the QDQ between
      Conv/Add/Gemm and Clip will be removed for DPU. The default is
      True.
   *  **RemoveQDQConvRelu**: (Boolean) If True, the QDQ between
      Conv/Add/Gemm and Relu will be removed for DPU. The default is
      True.
   *  **RemoveQDQConvLeakyRelu**: (Boolean) If True, the QDQ between
      Conv/Add/Gemm and LeakyRelu will be removed for DPU. The default
      is True.
   *  **RemoveQDQConvPRelu**: (Boolean) If True, the QDQ between
      Conv/Add/Gemm and PRelu will be removed for DPU. The default is
      True.
   *  **RemoveQDQConvGelu**: (Boolean) If True, the QDQ between
      Conv/Add/Gemm and Gelu will be removed. The default is False.
   *  **RemoveQDQMulAdd**: (Boolean) If True, the QDQ between
      Mul and Add will be removed for NPU. The default is False.
   *  **RemoveQDQBetweenOps**: (List of tuples (Strings, Strings) or None)
      This parameter accepts a list of tuples representing operation type
      pairs (e.g., Conv and Relu). If set, the QDQ between the specified
      pairs of operations will be removed for NPU. The default is None.
   *  **RemoveQDQInstanceNorm**: (Boolean) If True, the QDQ between
      InstanceNorm and Relu/LeakyRelu/PRelu will be removed for DPU. The
      default is False.
   *  **FoldBatchNorm**: (Boolean) If True, the BatchNormalization
      operation will be fused with Conv, ConvTranspose or Gemm
      operation. The BatchNormalization operation after Concat operation
      will also be fused, if the all input operations of the Concat
      operation are Conv, ConvTranspose or Gemm operatons.The default is
      True.
   *  **BF16WithClip**: (Boolean) If True, during BFloat16
      quantization, insert "Clip" node before customized "QuantizeLinear" node to
      add boundary protection for activation. The default is False.
   *  **BF16QDQToCast**: (Boolean) If True, during BFloat16
      quantization, replace QuantizeLinear/DeQuantizeLinear ops with Cast
      ops to accelerate BFloat16 quantized inference. The default is False.
   *  **FixShapes**: (String) Set the input and output shapes of the quantized
      model to a fixed shape by default if not explicitly specified. The
      example: 'FixShapes':'input_1:[1,224,224,3];input_2:[1,96,96,3];output_1:[1,100];output_2:[1,1000]'
   *  **MixedPrecisionTensor**: (Dictionary) A parameter used to specify
      the settings for mixed precision tensors. It is a dictionary where
      the keys are of the ExtendedQuantType/QuantType enumeration type, and
      the values are lists containing tensors that need to be processed
      using mixed precision.
      Example:"MixedPrecisionTensor":{quark.onnx.ExtendedQuantType.QBFloat16:['/stem/stem.2/Relu_output_0',
      'onnx::Conv_664', 'onnx::Conv_665']} **Note**:If there is a tensor
      with bias, 'Int32Bias' needs set to False.

   *  **AutoMixprecision**: (Dictionary) A parameter used to specify the
      settings for auto mixed precision.

      -  **DataSize**: (Int) Specifies the size of the data used for mix-precision. The entire data reader will be used by default.
      -  **TargetOpType**: (Set) The user defined op type set for mix-precision. The default value is ('Conv', 'ConvTranspose', 'Gemm', 'MatMul').
      -  **TargetQuantType**: (QuantType) Activation data type to be mixed in the model if 'ActTargetQuantType' is not given. Error will be raised if TargetQuantType is not specified.
      -  **ActTargetQuantType**: (QuantType) Activation data type to be mixed in the model.
         If both ActTargetQuantType and WeightTargetQuantType are not specified, the ActTargetQuantType will be same as TargetQuantType.
         If only ActTargetQuantType is not specified, the ActTargetQuantType will be the original activation_type.
      -  **WeightTargetQuantType**: (QuantType) Weight data type to be mixed in the model.
         If both ActTargetQuantType and WeightTargetQuantType are not specified, the ActTargetQuantType will be same as TargetQuantType.
         If only WeightTargetQuantType is not specified, the WeightTargetQuantType will be the original weight_type.
      -  **BiasTargetQuantType**: (QuantType) Bias data type to be mixed in the model.
         If BiasTargetQuantType is not specified and Int32Bias is True, the BiasTargetQuantType will be int32.
         If BiasTargetQuantType is not specified and Int32Bias is False, the BiasTargetQuantType will be same as WeightTargetQuantType.
      -  **DualQuantNodes**: (Bool) Some backend compilers require that two types of quantization nodes exist simultaneously on the tensors which connect two different precision nodes,
         for example, they require the tensor that connects BFP16 Conv and BF16 Reshape has a BFP node and a QDQ pair both. The default value is False.
      -  **OutputIndex**: (Int) The index of model output to be calculated for loss.
      -  **L2Target**: (Float) The L2 loss will be no larger than the L2Target.
         If L2Target is not specified, the model will be quantized to the target quant type.
      -  **Top1AccTarget**: (Float) The Top1 accuracy loss will be no larger than the Top1AccTarget.
         If Top1AccTarget is not specified, the model will be quantized to the target quant type.
      -  **EvaluateFunction**: (Function) The function to measure top1 accuracy loss. Input of the function is model output(numpy tensor),
         output of the function is top1 accuracy(between 0~1). If EvaluateFunction is not specified while Top1AccTarget is given, error will be raised.
      -  **NumTarget**: (Int) Specified the number of nodes for mix-precision to minimize the loss. The default value of NumTarget is 0.
      -  **TargetTensors**: (List) Specified the names of nodes to mix into the target quant type. It's a experimental option and will be deprecated in the future. The default value is [].
      -  **TargetIndices**: (List) Specified the indices (based on sensitivity analysis results) of the nodes to mix into the target quant type. The default value is [].
      -  **ExcludeIndices**: (List) Specified the indices (based on sensitivity analysis results) of the nodes not to mix into the target quant type. The default value is [].
      -  **NoInputQDQShared**: (Bool) If True, will skip the nodes who shared the input Q/DQ pair with other nodes. The default value is True.
      -  **AutoMixUseFastFT**: (Bool) If True, will perform fast finetune to improve accuracy after mixed a layer. The default value is False.

   *  **FoldRelu**: (Boolean) If True, the Relu will be fold to Conv
      when use ExtendedQuantFormat. The default is False.
   *  **CalibDataSize**: (Int) This parameter controls how many data are
      used for calibration. The default to using all the data in the
      calibration dataloader.
   *  **SaveTensorHistFig**: (Boolean) If True, save the tensor
      histogram to the file 'tensor_hist' in the working directory. The
      default is False.
   *  **QuantizeAllOpTypes**: (Boolean) If True, all operation types will be quantized.
      In the BF16 config, the default is True, while for others, the default is False.
   *  **WeightsOnly**: (Boolean) If True, only quantize weights of the
      model. The default is False.
   *  **AlignEltwiseQuantType**: (Boolean) If True, quantize weights of the node with the activation quant type if node type in [Mul, Add, Sub, Div, Min, Max] when quant_format is ExtendedQuantFormat.QDQ and enable_npu_cnn is False and enable_npu_transformer is False. The default is False.
   *  **EnableVaimlBF16**: (Boolean) If True, the bfloat16 quantized model with vitis qdq will be converted to a bfloat16 quantized model with bfloat16 weights stored as float32. Vaiml is the name of a compiler, the bfloat16 quantized model can be directly deployed on the compiler if the parameter is True. The default is False.
   *  **UseGPTQ**: (Boolean) If True, GPTQ algorithm will be applied to the
      model. The default is False.
   *  **GPTQParams**: (Dictionary) A parameter used to specify the
      settings for GPTQ.

      -  **Bits**: (int) The quantization bits used in GPTQ. The default is 8.
      -  **BlockSize**: (int) The block size in GPTQ determines
         how many columns of weights will be quantized for one update. The default is 128.
      -  **GroupSize**: (int) The group size in GPTQ determines how many columns of weights share one set of scale and zero-point. The default is -1.
      -  **PercDamp**: (int) Percent of the average Hessian diagonal to use for dampening. The default is 0.01.
      -  **ActOrder**: (Boolean) Determine whether to re-order Hessian matrix according the values of diag. The default is False.
      -  **PerChannel**: (Boolean) Determine whether perform per-channel quantization in GPTQ. The default is False.
      -  **MSE**: (Boolean) Determine whether to use MSE method to do data calibration in GPTQ. The default is False.

   *  **UseMatMulNBits**: (Boolean) If True, only quantize weights with nbits for MatMul of the
      model. The default is False.
   *  **MatMulNBitsParams**: (Dictionary) A parameter used to specify the
      settings for MatMulNBits Quantizer.

      -  **Algorithm**: (str) The algorithm in MatMulNBits Quantization determines which algorithm ("DEFAULT", "GPTQ", "HQQ") to be used to quantize weights. The default is "DEFAULT".
      -  **GroupSize**: (int) The block size in MatMulNBits Quantization determines how many weights share a scale. The default is 128.
      -  **Symmetric**: (Boolean) If True, symmetrize quantization for weights. The default is True.
      -  **Bits**: (int) The target bits to quantize. Only 4b quantization is supported for inference, additional bits support is planned.
      -  **AccuracyLevel**: (int) The quantization level of input, can be: 0(unset), 1(fp32), 2(fp16), 3(bf16), or 4(int8). The default is 0.


Table 7. Quantize Types can be selected for different Quantize Formats

+---------------------------+----------------------------------+---------------------------+
| quant_format              | quant_type                       | comments                  |
+===========================+==================================+===========================+
| QuantFormat.QDQ           | QuantType.QUInt16                |                           |
|                           | QuantType.QInt16                 |                           |
|                           | QuantType.QUInt8                 |                           |
|                           | QuantType.QInt8                  |                           |
|                           | QuantType.QUInt4                 |                           |
|                           | QuantType.QInt4                  |                           |
+---------------------------+----------------------------------+---------------------------+
| ExtendedQuantFormat.QDQ   | QuantType.QUInt8                 |                           |
|                           | QuantType.QInt8                  |                           |
|                           | ExtendedQuantType.QUInt16        |                           |
|                           | ExtendedQuantType.QInt16         |                           |
|                           | ExtendedQuantType.QFloat16       |                           |
|                           | ExtendedQuantType.QBFloat16      |                           |
|                           | ExtendedQuantType.QBFP           |                           |
|                           | ExtendedQuantType.QMX            |                           |
|                           | ExtendedQuantType.QUInt32        |                           |
|                           | ExtendedQuantType.QInt32         |                           |
+---------------------------+----------------------------------+---------------------------+

**Note**: For UINT4 and INT4 quantization types, ONNX Runtime version 1.19.0 or later is required.
Users must ensure that the ``calibration_method`` is a native ORT quantization method (MinMax, Percentile, etc.).

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
