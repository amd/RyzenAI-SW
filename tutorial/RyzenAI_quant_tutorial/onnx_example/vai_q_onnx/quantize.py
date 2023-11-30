#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy
import logging
import tempfile
from pathlib import Path

import onnx
import onnx.helper as helper
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod, create_calibrator
from onnxruntime.quantization.quantize import quantize_static as ort_quantize_static
from onnxruntime.quantization.quant_utils import QuantizationMode, QuantType, QuantFormat
from .calibrate import create_calibrator_power_of_two, PowerOfTwoMethod
from .optimize import optimize
from .equalization import cle_transforms, replace_all_clip6_to_relu
from .onnx_quantizer import VitisONNXQuantizer
from .qdq_quantizer import VitisExtendedQuantizer, VitisQDQQuantizer, VitisQDQDPUQuantizer, VitisBFPQuantizer
from .quant_utils import VAI_DOMAIN, VitisQuantType, VitisQuantFormat, get_exclude_nodes, RandomDataReader, check_onnx_model, run_onnx_model, print_quantize_info, is_ort_version_below_1_16, Int16Method, remove_initializer_from_input, fp32_nodes, print_fp32_nodes
from .registry import IntegerOpsRegistry, QLinearOpsRegistry, QDQRegistry, DPURegistry


def check_static_quant_arguments(quant_format, activation_type, weight_type):

    qwb_types = [
        VitisQuantType.QInt16, VitisQuantType.QUInt16, VitisQuantType.QInt32,
        VitisQuantType.QUInt32, VitisQuantType.QFloat16,
        VitisQuantType.QBFloat16
    ]

    if (activation_type in qwb_types or
            weight_type in qwb_types) and quant_format != VitisQuantFormat.QDQ:
        raise ValueError(
            "Only VitisQuantFormat.QDQ supports wide bits quantization types.")


def quantize_static(
    model_input,
    model_output,
    calibration_data_reader: CalibrationDataReader,
    quant_format=QuantFormat.QDQ,
    input_nodes=[],
    output_nodes=[],
    op_types_to_quantize=[],
    random_data_reader_input_shape=[],
    per_channel=False,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=[],
    nodes_to_exclude=[],
    optimize_model=True,
    use_external_data_format=False,
    calibrate_method=PowerOfTwoMethod.MinMSE,
    execution_providers=['CPUExecutionProvider'],
    enable_dpu=False,
    fp16_fp32_convert=False,
    convert_nchw_to_nhwc=False,
    debug_mode=False,
    include_cle=False,
    extra_options={},
    print_summary=True,
):
    print_quantize_info(model_input, model_output, calibration_data_reader,
                        quant_format, input_nodes, output_nodes,
                        op_types_to_quantize, random_data_reader_input_shape,
                        per_channel, reduce_range, activation_type, weight_type,
                        nodes_to_quantize, nodes_to_exclude, optimize_model,
                        use_external_data_format, calibrate_method,
                        execution_providers, enable_dpu, debug_mode,
                        fp16_fp32_convert, convert_nchw_to_nhwc, include_cle,
                        extra_options)
    if print_summary:
        fp32_nodes_dict = fp32_nodes(model_input)

    if debug_mode:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if convert_nchw_to_nhwc:
        from .utils.model_utils import convert_nchw_to_nhwc as convert_func
        logger.info(f"Start converting {model_input} ncwh to nhwc model.")
        try:
            origin_input_model = onnx.load(model_input)
            converted_model = convert_func(origin_input_model)
            converted_path = tempfile.TemporaryDirectory(prefix="vai.tools.")
            model_input = Path(
                converted_path.name).joinpath("converted.onnx").as_posix()
            onnx.save_model(converted_model,
                            model_input,
                            save_as_external_data=use_external_data_format)
        except Exception as e:
            logger.warning(f"Failed to convert nchw to nhwc beacuse {e}, ")

    mode = QuantizationMode.QLinearOps
    if calibration_data_reader is None:
        logger.info(
            "calibration_data_reader is None, using random data for calibration"
        )
        calibration_data_reader = RandomDataReader(
            model_input,
            input_shape=random_data_reader_input_shape,
            input_data_range=None
            if "RandomDataReaderInputDataRange" not in extra_options else
            extra_options["RandomDataReaderInputDataRange"])

    # TODO: Looking for alternative methods to replace the use of deepcopy
    try:
        check_onnx_model(model_input)
        test_dr = copy.deepcopy(calibration_data_reader)
        run_onnx_model(model_input, test_dr)
    except Exception as e:
        logger.debug(
            f"Fail to validate if the {model_input} is runnable "
            f"since the deepcopy of 'calibration_data_reader' failed. Exception: {e}"
        )

    if input_nodes or output_nodes:
        if nodes_to_exclude:
            nodes_to_exclude += get_exclude_nodes(model_input, input_nodes,
                                                  output_nodes)
        else:
            nodes_to_exclude = get_exclude_nodes(model_input, input_nodes,
                                                 output_nodes)

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        q_linear_ops = list(QLinearOpsRegistry.keys())
        qdq_ops = list(QDQRegistry.keys())
        if enable_dpu:
            dpu_ops = list(DPURegistry.keys())
            qdq_ops = list(set(dpu_ops + qdq_ops))
        op_types_to_quantize = list(set(q_linear_ops + qdq_ops))

    remove_input_init = True
    if "RemoveInputInit" in extra_options:
        remove_input_init = extra_options["RemoveInputInit"]
    if remove_input_init:
        try:
            model = onnx.load(model_input)
            model_rm_input_init = remove_initializer_from_input(model)
            model_rm_input_init_path = tempfile.TemporaryDirectory(
                prefix="vai.tools.")
            model_input = Path(model_rm_input_init_path.name).joinpath(
                "rm_input_init.onnx").as_posix()
            onnx.save_model(model_rm_input_init,
                            model_input,
                            save_as_external_data=use_external_data_format)
            logger.info("Removed initializers from input")
        except Exception as e:
            logger.debug(f"Fail to remove init from input because {e}")

    if fp16_fp32_convert:
        logger.info(f"Start converting {model_input} to float32 model.")
        try:
            from onnxsim import simplify
        except Exception as e:
            logger.error("onnxsim is not correctly installed. "
                         "Please install using 'python -m pip install onnxsim'")
            return
        from .tools import float16
        fp16_model = onnx.load(model_input)
        try:
            fp32_model = float16.convert_float16_to_float(fp16_model)
            try:
                model_simp, check = simplify(fp32_model)
                assert check, "Simplified ONNX model could not be validated"
                logger.info(
                    f"Convert {model_input} to float32 model sucessfully")
            except Exception as e2:
                logger.warning(f"Fail to Simplify ONNX model because of {e2}.")
                model_simp = fp32_model
        except Exception as e:
            logger.warning(f"Fail to convert fp16 to fp32 beacuse {e}, "
                           "skip fp16 to fp32 conversion.")
            model_simp = fp16_model

        fp32_path = tempfile.TemporaryDirectory(prefix="vai.tools.")
        model_input = Path(fp32_path.name).joinpath("fp32.onnx").as_posix()
        onnx.save_model(model_simp,
                        model_input,
                        save_as_external_data=use_external_data_format)
    logger.info("Loading model...")
    if is_ort_version_below_1_16():
        from onnxruntime.quantization.quant_utils import load_model
        try:
            model = load_model(Path(model_input), optimize_model)
        except Exception as e:
            logger.error(
                "Model loading failed as shape inference needs write access to the model input directory."
                "Please verify permissions of the model input directory.")
            return
    else:
        from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
        if optimize_model:
            from onnxruntime.quantization.quant_utils import optimize_model
            try:
                quant_tmp_dir = tempfile.TemporaryDirectory(
                    prefix="vai.quant.opt.")
                opt_model_path = Path(
                    quant_tmp_dir.name).joinpath("model.onnx").as_posix()
                optimize_model(Path(model_input), Path(opt_model_path))
                model = load_model_with_shape_infer(Path(opt_model_path))
            except Exception as e:
                logger.warning(
                    f"Failed to run quantization preprocessing with error of {e}. "
                    "Using original model. Please check.")
                try:
                    model = load_model_with_shape_infer(Path(model_input))
                except Exception as e:
                    logger.error(
                        "Model loading failed as shape inference needs write access to the model input directory."
                        "Please verify permissions of the model input directory."
                    )
                    return
        else:
            try:
                model = load_model_with_shape_infer(Path(model_input))
            except Exception as e:
                logger.error(
                    "Model loading failed as shape inference needs write access to the model input directory."
                    "Please verify permissions of the model input directory.")
                return

    clip6_to_relu6 = False
    if "ReplaceClip6Relu" in extra_options:
        clip6_to_relu6 = extra_options['ReplaceClip6Relu']

    if clip6_to_relu6:
        model = replace_all_clip6_to_relu(model, op_types_to_quantize,
                                          nodes_to_quantize, nodes_to_exclude)
        clip6relu_path = tempfile.TemporaryDirectory(prefix="vai.quant.")
        clip6relu_model_output = Path(
            clip6relu_path.name).joinpath("clip6relu_model.onnx").as_posix()
        onnx.save_model(model,
                        clip6relu_model_output,
                        save_as_external_data=use_external_data_format)
        model_input = clip6relu_model_output

    if include_cle:
        cle_balance_method = "max"
        cle_steps = 1
        cle_weight_threshold = 0.5
        cle_scale_append_bias = True
        cle_scale_use_threshold = True
        cle_total_layer_diff_threshold = 2e-7

        if "CLEBalanceMethod" in extra_options:
            cle_balance_method = extra_options['CLEBalanceMethod']
        if "CLEWeightThreshold" in extra_options:
            cle_weight_threshold = extra_options['CLEWeightThreshold']
        if "CLEScaleUseThreshold" in extra_options:
            cle_scale_use_threshold = extra_options['CLEScaleUseThreshold']
        if "CLEScaleAppendBias" in extra_options:
            cle_scale_append_bias = extra_options['CLEScaleAppendBias']
        if "CLESteps" in extra_options:
            cle_steps = extra_options['CLESteps']
        if "CLETotalLayerDiffThreshold" in extra_options:
            cle_total_layer_diff_threshold = extra_options[
                'CLETotalLayerDiffThreshold']

        model = cle_transforms(
            model,
            op_types_to_quantize,
            nodes_to_quantize,
            nodes_to_exclude,
            cle_steps,
            cle_balance_method,
            cle_weight_threshold,
            cle_scale_append_bias,
            cle_scale_use_threshold,
            cle_total_layer_diff_threshold,
        )

        cle_path = tempfile.TemporaryDirectory(prefix="vai.quant.")
        cle_model_output = Path(
            cle_path.name).joinpath("cle_model.onnx").as_posix()
        onnx.save_model(model,
                        cle_model_output,
                        save_as_external_data=use_external_data_format)
        model_input = cle_model_output

    int16_scale = False
    if "Int16Scale" in extra_options:
        int16_scale = extra_options["Int16Scale"]
    if int16_scale:
        if enable_dpu or quant_format in VitisQuantFormat:
            logger.error(
                "Int16Scale is an experimental feature"
                "and cannot be used simultaneously with enable_dpu and VitisQuantFormat"
            )
            return

    if not int16_scale and calibrate_method in CalibrationMethod:
        if is_ort_version_below_1_16():
            return ort_quantize_static(
                model_input, model_output, calibration_data_reader,
                quant_format, op_types_to_quantize, per_channel, reduce_range,
                activation_type, weight_type, nodes_to_quantize,
                nodes_to_exclude, optimize_model, use_external_data_format,
                calibrate_method, extra_options)
        else:
            return ort_quantize_static(
                model_input, model_output, calibration_data_reader,
                quant_format, op_types_to_quantize, per_channel, reduce_range,
                activation_type, weight_type, nodes_to_quantize,
                nodes_to_exclude, use_external_data_format, calibrate_method,
                extra_options)

    model.opset_import.append(helper.make_operatorsetid(VAI_DOMAIN, 1))

    if enable_dpu:
        logger.info(
            "enable_dpu is True, optimize the model for better hardware compatibility."
        )
        convert_bn_to_conv = True
        if "ConvertBNToConv" in extra_options:
            convert_bn_to_conv = extra_options["ConvertBNToConv"]
        convert_reduce_mean_to_global_avg_pool = True
        if "ConvertReduceMeanToGlobalAvgPool" in extra_options:
            convert_reduce_mean_to_global_avg_pool = extra_options[
                "ConvertReduceMeanToGlobalAvgPool"]
        split_large_kernel_pool = True
        if "SplitLargeKernelPool" in extra_options:
            split_large_kernel_pool = extra_options["SplitLargeKernelPool"]
        convert_split_to_slice = True
        if "ConvertSplitToSlice" in extra_options:
            convert_split_to_slice = extra_options["ConvertSplitToSlice"]
        fuse_instance_norm = False
        if "FuseInstanceNorm" in extra_options:
            fuse_instance_norm = extra_options["FuseInstanceNorm"]
        fuse_l2_norm = False
        if "FuseL2Norm" in extra_options:
            fuse_l2_norm = extra_options["FuseL2Norm"]
        model = optimize(
            model,
            op_types_to_quantize,
            nodes_to_quantize,
            nodes_to_exclude,
            convert_bn_to_conv,
            convert_reduce_mean_to_global_avg_pool,
            split_large_kernel_pool,
            convert_split_to_slice,
            fuse_instance_norm,
            fuse_l2_norm,
            convert_clip_to_relu=False,
        )

        if is_ort_version_below_1_16():
            from onnxruntime.quantization.quant_utils import save_and_reload_model
            model = save_and_reload_model(model)
        else:
            from onnxruntime.quantization.quant_utils import save_and_reload_model_with_shape_infer
            model = save_and_reload_model_with_shape_infer(model)

    calib_extra_options_keys = [
        ("CalibTensorRangeSymmetric", "symmetric"),
        ("CalibMovingAverage", "moving_average"),
        ("CalibMovingAverageConstant", "averaging_constant"),
        ("MinMSEMode", "minmse_mode"),
    ]
    calib_extra_options = {
        key: extra_options.get(name)
        for (name, key) in calib_extra_options_keys
        if name in extra_options
    }

    optimized_path = tempfile.TemporaryDirectory(prefix="vai.quant.")
    model_input = Path(
        optimized_path.name).joinpath("opt_model.onnx").as_posix()
    onnx.save_model(model,
                    model_input,
                    save_as_external_data=use_external_data_format)
    logger.info("Start calibration...")
    if calibrate_method in PowerOfTwoMethod:
        with tempfile.TemporaryDirectory(prefix="vai.quant.") as quant_tmp_dir:
            calibrator = create_calibrator_power_of_two(
                Path(model_input),
                op_types_to_quantize,
                augmented_model_path=Path(quant_tmp_dir).joinpath(
                    "augmented_model.onnx").as_posix(),
                activation_type=activation_type,
                method=calibrate_method,
                use_external_data_format=use_external_data_format,
                execution_providers=execution_providers,
                extra_options=calib_extra_options,
            )
            logger.info(
                "Start collecting data, runtime depends on your model size and the number of calibration dataset."
            )
            calibrator.collect_data(calibration_data_reader)
            if is_ort_version_below_1_16():
                tensors_range = calibrator.compute_range()
            elif calibrate_method == PowerOfTwoMethod.MinMSE:
                tensors_range = calibrator.compute_range()
                from onnxruntime.quantization.calibrate import TensorsData
                tensors_range = TensorsData(CalibrationMethod.MinMax,
                                            tensors_range)
            else:
                tensors_range = calibrator.compute_data()
            del calibrator
    else:
        with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
            calibrator = create_calibrator(
                Path(model_input),
                op_types_to_quantize,
                augmented_model_path=Path(quant_tmp_dir).joinpath(
                    "augmented_model.onnx").as_posix(),
                calibrate_method=calibrate_method,
                use_external_data_format=use_external_data_format,
                extra_options=calib_extra_options,
            )
            logger.info(
                "Start collecting data, runtime depends on your model size and the number of calibration dataset."
            )
            calibrator.collect_data(calibration_data_reader)
            if is_ort_version_below_1_16():
                tensors_range = calibrator.compute_range()
            else:
                tensors_range = calibrator.compute_data()
            del calibrator

    if enable_dpu:
        from .quant_utils import remove_qdq_op_type
        if extra_options.get("RemoveQDQConvLeakyRelu", False):
            remove_qdq_op_type.append("LeakyRelu")
        if extra_options.get("RemoveQDQConvPRelu", False):
            remove_qdq_op_type.append("PRelu")

        if is_ort_version_below_1_16():
            from onnxruntime.quantization.quant_utils import load_model
            model = load_model(Path(model_input), False)
        else:
            from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
            model = load_model_with_shape_infer(Path(model_input))
        # This optimization should after calibration
        convert_clip_to_relu = False
        if "ConvertClipToRelu" in extra_options:
            convert_clip_to_relu = extra_options["ConvertClipToRelu"]
        model = optimize(
            model,
            op_types_to_quantize,
            nodes_to_quantize,
            nodes_to_exclude,
            convert_bn_to_conv=False,
            convert_reduce_mean_to_global_avg_pool=False,
            split_large_kernel_pool=False,
            convert_split_to_slice=False,
            fuse_instance_norm=False,
            fuse_l2_norm=False,
            convert_clip_to_relu=convert_clip_to_relu,
        )
        if is_ort_version_below_1_16():
            from onnxruntime.quantization.quant_utils import save_and_reload_model
            model = save_and_reload_model(model)
        else:
            from onnxruntime.quantization.quant_utils import save_and_reload_model_with_shape_infer
            model = save_and_reload_model_with_shape_infer(model)

    check_static_quant_arguments(quant_format, activation_type, weight_type)
    if int16_scale:
        calibrate_method = Int16Method.MinMax

    if quant_format is QuantFormat.QOperator:
        quantizer = VitisONNXQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            extra_options,
        )
    elif quant_format is QuantFormat.QDQ and not enable_dpu:
        quantizer = VitisQDQQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            extra_options,
        )
    elif quant_format is QuantFormat.QDQ and enable_dpu:
        quantizer = VitisQDQDPUQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            extra_options,
        )
    elif quant_format is VitisQuantFormat.FixNeuron or quant_format is VitisQuantFormat.QDQ:
        quantizer = VitisExtendedQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            quant_format,
            True,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            extra_options,
        )
    # add Quantizer for BFP
    elif quant_format is VitisQuantFormat.BFPFixNeuron:
        quantizer = VitisBFPQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            extra_options,
        )

    quantizer.quantize_model()

    quantizer.model.save_model_to_file(model_output, use_external_data_format)
    if print_summary and fp32_nodes_dict:
        print_fp32_nodes(fp32_nodes_dict, model_output)
