#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from onnxruntime.quantization.calibrate import CalibraterBase, CalibrationDataReader, CalibrationMethod, MinMaxCalibrater, create_calibrator
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType, write_calibration_table
from .calibrate import create_calibrator_power_of_two, PowerOfTwoMethod
from .qdq_quantizer import VitisExtendedQuantizer
from .quant_utils import VitisQuantType, VitisQuantFormat, dump_model, Int16Method
from .quantize import QuantizationMode, quantize_static
from .version import __version__, __git_version__

try:
    # The custom op library may not have been compiled
    from vai_q_onnx.gen_files import _COP_DOMAIN, _COP_VERSION, _DEVICE_SUFFIX, get_library_path
except:
    # Try to import from original path but may raise an error when call get_library_path
    from vai_q_onnx.operators.custom_ops import _COP_DOMAIN, _COP_VERSION, _DEVICE_SUFFIX, get_library_path
