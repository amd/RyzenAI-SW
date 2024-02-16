#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os

# Build for which device
_DEVICE_SUFFIX = 'DEVICE_SED_MASK'

# Copied from custom_op_library.cc
_COP_DOMAIN = "com.vai.quantize"
_COP_VERSION = 1


def get_library_path(device: str = "cpu"):
    assert device in [
        "cpu", "CPU", "cuda", "CUDA", "rocm", "ROCM"
    ], "Valid devices are cpu/CPU or cuda/CUDA or rocm/ROCM, default is cpu."
    if device == "cuda" or device == "CUDA":
        lib_name = "libvai_custom_op_cuda.so"
    elif device == "rocm" or device == "ROCM":
        lib_name = "libvai_custom_op_rocm.so"
    else:
        lib_name = "libvai_custom_op.so"
    dir_path = os.path.dirname(__file__)
    lib_path = os.path.join(dir_path, lib_name)
    assert os.path.exists(
        lib_path), f"Lib {lib_path} does NOT exist, may not have been compiled."

    return lib_path
