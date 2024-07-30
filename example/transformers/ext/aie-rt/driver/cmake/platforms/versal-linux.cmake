###############################################################################
# Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################

set (CMAKE_SYSTEM_PROCESSOR "aarch64"            CACHE STRING "")
set (CROSS_PREFIX           "aarch64-linux-gnu-" CACHE STRING "")
include (cross-linux-g++)
