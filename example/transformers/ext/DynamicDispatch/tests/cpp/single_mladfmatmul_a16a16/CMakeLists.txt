# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

add_executable(test_single_mladfmatmul_a16a16 test_fusion.cpp)
dd_configure_test(test_single_mladfmatmul_a16a16 OFF)
