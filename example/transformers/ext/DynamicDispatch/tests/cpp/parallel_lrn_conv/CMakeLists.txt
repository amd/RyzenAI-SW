# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

add_executable(test_parallel_lrn_conv test_fusion.cpp)
dd_configure_test(test_parallel_lrn_conv OFF)
