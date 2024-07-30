# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

add_executable(test_mladfmatmul test_fusion.cpp)
dd_configure_test(test_mladfmatmul OFF)
