# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

find_package(aie_controller CONFIG QUIET)
if(NOT aie_controller_FOUND)
  message(STATUS "Using aie_controller from FetchContent")
  FetchContent_Declare(
    aie_ctrl
    GIT_REPOSITORY "https://gitenterprise.xilinx.com/VitisAI/aie_controller.git"
    GIT_TAG "v1.3"
  )
  FetchContent_MakeAvailable(aie_ctrl)
  set(aie_controller_FOUND TRUE)
endif()
