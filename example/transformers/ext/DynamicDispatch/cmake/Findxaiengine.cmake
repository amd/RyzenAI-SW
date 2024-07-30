# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

find_package(xaiengine CONFIG QUIET)
if(NOT xaiengine_FOUND)
  message(STATUS "Using xaiengine from FetchContent")
  FetchContent_Declare(
    AIERT
    GIT_REPOSITORY "https://gitenterprise.xilinx.com/bryanloz/aie-rt.git"
    GIT_TAG "main_aig"
    SOURCE_SUBDIR driver/src
  )
  set(XAIENGINE_BUILD_SHARED OFF CACHE INTERNAL "We want static library")
  FetchContent_MakeAvailable(AIERT)
  set(xaiengine_FOUND TRUE)
  add_library(xaiengine::xaiengine ALIAS xaiengine)
endif()
