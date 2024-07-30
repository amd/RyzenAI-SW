# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
include(FetchContent)

find_package(xaiengine CONFIG QUIET)
if(NOT xaiengine_FOUND)
  message(
    STATUS "No xaiengine found in cmake_prefix_path: ${CMAKE_PREFIX_PATH}"
  )
  get_filename_component(
    TRANSFORMERS_MODULE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY
  )
  set(xaiengine_PATH ${TRANSFORMERS_MODULE_DIR}/../ext/aie-rt)
  if(EXISTS "${xaiengine_PATH}/driver/src/CMakeLists.txt")
    message(STATUS "Using xaiengine from local directory: ${xaiengine_PATH}")
    FetchContent_Declare(
      xaiengine-fc SOURCE_DIR "${xaiengine_PATH}" SOURCE_SUBDIR driver/src
    )
  else()
    message(STATUS "Downloading xaiengine with FetchContent")
    FetchContent_Declare(
      xaiengine-fc
      GIT_REPOSITORY "https://gitenterprise.xilinx.com/ai-engine/aie-rt.git"
      GIT_TAG "eb21467b2e10f33e2278b9d1127da6e8f8376dab"
      SOURCE_SUBDIR driver/src
    )
  endif()
  set(XAIENGINE_BUILD_SHARED OFF CACHE INTERNAL "We want static library")
  FetchContent_MakeAvailable(xaiengine-fc)
  set(xaiengine_FOUND TRUE)
  if(NOT TARGET xaiengine::xaiengine)
    add_library(xaiengine::xaiengine ALIAS xaiengine)
  endif()
endif()
