# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
include(FetchContent)

find_package(DynamicDispatch CONFIG QUIET)
if(NOT DynamicDispatch_FOUND)
  message(
    STATUS "No DynamicDispatch found in cmake_prefix_path: ${CMAKE_PREFIX_PATH}"
  )
  get_filename_component(
    TRANSFORMERS_MODULE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY
  )
  set(DD_PATH ${TRANSFORMERS_MODULE_DIR}/../ext/DynamicDispatch)
  if(EXISTS "${DD_PATH}/CMakeLists.txt")
    message(STATUS "Using DynamicDispatch from local directory: ${DD_PATH}")
    FetchContent_Declare(dd SOURCE_DIR "${DD_PATH}")
  else()
    message(STATUS "Downloading DynamicDispatch with FetchContent")
    FetchContent_Declare(
      dd
      GIT_REPOSITORY
        "https://gitenterprise.xilinx.com/VitisAI/DynamicDispatch.git"
      GIT_TAG "66c31f4263a35dc04cea6ae82d6115620875425a"
    )
  endif()
  FetchContent_MakeAvailable(dd)
  set(DYNAMIC_DISPATCH_INCLUDE_DIRS "${dd_BINARY_DIR}/include")
  set(DynamicDispatch_FOUND TRUE)
  if(NOT TARGET DynamicDispatch::dyn_dispatch_core)
    add_library(DynamicDispatch::dyn_dispatch_core ALIAS dyn_dispatch_core)
    add_library(DynamicDispatch::transaction ALIAS transaction)
  endif()
else()
  cmake_path(SET CONDA_PREFIX $ENV{CONDA_PREFIX})
  set(DYNAMIC_DISPATCH_INCLUDE_DIRS
      ${CONDA_PREFIX}/include/ryzenai/dynamic_dispatch
  )
endif()

message(
  STATUS "DYNAMIC_DISPATCH_INCLUDE_DIRS : ${DYNAMIC_DISPATCH_INCLUDE_DIRS}"
)
