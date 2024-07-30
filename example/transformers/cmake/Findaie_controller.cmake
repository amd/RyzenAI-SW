# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
include(FetchContent)

find_package(aie_controller CONFIG QUIET)
if(NOT aie_controller_FOUND)
  message(
    STATUS "No aie_controller found in cmake_prefix_path: ${CMAKE_PREFIX_PATH}"
  )
  get_filename_component(
    TRANSFORMERS_MODULE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY
  )
  set(AIE_CONTROLLER_PATH ${TRANSFORMERS_MODULE_DIR}/../ext/aie_controller)
  if(EXISTS "${AIE_CONTROLLER_PATH}/CMakeLists.txt")
    message(
      STATUS "Using aie_controller from local directory: ${AIE_CONTROLLER_PATH}"
    )
    FetchContent_Declare(aie_ctrl SOURCE_DIR "${AIE_CONTROLLER_PATH}")
  else()
    message(STATUS "Downloading aie_controller with FetchContent")
    FetchContent_Declare(
      aie_ctrl
      GIT_REPOSITORY
        "https://gitenterprise.xilinx.com/VitisAI/aie_controller.git"
      GIT_TAG "v1.3"
    )
  endif()
  FetchContent_MakeAvailable(aie_ctrl)
  set(aie_controller_FOUND TRUE)
endif()
