# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

find_package(spdlog CONFIG QUIET)
if(NOT spdlog_FOUND)
  message(STATUS "Using spdlog from FetchContent")
  FetchContent_Declare(
    spdlog GIT_REPOSITORY "https://github.com/gabime/spdlog" GIT_TAG "v1.11.0"
  )
  FetchContent_MakeAvailable(spdlog)
endif()
