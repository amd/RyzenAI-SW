# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "Using nlohmann_json from FetchContent")
  FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  )
  FetchContent_MakeAvailable(json)
endif()
