# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.21)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(ParseVersion)

# sets version_core and other variables
dd_parse_version(${CMAKE_CURRENT_SOURCE_DIR}/VERSION)
project(DynamicDispatch VERSION ${version_core} LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
  set(CMAKE_C_STANDARD 99)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_CXX_EXTENSION OFF)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

set(DD_SRC_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)

if(MSVC)
  set(DD_DEFAULT_COMPILE_OPTIONS "/Zc:__cplusplus" "/W3")
endif()

include(FetchContent)
if(CMAKE_VERSION VERSION_GREATER "3.24")
  cmake_policy(SET CMP0135 NEW)
endif()

find_package(aie_controller REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(spdlog REQUIRED)
find_package(xaiengine REQUIRED)
find_package(XRT REQUIRED)
include(zlib_dep) # find zlib

option(UNIT_TEST_PERF_EN "Enable Performance checks in unit tests" OFF)
option(ENABLE_DD_TESTS "Enable tests" OFF)
option(ENABLE_SIMNOWLITE_BUILD "Build for linux simnowlite testing" OFF)
option(LOGGING_EN "Enable debug logging" OFF)
option(PERF_LOGGING_EN "Enable performance logging" OFF)
option(ENABLE_DD_PYTHON "Enable building the Python library" OFF)

if(ENABLE_SIMNOWLITE_BUILD)
  add_definitions(-DSIMNOWLITE_EN)
  message("SIMNOWLITE_EN is defined.")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.3")
      message(
        STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}"
      )
      message(FATAL_ERROR "\nTaskflow requires g++ at least v9.3")
    endif()
  endif()
endif()

if(LOGGING_EN)
  add_definitions(-DRYZENAI_TRACE)
  message("RYZENAI_TRACE is defined.")
endif()

if(PERF_LOGGING_EN)
  add_definitions(-DRYZENAI_PERF)
  message("RYZENAI_PERF is defined.")
endif()

# set the debug flag
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DRYZENAI_DEBUG)
  message("RYZENAI_DEBUG is defined.")
endif()

add_subdirectory(src)
add_subdirectory(transaction)

if(ENABLE_DD_TESTS)
  add_subdirectory(tests)
endif()
