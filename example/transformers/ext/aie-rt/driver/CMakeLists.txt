###############################################################################
# Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################

cmake_minimum_required (VERSION 3.16.3)

list (APPEND CMAKE_MODULE_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/platforms")

project (xaiengine VERSION 3.2)

include (CheckIncludeFiles)
include (CheckCSourceCompiles)
include (collect)
include (options)

enable_testing ()

add_subdirectory (src ${CMAKE_CURRENT_BINARY_DIR}/driver-src)

if (WITH_TESTS)
  add_subdirectory (tests)
endif (WITH_TESTS)
