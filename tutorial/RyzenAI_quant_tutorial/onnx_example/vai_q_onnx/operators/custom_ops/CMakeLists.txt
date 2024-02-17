#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

cmake_minimum_required(VERSION 3.5)
project (vai_custom_op)
message("PROJECT : vai_custom_op")
message("PROJECT DIRECTORY : ${PROJECT_SOURCE_DIR}")

#CXX flag settings
set(CMAKE_CXX_FLAGS "")

option(USE_CUDA "option for use cuda" OFF)
option(USE_ROCM "option for use rocm" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    message(STATUS "find CUDA version: " ${CUDA_VERSION_STRING})
    message("build with USE_CUDA")
    set(CMAKE_CXX_FLAGS "-DUSE_CUDA ${CMAKE_CXX_FLAGS}")
    set(THE_LIBRARY_NAME "${PROJECT_NAME}_cuda")
elseif(USE_ROCM)
    message("build with USE_ROCM")
    set(CMAKE_CXX_FLAGS "-DUSE_ROCM ${CMAKE_CXX_FLAGS}")
    set(THE_LIBRARY_NAME "${PROJECT_NAME}_rocm")
else()
    message("build with CPU")
    set(THE_LIBRARY_NAME "${PROJECT_NAME}")
endif()

option(DEBUG "option for debug" OFF)
if(DEBUG)
    message("build with DEBUG")
    set(CMAKE_CXX_FLAGS "-O0 -ggdb ${CMAKE_CXX_FLAGS}")
else()
    message("build with RELEASE")
    set(CMAKE_BUILE_TYPE RELEASE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

#Source code for Custom Op
set(HEADER ${PROJECT_SOURCE_DIR}/include/custom_op_library.h)
set(SOURCE ${PROJECT_SOURCE_DIR}/src/custom_op_library.cc
	   ${PROJECT_SOURCE_DIR}/src/custom_op_qdq.cc
       ${PROJECT_SOURCE_DIR}/src/custom_op_bfp.cc)

if(USE_CUDA)
    file(GLOB BFP_SOURCES ${PROJECT_SOURCE_DIR}/src/bfp/cuda/*)
    list(APPEND SOURCE ${BFP_SOURCES})
elseif(USE_ROCM)
else()
    file(GLOB BFP_SOURCES ${PROJECT_SOURCE_DIR}/src/bfp/cpu/*)
    list(APPEND SOURCE ${BFP_SOURCES})
endif()

add_library(${THE_LIBRARY_NAME} SHARED ${SOURCE})

#Include path to header files for Custom Op
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

include_directories(${PROJECT_SOURCE_DIR}/src/qdq)
include_directories(${PROJECT_SOURCE_DIR}/src/qdq/cpu)
include_directories(${PROJECT_SOURCE_DIR}/src/qdq/cuda)
include_directories(${PROJECT_SOURCE_DIR}/src/qdq/rocm)

set(ORT_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include/onnxruntime-1.15.0)
set(GSL_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include/gsl-4.0.0)
include_directories(${ORT_INCLUDE_PATH})
include_directories(${ORT_INCLUDE_PATH}/onnxruntime)
include_directories(${ORT_INCLUDE_PATH}/onnxruntime/core/session)
include_directories(${GSL_INCLUDE_PATH})

if(USE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(${THE_LIBRARY_NAME} ${CUDA_LIBRARIES})
endif()
