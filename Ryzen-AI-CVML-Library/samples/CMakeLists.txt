# Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# To compile all the Ryzen AI CVML sample applications, execute the following
# commands from the samples folder,
#
# For Windows
# > mkdir build
# > cmake -S %CD% -B %CD%\build -DOPENCV_INSTALL_ROOT=%OPENCV_INSTALL_ROOT%
# > cmake --build %CD%\build --config Release
#
# For Linux
# > mkdir build
# > cmake -S $PWD -B $PWD/build -DOPENCV_INSTALL_ROOT=$OPENCV_INSTALL_ROOT
# > cmake --build $PWD/build --config Release
#
# where OPENCV_INSTALL_ROOT is set to the location of the OpenCV library files.
#

#
# Build the sample apps inside Ryzen AI CVML release
#
cmake_minimum_required(VERSION 3.18)
project(cvml-sdk-samples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

#
# Please set opencv install root path, using the following line as an example,
#
file(TO_CMAKE_PATH "$ENV{OPENCV_INSTALL_ROOT}" OPENCV_INSTALL_ROOT)
find_package(OpenCV PATHS ${OPENCV_INSTALL_ROOT})
if (NOT OpenCV_FOUND)
    # Try default OpenCV config, workaround for older OpenCV versions
    if (WIN32)
        set(OpenCV_RUNTIME vc16)
        set(OpenCV_ARCH x64)
        set(OpenCV_DIR "${OPENCV_INSTALL_ROOT}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/lib")
        message(STATUS "First attempt to find OpenCV failed. Trying again with RUNTIME=${OpenCV_RUNTIME}, ARCH=${OpenCV_ARCH}")
    else ()
        set(OpenCV_DIR "${OPENCV_INSTALL_ROOT}/lib")
        message(STATUS "First attempt to find OpenCV failed. Trying again with OpenCV_DIR=${OPENCV_INSTALL_ROOT}/lib")
    endif ()
    find_package(OpenCV REQUIRED PATHS ${OPENCV_INSTALL_ROOT})
endif()

#
# Find Ryzen AI CVML library (also sets include directories)
#
if (NOT DEFINED AMD_CVML_SDK_ROOT)
    set(AMD_CVML_SDK_ROOT ..)
endif()
find_package(RyzenAILibrary REQUIRED PATHS ${AMD_CVML_SDK_ROOT})

set(CMAKE_CXX_FLAGS "-D_UNICODE -DUNICODE")
if (MSVC)
    add_compile_options(/EHsc)
endif ()

# Add the subdirectory of the samples to build
add_subdirectory(common-sample-utils)
file(GLOB SAMPLE_SRC_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} cvml-sample-*)
# Filter out any existing internal samples
file(GLOB INTERNAL_SAMPLE_SRC_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *internal*)
list(REMOVE_ITEM SAMPLE_SRC_DIRS ${INTERNAL_SAMPLE_SRC_DIRS})
FOREACH(ITER ${SAMPLE_SRC_DIRS})
    add_subdirectory(${ITER})
ENDFOREACH()
