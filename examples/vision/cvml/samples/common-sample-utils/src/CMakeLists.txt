# Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.

list(APPEND CMAKE_MODULE_PATH ${OPENCV_INSTALL_ROOT} ${AMD_CVML_SDK_ROOT} "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

find_package(OpenCV REQUIRED)

#
# Find Ryzen AI CVML library (also sets include directories)
#
find_package(RyzenAILibrary REQUIRED PATHS ${AMD_CVML_SDK_ROOT})

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include ${OpenCV_INCLUDE_DIRS})

add_library(common-sample-utils STATIC ${SOURCES})

target_include_directories(common-sample-utils
PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(common-sample-utils
    ${OpenCV_LIBS}
    ${RyzenAILibrary_LIBS}
)
