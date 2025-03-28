# Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.

find_package(OpenCV REQUIRED PATHS ${OPENCV_INSTALL_ROOT})

#
# Find Ryzen AI library (also sets include directories)
#
if (NOT DEFINED AMD_CVML_SDK_ROOT)
    set(AMD_CVML_SDK_ROOT ../..)
endif()
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
