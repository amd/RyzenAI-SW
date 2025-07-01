# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.

set(PROJECT_NAME cvml-sample-depth-estimation)
project(${PROJECT_NAME})

#
# Find OpenCV and set include directories
#
find_package(OpenCV REQUIRED PATHS ${OPENCV_INSTALL_ROOT})
include_directories(${OpenCV_INCLUDE_DIRS})

#
# Find Ryzen AI CVML library (also sets include directories)
#
if (NOT DEFINED AMD_CVML_SDK_ROOT)
    set(AMD_CVML_SDK_ROOT ../..)
endif()
find_package(RyzenAILibrary REQUIRED PATHS ${AMD_CVML_SDK_ROOT})

#
# Additional include folders for the sample
#
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common-sample-utils/include)

#
# Define source files, application and link libraries
#
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${RyzenAILibrary_LIBS}
    common-sample-utils
)

#
# Installation rules.
#
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION  ${AMD_CVML_SDK_ROOT}/samples/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION ${AMD_CVML_SDK_ROOT}/samples/${PROJECT_NAME})
