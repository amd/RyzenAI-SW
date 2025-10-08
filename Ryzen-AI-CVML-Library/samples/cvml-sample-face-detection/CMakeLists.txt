# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.

set(FEATURE_NAME cvml-sample-face-detection)
project(${FEATURE_NAME})

list(APPEND CMAKE_MODULE_PATH ${OPENCV_INSTALL_ROOT} ${AMD_CVML_SDK_ROOT} "${CMAKE_CURRENT_SOURCE_DIR}/../..")

find_package(OpenCV REQUIRED)
find_package(RyzenAILibrary REQUIRED)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../common-sample-utils/include)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${RyzenAILibrary_LIBS}
    common-sample-utils
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION  ${AMD_CVML_SDK_ROOT}/samples/${FEATURE_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION ${AMD_CVML_SDK_ROOT}/samples/${FEATURE_NAME})
