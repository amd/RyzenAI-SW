# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.

set(FEATURE_NAME cvml-sample-face-detection)
project(${FEATURE_NAME})

find_package(OpenCV REQUIRED PATHS ${OPENCV_INSTALL_ROOT})

if (NOT DEFINED AMD_CVML_SDK_ROOT)
    set(AMD_CVML_SDK_ROOT ../..)
endif()
find_package(RyzenAILibrary REQUIRED PATHS ${AMD_CVML_SDK_ROOT})

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${AMD_CVML_SDK_ROOT}/include ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../common-sample-utils/include)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${RyzenAILibrary_LIBS}
    common-sample-utils
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION  ${AMD_CVML_SDK_ROOT}/samples/${FEATURE_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION ${AMD_CVML_SDK_ROOT}/samples/${FEATURE_NAME})
