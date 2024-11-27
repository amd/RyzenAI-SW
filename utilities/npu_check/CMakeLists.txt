# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.13)

project("npu_util" VERSION 1.0.0.0)

add_executable(npu_util npu_util.cpp example.cpp)
    
set(CMAKE_DEBUG_POSTFIX "-d")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

target_compile_options(npu_util PRIVATE /Qspectre /ZH:SHA_256 /guard:cf /W3 /Zi /Zc:__cplusplus /std:c++latest)
target_link_options(npu_util PRIVATE "/DEBUG" "/CETCOMPAT")

target_include_directories(npu_util PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(FILES
    ${INSTALL_FILES_LIST}
    CONFIGURATIONS ${CMAKE_BUILD_TYPE}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}
)


         