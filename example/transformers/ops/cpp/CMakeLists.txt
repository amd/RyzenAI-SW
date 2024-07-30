# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.18)
project(_RyzenAI VERSION 0.0.1)

cmake_path(SET ENV_PYTORCH_AIE_PATH $ENV{PYTORCH_AIE_PATH})
list(APPEND CMAKE_MODULE_PATH ${ENV_PYTORCH_AIE_PATH}/cmake)

find_package(xaiengine REQUIRED)
find_package(aie_controller REQUIRED)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /W4 -march=native")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

find_package(Threads REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)

cmake_path(SET XRT_DIR $ENV{XRT_PATH})
find_package(XRT REQUIRED PATHS ${XRT_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

cmake_path(SET TRANSFORMERS_ROOT $ENV{PYTORCH_AIE_PATH})
set(OPS_ROOT ${TRANSFORMERS_ROOT}/ops)

set(LIBRARY_NAME "qlinear_2")
add_library(${LIBRARY_NAME} INTERFACE)

target_include_directories(
  ${LIBRARY_NAME}
  INTERFACE $<BUILD_INTERFACE:${OPS_ROOT}/cpp/qlinear_2>
            $<BUILD_INTERFACE:${OPS_ROOT}/cpp/utils>
            $<BUILD_INTERFACE:${XRT_INCLUDE_DIRS}>
            $<INSTALL_INTERFACE:include/ryzenai/ops/qlinear_2>
            $<INSTALL_INTERFACE:include/ryzenai/utils>
            $<INSTALL_INTERFACE:include>
)

target_link_libraries(
  ${LIBRARY_NAME} INTERFACE ${CMAKE_DL_LIBS} Threads::Threads XRT::xrt_coreutil
                            spdlog::spdlog Eigen3::Eigen aie_controller
)

if(MSVC)
  target_compile_options(
    ${LIBRARY_NAME} INTERFACE "/Zc:__cplusplus" "/W4" "/arch:AVX512"
  )
  target_compile_definitions(${LIBRARY_NAME} INTERFACE XAIE_FEATURE_MSVC)
else()
  target_compile_options(${LIBRARY_NAME} INTERFACE -mavx512f)
endif()

nanobind_add_module(${PROJECT_NAME} "./nanobind/pyryzenai.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARY_NAME})

if(WIN32)
  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION "lib")
endif()

# INSTALL EVERYTHING NEEDED FOR CONSUMERS
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/RyzenAIConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion
)
# CMAKE_INSTALL_PREFIX/include/ryzenai
install(FILES ${OPS_ROOT}/cpp/ryzenai/ryzenai.hpp DESTINATION include/ryzenai)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/qlinear_2/
  DESTINATION include/ryzenai/ops/qlinear_2
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/utils/
  DESTINATION include/ryzenai/utils
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
)
install(
  TARGETS ${LIBRARY_NAME}
  EXPORT ${LIBRARY_NAME}ConfigTargets
  INCLUDES
  DESTINATION include/ryzenai
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(
  EXPORT ${LIBRARY_NAME}ConfigTargets
  FILE RyzenAIConfigTargets.cmake
  NAMESPACE ryzenai::
  DESTINATION lib/cmake/ryzenai
)
install(FILES ${TRANSFORMERS_ROOT}/cmake/RyzenAIConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/RyzenAIConfigVersion.cmake
        DESTINATION lib/cmake/ryzenai
)
