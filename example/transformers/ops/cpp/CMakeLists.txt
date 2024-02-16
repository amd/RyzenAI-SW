cmake_minimum_required(VERSION 3.18)
project(RyzenAI)

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
endif()

find_package(Threads REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)

set (XRT_DIR $ENV{XRT_PATH})
find_package(XRT REQUIRED PATHS ${XRT_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set (THIRD_PARTY $ENV{THIRD_PARTY})
set (CONDA_PREFIX $ENV{CONDA_PREFIX})
set (DMLC_CORE "${THIRD_PARTY}/include/dmlc-core")
set (TRANSFORMERS_ROOT $ENV{PYTORCH_AIE_PATH})
set (OPS_ROOT ${TRANSFORMERS_ROOT}/ops)

file (GLOB SOURCES "./pybind/pyryzenai.cpp")
pybind11_add_module(RyzenAI ${SOURCES})

target_compile_definitions(
    ${PROJECT_NAME}
    PRIVATE
    -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${OPS_ROOT}/cpp/utils
    ${THIRD_PARTY}/include
    ${DMLC_CORE}/include
    ${THIRD_PARTY}/dlpack/include
    ${THIRD_PARTY}/include/maize
    ${XRT_INCLUDE_DIRS}
    ${CONDA_PREFIX}/Library/include
)

target_link_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${THIRD_PARTY}/lib
    ${THIRD_PARTY}/bin
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    ${CMAKE_DL_LIBS}
    Threads::Threads
    tvm_ipu_runtime
    XRT::xrt_coreutil
    maize
    spdlog::spdlog
    Eigen3::Eigen
)

target_compile_definitions(RyzenAI
                            PRIVATE)