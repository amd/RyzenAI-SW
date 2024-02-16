cmake_minimum_required(VERSION 3.18)
project(cpp_tests VERSION 0.0.1)
include(GoogleTest)
include(CTest)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set (XRT_DIR $ENV{XRT_PATH})
find_package(XRT REQUIRED PATHS ${XRT_DIR})
find_package(GTest REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)

set (TRANSFORMERS_ROOT $ENV{PYTORCH_AIE_PATH})
set (OPS_ROOT ${TRANSFORMERS_ROOT}/ops)

set (THIRD_PARTY $ENV{THIRD_PARTY})
set (DMLC_CORE "${THIRD_PARTY}/3rdparty/dmlc-core")

add_definitions(-DXAIE_FEATURE_MSVC)

set (SOURCES
    test_qlinear.cpp
    test_qlinear_2.cpp
    test_dynamicquantlinear.cpp
)

add_executable(cpp_tests ${SOURCES})

target_include_directories(
    cpp_tests
    PUBLIC
    ${GTEST_INCLUDE_DIRS}
    ${OPS_ROOT}/cpp/dynamic_quantlinear
    ${OPS_ROOT}/cpp/qlinear
    ${OPS_ROOT}/cpp/qlinear_2
    ${OPS_ROOT}/cpp/linear
    ${OPS_ROOT}/cpp/utils
    ${OPS_ROOT}/common
    ${THIRD_PARTY}/include
    ${DMLC_CORE}/include
	${XRT_INCLUDE_DIRS}
)

target_link_directories(    
    cpp_tests
    PUBLIC
    ${THIRD_PARTY}/lib
    ${THIRD_PARTY}/bin
)

target_link_libraries(
    cpp_tests
    PUBLIC
    ${CMAKE_DL_LIBS}
    tvm_ipu_runtime
    GTest::GTest
    GTest::Main
	XRT::xrt_coreutil
    maize
    spdlog::spdlog
    Eigen3::Eigen
)