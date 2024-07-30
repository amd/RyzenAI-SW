cmake_minimum_required(VERSION 3.18)
project(cpp_tests VERSION 0.0.1)
include(GoogleTest)
include(CTest)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /W4")
endif()

cmake_path(SET ENV_PYTORCH_AIE_PATH $ENV{PYTORCH_AIE_PATH})
list(APPEND CMAKE_MODULE_PATH ${ENV_PYTORCH_AIE_PATH}/cmake)

include(FetchContent)
find_package(xaiengine REQUIRED)
find_package(aie_controller REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

cmake_path(SET XRT_DIR $ENV{XRT_PATH})
find_package(XRT REQUIRED PATHS ${XRT_DIR})
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)

cmake_path(SET TRANSFORMERS_ROOT $ENV{PYTORCH_AIE_PATH})
set(OPS_ROOT ${TRANSFORMERS_ROOT}/ops)

add_definitions(-DXAIE_FEATURE_MSVC)
option(RYZENAI_PERF_ "Enable logging perf metrics" OFF)
if(RYZENAI_PERF_)
  add_definitions(-DRYZENAI_PERF)
endif()

set(SOURCES test_qlinear_2.cpp test_linear.cpp)
set(INCLUDE_DIRECTORIES
    ${GTEST_INCLUDE_DIRS} ${OPS_ROOT}/cpp/qlinear_2 ${OPS_ROOT}/cpp/linear
    ${OPS_ROOT}/cpp/utils ${XRT_INCLUDE_DIRS}
)
set(LINK_LIBRARIES
    ${CMAKE_DL_LIBS}
    GTest::GTest
    GTest::Main
    XRT::xrt_coreutil
    xaiengine::xaiengine
    spdlog::spdlog
    Eigen3::Eigen
    aie_controller
)

# cpp_tests
add_executable(cpp_tests ${SOURCES})

target_include_directories(
  cpp_tests PUBLIC ${INCLUDE_DIRECTORIES} ${AieRt_INCLUDE_DIRS}
)
target_link_libraries(cpp_tests PUBLIC ${LINK_LIBRARIES})
