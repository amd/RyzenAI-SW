cmake_minimum_required(VERSION 3.18)
project(system_plugins VERSION 0.0.1)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /W4")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set(XRT_DIR $ENV{XRT_PATH})
find_package(XRT REQUIRED PATHS ${XRT_DIR})
find_package(GTest REQUIRED)
find_package(spdlog REQUIRED)

set(TRANSFORMERS_ROOT $ENV{PYTORCH_AIE_PATH})
set(OPS_ROOT ${TRANSFORMERS_ROOT}/ops)

set(TVM_ROOT $ENV{TVM_PATH})
set(DMLC_CORE "${TVM_ROOT}/3rdparty/dmlc-core")

# Allows fully qualified directories for #include directives.
include_directories(${TRANSFORMERS_ROOT}/iree)

enable_testing()

function(add_gtest test_name test_filename)
  add_executable(${test_name} ${test_filename})
  gtest_discover_tests(${test_name})
  target_link_libraries(${test_name} GTest::gmock_main ${ARGN})
endfunction()

# Plugins
add_subdirectory(plugins)
