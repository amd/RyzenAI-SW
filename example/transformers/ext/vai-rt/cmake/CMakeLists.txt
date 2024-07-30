
cmake_minimum_required(VERSION 3.10)
project(VitisAI VERSION 1.0.0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(Options)
include(SourceVersions)
include(CMakeArgs)
include(Functions)

message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

if(BUILD_THIRD_PARTY)
  add_subdirectory(3rdparty/boost)
  add_subdirectory(3rdparty/glog)
  add_subdirectory(3rdparty/protobuf)
  add_subdirectory(3rdparty/pybind11)
  add_subdirectory(3rdparty/googletest)
else()
  include(FakeTargets)
endif()

add_subdirectory(unilog)
add_subdirectory(target_factory)
add_subdirectory(trace_logging)
add_subdirectory(xir)
add_subdirectory(vart)

if(BUILD_RT_ENGINE)
  add_subdirectory(rt_engine)
endif()

if(BUILD_GRAPH_ENGINE)
  add_subdirectory(graph_engine)
endif()

if(BUILD_TVM_ENGINE)
  add_subdirectory(tvm_engine)
endif()

if(BUILD_SINGLE_LIBRARY)
  add_subdirectory(vairt)
endif()

if(BUILD_TESTCASES)
  add_subdirectory(testcases)
endif()
