cmake_minimum_required(VERSION 3.5)
project(
  npu_multi_models
  VERSION 1.0.0
  LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus")

# unreferenced formal parameter
string(APPEND CMAKE_CXX_FLAGS " /wd4100")
string(APPEND CMAKE_CXX_FLAGS " /EHsc")

string(APPEND CMAKE_CXX_FLAGS " /wd4996")
string(APPEND CMAKE_CXX_FLAGS " /WX")
string(APPEND CMAKE_CXX_FLAGS " /W0")
#string(APPEND CMAKE_CXX_FLAGS " /EHsc")


include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"  # Windows local install
                    "${CMAKE_PREFIX_PATH}/include" 
			  )

link_directories("${ONNXRUNTIME_ROOTDIR}/lib"
                 "${CMAKE_PREFIX_PATH}/lib")

find_library(ORT_LIBRARY onnxruntime HINTS "${ONNXRUNTIME_ROOTDIR}/lib"
                                           REQUIRED)

#find_package(glog QUIET)
#find_package(vart REQUIRED util runner)
#find_package(vaip REQUIRED)
find_package(OpenCV COMPONENTS core highgui imgproc)
find_package(Eigen3) # bug in opencv.cmake.
#if(NOT glog_FOUND)
#  message(STATUS "cannot find glogConfig.cmake fallback to pkg-config")
#  find_package(PkgConfig)
#  pkg_search_module(PKG_GLOG REQUIRED IMPORTED_TARGET GLOBAL libglog)
#  add_library(glog::glog ALIAS PkgConfig::PKG_GLOG)
#endif(NOT glog_FOUND)

add_subdirectory(src)