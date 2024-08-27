cmake_minimum_required(VERSION 3.5)
project(
  yolov8
  VERSION 1.0.0
  LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus")
#
# unreferenced formal parameter
string(APPEND CMAKE_CXX_FLAGS " /wd4100")
# std::codecvt_utf8<wchar_t,1114111,(std::codecvt_mode)0>': warning STL4017:
# std::wbuffer_convert, std::wstring_convert, and the <codecvt> header
# (containing std::codecvt_mode, std::codecvt_utf8, std::codecvt_utf16, and
# std::codecvt_utf8_utf16) are deprecated in C++17. (The std::codecvt class
# template is NOT deprecated.) The C++ Standard doesn't provide equivalent
# non-deprecated functionality; consider using MultiByteToWideChar() and
# WideCharToMultiByte() from <Windows.h> instead. You can define
# _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING or
# _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS to acknowledge that you have received
# this warning.
#
string(APPEND CMAKE_CXX_FLAGS " /wd4996")
string(APPEND CMAKE_CXX_FLAGS " /WX-")
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)

set (RYZEN_AI_INSTALLATION_PATH $ENV{RYZEN_AI_INSTALLATION_PATH})

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "${RYZEN_AI_INSTALLATION_PATH}/onnxruntime")
else()
    set(ONNXRUNTIME_ROOTDIR "/usr/local")
  endif()
endif()

include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"  # Windows local install
                    "${CMAKE_PREFIX_PATH}/include")

link_directories("${ONNXRUNTIME_ROOTDIR}/lib"
                 "${CMAKE_PREFIX_PATH}/lib")

find_library(ORT_LIBRARY onnxruntime HINTS "${ONNXRUNTIME_ROOTDIR}/lib"
                                           REQUIRED)

find_package(Eigen3)
find_package(glog REQUIRED)
find_package(
  OpenCV
  COMPONENTS core highgui imgproc
  REQUIRED)
  
add_subdirectory(yolov8)