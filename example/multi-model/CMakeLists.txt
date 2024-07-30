cmake_minimum_required(VERSION 3.5)
project(
  ipu_multi_models
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

if(NOT LIB_ROOTDIR)
  if(WIN32)
    set(LIB_ROOTDIR "C:/Program Files")
  else()
    set(LIB_ROOTDIR "/usr/local")
  endif()
endif()


include_directories(
  "${ONNXRUNTIME_ROOTDIR}/include"
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime" 
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"
  "${LIB_ROOTDIR}/gflag/include"
  "${LIB_ROOTDIR}/glog/include"
  "${LIB_ROOTDIR}/opencv/include"
  )
link_directories(
  "${LIB_ROOTDIR}/glog/lib"
  "${LIB_ROOTDIR}/gflags/lib"
  "${LIB_ROOTDIR}opencv/x64/vc17/staticlib"
)
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
find_package(glog CONFIG REQUIRED)
# set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED)
find_package(
  OpenCV
  COMPONENTS core highgui imgproc
  REQUIRED)

add_subdirectory(src)
