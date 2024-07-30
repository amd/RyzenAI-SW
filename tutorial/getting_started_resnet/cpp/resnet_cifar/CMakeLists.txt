cmake_minimum_required(VERSION 3.5)
project(
  resnet_cifar
  VERSION 1.0.0
  LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python 3.10 EXACT COMPONENTS Interpreter Development)
set (RYZEN_AI_INSTALLATION_PATH $ENV{RYZEN_AI_INSTALLATION_PATH})


string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus")
#
# unreferenced formal parameter
string(APPEND CMAKE_CXX_FLAGS " /wd4100")
# std::codecvt_utf8<wchar_t,1114111,(std::codecvt_mode)0>': warning STL4017: std::wbuffer_convert, std::wstring_convert, and the <codecvt> header (containing std::codecvt_mode, std::codecvt_utf8, std::codecvt_utf16, and std::codecvt_utf8_utf16) are deprecated in C++17. (The std::codecvt class template is NOT deprecated.) The C++ Standard doesn't provide equivalent non-deprecated functionality; consider using MultiByteToWideChar() and WideCharToMultiByte() from <Windows.h> instead. You can define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING or _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS to acknowledge that you have received this warning.
# 
string(APPEND CMAKE_CXX_FLAGS " /wd4996")
string(APPEND CMAKE_CXX_FLAGS " /WX")

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "${RYZEN_AI_INSTALLATION_PATH}/onnxruntime")
else()
    set(ONNXRUNTIME_ROOTDIR "/usr/local")
  endif()
endif()

# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
# TODO: If we want to support additional EPs being loadable from a local install we also need to add EP specific
# directories under /include/onnxruntime/core/providers
include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"  # Windows local install
			  )

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

link_directories("${CMAKE_INSTALL_PREFIX}/lib")



find_package(OpenCV COMPONENTS core highgui imgproc REQUIRED)
#find_package(Eigen3) # bug in opencv.cmake.


add_executable(resnet_cifar resnet_cifar.cpp util/getopt.c)
target_include_directories(resnet_cifar 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
target_link_libraries(resnet_cifar ${ORT_LIBRARY} ${OpenCV_LIBS} onnxruntime)
install(TARGETS resnet_cifar RUNTIME DESTINATION bin)