cmake_minimum_required(VERSION 3.5)

project(resnet50 VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python 3.10 EXACT COMPONENTS Interpreter Development)

set (RYZEN_AI_INSTALLATION_PATH $ENV{RYZEN_AI_INSTALLATION_PATH})


set(ONNXRUNTIME_ROOTDIR "${RYZEN_AI_INSTALLATION_PATH}/onnxruntime")

include_directories("${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

link_directories("${CMAKE_INSTALL_PREFIX}/lib")



find_package(OpenCV COMPONENTS core highgui imgproc REQUIRED)

add_executable(resnet50 resnet50.cpp util/getopt.c)
target_include_directories(resnet50 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/util
)
target_link_libraries(resnet50 ${ORT_LIBRARY} ${OpenCV_LIBS} onnxruntime)
install(TARGETS resnet50 RUNTIME DESTINATION bin)

add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
  ${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/bin/DirectML.dll 
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>/DirectML.dll)

add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
  ${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/bin/onnxruntime.dll 
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>/onnxruntime.dll)