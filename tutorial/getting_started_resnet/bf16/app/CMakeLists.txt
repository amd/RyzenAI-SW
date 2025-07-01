cmake_minimum_required(VERSION 3.10)

project(app VERSION 1.0.0 LANGUAGES C CXX)

set(RYZEN_AI_INSTALLATION_PATH $ENV{RYZEN_AI_INSTALLATION_PATH})
set(ONNXRUNTIME_ROOTDIR "${RYZEN_AI_INSTALLATION_PATH}/onnxruntime")
set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")

message(STATUS "RYZEN_AI_INSTALLATION_PATH = ${RYZEN_AI_INSTALLATION_PATH}")
message(STATUS "ONNXRUNTIME_ROOTDIR        = ${ONNXRUNTIME_ROOTDIR}")
message(STATUS "CONDA_PREFIX               = ${CONDA_PREFIX}")

set(CMAKE_CONFIGURATION_TYPES Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_PREFIX .)
set(CMAKE_PREFIX_PATH .)
set(BUILD_SHARED_LIBS OFF)

add_compile_options(
  /Zc:__cplusplus
  /wd4100
  /wd4996
  /WX
  /std:c++latest
)

include_directories(
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"
)

add_compile_options(
    /Qspectre
    /ZH:SHA_256
    /guard:cf
    /W3
    /Zi
    /Zc:__cplusplus
)
add_link_options(
    /CETCOMPAT
    /DEBUG
)

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

add_executable(${PROJECT_NAME} main.cpp npu_util.cpp)

target_link_libraries(${PROJECT_NAME} onnxruntime)

# -- Copy the RyzenAI runtime DLLs in folder containing the executable

list(APPEND dll_list "${RYZEN_AI_INSTALLATION_PATH}/deployment/onnxruntime.dll")
list(APPEND dll_list "${RYZEN_AI_INSTALLATION_PATH}/deployment/onnxruntime_providers_shared.dll")
list(APPEND dll_list "${RYZEN_AI_INSTALLATION_PATH}/deployment/onnxruntime_providers_vitisai.dll")
list(APPEND dll_list "${RYZEN_AI_INSTALLATION_PATH}/deployment/onnxruntime_vitisai_ep.dll")
list(APPEND dll_list "${RYZEN_AI_INSTALLATION_PATH}/deployment/dyn_dispatch_core.dll")
list(APPEND dll_list "${RYZEN_AI_INSTALLATION_PATH}/deployment/flexmlrt.dll")
list(APPEND dll_list "${RYZEN_AI_INSTALLATION_PATH}/deployment/libutf8_validity.dll")
list(APPEND dll_list "${RYZEN_AI_INSTALLATION_PATH}/deployment/ryzenai_onnx_utils.dll")
list(APPEND dll_list "${RYZEN_AI_INSTALLATION_PATH}/deployment/zlib.dll")
list(APPEND dll_list "${RYZEN_AI_INSTALLATION_PATH}/deployment/xaiengine.dll")

foreach(DLL_FILE ${dll_list})
  add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLL_FILE} $<TARGET_FILE_DIR:${PROJECT_NAME}>
  )
endforeach()