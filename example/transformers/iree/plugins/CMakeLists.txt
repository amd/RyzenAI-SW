set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set(XRT_DIR $ENV{XRT_PATH})
find_package(XRT REQUIRED PATHS ${XRT_DIR})
find_package(spdlog REQUIRED)

set(OPS_ROOT ${TRANSFORMERS_ROOT}/ops)

set(TVM_ROOT $ENV{TVM_PATH})
set(DMLC_CORE "${TVM_ROOT}/3rdparty/dmlc-core")

add_definitions(-DXAIE_FEATURE_MSVC)

#==============================================================================#
# matmul
#==============================================================================#
add_library(iree_plugins_matmul matmul.cpp)

target_include_directories(
  iree_plugins_matmul
  PUBLIC ${OPS_ROOT}/cpp/qlinear
         ${OPS_ROOT}/cpp/dynamic_quantlinear
         ${OPS_ROOT}/cpp/linear
         ${OPS_ROOT}/cpp/utils
         ${OPS_ROOT}/common
         ${TVM_ROOT}/Deploy/include
         ${DMLC_CORE}/include
         ${TVM_ROOT}/Deploy/include/aie_controller/ps
         ${TVM_ROOT}/Deploy/include/aie_controller
         ${TRANSFORMERS_ROOT}/build/aie-rt/include
         ${XRT_INCLUDE_DIRS}
)

target_link_directories(
  iree_plugins_matmul PUBLIC ${TVM_ROOT}/Deploy/lib ${TVM_ROOT}/Deploy/bin
  ${TRANSFORMERS_ROOT}/build/aie-rt/Release
)

target_link_libraries(
  iree_plugins_matmul PUBLIC ${CMAKE_DL_LIBS} tvm_ipu_runtime XRT::xrt_coreutil
                             maize xaiengine spdlog::spdlog
)

add_gtest(iree_plugins_matmul_test matmul_test.cpp iree_plugins_matmul)

target_include_directories(
  iree_plugins_matmul_test
  PUBLIC ${OPS_ROOT}/cpp/qlinear
         ${OPS_ROOT}/cpp/dynamic_quantlinear
         ${OPS_ROOT}/cpp/linear
         ${OPS_ROOT}/cpp/utils
         ${OPS_ROOT}/common
         ${TVM_ROOT}/Deploy/include
         ${DMLC_CORE}/include
         ${TVM_ROOT}/Deploy/include/aie_controller/ps
         ${TVM_ROOT}/Deploy/include/aie_controller
         ${TRANSFORMERS_ROOT}/build/aie-rt/include
         ${XRT_INCLUDE_DIRS}
)

#==============================================================================#
# matmul_plugin
#==============================================================================#
add_library(iree_plugins_matmul_plugin SHARED system_plugin.c)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(
  iree_plugins_matmul_plugin PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON PREFIX ""
                                        OUTPUT_NAME "system_plugin"
)

target_link_libraries(iree_plugins_matmul_plugin PUBLIC iree_plugins_matmul)
