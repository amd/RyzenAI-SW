# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.18)
project(_ryzenai_torch_cpp)

cmake_path(SET ENV_PYTORCH_AIE_PATH $ENV{PYTORCH_AIE_PATH})
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/lib/site-packages/torch")
list(APPEND CMAKE_MODULE_PATH ${ENV_PYTORCH_AIE_PATH}/cmake)

find_package(xaiengine REQUIRED)
find_package(aie_controller REQUIRED)

set(OPS_ROOT ${ENV_PYTORCH_AIE_PATH}/ops)

find_package(Threads REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

set(XRT_DIR $ENV{XRT_PATH})
find_package(XRT REQUIRED PATHS ${XRT_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set(CMAKE_CXX_FLAGS "-march=native")

pybind11_add_module(
  ${PROJECT_NAME}
  "pybind_bindings.cpp"
  "src/linear.cpp"
  "src/mha.cpp"
  "src/scalar_mult.cpp"
  "src/silu_torch.cpp"
  "src/elemw_add_torch.cpp"
  "src/elemw_mul_torch.cpp"
  "src/bmm_torch.cpp"
  "src/softmax_torch.cpp"
  "src/mha_npu_torch.cpp"
  "src/mlp_npu_torch.cpp"
  "src/rope_torch.cpp"
  "src/rmsnorm_torch.cpp"
  "src/gemm_torch.cpp"
)

if(DEFINED RYZENAI_PERF)
  add_definitions(-DRYZENAI_PERF)
endif()

# -DHAVE_SNPRINTF needed because of : https://github.com/nlohmann/json/issues/1408#issuecomment-450559270
target_compile_definitions(
  ${PROJECT_NAME} PRIVATE -DXAIE_FEATURE_MSVC -DHAVE_SNPRINTF -DFMT_HEADER_ONLY
)
find_package(DynamicDispatch REQUIRED)

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE ${OPS_ROOT}/cpp/linear
          ${OPS_ROOT}/cpp/qlinear
          ${OPS_ROOT}/cpp/qlinear_2
          ${OPS_ROOT}/cpp/utils
          ${OPS_ROOT}/cpp/silu
          ${OPS_ROOT}/cpp/mul
          ${XRT_INCLUDE_DIRS}
          ${DYNAMIC_DISPATCH_INCLUDE_DIRS}
)

target_link_directories(
  ${PROJECT_NAME} PRIVATE "$ENV{CONDA_PREFIX}/lib/site-packages/torch/lib"
)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE xaiengine::xaiengine
          spdlog::spdlog
          ${TORCH_LIBRARIES}
          torch_python
          XRT::xrt_coreutil
          aie_controller
          DynamicDispatch::dyn_dispatch_core
)

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE "/Zc:__cplusplus" "/W4")
endif()
