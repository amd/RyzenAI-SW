# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

set(LIBRARY_NAME dyn_dispatch_core)

set(LIB_SRC
    ops/op_interface.cpp
    ops/op_builder.cpp
    ops/matmul/matmul.cpp
    ops/maxpool/maxpool.cpp
    ops/act_act_matmul_qdq/act_act_matmul_qdq.cpp
    ops/layernorm/layernorm.cpp
    ops/groupnorm/groupnorm.cpp
    ops/mhagprb/mhagprb.cpp
    ops/mhachannel/mhachannel.cpp
    ops/mhawindow/mhawindow.cpp
    ops/mha/mha.cpp
    ops/mhapsr/mhapsr.cpp
    ops/matmulgeluadd/matmulgeluadd.cpp
    ops/matmulbias/matmulbias.cpp
    ops/matmul_a16a16_mladf/matmul_a16a16_mladf.cpp
    ops/matmul_a16w8_mladf/matmul_a16w8_mladf.cpp
    ops/elwadd/elwadd.cpp
    ops/mladfadd/mladfadd.cpp
    ops/transpose/transpose.cpp
    ops/silu/silu.cpp
    ops/elwmul/elwmul.cpp
    ops/maskedsoftmax/maskedsoftmax.cpp
    ops/mladfmharope/mladfmharope.cpp
    ops/mladfrmsnorm/mladfrmsnorm.cpp
    ops/softmax_qdq/softmax_qdq.cpp
    ops/experimental/cube.cpp
    ops/experimental/square.cpp
    ops/pm_load/pm_load.cpp
    ops/record_timer/record_timer.cpp
    ops/mladfmatmulbias/mladfmatmulbias.cpp
    ops/bmm/bmm.cpp
    ops/mladfsoftmax/mladfsoftmax.cpp
    ops/mladfelwadd/mladfelwadd.cpp
    ops/mladfelwmul/mladfelwmul.cpp
    fusion_rt/fusion_rt.cpp
    fusion_rt/meta_utils.cpp
    passes/insert_pm_swap.cpp
    passes/insert_record_timer.cpp
    passes/assign_pdi_id_pass.cpp
    passes/generate_pdi_partitions_pass.cpp
    passes/analyze_buffer_reqs.cpp
    passes/optimize_scratch.cpp
    passes/split_max_partition_pass.cpp
    txn/txn_utils.cpp
    utils/xrt_context.cpp
    ops/conv/conv.cpp
    ops/concateOps/concateOps.cpp
    ops/gap/gap.cpp
    ops/iconv/iconv.cpp
    ops/conv2matmul/conv2matmul.cpp
    ops/gelu/gelu.cpp
    ops/silu_qdq/silu_qdq.cpp
    ops/matvecadd/matvecadd.cpp
    ops/elwmul_qdq/elwmul_qdq.cpp
    ops/nni_resize/nni_resize.cpp
    ops/concat/concat.cpp
    ops/slice/slice.cpp
    ops/xcom/conv/weight_shuffle.cpp
    ops/xcom/conv/conv.cpp
    txn_helper/txn_helper.cpp
    utils/utils.cpp
    ops/lstm/lstm.cpp
)

# Hook for simnow lite
set(HWEMU "")
if(ENABLE_SIMNOWLITE_BUILD)
  set(HWEMU ${XRT_HWEMU_LIBRARIES})
endif()

add_library(${LIBRARY_NAME} ${LIB_SRC})
target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  PUBLIC $<INSTALL_INTERFACE:include/ryzenai/dynamic_dispatch>
  PUBLIC $<INSTALL_INTERFACE:include>
  PRIVATE ${XRT_INCLUDE_DIRS}
  PRIVATE ${DD_SRC_INCLUDE_DIRS}
)
target_link_libraries(
  ${LIBRARY_NAME} PUBLIC nlohmann_json::nlohmann_json spdlog::spdlog
                         aie_controller xaiengine::xaiengine transaction
  PRIVATE ${XRT_COREUTIL_LIBRARIES} ${HWEMU}
)
target_compile_options(${LIBRARY_NAME} PRIVATE ${DD_DEFAULT_COMPILE_OPTIONS})
target_compile_definitions(${LIBRARY_NAME} PUBLIC XAIE_FEATURE_MSVC)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(
    ${LIBRARY_NAME} PUBLIC DYNAMIC_DISPATCH_BUILD_SHARED
  )
  target_compile_definitions(${LIBRARY_NAME} PRIVATE DYNAMIC_DISPATCH_EXPORT)
endif()

if(ENABLE_DD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
  )
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_DynamicDispatch "./python/bindings.cpp")
  target_include_directories(
    _DynamicDispatch PRIVATE ${XRT_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src
  )
  target_link_libraries(_DynamicDispatch PRIVATE ${LIBRARY_NAME})
  target_compile_options(_DynamicDispatch PRIVATE ${DD_DEFAULT_COMPILE_OPTIONS})
endif()

include(GNUInstallDirs)
# Install libraries
install(
  TARGETS ${LIBRARY_NAME}
  EXPORT ${PROJECT_NAME}-targets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ryzenai/dynamic_dispatch
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(ENABLE_DD_PYTHON AND SKBUILD)
  install(TARGETS _DynamicDispatch LIBRARY DESTINATION ".")
endif()

# Linux Can Be Fixed Later
if(MSVC)
  install(
    EXPORT ${PROJECT_NAME}-targets
    NAMESPACE ${PROJECT_NAME}::
    COMPONENT runtime
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  # Support Find Package
  include(CMakePackageConfigHelpers)

  configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/config.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake @ONLY
  )

  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion
  )

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    COMPONENT runtime DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ryzenai/dynamic_dispatch
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
  )
endif()
