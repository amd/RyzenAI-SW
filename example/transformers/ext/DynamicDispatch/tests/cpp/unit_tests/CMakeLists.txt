# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

add_executable(
  cpp_tests
  experimental/test_cube.cpp
  experimental/test_square.cpp
  test_act_act_matmul.cpp
  test_bmm.cpp
  test_concat.cpp
  test_concateOps.cpp
  test_conv.cpp
  test_conv2matmul.cpp
  test_elwadd.cpp
  test_elwmul.cpp
  test_elwmul_qdq.cpp
  test_gap.cpp
  test_gelu.cpp
  test_graphMode.cpp
  test_groupnorm.cpp
  test_iconv.cpp
  test_is_supported.cpp
  test_layernorm.cpp
  test_lstm_wts.cpp
  test_maskedsoftmax.cpp
  test_matmul.cpp
  test_matmul_a16a16_mladf.cpp
  test_matmul_a16w8_mladf.cpp
  test_matmulbias.cpp
  test_matmulgeluadd.cpp
  test_matvecadd.cpp
  test_mha.cpp
  test_mhachannel.cpp
  test_mhagprb.cpp
  test_mhapsr.cpp
  test_mhawindow.cpp
  test_mladfadd.cpp
  test_mladfelwadd.cpp
  test_mladfelwmul.cpp
  test_mladfmatmulbias.cpp
  test_mladfmharope.cpp
  test_mladfrmsnorm.cpp
  test_mladfsoftmax.cpp
  test_nni_resize.cpp
  test_silu.cpp
  test_silu_qdq.cpp
  test_slice.cpp
  test_softmax_qdq.cpp
  test_transpose.cpp
  test_xcom_conv2d.cpp
)

dd_configure_test(cpp_tests ON)
if(UNIT_TEST_PERF_EN)
  message(STATUS "Enabling performance checks in unit tests")
  target_compile_definitions(cpp_tests PRIVATE UNIT_TEST_PERF)
endif()
