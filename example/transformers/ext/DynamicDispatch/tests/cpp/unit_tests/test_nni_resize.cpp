/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "ops/ops_common/nni_resize_matrix.hpp"
#include <ops/nni_resize/nni_resize.hpp>

#include "enable_perf.hpp"

#include "test_common.hpp"

using namespace nni_resize_matrix;

template <typename InT = int8_t, typename OuT = int16_t>
int test_nni_resize(size_t H, size_t W, size_t C, bool debug = false,
                    const std::string &a_dtype = "int16",
                    const std::string &c_dtype = "int32",
                    const std::string &model_name = "PSF") {

  int err_count = 0;

  size_t Ho = H * 2;
  size_t Wo = W * 2;
  size_t Co = C;

  std::vector<size_t> a_shape = {H, W, C};
  std::vector<size_t> c_shape = {Ho, Wo, Co};

  std::vector<InT> a(H * W * C);
  std::vector<OuT> cpu_out(Ho * Wo * Co);
  std::vector<OuT> aie_out(Ho * Wo * Co);

  TensorMatrix<InT> ifm(H, W, C, a.data());
  TensorMatrix<OuT> expected_ofm(Ho, Wo, Co, cpu_out.data());
  TensorMatrix<OuT> aie_Y(Ho, Wo, Co, aie_out.data());

  initialize_random<InT>(a, H * W * C, 128, 0);

  // compute golden
  cpu_nni(ifm, expected_ofm, 2);
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::nni_resize nni_resize_ =
      ryzenai::nni_resize<InT, OuT>(a_dtype, c_dtype, true, attr);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), c_shape, c_dtype}};

  nni_resize_.debug(debug);

#ifdef UNIT_TEST_PERF
  PROFILE_THIS(nni_resize_.execute(input_Tensor, output_Tensor));
#else
  nni_resize_.execute(input_Tensor, output_Tensor);
#endif

  err_count = check_result(expected_ofm, aie_Y);

  return err_count;
}

// NNI 4x4
TEST(C4PSR_NNI_RESIZE_a16acc16, Kernel1) {
  int err_count = test_nni_resize<uint16_t, uint16_t>(
      32, 32, 640, false, "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_NNI_RESIZE_a16acc16, Kernel2) {
  int err_count = test_nni_resize<uint16_t, uint16_t>(
      8, 8, 1280, false, "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_NNI_RESIZE_a16acc16, Kernel3) {
  int err_count = test_nni_resize<uint16_t, uint16_t>(
      16, 16, 1280, false, "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
