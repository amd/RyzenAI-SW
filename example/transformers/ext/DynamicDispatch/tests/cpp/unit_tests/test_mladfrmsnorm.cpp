/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <tuple>

#include "mladfsoftmax_helpers.hpp"
#include <ops/mladfrmsnorm/mladfrmsnorm.hpp>

#include <stdexcept>

#include "enable_perf.hpp"

#include "test_common.hpp"

template <typename InT = uint16_t, typename WtsT = uint16_t,
          typename OuT = uint16_t>
int test_mladfrmsnorm(size_t M, size_t K, bool debug = false,
                      const std::string &a_dtype = "bfloat16",
                      const std::string &b_dtype = "bfloat16",
                      const std::string &c_dtype = "bfloat16",
                      const std::string &model_name = "LLAMA2") {
  int err_count = 0;

  std::vector<size_t> a_shape = {M, K};
  std::vector<size_t> wts_shape = {K};

  std::vector<InT> a(M * K);
  std::vector<InT> wts(K);

  // compute aie
  std::vector<OuT> aie_out(M * K, garbage_value);
  std::vector<OuT> reference_out(M * K);

  // Using golden teste vectors from MLLIB
  // https://gitenterprise.xilinx.com/AIELibs/mllib/tree/716e81ac7bf6fd135c86d54eb51435c6a1a3f403/internal/examples/rmsnorm_2x4x4/data
  std::string data_path_prefix = OpInterface::get_dod_base_dir() + "\\" +
                                 "tests" + "\\" + "cpp" + "\\" + "unit_tests" +
                                 "\\" + "testDataMladf" + "\\" +
                                 "llama2_2x4x4_mladfrmsnorm_2048_4096" + "\\";
  std::string a_bin_path = data_path_prefix + "ifm32.bin";
  std::string wts_bin_path = data_path_prefix + "wts32.bin";
  std::string ofm_bin_path = data_path_prefix + "ofm32.bin";

  mladfsoftmax_helpers::read_bin_to_vector(a_bin_path, a);
  mladfsoftmax_helpers::read_bin_to_vector(wts_bin_path, wts);
  mladfsoftmax_helpers::read_bin_to_vector(ofm_bin_path, reference_out);

  ryzenai::rms_norm mladfrmsnorm_ =
      ryzenai::rms_norm<InT, WtsT, OuT>(a_dtype, true);

  std::vector<Tensor> const_Tensor;
  std::vector<Tensor> input_Tensor;

  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor wts_T = {wts.data(), wts_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(wts_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  mladfrmsnorm_.debug(debug);
  mladfrmsnorm_.initialize_const_params(const_Tensor);
#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(mladfrmsnorm_.execute(input_Tensor, output_Tensor));
#else
  mladfrmsnorm_.execute(input_Tensor, output_Tensor);
#endif

  err_count = dd::count_errors_bfloat16vsbfloat16(
      reference_out, aie_out, a_shape, mladfrmsnorm_.EPSILON);

  return err_count;
}

// mladfrmsnorm
TEST(LLAMA2_MLADFRMSNORM_Testa16, Kernel2048x4096) {
  int err_count = test_mladfrmsnorm<uint16_t, uint16_t, uint16_t>(
      2048, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
