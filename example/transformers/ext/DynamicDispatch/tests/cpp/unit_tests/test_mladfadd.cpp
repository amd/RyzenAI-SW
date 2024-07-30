/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <tuple>

#include "../src/ops/ops_common/matmul_matrix.hpp"
#include <ops/mladfadd/mladfadd.hpp>
#include <stdexcept>

#include "enable_perf.hpp"

#include "test_common.hpp"

using namespace matmul_matrix;
template <typename LhsT = int16_t, typename RhsT = int16_t,
          typename OuT = int16_t>
int test_mladfadd(size_t M, size_t K, bool debug = false,
                  const std::string &a_dtype = "bfloat16",
                  const std::string &b_dtype = "bfloat16",
                  const std::string &c_dtype = "bfloat16",
                  const std::string &model_name = "LLAMA2") {
  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);

  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ms, Ks};

  std::vector<LhsT> a(M * K);
  std::vector<LhsT> b(M * K);
  std::vector<float> cpu_out(M * K);
  std::vector<OuT> aie_out(M * K, garbage_value);

  dd::initialize_random_bfloat16(a, 40);
  dd::initialize_random_bfloat16(b, 40);

  // compute golden
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < K; c++) {
      cpu_out.at(r * K + c) = bfloat16_to_float(a.at(r * K + c)) +
                              bfloat16_to_float(b.at(r * K + c));
    }
  }
  ryzenai::mladf_add mladfadd_ =
      ryzenai::mladf_add<LhsT, RhsT, OuT>(a_dtype, true);

  std::vector<Tensor> const_Tensor;
  std::vector<Tensor> input_Tensor;

  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor b_T = {b.data(), a_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(b_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  mladfadd_.debug(debug);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(mladfadd_.execute(input_Tensor, output_Tensor));
#else
  mladfadd_.execute(input_Tensor, output_Tensor);
#endif

  err_count = dd::count_errors_floatvsbfloat16(cpu_out, aie_out, a_shape, 4);

  return err_count;
}

// Add
TEST(LLAMA2_MLADFADD_Testa16, Kernel4096x4096) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      4096, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
