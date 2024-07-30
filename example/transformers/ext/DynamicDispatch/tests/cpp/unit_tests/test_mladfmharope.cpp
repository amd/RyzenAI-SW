/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <tuple>

#include <ops/mladfmharope/mladfmharope.hpp>

#include <stdexcept>

#include "enable_perf.hpp"

#include "test_common.hpp"

template <typename InT = uint16_t, typename TrigT = uint16_t,
          typename OuT = uint16_t>
int test_mladfmharope(size_t B, size_t M, size_t K, bool debug = false,
                      const std::string &a_dtype = "bfloat16",
                      const std::string &b_dtype = "bfloat16",
                      const std::string &c_dtype = "bfloat16",
                      const std::string &model_name = "LLAMA2") {
  int err_count = 0;

  std::vector<size_t> a_shape = {B, M, K};
  std::vector<size_t> trig_shape = {2, M, K};

  // simple test vector for functionality
  // ifm = all ones
  // trig = all ones
  std::vector<InT> a(B * M * K, dd::float_to_bfloat16(1.0f));
  std::vector<InT> trig(2 * M * K, dd::float_to_bfloat16(1.0f));
  //  ==> Rope = half zeros half two
  // TODO: I believe these should be interleaved but current kernel has
  // contigous K/2 0s and then 2s
  std::vector<float> cpu_float(B * M * K, 0.0f);
  for (int i = 0; i < cpu_float.size(); ++i) {
    if (i % K >= K / 2) {
      cpu_float.at(i) = 2.0f;
    }
  }

  // compute aie
  std::vector<OuT> aie_out(B * M * K, garbage_value);

  ryzenai::mha_rope mladfmharope_ =
      ryzenai::mha_rope<InT, TrigT, OuT>(a_dtype, true);

  std::vector<Tensor> const_Tensor;
  std::vector<Tensor> input_Tensor;

  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor trig_T = {trig.data(), trig_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(trig_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  mladfmharope_.debug(debug);
  mladfmharope_.initialize_const_params(const_Tensor);
#ifdef UNIT_TEST_PERF
  LOG_THIS("B = " << B << ", M = " << M << ", K = " << K);
  PROFILE_THIS(mladfmharope_.execute(input_Tensor, output_Tensor));
#else
  mladfmharope_.execute(input_Tensor, output_Tensor);
#endif

  err_count = dd::count_errors_floatvsbfloat16(cpu_float, aie_out, a_shape,
                                               mladfmharope_.EPSILON);

  return err_count;
}

// MLADFMHAROPE
TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x4096x128) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 4096, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x128x128) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
