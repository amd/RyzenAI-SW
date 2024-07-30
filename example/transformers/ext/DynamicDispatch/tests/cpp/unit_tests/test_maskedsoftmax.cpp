/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <tuple>

#include <ops/maskedsoftmax/maskedsoftmax.hpp>

#include <stdexcept>

#include "enable_perf.hpp"

#include "maskedsoftmax_helpers.hpp"
#include "test_common.hpp"

template <typename InT = uint16_t, typename MaskT = uint16_t,
          typename OuT = uint16_t>
int test_maskedsoftmax(size_t B, size_t M, size_t K, bool debug = false,
                       const std::string &a_dtype = "bfloat16",
                       const std::string &b_dtype = "bfloat16",
                       const std::string &c_dtype = "bfloat16",
                       const std::string &model_name = "LLAMA2") {
  int err_count = 0;

  std::vector<size_t> a_shape = {B, M, K};
  std::vector<size_t> mask_shape = {1, M, K};

  std::vector<InT> a(B * M * K);
  // Range taken from
  // https://gitenterprise.xilinx.com/AIELibs/mllib/blob/dev/internal/models/python/restructured/operators/Transformers/SoftMax.py#L348
  dd::initialize_random_bfloat16(a, 5);

  std::vector<InT> mask(
      M * K, dd::float_to_bfloat16(-std::numeric_limits<float>::infinity()));
  // zero out lower triangluar to use a casual mask
  dd::initialize_lowertriangular(mask, M, K, dd::float_to_bfloat16(0.0));

  // compute golden
  std::vector<float> cpu_float = maskedsoftmax_helpers::golden_maskedsoftmax(
      {B, M, K}, a, mask,
      ryzenai::masked_softmax<uint16_t, uint16_t,
                              uint16_t>::DEFAULT_PREMASK_SCALE);

  // compute aie
  std::vector<OuT> aie_out(B * M * K,
                           /*garbage_value*/ dd::float_to_bfloat16(1.0));

  ryzenai::masked_softmax maskedsoftmax_ =
      ryzenai::masked_softmax<InT, MaskT, OuT>(a_dtype, true);

  std::vector<Tensor> const_Tensor;
  std::vector<Tensor> input_Tensor;

  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor mask_T = {mask.data(), mask_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(mask_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  maskedsoftmax_.debug(debug);
  maskedsoftmax_.initialize_const_params(const_Tensor);
#ifdef UNIT_TEST_PERF
  LOG_THIS("B = " << B << ", M = " << M << ", K = " << K);
  PROFILE_THIS(maskedsoftmax_.execute(input_Tensor, output_Tensor));
#else
  maskedsoftmax_.execute(input_Tensor, output_Tensor);
#endif

  err_count = dd::count_errors_floatvsbfloat16(cpu_float, aie_out, a_shape,
                                               maskedsoftmax_.EPSILON);

  return err_count;
}

// MASKEDSOFTMAX
TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x2048x2048) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 2048, 2048, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
