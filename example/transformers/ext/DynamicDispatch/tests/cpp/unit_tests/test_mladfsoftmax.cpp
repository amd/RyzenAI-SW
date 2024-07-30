/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */
#include "ops/ops_common/matmul_matrix.hpp"
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <ops/mladfsoftmax/mladfsoftmax.hpp>
#include <tuple>

#include <stdexcept>

#include "enable_perf.hpp"
#include "mladfsoftmax_helpers.hpp"
#include "test_common.hpp"
using namespace matmul_matrix;

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_mladfsoftmax(const std::vector<size_t> shape, bool debug = false,
                      const std::string &a_dtype = "uint16",
                      const std::string &b_dtype = "uint8",
                      const std::string &c_dtype = "uint16",
                      const std::string &model_name = "PSS") {
  if (2 != shape.size()) {
    throw std::invalid_argument("Now only support two dimension!");
  }
  int err_count = 0;
  // hardcoded for PSS/PST
  size_t single_const_size = 4096 + 64;
  size_t M = shape[0];
  size_t K = shape[0];
  std::vector<size_t> a_shape = shape;
  std::vector<size_t> b_shape = {single_const_size};

  std::vector<InT> a(M * K);
  std::vector<WgT> b(single_const_size * 2);
  std::vector<OuT> cpu_out(M * K);
  std::vector<OuT> aie_out(M * K);

  OutMatrix<OuT, 1, 1> cpu_Q_Y(M, K, cpu_out.data());
  OutMatrix<OuT, 1, 1> aie_Y(M, K, aie_out.data());

  // read ifm, ofm, lut and param from files
  std::string data_path_prefix = "tests/cpp/unit_tests/testDataMladf/";
  // std::string model_part = "pss_softmax_4096_4096/";
  std::string model_part = "pst_softmax_4096_4096/";
  std::string a_bin_path = data_path_prefix + model_part + "ifm.bin";
  std::string ofm_bin_path = data_path_prefix + model_part + "ofm.bin";

  mladfsoftmax_helpers::read_bin_to_vector(a_bin_path, a);
  mladfsoftmax_helpers::read_bin_to_vector(ofm_bin_path, cpu_out);
  ryzenai::mladf_softmax mladf_softmax_ =
      ryzenai::mladf_softmax<InT, WgT, OuT>(a_dtype, false);
  // Here use the lut for pst. For PSS model, need to change to pss_lut_array
  // and pss_param_array
  std::vector<Tensor> const_Tensor;

  const static uint16_t ifm_zp[1] = {32696};
  const static float ifm_scale[1] = {0.0006733822519890964};
  const static uint16_t ofm_zp[1] = {0};
  const static float ofm_scale[1] = {0.000015222542060655542};

  const std::string zp_type = "uint16";
  const std::string scale_type = "float";
  Tensor ifm_zp_tensor = {(void *)ifm_zp, {1}, zp_type};
  Tensor ifm_scale_tensor = {(void *)ifm_scale, {1}, scale_type};
  Tensor ofm_zp_tensor = {(void *)ofm_zp, {1}, zp_type};
  Tensor ofm_scale_tensor = {(void *)ofm_scale, {1}, scale_type};

  const size_t rtp_byte_size = 64;
  std::vector<uint8_t> rtp_vec(rtp_byte_size, 0);
  rtp_vec[62] = 131;
  rtp_vec[63] = 199;
  Tensor rtp_tensor = {(void *)rtp_vec.data(), {rtp_byte_size}, "uint8"};

  const_Tensor.push_back(ifm_scale_tensor);
  const_Tensor.push_back(ifm_zp_tensor);
  const_Tensor.push_back(ofm_scale_tensor);
  const_Tensor.push_back(ofm_zp_tensor);
  const_Tensor.push_back(rtp_tensor);

  std::vector<Tensor> input_Tensor;
  std::vector<Tensor> output_Tensor;

  Tensor a_T = {a.data(), a_shape, a_dtype};
  Tensor c_T = {aie_out.data(), a_shape, c_dtype};

  input_Tensor.push_back(a_T);
  output_Tensor.push_back(c_T);

  mladf_softmax_.debug(debug);
  mladf_softmax_.set_params(model_name, a_shape);
  mladf_softmax_.initialize_const_params(const_Tensor);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(mladf_softmax_.execute(input_Tensor, output_Tensor));
#else
  mladf_softmax_.execute(input_Tensor, output_Tensor);
#endif
  err_count = check_result(cpu_Q_Y, aie_Y);
  return err_count;
}

TEST(PSS_MLADFSOFTMAX, Kernel4096x4096) {
  int err_count = test_mladfsoftmax<uint16_t, uint8_t, uint16_t>(
      {4096, 4096}, false, "uint16", "uint8", "uint16", "PSS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
