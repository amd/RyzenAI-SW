#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/mladfelwadd/mladfelwadd.hpp>

#include "mladfelwadd_helpers.hpp"
#include "test_common.hpp"
using namespace matmul_matrix;

// Template function to test the MLADFElwADD operation
template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_mladfelwadd(std::vector<std::vector<size_t>> inputs_shape,
                     bool debug = false, const std::string &a_dtype = "int16",
                     const std::string &b_dtype = "int16",
                     const std::string &c_dtype = "int16",
                     const std::string &model_name = "PST") {

  int err_count = 0;
  mladfelwadd_helpers::process_shape(inputs_shape.at(0));
  mladfelwadd_helpers::process_shape(inputs_shape.at(1));
  auto bd_type = mladfelwadd_helpers::determine_broadcast_type(inputs_shape);

  std::vector<size_t> a_shape = inputs_shape.at(0);
  std::vector<size_t> b_shape = inputs_shape.at(1);

  size_t M = a_shape[0];
  size_t K = std::accumulate(a_shape.begin() + 1, a_shape.end(), size_t{1},
                             std::multiplies{});

  size_t b_size = std::accumulate(b_shape.begin(), b_shape.end(), size_t{1},
                                  std::multiplies{});
  size_t qdq_size = 24;
  std::vector<size_t> qdq_params_shape = {qdq_size};

  std::vector<InT> a(M * K);
  std::vector<WgT> b(b_size);
  std::vector<OuT> cpu_out(M * K);

  std::vector<OuT> aie_out(M * K);
  std::vector<int8_t> qdq_params(qdq_size);
  OutMatrix<OuT, 1, 1> aie_Y(M, K, aie_out.data());
  OutMatrix<OuT, 1, 1> cpu_Q_Y(M, K, cpu_out.data());

  srand(0xABCD);

  initialize_random<InT>(a, M * K, 65535, 1);
  initialize_random<WgT>(b, b_size, 65535, 1);
  int32_t matA_zero_point = 41733;
  double matA_scale = 0.0008634580299258232;
  int32_t matB_zero_point = 19916;
  double matB_scale = 0.0001138541119871661;
  OuT matC_zero_point = 42933;
  double matC_scale = 0.0008906692964956164;

  // Compute qdq parameters
  auto results = mladfelwadd_helpers::compute_qdq(
      matA_scale, matB_scale, matC_scale, matA_zero_point, matB_zero_point,
      matC_zero_point);
  int ifm1_coeff, ifm2_coeff, zero_point_coeff;
  int8_t ofm_shift, ifm1_shift, ifm2_shift, zero_point_shift;
  std::tie(ifm1_coeff, ifm2_coeff, zero_point_coeff, ofm_shift, ifm1_shift,
           ifm2_shift, zero_point_shift) = results;

  uint32_t ifmsv_size = M * K;
  // Assign qdq_params
  mladfelwadd_helpers::assign_qdq_params(
      qdq_params, ifm1_coeff, ifm2_coeff, zero_point_coeff, ofm_shift,
      ifm1_shift, ifm2_shift, zero_point_shift, ifmsv_size, bd_type);
  // Compute CPU output
  mladfelwadd_helpers::compute_cpu_output(
      a, b, cpu_out, M, K, ifm1_coeff, ifm2_coeff, zero_point_coeff, ofm_shift,
      ifm1_shift, ifm2_shift, zero_point_shift, bd_type);

  ryzenai::ml_adf_elw_add mladfelwadd_ =
      ryzenai::ml_adf_elw_add<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false);

  std::vector<Tensor> const_Tensor;
  Tensor b_T = {b.data(), b_shape, b_dtype};
  Tensor qdq_tensor = {qdq_params.data(), qdq_params_shape, "int8_t"};
  const_Tensor.push_back(b_T);
  const_Tensor.push_back(qdq_tensor);

  std::vector<Tensor> input_tensor;
  Tensor a_T = {a.data(), a_shape, a_dtype};
  Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_tensor.push_back(a_T);

  std::vector<Tensor> output_tensor;
  output_tensor.push_back(c_T);
  mladfelwadd_.debug(debug);
  mladfelwadd_.set_params(model_name, a_shape);
  mladfelwadd_.initialize_const_params(const_Tensor);
#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(mladfelwadd_.execute(input_tensor, output_tensor));
#else
  mladfelwadd_.execute(input_tensor, output_tensor);
#endif
  err_count = check_add_result(cpu_Q_Y, aie_Y, 0.01);
  return err_count;
}

// // Test case for MLADFElwADD kernel
TEST(PST_MLADFELWADD_A16W16, Kernel1) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{1, 128, 256, 256}, {1, 128, 1, 1}}, false, "uint16", "uint16", "uint16",
      "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel2) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{128, 512, 512}, {128, 1, 1}}, false, "uint16", "uint16", "uint16",
      "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel3) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{256, 128, 128}, {256, 1, 1}}, false, "uint16", "uint16", "uint16",
      "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel4) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{256, 256, 256}, {256, 1, 1}}, false, "uint16", "uint16", "uint16",
      "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel5) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{256, 512, 512}, {256, 1, 1}}, false, "uint16", "uint16", "uint16",
      "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel6) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{512, 64, 64}, {512, 1, 1}}, false, "uint16", "uint16", "uint16", "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel7) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{512, 128, 128}, {512, 1, 1}}, false, "uint16", "uint16", "uint16",
      "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel8) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{512, 256, 256}, {512, 1, 1}}, false, "uint16", "uint16", "uint16",
      "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel9) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{512, 4096}, {512, 1}}, false, "uint16", "uint16", "uint16", "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel10) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{4096, 512}, {512}}, false, "uint16", "uint16", "uint16", "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel11) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{1, 128, 512, 512}, {1, 128, 512, 512}}, false, "uint16", "uint16",
      "uint16", "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel12) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{256, 256, 256}, {256, 256, 256}}, false, "uint16", "uint16", "uint16",
      "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel13) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{512, 128, 128}, {512, 128, 128}}, false, "uint16", "uint16", "uint16",
      "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_MLADFELWADD_A16W16, Kernel14) {
  int err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
      {{512, 64, 64}, {512, 64, 64}}, false, "uint16", "uint16", "uint16",
      "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
