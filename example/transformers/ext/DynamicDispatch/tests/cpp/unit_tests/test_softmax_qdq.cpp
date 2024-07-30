/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/lrn_matrix.hpp"
#include "ops/ops_common/mhagprb_matrix.hpp"
#include <ops/softmax_qdq/softmax_qdq.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace lrn_matrix;
using namespace mhagprb_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_softmax_qdq(int M, int N, bool debug = false,
                     const std::string &a_dtype = "int16",
                     const std::string &b_dtype = "int16",
                     const std::string &c_dtype = "int16",
                     const std::string &model_name = "PSF") {

  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> gamma_shape = {Ns};
  std::vector<size_t> beta_shape = {Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
  std::vector<size_t> qdq_params_shape = {num_qdq_nodes,
                                          lrn_matrix::QDQparam_size};

  std::vector<InT> a(M * N);
  std::vector<float> gamma(N); // for CPU calculation
  std::vector<float> beta(N);  // for CPU calculation
  std::vector<WgT> aie_gamma(N);
  std::vector<WgT> aie_beta(N);
  std::vector<OutT> cpu_out(M * N);
  std::vector<OutT> aie_out(M * N);
  std::vector<OutT> model_out(M * N);
  std::vector<int32_t> qdq_params(lrn_matrix::QDQparam_size * num_qdq_nodes);

  // std::vector<WgT> b(2 * N);
  // BiasVector<WgT, 1> bias(N, b.data());
#ifdef RANDOM_DATA
  int32_t is_input_uint16 = 0;

  if (a_dtype == "uint16") {
    is_input_uint16 = 1;
  }

  srand(0xABCD);
  initialize_random<InT>(a, M * N, 40000, 100);

  // quantize output
  float sc_float = 0.1;
  int16_t sc_out = float2bfloat(1.0 / sc_float); // bfloat16
  OutT zp_out = 129;
  float sc_in = 0.03;
  InT zp_in = 128;

  qdq_params[0] = (int32_t)sc_out;
  qdq_params[1] = (int32_t)zp_out;
  // for PSH, user needs to set it based on Q datatype
  qdq_params[lrn_isint16_idx] = 1; // for PSR, this is enable quant at output
  qdq_params[3] = float2bfloat(sc_in);
  qdq_params[4] = zp_in;
  qdq_params[5] = is_input_uint16; // if 1, enalbe de-quant at input

  // DQ before SM
  qdq_params[(16 * 4) + 0] = 27389;
  qdq_params[(16 * 4) + 1] =
      float_to_bfloat16(0.00043027219362556934 * 1.442695041).value;

  // Q after SM
  qdq_params[(16 * 5) + 0] = 0;
  qdq_params[(16 * 5) + 1] = float_to_bfloat16(1.0 / 0.0000075458247).value;

  std::vector<std::vector<InT>> In(M);
  std::vector<std::vector<float>> Out(M);
  ActMatrix<OutT, 1, 1> cpu_Y(M, N, cpu_out.data());

  if (is_input_uint16 == 1) {
    std::vector<uint16_t> a_dq(M * N);
    dequant_to_bfloat(a, a_dq, zp_in, sc_in);
    for (int r = 0; r < M; r++) {
      for (int c = 0; c < N; c++) {
        In[r].push_back(a_dq[r * N + c]);
      }
    }
  } else {
    // initialize golden inputs
    for (int r = 0; r < M; r++) {
      for (int c = 0; c < N; c++) {
        In[r].push_back(a[r * N + c]);
      }
    }
  }

  // compute golden
  compute_lrn_bfloat16(In, gamma, beta, Out);

  if (c_dtype == "uint16") {
    q_bfloat2uint16(Out, float2bfloat(sc_float), zp_out, cpu_Y);
  } else {
    q_bfloat2uint8(Out, float2bfloat(sc_float), zp_out, cpu_Y);
  }

#else
  ActMatrix<OutT, 1, 1> model_Y(M, N, model_out.data());
  std::string fld_name;
  if (M == 256 && N == 256) {
    fld_name = "//SELF_SM_256//";
  } else if (M == 1024 && N == 1024) {
    fld_name = "//SELF_SM_1024//";
  } else if (M == 4096 && N == 4096) {
    fld_name = "//SELF_SM_4096//";
  }

  read_bin_file(OpInterface::get_dod_base_dir() + fld_name + "ifm.bin",
                reinterpret_cast<char *>(a.data()));

  read_bin_file(OpInterface::get_dod_base_dir() + fld_name + "ofm.bin",
                reinterpret_cast<char *>(model_out.data()));

  // MuL DQ - dummy, not used in kernel
  qdq_params[(16 * 0) + 0] = 0;
  qdq_params[(16 * 0) + 1] = float_to_bfloat16(1.0).value;
  // MuL Q - dummy, not used in kernel
  qdq_params[(16 * 1) + 0] = 0;
  qdq_params[(16 * 1) + 1] = float_to_bfloat16(1.0).value;

#if 0
  // DQ before SM
  qdq_params[(16 * 4) + 0] = 35625;
  qdq_params[(16 * 4) + 1] =
      float_to_bfloat16(0.0003217620251234621 * 1.442695041).value;

  // Q after SM
  qdq_params[(16 * 5) + 0] = 0;
  qdq_params[(16 * 5) + 1] = float_to_bfloat16(1.0 / 0.000009475293).value;
#endif

  read_bin_file(OpInterface::get_dod_base_dir() + fld_name + "wgt.bin",
                reinterpret_cast<char *>(qdq_params.data()));

#endif

  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::softmax_qdq softmax_qdq_ = ryzenai::softmax_qdq<InT, WgT, OutT>(
      a_dtype, b_dtype, c_dtype, false, attr);

  softmax_qdq_.debug(debug);
  softmax_qdq_.set_params(model_name, a_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int32"}};

  softmax_qdq_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(softmax_qdq_.execute(input_Tensor, output_Tensor));
#else
  softmax_qdq_.execute(input_Tensor, output_Tensor);
#endif
  // #ifndef RANDOM_DATA
  //   std::string ref_bin_name =
  //       OpInterface::get_dod_base_dir() + fld_name + "//golden.bin";
  //   read_bin_file(ref_bin_name, reinterpret_cast<char *>(cpu_out.data()));
  //
  //   for (int r = 196 * N; r < N * M; r++) {
  //     aie_out[r] = 0;
  //     cpu_out[r] = 0;
  //   }
  // #endif
  //  compare results
  double const PERCENT_ERR_TOLERANCE = 1;
  uint16_t const ABSOLUTE_ERR_TOLERANCE = 20;
  uint16_t max_error = 0;
  int mismatch_count = 0;
#if 1
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {

#ifdef RANDOM_DATA
      uint16_t err = std::abs(aie_out[r * N + c] - cpu_Y.at(r, c));
      double err_percentage = std::abs(aie_out[r * N + c] - cpu_Y.at(r, c)) /
                              (double)std::abs(cpu_Y.at(r, c)) * 100;
#else
      uint16_t err = std::abs(aie_out[r * N + c] - model_Y.at(r, c));
      double err_percentage = std::abs(aie_out[r * N + c] - model_Y.at(r, c)) /
                              (double)std::abs(model_Y.at(r, c)) * 100;
#endif
      if (err > max_error)
        max_error = err;

      if ((err_percentage > PERCENT_ERR_TOLERANCE) &&
          (err > ABSOLUTE_ERR_TOLERANCE)) {
        // printf("ERROR: ofm[%d]: Expected out: %d, Generated out: %d, Error
        // Percentage : %f, Absolute Error: %d\n", i, *(gold_ptr + i), *(ptr +
        // i), err_percentage, err);
        mismatch_count++;
      } else {
        // printf("PASS: ofm[%d]: Expected out: %d, Generated out: %d, Error
        // Percentage : %f, Absolute Error: %d\n", i, *(gold_ptr + i), *(ptr +
        // i), err_percentage, err);
      }
    }
  }
#endif
  std::cout << "TEST EXECUTION COMPLETED!" << std::endl;
  std::cout << "Mismatch count = " << mismatch_count << std::endl;
  std::cout << "Maximum error = " << max_error << std::endl;

  return err_count;
}

TEST(PSR_Softmax_Testa16w8, Kernel1) {
  int err_count = test_softmax_qdq<int16_t, int16_t, uint16_t>(
      256, 256, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_Softmax_Testa16w8, Kernel2) {
  int err_count = test_softmax_qdq<int16_t, int16_t, uint16_t>(
      1024, 1024, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_Softmax_Testa16w8, Kernel3) {
  int err_count = test_softmax_qdq<int16_t, int16_t, uint16_t>(
      4096, 4096, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_Softmax_Testa16w8, Kernel1) {
  int err_count = test_softmax_qdq<int16_t, int16_t, uint16_t>(
      256, 256, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_Softmax_Testa16w8, Kernel2) {
  int err_count = test_softmax_qdq<int16_t, int16_t, uint16_t>(
      1024, 1024, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_Softmax_Testa16w8, Kernel3) {
  int err_count = test_softmax_qdq<int16_t, int16_t, uint16_t>(
      4096, 4096, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
