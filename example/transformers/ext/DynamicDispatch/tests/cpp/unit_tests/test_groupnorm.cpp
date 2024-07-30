/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/lrn_matrix.hpp"
#include <ops/groupnorm/groupnorm.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace lrn_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_gpn(int M, int N, int GB, bool debug = false,
             const std::string &a_dtype = "int16",
             const std::string &b_dtype = "int16",
             const std::string &c_dtype = "int16",
             const std::string &model_name = "PSF") {

  int err_count = 0;

  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  size_t GBs = static_cast<size_t>(GB);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> gamma_shape = {GBs};
  std::vector<size_t> beta_shape = {GBs};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> a(M * N);
  std::vector<float> gamma(GB); // for CPU calculation
  std::vector<float> beta(GB);  // for CPU calculation
  std::vector<WgT> aie_gamma(GB);
  std::vector<WgT> aie_beta(GB);
  std::vector<OutT> cpu_out(M * N);
  std::vector<OutT> aie_out(M * N);
  std::vector<int32_t> qdq_params(QDQparam_size);

  std::vector<WgT> b(2 * GB);
  BiasVector<WgT, 1> bias(GB, b.data());
  ActMatrix<OutT, 1, 1> cpu_Y(M, N, cpu_out.data());

  int32_t is_input_uint16 = 0;

  if (a_dtype == "uint16") {
    is_input_uint16 = 1;
  }

  int32_t is_output_uint16 = 0;

  if (c_dtype == "uint16") {
    is_output_uint16 = 1;
  }
#ifdef RANDOM_DATA
  srand(0xABCD);
  if (is_input_uint16 == 1) {
    initialize_random<InT>(a, M * N, 40000, 100);
  } else {
    initialize_random_bfloat16(a, M * N, -20, 20);
  }
  initialize_random_bfloat16(b, 2 * GB, -1, 1);
  // init_random_bias(bias, -2, 2); // float to bfloat16

  for (int c = 0; c < GB; c++) {
    gamma[c] = (bfloat2float(bias.gamma(c)));
    beta[c] = (bfloat2float(bias.beta(c)));
    aie_gamma[c] = bias.gamma(c);
    aie_beta[c] = bias.beta(c);
  }

  // quantize output
  float sc_float = 0.1;
  int16_t sc_out = float2bfloat(1.0 / sc_float); // bfloat16
  OutT zp_out = 129;
  float sc_in = 0.03;
  InT zp_in = 128;

  qdq_params[0] = (int32_t)sc_out;
  qdq_params[1] = (int32_t)zp_out;
  qdq_params[2] = is_output_uint16; // for PSR, this is enable quant at output
  qdq_params[3] = float2bfloat(sc_in);
  qdq_params[4] = (is_input_uint16 == 0) ? 0 : zp_in;
  qdq_params[5] = is_input_uint16; // if 1, enalbe de-quant at input

  std::vector<std::vector<InT>> In(M);
  std::vector<std::vector<float>> Out(M);

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
  compute_gpn_bfloat16(In, gamma, beta, Out);

  if (c_dtype == "uint16") {
    q_bfloat2uint16(Out, float2bfloat(sc_float), zp_out, cpu_Y);
  } else { // bfloat
    int num_rows = Out.size();
    int num_cols = Out[0].size();
    for (int r = 0; r < num_rows; r++) {
      for (int c = 0; c < num_cols; c++) {
        cpu_Y.at(r, c) = float2bfloat(Out[r][c]);
      }
    }
  }
#else
  std::string data_folder = OpInterface::get_dod_base_dir() +
                            "//bin_files//PSR_groupnorm_" + std::to_string(M) +
                            "_" + std::to_string(N) + "//data//";

  // read_bin_file(data_folder + "input.bin",
  // reinterpret_cast<char*>(a.data())); read_bin_file(data_folder +
  //"gamma.const", reinterpret_cast<char*>(aie_gamma.data()));
  // read_bin_file(data_folder + "beta.const",
  // reinterpret_cast<char*>(aie_beta.data())); read_bin_file(data_folder +
  //"qdq.const", reinterpret_cast<char*>(qdq_params.data()));

  std::string ifm_filename = data_folder + "ifm_uint16.txt";
  std::string wgt_filename = data_folder + "weight_uint16.txt";
  std::string wgt_scale_filename = data_folder + "weight_scale_float32.txt";
  std::string wgt_zp_filename = data_folder + "weight_zp_uint16.txt";
  std::string bias_filename = data_folder + "bias_uint16.txt";
  std::string bias_scale_filename = data_folder + "bias_scale_float32.txt";
  std::string bias_zp_filename = data_folder + "bias_zp_uint16.txt";
  std::string ofm_filename = data_folder + "ofm_float32.txt";
  std::string sc_in_filename = data_folder + "sc_in_float32.txt";
  std::string zp_in_filename = data_folder + "zp_in_uint16.txt";

  std::vector<uint32_t> aint(M * N);
  read_data_file<uint32_t>(ifm_filename, (uint32_t *)aint.data());

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a[i * N + j] = (uint16_t)aint[j * M + i];
    }
  }

  std::vector<uint32_t> weight(GB);
  std::vector<uint32_t> bias_in(GB);
  uint16_t wgt_zp, bias_zp;
  float wgt_scale, bias_scale;
  read_data_file<float>(wgt_scale_filename, (float *)&wgt_scale);
  read_data_file<uint16_t>(wgt_zp_filename, (uint16_t *)&wgt_zp);
  read_data_file<uint32_t>(wgt_filename, (uint32_t *)weight.data());
  read_data_file<float>(bias_scale_filename, (float *)&bias_scale);
  read_data_file<uint16_t>(bias_zp_filename, (uint16_t *)&bias_zp);
  read_data_file<uint32_t>(bias_filename, (uint32_t *)bias_in.data());
  for (int i = 0; i < GB; i++) {
    float out_dq = ((uint16_t)weight[i] - wgt_zp) * wgt_scale;
    aie_gamma[i] = float2bfloat(out_dq);
    out_dq = ((uint16_t)bias_in[i] - bias_zp) * bias_scale;
    aie_beta[i] = float2bfloat(out_dq);
  }

  float sc_in, sc_float = 0;
  uint16_t zp_in, zp_out = 0;
  read_data_file<float>(sc_in_filename, (float *)&sc_in);
  read_data_file<uint16_t>(zp_in_filename, (uint16_t *)&zp_in);

  qdq_params[0] = 0;
  qdq_params[1] = 0;
  qdq_params[2] = is_output_uint16; // for PSR, this is enable quant at output
  qdq_params[3] = float2bfloat(sc_in);
  qdq_params[4] = (is_input_uint16 == 0) ? 0 : zp_in;
  qdq_params[5] = is_input_uint16; // if 1, enalbe de-quant at input

  if (is_input_uint16 == 0) { // a is bfloat
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float out_dq = ((uint16_t)aint[j * M + i] - zp_in) * sc_in;
        a[i * N + j] = float2bfloat(out_dq);
      }
    }
  }

  std::vector<float> golden(M * N);
  read_data_file<float>(ofm_filename, (float *)golden.data());

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      cpu_out[i * N + j] = float2bfloat(golden[j * M + i]);
    }
  }

  for (int c = 0; c < N; c++) {
    gamma[c] = (bfloat2float(aie_gamma[c]));
    beta[c] = (bfloat2float(aie_beta[c]));
  }
#endif
  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::groupnorm groupnorm_ = ryzenai::groupnorm<InT, WgT, OutT>(
      a_dtype, b_dtype, c_dtype, false, attr);

  groupnorm_.debug(debug);
  std::vector<size_t> perform_shape = {Ms, Ns, GBs};
  groupnorm_.set_params(model_name, perform_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{aie_gamma.data(), gamma_shape, b_dtype},
                  {aie_beta.data(), beta_shape, b_dtype},
                  {qdq_params.data(), qdq_params_shape, "int32"}};

  groupnorm_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(groupnorm_.execute(input_Tensor, output_Tensor));
#else
  groupnorm_.execute(input_Tensor, output_Tensor);
#endif

  // compare results
  float L2_norm = 0;
  if (is_output_uint16) {
    int max_error = 0;
    int error_limit = 40;
    for (int r = 0; r < M; r++) {
      for (int c = 0; c < N; c++) {
        int32_t diff = std::abs(aie_out[r * N + c] - cpu_Y.at(r, c));
        L2_norm += ((float)diff * (float)diff);
        if (diff > error_limit) {
          std::cout << "ERROR: Y[" << r << ", " << c << "]: "
                    << "Expected: " << (int)cpu_Y.at(r, c) << ", "
                    << "Received: " << (int)aie_out[r * N + c] << "\n";
          err_count++;
        }
        max_error = (diff > max_error) ? diff : max_error;
      }
    }
    if (max_error <= error_limit) {
      err_count = 0;
    }
    LOG_THIS("Maximum Difference : " << max_error);
  } else {
    float max_error = 0;
    float error_limit = 0.1;
    for (int r = 0; r < M; r++) {
      for (int c = 0; c < N; c++) {
        float diff = std::abs(bfloat2float(cpu_Y.at(r, c)) -
                              bfloat2float(aie_out[r * N + c]));
        L2_norm += (diff * diff);
        if (diff > error_limit || std::isnan(diff)) {
          std::cout << "ERROR: Y[" << r << ", " << c << "]: "
                    << "Expected: " << bfloat2float(cpu_Y.at(r, c)) << ","
                    << "Received: " << bfloat2float(aie_out[r * N + c]) << ","
                    << "Diff: " << diff << "\n";
          err_count++;
        }
        max_error = (diff > max_error) ? diff : max_error;
      }
    }
    if (max_error <= error_limit) {
      err_count = 0;
    }
    LOG_THIS("Maximum Difference : " << max_error);
  }

  L2_norm = sqrt(L2_norm);
  std::cout << "L2_norm is " << L2_norm << std::endl;
  return err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel1) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      64, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel2) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      64, 2560, 2560, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel3) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel4) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 2560, 2560, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel5) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 1920, 1920, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel6) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 640, 640, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel7) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      4096, 320, 320, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel8) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 1920, 1920, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel9) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 960, 960, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel10) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel11) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      4096, 640, 640, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel12) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      4096, 960, 960, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel13) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 320, 320, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GPN_Testabf16wbf16, Kernel14) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 640, 640, false, "bfloat16", "bfloat16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel1) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      64, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel2) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      64, 2560, 2560, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel3) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel4) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 2560, 2560, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel5) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 1920, 1920, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel6) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 640, 640, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel7) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      4096, 320, 320, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel8) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 1920, 1920, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel9) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 960, 960, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel10) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel11) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      4096, 640, 640, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel12) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      4096, 960, 960, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel13) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 320, 320, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel14) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 640, 640, false, "bfloat16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel15) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      64, 1280, 1280, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel16) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      64, 2560, 2560, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel17) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 1280, 1280, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel18) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 2560, 2560, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel19) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 1920, 1920, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel20) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 640, 640, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel21) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      4096, 320, 320, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel22) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 1920, 1920, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel23) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 960, 960, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel24) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 1280, 1280, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel25) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      4096, 640, 640, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel26) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      4096, 960, 960, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel27) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      1024, 320, 320, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GPN_Testabf16wbf16, Kernel28) {
  int err_count = test_gpn<int16_t, int16_t, uint16_t>(
      256, 640, 640, false, "uint16", "bfloat16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
