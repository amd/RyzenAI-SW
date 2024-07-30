/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "matrix_formatting.h"
#include <bmm.hpp>

template <typename T>
static std::vector<T> load_bin(const std::string &filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::invalid_argument("Couldn't open file : " + filename);
  }

  std::istreambuf_iterator<char> begin_it{ifs};
  std::istreambuf_iterator<char> end_it;
  std::vector<char> data(begin_it, end_it);
  std::vector<T> data_final;
  data_final.resize(data.size() / sizeof(T));
  std::memcpy(data_final.data(), data.data(), data.size());
  return data_final;
}

template <typename InOutT = uint16_t>
int test_bmm_realdata(int B, int M, int K, int N, bool debug = false,
                      const std::string &a_dtype = "bfloat16",
                      const std::string &b_dtype = "bfloat16",
                      const std::string &c_dtype = "bfloat16") {

  auto NUM_EXECUTE_ITERATIONS_ =
      Utils::get_env_var("NUM_EXECUTE_ITERATIONS", "1");

  std::tuple<int, int> a_shape = {B * M, K};
  std::tuple<int, int> b_shape = {B * N, K};

  bool is_bmm2 = (K == 2048);
  std::string a_golden = "query_states.bin";
  std::string b_golden = "key_states.bin";
  if (is_bmm2) {
    a_golden = "attn_weights.bin";
    b_golden = "value_states.bin";
  }
  std::string bmm_type = is_bmm2 ? "bmm2" : "bmm1";
  std::string gold_path = "./dump/" + bmm_type + "/" + to_string(M) + "x" +
                          to_string(K) + "x" + to_string(N) + "/";

  auto a = load_bin<InOutT>(gold_path + a_golden);
  auto b = load_bin<InOutT>(gold_path + b_golden);
  auto c_gold = load_bin<InOutT>(gold_path + "c.bin");

  int32_t garbage_value = 0xCDCDCDCD;
  std::vector<InOutT> c(B * M * N, garbage_value);

  auto bmm_runner = ryzenai::bmm<InOutT>(a_dtype, b_dtype, c_dtype);
  bmm_runner.debug(debug);

  for (int i = 0; i < stoi(NUM_EXECUTE_ITERATIONS_); i++)
    bmm_runner.execute(a.data(), b.data(), c.data(), a_shape, b_shape);

  float const EPSILON = 1.0;
  int err_count = 0;
  float err_max = 0;
  float err_min = 0;
  float err_total = 0;
  for (int i = 0; i < c.size(); i++) {
    float err = std::abs(ryzenai::bfloat16_to_float(c_gold[i]) -
                         ryzenai::bfloat16_to_float(c[i]));
    if (std::abs(err_max) < std::abs(err)) {
      err_max = err;
    }
    if (i == 0) {
      err_min = err;
    } else if (std::abs(err_min) > std::abs(err)) {
      err_min = err;
    }
    err_total += err;
    if (err > EPSILON * std::sqrt(K)) {
      if (err_count < 20) {
        std::cout << "index " << i << " real "
                  << ryzenai::bfloat16_to_float(c[i]) << " vs golden "
                  << ryzenai::bfloat16_to_float(c_gold[i]) << std::endl;
      }
      err_count++;
    }
  }
  std::cout << "err_max: " << err_max << ", err_min: " << err_min
            << ", err_mean: " << err_total / c.size() << std::endl;

  return err_count;
}

template <typename T>
static void initialize_random(std::vector<T> &vec, size_t size, int data_max,
                              std::string dtype = "int8") {
  auto data_min = -(data_max + 1);
  for (int i = 0; i < size; i++) {
    if (dtype == "bfloat16") {
      vec[i] = ryzenai::rand_bfloat16(float(data_max));
    } else if (dtype == "uint4") {
      vec[i] = ryzenai::rand_uint4(data_max);
    } else if (dtype == "int4") {
      vec[i] = ryzenai::rand_int4(data_max);
    } else if (std::is_same<T, float>::value) {
      vec[i] = (2.0 * (rand() / (float)RAND_MAX) - 1.0) * data_max;
    } else {
      vec[i] = (T)(rand() % (data_max - data_min + 1)) + data_min;
    }
  }
}

template <typename InT = int16_t, typename WgT = int16_t, typename OutT = float>
static void matmul_batch(std::vector<InT> &a_in, std::vector<WgT> &b_in,
                         std::vector<OutT> &c_in, std::tuple<int, int> &a_shape,
                         std::tuple<int, int> &b_shape, int batch_in,
                         bool trans = false) {
  int batch = batch_in;
  int M = std::get<0>(a_shape);
  int K = std::get<1>(a_shape);
  int N = std::get<1>(b_shape);
  if (trans)
    N = std::get<0>(b_shape);

  size_t a_2d_size = M * K;
  size_t b_2d_size = K * N;
  size_t c_2d_size = M * N;

  for (int bat_id = 0; bat_id < batch; bat_id++) {
    InT *a = a_in.data() + (bat_id * a_2d_size);
    WgT *b = b_in.data() + (bat_id * b_2d_size);
    OutT *c = c_in.data() + (bat_id * c_2d_size);
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        OutT sum = 0;
        for (int k = 0; k < K; k++) {
          OutT a_f = ryzenai::bfloat16_to_float(a[i * K + k]);
          OutT b_f = 0;
          if (trans) {
            b_f = ryzenai::bfloat16_to_float(b[j * K + k]);
          } else {
            b_f = ryzenai::bfloat16_to_float(b[k * N + j]);
          }
          sum += a_f * b_f;
        }
        c[i * N + j] = sum;
      }
    }
  }

  return;
}

template <typename InOutT = uint16_t>
int test_bmm_randomdata(int B, int M, int K, int N, bool debug = false,
                        const std::string &a_dtype = "bfloat16",
                        const std::string &b_dtype = "bfloat16",
                        const std::string &c_dtype = "bfloat16",
                        bool compare_golden = true) {

  auto NUM_EXECUTE_ITERATIONS_ =
      Utils::get_env_var("NUM_EXECUTE_ITERATIONS", "1");

  int32_t garbage_value = 0xCDCDCDCD;
  std::vector<InOutT> a(B * M * K);
  std::vector<InOutT> b(B * K * N);
  std::vector<InOutT> c(B * M * N, garbage_value);
  std::vector<float> c_gold(B * M * N, garbage_value);
  srand(42);
  initialize_random<InOutT>(a, B * M * K, 2, "bfloat16");
  initialize_random<InOutT>(b, B * K * N, 2, "bfloat16");

  std::tuple<int, int> a_shape = {B * M, K};
  std::tuple<int, int> b_shape = {B * K, N};
  if (K == 128)
    b_shape = {B * N, K};

  auto bmm_runner = ryzenai::bmm<InOutT>(a_dtype, b_dtype, c_dtype);
  bmm_runner.debug(debug);

  for (int i = 0; i < stoi(NUM_EXECUTE_ITERATIONS_); i++)
    bmm_runner.execute(a.data(), b.data(), c.data(), a_shape, b_shape);

  if (!compare_golden)
    return 0;

  std::tuple<int, int> a_dim{M, K};
  std::tuple<int, int> b_dim{K, N};
  bool trans = false;
  if (K == 128) {
    trans = true;
    b_dim = {N, K};
  }
  matmul_batch<InOutT, InOutT, float>(a, b, c_gold, a_dim, b_dim, B, trans);

  float const EPSILON = 1.0;
  int err_count = 0;
  float err_max = 0;
  float err_min = 0;
  float err_total = 0;
  for (int i = 0; i < c.size(); i++) {
    float err = std::abs(c_gold[i] - ryzenai::bfloat16_to_float(c[i]));
    if (std::abs(err_max) < std::abs(err)) {
      err_max = err;
    }
    if (i == 0) {
      err_min = err;
    } else if (std::abs(err_min) > std::abs(err)) {
      err_min = err;
    }
    err_total += err;
    if (err > EPSILON * std::sqrt(K)) {
      if (err_count < 20) {
        std::cout << "index " << i << " real "
                  << ryzenai::bfloat16_to_float(c[i]) << " vs golden "
                  << c_gold[i] << std::endl;
      }
      err_count++;
    }
  }
  std::cout << "err_max: " << err_max << ", err_min: " << err_min
            << ", err_mean: " << err_total / c.size() << std::endl;

  return err_count;
}

// bmm1
TEST(Bmm_Testw16a16_Random, Kernel1) {
  auto comp_gold = Utils::get_env_var("SKIP_COMP_GOLDEN", "").empty();
  int err_count =
      test_bmm_randomdata<uint16_t>(32, 2048, 128, 2048, false, "bfloat16",
                                    "bfloat16", "bfloat16", comp_gold);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// bmm2
TEST(Bmm_Testw16a16_Random, Kernel2) {
  auto comp_gold = Utils::get_env_var("SKIP_COMP_GOLDEN", "").empty();
  int err_count =
      test_bmm_randomdata<uint16_t>(32, 2048, 2048, 128, false, "bfloat16",
                                    "bfloat16", "bfloat16", comp_gold);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// bmm1
TEST(Bmm_Testw16a16_Real, Kernel1) {
  int err_count = test_bmm_realdata<uint16_t>(
      32, 2048, 128, 2048, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// bmm2
TEST(Bmm_Testw16a16_Real, Kernel2) {
  int err_count = test_bmm_realdata<uint16_t>(
      32, 2048, 2048, 128, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
