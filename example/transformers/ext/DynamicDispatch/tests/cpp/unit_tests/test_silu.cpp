/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

#include "../src/ops/ops_common/matmul_matrix.hpp"
#include <ops/silu/silu.hpp>

#include "enable_perf.hpp"

#include "test_common.hpp"

namespace {
// Function to count the number of lines in the file
size_t countLines(const std::string &filename) {
  std::ifstream file(filename);
  size_t lineCount = 0;
  std::string line;
  while (std::getline(file, line)) {
    ++lineCount;
  }
  return lineCount;
}
// Function to load hex values from a file into a vector
bool loadHexValues(const std::string &filename,
                   std::vector<uint16_t> &hexValues, float force_value) {
  size_t lineCount = countLines(filename);
  hexValues.resize(lineCount * 2); // Each line contains 2 hex values
  bool do_once = false;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file." << std::endl;
    return false;
  }

  std::string line;
  size_t index = 0;
  while (std::getline(file, line)) {
    if (line.length() != 8) {
      std::cerr << "Invalid line length: " << line << std::endl;
      continue;
    }

    std::string highStr = line.substr(0, 4);
    std::string lowStr = line.substr(4, 4);

    uint16_t highValue;
    uint16_t lowValue;

    std::stringstream highConverter;
    std::stringstream lowConverter;

    highConverter << std::hex << highStr;
    highConverter >> highValue;

    lowConverter << std::hex << lowStr;
    lowConverter >> lowValue;
    if (!do_once) {
      std::cout << " highValue " << dd::bfloat16_to_float(highValue)
                << std::endl;
      printf("highValue %f, %d \n", dd::bfloat16_to_float(highValue),
             highValue);
      std::cout << " lowValue " << dd::bfloat16_to_float(lowValue) << std::endl;
      printf("lowValue %f, %d \n", dd::bfloat16_to_float(lowValue), highValue);
      do_once = true;
    }

    hexValues.at(index++) = dd::float_to_bfloat16(
        force_value); //(highValue==0&&force_value)?
                      // dd::float_to_bfloat16(6.0f):highValue;
    hexValues.at(index++) = dd::float_to_bfloat16(
        force_value); //(lowValue==0&&force_value)?
                      // dd::float_to_bfloat16(6.0f):lowValue;
  }

  file.close();
  return true;
}
} // namespace

template <typename InT = uint16_t, typename OuT = uint16_t>
int test_silu(size_t M, size_t K, bool debug = false,
              const std::string &a_dtype = "bfloat16",
              const std::string &c_dtype = "bfloat16",
              const std::string &model_name = "LLAMA2",
              const bool use_reference_data = false) {
  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);

  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ms, Ks};

  std::vector<InT> a(M * K);
  std::vector<InT> b(M * K);
  std::vector<float> cpu_float(M * K);
  std::vector<OuT> aie_out(M * K, garbage_value);

  dd::initialize_random_bfloat16(a, 42);
  // compute golden
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < K; c++) {
      float x = dd::bfloat16_to_float(a.at(r * K + c));
      float sigmoid = 1 / (std::exp(-x) + 1);
      float fSilu = x * sigmoid;
      cpu_float.at(r * K + c) = fSilu;
    }
  }

  ryzenai::silu silu_ = ryzenai::silu<InT, OuT>(a_dtype, true);

  std::vector<Tensor> const_Tensor;

  std::vector<Tensor> input_Tensor;
  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  silu_.debug(debug);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(silu_.execute(input_Tensor, output_Tensor));
#else
  silu_.execute(input_Tensor, output_Tensor);
#endif
  err_count = dd::count_errors_floatvsbfloat16(cpu_float, aie_out, a_shape,
                                               silu_.EPSILON);
  return err_count;
}

// SiLU
TEST(LLAMA2_SILU_Testa16, Kernel2048x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(2048, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1024x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(1024, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel512x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(512, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel256x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(256, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel128x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(128, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(1, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
