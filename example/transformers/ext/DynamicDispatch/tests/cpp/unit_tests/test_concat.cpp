/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include <ops/concat/concat.hpp>

#include "enable_perf.hpp"

#include "test_common.hpp"

template <typename InT = int8_t, typename OuT = int16_t>
int test_concat(size_t MA, size_t KA, size_t MB, size_t KB, size_t N,
                bool debug = false, const std::string &a_dtype = "int16",
                const std::string &c_dtype = "int32",
                const std::string &model_name = "PSF") {

  int err_count = 0;

  if (MA != MB) {
    throw std::invalid_argument(
        "Matrix A and B must have the same number of rows.");
  }
  std::vector<size_t> a_shape = {MA, KA};
  std::vector<size_t> b_shape = {MB, KB};
  size_t MC = MA;
  size_t KC = KA + N * KB;
  std::vector<size_t> c_shape = {MC, KC};

  std::vector<InT> a(MA * KA);
  std::vector<InT> b(MB * KB * N);
  std::vector<OuT> cpu_out(MC * KC);
  std::vector<OuT> aie_out(MC * KC);

  initialize_random<InT>(a, MA * KA, 128, 0);
  initialize_random<InT>(b, MB * KB * N, 128, 0);

  // compute golden
  for (int i = 0; i < MA; ++i) {
    for (int j = 0; j < KA; ++j) {
      cpu_out[(i * KC) + j] = a[(i * KA) + j];
    }
  }
  for (int n = 0; n < N; ++n) {
    for (int i = 0; i < MB; ++i) {
      for (int j = 0; j < KB; ++j) {
        cpu_out[(i * KC) + KA + (n * KB) + j] = b[n * MB * KB + (i * KB) + j];
      }
    }
  }
  // run AIE
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::concat concat_ =
      ryzenai::concat<InT, OuT>(a_dtype, c_dtype, true, attr);

  std::vector<Tensor> input_Tensor;
  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  input_Tensor.push_back(a_T);

  for (int n = 0; n < N; n++) {
    InT *ptr_matB = &b[n * MB * KB];
    struct Tensor b_T = {ptr_matB, b_shape, a_dtype};
    input_Tensor.push_back(b_T);
  }

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), c_shape, c_dtype}};

  concat_.debug(debug);

#ifdef UNIT_TEST_PERF
  PROFILE_THIS(concat_.execute(input_Tensor, output_Tensor));
#else
  concat_.execute(input_Tensor, output_Tensor);
#endif
  // Check results
  for (int i = 0; i < MC; ++i) {
    for (int j = 0; j < KC; ++j) {
      InT ref = cpu_out[(i * KC) + j];
      InT act = aie_out[(i * KC) + j];
      if (ref != act) {
        std::cout << "ERROR: [" << i << ", " << j << "]: "
                  << "Expected: " << ref << ", "
                  << "Received: " << act << "\n";
        err_count += 1;
      }
    }
  }

  return err_count;
}

// CONCAT 4x4
TEST(C4PSR_CONCAT_a16, Kernel1) {
  int err_count = test_concat<uint16_t, uint16_t>(64, 64, 64, 64, 19, false,
                                                  "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONCAT_a16, Kernel2) {
  int err_count = test_concat<uint16_t, uint16_t>(256, 64, 256, 64, 19, false,
                                                  "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONCAT_a16, Kernel3) {
  int err_count = test_concat<uint16_t, uint16_t>(1024, 64, 1024, 64, 9, false,
                                                  "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONCAT_a16, Kernel4) {
  int err_count = test_concat<uint16_t, uint16_t>(4096, 64, 4096, 64, 4, false,
                                                  "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONCAT_a16, Kernel5) {
  int err_count = test_concat<uint16_t, uint16_t>(
      4096, 640, 4096, 320, 1, false, "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONCAT_a16, Kernel6) {
  int err_count = test_concat<uint16_t, uint16_t>(64, 1280, 64, 1280, 1, false,
                                                  "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONCAT_a16, Kernel7) {
  int err_count = test_concat<uint16_t, uint16_t>(256, 1280, 256, 640, 1, false,
                                                  "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONCAT_a16, Kernel8) {
  int err_count = test_concat<uint16_t, uint16_t>(
      256, 1280, 256, 1280, 1, false, "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONCAT_a16, Kernel9) {
  int err_count = test_concat<uint16_t, uint16_t>(
      1024, 640, 1024, 320, 1, false, "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONCAT_a16, Kernel10) {
  int err_count = test_concat<uint16_t, uint16_t>(
      1024, 640, 1024, 640, 1, false, "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONCAT_a16, Kernel11) {
  int err_count = test_concat<uint16_t, uint16_t>(
      1024, 1280, 1024, 640, 1, false, "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONCAT_a16, Kernel12) {
  int err_count = test_concat<uint16_t, uint16_t>(
      4096, 320, 4096, 320, 1, false, "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
