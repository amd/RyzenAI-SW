/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include <ops/slice/slice.hpp>

#include "enable_perf.hpp"

#include "test_common.hpp"

template <typename InT = int8_t, typename OuT = int16_t>
int test_slice(size_t M, size_t K, int sIdx, bool debug = false,
               const std::string &a_dtype = "int16",
               const std::string &c_dtype = "int32",
               const std::string &model_name = "PSF") {

  int err_count = 0;

  size_t Mo = M;
  size_t Ko = K / 2;

  std::vector<size_t> a_shape = {M, K};
  std::vector<size_t> c_shape = {Mo, Ko};

  std::vector<InT> a(M * K);
  std::vector<OuT> cpu_out(Mo * Ko);
  std::vector<OuT> aie_out(Mo * Ko);

  initialize_random<InT>(a, M * K, 128, 0);

  // compute golden
  int Wout_start = sIdx * Ko;
  for (int i = 0; i < Mo; ++i) {
    for (int j = 0; j < Ko; ++j) {
      cpu_out[(i * Ko) + j] = a[(i * K) + Wout_start + j];
    }
  }
  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
    attr["slice_idx"] = std::vector<int>{sIdx};
  }
  ryzenai::slice slice_ =
      ryzenai::slice<InT, OuT>(a_dtype, c_dtype, true, attr);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), c_shape, c_dtype}};

  slice_.debug(debug);

#ifdef UNIT_TEST_PERF
  PROFILE_THIS(slice_.execute(input_Tensor, output_Tensor));
#else
  slice_.execute(input_Tensor, output_Tensor);
#endif
  for (int i = 0; i < Mo; ++i) {
    for (int j = 0; j < Ko; ++j) {
      InT ref = cpu_out[(i * Ko) + j];
      InT act = aie_out[(i * Ko) + j];
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

// NNI 4x4
TEST(C4PSR_SLICE_a16, Kernel1) {
  int err_count = test_slice<uint16_t, uint16_t>(64, 10240, 0, false, "uint16",
                                                 "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SLICE_a16, Kernel2) {
  int err_count = test_slice<uint16_t, uint16_t>(256, 10240, 0, false, "uint16",
                                                 "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SLICE_a16, Kernel3) {
  int err_count = test_slice<uint16_t, uint16_t>(1024, 5120, 0, false, "uint16",
                                                 "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SLICE_a16, Kernel4) {
  int err_count = test_slice<uint16_t, uint16_t>(4096, 2560, 0, false, "uint16",
                                                 "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SLICE_a16, Kernel5) {
  int err_count = test_slice<uint16_t, uint16_t>(64, 10240, 1, false, "uint16",
                                                 "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SLICE_a16, Kernel6) {
  int err_count = test_slice<uint16_t, uint16_t>(256, 10240, 1, false, "uint16",
                                                 "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SLICE_a16, Kernel7) {
  int err_count = test_slice<uint16_t, uint16_t>(1024, 5120, 1, false, "uint16",
                                                 "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SLICE_a16, Kernel8) {
  int err_count = test_slice<uint16_t, uint16_t>(4096, 2560, 1, false, "uint16",
                                                 "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
