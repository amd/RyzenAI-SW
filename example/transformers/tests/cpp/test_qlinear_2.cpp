/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
//#include <fstream>

#include "matrix_formatting.h"
#include <qlinear_2.hpp>

template <typename T>
static void initialize_random(std::vector<T> &vec, size_t size, int data_max, std::string b_dtype = "int8") {
  auto data_min = -(data_max + 1);
  for (int i = 0; i < size; i++) {
    if(b_dtype == "bloat16"){
      vec[i] = ryzenai::rand_bfloat16(float(data_max));
    }
    else if(b_dtype == "uint4"){ 
      vec[i] = ryzenai::rand_uint4(data_max);
    }else if (b_dtype == "int4"){
      vec[i] = ryzenai::rand_int4(data_max);
    }else if (std::is_same<T, float>::value) {
      vec[i] = (2.0 * (rand() / (float)RAND_MAX) - 1.0) * data_max;
    } else {
      vec[i] = (T)(rand() % (data_max - data_min + 1)) + data_min;
    }
  }
}

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int32_t>
static void matmul(std::vector<InT> &a, std::vector<WgT> &b,
                   std::vector<OutT> &c, std::tuple<int, int> &a_shape,
                   std::tuple<int, int> &b_shape, bool apply_srs = false,
                   OutT shift = 12, OutT srs_min = -128, OutT srs_max = 127) {
  int M = std::get<0>(a_shape);
  int K = std::get<1>(a_shape);
  int N = std::get<1>(b_shape);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      OutT sum = 0;
      for (int k = 0; k < K; k++) {
        sum += OutT(a[i * K + k]) * OutT(b[k * N + j]);
      }
      if constexpr (!std::is_same_v<OutT, float>) {
        if (apply_srs) {
          sum = sum >> shift;
          sum = (sum < srs_min) ? srs_min : sum;
          sum = (sum > srs_max) ? srs_max : sum;
        }
      }
      c[i * N + j] = sum;
    }
  }
}

/*
 * GeMM Unit Test
 *
 * X x Y x Z is the maximum supported kernel size
 * M x K x N is the actual GeMM shape
 *
 *    NOTE: When M < X, the AIE wrapper can transparently call a kernel
 *          with a smaller value for X by selecting a different DPU sequence.
 *          This allows us to quickly execute token generation (when M == 1)
 *          and prefill phase (where M may be large).
 *
 */
template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int32_t>
int test_matmul(int M, int K, int N, bool debug = false,
                const std::string &a_dtype = "int8",
                const std::string &b_dtype = "int8",
                const std::string &c_dtype = "int32", int group_size = 32) {
  // NOTE: The actual kernel shape is decided in
  //       run_aie and initialize_weights to be compatible
  //       with the pytorch runtime, so these shapes don't
  //       matter. They exist for backwards compatibility.
  //       We use 8 x 2k x 2k as dummy values here.
  if (a_dtype == "int8" || a_dtype == "int16") {
    std::tuple<int, int> kernel_x_shape = {32, 4096};
    std::tuple<int, int> kernel_y_shape = {4096, 4096};
    std::tuple<int, int> a_shape = {M, K};
    std::tuple<int, int> b_shape = {K, N};

    std::vector<InT> a(M * K);
    std::vector<WgT> b(K * N);
    int32_t garbage_value = 0xCDCDCDCD;
    std::vector<OuT> c(M * N, garbage_value);
    std::vector<OuT> c_golden(M * N, garbage_value);

    srand(42);
    initialize_random<InT>(a, M * K, 127);
    initialize_random<WgT>(b, K * N, 127);
    matmul<InT, WgT, OuT>(a, b, c_golden, a_shape, b_shape);

    ryzenai::qlinear_2 qlin =
        ryzenai::qlinear_2<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype);
    qlin.debug(debug);
    qlin.initialize_weights(b.data(), b_shape, group_size);
    qlin.execute(a.data(), a_shape, c.data());

    int err_count = 0;
    for (int i = 0; i < c.size(); i++) {
      OuT err = std::abs(c_golden[i] - c[i]);
      if (err > 0) {
        // std::cout << "c[" << i << "]: "
        //           << "Expected: " << c_golden[i] << ", "
        //           << "Received: " << c[i] << "\n";
        err_count += 1;
      }
    }
    return err_count;
  } else if (b_dtype == "int4" || b_dtype == "uint4") {
    std::tuple<int, int> a_shape = {M, K};
    std::tuple<int, int> b_shape = {K, N};

    std::vector<InT> a(M * K);
    std::vector<float> bias(N);
    std::vector<float> scales(K * N / group_size);
    std::vector<WgT> b(K * N);
    std::vector<WgT> zeros(K * N / group_size);
    int32_t garbage_value = 0xCDCDCDCD;
    std::vector<OuT> c(M * N, garbage_value);
    std::vector<OuT> c_golden(M * N, garbage_value);
    ryzenai::QuantMatrix tiled_B(K, N);
    tiled_B.data = (ryzenai::CoreSubv *)(std::malloc(tiled_B.data_size));
    srand(42);
    initialize_random<InT>(a, M * K, 42, "bloat16");
    initialize_random<WgT>(b, K * N, 7, b_dtype);
    initialize_random<WgT>(zeros, K * N / group_size, 7, b_dtype);
    initialize_random<float>(bias, N, 1);
    initialize_random<float>(scales, K * N / group_size, 1);

  //  std::ofstream input_a, input_b;
  //  input_a.open("input_a.txt"); 
  //  input_b.open("input_b.txt"); 
    // compute golden
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        c_golden[m * N + n] = bias[n];
        for (int k = 0; k < K; ++k) {
          float x = ryzenai::bfloat16_to_float(a[m * K + k]);
          int g_idx = (k / group_size); 
          float y = (b[k * N + n]-zeros[g_idx * N + n]) * ryzenai::bfloat16_rnd_even(scales[g_idx * N  + n ]);
          // if(n==0) input_a << ryzenai::bfloat16_rnd_even(x)<<std::endl;
          // if(m==0) {
          //   //input_b << ryzenai::bfloat16_rnd_even(y)<<std::endl;
          // }
          c_golden[m * N + n] +=
              ryzenai::bfloat16_rnd_even(x) * ryzenai::bfloat16_rnd_even(y);
        }
      }
    }
    // input_b.close();
    // input_a.close();
    // execute on aie
    ryzenai::qlinear_2 qlin =
        ryzenai::qlinear_2<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype);
    qlin.debug(debug);
    qlin.initialize_weights_int4(b.data(), zeros.data(), (float *)scales.data(),
                                 (float *)bias.data(), b_shape, group_size);
    qlin.execute(a.data(), a_shape, c.data());
    float const EPSILON = 1.0;
    int err_count = 0;
    float err_max = 0;
    float err_min = 0;
    float err_total = 0;
    for (int i = 0; i < c.size(); i++) {
      OuT err = std::abs(c_golden[i] - c[i]);
      if (std::abs(err_max) < std::abs(err)) {
        err_max = err;
      }
      if (i == 0) {
        err_min = err;
      }
      else if (std::abs(err_min) > std::abs(err)) 
      {
        err_min = err;
      }
      err_total += err; 
      if (err > EPSILON) {
        //  std::cout << "c[" << i << "]: "
        //            << "Expected: " << c_golden[i] << ", "
        //            << "Received: " << c[i] << "\n";
        err_count++;
      }
    }
    //printf("err_max: %f, err_min: %f, err_mean: %f \n", err_max, err_min, err_total/c.size());
    std::free((void *)tiled_B.data);
    return err_count;
  }
}

/* Test kernel dimensions without tiling */

/* Test kernels with M = 1*/

TEST(Qlinear_2Testw8a8, Kernel1) {
  int err_count = test_matmul(1, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel2) {
  int err_count = test_matmul(1, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel3) {
  int err_count = test_matmul(1, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test kernels with M = 8 */

TEST(Qlinear_2Testw8a8, Kernel4) {
  int err_count = test_matmul(8, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel5) {
  int err_count = test_matmul(8, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel6) {
  int err_count = test_matmul(8, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test kernels with M = 16 */

TEST(Qlinear_2Testw8a8, Kernel7) {
  int err_count = test_matmul(16, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel8) {
  int err_count = test_matmul(16, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel9) {
  int err_count = test_matmul(16, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test kernels with M = 32 */

TEST(Qlinear_2Testw8a8, Kernel10) {
  int err_count = test_matmul(32, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel11) {
  int err_count = test_matmul(32, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel12) {
  int err_count = test_matmul(32, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test tiling up to OPT vocabulary output dimension */

TEST(Qlinear_2Testw8a8, Tiling1) {
  int err_count = test_matmul(8, 2048, 50272);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test tiling in all three dimensions with padding */

TEST(Qlinear_2Testw8a8, Tiling2) {
  int err_count = test_matmul(42, 5000, 2500);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Tiling3) {
  int err_count = test_matmul(42, 16000, 2500);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* Test padding multiple token input with OPT 1.3B dimensions */

TEST(Qlinear_2Testw8a8, Padding1) {
  int err_count = test_matmul(12, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Padding2) {
  int err_count = test_matmul(12, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Padding3) {
  int err_count = test_matmul(12, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/*Llama2 shapes tests*/
TEST(Qlinear_2Testw8a16, Kernel1) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        32, 4096, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
      std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw8a16, Kernel2) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        32, 4096, 11008, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
     std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw8a16, Kernel3) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        32, 11008, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
    std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw8a16, Kernel4) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        64, 4096, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
    std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw8a16, Kernel5) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        64, 11008, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
    std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw8a16, Kernel6) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        64, 4096, 11008, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
    std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw8a16, Kernel7) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        32, 4096, 32000, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
    std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw8a16, Kernel8) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        1, 4096, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
    std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw8a16, Kernel9) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        1, 4096, 11008, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
    std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw8a16, Kernel10) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        1, 4096, 32000, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
    std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw8a16, Kernel11) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        1, 11008, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
    std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw8a16, KernelTiling1) {
  if (std::getenv("DEVICE") == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        1, 16000, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  }
  else {
    std::cout << "Test not supported on " << std::getenv("DEVICE") << std::endl;
  }
}

TEST(Qlinear_2Testw4a16, Kernel1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 12288, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 11008, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, Kernel5) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 12288, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, Kernel6) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 11008, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel7) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, Kernel8) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 12288, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, Kernel9) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 11008, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelTiling1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 32000, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelTiling2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      64, 4096, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelTiling3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 8192, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelTiling4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 4608, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelPadding1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 2048, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelPadding2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 2048, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, Kernelgroup1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 8192, false, "bfloat16", "uint4", "float32", 64);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, Kernelgroup2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 8192, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelStartCoding1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 6400, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelStartCoding2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6144, 6400, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelStartCoding3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6400, 6144, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelStartCoding4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6400, 6144, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelStartCoding5) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 24576, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelStartCoding6) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6144, 24576, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelStartCoding7) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 24576, 6144, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelStartCoding8) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 24576, 6144, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelStartCoding9) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 49152, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw4a16, KernelStartCoding10) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 49152, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 12288, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 11008, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel5) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 12288, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel6) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 11008, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel7) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel8) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 12288, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel9) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 11008, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelTiling1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 32000, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelTiling2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      64, 4096, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelTiling3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 8192, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelTiling4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 4608, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelPadding1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 2048, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelPadding2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 2048, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernelgroup1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 8192, false, "bfloat16", "int4", "float32", 64);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernelgroup2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 8192, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, KernelStartCoding1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 6400, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6144, 6400, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6400, 6144, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6400, 6144, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding5) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 24576, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding6) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6144, 24576, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding7) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 24576, 6144, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding8) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 24576, 6144, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding9) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 49152, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding10) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 49152, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test debug feature */
TEST(Qlinear_2Testw8a8, Debug) {
  int err_count = test_matmul(1, 2048, 2048, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
