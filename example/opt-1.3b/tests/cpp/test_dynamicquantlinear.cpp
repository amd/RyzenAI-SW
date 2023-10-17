#include <dynamic_quantlinear.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <xrt/xrt_bo.h>

template <typename T>
static void initialize_random(std::vector<T> &vector, size_t size, int data_max,
                              float scale) {
  auto ptr = vector.data();
  int data_min = -(data_max + 1);
  for (int i = 0; i < size; i++) {
    ptr[i] = ((rand() % (data_max - data_min + 1)) + data_min) * scale;
  }
}

template <typename InT, typename OutT>
static void matmul(std::vector<InT> &a, std::vector<InT> &b,
                   std::vector<OutT> &c, std::tuple<int, int> &a_shape,
                   std::tuple<int, int> &b_shape, float x_scale, float y_scale,
                   int out_scale) {
  auto a_ptr = a.data();
  auto b_ptr = b.data();
  auto c_ptr = c.data();
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      OutT sum = 0;
      for (int k = 0; k < K; k++) {
        sum += std::round(((OutT)a[i * K + k]) / x_scale) *
               std::round(((OutT)b[k * N + j]) / y_scale);
      }
      c_ptr[i * N + j] =
          (float)((sum / out_scale)) * x_scale * y_scale * out_scale;
    }
  }
}

void save_bin_fime(int8_t *data, size_t size, std::string file) {
  std::ofstream ofs(file, std::ios::binary | std::ios::out);
  ofs.write((const char *)data, size);
  ofs.close();
}

// Test 4x2kx2k dll
TEST(dyqlinearTest, dyqlinearBasic) {

  const std::string aie_kernel_dll =
      Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll";

  std::tuple<int, int> a_shape = {4, 2048};
  std::tuple<int, int> b_shape = {2048, 2048};
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);

  std::vector<float> c(M * N, 0);
  std::vector<float> golden(M * N, 0);

  initialize_random<float>(a, M * K, 2, 43.0 / 128);
  initialize_random<float>(b, K * N, 2, 0.22 / 128);

  float max = Utils::abs_max<float>(
      a.data(), (int)(std::get<0>(a_shape) * std::get<1>(a_shape)));
  float x_scale = (float)(max / 128);
  float y_scale = 0.22 / 128;
  int in_scale = 1;
  int out_scale = 1;

  ryzenai::dynamicquantlinear lin = ryzenai::dynamicquantlinear<float, float>(
      aie_kernel_dll, a_shape, b_shape, y_scale, out_scale, 1, "linear.txt");
  lin.initialize_weights_data(b.data(), b_shape, y_scale);
  lin.execute_aie(a.data(), a_shape, c.data());
  matmul<float, float>(a, b, golden, a_shape, b_shape, x_scale, y_scale,
                       out_scale);

  for (int i = 0; i < c.size(); i++) {
    auto err = std::abs(golden.data()[i] - c.data()[i]);
    EXPECT_TRUE(err <= std::abs(golden.data()[i])) << "Absolute error > golden";
  }
}

// Test 8x2kx2k dll
TEST(dyqlinearTest, dyqlinearBasic2) {

  const std::string aie_kernel_dll =
      Utils::get_dll_path() + "libGemmQnnAie_8x2048_2048x2048.dll";

  std::tuple<int, int> a_shape = {8, 2048};
  std::tuple<int, int> b_shape = {2048, 2048};
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);

  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c(M * N, 0);
  std::vector<float> golden(M * N, 0);

  initialize_random<float>(a, M * K, 2, 43.0 / 128);
  initialize_random<float>(b, K * N, 2, 0.22 / 128);

  float max = Utils::abs_max<float>(
      a.data(), (int)(std::get<0>(a_shape) * std::get<1>(a_shape)));
  float x_scale = (float)(max / 128);
  float y_scale = 0.22 / 128;
  int in_scale = 1;
  int out_scale = 1;

  ryzenai::dynamicquantlinear lin = ryzenai::dynamicquantlinear<float, float>(
      aie_kernel_dll, a_shape, b_shape, y_scale, out_scale, 1, "linear.txt");

  lin.initialize_weights_data(b.data(), b_shape, y_scale);
  lin.execute_aie(a.data(), a_shape, c.data());

  matmul<float, float>(a, b, golden, a_shape, b_shape, x_scale, y_scale,
                       out_scale);

  for (int i = 0; i < c.size(); i++) {
    auto err = std::abs(golden.data()[i] - c.data()[i]);
    EXPECT_TRUE(err <= std::abs(golden.data()[i]))
        << golden.data()[i] << "  " << c.data()[i] << "Absolute error > golden";
  }
}

// Tiling with 4x2kx2k dll
TEST(dyqlinearTest, dyqTiling1) {

  const std::string aie_kernel_dll =
      Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll";

  std::tuple<int, int> kernel_x_shape = {4, 2048};
  std::tuple<int, int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {8, 2048};
  std::tuple<int, int> b_shape = {2048, 2048};
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);

  std::vector<float> a(M * K);
  std::vector<float> b(K * N);

  std::vector<float> c(M * N, 0);
  std::vector<float> golden(M * N, 0);

  initialize_random<float>(a, M * K, 2, 43.0 / 128);
  initialize_random<float>(b, K * N, 2, 0.22 / 128);

  float max = Utils::abs_max<float>(
      a.data(), (int)(std::get<0>(a_shape) * std::get<1>(a_shape)));
  float x_scale = (float)(max / 128);
  float y_scale = 0.22 / 128;

  ryzenai::dynamicquantlinear lin = ryzenai::dynamicquantlinear<float, float>(
      aie_kernel_dll, kernel_x_shape, kernel_y_shape, y_scale, 1, 1,
      "linear.txt");

  lin.initialize_weights_data(b.data(), b_shape, y_scale);
  lin.execute_aie(a.data(), a_shape, c.data());

  matmul<float, float>(a, b, golden, a_shape, b_shape, x_scale, y_scale, 1);
  for (int i = 0; i < c.size(); i++) {
    auto err = std::abs((golden.data()[i]) - c.data()[i]);
    EXPECT_TRUE(err <= std::abs(golden.data()[i]))
        << "Absolute error >  golden";
  }
}

// Padding & Tiling with 4x2kx2k dll
TEST(dyqlinearTest, dyqPaddingTiling) {

  const std::string aie_kernel_dll =
      Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll";

  std::tuple<int, int> kernel_x_shape = {4, 2048};
  std::tuple<int, int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {1, 2048};
  std::tuple<int, int> b_shape = {2048, 2048};
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);

  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c(M * N, 0);

  initialize_random<float>(a, M * K, 2, 43.0 / 128);
  initialize_random<float>(b, K * N, 2, 0.22 / 128);

  std::vector<float> golden(M * N, 0);
  float max = Utils::abs_max<float>(
      a.data(), (int)(std::get<0>(a_shape) * std::get<1>(a_shape)));
  float x_scale = (float)(max / 128);
  float y_scale = 0.22 / 128;
  int out_scale = 1;

  ryzenai::dynamicquantlinear lin = ryzenai::dynamicquantlinear<float, float>(
      aie_kernel_dll, kernel_x_shape, kernel_y_shape, y_scale, 1, 1,
      "linear.txt");

  lin.initialize_weights_data(b.data(), b_shape, y_scale);
  lin.execute_aie(a.data(), a_shape, c.data());

  matmul<float, float>(a, b, golden, a_shape, b_shape, x_scale, y_scale,
                       out_scale);
  for (int i = 0; i < c.size(); i++) {

    auto err = std::abs((golden.data()[i]) - c.data()[i]);
    EXPECT_TRUE(err <= std::abs(golden.data()[i])) << "Absolute error > golden";
  }
}

// weights file load with 4x2kx2k dll
TEST(dyqlinearTest, dyqWeightsFileLoad) {
  GTEST_SKIP();
  const std::string aie_kernel_dll =
      Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll";

  std::tuple<int, int> kernel_x_shape = {4, 2048};
  std::tuple<int, int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {1, 2048};
  std::tuple<int, int> b_shape = {2048, 2048};
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);

  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c(M * N, 0);
  std::vector<int8_t> b_q(K * N);

  initialize_random<float>(a, M * K, 2, 43.0 / 128);
  initialize_random<float>(b, K * N, 2, 0.22 / 128);
  std::vector<float> golden(M * N, 0);
  float max = Utils::abs_max<float>(
      a.data(), (int)(std::get<0>(a_shape) * std::get<1>(a_shape)));
  float x_scale = (float)(max / 128);
  float y_scale = 0.22 / 128;
  int out_scale = 1;

  std::string wts_file = "wts.bin";

  // weights quant
  for (int i = 0; i < K * N; i++)
    b_q[i] = (int8_t)std::round(b[i] / x_scale);

  ryzenai::dynamicquantlinear lin = ryzenai::dynamicquantlinear<float, float>(
      aie_kernel_dll, kernel_x_shape, kernel_y_shape, y_scale, 1, 1,
      "linear.txt");

  // prepare wts bin file
  save_bin_fime(b_q.data(), K * N, wts_file);
  lin.initialize_weights_bin(wts_file, b_shape);

  lin.execute_aie(a.data(), a_shape, c.data());

  matmul<float, float>(a, b, golden, a_shape, b_shape, x_scale, y_scale,
                       out_scale);
  for (int i = 0; i < c.size(); i++) {
    auto err = std::abs((golden.data()[i]) - c.data()[i]);
    EXPECT_TRUE(err <= std::abs(golden.data()[i])) << "Absolute error > golden";
  }
}

// 2 dll
TEST(dyqlinearTest, dyq2dll) {
  GTEST_SKIP();
  const std::vector<std::string> aie_kernel_dlls = {
      Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll",
      Utils::get_dll_path() + "libGemmQnnAie_8x2048_2048x2048.dll",
  };

  std::vector<std::tuple<int, int>> kernel_x_shapes = {{4, 2048}, {8, 2048}};
  std::tuple<int, int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {8, 8192};
  std::tuple<int, int> b_shape = {8192, 2048};
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);

  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c(M * N, 0);
  std::vector<int8_t> b_q(K * N);

  initialize_random<float>(a, M * K, 2, 43.0 / 128);
  initialize_random<float>(b, K * N, 2, 0.22 / 128);
  std::vector<float> golden(M * N, 0);
  float max = Utils::abs_max<float>(
      a.data(), (int)(std::get<0>(a_shape) * std::get<1>(a_shape)));
  float x_scale = (float)(max / 128);
  float y_scale = 0.22 / 128;
  int out_scale = 1;

  std::string wts_file = "wts.bin";

  // weights quant
  for (int i = 0; i < K * N; i++)
    b_q[i] = (int8_t)std::round(b[i] / x_scale);

  ryzenai::dynamicquantlinear lin = ryzenai::dynamicquantlinear<float, float>(
      aie_kernel_dlls, kernel_x_shapes, kernel_y_shape, y_scale, 1, 1, 2, 4,
      "linear.txt");

  // prepare wts bin file
  save_bin_fime(b_q.data(), K * N, wts_file);
  lin.initialize_weights_bin(wts_file, b_shape);

  lin.execute_aie(a.data(), a_shape, c.data());

  matmul<float, float>(a, b, golden, a_shape, b_shape, x_scale, y_scale,
                       out_scale);
  for (int i = 0; i < c.size(); i++) {
    auto err = std::abs((golden.data()[i]) - c.data()[i]);
    EXPECT_TRUE(err <= std::abs(golden.data()[i])) << "Absolute error > golden";
  }
}
