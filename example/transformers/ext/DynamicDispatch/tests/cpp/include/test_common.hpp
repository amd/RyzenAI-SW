#pragma once
#ifndef _TEST_COMMON_HPP_
#define _TEST_COMMON_HPP_

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

constexpr int32_t garbage_value = 0xCD;
const std::string PSF_A8W8_QDQ_XCLBIN_REL_PATH =
    "xclbin/stx/4x2_psf_model_a8w8_qdq.xclbin";
const std::string PSJ_A16W8_QDQ_XCLBIN_REL_PATH =
    "xclbin/stx/4x2_psj_model_a16w8_qdq.xclbin";
const std::string PSH_A16W8_QDQ_XCLBIN_REL_PATH =
    "xclbin/stx/4x2_psh_model_a16w8_qdq.xclbin";
const std::string MLADF_GEMM_4x4_A16FW4ACC16F_XCLBIN_PATH =
    "xclbin/stx/mladf_gemm_4x4_a16fw4acc16f.xclbin";
const std::string XCOM_4x4_XCLBIN_REL_PATH = "xclbin/stx/4x4_dpu.xclbin";
const std::string XCOM_4x4_Q_XCLBIN_REL_PATH =
    "xclbin/stx/4x4_dpu_qconv_qelew_add.xclbin";
const std::string MLADF_SOFTMAX_A16_XCLBIN_PATH =
    "xclbin/stx/mladf_4x2_matmul_softmax_a16w16.xclbin";
const std::string
    LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_REL_PATH =
        "xclbin/stx/llama2_mladf_2x4x4_gemmbfp16_silu_mul_mha_rms_rope.xclbin";
const std::string MLADF_4x2_GEMM_A16A16_XCLBIN_PATH =
    "xclbin/stx/mladf_4x2_matmul_softmax_a16w16.xclbin";
const std::string MLADF_4x2_GEMM_A16W8_XCLBIN_PATH =
    "xclbin/stx/mladf_4x2_gemm_a16w8_qdq.xclbin";

const std::string MLADF_4x2_ELWADD_A16W16_QDQ_XCLBIN_PATH =
    "xclbin/stx/mladf_4x2_add_a16.xclbin";
const std::string MLADF_4x2_ELWMUL_A16W16_QDQ_XCLBIN_PATH =
    "xclbin/stx/mladf_4x2_mul_a16.xclbin";
template <typename T>
static void initialize_random(std::vector<T> &vec, size_t size,
                              T data_max = std::numeric_limits<T>::max(),
                              T data_min = std::numeric_limits<T>::min()) {
  for (size_t i = 0; i < size; i++) {
    vec.at(i) = (rand() % (data_max - data_min)) + data_min;
  }
}

namespace dd {
static float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *dst = (uint8_t *)&y;
  std::memcpy(&dst[2], src, sizeof(uint16_t));
  return y;
}

static inline uint16_t float_to_bfloat16(float x) {
  uint32_t i;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *tmp = (uint8_t *)&i;
  // copy float to uint32_t
  std::memcpy(tmp, src, sizeof(float));
  // round to nearest even
  uint32_t lsb = (i >> 16) & 0x1;
  uint32_t bias = 0x7fff + lsb;
  i += bias;
  // extract upper half of input
  uint16_t y = uint16_t(i >> 16);
  return y;
}

static uint16_t rand_bfloat16(float range = 1.0) {
  float x = range * (2.0 * (rand() / (float)RAND_MAX) - 1.0);
  return float_to_bfloat16(x);
}

static void initialize_random_bfloat16(std::vector<uint16_t> &vec,
                                       int data_max) {
  auto data_min = -(data_max + 1);
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = rand_bfloat16(float(data_max));
  }
}

static void initialize_lowertriangular(std::vector<uint16_t> &vec, int M, int N,
                                       uint16_t value) {
  std::memset(vec.data(), 0, M * N * sizeof(uint16_t));
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      if (n <= m)
        vec[M * m + n] = value;
    }
  }
}

static int count_errors_floatvsbfloat16(std::vector<float> cpu_Y,
                                        std::vector<uint16_t> aie_Y,
                                        std::vector<size_t> tensor_shape,
                                        float error_tolerance = 0.01) {
  size_t num_rows;
  size_t num_cols;
  size_t num_batch;
  if (tensor_shape.size() == 3) {
    num_batch = tensor_shape[0];
    num_rows = tensor_shape[1];
    num_cols = tensor_shape[2];
  } else if (tensor_shape.size() == 2) {
    // no harm with batch size being 1
    num_batch = 1;
    num_rows = tensor_shape[0];
    num_cols = tensor_shape[1];
  } else {
    throw std::runtime_error(
        "count_errors_floatvsbfloat16 only supports either rank 2 [Rows,Cols] "
        "or rank3 [Batch,Rows,Cols] comparisson");
  }
  int fail = 0;
  int err_count = 0;
  float max_diff = 0.0;
  for (int b = 0; b < num_batch; ++b) {
    for (int r = 0; r < num_rows; ++r) {
      for (int c = 0; c < num_cols; ++c) {
        float diff =
            std::abs(cpu_Y.at(b * num_cols * num_rows + r * num_cols + c) -
                     bfloat16_to_float(
                         aie_Y.at(b * num_cols * num_rows + r * num_cols + c)));
        if (diff > max_diff)
          max_diff = diff;
        if (diff > error_tolerance) {
          // printf("ERROR: Y[%d][%d][%d] Expected: %f, %d, Received: %f, %d
          // \n",
          //        b, r, c, cpu_Y.at(b * num_cols * num_rows + r * num_cols +
          //        c), cpu_Y.at(b * num_cols * num_rows + r * num_cols + c),
          //        bfloat16_to_float(
          //            aie_Y.at(b * num_cols * num_rows + r * num_cols + c)),
          //        aie_Y.at(b * num_cols * num_rows + r * num_cols + c));
          fail = 1;
          err_count++;
        }
      }
    }
  }
  std::cout << "max_diff is " << max_diff << std::endl;
  return err_count;
}

static int count_errors_bfloat16vsbfloat16(std::vector<uint16_t> cpu_Y,
                                           std::vector<uint16_t> aie_Y,
                                           std::vector<size_t> tensor_shape,
                                           float error_tolerance = 0.01) {
  size_t num_rows;
  size_t num_cols;
  size_t num_batch;
  if (tensor_shape.size() == 3) {
    num_batch = tensor_shape[0];
    num_rows = tensor_shape[1];
    num_cols = tensor_shape[2];
  } else if (tensor_shape.size() == 2) {
    // no harm with batch size being 1
    num_batch = 1;
    num_rows = tensor_shape[0];
    num_cols = tensor_shape[1];
  } else {
    throw std::runtime_error(
        "count_errors_floatvsbfloat16 only supports either rank 2 [Rows,Cols] "
        "or rank3 [Batch,Rows,Cols] comparisson");
  }
  int fail = 0;
  int err_count = 0;
  float max_diff = 0.0;
  for (int b = 0; b < num_batch; ++b) {
    for (int r = 0; r < num_rows; ++r) {
      for (int c = 0; c < num_cols; ++c) {
        float diff =
            std::abs(bfloat16_to_float(
                         cpu_Y.at(b * num_cols * num_rows + r * num_cols + c)) -
                     bfloat16_to_float(
                         aie_Y.at(b * num_cols * num_rows + r * num_cols + c)));
        if (diff > max_diff)
          max_diff = diff;
        if (diff > error_tolerance) {
          // printf("ERROR: Y[%d][%d][%d] Expected: %f, %d, Received: %f, %d
          // \n",
          //        b, r, c, cpu_Y.at(b * num_cols * num_rows + r * num_cols +
          //        c), cpu_Y.at(b * num_cols * num_rows + r * num_cols + c),
          //        bfloat16_to_float(
          //            aie_Y.at(b * num_cols * num_rows + r * num_cols + c)),
          //        aie_Y.at(b * num_cols * num_rows + r * num_cols + c));
          fail = 1;
          err_count++;
        }
      }
    }
  }
  std::cout << "max_diff is " << max_diff << std::endl;
  return err_count;
}

} // namespace dd

static inline void confirmOpen(std::ofstream &file) {
  if (!file) {
    std::cerr << "Error: File could not be opened." << std::endl;
    throw;
  }
}

static void rand_init_int(int8_t *ptr, size_t size) {
  srand(32);
  for (int i = 0; i < size; i++) {
    int8_t r = 16; // static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    ptr[i] = r;
  }
}

#endif
