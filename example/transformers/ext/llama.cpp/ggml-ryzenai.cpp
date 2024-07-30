
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/log.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-ryzenai.h"
#include "ggml.h"

// These includes aren't needed for emulation mode
#ifndef RYZENAI_EMULATION
// #include <ryzenai/ryzenai.hpp>
#include <ryzenai/ops/qlinear_2/qlinear_2.hpp>
#include <ryzenai/utils/dtype_utils.h>
#endif

// Macro for wrapping function calls in try catch
#define TRY_CATCH(expression)                                                  \
  try {                                                                        \
    expression                                                                 \
  } catch (const std::exception &e) {                                          \
    std::cerr << "Exception caught: " << e.what() << std::endl;                \
    throw e;                                                                   \
  } catch (...) {                                                              \
    std::cerr << "Unknown exception caught." << std::endl;                     \
    throw;                                                                     \
  }

namespace {

// Transpose inner most dimensions of a vector interpreted as a 4D tensor
template <typename T>
std::vector<T> transpose(const std::vector<T> &tensor,
                         const std::tuple<int, int, int, int> &shapes) {

  const auto &[ne00, ne01, ne02, ne03] = shapes;

  std::vector<T> transposed_tensor(ne00 * ne01 * ne02 * ne03);

  int64_t s3 = ne00 * ne01 * ne02;
  int64_t s2 = ne00 * ne01;

  // Transpose the two innermost dimensions
  for (int64_t b3 = 0; b3 < ne03; ++b3) {
    for (int64_t b2 = 0; b2 < ne02; ++b2) {

      for (int64_t i = 0; i < ne01; ++i) {
        for (int64_t j = 0; j < ne00; ++j) {
          transposed_tensor[b3 * s3 + b2 * s2 + j * ne01 + i] =
              tensor[b3 * s3 + b2 * s2 + i * ne00 + j];
        }
      }
    }
  }

  return transposed_tensor;
}

// A quantized ggml tensor will have its weights and scales packed contiguously
// i.e. two int4 packed into int8
// We need to unpack the parameters into vectors to make them easier to use
void unpack_row_q4_0(const char *xx, int k, std::vector<int8_t> &weights,
                     std::vector<int8_t> &zeros, std::vector<float> &scales) {

  const auto *x = reinterpret_cast<const block_q4_0 *>(xx);

  static const int qk = QK4_0;

  assert(k % qk == 0);
  if (k % qk) {
    std::cerr << "Assertion failed: "
              << "A row must be divisible by " << qk << std::endl;
    std::abort();
  }

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = GGML_FP16_TO_FP32(x[i].d);

    scales.push_back(d);
    //zeros.push_back(8);

    for (int j = 0; j < qk / 2; ++j) {
      weights.push_back(x[i].qs[j] & 0xF);
    }
    for (int j = 0; j < qk / 2; ++j) {
      weights.push_back(x[i].qs[j] >> 4);
    }
  }
}

} // Anonymous namespace

#ifndef RYZENAI_EMULATION

// Currently matrix multiplication depends on qlinear_2 as the primary op
using op_t = ryzenai::qlinear_2<int16_t, int8_t, float, float>;

// Lockable Singleton
// Owns our qlinear_2 ops
class RyzenAIContext {

  RyzenAIContext() = default;
  RyzenAIContext(const RyzenAIContext &) = delete;
  RyzenAIContext &operator=(const RyzenAIContext &) = delete;
  std::mutex mtx_;

public:
  // Get the singleton instance
  static RyzenAIContext &getInstance() {
    static RyzenAIContext instance;
    return instance;
  }

  void lock() { mtx_.lock(); }
  void unlock() { mtx_.unlock(); }

  std::unordered_map<std::string_view, op_t> map;
};
#endif

// We can add our own device specific initialization here at startup time
// Today, we do all init on the fly
void ggml_ryzenai_init(void) {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  initialized = true;
}

// This function is used to check if RyzenAI can offload the specific matrix
// multiplication It considers that we only want to accelerate large mmult, and
// we only support Q4_0 quantization scheme
bool ggml_ryzenai_can_mul_mat(const struct ggml_tensor *src0,
                              const struct ggml_tensor *src1,
                              const struct ggml_tensor *dst) {

  const int64_t ne3 = dst->ne[3];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne0 = dst->ne[0];

  // TODO: find the optimal values for these
  if (src0->type == GGML_TYPE_Q4_0 && // Check this
      ggml_is_contiguous(src1) && src1->type == GGML_TYPE_F32 &&
      dst->type == GGML_TYPE_F32 &&
      ne3 == 1 && // Not currently supporting batched matrix multiplication
      ne2 == 1 && ((ne0 >= 4096))) {
    return true;
  }

  return false;
}

void ggml_ryzenai_mul_mat_emu(const struct ggml_tensor *src0,
                              const struct ggml_tensor *src1,
                              struct ggml_tensor *dst, void *wdata,
                              size_t wsize);
void ggml_ryzenai_mul_mat_ryz(const struct ggml_tensor *src0,
                              const struct ggml_tensor *src1,
                              struct ggml_tensor *dst, void *wdata,
                              size_t wsize);

// Entry point from ggml cpu backend into RyzenAI backend
void ggml_ryzenai_mul_mat(const struct ggml_tensor *src0,
                          const struct ggml_tensor *src1,
                          struct ggml_tensor *dst, void *wdata, size_t wsize) {
  // Work Buffer is unused for now
  (void)wdata;
  (void)wsize;
  GGML_ASSERT(ggml_ryzenai_can_mul_mat(src0, src1, dst));
  // LOG("ggml_compute_forward_mul_mat called\n");
  // LOG("src0: %s dtype: %d (%lld,%lld,%lld,%lld)\n", src0->name, src0->type,
  // src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]); LOG("src1: %s dtype:
  // %d (%lld,%lld,%lld,%lld)\n", src1->name, src1->type, src1->ne[0],
  // src1->ne[1], src1->ne[2], src1->ne[3]); LOG("dst: %s dtype: %d
  // (%lld,%lld,%lld,%lld)\n", dst->name, dst->type, dst->ne[0], dst->ne[1],
  // dst->ne[2], dst->ne[3]);

#ifdef RYZENAI_EMULATION
  ggml_ryzenai_mul_mat_emu(src0, src1, dst, wdata, wsize);
#else
  ggml_ryzenai_mul_mat_ryz(src0, src1, dst, wdata, wsize);
#endif
}

// This function is to be called when we want to software emulate matrix
// multiplication It is useful to understand the actual compute that is expected
// by the framework It does not aim to emulate our hardware
void ggml_ryzenai_mul_mat_emu(const struct ggml_tensor *src0,
                              const struct ggml_tensor *src1,
                              struct ggml_tensor *dst, void *wdata,
                              size_t wsize) {
  // Work Buffer is unused for now
  (void)wdata;
  (void)wsize;
  GGML_ASSERT(ggml_ryzenai_can_mul_mat(src0, src1, dst));

  GGML_TENSOR_BINARY_OP_LOCALS

  void *w = src0->data;

  /*
  // Alternative method for emulating the matrix multiplication
  // In this method, we directly use the dequantization function from ggml

  ggml_to_float_t const to_float = (ggml_to_float_t)dequantize_row_q4_0;

  std::vector<float> dequantized_weights(ne00 * ne01 * ne02 * ne03);
  float *aptr = dequantized_weights.data();
  // Attempt to convert Q4_K to floats
  for (int64_t i03 = 0; i03 < ne03; ++i03) {
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
      for (int64_t i01 = 0; i01 < ne01; ++i01) {
        to_float((const char *)w + i01 * nb01 + i02 * nb02 + i03 * nb03,
                 aptr + i01 * ne00 + i02 * ne01 + i03 * ne02, ne00);
      }
    }
  }

  */

  std::vector<int8_t> weights; // int4 weights
  std::vector<int8_t> zeros;
  std::vector<float> mins;
  std::vector<float> scales;
  // std::vector<float> bias; // Not required for emulation

  // Unpack weights, zeros, scales
  for (int64_t i03 = 0; i03 < ne03; ++i03) {
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
      for (int64_t i01 = 0; i01 < ne01; ++i01) {
        unpack_row_q4_0((const char *)w + i01 * nb01 + i02 * nb02 + i03 * nb03,
                        ne00, weights, zeros, scales);
      }
    }
  }

  // Convert weights, zeros, scales to floating point weights (Dequantize)
  // We will change the scalar and the zero after every 32 elements
  std::vector<float> A(ggml_nelements(src0));
  int64_t sidx = 0;
  for (int64_t i = 0; i < weights.size(); ++i) {
    A[i] = scales[sidx] * (weights[i] - zeros[sidx]);
    if ((i + 1) % 32 == 0) {
      ++sidx;
    }
  }

  float *aptr = A.data();

  float *bptr = (float *)(src1->data);

  float *cptr = (float *)(dst->data);

  // Compute C^T
  // The input matricies A and B have the same width for cache performance
  // For some reason, the result of the computation is C^T
  for (int i3 = 0; i3 < ne3; ++i3) {
    for (int i2 = 0; i2 < ne2; ++i2) {
      for (int i1 = 0; i1 < ne1; ++i1) {   // rows of C, rows of B
        for (int i0 = 0; i0 < ne0; ++i0) { // cols of C, rows of A
          cptr[i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0] = 0;
          for (int k = 0; k < ne10; ++k) { // Shared dimension
            cptr[i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0] +=
                aptr[i3 * ne02 * ne01 * ne00 + i2 * ne01 * ne00 + i0 * ne00 +
                     k] *
                bptr[i3 * ne12 * ne11 * ne10 + i2 * ne11 * ne10 + i1 * ne10 +
                     k];
          }
        }
      }
    }
  }
}

void ggml_ryzenai_mul_mat_ryz(const struct ggml_tensor *src0,
                              const struct ggml_tensor *src1,
                              struct ggml_tensor *dst, void *wdata,
                              size_t wsize) {

  // Work Buffer is unused for now
  (void)wdata;
  (void)wsize;

  GGML_ASSERT(ggml_ryzenai_can_mul_mat(src0, src1, dst));

  GGML_TENSOR_BINARY_OP_LOCALS

#ifndef RYZENAI_EMULATION

  auto &ctx = RyzenAIContext::getInstance();

  // Assume src0 is always the weight tensor
  // Assume src1 is always the inputs
  // This is always true when I run LLama2
  // However, this assumption may not be globally safe, probably should check

  // Check if we need to create executor (qlinear_2 object) for this weight
  // tensor
  auto key = std::string_view(src0->name);
  ctx.lock();
  if (ctx.map.count(key) == 0) { // No executor

    TRY_CATCH(ctx.map.emplace(
        std::piecewise_construct, std::forward_as_tuple(key),
        std::forward_as_tuple("bfloat16", "uint4", "float32")););

    std::vector<int8_t> weights; // int4 weights
    weights.reserve(ggml_nelements(src0));
    std::vector<int8_t> zeros(ggml_nelements(src0)/32, 8); // Will be a vector of 8s for Q4_0 quantization scheme
    std::vector<float> scales;
    scales.reserve(ggml_nelements(src0)/32);
    std::vector<float> bias(ne01, 0); // Vector of zeros, should have size N in MxK * K*N

    // Unpack weights, zeros, scales
    void *w = src0->data;
    for (int64_t i03 = 0; i03 < ne03; ++i03) {
      for (int64_t i02 = 0; i02 < ne02; ++i02) {
        for (int64_t i01 = 0; i01 < ne01; ++i01) {
          unpack_row_q4_0((const char *)w + i01 * nb01 + i02 * nb02 +
                              i03 * nb03,
                          ne00, weights, zeros, scales);
        }
      }
    }

    // Need to transpose the weights and scales
    // Why you ask?
    // GGML's matmul does A * B^T = C^T // B is already assumed transposed
    // This is for maximal cache hit efficiency
    // qlinear_2 does A * B = C where B must be the weights
    // Here we see that the weights are presented as A, so we need to swap A and
    // B I make use of the matrix multiplication transpose property B^T * A^T =
    // C^T So if we transpose the weights, and feed the input directly in,
    // qlinear_2 will compute C^T as ggml expects.
    auto transposed_weights =
        transpose(weights, std::make_tuple(ne00, ne01, ne02, ne03));

    auto transposed_scales = transpose(
        scales, std::make_tuple(ne00 / 32, ne01, ne02,
                                ne03)); // 1 scale for every 32 elements

    auto w_shape = make_tuple(
        ne00, ne01); // qlinear_2 expects KxN = w_shape[0] x w_shape[1]

    TRY_CATCH(ctx.map.at(key).initialize_weights_int4(
        transposed_weights.data(), zeros.data(), transposed_scales.data(),
        bias.data(), w_shape););
  }
  ctx.unlock();

  // Kernel can only accept bfloat16
  // Need to convert inputs from F32 to bfloat16
  const static bool use_avx = ryzenai::check_avx512_and_bf16_support();
  vector<int16_t> bfloatInputs(ggml_nelements(src1));
  ryzenai::float_buffer_to_bfloat16((float *)(src1->data), ggml_nelements(src1),
                                    (uint16_t *)bfloatInputs.data(), use_avx);

  // TODO
  // Add loops to support 4D tensor with batched matrix multiplication
  // Do I need to acquire some lock while executing?
  TRY_CATCH(ctx.map.at(key).execute(bfloatInputs.data(),
                                    std::make_tuple((int)ne11, (int)ne10),
                                    (float *)(dst->data)););
#endif
}
