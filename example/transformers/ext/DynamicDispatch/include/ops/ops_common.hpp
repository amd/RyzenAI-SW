#pragma once
#include <tuple>
#include <vector>

namespace ryzenai {
struct matrix_shapes {
  // capture M, K, N of the shape supported.
  int64_t M;
  int64_t K;
  int64_t N;

  matrix_shapes(int64_t M, int64_t K, int64_t N) : M(M), K(K), N(N) {}
};

struct conv_shapes {
  /* This shape only supports square filter dimention i.e. FxF */
  // capture zp, F, K, N of the shape supported.
  int64_t Z; /* Zero point */
  int64_t F; /* Filter size : Typically F x F*/
  int64_t K; /* Number of input channels */
  int64_t N; /* Number of output channels */

  conv_shapes(int64_t Z, int64_t F, int64_t K, int64_t N)
      : Z(Z), F(F), K(K), N(N) {}
};

struct mladf_matrix_shapes {
  // capture M, K, N, Gs of the shape supported.
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t Gs;

  mladf_matrix_shapes(int64_t M, int64_t K, int64_t N, int64_t Gs = 0)
      : M(M), K(K), N(N), Gs(Gs) {}
};

const std::string PSF_A8W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psf_model_a8w8_qdq.xclbin";
const std::string PSJ_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psj_model_a16w8_qdq.xclbin";
const std::string PSH_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psh_model_a16w8_qdq.xclbin";
const std::string PSI_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psi_model_a16w8_qdq.xclbin";
const std::string PSQ_A8W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psq_model_a8w8_qdq.xclbin";
const std::string PSQ2_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psq2_model_a16w8_qdq.xclbin";
const std::string PSR_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psr_model_a16w8_qdq.xclbin";
const std::string PSS_A16A16_QDQ_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x2_matmul_softmax_a16w16.xclbin";
const std::string PST_A16A16_QDQ_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x2_pss_a16a16_qdq.xclbin";
const std::string MLADF_4x4_GEMM_SILU_MUL_A16FW4_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x4_gemm_silu_mul_a16fw4.xclbin";
const std::string PSR4x4_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x4_psr_model_a16w8_qdq.xclbin";
const std::string MLADF_GEMM_4x4_A16FW4ACC16F_XCLBIN_PATH =
    "/xclbin/stx/mladf_gemm_4x4_a16fw4acc16f.xclbin";
const std::string MLADF_2x4x4_MASKEDSOFTMAX_A16F_XCLBIN_PATH =
    "/xclbin/stx/mladf_2x4x4_maskedsoftmax_a16f.xclbin";
const std::string BMM_A16W16_65536_128_2048_XCLBIN_PATH =
    "/xclbin/stx/2x4x4_bmm_model_a16w16_65536_128_2048.xclbin";
const std::string BMM_A16W16_65536_2048_128_XCLBIN_PATH =
    "/xclbin/stx/2x4x4_bmm_model_a16w16_65536_2048_128.xclbin";
const std::string MLADF_4x2_GEMM_A16W8_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x2_gemm_a16w8_qdq.xclbin";
const std::string MLADF_4x2_GEMM_A16W16_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x2_gemm_a16w16_qdq.xclbin";
const std::string BMM_A16W16_XCLBIN_PATH =
    "/xclbin/stx/2x4x4_bmm_model_a16w16.xclbin";
const std::string XCOM_4x4_XCLBIN_PATH = "/xclbin/stx/4x4_dpu.xclbin";
const std::string XCOM_4x4_Q_XCLBIN_PATH =
    "/xclbin/stx/4x4_dpu_qconv_qelew_add.xclbin";
const std::string MLADF_SOFTMAX_A16_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x2_matmul_softmax_a16w16.xclbin";
const std::string
    LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH =
        "/xclbin/stx/llama2_mladf_2x4x4_gemmbfp16_silu_mul_mha_rms_rope.xclbin";
const std::string MLADF_4x2_ELWADD_A16W16_QDQ_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x2_add_a16.xclbin";
const std::string MLADF_4x2_ELWMUL_A16_QDQ_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x2_mul_a16.xclbin";

namespace utils {
template <class Tuple,
          class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<T> tuple_to_vector(Tuple &&tuple) {
  return std::apply(
      [](auto &&...elems) {
        return std::vector<T>{std::forward<decltype(elems)>(elems)...};
      },
      std::forward<Tuple>(tuple));
}

template <class Types>
Types running_product_with_skips(const std::vector<Types> &nums,
                                 const std::vector<size_t> &skip_indexes = {}) {
  Types product_of_all =
      std::accumulate(nums.begin(), nums.end(), 1LL, std::multiplies<Types>());
  Types product_of_skips{1};
  if (skip_indexes.size() != 0) {
    product_of_skips =
        std::accumulate(skip_indexes.begin(), skip_indexes.end(), 1LL,
                        [&](Types acc, size_t index) {
                          return index < nums.size() ? acc * nums[index] : acc;
                        });
  }

  return product_of_all / product_of_skips;
}

template <class Types>
Types max_element_count_with_skips(
    const std::vector<std::vector<Types>> &supported_shapes,
    const std::vector<size_t> skip_indexes = {}) {
  auto max_product_iter =
      max_element(supported_shapes.begin(), supported_shapes.end(),
                  [&](const auto &t1, const auto &t2) {
                    return running_product_with_skips(t1, skip_indexes) <
                           running_product_with_skips(t2, skip_indexes);
                  });
  return running_product_with_skips(*max_product_iter, skip_indexes);
}

} // namespace utils
} // namespace ryzenai
