#pragma once

/***************************************************
 * NO NOT INCLUDE NON-STD HEADERS HERE
 * KEEP FUNCTIONS HERE INDEPENDENT OF XCOMPILER IMPL
 **************************************************/
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace ryzenai {
namespace xcom {

struct WGTShuffleParam {
  std::int32_t input_ic_;
  std::int32_t output_c_;
  std::int32_t align_oc_;
  std::int32_t align_ic_;
  std::int32_t kernel_h_;
  std::int32_t kernel_w_;
  std::int32_t align_kernel_w_;
  std::int32_t stride_w_;
  std::int32_t pad_left;
  std::int32_t chl_augmentation_opt;
  std::int32_t mode;
  std::int32_t oc_mt;
  std::int32_t oc_per_aie;
  std::int32_t ic_per_aie;
  std::int32_t OCPf;
  std::int32_t BIAS_DUP_NUM;
  std::int32_t iter_ocg_;
  std::int32_t iter_icg_;
  std::int32_t tile_ocg_;
  std::int32_t tile_icg_;
  std::int32_t OCp;
  std::int32_t ICp;
  std::int32_t is_first_conv;
  std::int32_t split_num;
  std::int32_t RowNum;
  std::int32_t enable_col_num;
  std::int32_t x_zp;
  std::int32_t y_zp;
  std::int32_t w_zp;
  std::int32_t prelu_in;
  std::int32_t prelu_shift;
  std::int32_t tile_scale;
  std::int32_t in_width;
  std::int32_t wgt_width;
  float x_s;
  float y_s;
  float w_s;
  bool wgt_is_int8;
  bool wgt_is_uint8;
  bool is_prelu;
  bool is_fused_with_tile = false;
};

static_assert(sizeof(WGTShuffleParam) == 152);

/**
 * @brief Compile-time data shuffling for qlinear conv.
 *
 * This function takes three buffers and one param struct
 * as input and returns the shuffled data buffer.
 *
 * @param[in] wgt_orig The unshuffled weight.
 * @param[in] bias_data The bias.
 * @param[in] info_buf The param buffer for kernel.
 * @return The shuffled buffer including wgt, bias, param, qdq coeff.
 */
std::vector<char> qdq_conv_data_shuffle(std::vector<char> &wgt_orig,
                                        std::vector<std::int32_t> &bias_data,
                                        std::vector<std::int32_t> &info_buf,
                                        WGTShuffleParam &param);

} // namespace xcom
} // namespace ryzenai
