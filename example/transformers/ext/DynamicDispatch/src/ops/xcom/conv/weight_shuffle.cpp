#include <ops/xcom/conv/weight_shuffle.hpp>

#include <cassert>
#include <cmath>
#include <fstream>
#include <limits>

namespace ryzenai {
namespace xcom {

// TODO: Those helper functions are copied from other headers to break
// dependency, ideally some standalone header should hold these definitions.

/// Helper functions begin
template <typename T> void row2col(T *A, int dst_h, int dst_w) {
  std::vector<T> B(dst_h * dst_w);
  std::copy_n(A, B.size(), B.data());
  for (int idx_h = 0; idx_h < dst_h; idx_h++) {
    for (int idx_w = 0; idx_w < dst_w; idx_w++) {
      int src_addr = idx_w * dst_h + idx_h;
      int dst_addr = idx_h * dst_w + idx_w;
      A[dst_addr] = B[src_addr];
    }
  }
}
std::vector<char>
get_char_vec_from_int_copied(const std::vector<std::int32_t> &vec_int) {
  std::vector<char> vec_char(vec_int.size() * sizeof(std::int32_t), 0);
  std::memcpy(vec_char.data(), vec_int.data(), vec_char.size());
  return vec_char;
}
std::int32_t get_int32_representation_from_float_copied(float value) {
  uint32_t floatBits;
  std::memcpy(&floatBits, &value, sizeof(value));
  uint32_t exponent = (floatBits >> 23) & 0xFF;
  uint32_t mantissa = floatBits & 0x7FFFFF;
  // If the mantissa and exponent are equal to 0, return 0 directly.
  if ((0 == exponent) && (0 == mantissa))
    return 0;
  uint32_t result = 0;
  result |= (floatBits & 0x80000000);
  result = result >> 1;
  if (value < 0) {
    result |= (~((1 << 30) | ((floatBits & 0x007FFFFF) << 7)) + 1);
  } else {
    result |= ((1 << 30) | ((floatBits & 0x007FFFFF) << 7));
  }
  std::int32_t result_int = 0;
  std::memcpy(&result_int, &result, sizeof(result));
  return result_int;
}
std::int32_t convert_float_to_qint_copied(float in_f) {
  std::int32_t ret{0};
  std::memcpy(&ret, &in_f, sizeof in_f);
  ret &= 0x7fffffff; // Remove sign bit
  return ret;
}
std::int32_t get_shift_from_int32_rep_copied(std::int32_t rep) {
  std::int32_t shift =
      127 - (((rep >> 23) & 255) + 1) + (8 * sizeof(std::int32_t) - 1);
  return shift;
}
int64_t calculate_conv_shift(const int &a_bit_width, const int &w_bit_width,
                             int64_t K) {
  int64_t shift = 0;
  assert((a_bit_width + w_bit_width <= 24) && "Unsupported bit_width");
  if (a_bit_width + w_bit_width == 24) {
    // 0 < K < 32768
    shift = std::min(std::max(24 + int(std::ceil(std::log2(K))) - 32, 0), 7);
  }
  return shift;
}
struct QdqConvHeader {
  std::int32_t qdq_m = 0;
  std::int32_t qdq_n = 0;
  std::int32_t qdq_addr_config = 0;
  std::int32_t conv_shift = 0;
  std::int32_t idx_4_placeholder = 0;
  std::int32_t idx_5_placeholder = 0;
  std::int32_t shift_out = 0;
  std::int32_t idx_7_placeholder = 0;
  std::int64_t c0_front = 0;
  std::int32_t c1_front = 0;
  std::int32_t c2_front = 0;
  std::int32_t prelu_cofficient = 0;
  std::int32_t idx_13_placeholder = 0;
  std::int32_t idx_14_placeholder = 0;
  std::int32_t idx_15_placeholder = 0;
};
const std::int32_t QDQ_HEADER_BYTES = 64;
template <typename Dtype>
std::vector<Dtype> restreamize(const std::vector<char> &ori) {
  std::vector<Dtype> ret;
  auto vec_dtype_len = ori.size() * sizeof(char) / sizeof(Dtype);
  ret.reserve(vec_dtype_len);
  const Dtype *dtype_ptr = reinterpret_cast<const Dtype *>(ori.data());
  for (auto idx = 0U; idx < vec_dtype_len; idx++) {
    ret.push_back(dtype_ptr[idx]);
  }
  return ret;
}
std::vector<char> convert_qdq_conv_header_struct_to_char_vec_copied(
    QdqConvHeader &qdq_conv_header) {
  // TODO: Hack and convert struct to vector to bypass padding issue for memcpy
  auto const qdq_conv_header_ptr = reinterpret_cast<char *>(&qdq_conv_header);
  std::vector<char> qdq_conv_header_vec(qdq_conv_header_ptr,
                                        qdq_conv_header_ptr + QDQ_HEADER_BYTES);
  return qdq_conv_header_vec;
}
/// Helper functions end

std::vector<char> qdq_conv_data_shuffle(std::vector<char> &wgt_orig,
                                        std::vector<std::int32_t> &bias_data,
                                        std::vector<std::int32_t> &info_buf,
                                        WGTShuffleParam &param) {
  // std::ofstream fout;
  // fout.open("wgt_params.bin", std::ios::out | std::ios::binary);
  // fout.write(reinterpret_cast<const char *>(&param),
  // sizeof(WGTShuffleParam)); fout.close(); fout.clear();
  //
  //  Step 1: Shuffle wgt based on alignment, and reduce in OC
  //
  std::vector<char> aligned_wgt_ddr;
  aligned_wgt_ddr.resize(param.align_oc_ * param.kernel_h_ *
                             param.align_kernel_w_ * param.align_ic_,
                         0);
  if (param.is_fused_with_tile) {
    for (std::int32_t scale_idx = 0; scale_idx < param.tile_scale;
         scale_idx++) {
      std::int32_t src_oc_block = param.output_c_ / param.tile_scale;
      std::int32_t dst_oc_block = param.align_oc_ / param.tile_scale;
      for (int idx_oc = 0; idx_oc < src_oc_block; idx_oc++) {
        for (int idx_ih = 0; idx_ih < param.kernel_h_; idx_ih++) {
          for (int idx_iw = 0; idx_iw < param.kernel_w_; idx_iw++) {
            size_t src_idx =
                (scale_idx * src_oc_block + idx_oc) *
                    (param.kernel_h_ * param.kernel_w_ * param.input_ic_) +
                idx_ih * (param.kernel_w_ * param.input_ic_) +
                idx_iw * param.input_ic_;
            size_t dst_idx =
                (scale_idx * dst_oc_block + idx_oc) *
                    (param.kernel_h_ * param.align_kernel_w_ *
                     param.align_ic_) +
                idx_ih * (param.align_kernel_w_ * param.align_ic_) +
                idx_iw * param.align_ic_;
            assert(src_idx < wgt_orig.size() && "Conv weights shuffle failed");
            assert(dst_idx < aligned_wgt_ddr.size() &&
                   "Conv weights shuffle failed");
            std::copy_n(wgt_orig.data() + src_idx, param.input_ic_,
                        aligned_wgt_ddr.data() + dst_idx);
          }
        }
      }
    }
  } else {
    for (int idx_oc = 0; idx_oc < param.output_c_; idx_oc++) {
      for (int idx_ih = 0; idx_ih < param.kernel_h_; idx_ih++) {
        for (int idx_iw = 0; idx_iw < param.kernel_w_; idx_iw++) {
          auto ic_offset =
              (param.chl_augmentation_opt && param.pad_left % 2 != 0)
                  ? param.align_ic_
                  : 0;
          size_t src_idx =
              idx_oc * (param.kernel_h_ * param.kernel_w_ * param.input_ic_) +
              idx_ih * (param.kernel_w_ * param.input_ic_) +
              idx_iw * param.input_ic_;
          size_t dst_idx = idx_oc * (param.kernel_h_ * param.align_kernel_w_ *
                                     param.align_ic_) +
                           idx_ih * (param.align_kernel_w_ * param.align_ic_) +
                           idx_iw * param.align_ic_ + ic_offset;
          assert(src_idx < wgt_orig.size() && "Conv weights shuffle failed");
          assert(dst_idx < aligned_wgt_ddr.size() &&
                 "Conv weights shuffle failed");
          std::copy_n(wgt_orig.data() + src_idx, param.input_ic_,
                      aligned_wgt_ddr.data() + dst_idx);
        }
      }
    }
  }

  // Reduce weight data along oc

  auto wgt_reduced_func = [&](const auto &aligned_wgt_ddr) {
    std::vector<float> wgt_reduced(param.align_oc_, 0);
    for (std::int32_t idx_oc = 0; idx_oc < param.output_c_; idx_oc++) {
      // reduce for each oc
      float sum_oc = 0;
      for (std::int32_t idx_ih = 0; idx_ih < param.kernel_h_; idx_ih++) {
        for (std::int32_t idx_iw = 0; idx_iw < param.kernel_w_; idx_iw++) {
          std::int32_t dst_idx =
              idx_oc *
                  (param.kernel_h_ * param.align_kernel_w_ * param.align_ic_) +
              idx_ih * (param.align_kernel_w_ * param.align_ic_) +
              idx_iw * param.align_ic_;
          for (std::int32_t idx_ic = 0; idx_ic < param.input_ic_; idx_ic++) {
            sum_oc += aligned_wgt_ddr[dst_idx + idx_ic];
          }
        }
      }
      wgt_reduced[idx_oc] = sum_oc;
    }
    return wgt_reduced;
  };

  std::vector<float> wgt_reduced;
  if (param.wgt_is_int8) {
    wgt_reduced = wgt_reduced_func(restreamize<char>(aligned_wgt_ddr));

  } else if (param.wgt_is_uint8) {
    wgt_reduced = wgt_reduced_func(restreamize<unsigned char>(aligned_wgt_ddr));
  }

  //
  // Step 2: Prepare bias and param data
  //

  bias_data.resize(param.align_oc_, 0);
  if (param.is_fused_with_tile) {
    std::vector<std::int32_t> tile_bias_shuffle(param.align_oc_, 0);
    std::int32_t src_oc_block = param.output_c_ / param.tile_scale;
    std::int32_t dst_oc_block = param.align_oc_ / param.tile_scale;
    for (std::int32_t scale_idx = 0; scale_idx < param.tile_scale;
         scale_idx++) {
      size_t src_idx = scale_idx * src_oc_block;
      size_t dst_idx = scale_idx * dst_oc_block;
      std::copy_n(bias_data.data() + src_idx, src_oc_block,
                  tile_bias_shuffle.data() + dst_idx);
    }
    bias_data = tile_bias_shuffle;
  }
  std::int32_t ddr2mt_oc = param.align_oc_ / param.oc_mt;

  std::int32_t info_buf_byte_size = info_buf.size() * sizeof(std::int32_t);

  //
  // Step 3: Compute QDQ coefficients
  // Compute coefficient 2 (new formula for 2 term)
  float c2_f = param.y_s / (param.x_s * param.w_s);
  std::vector<float> c2_vec_ocpf_float(param.OCPf, c2_f);
  std::vector<std::int32_t> c2_vec_ocpf_int(param.OCPf, 0);

  for (std::size_t idx = 0; idx < c2_vec_ocpf_int.size(); idx++) {
    c2_vec_ocpf_int[idx] =
        get_int32_representation_from_float_copied(c2_vec_ocpf_float[idx]);
    if (param.w_zp != 0) {
      c2_vec_ocpf_int[idx] = c2_vec_ocpf_int[idx] >> 7;
    }
  }
  std::vector<char> c2_vec_char = get_char_vec_from_int_copied(c2_vec_ocpf_int);
  std::int32_t c2_int = get_int32_representation_from_float_copied(c2_f);
  std::int32_t c2_qint = convert_float_to_qint_copied(c2_f);
  std::int32_t shift_out = get_shift_from_int32_rep_copied(c2_qint);
  std::int32_t single_filter_size =
      param.kernel_h_ * param.kernel_w_ * param.input_ic_;
  // calculate the conv shift by input width, weights width and single filter
  // size.
  auto conv_shift =
      calculate_conv_shift(param.in_width, param.wgt_width, single_filter_size);
  std::int64_t c2_int_64 = c2_int;
  if (param.w_zp != 0) {
    c2_int_64 = c2_int >> 7;
    shift_out -= 7;
    if (-c2_int_64 * param.w_zp < INT32_MIN ||
        -c2_int_64 * param.w_zp > INT32_MAX) {
      c2_int_64 = c2_int_64 >> 1;
      shift_out = shift_out - 1;
    }
  }
  QdqConvHeader qdq_header_conv;
  // TODO: what does 4 mean for conv?
  qdq_header_conv.qdq_addr_config = 4;

  // Update shift out in qdq header
  qdq_header_conv.shift_out = shift_out;
  ;
  // Update C2 front in qdq header
  qdq_header_conv.c2_front = c2_int_64 << conv_shift;
  // Update C1 front in qdq header
  qdq_header_conv.c1_front = -param.w_zp * c2_int_64;
  // Update conv shift in qdq header
  qdq_header_conv.conv_shift = conv_shift;
  if (param.is_prelu) {
    qdq_header_conv.prelu_cofficient =
        ((std::int64_t)c2_int * param.prelu_in >> param.prelu_shift);
  }
#ifdef XCOM_DEBUG_MODE
  bool limit =
      (-param.w_zp * c2_int_64 >= std::numeric_limits<int32_t>::min()) &&
      (-param.w_zp * c2_int_64 <= std::numeric_limits<int32_t>::max());
  assert(limit && "Only supports int32 C1,");
#endif

  std::vector<std::int32_t> c1_vec_ocpf_int(param.OCPf,
                                            -param.w_zp * c2_int_64);
  std::vector<char> c1_vec_char = get_char_vec_from_int_copied(c1_vec_ocpf_int);

  //
  // Step 4: Compute bias coefficients and shuffle weights
  //         Returned data includes qdq header, coefficients, and weights
  //

  // Aggregate the final returned size
  std::int32_t total_wgt_size = aligned_wgt_ddr.size() + info_buf_byte_size;
  // Compute actual bias size
  std::int32_t bias_size_per_copy = param.oc_per_aie * param.BIAS_DUP_NUM;
  // Type is changed from int8 to int32 compared to conv2d-fix
  bias_size_per_copy *= sizeof(std::int32_t);
  // QDQ header has 16 ints, don't use sizeof on struct due to padding
  bias_size_per_copy += QDQ_HEADER_BYTES;
  std::int32_t bias_size_total =
      ddr2mt_oc * param.iter_ocg_ * param.iter_icg_ * bias_size_per_copy;
  total_wgt_size += bias_size_total;
  std::vector<char> shuffled_wgt_ddr(total_wgt_size, 0); // return data

  // Step 4a: copy param (info_buf, only copied once)
  std::size_t idx_dst = 0;
  memcpy(shuffled_wgt_ddr.data() + idx_dst, info_buf.data(),
         info_buf_byte_size);
  idx_dst += info_buf_byte_size;

  std::int32_t one_kernel_size =
      param.kernel_h_ * param.align_kernel_w_ * param.align_ic_;
  std::int32_t wreg_size = param.OCPf * param.ICp;
  std::int32_t oc_blk_loop = param.align_oc_ / param.oc_mt;
  std::int32_t kernel_w_stride = param.is_first_conv ? param.stride_w_ : 1;

  std::int32_t split_iter_icg = param.iter_icg_ / param.split_num;
  std::int32_t align_num_by_core = 1;
  std::int32_t align_num_by_col = 1;
  std::int32_t align_oc_per_col = 0;
  if (param.mode == 0 || param.mode == 3) {
    align_num_by_core = param.RowNum;
    align_num_by_col = param.enable_col_num;
    align_oc_per_col = param.align_oc_ / param.enable_col_num;
    param.oc_mt = param.oc_mt * param.RowNum;
    oc_blk_loop = align_oc_per_col / param.oc_mt;
  }

  for (std::int32_t idx_col = 0; idx_col < align_num_by_col; idx_col++) {
    for (std::int32_t idx_oc_blk = 0; idx_oc_blk < oc_blk_loop;
         idx_oc_blk++) { // oc_blk_loop
      for (std::int32_t idx_icpartition = 0; idx_icpartition < param.split_num;
           idx_icpartition++) {
        for (std::int32_t idx_core = 0; idx_core < align_num_by_core;
             idx_core++) {
          for (std::int32_t idx_iter_ocg = 0; idx_iter_ocg < param.iter_ocg_;
               idx_iter_ocg++) {
            // basic aietile block
            std::size_t ocbase =
                (idx_col * align_oc_per_col + idx_oc_blk * param.oc_mt +
                 idx_iter_ocg * param.oc_per_aie +
                 idx_core * param.iter_ocg_ * param.oc_per_aie) *
                one_kernel_size;
            std::size_t bias_addr =
                idx_col * align_oc_per_col + idx_oc_blk * param.oc_mt +
                idx_iter_ocg * param.oc_per_aie +
                idx_core * param.iter_ocg_ * param.oc_per_aie;
            assert(bias_addr + param.oc_per_aie <= bias_data.size() &&
                   "Conv weights shuffle failed");
            for (std::int32_t idx_iter_icg = 0; idx_iter_icg < split_iter_icg;
                 idx_iter_icg++) {
              assert(idx_dst + param.oc_per_aie * param.BIAS_DUP_NUM <=
                         shuffled_wgt_ddr.size() &&
                     "Conv weights shuffle failed.");
              // Step 4c_1: Compute QDQ header: c0_front
              {
                for (std::int32_t idx_dup = 0; idx_dup < param.oc_per_aie;
                     idx_dup += param.OCPf) {
                  // For each OCPf (8) OC, keep the compute here for
                  // channel-wise
                  // Get bias data
                  auto bias_data_ocpf = std::vector<std::int32_t>(
                      bias_data.begin() + bias_addr + idx_dup,
                      bias_data.begin() + bias_addr + idx_dup + param.OCPf);
                  // Get wgt data
                  auto wgt_data_ocpf = std::vector<std::int32_t>(
                      wgt_reduced.begin() + idx_dup +
                          idx_iter_ocg * param.oc_per_aie +
                          idx_core * param.iter_ocg_ * param.oc_per_aie +
                          idx_oc_blk * param.oc_mt,
                      wgt_reduced.begin() + idx_dup +
                          idx_iter_ocg * param.oc_per_aie +
                          idx_core * param.iter_ocg_ * param.oc_per_aie +
                          idx_oc_blk * param.oc_mt + param.OCPf);
                  std::int64_t c_wgt = std::int64_t(param.kernel_h_) *
                                       param.kernel_w_ * param.input_ic_ *
                                       int64_t(param.x_zp) * param.w_zp;
                  std::int64_t first_c0_int =
                      -c2_int_64 * (param.x_zp * wgt_data_ocpf[0] - c_wgt -
                                    bias_data_ocpf[0]) +
                      (param.y_zp << shift_out);
                  qdq_header_conv.c0_front = first_c0_int;
                }
              }

              // Copy qdq header for each aie tile
              std::vector<char> qdq_header_vec =
                  convert_qdq_conv_header_struct_to_char_vec_copied(
                      qdq_header_conv);
              std::copy_n(qdq_header_vec.data(), qdq_header_vec.size(),
                          shuffled_wgt_ddr.data() + idx_dst);
              idx_dst += qdq_header_vec.size();

              // Step 4c_2, compute coefficients and copy to buffer
              for (std::int32_t idx_dup = 0; idx_dup < param.oc_per_aie;
                   idx_dup += param.OCPf) {
                // Compute C0, for two-term symmetric wgts
                std::vector<std::int64_t> c0_vec_int64(param.OCPf, 0);
                // Compute C3, for leakyrelu
                std::vector<std::int64_t> c3_vec_int64(param.OCPf, 0);
                {
                  std::int64_t c_wgt = int64_t(param.kernel_h_) *
                                       param.kernel_w_ * param.input_ic_ *
                                       param.x_zp * param.w_zp;
                  // Get bias data
                  auto bias_data_ocpf = std::vector<std::int32_t>(
                      bias_data.begin() + bias_addr + idx_dup,
                      bias_data.begin() + bias_addr + idx_dup + param.OCPf);
                  // Get wgt data
                  auto wgt_data_ocpf = std::vector<std::int32_t>(
                      wgt_reduced.begin() + idx_dup +
                          idx_iter_ocg * param.oc_per_aie +
                          idx_core * param.iter_ocg_ * param.oc_per_aie +
                          idx_oc_blk * param.oc_mt + idx_col * align_oc_per_col,
                      wgt_reduced.begin() + idx_dup +
                          idx_iter_ocg * param.oc_per_aie +
                          idx_core * param.iter_ocg_ * param.oc_per_aie +
                          idx_oc_blk * param.oc_mt +
                          idx_col * align_oc_per_col + param.OCPf);
                  // Compute OCPf C0
                  for (std::int32_t i_oc = 0; i_oc < param.OCPf; i_oc++) {
                    std::int64_t new_y_zp = int64_t(param.y_zp) << shift_out;
                    std::int64_t tmp_value_a =
                        -int64_t(param.x_zp) * wgt_data_ocpf[i_oc] + c_wgt +
                        bias_data_ocpf[i_oc];
                    std::int64_t tmp_value_b = c2_int_64 * tmp_value_a;
                    std::int64_t c0_int = tmp_value_b + new_y_zp;
                    c0_vec_int64[i_oc] = c0_int;
                    std::int64_t c3_int = tmp_value_b * param.prelu_in +
                                          (new_y_zp << param.prelu_shift);
                    c3_vec_int64[i_oc] = c3_int >> param.prelu_shift;
                  }
                }
                // Copy C0
                std::vector<char> c0_vec_char(param.OCPf * sizeof(std::int64_t),
                                              0);
                std::vector<char> c3_vec_char(param.OCPf * sizeof(std::int64_t),
                                              0);
                std::memcpy(c0_vec_char.data(), c0_vec_int64.data(),
                            param.OCPf * sizeof(std::int64_t));
                std::copy_n(c0_vec_char.data(), c0_vec_char.size(),
                            shuffled_wgt_ddr.data() + idx_dst);
                idx_dst += c0_vec_char.size();
                if (param.is_prelu) {
                  // Copy C3, which has been computed
                  std::memcpy(c3_vec_char.data(), c3_vec_int64.data(),
                              param.OCPf * sizeof(std::int64_t));
                  std::copy_n(c3_vec_char.data(), c3_vec_char.size(),
                              shuffled_wgt_ddr.data() + idx_dst);
                  idx_dst += c3_vec_char.size();
                } else {
                  // Copy C1, which has been computed
                  std::copy_n(c1_vec_char.data(), c1_vec_char.size(),
                              shuffled_wgt_ddr.data() + idx_dst);
                  idx_dst += c1_vec_char.size();
                  // Copy C2, which has been computed
                  std::copy_n(c2_vec_char.data(), c2_vec_char.size(),
                              shuffled_wgt_ddr.data() + idx_dst);
                  idx_dst += c2_vec_char.size();
                }
              }

              // Step 4c_3, Shuffle wgt data
              for (std::int32_t idx_tile_ocg = 0;
                   idx_tile_ocg < param.tile_ocg_; idx_tile_ocg++) {
                for (std::int32_t idx_tile_icg = 0;
                     idx_tile_icg < param.tile_icg_; idx_tile_icg++) {
                  for (std::int32_t idx_kh = 0; idx_kh < param.kernel_h_;
                       idx_kh++) {
                    for (std::int32_t idx_kw = 0;
                         idx_kw < param.align_kernel_w_;
                         idx_kw += kernel_w_stride) {
                      for (std::int32_t idx_ocpf = 0; idx_ocpf < 2;
                           idx_ocpf++) {
                        std::size_t icbase;
                        icbase = idx_icpartition *
                                     (param.align_ic_ / param.split_num) +
                                 idx_iter_icg * param.ic_per_aie +
                                 idx_tile_icg * param.ICp;
                        std::vector<std::int8_t> tmp(wreg_size);
                        std::int32_t idx_tmp = 0;
                        for (std::int32_t idx_ocp = 0; idx_ocp < param.OCPf;
                             idx_ocp++) {
                          std::size_t addr =
                              ocbase +
                              (idx_tile_ocg * param.OCp +
                               idx_ocpf * param.OCPf + idx_ocp) *
                                  one_kernel_size +
                              (idx_kh * param.align_kernel_w_ + idx_kw) *
                                  param.align_ic_ +
                              icbase;
                          assert(addr + param.ICp <= aligned_wgt_ddr.size() &&
                                 "Conv weights shuffle failed.");
                          std::copy_n(aligned_wgt_ddr.data() + addr, param.ICp,
                                      tmp.data() + idx_tmp);
                          idx_tmp += param.ICp;
                        }
                        row2col(tmp.data(), param.ICp, param.OCPf);
                        assert(idx_dst + wreg_size <= shuffled_wgt_ddr.size() &&
                               "Conv weights shuffle failed.");
                        std::copy_n(tmp.data(), tmp.size(),
                                    shuffled_wgt_ddr.data() + idx_dst);
                        idx_dst += wreg_size;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // fout.open("shuffled_wgt.bin", std::ios::out | std::ios::binary);
  // fout.write(shuffled_wgt_ddr.data(), shuffled_wgt_ddr.size());
  // fout.close();
  return shuffled_wgt_ddr;
}

} // namespace xcom
} // namespace ryzenai
