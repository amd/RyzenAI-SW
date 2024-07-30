/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <any>
#include <cassert>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include <utility>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include <utils/dpu_mdata.hpp>

// Subroutines to read the transaction binary
#include "op_buf.hpp"
#include "op_types.h"

#include "utils/txn_container.hpp"
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>
#include <ops/op_interface.hpp>
#include <utils/instruction_registry.hpp>
#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/txn_container.hpp>
#include <utils/utils.hpp>
#include <xrt_context/xrt_context.hpp>

#include "ops/ops_common/mladf_matmul_matrix.hpp"

namespace ryzenai {

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
std::string mladfmatmulbias<InT, WtT, AccT, OutT>::get_instr_key(
    std::string prefix, int m, int k, int n, int grp_size) {
  if (grp_size)
    return "mladfmatmulbias_" + prefix + "_" + std::to_string(m) + "_" +
           std::to_string(k) + "_" + std::to_string(n) + "_" +
           std::to_string(grp_size);
  else
    return "mladfmatmulbias_" + prefix + "_" + std::to_string(m) + "_" +
           std::to_string(k) + "_" + std::to_string(n);
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<mladf_matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N, mat.Gs);
    instructions.push_back(std::make_pair(key, false));
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
}

template <typename InT, typename WtT, typename AccT, typename OutT>
mladfmatmulbias<InT, WtT, AccT, OutT>::mladfmatmulbias(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt) {
  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  initialized_ = false;
  DPU_DIR = OpInterface::get_dod_base_dir() + "//transaction//" + "stx" +
            "//mladfmatmulbias//";
  txnbin_a_header = {{"bfloat16", "a16f"}};
  txnbin_b_header = {{"uint4", "w4"}};
  txnbin_acc_header = {{"bfloat16", "acc16f"}};

  default_shapes_ = {
      {"mladf_2x4x4_a16fw4acc16f",
       {
           mladf_matrix_shapes(1, 4096, 4096, 128),
           mladf_matrix_shapes(1, 4096, 11008, 128),
           mladf_matrix_shapes(1, 4096, 12288, 128),
           mladf_matrix_shapes(1, 4096, 22528, 128),
           mladf_matrix_shapes(1, 4096, 32768, 32),
           mladf_matrix_shapes(1, 11008, 4096, 128),
           mladf_matrix_shapes(128, 4096, 4096, 128),
           mladf_matrix_shapes(128, 4096, 11008, 128),
           mladf_matrix_shapes(128, 4096, 12288, 128),
           mladf_matrix_shapes(128, 4096, 22528, 128),
           mladf_matrix_shapes(128, 4096, 32768, 32),
           mladf_matrix_shapes(128, 11008, 4096, 128),
           mladf_matrix_shapes(2048, 4096, 4096, 128),
           mladf_matrix_shapes(2048, 4096, 11008, 128),
           mladf_matrix_shapes(2048, 4096, 12288, 128),
           mladf_matrix_shapes(2048, 4096, 22528, 128),
           mladf_matrix_shapes(2048, 4096, 32768, 32),
           mladf_matrix_shapes(2048, 11008, 4096, 128),
       }},
  };

  /* Set the appropriate kernel group id based on the DPU sequence execution
   * flow or transaction binary flow The transaction binary flow is enabled
   * only
   * for w4a16 and w3a16 GEMMs
   */
  if (a_dtype_ == "bfloat16") {
    a_dtype_size_ = sizeof(uint16_t);
  }
  if (b_dtype == "uint4") {
    sign = 0;
  } else {
    sign = 1;
  }

  mladfmatmulbias_id_ = mladfmatmulbias_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dod_base_dir() +
      ryzenai::LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH;
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("xclbin fname : {}", XCLBIN_FNAME));

  txn_fname_prefix_ = "mladf_2x4x4_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("txn_fname_prefix : {}", txn_fname_prefix_));

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  // superkernel parameters not set through SHIM DDR
  params_bytes = 0;
  KERNEL_M_MAX = 32; // m dim of design is 32; initialize_weights() uses
                     // KERNEL_M_MAX by default

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header =
        "mladfmatmulbias_id M K N kernel_m kernel_k kernel_n Execute"
        "time(ns) num_aie_runs run_aie_time(ns) A_Pad_time(ns) "
        "C_Pad_time(ns) C_depad_time(ns) A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) CPU_accum_time(ns) "
        "Avg_time_per_aie_run(ns) group_size\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE(
      "[MLADFMATMULBIAS] ID: " + std::to_string(mladfmatmulbias_id_) +
      ", XCLBIN: " + XCLBIN_FNAME + ", (a_dtype, b_dtype, c_dtype): (" +
      a_dtype_ + ", " + b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::set_kernel_shapes_kn_mladf() {
  if (a_dtype_ != "bfloat16") {
    /*Current support is only for Int8 and Bf16 activation type*/
    throw std::runtime_error(
        "No Kernel exists for the current activation data type");
  }
  if (a_dtype_ == "bfloat16") {
    KERNEL_M_MAX = 2048;
    if (w_shape_[0] > 4096) {
      // Use 11008 x 4096 as the kernel
      kernel_x_shape_[1] = 11008;
      kernel_y_shape_[0] = 11008;
      kernel_y_shape_[1] = 4096;
      kernel_z_shape_[1] = 4096;
    } else if (w_shape_[1] > 22528) {
      // Use 4096 x 32768 as the kernel
      kernel_x_shape_[1] = 4096;
      kernel_y_shape_[0] = 4096;
      kernel_y_shape_[1] = 32768;
      kernel_z_shape_[1] = 32768;
    } else if (w_shape_[1] > 12288) {
      // Use 4096 x 22528 as the kernel
      kernel_x_shape_[1] = 4096;
      kernel_y_shape_[0] = 4096;
      kernel_y_shape_[1] = 22528;
      kernel_z_shape_[1] = 22528;
    } else if (w_shape_[1] > 4096) {
      // Use 4096 x 11008 as the kernel
      kernel_x_shape_[1] = 4096;
      kernel_y_shape_[0] = 4096;
      kernel_y_shape_[1] = 12288;
      kernel_z_shape_[1] = 12288;
    } else if (w_shape_[1] > 2048) {
      // Use 4096 x 4096 as the kernel
      kernel_x_shape_[1] = 4096;
      kernel_y_shape_[0] = 4096;
      kernel_y_shape_[1] = 4096;
      kernel_z_shape_[1] = 4096;
    } else {
      // a debug kernel 8 x 256 x 2048
      kernel_x_shape_[1] = 256;
      kernel_y_shape_[0] = 256;
      kernel_y_shape_[1] = 2048;
      kernel_z_shape_[1] = 2048;
    }
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  // get original arguments from const Tensors
  int8_t *weights = (int8_t *)const_params.at(0).data;
  int8_t *zeros = (int8_t *)const_params.at(3).data;
  float *scales = (float *)const_params.at(2).data;
  float *bias = (float *)const_params.at(1).data;
  std::tuple<int, int> w_shape = {const_params.at(0).shape.at(0),
                                  const_params.at(0).shape.at(1)};

  int group_size = const_params.at(2).shape.at(0);
  // Note: for mladf int8 gemm we had to change group id to 0
  const int group_id = 0;
  w_shape_[0] = std::get<0>(w_shape);
  w_shape_[1] = std::get<1>(w_shape);

  set_kernel_shapes_kn_mladf();

  // Use largest M dimension as the default. This has to correspond
  // to one of the available kernel sizes.
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;

  /* Create input/output BOs */
  const int A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  const int C_BO_SIZE =
      (kernel_x_shape_[1] != 11008)
          ? (kernel_z_shape_[0] * kernel_z_shape_[1] * sizeof(OutT))
          : (kernel_z_shape_[0] * kernel_z_shape_[1] * sizeof(OutT) * 2);
  if (!initialized_) {

    a_bo_ =
        xrt::bo(xrt_ctx_->get_context(), A_BO_SIZE, xrt::bo::flags::host_only,
                xrt_ctx_->get_kernel().group_id(group_id));
    c_bo_ =
        xrt::bo(xrt_ctx_->get_context(), C_BO_SIZE, xrt::bo::flags::host_only,
                xrt_ctx_->get_kernel().group_id(group_id));

    const int A_BO_SIZE_TOKEN = (1 * kernel_x_shape_[1] * a_dtype_size_);
    const int C_BO_SIZE_TOKEN = (kernel_x_shape_[1] != 11008)
                                    ? 1 * kernel_z_shape_[1] * sizeof(OutT)
                                    : 1 * kernel_z_shape_[1] * sizeof(OutT) * 2;
    a_bo_token_ = xrt::bo(xrt_ctx_->get_context(), A_BO_SIZE_TOKEN,
                          xrt::bo::flags::host_only,
                          xrt_ctx_->get_kernel().group_id(group_id));

    c_bo_token_ = xrt::bo(xrt_ctx_->get_context(), C_BO_SIZE_TOKEN,
                          xrt::bo::flags::host_only,
                          xrt_ctx_->get_kernel().group_id(group_id));
    initialized_ = true;
  }

  /* Create weight BOs */
  // Create a BO for weight block and initialize to zero
  //    NOTE: We must initialize to zero here because the weight matrix
  //          shape might not be an integer multiple of the block size.
  //          Initializing the BOs to zero takes care of the padding
  //          without allocating any extra scratch space.

  // For int4 quantization the buffer also contains bias, zeros, and scales
  // the weights are tiled in zigzag w4 aligned subvolumes of 32x128 tiles
  // the first subvolume consists of bias that is padded with zeros
  // Rest of the subvolumes consist weights+scales+zeros in each tile
  // QuantMatrix class has helper functions to write the data into the
  // correct index

  w_padded_shape_[0] = Utils::ceil_for_me(w_shape_[0], kernel_y_shape_[0]);
  w_padded_shape_[1] = Utils::ceil_for_me(w_shape_[1], kernel_y_shape_[1]);
  // The bfp16 kernel uses a block size of 4 for any gemm shape.
  int blk_size = 4;

  mladfQuantMatrix<64, 32, 32, 32> buff_B1(kernel_y_shape_[0],
                                           kernel_y_shape_[1], blk_size);
  mladfQuantMatrix<64, 128, 32, 128> buff_B2(kernel_y_shape_[0],
                                             kernel_y_shape_[1], blk_size);
  // iterate over kernel shaped blocks of the weight matrix
  for (int64_t rb = 0; rb < w_padded_shape_[0]; rb += kernel_y_shape_[0]) {
    for (int64_t cb = 0; cb < w_padded_shape_[1]; cb += kernel_y_shape_[1]) {
      auto b_format_start = GET_ELAPSED_TIME_NS();

      int block_size =
          (group_size < 128) ? buff_B1.data_size : buff_B2.data_size;
      xrt::bo bo_;
      bo_ = xrt::bo(xrt_ctx_->get_context(), block_size,
                    xrt::bo::flags::host_only,
                    xrt_ctx_->get_kernel().group_id(group_id));
      auto bo_map = bo_.map<WtT *>();
      memset((void *)bo_map, 0, block_size);
      buff_B1.data = (mladfCoreSubv<32, 32, 32> *)bo_map;
      buff_B2.data = (mladfCoreSubv<128, 32, 128> *)bo_map;

      // first pack the bias (bf16)
      for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1]; ++c) {
        if (rb == 0) {
          (group_size < 128)
              ? buff_B1.bias(c) = ryzenai::float_to_bfloat16(bias[cb + c])
              : buff_B2.bias(c) = ryzenai::float_to_bfloat16(bias[cb + c]);
        }
      }
      // format quantized weights (int4/uint4)
      for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0]; ++r) {
        for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
             c += 2) {
          // NOTE: int8_t weights will be sign extended to int
          int x = weights[((rb + r) * w_shape_[1]) + (cb + c)];
          int y = weights[((rb + r) * w_shape_[1]) + (cb + c) + 1];
          if (b_dtype_ == "int4") {
            (group_size < 128)
                ? buff_B1.quant(r, c) = ryzenai::pack_v2int4(x, y)
                : buff_B2.quant(r, c) = ryzenai::pack_v2int4(x, y);
          } else {
            (group_size < 128)
                ? buff_B1.quant(r, c) = ryzenai::pack_v2uint4(x, y)
                : buff_B2.quant(r, c) = ryzenai::pack_v2uint4(x, y);
          }
        }
      }

      // Select the supported group_size
      if (group_size >= 128) {
        assert(group_size % 128 == 0 &&
               "group_size should be div by 32 or 128");
        grp_size_ = 128;
      } else if (group_size >= 32) {
        assert(group_size % 32 == 0 && "group_size should be div by 32 or 128");
        grp_size_ = 32;
      }

      int repeat_count = group_size / grp_size_;
      // format the scales (bf16)
      for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0];
           r += group_size) {
        for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1]; c++) {
          for (int g = 0; g < repeat_count; g++) {
            (group_size < 128)
                ? buff_B1.scale(r + g * grp_size_, c) =
                      ryzenai::float_to_bfloat16(scales[(
                          ((rb + r) * w_shape_[1] / group_size) + (cb + c))])
                : buff_B2.scale(r + g * grp_size_, c) =
                      ryzenai::float_to_bfloat16(scales[(
                          ((rb + r) * w_shape_[1] / group_size) + (cb + c))]);
          }
        }
      }

      // format the zeros (int4)
      for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0];
           r += group_size) {
        for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
             c += 2) {
          int index = ((rb + r) * w_shape_[1] / (group_size)) + (cb + c);
          int x = zeros[index];
          int y = zeros[index + 1];
          int8_t pack_zeros;
          if (b_dtype_ == "int4") {
            pack_zeros = ryzenai::pack_v2int4(x, y);
          } else {
            pack_zeros = ryzenai::pack_v2uint4(x, y);
          }
          for (int g = 0; g < repeat_count; g++) {
            (group_size < 128)
                ? buff_B1.zero(r + g * grp_size_, c) = pack_zeros
                : buff_B2.zero(r + g * grp_size_, c) = pack_zeros;
          }
        }
      }
      auto b_format_stop = GET_ELAPSED_TIME_NS();
      b_format_time_ += b_format_stop - b_format_start;

      auto b_sync_start = GET_ELAPSED_TIME_NS();
      bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      auto b_sync_stop = GET_ELAPSED_TIME_NS();
      b_sync_time_ = b_sync_stop - b_sync_start;

      weights_bo_.push_back(bo_);
    }
  }
}

// specialization for ML-ADF, invoked in set_kernel_shapes_m if is_mladf_enabled
// is TRUE
template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::set_kernel_shapes_m_mladf(
    int64_t input_m) {
  if (a_dtype_ == "bfloat16") {
    if (input_m == 1)
      kernel_x_rows = 1;
    else if (input_m < 800)
      kernel_x_rows = 128;
    else if (input_m < 2048)
      kernel_x_rows = 800;
    else
      kernel_x_rows = 2048;
  } else
    throw std::runtime_error(
        "No Kernel exists for the chosen activation shape and data type");
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::run_aie(InT *a, xrt::bo &w_bo,
                                                    int64_t *input_shape) {
  // NOTE: Here we select the DPU sequence to use based on the
  //       number of rows in the input. This allows us to optimize
  //       kernels for both prefill and token generation phases
  //       of LLM inference. All kernels share the same weight
  //       buffer. The input buffer is allocated to be big enough
  //       for the largest kernel.
  //

  auto a_bo_run_aie = a_bo_;
  auto c_bo_run_aie = c_bo_;
  if (input_shape[0] == 1) {
    a_bo_run_aie = a_bo_token_;
    c_bo_run_aie = c_bo_token_;
  }

  set_kernel_shapes_m_mladf(input_shape[0]);

  std::string instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                             std::to_string(kernel_x_rows) + "_" +
                             std::to_string(kernel_x_shape_[1]) + "_" +
                             std::to_string(kernel_y_shape_[1]) + "_" +
                             std::to_string(grp_size_);

  RYZENAI_LOG_TRACE("instr_bo_key = " + instr_bo_key);

  // Pad and copy input activation to host BO memory
  // NOTE: BOs are allocated in the constructor to
  //       support the largest kernel size, so all of these
  //       memory accesses will be within bounds

  /*
   * Format A matrix for BF16 kernel
   * Since bf16 is not a natively supported dtype we
   * use int16_t buffers for activations and populate them
   */
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();

  const xrt::bo &instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;

  uint16_t *a_map = a_bo_run_aie.map<uint16_t *>();
  memset((void *)a_map, 0, a_bo_run_aie.size());
  uint16_t *a_u16 = reinterpret_cast<uint16_t *>(a);

  auto a_sz = a_shape_[0] * a_shape_[1];

  for (int i = 0; i < input_shape[0]; ++i) {
    // copy row from the source tile
    memcpy((void *)&a_map[i * kernel_x_shape_[1]], (void *)&a[i * a_shape_[1]],
           input_shape[1] * a_dtype_size_);
  }
  //  append params at the end of A tensor
  /* auto dev_params = (ParamSubv *)&a_map[kernel_x_rows * kernel_x_shape_[1]];
   dev_params->gen_params(kernel_x_rows, kernel_x_shape_[1],
                          kernel_y_shape_[1], grp_size_, sign);*/

  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();
  int instr_bo_words = instr_bo.size() / sizeof(int);
  // sync input activation to device memory
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();

  a_bo_run_aie.sync(XCL_BO_SYNC_BO_TO_DEVICE,
                    (kernel_x_rows * kernel_x_shape_[1] * a_dtype_size_), 0);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  auto kernel_ = xrt_ctx_->get_kernel();

  xrt::run run;
  // launch the GEMM kernel
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();

  // kernel call for GEMM that supports transaction binary flow
  if (kernel_x_shape_[1] != 11008) {
    run = kernel_(2, instr_bo, instr_bo_words,
                  a_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                  w_bo.address() + DDR_AIE_ADDR_OFFSET,
                  c_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET, 0);
  } else {
    run = kernel_(2, instr_bo, instr_bo_words,
                  a_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                  w_bo.address() + DDR_AIE_ADDR_OFFSET,
                  c_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                  c_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET, 0);
  }
  run.wait2();
  num_run_aie_++;

  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  // sync output activation to host memory
  int64_t c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_run_aie.sync(XCL_BO_SYNC_BO_FROM_DEVICE,
                    kernel_x_rows * kernel_z_shape_[1] * sizeof(OutT), 0);
  int64_t c_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ += a_copy_stop - a_copy_start;
  a_sync_time_ += a_sync_stop - a_sync_start;
  c_sync_time_ += c_sync_stop - c_sync_start;
  run_aie_time_ += run_aie_stop - run_aie_start;
}
template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::set_shape(
    std::vector<int> a_shape) {
  a_shape_[0] = a_shape[0];
  a_shape_[1] = a_shape[1];
  c_shape_[0] = a_shape[0];
  c_shape_[1] = w_shape_[1];
}
template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::execute(
    std::vector<xrt::bo> &inputs, std::vector<xrt::bo> &outputs) {
  int64_t input_shape[2];
  input_shape[0] = std::min(a_shape_[0], kernel_x_shape_[0]);
  input_shape[1] = std::min(a_shape_[1], kernel_x_shape_[1]);

  set_kernel_shapes_m_mladf(input_shape[0]);

  std::string instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                             std::to_string(kernel_x_rows) + "_" +
                             std::to_string(kernel_x_shape_[1]) + "_" +
                             std::to_string(kernel_y_shape_[1]) + "_" +
                             std::to_string(grp_size_);
  const xrt::bo &instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;

  int instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();

  xrt::run run;

  // kernel call for GEMM that supports transaction binary flow
  if (kernel_x_shape_[1] != 11008) {
    run = kernel_(2, instr_bo, instr_bo_words,
                  inputs[0].address() + DDR_AIE_ADDR_OFFSET,
                  inputs[1].address() + DDR_AIE_ADDR_OFFSET,
                  outputs[0].address() + DDR_AIE_ADDR_OFFSET, 0);
  } else {
    run = kernel_(2, instr_bo, instr_bo_words,
                  inputs[0].address() + DDR_AIE_ADDR_OFFSET,
                  inputs[1].address() + DDR_AIE_ADDR_OFFSET,
                  outputs[0].address() + DDR_AIE_ADDR_OFFSET,
                  outputs[0].address() + DDR_AIE_ADDR_OFFSET, 0);
  }
  run.wait2();
  if (c_shape_[1] < kernel_z_shape_[1]) {

    outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto c_map = outputs[0].map<AccT *>();
    int64_t output_shape[2];
    output_shape[0] = std::min(c_shape_[0], kernel_z_shape_[0]);
    output_shape[1] = std::min(c_shape_[1], kernel_z_shape_[1]);

    for (int i = 0; i < output_shape[0]; ++i) {
      memcpy((void *)&c_map[i * c_shape_[1]],
             (void *)&c_map[i * kernel_z_shape_[1]],
             output_shape[1] * sizeof(AccT));
    }
  }
}
template <typename InT, typename WtT, typename AccT, typename OutT>
std::vector<xrt::bo> mladfmatmulbias<InT, WtT, AccT, OutT>::get_inputs(int M) {
  if (M == 1) {
    return {a_bo_token_};
  } else {
    return {a_bo_};
  }
}
template <typename InT, typename WtT, typename AccT, typename OutT>
std::vector<xrt::bo> mladfmatmulbias<InT, WtT, AccT, OutT>::get_const() {
  return weights_bo_;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
std::vector<xrt::bo> mladfmatmulbias<InT, WtT, AccT, OutT>::get_outputs(int M) {
  if (M == 1) {
    return {c_bo_token_};
  } else {
    return {c_bo_};
  }
}
template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::execute(
    //    InT *a, const std::tuple<int, int> &a_shape, OutT *c) {
    const std::vector<Tensor> &input_Tensor,
    std::vector<Tensor> &output_Tensor) {
  // get original arguments from input/output Tensor
  InT *a = (InT *)input_Tensor.at(0).data;
  std::tuple<int, int> a_shape = {input_Tensor.at(0).shape.at(0),
                                  input_Tensor.at(0).shape.at(1)};
  OutT *c = (OutT *)output_Tensor.at(0).data;
  int64_t exec_start = GET_ELAPSED_TIME_NS();
  a_sync_time_ = 0;
  c_sync_time_ = 0;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  a_shape_[0] = std::get<0>(a_shape);
  a_shape_[1] = std::get<1>(a_shape);
  c_shape_[0] = std::get<0>(a_shape);
  c_shape_[1] = w_shape_[1];

  AccT *c_acc;
  if constexpr (std::is_same_v<AccT, OutT>) {
    c_acc = reinterpret_cast<AccT *>(c);
  } else {
    if (c_acc_vec_.size() != (c_shape_[0] * c_shape_[1])) {
      c_acc_vec_.resize(c_shape_[0] * c_shape_[1]);
    }
    c_acc = c_acc_vec_.data();
  }
  for (int64_t ra = 0; ra < a_shape_[0]; ra += kernel_x_shape_[0]) {
    for (int64_t cb = 0; cb < w_shape_[1]; cb += kernel_y_shape_[1]) {
      int k = 0;
      // compute row major tile index for weight BOs
      int64_t tile_pitch = w_padded_shape_[1] / kernel_y_shape_[1];
      int64_t tile_row = k / kernel_y_shape_[0];
      int64_t tile_col = cb / kernel_y_shape_[1];
      int64_t tile_idx = tile_row * tile_pitch + tile_col;
      // compute shape of current input tile
      int64_t input_shape[2];
      input_shape[0] = std::min(a_shape_[0] - ra, kernel_x_shape_[0]);
      input_shape[1] = std::min(a_shape_[1] - k, kernel_x_shape_[1]);

      run_aie(&a[ra * a_shape_[1] + k], weights_bo_[tile_idx], input_shape);

      // compute shape of current output tile
      int64_t output_shape[2];
      output_shape[0] = std::min(c_shape_[0] - ra, kernel_z_shape_[0]);
      output_shape[1] = std::min(c_shape_[1] - cb, kernel_z_shape_[1]);

      int64_t c_copy_start = GET_ELAPSED_TIME_NS();
      // initialize the output tile
      auto c_map = c_bo_.map<AccT *>();
      if (output_shape[0] == 1) {
        c_map = c_bo_token_.map<AccT *>();
      } else {
        c_map = c_bo_.map<AccT *>();
      }
      for (int i = 0; i < output_shape[0]; ++i) {
        memcpy((void *)&c_acc[(ra + i) * c_shape_[1] + cb],
               (void *)&c_map[i * kernel_z_shape_[1]],
               output_shape[1] * sizeof(AccT));
      }

      int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
      c_copy_time_ += (c_copy_stop - c_copy_start);
    }
  }

  int64_t exec_end = GET_ELAPSED_TIME_NS();

  int64_t a_pad_time = 0;
  int64_t c_pad_time = 0;
  int64_t cpu_depad_time = 0;

  if (debug_) {
    // Write input / output matrix to file.
    std::string a_fname = "ryzenai_qlinear2_" +
                          std::to_string(mladfmatmulbias_id_) + "_" +
                          std::to_string(num_execute_) + "_a.txt";
    std::string c_fname = "ryzenai_qlinear2_" +
                          std::to_string(mladfmatmulbias_id_) + "_" +
                          std::to_string(num_execute_) + "_c.txt";

    Utils::write_buffer_to_file(a, a_shape_[0] * a_shape_[1], a_fname);
    Utils::write_buffer_to_file(c, c_shape_[0] * c_shape_[1], c_fname);
  }
  num_execute_++;

  RYZENAI_LOG_INFO(
      std::to_string(mladfmatmulbias_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) +
      " " + std::to_string(kernel_x_rows) + " " +
      std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_pad_time) + " " + std::to_string(c_pad_time) + " " +
      std::to_string(cpu_depad_time) + " " + std::to_string(a_copy_time_) +
      " " + std::to_string(a_sync_time_) + " " + std::to_string(c_copy_time_) +
      " " + std::to_string(c_sync_time_) + " " + std::to_string(cpu_acc_time_) +
      " " + std::to_string((double)run_aie_time_ / num_run_aie_) + " " +
      std::to_string(grp_size_) + "\n");
}

static std::tuple<int, int, int> fit_MKN(const std::vector<Tensor> &input) {
  // input[0] --> input
  // input[1] --> wts
  int M = input.at(0).shape.size() == 3 ? input.at(0).shape.at(1)
                                        : input.at(0).shape.at(0);
  int K = input.at(1).shape.at(0);
  int N = input.at(1).shape.at(1);

  return std::make_tuple(M, K, N);
}
template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Matmul initialize_const_params(ptr) ...");

  DOD_THROW_IF(
      (const_params.size() != 4) || (const_params.at(0).shape.size() != 2),
      OpsFusion::dod_format("Unsupported const spec for Matmul\n") +
          OpsFusion::dod_format(
              "(Details : #const params == 1 ({}), Const param dim == 2 ({})",
              const_params.size(), const_params.at(0).shape.size()));

  int8_t *weights = (int8_t *)const_params.at(0).data;
  int8_t *zeros = (int8_t *)const_params.at(3).data;
  float *scales = (float *)const_params.at(2).data;
  float *bias = (float *)const_params.at(1).data;
  std::tuple<int, int> w_shape = {const_params.at(0).shape.at(0),
                                  const_params.at(0).shape.at(1)};
  int group_size = 128;
  std::string key = "group_size";
  group_size = std::any_cast<std::vector<int>>(attr.find(key)->second)[0];
  // Note: for mladf int8 gemm we had to change group id to 0
  const int group_id = 0;

  w_shape_[0] = std::get<0>(w_shape);
  w_shape_[1] = std::get<1>(w_shape);
  set_kernel_shapes_kn_mladf();
  // Use largest M dimension as the default. This has to correspond
  // to one of the available kernel sizes.
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;

  /* Create weight BOs */
  // Create a BO for weight block and initialize to zero
  //    NOTE: We must initialize to zero here because the weight matrix
  //          shape might not be an integer multiple of the block size.
  //          Initializing the BOs to zero takes care of the padding
  //          without allocating any extra scratch space.

  // For int4 quantization the buffer also contains bias, zeros, and scales
  // the weights are tiled in zigzag w4 aligned subvolumes of 32x128 tiles
  // the first subvolume consists of bias that is padded with zeros
  // Rest of the subvolumes consist weights+scales+zeros in each tile
  // QuantMatrix class has helper functions to write the data into the
  // correct index

  w_padded_shape_[0] = Utils::ceil_for_me(w_shape_[0], kernel_y_shape_[0]);
  w_padded_shape_[1] = Utils::ceil_for_me(w_shape_[1], kernel_y_shape_[1]);
  // The bfp16 kernel uses a block size of 4 for any gemm shape.
  int blk_size = 4;
  mladfQuantMatrix<64, 32, 32, 32> buff_B1(kernel_y_shape_[0],
                                           kernel_y_shape_[1], blk_size);
  mladfQuantMatrix<64, 128, 32, 128> buff_B2(kernel_y_shape_[0],
                                             kernel_y_shape_[1], blk_size);

  // iterate over kernel shaped blocks of the weight matrix
  for (int64_t rb = 0; rb < w_padded_shape_[0]; rb += kernel_y_shape_[0]) {
    for (int64_t cb = 0; cb < w_padded_shape_[1]; cb += kernel_y_shape_[1]) {
      auto b_format_start = GET_ELAPSED_TIME_NS();

      int block_size =
          (group_size < 128) ? buff_B1.data_size : buff_B2.data_size;
      auto bo_map = dest;
      memset((void *)bo_map, 0, block_size);

      buff_B1.data = (mladfCoreSubv<32, 32, 32> *)bo_map;
      buff_B2.data = (mladfCoreSubv<128, 32, 128> *)bo_map;

      // first pack the bias (bf16)
      for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1]; ++c) {
        if (rb == 0) {
          (group_size < 128)
              ? ryzenai::float_to_bfloat16(buff_B1.bias(c) = bias[cb + c])
              : ryzenai::float_to_bfloat16(buff_B2.bias(c) = bias[cb + c]);
        }
      }
      // format quantized weights (int4/uint4)
      for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0]; ++r) {
        for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
             c += 2) {
          // NOTE: int8_t weights will be sign extended to int
          int x = weights[((rb + r) * w_shape_[1]) + (cb + c)];
          int y = weights[((rb + r) * w_shape_[1]) + (cb + c) + 1];
          if (b_dtype_ == "int4") {
            (group_size < 128)
                ? buff_B1.quant(r, c) = ryzenai::pack_v2int4(x, y)
                : buff_B2.quant(r, c) = ryzenai::pack_v2int4(x, y);
          } else {
            (group_size < 128)
                ? buff_B1.quant(r, c) = ryzenai::pack_v2uint4(x, y)
                : buff_B2.quant(r, c) = ryzenai::pack_v2uint4(x, y);
          }
        }
      }

      // Select the supported group_size
      if (group_size >= 128) {
        assert(group_size % 128 == 0 &&
               "group_size should be div by 32 or 128");
        grp_size_ = 128;
      } else if (group_size >= 32) {
        assert(group_size % 32 == 0 && "group_size should be div by 32 or 128");
        grp_size_ = 32;
      }

      int repeat_count = group_size / grp_size_;
      // format the scales (bf16)
      for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0];
           r += group_size) {
        for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1]; c++) {
          for (int g = 0; g < repeat_count; g++) {
            (group_size < 128)
                ? buff_B1.scale(r + g * grp_size_, c) =
                      ryzenai::float_to_bfloat16(scales[(
                          ((rb + r) * w_shape_[1] / group_size) + (cb + c))])
                : buff_B2.scale(r + g * grp_size_, c) =
                      ryzenai::float_to_bfloat16(scales[(
                          ((rb + r) * w_shape_[1] / group_size) + (cb + c))]);
          }
        }
      }

      // format the zeros (int4)
      for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0];
           r += group_size) {
        for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
             c += 2) {
          int index = ((rb + r) * w_shape_[1] / (group_size)) + (cb + c);
          int x = zeros[index];
          int y = zeros[index + 1];
          int8_t pack_zeros;
          if (b_dtype_ == "int4") {
            pack_zeros = ryzenai::pack_v2int4(x, y);
          } else {
            pack_zeros = ryzenai::pack_v2uint4(x, y);
          }
          for (int g = 0; g < repeat_count; g++) {
            (group_size < 128)
                ? buff_B1.zero(r + g * grp_size_, c) = pack_zeros
                : buff_B2.zero(r + g * grp_size_, c) = pack_zeros;
          }
        }
      }
      auto b_format_stop = GET_ELAPSED_TIME_NS();
      b_format_time_ += b_format_stop - b_format_start;
    }
  }
  RYZENAI_LOG_TRACE("Matmul initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::vector<uint8_t>
mladfmatmulbias<InT, WtT, AccT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [M, K, N] = fit_MKN(input);
  std::string txn_key = get_instr_key(txn_fname_prefix_, M, K, N, 128); // TODO
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::vector<uint8_t>
mladfmatmulbias<InT, WtT, AccT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  return {};
}

template <typename InT, typename WtT, typename AccT, typename OutT>
std::vector<OpArgMap> mladfmatmulbias<InT, WtT, AccT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  // TODO: impl for mladfmatmulbias
  // input --> [input, weights, output]
  // Check if IO buffers have batch.
  w_shape_[0] = input.at(1).shape.at(0);
  w_shape_[1] = input.at(1).shape.at(1);
  set_kernel_shapes_kn_mladf();
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;

  int blk_size = 4;
  mladfQuantMatrix<64, 32, 32, 32> buff_B1(kernel_y_shape_[0],
                                           kernel_y_shape_[1], blk_size);
  mladfQuantMatrix<64, 128, 32, 128> buff_B2(kernel_y_shape_[0],
                                             kernel_y_shape_[1], blk_size);
  // iterate over kernel shaped blocks of the weight matrix
  int group_size = 128;
  std::string key = "group_size";
  group_size = std::any_cast<std::vector<int>>(attr.find(key)->second)[0];
  int block_size = (group_size < 128) ? buff_B1.data_size : buff_B2.data_size;
  size_t const_params_bo_size = block_size;
  size_t input_bo_size =
      (input.at(0).shape.at(2) * input.at(0).shape.at(1) * sizeof(InT));
  size_t output_bo_size =
      (input.at(0).shape.at(2) != 11008)
          ? (input.at(5).shape.at(2) * input.at(5).shape.at(1) * sizeof(OutT))
          : (input.at(5).shape.at(2) * input.at(5).shape.at(1) * sizeof(OutT)) *
                2;
  size_t super_kernel_size = get_super_kernel_params(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 5, 0, output_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 3, 5, 0, output_bo_size}};
  // TODO: ostringstream compiled fail
  // RYZENAI_LOG_TRACE(OpsFusion::dod_format("Matmulbias Argmap : {}",
  // cvt_to_string(arg_map)));
  return arg_map;
}
template <typename InT, typename WtT, typename AccT, typename OutT>
std::once_flag mladfmatmulbias<InT, WtT, AccT, OutT>::logger_flag_;
template <typename InT, typename WtT, typename AccT, typename OutT>
uint64_t mladfmatmulbias<InT, WtT, AccT, OutT>::mladfmatmulbias_count = 0;

template <typename InT, typename WtT, typename AccT, typename OutT>
std::once_flag mladfmatmulbias<InT, WtT, AccT, OutT>::instr_reg_flag_;
template class mladfmatmulbias<int16_t, uint8_t, int16_t, int16_t>;
template class mladfmatmulbias<int16_t, int8_t, int16_t, int16_t>;
} // namespace ryzenai
