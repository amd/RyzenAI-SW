/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __QLINEAR_2_H__
#define __QLINEAR_2_H__

#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include "dpu_kernel_metadata.hpp"

// Subroutines to read the transaction binary
#include "op_buf.hpp"
#include "op_types.h"

#include "instruction_registry.hpp"
#include "xrt_context.hpp"

// AIE Driver header
#include "xaiengine.h"

// Headers for DPU sequence parsing
#include "buffer_ops.h"

// Headers for superkernel instruction generation
#include "super_instr.h"

// Headers for weight matrix formatting
#include "matrix_formatting.h"
#include "wgt_matrix.h"

#include "logging.h"
#include "utils.h"

#include <type_traits>

namespace ryzenai {
struct matrix_shapes {
  // capture M, K, N of the shape supported.
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t Gs;
  matrix_shapes(int64_t M, int64_t K, int64_t N) : M(M), K(K), N(N) { Gs = 0; }
  matrix_shapes(int64_t M, int64_t K, int64_t N, int64_t Gs)
      : M(M), K(K), N(N), Gs(Gs) {}
};

/*
 * qlinear_2 is an experimental class to offload int8_t * int8_t matrix
 * multiplications to AIE. this class uses lite runtime stack to interface
 * with
 * XRT and submit jobs to IPU. Even though the instructions in this template
 * supports transaction format, it can be extended to support DPU sequence
 * format.
 */
template <typename InT, typename WtT, typename AccT, typename OutT = AccT>
class qlinear_2 {
private:
  static const std::string DPU_DIR;
  static const std::map<std::string, std::string> xclbin_a_header;
  static const std::map<std::string, std::string> xclbin_b_header;
  static const std::map<std::string, std::string> xclbin_acc_header;
  static const std::map<std::string, std::string> txnbin_a_header;
  static const std::map<std::string, std::string> txnbin_b_header;
  static const std::map<std::string, std::string> txnbin_acc_header;
  static const std::map<std::string, std::vector<matrix_shapes>>
      default_shapes_;
  static const bool use_avx;

  /* M x K dimension of base matmul being offloaded to AIE */
  int64_t kernel_x_shape_[2];
  /* K x N dimension of base matmul being offloaded to AIE */
  int64_t kernel_y_shape_[2];
  /* M x N dimension of base matmul being offloaded to AIE */
  int64_t kernel_z_shape_[2];
  /*Kernel shape selected in runtime*/
  int64_t kernel_x_rows;
  /* Max Kernel M size supported for a given model*/
  int KERNEL_M_MAX;
  /* bytes required for params*/
  int params_bytes;
  /* actual M x K of matrix A */
  int64_t a_shape_[2];
  /* actual M x N of matrix C */
  int64_t c_shape_[2];
  /* actual K x N of matrix W */
  int64_t w_shape_[2];
  /* padded shape of weight matrix */
  int64_t w_padded_shape_[2];
  /* xrt context handle */
  xrt_context *xrt_ctx_;
  static instruction_registry instr_reg_;
  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_token_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_token_;
  /* vector of XRT BOs for tiled and reformtted weight matrix */
  std::vector<xrt::bo> weights_bo_;
  /* size for activation dtype */
  int a_dtype_size_;

  /*group size selected for this instantiation */
  int grp_size_;
  int sign;

  /* variables to store profile data */
  int64_t a_copy_time_;
  int64_t a_sync_time_;
  int64_t b_copy_time_;
  int64_t b_format_time_;
  int64_t b_sync_time_;
  int64_t c_copy_time_;
  int64_t c_sync_time_;
  int64_t run_aie_time_;
  int64_t cpu_acc_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t qlinear_2_id_;
  static uint64_t qlinear_2_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string b_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;
  /* variable to choose MLADF design*/
  int is_mladf_enabled_ = 0;
  enum MLADF_E { NOT_MLADF = 0, M4x4 = 1, M2x4x4 = 2 };
  std::map<std::string, MLADF_E> mladf_map{{"4x4", M4x4}, {"2x4x4", M2x4x4}};

  /* Temporary CPU buffer to hold accumulation */
  std::vector<AccT> c_acc_vec_;

  /*
   * run matrix multiplication on AIE
   *
   * execute matmul on AIE with dimensions kernel_x_shape, kernel_y_shape,
   * kernel_z_shape
   *
   * @param a pointer to activation(a) matrix of shape kernel_x_shape
   * @param w_bo xrt_bo containing tiled, formatted weight matrix of shape
   * kernel_y_shape
   * @param input_shape shape of the input activation matrix
   *
   * @return none
   */
  void run_aie(InT *a, xrt::bo &w_bo, int64_t *input_shape);

  /* Utility function to set the kernel shape based on the weights dimensions
   * Pick kernel shape using weight matrix size
   * Select OPT shapes when a_type is int8
   * Select Llamav2 shapes when a_type is int16
   * Need to fix this to pick shapes independent of the datatype*/
  void set_kernel_shapes_kn();

  // Specialization of set_kernel_shapes_kn for MLADF.
  void set_kernel_shapes_kn_mladf();

  /* Utility function to set the m dimension of the kernel based on the
   * activations.
   * Select OPT shapes when a_type is int8
   * Select Llamav2 shapes when a_type is int16*/
  void set_kernel_shapes_m(int64_t input_m);

  // Specialization of set_kernel_shapes_m and for MLADF.
  void set_kernel_shapes_m_mladf(int64_t input_m);

  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(std::string prefix, int m, int k, int n,
                            int grp_size);

public:
  /*
   * qlinear_2 class constructor
   *
   * @param kernel_x_shape tuple containing of M x K dimension base matmul
   * supported on IPU
   * @param kernel_y_shape tuple containing of K x N dimension base matmul
   * supported on IPU
   *
   * NOTE: If the input shape has a smaller M dimension than the kernel
   * shape initialized here, the execute function can transparently
   * call a smaller GeMM to reduce padding overhead. The kernel
   * shape passed here should have largest supported M dimension.
   *
   */
  qlinear_2(const std::string &a_dtype, const std::string &b_dtype,
            const std::string &c_dtype);
  /*
   * @param weights pointer to the row-major quantized weights
   *        NOTE: Each int4 weight is stored in a full byte
   * @param zeros pointer to the row-major zero points
   *        NOTE: Each int4 zero point is stored in a full byte
   * @param scales pointer to the fp32 scaling values
   * @param bias pointer to the fp32 bias values
   * @param w_shape tuple containing the shape of the weight matrix
   * @param group_size the number of weight elements per quantization group
   *
   * @return none
   */
  void qlinear_2::initialize_weights_int4(int8_t *weights, int8_t *zeros,
                                          float *scales, float *bias,
                                          const std::tuple<int, int> &w_shape,
                                          int group_size = 32);
  /*
   * copy weight matrix into XRT BOs with padding and tiling
   *
   * this method works for mladf kernels, it copies the weight matrix into XRT
   * BOs. This is re-used for every matmul performed for this object with
   * different activations. weight matrix is padded, tiled and reformatted while
   * copying to XRT BOs. padding is done to align with kernel_y_shape each tile
   * of the weight matrix is of shape kernel_y_shape this method also reformats
   * the matrix b/weight matrix as required by AIE/IPU matmul implementation
   *
   * @param weights pointer to the weight matrix
   * @param zeros pointer to the row-major zero points
   *        NOTE: Each int4 zero point is stored in a full byte
   * @param scales pointer to the bfloat16 scaling values
   * @param bias pointer to the bfloat16 bias values
   * @param w_shape tuple containing the shape of the weight matrix
   * @param group_size indicate the number of groups in weights
   *
   * @return none
   */
  void qlinear_2::initialize_weights_int4_mladf(
      int8_t *weights, int8_t *zeros, float *scales, float *bias,
      const std::tuple<int, int> &w_shape, int group_size = 128);

  /*
   * copy weight matrix into XRT BOs with padding and tiling
   *
   * this method copies the weight matrix into XRT BOs. This is re-used for
   * every matmul performed for this object with different activations. weight
   * matrix is padded, tiled and reformatted while copying to XRT BOs. padding
   * is done to align with kernel_y_shape each tile of the weight matrix is of
   * shape kernel_y_shape this method also reformats the matrix b/weight matrix
   * as required by AIE/IPU matmul implementation
   *
   * @param weights pointer to the weight matrix
   * @param w_shape tuple containing the shape of the weight matrix
   *
   * @return none
   */
  void qlinear_2::initialize_weights(WtT *weights,
                                     const std::tuple<int, int> &w_shape,
                                     int group_size = 32);

  /*
   * execute matrix multiplication c = a * w
   *
   * perform matmul c = a * w. w is stored in the object with
   * initialize_weights
   * method.
   *
   * @param a pointer to activation matrix
   * @param a_shape tuple containing the shape of the activation matrix
   * @param c pointer to store the result of matmul
   *
   * @return none
   */
  void execute(InT *a, const std::tuple<int, int> &a_shape, OutT *c);

  /*
   * method to set debug flag
   *
   * When the debug flag is set, execute method will write input, weights and
   * output matrices to a field. the filename will be
   * ryzenai_qlinear2_<execute_num>_<matrix>.txt
   *
   * @param debug bool value to enable disable debug feature. turned off by
   * default
   *
   * @return none
   */
  void debug(bool enable);
};

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::string qlinear_2<InT, WtT, AccT, OutT>::DPU_DIR =
    std::string(Utils::get_env_var("PYTORCH_AIE_PATH")) + "\\dll\\" +
    std::string(Utils::get_env_var("DEVICE")) + "\\qlinear_2\\";

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::map<std::string, std::string>
    qlinear_2<InT, WtT, AccT, OutT>::txnbin_a_header = {
        {"int8", "a8"}, {"int16", "a16"}, {"bfloat16", "a16f"}};

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::map<std::string, std::string>
    qlinear_2<InT, WtT, AccT, OutT>::txnbin_b_header = {
        {"int8", "w8"}, {"int4", "w4"}, {"uint4", "w4"}, {"bfloat16", "w16f"}};

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::map<std::string, std::string>
    qlinear_2<InT, WtT, AccT, OutT>::txnbin_acc_header = {
        {"int32", "acc32"},
        {"int64", "acc64"},
        {"bfloat16", "acc16f"},
        {"float32", "acc32f"}};

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::map<std::string, std::string>
    qlinear_2<InT, WtT, AccT, OutT>::xclbin_a_header = {
        {"int8", "a8"}, {"int16", "a16"}, {"bfloat16", "a16f"}};

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::map<std::string, std::string>
    qlinear_2<InT, WtT, AccT, OutT>::xclbin_b_header = {
        {"int8", "w8"}, {"int4", "w4"}, {"uint4", "w4"}, {"bfloat16", "w16f"}};

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::map<std::string, std::string>
    qlinear_2<InT, WtT, AccT, OutT>::xclbin_acc_header = {
        {"int32", "acc32"},
        {"int64", "acc64"},
        {"bfloat16", "acc16f"},
        {"float32", "acc32f"}};

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::map<std::string, std::vector<matrix_shapes>>
    qlinear_2<InT, WtT, AccT, OutT>::default_shapes_ = {
        {"mladf_a8w8acc32",
         {
             matrix_shapes(8, 2048, 2048),
             matrix_shapes(16, 2048, 2048),
             matrix_shapes(32, 2048, 2048),
             matrix_shapes(8, 2048, 8192),
             matrix_shapes(16, 2048, 8192),
             matrix_shapes(32, 2048, 8192),
             matrix_shapes(8, 2048, 51200),
             matrix_shapes(16, 2048, 51200),
             matrix_shapes(32, 2048, 51200),
             matrix_shapes(8, 8192, 2048),
             matrix_shapes(16, 8192, 2048),
             matrix_shapes(32, 8192, 2048),
         }},
        {"mladf_4x4_a16fw4acc16f",
         {
             // debug
             matrix_shapes(8, 256, 2048, 128),
             matrix_shapes(128, 256, 2048, 32),

             // formal shapes
             matrix_shapes(1, 11008, 4096, 128),
             matrix_shapes(1, 4096, 4096, 128),
             matrix_shapes(1, 4096, 12288, 128),
             matrix_shapes(1, 4096, 22528, 128),
             matrix_shapes(1, 4096, 32768, 32),

             matrix_shapes(128, 11008, 4096, 128),
             matrix_shapes(128, 4096, 4096, 128),
             matrix_shapes(128, 4096, 12288, 128),
             matrix_shapes(128, 4096, 22528, 128),
             matrix_shapes(128, 4096, 32768, 32),

             matrix_shapes(2048, 11008, 4096, 128),
             matrix_shapes(2048, 4096, 4096, 128),
             matrix_shapes(2048, 4096, 12288, 128),
             matrix_shapes(2048, 4096, 22528, 128),
             matrix_shapes(2048, 4096, 32768, 32),

             matrix_shapes(8, 256, 2048, 32),
         }},
        {"mladf_2x4x4_a16fw4acc16f",
         {
             // formal shapes
             matrix_shapes(1, 11008, 4096, 128),
             matrix_shapes(1, 4096, 4096, 128),
             matrix_shapes(1, 4096, 12288, 128),
             matrix_shapes(1, 4096, 22528, 128),
             matrix_shapes(1, 4096, 32768, 32),

             matrix_shapes(128, 11008, 4096, 128),
             matrix_shapes(128, 4096, 4096, 128),
             matrix_shapes(128, 4096, 12288, 128),
             matrix_shapes(128, 4096, 22528, 128),
             matrix_shapes(128, 4096, 32768, 32),

             matrix_shapes(2048, 11008, 4096, 128),
             matrix_shapes(2048, 4096, 4096, 128),
             matrix_shapes(2048, 4096, 12288, 128),
             matrix_shapes(2048, 4096, 22528, 128),
             matrix_shapes(2048, 4096, 32768, 32),
         }},
        {
            "a8w8acc32",
            {
                matrix_shapes(1, 2048, 2048),
                matrix_shapes(8, 2048, 2048),
                matrix_shapes(16, 2048, 2048),
                matrix_shapes(32, 2048, 2048),
                matrix_shapes(64, 2048, 2048),
                matrix_shapes(1, 2048, 8192),
                matrix_shapes(8, 2048, 8192),
                matrix_shapes(16, 2048, 8192),
                matrix_shapes(32, 2048, 8192),
                matrix_shapes(64, 2048, 8192),
                matrix_shapes(1, 8192, 2048),
                matrix_shapes(8, 8192, 2048),
                matrix_shapes(16, 8192, 2048),
                matrix_shapes(32, 8192, 2048),
                matrix_shapes(64, 8192, 2048),
            },
        },
        {
            "a16w8acc64",
            {
                matrix_shapes(1, 4096, 4096),
                matrix_shapes(8, 4096, 4096),
                matrix_shapes(32, 4096, 4096),
                matrix_shapes(1, 4096, 11008),
                matrix_shapes(8, 4096, 11008),
                matrix_shapes(32, 4096, 11008),
                matrix_shapes(1, 11264, 4096),
                matrix_shapes(8, 11264, 4096),
                matrix_shapes(32, 11264, 4096),
            },
        },
        {
            "a16fw4acc32f",
            {
                matrix_shapes(1, 4096, 4096, 32),
                matrix_shapes(8, 4096, 4096, 32),
                matrix_shapes(32, 4096, 4096, 32),
                matrix_shapes(1, 4096, 12288, 32),
                matrix_shapes(8, 4096, 12288, 32),
                matrix_shapes(32, 4096, 12288, 32),
                matrix_shapes(1, 11008, 4096, 32),
                matrix_shapes(8, 11008, 4096, 32),
                matrix_shapes(32, 11008, 4096, 32),
                matrix_shapes(1, 4096, 32768, 32),
                matrix_shapes(8, 4096, 32768, 32),
                matrix_shapes(32, 4096, 32768, 32),
                matrix_shapes(1, 4096, 4096, 128),
                matrix_shapes(8, 4096, 4096, 128),
                matrix_shapes(32, 4096, 4096, 128),
                matrix_shapes(1, 4096, 12288, 128),
                matrix_shapes(8, 4096, 12288, 128),
                matrix_shapes(32, 4096, 12288, 128),
                matrix_shapes(1, 11008, 4096, 128),
                matrix_shapes(8, 11008, 4096, 128),
                matrix_shapes(32, 11008, 4096, 128),
                matrix_shapes(1, 4096, 32768, 128),
                matrix_shapes(8, 4096, 32768, 128),
                matrix_shapes(32, 4096, 32768, 128),

            },
        },
};

template <typename InT, typename WtT, typename AccT, typename OutT>
const bool qlinear_2<InT, WtT, AccT, OutT>::use_avx =
    ryzenai::check_avx512_and_bf16_support();

template <typename InT, typename WtT, typename AccT, typename OutT>
std::once_flag qlinear_2<InT, WtT, AccT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename AccT, typename OutT>
uint64_t qlinear_2<InT, WtT, AccT, OutT>::qlinear_2_count = 0;

template <typename InT, typename WtT, typename AccT, typename OutT>
instruction_registry qlinear_2<InT, WtT, AccT, OutT>::instr_reg_;

template <typename InT, typename WtT, typename AccT, typename OutT>
std::once_flag qlinear_2<InT, WtT, AccT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename AccT, typename OutT>
void qlinear_2<InT, WtT, AccT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
std::string qlinear_2<InT, WtT, AccT, OutT>::get_instr_key(std::string prefix,
                                                           int m, int k, int n,
                                                           int grp_size = 0) {
  if (grp_size)
    return prefix + "_" + std::to_string(m) + "_" + std::to_string(k) + "_" +
           std::to_string(n) + "_" + std::to_string(grp_size) + ".bin";
  else
    return prefix + "_" + std::to_string(m) + "_" + std::to_string(k) + "_" +
           std::to_string(n) + ".bin";
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void qlinear_2<InT, WtT, AccT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N, mat.Gs);
    instructions.push_back(std::make_pair(key, false));
  }

  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions, DPU_DIR);
}

template <typename InT, typename WtT, typename AccT, typename OutT>
qlinear_2<InT, WtT, AccT, OutT>::qlinear_2(const std::string &a_dtype,
                                           const std::string &b_dtype,
                                           const std::string &c_dtype) {
  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
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

  /* The MLADF design is used if the environment variable has been set.
   * Currently only a8w8a32, a16fw3acc16f and a16fw4acc16f are supported, the
   * non-MLADF implementation is used as a fallback for other data types.
   */
  std::string device = Utils::get_env_var("DEVICE");
  bool is_dtype_supported_mladf =
      (device == "phx" && a_dtype_ == "int8" && b_dtype_ == "int8" &&
       c_dtype_ == "int32") ||
      (device == "stx" && a_dtype_ == "bfloat16" &&
       (b_dtype_ == "int4" || b_dtype_ == "uint4") && c_dtype_ == "bfloat16");
  std::string mladf_str = Utils::get_env_var("MLADF");
  if ((mladf_str == "4x4" || mladf_str == "2x4x4") &&
      is_dtype_supported_mladf) {
    is_mladf_enabled_ = int(mladf_map.at(mladf_str));
  }

  qlinear_2_id_ = qlinear_2_count++;
  /*select xclbin based on the input/output types*/

  std::string XCLBIN_FNAME;
  if (is_mladf_enabled_) {
    XCLBIN_FNAME = Utils::get_env_var("PYTORCH_AIE_PATH") + "\\xclbin\\" +
                   Utils::get_env_var("DEVICE") +
                   ((is_mladf_enabled_ == M4x4) ? "\\mladf_gemm_4x4_"
                                                : "\\mladf_gemm_2x4x4_") +
                   xclbin_a_header.at(a_dtype_) + xclbin_b_header.at(b_dtype_) +
                   xclbin_acc_header.at(c_dtype_) + ".xclbin";

    txn_fname_prefix_ = std::string("mladf_") +
                        ((is_mladf_enabled_ == M4x4) ? "4x4_" : "2x4x4_") +
                        txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);
  } else {
    XCLBIN_FNAME = Utils::get_env_var("PYTORCH_AIE_PATH") + "\\xclbin\\" +
                   Utils::get_env_var("DEVICE") + "\\gemm_4x4_" +
                   xclbin_a_header.at(a_dtype_) + xclbin_b_header.at(b_dtype_) +
                   xclbin_acc_header.at(c_dtype_) + ".xclbin";

    txn_fname_prefix_ = txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);
  }
  RYZENAI_LOG_TRACE("XCLBIN_FNAME: " + XCLBIN_FNAME.c_str() +
                    ", txn_fname_prefix_: " + txn_fname_prefix_);

  xrt_ctx_ = &xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  if (a_dtype_ == "int8") {
    params_bytes = SEQ_BYTES;
    KERNEL_M_MAX = 64;
  } else if (a_dtype_ == "int16") {
    params_bytes = SEQ_BYTES;
    KERNEL_M_MAX = 32;
  } else if (a_dtype_ == "bfloat16") {
    params_bytes = M_SUBV * K_SUBV * 2;
    KERNEL_M_MAX = 32;
  }

  if (is_mladf_enabled_) {
    // superkernel parameters not set through SHIM DDR
    params_bytes = 0;
    KERNEL_M_MAX = 32; // m dim of design is 32; initialize_weights() uses
                       // KERNEL_M_MAX by default
  }

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
        "qlinear_2_id M K N kernel_m kernel_k kernel_n Execute"
        "time(ns) num_aie_runs run_aie_time(ns) A_Pad_time(ns) "
        "C_Pad_time(ns) C_depad_time(ns) A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) CPU_accum_time(ns) "
        "Avg_time_per_aie_run(ns) group_size\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[QLINEAR_2] ID: " + std::to_string(qlinear_2_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void qlinear_2<InT, WtT, AccT, OutT>::set_kernel_shapes_kn() {
  if (is_mladf_enabled_) {
    return set_kernel_shapes_kn_mladf();
  }
  if (a_dtype_ == "int8") {
    if ((w_shape_[0] == 2048 && w_shape_[1] == 2048) ||
        (w_shape_[0] == 2048 && w_shape_[1] == 8192) ||
        (w_shape_[0] == 8192 && w_shape_[1] == 2048)) {
      // Update kernel shape to match weight matrix if a
      // supported kernel exists
      kernel_x_shape_[1] = w_shape_[0];
      kernel_y_shape_[0] = w_shape_[0];
      kernel_y_shape_[1] = w_shape_[1];
      kernel_z_shape_[1] = w_shape_[1];
    } else if (w_shape_[0] > 2048 && w_shape_[1] <= 2048) {
      // Use 8192 x 2048 as the kernel
      kernel_x_shape_[1] = 8192;
      kernel_y_shape_[0] = 8192;
      kernel_y_shape_[1] = 2048;
      kernel_z_shape_[1] = 2048;
    } else if (w_shape_[0] <= 2048 && w_shape_[1] > 2048) {
      // Use 2048 x 8192 as the kernel
      kernel_x_shape_[1] = 2048;
      kernel_y_shape_[0] = 2048;
      kernel_y_shape_[1] = 8192;
      kernel_z_shape_[1] = 8192;
    } else {
      // Use 2048 x 2048 as the default kernel
      kernel_x_shape_[1] = 2048;
      kernel_y_shape_[0] = 2048;
      kernel_y_shape_[1] = 2048;
      kernel_z_shape_[1] = 2048;
    }
  } else if (a_dtype_ == "int16") {
    if ((w_shape_[0] == 4096 && w_shape_[1] == 4096) ||
        (w_shape_[0] == 4096 && w_shape_[1] == 11008) ||
        (w_shape_[0] == 11008 && w_shape_[1] == 4096)) {
      // Update kernel shape to match weight matrix if a
      // supported kernel exists
      kernel_x_shape_[1] = (w_shape_[0] == 11008 ? 11264 : w_shape_[0]);
      kernel_y_shape_[0] = kernel_x_shape_[1];
      kernel_y_shape_[1] = w_shape_[1];
      kernel_z_shape_[1] = kernel_y_shape_[1];
    } else if (w_shape_[1] > 4096) {
      // Use 4096 x 11008 as the kernel
      kernel_x_shape_[1] = 4096;
      kernel_y_shape_[0] = 4096;
      kernel_y_shape_[1] = 11008;
      kernel_z_shape_[1] = 11008;
    } else if (w_shape_[0] > 4096) {
      // Use 11008 x 4096 as the kernel
      kernel_x_shape_[1] = 11264;
      kernel_y_shape_[0] = 11264;
      kernel_y_shape_[1] = 4096;
      kernel_z_shape_[1] = 4096;
    } else {
      // Use 4096 x 4096 as the default kernel
      kernel_x_shape_[1] = 4096;
      kernel_y_shape_[0] = 4096;
      kernel_y_shape_[1] = 4096;
      kernel_z_shape_[1] = 4096;
    }
  } else if (a_dtype_ == "bfloat16") {
    KERNEL_M_MAX = 32;
    if ((w_shape_[0] == 4096 && w_shape_[1] == 4096) ||
        (w_shape_[0] == 4096 && w_shape_[1] == 11008) ||
        (w_shape_[0] == 11008 && w_shape_[1] == 4096)) {
      kernel_x_shape_[1] = w_shape_[0];
      kernel_y_shape_[0] = kernel_x_shape_[1];
      kernel_y_shape_[1] = (w_shape_[1] == 11008 ? 12288 : w_shape_[1]);
      kernel_z_shape_[1] = kernel_y_shape_[1];
    } else if (w_shape_[0] > 4096) {
      // Use 11008 x 4096 as the kernel
      kernel_x_shape_[1] = 11008;
      kernel_y_shape_[0] = 11008;
      kernel_y_shape_[1] = 4096;
      kernel_z_shape_[1] = 4096;
    } else if (w_shape_[1] > 12288) {
      // Use 4096 x 32768 as the kernel
      kernel_x_shape_[1] = 4096;
      kernel_y_shape_[0] = 4096;
      kernel_y_shape_[1] = 32768;
      kernel_z_shape_[1] = 32768;
    } else if (w_shape_[1] > 4096) {
      // Use 4096 x 11008 as the kernel
      kernel_x_shape_[1] = 4096;
      kernel_y_shape_[0] = 4096;
      kernel_y_shape_[1] = 12288;
      kernel_z_shape_[1] = 12288;
    } else {
      kernel_x_shape_[1] = 4096;
      kernel_y_shape_[0] = 4096;
      kernel_y_shape_[1] = 4096;
      kernel_z_shape_[1] = 4096;
    }
  } else {
    /*Current support is only for Int8 and Int16 activation types*/
    throw std::runtime_error(
        "No Kernel exists for the current activation data type");
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void qlinear_2<InT, WtT, AccT, OutT>::set_kernel_shapes_kn_mladf() {
  if ((a_dtype_ != "int8") && (a_dtype_ != "bfloat16")) {
    /*Current support is only for Int8 and Bf16 activation type*/
    throw std::runtime_error(
        "No Kernel exists for the current activation data type");
  }
  if (a_dtype_ == "int8") {
    if ((w_shape_[0] == 2048 && w_shape_[1] == 2048) ||
        (w_shape_[0] == 2048 && w_shape_[1] == 8192) ||
        (w_shape_[0] == 8192 && w_shape_[1] == 2048)) {
      // Update kernel shape to match weight matrix if a
      // supported kernel exists.
      kernel_x_shape_[1] = w_shape_[0];
      kernel_y_shape_[0] = w_shape_[0];
      kernel_y_shape_[1] = w_shape_[1];
      kernel_z_shape_[1] = w_shape_[1];
    } else if (w_shape_[1] >= 51200) {
      // Use 2048 x 51200 as the kernel shape if N is very large.
      kernel_x_shape_[1] = 2048;
      kernel_y_shape_[0] = 2048;
      kernel_y_shape_[1] = 51200;
      kernel_z_shape_[1] = 51200;
    } else if (w_shape_[1] > 2048 && w_shape_[0] <= 2048) {
      // Use 2048 x 8192 as the kernel shape if N is large.
      kernel_x_shape_[1] = 2048;
      kernel_y_shape_[0] = 2048;
      kernel_y_shape_[1] = 8192;
      kernel_z_shape_[1] = 8192;
    } else if (w_shape_[0] > 2048 && w_shape_[1] <= 2048) {
      // Use 8192 x 2048 as the kernel shape if K is large.
      kernel_x_shape_[1] = 8192;
      kernel_y_shape_[0] = 8192;
      kernel_y_shape_[1] = 2048;
      kernel_z_shape_[1] = 2048;
    } else {
      // Use 2048 x 2048 as the default kernel shape.
      kernel_x_shape_[1] = 2048;
      kernel_y_shape_[0] = 2048;
      kernel_y_shape_[1] = 2048;
      kernel_z_shape_[1] = 2048;
    }
  } else if (a_dtype_ == "bfloat16") {
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
void qlinear_2<InT, WtT, AccT, OutT>::initialize_weights_int4(
    int8_t *weights, int8_t *zeros, float *scales, float *bias,
    const std::tuple<int, int> &w_shape, int group_size) {
  w_shape_[0] = std::get<0>(w_shape);
  w_shape_[1] = std::get<1>(w_shape);

  set_kernel_shapes_kn();
  // Use largest M dimension as the default. This has to correspond
  // to one of the available kernel sizes.
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;

  /* Create input/output BOs */
  const int A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_) + params_bytes;
  const int C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * sizeof(AccT));
  a_bo_ = xrt::bo(xrt_ctx_->get_context(), A_BO_SIZE, xrt::bo::flags::host_only,
                  xrt_ctx_->get_kernel().group_id(8));
  c_bo_ = xrt::bo(xrt_ctx_->get_context(), C_BO_SIZE, xrt::bo::flags::host_only,
                  xrt_ctx_->get_kernel().group_id(8));

  const int A_BO_SIZE_TOKEN =
      (1 * kernel_x_shape_[1] * a_dtype_size_) + params_bytes;
  const int C_BO_SIZE_TOKEN = 1 * kernel_z_shape_[1] * sizeof(AccT);
  a_bo_token_ =
      xrt::bo(xrt_ctx_->get_context(), A_BO_SIZE_TOKEN,
              xrt::bo::flags::host_only, xrt_ctx_->get_kernel().group_id(8));
  c_bo_token_ =
      xrt::bo(xrt_ctx_->get_context(), C_BO_SIZE_TOKEN,
              xrt::bo::flags::host_only, xrt_ctx_->get_kernel().group_id(8));

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
  QuantMatrix<8, 32, 128, 32> buff_B1(kernel_y_shape_[0], kernel_y_shape_[1]);
  QuantMatrix<8, 128, 128, 128> buff_B2(kernel_y_shape_[0], kernel_y_shape_[1]);
  // iterate over kernel shaped blocks of the weight matrix
  for (int64_t rb = 0; rb < w_padded_shape_[0]; rb += kernel_y_shape_[0]) {
    for (int64_t cb = 0; cb < w_padded_shape_[1]; cb += kernel_y_shape_[1]) {
      xrt::bo bo_;
      auto b_format_start = GET_ELAPSED_TIME_NS();

      int block_size =
          (group_size < 128) ? buff_B1.data_size : buff_B2.data_size;
      bo_ = xrt::bo(xrt_ctx_->get_context(), block_size,
                    xrt::bo::flags::host_only,
                    xrt_ctx_->get_kernel().group_id(8));
      auto bo_map = bo_.map<WtT *>();
      memset((void *)bo_map, 0, block_size);
      buff_B1.data = (CoreSubv<32, 128, 32> *)bo_map;
      buff_B2.data = (CoreSubv<128, 128, 128> *)bo_map;

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
        assert(group_size % 128 == 0, "group_size should be div by 32 or 128");
        grp_size_ = 128;
      } else if (group_size >= 32) {
        assert(group_size % 32 == 0, "group_size should be div by 32 or 128");
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

template <typename InT, typename WtT, typename AccT, typename OutT>
void qlinear_2<InT, WtT, AccT, OutT>::initialize_weights_int4_mladf(
    int8_t *weights, int8_t *zeros, float *scales, float *bias,
    const std::tuple<int, int> &w_shape, int group_size) {
  // Note: for mladf int8 gemm we had to change group id to 0
  const int group_id = is_mladf_enabled_ ? 0 : 8;
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
  // Reserve double the size of C for the intermediate result computed in case
  // of K=11k and 2x4x4 overlay.
  const int C_BO_SIZE =
      (kernel_x_shape_[1] != 11008)
          ? (kernel_z_shape_[0] * kernel_z_shape_[1] * sizeof(OutT))
          : (kernel_z_shape_[0] * kernel_z_shape_[1] * sizeof(OutT) * 2);
  a_bo_ = xrt::bo(xrt_ctx_->get_context(), A_BO_SIZE, xrt::bo::flags::host_only,
                  xrt_ctx_->get_kernel().group_id(group_id));
  c_bo_ = xrt::bo(xrt_ctx_->get_context(), C_BO_SIZE, xrt::bo::flags::host_only,
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
  // The used L1 subvolume sizes are identical for the 2x4x4 and 1x4x4 overlay.
  mladfQuantMatrix<64, 32, 32, 32> buff_B1(kernel_y_shape_[0],
                                           kernel_y_shape_[1], blk_size);
  mladfQuantMatrix<64, 128, 32, 128> buff_B2(kernel_y_shape_[0],
                                             kernel_y_shape_[1], blk_size);
  // iterate over kernel shaped blocks of the weight matrix
  for (int64_t rb = 0; rb < w_padded_shape_[0]; rb += kernel_y_shape_[0]) {
    for (int64_t cb = 0; cb < w_padded_shape_[1]; cb += kernel_y_shape_[1]) {
      xrt::bo bo_;
      auto b_format_start = GET_ELAPSED_TIME_NS();

      int block_size =
          (group_size < 128) ? buff_B1.data_size : buff_B2.data_size;
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
        assert(group_size % 128 == 0, "group_size should be div by 32 or 128");
        grp_size_ = 128;
      } else if (group_size >= 32) {
        assert(group_size % 32 == 0, "group_size should be div by 32 or 128");
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

template <typename InT, typename WtT, typename AccT, typename OutT>
void qlinear_2<InT, WtT, AccT, OutT>::initialize_weights(
    WtT *weights, const std::tuple<int, int> &w_shape, int group_size) {
  const int group_id = is_mladf_enabled_ ? 0 : 8;

  grp_size_ = group_size;
  w_shape_[0] = std::get<0>(w_shape);
  w_shape_[1] = std::get<1>(w_shape);

  weights_bo_.clear();
  set_kernel_shapes_kn();

  // Use largest M dimension as the default
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;

  /* Create input/output BOs */
  const int A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_) + params_bytes;
  const int C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * sizeof(AccT));

  a_bo_ = xrt::bo(xrt_ctx_->get_context(), A_BO_SIZE, xrt::bo::flags::host_only,
                  xrt_ctx_->get_kernel().group_id(group_id));
  c_bo_ = xrt::bo(xrt_ctx_->get_context(), C_BO_SIZE, xrt::bo::flags::host_only,
                  xrt_ctx_->get_kernel().group_id(group_id));

  const int A_BO_SIZE_TOKEN =
      (1 * kernel_x_shape_[1] * a_dtype_size_) + params_bytes;
  const int C_BO_SIZE_TOKEN = 1 * kernel_z_shape_[1] * sizeof(AccT);
  a_bo_token_ = xrt::bo(xrt_ctx_->get_context(), A_BO_SIZE_TOKEN,
                        xrt::bo::flags::host_only,
                        xrt_ctx_->get_kernel().group_id(group_id));
  c_bo_token_ = xrt::bo(xrt_ctx_->get_context(), C_BO_SIZE_TOKEN,
                        xrt::bo::flags::host_only,
                        xrt_ctx_->get_kernel().group_id(group_id));

  /* Create weight BOs */

  if (debug_) {
    std::string w_fname = "ryzenai_qlinear2_" + std::to_string(qlinear_2_id_) +
                          "_" + std::to_string(num_execute_) + "_b.txt";

    Utils::write_buffer_to_file(weights, w_shape_[0] * w_shape_[1], w_fname);
  }

  w_padded_shape_[0] = Utils::ceil_for_me(w_shape_[0], kernel_y_shape_[0]);
  w_padded_shape_[1] = Utils::ceil_for_me(w_shape_[1], kernel_y_shape_[1]);
  // iterate over kernel shaped blocks of the weight matrix
  for (int64_t rb = 0; rb < w_padded_shape_[0]; rb += kernel_y_shape_[0]) {
    for (int64_t cb = 0; cb < w_padded_shape_[1]; cb += kernel_y_shape_[1]) {

      // Create a BO for weight block and initialize to zero
      //    NOTE: We must initialize to zero here because the weight matrix
      //          shape might not be an integer multiple of the block size.
      //          Initializing the BOs to zero takes care of the padding
      //          without allocating any extra scratch space.

      xrt::bo bo_;
      auto b_format_start = GET_ELAPSED_TIME_NS();
      int block_size =
          kernel_y_shape_[0] * kernel_y_shape_[1] * sizeof(WtT); // zeros
      bo_ = xrt::bo(xrt_ctx_->get_context(), block_size,
                    xrt::bo::flags::host_only,
                    xrt_ctx_->get_kernel().group_id(group_id));
      auto bo_map = bo_.map<WtT *>();
      memset((void *)bo_map, 0, block_size);

      WgtMatrix<WtT> block_4(bo_map, kernel_y_shape_[0], kernel_y_shape_[1]);
      WgtMatrix<WtT, 2> block_2(bo_map, kernel_y_shape_[0], kernel_y_shape_[1]);

      // Re-arrange weight matrix block
      if (kernel_y_shape_[0] > 8192) {
        for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0]; ++r) {
          for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1]; ++c) {
            block_2(r, c) = weights[((rb + r) * w_shape_[1]) + (cb + c)];
          }
        }
      } else {
        for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0]; ++r) {
          for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1]; ++c) {
            block_4(r, c) = weights[((rb + r) * w_shape_[1]) + (cb + c)];
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

template <typename InT, typename WtT, typename AccT, typename OutT>
void qlinear_2<InT, WtT, AccT, OutT>::set_kernel_shapes_m(int64_t input_m) {
  // NOTE: kernel_x_rows has to be at least as large as input_m,
  // since the whole input has to be covered in one AIE run.
  if (is_mladf_enabled_)
    set_kernel_shapes_m_mladf(input_m);
  else if (input_m == 1)
    kernel_x_rows = 1;
  else if (input_m <= 8)
    kernel_x_rows = 8;
  else if (input_m <= 16 && a_dtype_ == "int8")
    kernel_x_rows = 16;
  else if (a_dtype_ == "bfloat16" || input_m <= 32)
    kernel_x_rows = 32;
  else if (a_dtype_ == "int16" || a_dtype_ == "int8")
    kernel_x_rows = 64;
  else
    throw std::runtime_error(
        "No Kernel exists for the chosen activation shape and data type");
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void qlinear_2<InT, WtT, AccT, OutT>::set_kernel_shapes_m_mladf(
    int64_t input_m) {
  if (a_dtype_ == "int8") {
    if (input_m <= 16)
      kernel_x_rows = 16;
    else
      kernel_x_rows = 32;
  } else if (a_dtype_ == "bfloat16") {
    if (input_m == 1)
      kernel_x_rows = 1;
    else if (input_m < 2048)
      kernel_x_rows = 128;
    else
      kernel_x_rows = 2048;
  } else
    throw std::runtime_error(
        "No Kernel exists for the chosen activation shape and data type");
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void qlinear_2<InT, WtT, AccT, OutT>::run_aie(InT *a, xrt::bo &w_bo,
                                              int64_t *input_shape) {
  // NOTE: Here we select the DPU sequence to use based on the
  //       number of rows in the input. This allows us to optimize
  //       kernels for both prefill and token generation phases
  //       of LLM inference. All kernels share the same weight
  //       buffer. The input buffer is allocated to be big enough
  //       for the largest kernel.
  //
  xrt::bo *instr_bo = nullptr;

  auto a_bo_run_aie = a_bo_;
  auto c_bo_run_aie = c_bo_;
  if (input_shape[0] == 1) {
    a_bo_run_aie = a_bo_token_;
    c_bo_run_aie = c_bo_token_;
  }

  set_kernel_shapes_m(input_shape[0]);

  auto instr_bo_key = txn_fname_prefix_ + "_" + std::to_string(kernel_x_rows) +
                      "_" + std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]);
  RYZENAI_LOG_TRACE("instr_bo_key = " + instr_bo_key.c_str());

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
  if (a_dtype_ == "bfloat16") {

    instr_bo = &instr_reg_
                    .get_instr_bo(instr_bo_key + "_" +
                                  std::to_string(grp_size_) + ".bin")
                    .second;

    uint16_t *a_map = a_bo_run_aie.map<uint16_t *>();
    memset((void *)a_map, 0, a_bo_run_aie.size());
    uint16_t *a_u16 = reinterpret_cast<uint16_t *>(a);

    auto a_sz = a_shape_[0] * a_shape_[1];

    for (int i = 0; i < input_shape[0]; ++i) {
      // copy row from the source tile
      memcpy((void *)&a_map[i * kernel_x_shape_[1]],
             (void *)&a[i * a_shape_[1]], input_shape[1] * a_dtype_size_);
    }
    //  append params at the end of A tensor
    if (!is_mladf_enabled_) {
      auto dev_params = (ParamSubv *)&a_map[kernel_x_rows * kernel_x_shape_[1]];
      dev_params->gen_params(kernel_x_rows, kernel_x_shape_[1],
                             kernel_y_shape_[1], grp_size_, sign);
    }
  }
  // For Int8/Int16 kernels, A is not formatted.
  // we simply copy the matrix into the BO, and zero pad it when necessary
  else {
    instr_bo = &instr_reg_.get_instr_bo(instr_bo_key + ".bin").second;
    InT *a_map = a_bo_run_aie.map<InT *>();
    memset((void *)a_map, 0, a_bo_run_aie.size());

    for (int i = 0; i < input_shape[0]; ++i) {
      // copy row from the source tile
      memcpy((void *)&a_map[i * kernel_x_shape_[1]],
             (void *)&a[i * a_shape_[1]], input_shape[1] * a_dtype_size_);
    }
    // Initialize the superkernel instruction sequence
    // NOTE: the superkernel instruction sequence is initialized at the
    //       offset after the IFM tensor with size params_bytes
    if (!is_mladf_enabled_) {
      init_gemm_instr_ddr(
          (int8_t *)(&a_map[kernel_x_rows * kernel_x_shape_[1]]), kernel_x_rows,
          kernel_x_shape_[1], kernel_y_shape_[1], KERNEL_M_MAX, 128, 64,
          128 / (16 * a_dtype_size_));
    }
  }

  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();
  int instr_bo_words = instr_bo->size() / sizeof(int);
  // sync input activation to device memory
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();

  if (!is_mladf_enabled_) {
    a_bo_run_aie.sync(
        XCL_BO_SYNC_BO_TO_DEVICE,
        (kernel_x_rows * kernel_x_shape_[1] * a_dtype_size_) + params_bytes, 0);
  } else {
    a_bo_run_aie.sync(XCL_BO_SYNC_BO_TO_DEVICE,
                      (kernel_x_rows * kernel_x_shape_[1] * a_dtype_size_), 0);
  }
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  auto kernel_ = xrt_ctx_->get_kernel();

  xrt::run run;
  // launch the GEMM kernel
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  // kernel call for GEMM that supports transaction binary flow
  uint32_t *b_map = w_bo.map<uint32_t *>();
  if (is_mladf_enabled_) {
    if (kernel_x_shape_[1] != 11008) {
      run = kernel_(2, *instr_bo, instr_bo_words,
                    a_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                    w_bo.address() + DDR_AIE_ADDR_OFFSET,
                    c_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET, 0);
    } else {
      run = kernel_(2, *instr_bo, instr_bo_words,
                    a_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                    w_bo.address() + DDR_AIE_ADDR_OFFSET,
                    c_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                    c_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET, 0);
    }
  } else {
    run = kernel_(2, *instr_bo, instr_bo_words,
                  c_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                  a_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                  w_bo.address() + DDR_AIE_ADDR_OFFSET, 0, 0);
  }
  run.wait2();
  num_run_aie_++;
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  // sync output activation to host memory
  int64_t c_sync_start = GET_ELAPSED_TIME_NS();
  if (is_mladf_enabled_) {
    c_bo_run_aie.sync(XCL_BO_SYNC_BO_FROM_DEVICE,
                      kernel_x_rows * kernel_z_shape_[1] * sizeof(OutT), 0);
  } else {
    c_bo_run_aie.sync(XCL_BO_SYNC_BO_FROM_DEVICE,
                      kernel_x_rows * kernel_z_shape_[1] * sizeof(AccT), 0);
  }
  int64_t c_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ += a_copy_stop - a_copy_start;
  a_sync_time_ += a_sync_stop - a_sync_start;
  c_sync_time_ += c_sync_stop - c_sync_start;
  run_aie_time_ += run_aie_stop - run_aie_start;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void qlinear_2<InT, WtT, AccT, OutT>::execute(
    InT *a, const std::tuple<int, int> &a_shape, OutT *c) {
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
  // for MLADF case, leave result as bf16 and do not allow accumulation
  // along K over several AIE calls

  if (is_mladf_enabled_ && c_dtype_ == "bfloat16") {
    if (kernel_x_shape_[1] > a_shape_[1])
      throw std::runtime_error("For ML-ADF bf16 / bfp16 gemm, no accumulation "
                               "across several AIE calls is supported.");
    if (!std::is_same_v<AccT, OutT>)
      throw std::runtime_error("For ML-ADF bf16 / bfp16 gemm, the accumulation "
                               "and out type must be identical.");
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

      // accumulate over inner dimension
      for (k = kernel_x_shape_[1]; k < a_shape_[1]; k += kernel_x_shape_[1]) {
        tile_row = k / kernel_y_shape_[0];
        tile_idx = tile_row * tile_pitch + tile_col;
        input_shape[1] = std::min(a_shape_[1] - k, kernel_x_shape_[1]);
        run_aie(&a[ra * a_shape_[1] + k], weights_bo_[tile_idx], input_shape);

        int64_t cpu_acc_start = GET_ELAPSED_TIME_NS();
        if (is_mladf_enabled_) {
          throw std::runtime_error(
              "mladf Kernel doesn't support host accumulation.");
        } else {
          for (int i = 0; i < output_shape[0]; ++i) {
            for (int j = 0; j < output_shape[1]; ++j) {
              c_acc[(ra + i) * c_shape_[1] + (cb + j)] +=
                  c_map[i * kernel_z_shape_[1] + j];
            }
          }
        }

        int64_t cpu_acc_stop = GET_ELAPSED_TIME_NS();
        cpu_acc_time_ += cpu_acc_stop - cpu_acc_start;
      }
    }
  }

  if (!is_mladf_enabled_)
    if constexpr (!std::is_same_v<AccT, OutT>) {
      // Assume AccT is float and OutT is bfloat16 for now, indicated by 4th
      // template being int16_t
      static_assert(std::is_same_v<AccT, float>, "AccT must be float");
      static_assert(std::is_same_v<OutT, int16_t>, "OutT must be int16_t");
      float_buffer_to_bfloat16(c_acc, c_acc_vec_.size(), (uint16_t *)c,
                               use_avx);
    }

  int64_t exec_end = GET_ELAPSED_TIME_NS();

  int64_t a_pad_time = 0;
  int64_t c_pad_time = 0;
  int64_t cpu_depad_time = 0;

  if (debug_) {
    // Write input / output matrix to file.
    std::string a_fname = "ryzenai_qlinear2_" + std::to_string(qlinear_2_id_) +
                          "_" + std::to_string(num_execute_) + "_a.txt";
    std::string c_fname = "ryzenai_qlinear2_" + std::to_string(qlinear_2_id_) +
                          "_" + std::to_string(num_execute_) + "_c.txt";

    Utils::write_buffer_to_file(a, a_shape_[0] * a_shape_[1], a_fname);
    Utils::write_buffer_to_file(c, c_shape_[0] * c_shape_[1], c_fname);
  }
  num_execute_++;

  RYZENAI_LOG_INFO(
      std::to_string(qlinear_2_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) + " " +
      std::to_string(kernel_x_rows) + " " + std::to_string(kernel_x_shape_[1]) +
      " " + std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_pad_time) + " " + std::to_string(c_pad_time) + " " +
      std::to_string(cpu_depad_time) + " " + std::to_string(a_copy_time_) +
      " " + std::to_string(a_sync_time_) + " " + std::to_string(c_copy_time_) +
      " " + std::to_string(c_sync_time_) + " " + std::to_string(cpu_acc_time_) +
      " " + std::to_string((double)run_aie_time_ / num_run_aie_) + " " +
      std::to_string(grp_size_) + "\n");
}

} // namespace ryzenai

#endif /* __QLINEAR_2_H__ */
