/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __silu_H__
#define __silu_H__

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
  int64_t N;
  matrix_shapes(int64_t M, int64_t N) : M(M), N(N) {}
  bool operator==(const matrix_shapes &other) const {
    return other.M == M && other.N == N;
  }
};

/*
 * silu is an experimental class to offload int8_t * int8_t matrix
 * silutiplications to AIE. this class uses lite runtime stack to interface with
 * XRT and submit jobs to IPU. Even though the instructions in this template
 * supports transaction format, it can be extended to support DPU sequence
 * format.
 */
template <typename InOutT = int16_t> class silu {
private:
  static const std::string DPU_DIR;
  static const std::map<std::string, std::string> xclbin_a_header;
  static const std::map<std::string, std::string> txnbin_a_header;
  static const std::map<std::string, std::string> xclbin_b_header;
  static const std::map<std::string, std::string> txnbin_b_header;
  static const std::map<std::string, std::string> xclbin_c_header;
  static const std::map<std::string, std::string> txnbin_c_header;
  // AXEL: check if this is needed
  std::map<std::string, std::vector<matrix_shapes>> default_shapes_;
  static const bool use_avx;

  /* M x K dimension of base matsilu being offloaded to AIE */
  int64_t kernel_x_shape_[2];
  /* K x N dimension of base matsilu being offloaded to AIE */
  int64_t kernel_y_shape_[2];
  /* M x N dimension of base matsilu being offloaded to AIE */
  int64_t kernel_z_shape_[2];
  /*Kernel shape selected in runtime*/
  int64_t kernel_x_rows;
  /* Max Kernel M size supported for a given model*/
  int KERNEL_M_MAX;
  /* bytes required for params*/
  int params_bytes;
  /* actual M x K of matrix A */
  int64_t a_shape_[2];
  /* actual M x K of matrix A */
  int64_t b_shape_[2];
  /* actual M x N of matrix C */
  int64_t c_shape_[2];
  /* xrt context handle */
  xrt_context *xrt_ctx_;
  static instruction_registry instr_reg_;
  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled lhs activation matrix */
  xrt::bo a_bo_;
  /* XRT BO for tiled rhs activation matrix */
  xrt::bo b_bo_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_;
  /* XRT BO for tiled lhs activation matrix */
  xrt::bo a_bo_token_;
  /* XRT BO for tiled rhs activation matrix */
  xrt::bo b_bo_token_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_token_;

  /* size for activation dtype */
  int a_dtype_size_;

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
  uint64_t silu_id_;
  static uint64_t silu_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string b_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;
  /* variable to choose MLADF design*/
  bool is_mladf_enabled_;

  /*
   * run matrix silutiplication on AIE
   *
   * execute matsilu on AIE with dimensions kernel_x_shape, kernel_y_shape,
   * kernel_z_shape
   *
   * @param a pointer to activation(a) matrix of shape kernel_x_shape
   * @param w_bo xrt_bo containing tiled, formatted weight matrix of shape
   * kernel_y_shape
   * @param input_shape shape of the input activation matrix
   *
   * @return none
   */
  void run_aie(InOutT *a, InOutT *b, InOutT *c, int64_t *input_shape);

  /* Utility function to set the m dimension of the kernel based on the
   * activations.
   * Select OPT shapes when a_type is int8
   * Select Llamav2 shapes when a_type is int16*/
  void set_kernel_shapes_mn(int64_t *input_m);

  // Specialization of set_kernel_shapes_m and for MLADF.
  void set_kernel_shapes_mn_mladf(int64_t *input_m);

  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(std::string prefix, int m, int n);

public:
  /*
   * silu class constructor
   *
   * @param kernel_x_shape tuple containing of M x K dimension base matsilu
   * supported on IPU
   * @param kernel_y_shape tuple containing of K x N dimension base matsilu
   * supported on IPU
   *
   * NOTE: If the input shape has a smaller M dimension than the kernel
   * shape initialized here, the execute function can transparently
   * call a smaller GeMM to reduce padding overhead. The kernel
   * shape passed here should have largest supported M dimension.
   *
   */
  silu(const std::string &a_dtype, const std::string &b_dtype,
       const std::string &c_dtype);

  /*
   * execute matrix silutiplication c = a * w
   *
   * perform matsilu c = a * w. w is stored in the object with initilize_weights
   * method.
   *
   * @param a pointer to activation matrix
   * @param a_shape tuple containing the shape of the activation matrix
   * @param c pointer to store the result of matsilu
   *
   * @return none
   */
  void execute(InOutT *a, InOutT *b, InOutT *c,
               const std::tuple<int, int> &a_shape);

  /*
   * method to set debug flag
   *
   * When the debug flag is set, execute method will write input, weights and
   * output matricies to a filed. the filename will be
   * ryzenai_qlinear2_<execute_num>_<matrix>.txt
   *
   * @param debug bool value to enable disable debug feature. turned off by
   * default
   *
   * @return none
   */
  void debug(bool enable);
};

template <typename InOutT>
const std::string silu<InOutT>::DPU_DIR =
    std::string(Utils::get_env_var("PYTORCH_AIE_PATH")) + "\\dll\\" +
    std::string(Utils::get_env_var("DEVICE")) + "\\silu\\";

template <typename InOutT>
const std::map<std::string, std::string> silu<InOutT>::txnbin_a_header = {
    /*{"int8", "a8"}, {"int16", "a16"},*/ {"bfloat16", "a16f"}};
template <typename InOutT>
const std::map<std::string, std::string> silu<InOutT>::txnbin_b_header = {
    /*{"int8", "a8"}, {"int16", "a16"},*/ {"bfloat16", "b16f"}};
template <typename InOutT>
const std::map<std::string, std::string> silu<InOutT>::txnbin_c_header = {
    /*{"int8", "a8"}, {"int16", "a16"},*/ {"bfloat16", "c16f"}};
template <typename InOutT>
const std::map<std::string, std::string> silu<InOutT>::xclbin_a_header = {
    /* {"int8", "a8"}, {"int16", "a16"},*/ {"bfloat16", "a16f"}};
template <typename InOutT>
const std::map<std::string, std::string> silu<InOutT>::xclbin_b_header = {
    /* {"int8", "a8"}, {"int16", "a16"},*/ {"bfloat16", "b16f"}};
template <typename InOutT>
const std::map<std::string, std::string> silu<InOutT>::xclbin_c_header = {
    /* {"int8", "a8"}, {"int16", "a16"},*/ {"bfloat16", "c16f"}};

// TODO(uKernel): ???? do we need avx?
template <typename InOutT>
const bool silu<InOutT>::use_avx = ryzenai::check_avx512_and_bf16_support();

template <typename InOutT> std::once_flag silu<InOutT>::logger_flag_;

template <typename InOutT> uint64_t silu<InOutT>::silu_count = 0;

template <typename InOutT> instruction_registry silu<InOutT>::instr_reg_;

template <typename InOutT> std::once_flag silu<InOutT>::instr_reg_flag_;

template <typename InOutT> void silu<InOutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InOutT>
std::string silu<InOutT>::get_instr_key(std::string prefix, int m, int n) {

  return prefix + "_" + std::to_string(m) + "_" + std::to_string(n) + ".bin";
}

template <typename InOutT> void silu<InOutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;

  assert(default_shapes_.find(txn_fname_prefix_) == default_shapes_.end() &&
         "not found in defaul_shapes_");

  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = get_instr_key(txn_fname_prefix_, mat.M, mat.N);
    instructions.push_back(std::make_pair(key, false));
  }

  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions, DPU_DIR);
}

template <typename InOutT>
silu<InOutT>::silu(const std::string &a_dtype, const std::string &b_dtype,
                   const std::string &c_dtype) {
  default_shapes_.insert(
      {std::string{"mladf_silu_4x4_a16fb16fc16f"},
       std::vector<matrix_shapes>{
           matrix_shapes(1, 11008), matrix_shapes(128, 11008),
           matrix_shapes(256, 11008), matrix_shapes(512, 11008),
           matrix_shapes(1024, 11008), matrix_shapes(2048, 11008),
           // TBDmatrix_shapes(800, 11008),
           // TBDmatrix_shapes(2000, 11008),
       }});
  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  assert(a_dtype == "bfloat16");
  assert(c_dtype == a_dtype);
  if (a_dtype_ == "bfloat16") {
    a_dtype_size_ = sizeof(InOutT);
  }

  /* The MLADF design is used if the environment variable has been set.
   */
  is_mladf_enabled_ = !std::string(Utils::get_env_var("MLADF")).empty() &&
                      (a_dtype_ == "bfloat16" && b_dtype_ == "bfloat16" &&
                       c_dtype_ == "bfloat16");
  silu_id_ = silu_count++;

  txn_fname_prefix_ = "mladf_silu_4x4_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_c_header.at(c_dtype_);

  assert(is_mladf_enabled_ &&
         "Currently no non-MLDADF enabled version supported");
  std::string XCLBIN_FNAME{};
  if (is_mladf_enabled_) {
    XCLBIN_FNAME = Utils::get_env_var("PYTORCH_AIE_PATH") + "\\xclbin\\" +
                   Utils::get_env_var("DEVICE") +
                   "\\mladf_4x4_gemm_silu_mul_a16fw4" + ".xclbin";
  }

  RYZENAI_LOG_TRACE("XCLBIN_FNAME: " + XCLBIN_FNAME.c_str() +
                    ", txn_fname_prefix_: " + txn_fname_prefix_);

  xrt_ctx_ = &xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  if (is_mladf_enabled_) {
    // superkernel parameters not set through SHIM DDR
    params_bytes = 0;
    KERNEL_M_MAX = 1; // TODO(uKernel): change this to at least 2000 as that
                      // appears in LLama2
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
    std::string header = "silu_id M N kernel_m kernel_n num_aie_runs Execute"
                         "time(ns) run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[silu] ID: " + std::to_string(silu_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME + ", (a_dtype, c_dtype): (" +
                    a_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InOutT>
void silu<InOutT>::set_kernel_shapes_mn(int64_t *input_m) {
  // NOTE: kernel_x_rows has to be at least as large as input_m,
  // since the whole input has to be covered in one AIE run.
  if (is_mladf_enabled_)
    set_kernel_shapes_mn_mladf(input_m);
  else
    throw std::runtime_error(
        "No Kernel exists for the chosen activation shape and data type");
}

template <typename InOutT>
void silu<InOutT>::set_kernel_shapes_mn_mladf(int64_t *input_m) {
  matrix_shapes inputShape{input_m[0], input_m[1]};
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  const auto itToShape =
      std::find(supported_shapes.begin(), supported_shapes.end(), inputShape);
  if (itToShape == supported_shapes.end()) {
    // TODO(uKernel): we need to find a strategy to pad or tile
    std::runtime_error("inputShape not found in list of valid txns");
  }
  kernel_x_rows = input_m[0];
  kernel_x_shape_[0] = input_m[0];
  kernel_x_shape_[1] = input_m[1];
}

template <typename InOutT>
void silu<InOutT>::run_aie(InOutT *a, InOutT *b, InOutT *c,
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
  auto b_bo_run_aie = b_bo_;
  auto c_bo_run_aie = c_bo_;
  // TODO(uKernel): anything special we need to do when its 1 token?
  // if (input_shape[0] == 1) {
  //   a_bo_run_aie = a_bo_token_;
  //   c_bo_run_aie = c_bo_token_;
  // }

  set_kernel_shapes_mn(input_shape);

  auto instr_bo_key = txn_fname_prefix_ + "_" +
                      std::to_string(kernel_x_shape_[0]) + "_" +
                      std::to_string(kernel_x_shape_[1]);
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
  auto a_sz = input_shape[0] * input_shape[1];
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  if (a_dtype_ == "bfloat16") {

    instr_bo = &instr_reg_.get_instr_bo(instr_bo_key + ".bin").second;

    uint16_t *a_map = a_bo_run_aie.map<uint16_t *>();
    memset((void *)a_map, 0, a_bo_run_aie.size());
    uint16_t *a_u16 = reinterpret_cast<uint16_t *>(a);
    memcpy((void *)a_map, (void *)a, a_sz * a_dtype_size_);

    // TODO(uKernel): do we need to add a parameter?
    //   append params at the end of A tensor
    // auto dev_params = (ParamSubv *)&a_map[kernel_x_rows *
    // kernel_x_shape_[1]]; dev_params->gen_params(kernel_x_rows,
    // kernel_x_shape_[1],
    //                        kernel_y_shape_[1], grp_size_, sign);
  }
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();
  a_copy_time_ = a_copy_stop - a_copy_start;

  // TODO(uKernel): this is a copy-paste from above -> refactor
  int64_t b_copy_start = GET_ELAPSED_TIME_NS();
  if (b_dtype_ == "bfloat16") {
    uint16_t *b_map = b_bo_run_aie.map<uint16_t *>();
    memset((void *)b_map, 0, b_bo_run_aie.size());
    uint16_t *b_u16 = reinterpret_cast<uint16_t *>(b);
    memcpy((void *)b_map, (void *)b, a_sz * a_dtype_size_);

    // TODO(uKernel): do we need to add a parameter?
    //   append params at the end of A tensor
    // auto dev_params = (ParamSubv *)&a_map[kernel_x_rows *
    // kernel_x_shape_[1]]; dev_params->gen_params(kernel_x_rows,
    // kernel_x_shape_[1],
    //                        kernel_y_shape_[1], grp_size_, sign);
  }
  int64_t b_copy_stop = GET_ELAPSED_TIME_NS();
  b_copy_time_ = b_copy_stop - b_copy_start;

  int instr_bo_words = instr_bo->size() / sizeof(int);
  // sync input activation to device memory
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_run_aie.sync(
      XCL_BO_SYNC_BO_TO_DEVICE,
      (input_shape[0] * input_shape[1] * a_dtype_size_) /*+ params_bytes*/, 0);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();
  a_sync_time_ += a_sync_stop - a_sync_start;

  // TODO(uKernel): copy paste from above -> refactor
  int64_t b_sync_start = GET_ELAPSED_TIME_NS();
  ;
  // b_bo_run_aie.sync(
  //     XCL_BO_SYNC_BO_TO_DEVICE,
  //(input_shape[0] *input_shape[1] * a_dtype_size_) /*+ params_bytes*/, 0);
  int64_t b_sync_stop = GET_ELAPSED_TIME_NS();
  b_sync_time_ += b_sync_stop - b_sync_start;

  auto kernel_ = xrt_ctx_->get_kernel();

  xrt::run run;
  // launch the GEMM kernel
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  // kernel call for GEMM that supports transaction binary flow
  run = kernel_(2, *instr_bo, instr_bo_words,
                a_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                // b_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                c_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET, 0, 0, 0);
  run.wait2();
  num_run_aie_++;
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += run_aie_stop - run_aie_start;

  // sync output activation to host memory
  int64_t c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_run_aie.sync(XCL_BO_SYNC_BO_FROM_DEVICE,
                    (input_shape[0] * input_shape[1] * a_dtype_size_), 0);
  int64_t c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += c_sync_stop - c_sync_start;

  // copy c_bo to host memory

  int64_t c_copy_start = GET_ELAPSED_TIME_NS();
  InOutT *c_bo_map = c_bo_run_aie.map<InOutT *>();
  memcpy((void *)c, (void *)c_bo_map,
         input_shape[0] * input_shape[1] * a_dtype_size_);
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
}

template <typename InOutT>
void silu<InOutT>::execute(InOutT *a, InOutT *b, InOutT *c,
                           const std::tuple<int, int> &input_shape) {
  int64_t exec_start = GET_ELAPSED_TIME_NS();
  a_sync_time_ = 0;
  a_copy_time_ = 0;

  b_sync_time_ = 0;
  b_copy_time_ = 0;

  c_copy_time_ = 0;
  c_sync_time_ = 0;

  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  a_shape_[0] = std::get<0>(input_shape);
  a_shape_[1] = std::get<1>(input_shape);
  b_shape_[0] = a_shape_[0];
  b_shape_[1] = a_shape_[1];
  c_shape_[0] = a_shape_[0];
  c_shape_[1] = a_shape_[1];

  a_bo_ =
      xrt::bo(xrt_ctx_->get_device(), a_shape_[0] * a_shape_[1] * a_dtype_size_,
              XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
  b_bo_ =
      xrt::bo(xrt_ctx_->get_device(), b_shape_[0] * b_shape_[1] * a_dtype_size_,
              XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
  c_bo_ =
      xrt::bo(xrt_ctx_->get_device(), c_shape_[0] * c_shape_[1] * a_dtype_size_,
              XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
  run_aie(a, b, c, a_shape_);

  // TODO(uKernel): do we need to do something specific when token == 1?
  // if (output_shape[0] == 1) {
  //   c_map = c_bo_token_.map<AccT *>();
  // } else {
  //   c_map = c_bo_.map<AccT *>();
  // }

  int64_t exec_end = GET_ELAPSED_TIME_NS();

  int64_t exec_time = exec_end - exec_start;

  int64_t a_pad_time = 0;
  int64_t c_pad_time = 0;
  int64_t cpu_depad_time = 0;

  num_execute_++;

  RYZENAI_LOG_INFO(std::to_string(silu_id_) + " " +           // 0
                   std::to_string(a_shape_[0]) + " " +        // 1
                   std::to_string(a_shape_[1]) + " " +        // 2
                   std::to_string(kernel_x_shape_[0]) + " " + // 3
                   std::to_string(kernel_x_shape_[1]) + " " + // 4
                   std::to_string(num_run_aie_) + " " +       // 5
                   std::to_string(exec_time) + " " +          // 6
                   std::to_string(run_aie_time_) + " " +      // 7
                   std::to_string(a_copy_time_) + " " +       // 8
                   std::to_string(a_sync_time_) + " " +       // 9
                   std::to_string(c_copy_time_) + " " +       // 10
                   std::to_string(c_sync_time_) + " " +       // 11
                   std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

} // namespace ryzenai

#endif /* __silu_H__ */
