/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __softmax_H__
#define __softmax_H__

#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
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
size_t tuple_product(const std::tuple<int, int, int> &tuple) {
  return std::get<0>(tuple) * std::get<1>(tuple) * std::get<2>(tuple);
}
/*
 * softmax is an experimental class to offload int8_t * int8_t matrix
 * softmaxtiplications to AIE. this class uses lite runtime stack to interface
 * with XRT and submit jobs to IPU. Even though the instructions in this
 * template supports transaction format, it can be extended to support DPU
 * sequence format.
 */
template <typename InOutT = int16_t> class softmax {
private:
  static const std::string DPU_DIR;

  std::map<std::string, std::string> txnbin_operand_header;
  // AXEL: check if this is needed
  std::map<std::string, std::vector<std::tuple<int, int, int>>> default_shapes_;

  int64_t kernel_x_shape_[3];
  int64_t operand_size_in_bytes_;
  int64_t mask_size_in_bytes_;
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
  uint64_t softmax_id_;
  static uint64_t softmax_count;
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
  void run_aie(InOutT *a, InOutT *b, InOutT *c,
               std::tuple<int, int, int> inputA_shape);

  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(std::string prefix, int b, int m, int n);

public:
  /*
   * softmax class constructor
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
  softmax(const std::string &a_dtype, const std::string &b_dtype,
          const std::string &c_dtype);

  /*
   * execute matrix multiplication c = a * w
   *
   * perform matmul c = a * w. w is stored in the object with initilize_weights
   * method.
   *
   * @param a pointer to activation matrix
   * @param a_shape tuple containing the shape of the activation matrix
   * @param c pointer to store the result of matmul
   *
   * @return none
   */
  void execute(InOutT *a, InOutT *b, InOutT *c,
               const std::tuple<int, int, int> &a_shape);

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
const std::string softmax<InOutT>::DPU_DIR =
    std::string(Utils::get_env_var("PYTORCH_AIE_PATH")) + "\\dll\\" +
    std::string(Utils::get_env_var("DEVICE")) + "\\softmax\\";

template <typename InOutT> std::once_flag softmax<InOutT>::logger_flag_;

template <typename InOutT> uint64_t softmax<InOutT>::softmax_count = 0;

template <typename InOutT> instruction_registry softmax<InOutT>::instr_reg_;

template <typename InOutT> std::once_flag softmax<InOutT>::instr_reg_flag_;

template <typename InOutT> void softmax<InOutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InOutT>
std::string softmax<InOutT>::get_instr_key(std::string prefix, int b, int m,
                                           int n) {

  return prefix + "_" + std::to_string(b) + "_" + std::to_string(m) + "_" +
         std::to_string(n) + ".bin";
}

template <typename InOutT> void softmax<InOutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;

  assert(default_shapes_.find(txn_fname_prefix_) == default_shapes_.end() &&
         "not found in default_shapes_");
  txnbin_operand_header = {{"bfloat16", "a16"}};
  std::vector<std::tuple<int, int, int>> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto tensor = supported_shapes[i];
    auto key = get_instr_key(txn_fname_prefix_, get<0>(tensor), get<1>(tensor),
                             get<2>(tensor));
    instructions.push_back(std::make_pair(key, false));
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions, DPU_DIR);
}

template <typename InOutT>
softmax<InOutT>::softmax(const std::string &a_dtype, const std::string &b_dtype,
                         const std::string &c_dtype) {

  default_shapes_["maskedsoftmax_a16"] =
      std::vector<std::tuple<int, int, int>>();
  default_shapes_["maskedsoftmax_a16"].push_back(
      std::make_tuple(32, 2048, 2048));

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
  softmax_id_ = softmax_count++;
  txnbin_operand_header = {{"bfloat16", "a16"}};
  txn_fname_prefix_ = "maskedsoftmax_" + txnbin_operand_header.at(a_dtype_);

  // assert(is_mladf_enabled_ &&
  //        "Currently no non-MLDADF enabled version supported");

  std::string XCLBIN_FNAME{};
  // if (is_mladf_enabled_) {
  XCLBIN_FNAME = Utils::get_env_var("PYTORCH_AIE_PATH") + "\\xclbin\\" +
                 Utils::get_env_var("DEVICE") +
                 "\\mladf_gemm_2x4x4_a16fw4acc16f" + ".xclbin";
  //}
  RYZENAI_LOG_TRACE("XCLBIN_FNAME: " + XCLBIN_FNAME.c_str() +
                    ", txn_fname_prefix_: " + txn_fname_prefix_);

  xrt_ctx_ = &xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });

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
    std::string header = "softmax_id M N kernel_b kernel_m kernel_k "
                         "num_aie_runs Executetime(ns) run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "B_copy_time(ns) B_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });
  // TODO not really sure we need this member
  kernel_x_shape_[0] = 32;
  kernel_x_shape_[1] = 2048;
  kernel_x_shape_[2] = 2048;
  auto shapeOperand = std::make_tuple(32, 2048, 2048);
  operand_size_in_bytes_ = tuple_product(shapeOperand) * sizeof(InOutT);
  mask_size_in_bytes_ =
      tuple_product(std::make_tuple(1, 2048, 2048)) * sizeof(InOutT);

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));

  b_bo_ = xrt::bo(xrt_ctx_->get_device(), mask_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
  RYZENAI_LOG_TRACE("[softmax] ID: " + std::to_string(softmax_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME + ", (a_dtype, c_dtype): (" +
                    a_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InOutT>
void softmax<InOutT>::run_aie(InOutT *a, InOutT *b, InOutT *c,
                              std::tuple<int, int, int> a_shape) {
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

  auto instr_bo_key = txn_fname_prefix_ + "_" +
                      std::to_string(kernel_x_shape_[0]) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_x_shape_[2]);
  ;
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

    instr_bo = &instr_reg_.get_instr_bo(instr_bo_key + ".bin").second;

    uint16_t *a_map = a_bo_run_aie.map<uint16_t *>();
    uint16_t *a_u16 = reinterpret_cast<uint16_t *>(a);
    memcpy((void *)a_map, (void *)a, operand_size_in_bytes_);

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
    uint16_t *b_u16 = reinterpret_cast<uint16_t *>(b);
    memcpy((void *)b_map, (void *)b, mask_size_in_bytes_);
  }

  int64_t b_copy_stop = GET_ELAPSED_TIME_NS();
  b_copy_time_ = b_copy_stop - b_copy_start;

  int instr_bo_words = instr_bo->size() / sizeof(int);
  // sync input activation to device memory
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();

  a_bo_run_aie.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();
  a_sync_time_ += a_sync_stop - a_sync_start;

  // TODO(uKernel): copy paste from above -> refactor
  int64_t b_sync_start = GET_ELAPSED_TIME_NS();

  b_bo_run_aie.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t b_sync_stop = GET_ELAPSED_TIME_NS();
  b_sync_time_ += b_sync_stop - b_sync_start;

  auto kernel_ = xrt_ctx_->get_kernel();

  xrt::run run;
  // launch the GEMM kernel
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();

  run = kernel_(2, *instr_bo, instr_bo_words,
                a_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                b_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
                c_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET, 0);

  run.wait2();
  num_run_aie_++;
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ = run_aie_stop - run_aie_start;
  // sync output activation to host memory
  int64_t c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_run_aie.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  //(outputC_shape[0] * outputC_shape[1] * a_dtype_size_), 0);
  int64_t c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += c_sync_stop - c_sync_start;

  // copy c_bo to host memory

  int64_t c_copy_start = GET_ELAPSED_TIME_NS();
  InOutT *c_bo_map = c_bo_run_aie.map<InOutT *>();
  memcpy((void *)c, (void *)c_bo_map, operand_size_in_bytes_);
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
}

template <typename InOutT>
void softmax<InOutT>::execute(InOutT *a, InOutT *b, InOutT *c,
                              const std::tuple<int, int, int> &a_shape) {
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

  run_aie(a, b, c, a_shape);

  int64_t exec_end = GET_ELAPSED_TIME_NS();
  int64_t exec_time = exec_end - exec_start;

  num_execute_++;

  RYZENAI_LOG_INFO(std::to_string(softmax_id_) + " " +          // 0
                   std::to_string(std::get<0>(a_shape)) + " " + // 1
                   std::to_string(std::get<1>(a_shape)) + " " + // 2
                   std::to_string(kernel_x_shape_[0]) + " " +   // 3
                   std::to_string(kernel_x_shape_[1]) + " " +   // 4
                   std::to_string(kernel_x_shape_[2]) + " " +   // 5
                   std::to_string(num_run_aie_) + " " +         // 6
                   std::to_string(exec_time) + " " +            // 7
                   std::to_string(run_aie_time_) + " " +        // 8
                   std::to_string(a_copy_time_) + " " +         // 9
                   std::to_string(a_sync_time_) + " " +         // 10
                   std::to_string(b_copy_time_) + " " +         // 11
                   std::to_string(b_sync_time_) + " " +         // 12
                   std::to_string(c_copy_time_) + " " +         // 13
                   std::to_string(c_sync_time_) + " " +         // 14
                   std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

} // namespace ryzenai

#endif /* __softmax_H__ */
