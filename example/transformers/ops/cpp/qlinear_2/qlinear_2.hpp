/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __QLINEAR_2_H__
#define __QLINEAR_2_H__

#include <fstream>
#include <iostream>
#include <map>
#include <tuple>

// Eigen headers
#include <Eigen/Eigen>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include "dpu_kernel_metadata.hpp"

// Headers for DPU sequence parsing
#include "buffer_ops.h"

// Headers for superkernel instruction generation
#include "super_instr.h"

// Headers for weight matrix formatting
#include "matrix_formatting.h"
#include "wgt_matrix.h"

#include "xrt_context.hpp"

#include "logging.h"
#include "utils.h"

// #define ENABLE_SAFETY_CHECKS

namespace ryzenai {

typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrix8i;
typedef Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrix32i;

/*
 * qlinear_2 is an experimental class to offload int8_t * int8_t matrix
 * multiplications to AIE. this class uses lite runtime stack to interface with
 * XRT and submit jobs to IPU. Even though the instructions in this template
 * supports transaction format, it can be extended to support DPU sequence
 * format.
 */
template <typename InT, typename WtT, typename OutT> class qlinear_2 {
private:
  static const int DUMMY_BO_SIZE = 16;
  static const std::string DPU_DIR;
  static const std::map<std::string, std::string> xclbin_a_header;
  static const std::map<std::string, std::string> xclbin_b_header;
  static const std::map<std::string, std::string> xclbin_acc_header;

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
  /* actual M x N of matrix A */
  int64_t c_shape_[2];
  /* actual K x N of matrix A */
  int64_t w_shape_[2];
  /* padded shape of weight matrix */
  int64_t w_padded_shape_[2];
  /* xrt context handle */
  xrt_context *xrt_ctx_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_token_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_token_;
  /* NOTE: the kernel function signature for DPU firmware
   *       expects two more BDs than we need, so we allocate
   *       these dummy BDs */
  /* XRT BO for dummy input 1 */
  xrt::bo dummy_bo1_;
  /* XRT BO for dummy input 2 */
  xrt::bo dummy_bo2_;
  /* XRT BOs for DPU sequences */
  xrt::bo instr_bo_1_;
  xrt::bo instr_bo_8_;
  xrt::bo instr_bo_16_;
  xrt::bo instr_bo_32_;
  xrt::bo instr_bo_64_;
  /* vector of XRT BOs for tiled and reformtted weight matrix */
  std::vector<xrt::bo> weights_bo_;

   /* size for activation dtype*/
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

  uint64_t qlinear_2_id_;
  static uint64_t qlinear_2_count;

  /* debug flag */
  bool debug_ = false;

  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string b_dtype_;
  std::string c_dtype_;

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

  /*
   * run matrix multiplication on CPU
   *
   * execute matmul on AIE with dimensions kernel_x_shape, kernel_y_shape,
   * kernel_z_shape
   *
   * @param a pointer to activation(a) matrix of shape kernel_x_shape
   * @param w_bo xrt_bo containing tiled, formatted weight matrix of shape
   * kernel_y_shape
   * @param c pointer to store matmul result of shape kernel_z_shape
   * @param a_shape shape of the input activation matrix(a_shape_)
   *
   * @return none
   */
  void run_cpu(InT *a, xrt::bo &w_bo, OutT *c, int64_t *a_shape);

  /*
   * create instruction buffer for IPU to execute matmul
   *
   * @param dpu_fname filename of DPU sequence
   * @return the buffer object containing the instructions
   */
  xrt::bo create_instructions(std::string dpu_fname);

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
  void qlinear_2::initialize_weights_int4(
      int8_t* weights, int8_t* zeros, float* scales, float* bias,
      const std::tuple<int, int>& w_shape,
      int group_size = 32);
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

  /* Utility function to set the kernel shape based on the weights dimensions
   * Pick kernel shape using weight matrix size
   * Select OPT shapes when a_type is int8
   * Select Llamav2 shapes when a_type is int16
   * Need to fix this to pick shapes independent of the datatype*/
  void set_kernel_shapes();

  /* Utility function to pick instruction based on input shapes*/
  void create_instr_code();

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
  void execute(InT *a, const std::tuple<int, int> &a_shape, OutT *c);

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

template <typename InT, typename WtT, typename OutT>
qlinear_2<InT, WtT, OutT>::qlinear_2(const std::string &a_dtype,
                                     const std::string &b_dtype,
                                     const std::string &c_dtype) {

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  if (a_dtype_ == "bfloat16"){
    a_dtype_size_ = sizeof(uint16_t); 
  }
  qlinear_2_id_ = qlinear_2_count++;

  /*select xclbin based on the input/output types*/
  const std::string XCLBIN_FNAME =
      std::string(Utils::get_env_var("PYTORCH_AIE_PATH")) + "\\xclbin\\" +
      std::string(Utils::get_env_var("DEVICE")) + "\\gemm_4x4_" +
      xclbin_a_header.at(a_dtype_) + xclbin_b_header.at(b_dtype_) +
      xclbin_acc_header.at(c_dtype_) + ".xclbin";

  xrt_ctx_ = &xrt_context::get_instance(XCLBIN_FNAME);

  dummy_bo1_ =
      xrt::bo(xrt_ctx_->get_device(), DUMMY_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(4));
  dummy_bo2_ =
      xrt::bo(xrt_ctx_->get_device(), DUMMY_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(7));

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
        "time(us) num_aie_runs run_aie_time(ns) A_Pad_time(ns) "
        "C_Pad_time(ns) C_depad_time(ns) A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) CPU_accum_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });
}

template <typename InT, typename WtT, typename OutT>
const std::string qlinear_2<InT, WtT, OutT>::DPU_DIR =
    std::string(Utils::get_env_var("PYTORCH_AIE_PATH")) + "\\dll\\" +
    std::string(Utils::get_env_var("DEVICE")) + "\\qlinear_2\\";

template <typename InT, typename WtT, typename OutT>
const std::map<std::string, std::string>
    qlinear_2<InT, WtT, OutT>::xclbin_a_header = {
        {"int8", "a8"}, {"int16", "a16"}, {"bfloat16", "a16f"}};

template <typename InT, typename WtT, typename OutT>
const std::map<std::string, std::string>
    qlinear_2<InT, WtT, OutT>::xclbin_b_header = {
        {"int8", "w8"}, {"int4", "w3"}, {"uint4", "w4"}, {"bfloat16", "w16f"}};

template <typename InT, typename WtT, typename OutT>
const std::map<std::string, std::string>
    qlinear_2<InT, WtT, OutT>::xclbin_acc_header = {
        {"int32", "acc32"}, {"int64", "acc64"}, {"float32", "acc32f"}};
template <typename InT, typename WtT, typename OutT>
std::once_flag qlinear_2<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t qlinear_2<InT, WtT, OutT>::qlinear_2_count = 0;

template <typename InT, typename WtT, typename OutT>
void qlinear_2<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
xrt::bo qlinear_2<InT, WtT, OutT>::create_instructions(std::string dpu_fname) {

  size_t instr_bo_words = get_instr_size(dpu_fname);
  if (instr_bo_words == 0) {
    throw std::runtime_error(
        "Error: DPU instruction sequence has length zero.");
  }

  xrt::bo instr_bo =
      xrt::bo(xrt_ctx_->get_device(), instr_bo_words * sizeof(int),
              XCL_BO_FLAGS_CACHEABLE, xrt_ctx_->get_kernel().group_id(5));
  init_hex_buf(instr_bo, instr_bo_words, dpu_fname);
  instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  return instr_bo;
}

template <typename InT, typename WtT, typename OutT>
void qlinear_2<InT, WtT, OutT>::set_kernel_shapes() {
  if (a_dtype_ == "int8") {
    KERNEL_M_MAX = 32;
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
    KERNEL_M_MAX = 32;
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

template <typename InT, typename WtT, typename OutT>
void qlinear_2<InT, WtT, OutT>::create_instr_code() {
  if (a_dtype_ == "int8") {
    params_bytes = SEQ_BYTES;
    if (kernel_y_shape_[0] == 2048 && kernel_y_shape_[1] == 2048) {
      instr_bo_1_ = create_instructions(DPU_DIR + "a8w8acc32_1_2k_2k.txt");
      instr_bo_8_ = create_instructions(DPU_DIR + "a8w8acc32_8_2k_2k.txt");
      instr_bo_16_ = create_instructions(DPU_DIR + "a8w8acc32_16_2k_2k.txt");
      instr_bo_32_ = create_instructions(DPU_DIR + "a8w8acc32_32_2k_2k.txt");
    } else if (kernel_y_shape_[0] == 2048 && kernel_y_shape_[1] == 8192) {
      instr_bo_1_ = create_instructions(DPU_DIR + "a8w8acc32_1_2k_8k.txt");
      instr_bo_8_ = create_instructions(DPU_DIR + "a8w8acc32_8_2k_8k.txt");
      instr_bo_16_ = create_instructions(DPU_DIR + "a8w8acc32_16_2k_8k.txt");
      instr_bo_32_ = create_instructions(DPU_DIR + "a8w8acc32_32_2k_8k.txt");
    } else if (kernel_y_shape_[0] == 8192 && kernel_y_shape_[1] == 2048) {
      instr_bo_1_ = create_instructions(DPU_DIR + "a8w8acc32_1_8k_2k.txt");
      instr_bo_8_ = create_instructions(DPU_DIR + "a8w8acc32_8_8k_2k.txt");
      instr_bo_16_ = create_instructions(DPU_DIR + "a8w8acc32_16_8k_2k.txt");
      instr_bo_32_ = create_instructions(DPU_DIR + "a8w8acc32_32_8k_2k.txt");
    } else {
      // NOTE: This code is not reachable, but is included to
      //       defend against future bugs.
      throw std::runtime_error("No DPU sequence exists for kernel shape!");
    }
  } else if (a_dtype_ == "int16") {
    params_bytes = SEQ_BYTES;
    if (kernel_y_shape_[0] == 4096 && kernel_y_shape_[1] == 4096) {
      instr_bo_1_ = create_instructions(DPU_DIR + "a16w8acc64_1_4k_4k.txt");
      instr_bo_8_ = create_instructions(DPU_DIR + "a16w8acc64_8_4k_4k.txt");
      instr_bo_32_ = create_instructions(DPU_DIR + "a16w8acc64_32_4k_4k.txt");
      instr_bo_64_ = create_instructions(DPU_DIR + "a16w8acc64_64_4k_4k.txt");
    } else if (kernel_y_shape_[0] == 4096 && kernel_y_shape_[1] == 11008) {
      instr_bo_1_ = create_instructions(DPU_DIR + "a16w8acc64_1_4k_11k.txt");
      instr_bo_8_ = create_instructions(DPU_DIR + "a16w8acc64_8_4k_11k.txt");
      instr_bo_32_ = create_instructions(DPU_DIR + "a16w8acc64_32_4k_11k.txt");
      instr_bo_64_ = create_instructions(DPU_DIR + "a16w8acc64_64_4k_11k.txt");
    } else if (kernel_y_shape_[0] == 11264 && kernel_y_shape_[1] == 4096) {
      instr_bo_1_ = create_instructions(DPU_DIR + "a16w8acc64_1_11k_4k.txt");
      instr_bo_8_ = create_instructions(DPU_DIR + "a16w8acc64_8_11k_4k.txt");
      instr_bo_32_ = create_instructions(DPU_DIR + "a16w8acc64_32_11k_4k.txt");
      instr_bo_64_ = create_instructions(DPU_DIR + "a16w8acc64_64_11k_4k.txt");
    } else {
      // NOTE: This code is not reachable, but is included to
      //       defend against future bugs.
      throw std::runtime_error("No DPU sequence exists for kernel shape!");
    }
  } else if (a_dtype_ == "bfloat16") {
    params_bytes = CORE_IN1_SIZE;
    if (kernel_y_shape_[0] == 4096 && kernel_y_shape_[1] == 4096) {
      instr_bo_1_ = create_instructions(DPU_DIR + "a16fw4acc32f_1_4k_4k.txt");
      instr_bo_8_ = create_instructions(DPU_DIR + "a16fw4acc32f_8_4k_4k.txt");
      instr_bo_32_ = create_instructions(DPU_DIR + "a16fw4acc32f_32_4k_4k.txt");
    } else if (kernel_y_shape_[0] == 11008 && kernel_y_shape_[1] == 4096) {
      instr_bo_1_ = create_instructions(DPU_DIR + "a16fw4acc32f_1_11k_4k.txt");
      instr_bo_8_ = create_instructions(DPU_DIR + "a16fw4acc32f_8_11k_4k.txt");
      instr_bo_32_ =
          create_instructions(DPU_DIR + "a16fw4acc32f_32_11k_4k.txt");
    } else if (kernel_y_shape_[0] == 4096 && kernel_y_shape_[1] == 12288) {
      instr_bo_1_ = create_instructions(DPU_DIR + "a16fw4acc32f_1_4k_11k.txt");
      instr_bo_8_ = create_instructions(DPU_DIR + "a16fw4acc32f_8_4k_11k.txt");
      instr_bo_32_ =
          create_instructions(DPU_DIR + "a16fw4acc32f_32_4k_11k.txt");
    } else {
      // NOTE: This code is not reachable, but is included to
      //       defend against future bugs.
      std::cout << "kernel_y_shape_[0] " << kernel_y_shape_[0]
                << " kernel_y_shape_[1] " << kernel_y_shape_[1] << "\n";
      throw std::runtime_error("No DPU sequence exists for kernel shape!");
    }
  }
}


template <typename InT, typename WtT, typename OutT>
void qlinear_2<InT, WtT, OutT>::initialize_weights_int4(
    int8_t* weights, int8_t* zeros, float* scales, float* bias,
    const std::tuple<int, int>& w_shape, int group_size)
{

  w_shape_[0] = std::get<0>(w_shape);
  w_shape_[1] = std::get<1>(w_shape);

  set_kernel_shapes();
  // Use largest M dimension as the default
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;
  
  /* Create DPU sequence BOs */
  create_instr_code();
  /* Create input/output BOs */
  const int A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_) + params_bytes;
  const int C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * sizeof(OutT));
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(1));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(3));

  const int A_BO_SIZE_TOKEN =
      (1 * kernel_x_shape_[1] * a_dtype_size_) + params_bytes;
  const int C_BO_SIZE_TOKEN = 1 * kernel_z_shape_[1] * sizeof(OutT);
  a_bo_token_ =
      xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE_TOKEN, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(1));
  c_bo_token_ =
      xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE_TOKEN, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(3));


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
  // iterate over kernel shaped blocks of the weight matrix
  for (int64_t rb = 0; rb < w_padded_shape_[0]; rb += kernel_y_shape_[0]) {
    for (int64_t cb = 0; cb < w_padded_shape_[1]; cb += kernel_y_shape_[1]) {

      xrt::bo bo_;
      auto b_format_start = GET_ELAPSED_TIME_NS();

      QuantMatrix buff_B(kernel_y_shape_[0], kernel_y_shape_[1]);
      int block_size = buff_B.data_size;
      
      bo_ =
          xrt::bo(xrt_ctx_->get_device(), block_size, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(2));
      auto bo_map = bo_.map<WtT *>();
      memset((void *)bo_map, 0, block_size);
      buff_B.data = (CoreSubv *)bo_map;
      
      // first pack the bias (bf16)
      for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1]; ++c) {
        if (rb == 0) {
          buff_B.bias(c) = ryzenai::float_to_bfloat16(bias[cb + c]);
        }
      }
      // format quantized weights (int4/uint4)
      for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0]; ++r) {
        for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
              c += 2) {
              // NOTE: int8_t weights will be sign extended to int
          int x = weights[((rb + r) * w_shape_[1]) + (cb + c)];
          int y = weights[((rb + r) * w_shape_[1]) + (cb + c) + 1];
          if(b_dtype_ == "int4"){
            buff_B.quant(r, c) = ryzenai::pack_v2int4(x,y);
          }
          else{
            buff_B.quant(r, c) = ryzenai::pack_v2uint4(x,y);
          }  
        }
      }
      int repeat_count = group_size / GRP_SIZE;
      // format the scales (bf16)
      for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0]; r += group_size) {
        for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
              c++) {
                for(int g = 0; g<repeat_count; g++)
                {
                  buff_B.scale(r + g*GRP_SIZE, c ) = ryzenai::float_to_bfloat16(scales[(( (rb + r)*w_shape_[1] / group_size) + (cb + c)) ]);
                }
        }
      }
      // format the zeros (int4)
      for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0]; r += group_size) {
        for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
              c += 2) {
              int index =  ((rb + r) * w_shape_[1]/ (group_size)) + (cb + c)  ;
              int x = zeros[index];
              int y = zeros[index + 1];
              int8_t pack_zeros;
              if(b_dtype_ == "int4"){
                pack_zeros = ryzenai::pack_v2int4(x,y);
              }
              else{
                pack_zeros = ryzenai::pack_v2uint4(x,y);
              } 
              for(int g = 0; g<repeat_count; g++){
                buff_B.zero(r+ g*GRP_SIZE, c) = pack_zeros;
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


template <typename InT, typename WtT, typename OutT>
void qlinear_2<InT, WtT, OutT>::initialize_weights(
    WtT *weights, const std::tuple<int, int> &w_shape, int group_size) {

  w_shape_[0] = std::get<0>(w_shape);
  w_shape_[1] = std::get<1>(w_shape);

  set_kernel_shapes();

  // Use largest M dimension as the default
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;
  
  /* Create DPU sequence BOs */
  create_instr_code();

  /* Create input/output BOs */
  const int A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_) + params_bytes;
  const int C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * sizeof(OutT));
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(1));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(3));

  const int A_BO_SIZE_TOKEN =
      (1 * kernel_x_shape_[1] * a_dtype_size_) + params_bytes;
  const int C_BO_SIZE_TOKEN = 1 * kernel_z_shape_[1] * sizeof(OutT);
  a_bo_token_ =
      xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE_TOKEN, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(1));
  c_bo_token_ =
      xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE_TOKEN, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(3));


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
      bo_ =
          xrt::bo(xrt_ctx_->get_device(), block_size, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(2));
      auto bo_map = bo_.map<WtT *>();
      memset((void *)bo_map, 0, block_size);

      WgtMatrix<WtT> block_4(bo_map, kernel_y_shape_[0], kernel_y_shape_[1]);
      WgtMatrix<WtT, 2> block_2(bo_map, kernel_y_shape_[0],
                                kernel_y_shape_[1]);

      // Re-arrange weight matrix block
      if (kernel_y_shape_[0] > 8192) {
        for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0]; ++r) {
          for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
                ++c) {
            block_2(r, c) = weights[((rb + r) * w_shape_[1]) + (cb + c)];
          }
        }
      } else {
        for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0]; ++r) {
          for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
                ++c) {
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

template <typename InT, typename WtT, typename OutT>
static void cpu_matmul(InT *a, WtT *b, OutT *c, int64_t *a_shape,
                       int64_t *b_shape) {
  auto a_ptr = a;
  auto b_ptr = b;
  auto c_ptr = c;
  auto M = a_shape[0];
  auto K = a_shape[1];
  auto N = b_shape[1];
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      OutT sum = 0;
      for (int k = 0; k < K; k++) {
        sum += ((OutT)a[i * K + k]) * ((OutT)b[k * N + j]);
      }
      c_ptr[i * N + j] = sum;
    }
  }
}

template <typename InT, typename WtT, typename OutT>
void qlinear_2<InT, WtT, OutT>::run_cpu(InT *a, xrt::bo &w_bo, OutT *c,
                                        int64_t *a_shape) {
  w_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  auto w_map = w_bo.map<WtT *>();
  auto a_map = a_bo_.map<InT *>();
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  Utils::_copy_tile<InT>(a, a_map, &kernel_x_shape_[0], &a_shape[1]);
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();

  cpu_matmul<InT, WtT, OutT>(a_map, w_map, c, kernel_x_shape_, kernel_y_shape_);

  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  Matrix32i A =
      Eigen::Map<Matrix8i>(a_map, kernel_x_shape_[0], kernel_x_shape_[1])
          .cast<Matrix32i::Scalar>();
  Matrix32i B =
      Eigen::Map<Matrix8i>(w_map, kernel_y_shape_[0], kernel_y_shape_[1])
          .cast<Matrix32i::Scalar>();
  Matrix32i C = A * B;
  memcpy(c, &C(0), sizeof(int32_t) * kernel_z_shape_[0] * kernel_z_shape_[1]);
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += run_aie_stop - run_aie_start;
  num_run_aie_++;
}

template <typename InT, typename WtT, typename OutT>
void qlinear_2<InT, WtT, OutT>::run_aie(InT *a, xrt::bo &w_bo,
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
    kernel_x_rows = 1;
    instr_bo = &instr_bo_1_;
    a_bo_run_aie = a_bo_token_;
    c_bo_run_aie = c_bo_token_;
  } else if (input_shape[0] <= 8) {
    kernel_x_rows = 8;
    instr_bo = &instr_bo_8_;
  } else if (input_shape[0] <= 16 && a_dtype_ == "int8") {
    kernel_x_rows = 16;
    instr_bo = &instr_bo_16_;
  } else {
    if (a_dtype_ == "int8" || a_dtype_ == "bfloat16" ||
        (input_shape[0] <= 32 && a_dtype_ == "int16")) {
      kernel_x_rows = 32;
      instr_bo = &instr_bo_32_;
    } else if (a_dtype_ == "int16") {
      kernel_x_rows = 64;
      instr_bo = &instr_bo_64_;
    }
  }
  size_t instr_bo_words = instr_bo->size() / sizeof(int);
  // Pad and copy input activation to host BO memory
  // NOTE: BOs are allocated in the constructor to
  //       support the largest kernel size, so all of these
  //       memory accesses will be within bounds
  
  /*
  * Format A matrix for BF16 kernel
  * Since bf16 is not a natively supported dtype we
  * use int16_t buffers for activations and populate them
  */ 
  // std::ofstream formatted_a;
  // formatted_a.open("formatted_a.txt"); 
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  if (a_dtype_ == "bfloat16") {
    uint16_t *a_map = a_bo_run_aie.map<uint16_t *>();
    memset((void *)a_map, 0, a_bo_run_aie.size());
    ActMatrix<uint16_t> buff_A(
        kernel_x_rows, kernel_x_shape_[1],
        (kernel_x_rows > KERNEL_M_MAX) ? KERNEL_M_MAX : kernel_x_rows, K_SUBV); // 8, 4096, 4096, 32
    buff_A.data = a_map;
    uint16_t* a_u16 = reinterpret_cast<uint16_t*>(a);

    auto a_sz = a_shape_[0] * a_shape_[1];
    for (int i = 0; i < input_shape[0]; ++i) {
      for (int j = 0; j < input_shape[1]; ++j) {
        auto in_idx = i * a_shape_[1] + j;
#ifdef ENABLE_SAFETY_CHECKS
        if(in_idx >= a_sz) {
          std::cout << "[ERROR] Input buffer overflow : " << in_idx << " >= " << a_sz << std::endl;
          std::cout << "a_shape : " << a_shape_[0] << "x" << a_shape_[1] << '\n'
                    << "w_shape : " << w_shape_[0] << "x" << w_shape_[1] << '\n'
                    << "c_shape : " << c_shape_[0] << "x" << c_shape_[1] << '\n'
                    << "i, j : " << i << "x" << j << std::endl;
          throw std::runtime_error("Out of Bound Access");
        }
#endif        
        auto tmp = a_u16[in_idx];
        buff_A.act(i, j) = tmp;
        //formatted_a<< bfloat16_to_float(buff_A.act(i, j)) <<std::endl;
      }
    }
    //formatted_a.close();
    // append params at the end of A tensor
    auto dev_params = (ParamSubv *)&a_map[kernel_x_rows * kernel_x_shape_[1]];
    dev_params->gen_params(kernel_x_rows, kernel_x_shape_[1],
                           kernel_y_shape_[1]);
  }
  // For Int8/Int16 kernels, A is not formatted.
  // we simply copy the matrix into the BO, and zero pad it when necessary
  else {
    InT *a_map = a_bo_run_aie.map<InT *>();
    memset((void *)a_map, 0, a_bo_run_aie.size());

    for (int i = 0; i < input_shape[0]; ++i) {
      // copy row from the source tile
      memcpy((void *)&a_map[i * kernel_x_shape_[1]],
             (void *)&a[i * a_shape_[1]], input_shape[1] * a_dtype_size_);
      // pad input row with trailing zeros
      memset((void *)&a_map[(i * kernel_x_shape_[1] + input_shape[1])], 0,
             (kernel_x_shape_[1] - input_shape[1]) * a_dtype_size_);
    }
    // Initialize the superkernel instruction sequence
    // NOTE: the superkernel instruction sequence is initialized at the
    //       offset after the IFM tensor with size params_bytes
    init_gemm_instr_ddr((int8_t *)(&a_map[kernel_x_rows * kernel_x_shape_[1]]),
                        kernel_x_rows, kernel_x_shape_[1], kernel_y_shape_[1],
                        KERNEL_M_MAX, 128, 64, 128 / (16 * a_dtype_size_));
  }
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();

  // sync input activation to device memory
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_run_aie.sync(
      XCL_BO_SYNC_BO_TO_DEVICE,
      (kernel_x_rows * kernel_x_shape_[1] * a_dtype_size_) + params_bytes, 0);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  // launch the GEMM kernel
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  auto kernel_ = xrt_ctx_->get_kernel();

  auto run = kernel_(1, a_bo_run_aie, w_bo, c_bo_run_aie, dummy_bo1_, *instr_bo,
                     instr_bo_words, dummy_bo2_);
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

template <typename InT, typename WtT, typename OutT>
void qlinear_2<InT, WtT, OutT>::execute(InT *a,
                                        const std::tuple<int, int> &a_shape,
                                        OutT *c) {
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

  a_shape_[0] = std::get<0>(a_shape);
  a_shape_[1] = std::get<1>(a_shape);
  c_shape_[0] = std::get<0>(a_shape);
  c_shape_[1] = w_shape_[1];

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

      // initialize the output tile
      auto c_map = c_bo_.map<OutT *>();
      if (output_shape[0] == 1) {
        c_map = c_bo_token_.map<OutT *>();
      } else {
        c_map = c_bo_.map<OutT *>();
      }

      ActMatrix<float> buff_C(kernel_x_rows, kernel_z_shape_[1], kernel_x_rows, N_SUBV);
      buff_C.data = (float*) c_map;
      //TODO:Fix output activation tiling with memtile BDs for BF16xInt4 kernel
      if(c_dtype_ == "float32" && a_dtype_ == "bfloat16")
      {
        for(int i = 0; i < output_shape[0]; ++i){
          for(int j = 0; j < output_shape[1]; j++){
            c[(ra + i) * c_shape_[1] + (cb + j)]  = buff_C.act(i, j);
          }
        }
      }
      else{
        for (int i = 0; i < output_shape[0]; ++i) {
        memcpy((void *)&c[(ra + i) * c_shape_[1] + cb],
               (void *)&c_map[i * kernel_z_shape_[1]],
               output_shape[1] * sizeof(OutT));
        }
      }
      
      // accumulate over inner dimension
      for (k = kernel_x_shape_[1]; k < a_shape_[1]; k += kernel_x_shape_[1]) {
        tile_row = k / kernel_y_shape_[0];
        tile_idx = tile_row * tile_pitch + tile_col;
        input_shape[1] = std::min(a_shape_[1] - k, kernel_x_shape_[1]);

        run_aie(&a[ra * a_shape_[1] + k], weights_bo_[tile_idx], input_shape);

        int64_t cpu_acc_start = GET_ELAPSED_TIME_NS();
        for (int i = 0; i < output_shape[0]; ++i) {
          //TODO:Fix output activation tiling with memtile BDs for BF16xInt4 kernel 
          for (int j = 0; j < output_shape[1]; ++j) {
            if(c_dtype_ == "float32" && a_dtype_ == "bfloat16"){
              c[(ra + i) * c_shape_[1] + (cb + j)] +=
                buff_C.act(i,j);  
            }
            else{
               c[(ra + i) * c_shape_[1] + (cb + j)] +=
                c_map[i * kernel_z_shape_[1] + j]; 
            }
            
          }
        }
        int64_t cpu_acc_stop = GET_ELAPSED_TIME_NS();
        cpu_acc_time_ += cpu_acc_stop - cpu_acc_start;
      }
    }
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
      " " + std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

} // namespace ryzenai

#endif /* __QLINEAR_2_H__ */
