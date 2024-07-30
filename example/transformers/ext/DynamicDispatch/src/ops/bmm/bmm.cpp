/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <any>
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

#include <utils/instruction_registry.hpp>
#include <xrt_context/xrt_context.hpp>

#include "utils/txn_container.hpp"
#include <ops/bmm/bmm.hpp>
#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/txn_container.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "xaiengine.h"

namespace ryzenai {

static std::tuple<size_t, size_t, size_t>
extract_MKN(const std::vector<Tensor> &inputs) {
  size_t M;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
  } else if (inputs.at(0).shape.size() == 3) { // has batch_dim
    M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1);
  } else {
    throw std::runtime_error("Input Shape is not supported");
  }

  size_t K = inputs.at(1).shape.at(0);
  size_t N = inputs.at(1).shape.at(1);
  return std::make_tuple(M, K, N);
}

template <typename InT, typename WtT, typename OutT>
std::tuple<size_t, size_t> bmm<InT, WtT, OutT>::map_padded_shape(size_t M,
                                                                 size_t N) {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<matrix_shapes> &supported_shapes = iter->second;
  size_t Mo = M;
  size_t No = N;
  int fidx = 0;
  int f_found = 0;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    if (M == mat.M && N == mat.K) {
      fidx = i;
      f_found = 1;
      break;
    }
  }

  if (f_found == 1) {
    iter = default_shapes_.find(txn_fname_prefix_);
    const std::vector<matrix_shapes> &actual_shapes = iter->second;
    auto mat = actual_shapes.at(fidx);
    Mo = mat.M;
    No = mat.K;
  } else {
    throw std::runtime_error("Can not find the shape");
  }
  return std::make_tuple(Mo, No);
}

/*
 * bmm is an experimental class to offload bf16 * bf16  (int16_t) matrix
 * multiplications to AIE. this class uses lite runtime stack to interface with
 * XRT and submit jobs to IPU. Even though the instructions in this template
 * supports transaction format, it can be extended to support DPU sequence
 * format.
 */

/* Utility function to set the kernel shape based on the weights dimensions
 * Pick kernel shape using weight matrix size
 * Select OPT shapes when a_type is int8
 * Select Llamav2 shapes when a_type is int16
 * Need to fix this to pick shapes independent of the datatype*/
template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::set_kernel_shapes() {
  // Use largest M dimension as the default
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = a_shape_[0];
  kernel_z_shape_[0] = a_shape_[0];
  kernel_x_shape_[1] = a_shape_[1];
  kernel_y_shape_[0] = w_shape_[0];
  kernel_y_shape_[1] = w_shape_[1];
  kernel_z_shape_[1] = w_shape_[1];
}

/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = "bmm_" + get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N);
    instructions.push_back(std::make_pair(key, false));
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
std::string bmm<InT, WtT, OutT>::get_instr_key(std::string prefix, size_t m,
                                               size_t k, size_t n) {
  return prefix + "_" + std::to_string(m) + "_" + std::to_string(k) + "_" +
         std::to_string(n);
}

/*
 * bmm class constructor
 *
 * @param kernel_x_shape tuple containing of M x K dimension base bmm
 * supported on IPU
 * @param kernel_y_shape tuple containing of K x N dimension base bmm
 * supported on IPU
 *
 * NOTE: If the input shape has a smaller M dimension than the kernel
 * shape initialized here, the execute function can transparently
 * call a smaller BMM to reduce padding overhead. The kernel
 * shape passed here should have largest supported M dimension.
 *
 */
template <typename InT, typename WtT, typename OutT>
bmm<InT, WtT, OutT>::bmm(const std::string &a_dtype, const std::string &b_dtype,
                         const std::string &c_dtype, bool load_xrt) {

  txnbin_a_header = {{"uint16_t", "a16"}, {"bfloat16", "a16"}};
  txnbin_b_header = {{"uint16_t", "w16"}, {"bfloat16", "w16"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["bmm_a16w16"] = std::vector<matrix_shapes>{};
  default_shapes_["bmm_a16w16"].emplace_back(65536, 128, 2048);
  default_shapes_["bmm_a16w16"].emplace_back(65536, 2048, 128);

  // raw shape is the actual shape from ONNX
  raw_shapes_["bmm_a16w16"] = std::vector<matrix_shapes>{};
  raw_shapes_["bmm_a16w16"].emplace_back(65536, 128, 2048);
  raw_shapes_["bmm_a16w16"].emplace_back(65536, 2048, 128);

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  bmm_id_ = bmm_count++;

  txn_fname_prefix_ =
      "bmm_" + txnbin_a_header.at(a_dtype) + txnbin_b_header.at(b_dtype);
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("txn_fname_prefix : {}", txn_fname_prefix_));

  KERNEL_M_MAX = 32 * 2048;
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
    std::string header = "bmm_id M K N kernel_m kernel_k kernel_n Execute"
                         "time(ns) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });
}

template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::set_params(const std::string &model_name,
                                     std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  a_shape_[0] = input_shape.at(0);
  a_shape_[1] = input_shape.at(1);

  // TODO this is for MHA for shor to call execute with xrt::bo
  if (a_shape_[1] == 128) {
    w_shape_[0] = 32 * 128;
    w_shape_[1] = 2048;
  } else {
    w_shape_[0] = 32 * 2048;
    w_shape_[1] = 128;
  }
  set_kernel_shapes();

  auto [M, K] = map_padded_shape(input_shape.at(0), input_shape.at(1));
  KERNEL_M_MAX = (int)M;
  if (model_name == "BMM" || model_name == "BMM1" || model_name == "BMM2") {
    XCLBIN_FNAME =
        OpInterface::get_dod_base_dir() +
        ryzenai::LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });

  // allocate instruction bo
  kernel_x_rows = (int64_t)M;

  // a16w16_65536_128_2048?
  auto instr_bo_key = "bmm_" + txn_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]);

  // make the following class variables
  instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;
  instr_bo_words = instr_bo.size() / sizeof(int);
  kernel_ = xrt_ctx_->get_kernel();
  allocate_inputs();
  allocate_outputs();
}

/*
 * copy weight matrix into XRT BOs with padding and tiling
 *
 * this method copies the weight matrix into XRT BOs. This is re-used for
 * every bmm performed for this object with different activations. weight
 * matrix is padded, tiled and reformatted while copying to XRT BOs. padding
 * is done to align with kernel_y_shape each tile of the weight matrix is of
 * shape kernel_y_shape this method also reformats the matrix b/weight matrix
 * as required by AIE/IPU bmm implementation
 *
 * @param weights pointer to the weight matrix
 * @param w_shape tuple containing the shape of the weight matrix
 *
 * @return none
 */

template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("bmm initialize_const_params ...");

  DOD_THROW_IF(
      (const_params.size() != 1) || (const_params.at(0).shape.size() != 2),
      OpsFusion::dod_format("Unsupported const spec for bmm\n") +
          OpsFusion::dod_format(
              "(Details : #const params == 1 ({}), Const param dim == 2 ({})",
              const_params.size(), const_params.at(0).shape.size()));

  w_shape_[0] = const_params.at(0).shape.at(0);
  w_shape_[1] = const_params.at(0).shape.at(1);
  set_kernel_shapes();
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  auto b_format_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  auto weights = (WtT *)const_params.at(0).data;
  auto weight_size = w_shape_[0] * w_shape_[1] * b_dtype_size_;
  memcpy((void *)(static_cast<WtT *>(b_bo_map)), (void *)weights, weight_size);
  auto b_format_stop = GET_ELAPSED_TIME_NS();
  b_format_time_ += b_format_stop - b_format_start;
  auto b_copy_stop = GET_ELAPSED_TIME_NS();
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();
  b_copy_time_ = b_copy_stop - b_copy_start;
  b_sync_time_ = b_sync_stop - b_sync_start;
  RYZENAI_LOG_TRACE("bmm initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                  std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("bmm execute ...");

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;
  int64_t exec_start = GET_ELAPSED_TIME_NS();
  a_shape_[0] = input.at(0).shape.at(0);
  a_shape_[1] = input.at(0).shape.at(1);
  c_shape_[0] = a_shape_[0];
  c_shape_[1] = w_shape_[1];
  auto aie_out = (OutT *)output.at(0).data;
  auto a = (InT *)input.at(0).data;

  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();

  memcpy((void *)a_bo_map, (void *)a,
         (a_shape_[0] * a_shape_[1] * a_dtype_size_));
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();
  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  std::vector<xrt::bo> inputs = {a_bo_, b_bo_};
  std::vector<xrt::bo> outputs = {c_bo_};
  execute(inputs, outputs);
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  // sync output activation to host memory
  int64_t c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  OutT *c_bo_map = c_bo_.map<OutT *>();
  int64_t c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += c_sync_stop - c_sync_start;
  run_aie_time_ += run_aie_stop - run_aie_start;
  int64_t c_copy_start = GET_ELAPSED_TIME_NS();
  memcpy((void *)aie_out, (void *)c_bo_map,
         (c_shape_[0] * c_shape_[1] * c_dtype_size_));
  int64_t c_copy_end = GET_ELAPSED_TIME_NS();
  c_copy_time_ += c_copy_end - c_copy_start;
  int64_t exec_end = GET_ELAPSED_TIME_NS();
  RYZENAI_LOG_INFO(
      std::to_string(bmm_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) + " " +
      std::to_string(kernel_x_rows) + " " + std::to_string(a_shape_[1]) + " " +
      std::to_string(w_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
  RYZENAI_LOG_TRACE("bmm execute ... DONE");
}

template <typename InT, typename WtT, typename OutT>
std::vector<xrt::bo> bmm<InT, WtT, OutT>::allocate_inputs() {
  size_t B_BO_SIZE = kernel_y_shape_[0] * kernel_y_shape_[1] * b_dtype_size_;
  size_t A_BO_SIZE = kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_;

  auto xDevice = xrt_ctx_->get_device();
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  return {a_bo_, b_bo_};
}

template <typename InT, typename WtT, typename OutT>
std::vector<xrt::bo> bmm<InT, WtT, OutT>::allocate_outputs() {
  size_t C_BO_SIZE = kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_;
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  return {c_bo_};
}

template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::execute(std::vector<xrt::bo> &input,
                                  std::vector<xrt::bo> &output) {
  // launch the BMM kernel
  run = kernel_(2, instr_bo, instr_bo_words,
                input[0].address() + DDR_AIE_ADDR_OFFSET,
                input[1].address() + DDR_AIE_ADDR_OFFSET,
                output[0].address() + DDR_AIE_ADDR_OFFSET, 0, 0);
  run.wait2();
}

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
template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> bmm<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  std::string txn_key = "bmm_" + get_instr_key(txn_fname_prefix_, Mo, Ko, N);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Instruction fname : {}", txn_key));
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());
  return data;
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> bmm<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  // input --> [input, weights, output]
  // Check if IO buffers have batch.
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko] = map_padded_shape(M, K);

  size_t const_params_bo_size = (Ko * N * b_dtype_size_);
  size_t input_bo_size = (Mo * Ko * a_dtype_size_);
  size_t output_bo_size = (Mo * N * c_dtype_size_);

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size}};
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("bmm Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
std::once_flag bmm<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t bmm<InT, WtT, OutT>::bmm_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag bmm<InT, WtT, OutT>::instr_reg_flag_;

template class bmm<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
