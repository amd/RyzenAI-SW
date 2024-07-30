/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include <utils/dpu_mdata.hpp>

#include <utils/instruction_registry.hpp>
#include <utils/txn_container.hpp>
#include <xrt_context/xrt_context.hpp>

#include <ops/mladfsoftmax/mladfsoftmax.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

namespace ryzenai {

namespace {

// convert float to int in bytes
union IntFloatUnion {
  int32_t i;
  float f;
};

// table size
static const int lut_array_sz = 1024;
// param array size
static const int param_array_sz = 16;
// param array template
static int32_t param_array[param_array_sz] = {
    // meaning in bytes
    0x10001, // int16 | 0-1:   w_iter         |   2-3: h_iter
    0x10200, // int16 | 4-5:   num_depth_iter |   6-7: ch_iter
    0x0,     // int32 | 8-11:  num_remain, vaild number in  last v16
    0x1,     // int32 | 12-15: tile_cnt, row number
    0x100,   // int32 | 16-19: loop_cnt, column number/16,(row=4096,column=4096)
    0x1000,  // int32 | 20-23: lut_size, 4096
    0x47804e06, // float | 24-29: ofm_s, ofm scale=1/scale_q
    0x0,        // uint16 | 30-31: ofm_zp, ofm zero point
    // not used for now
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

// rtp param is used for unified xclbin with mladfmatmul, only last two bytes
// matter. softmax op do receives rtp from fusionruntime or unicase test, but it
// does not use it. op only use this rtp array defined here instead, which means
// we only need to change rtp value here if necessary.
static const size_t RTP_BYTE_SIZE = 64;
static uint8_t rtp[RTP_BYTE_SIZE] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131, 199};

std::string getXCLBinName() {
  return OpInterface::get_dod_base_dir() + MLADF_SOFTMAX_A16_XCLBIN_PATH;
}

} // namespace

template <typename InT, typename WtT, typename OutT>
std::string mladf_softmax<InT, WtT, OutT>::get_instr_key(
    std::string prefix, const std::vector<size_t> &dimensions) {
  std::string key = "mladfsoftmax_" + prefix;
  for (const int &dim : dimensions) {
    key += "_" + std::to_string(dim);
  }
  return key;
}

template <typename InT, typename WtT, typename OutT>
void mladf_softmax<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::vector<size_t>> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto shape = supported_shapes[i];
    auto key = get_instr_key(txn_fname_prefix_, shape);
    instructions.push_back(std::make_pair(key, false));
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
void mladf_softmax<InT, WtT, OutT>::set_params(const std::string &model_name,
                                               std::vector<size_t> a_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "PST" || model_name == "PSS") {
    XCLBIN_FNAME = getXCLBinName();
  } else {
    throw std::invalid_argument("model_name is not supported");
  }
  kernel_x_shape_ = a_shape;
  kernel_z_shape_ = a_shape;
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

static vector<float> cal_lut(float scale) {
  std::vector<float> lut;
  std::vector<float> lut_temp;
  int bandwidth = 256;
  int stride = 8;
  int i_stride = 512 / (2 * stride);
  lut.resize(512);
  lut_temp.resize(512);
  int index = 0;
  for (int n = 0; n < 1; ++n) {
    for (int i = 0; i < i_stride; ++i) {
      for (int m = 0; m < 2; ++m) {
        for (int k = 0; k < stride; ++k) {
          float x = (float)(uint8_t)(k + i * stride);
          float x_char = (float)(int8_t)(uint8_t)(k + i * stride);
          lut[index] = exp(x * scale);
          lut_temp[index] = exp(x_char * scale * pow(2, 8));
          index++;
        }
      }
    }
  }
  lut.insert(lut.end(), lut_temp.begin(), lut_temp.end());
  return lut;
}

template <typename InT, typename WtT, typename OutT>
void mladf_softmax<InT, WtT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("MLADFSOFTMAX initialize_const_params(ptr) ...");

  size_t lut_len = sizeof(float) * lut_array_sz;
  size_t param_len = sizeof(int32_t) * param_array_sz;
  size_t total_size = 2 * (lut_len + param_len);

  float ifm_scale, ofm_scale;
  uint16_t ifm_zp, ofm_zp;
  if (const_params.at(0).dtype == "float" &&
      const_params.at(2).dtype == "float") {
    ifm_scale = *static_cast<float *>(const_params.at(0).data);
    ofm_scale = *static_cast<float *>(const_params.at(2).data);
  } else {
    throw std::invalid_argument("Unsupported dtype.");
  }
  if (const_params.at(1).dtype == "uint16" &&
      const_params.at(3).dtype == "uint16") {
    ifm_zp = *static_cast<uint16_t *>(const_params.at(1).data);
    ofm_zp = *static_cast<uint16_t *>(const_params.at(3).data);
  } else {
    throw std::invalid_argument("Unsupported dtype.");
  }
  // set ofm qdq param to param array
  IntFloatUnion converter;
  converter.f = 1 / ofm_scale;
  // 1 / ofm_scale
  param_array[6] = converter.i;
  // ofm zero point
  param_array[7] = static_cast<int32_t>(ofm_zp);

  // calculate table based on scale
  std::vector<float> lut_vec = cal_lut(ifm_scale);
  auto lut_array = lut_vec.data();

  std::vector<char> buffer(total_size);
  char *buffer_ptr = buffer.data();
  // Copy param, lut, param, lut in order into the buffer
  memcpy(buffer_ptr, (void *)param_array, param_len);
  memcpy(buffer_ptr + param_len, (void *)lut_array, lut_len);
  memcpy(buffer_ptr + param_len + lut_len, (void *)param_array, param_len);
  memcpy(buffer_ptr + param_len + lut_len + param_len, (void *)lut_array,
         lut_len);
  memcpy(dest, buffer_ptr, total_size);

  RYZENAI_LOG_TRACE("MLADFSOFTMAX initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void mladf_softmax<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {

  operand_size_in_bytes_ =
      std::accumulate(kernel_x_shape_.begin(), kernel_x_shape_.end(), size_t{1},
                      std::multiplies{}) *
      sizeof(InT);
  const_input_size_ =
      (sizeof(float) * lut_array_sz + sizeof(int32_t) * param_array_sz) * 2;

  rtp_bo_ = xrt::bo(xrt_ctx_->get_device(), RTP_BYTE_SIZE,
                    XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));

  b_bo_ = xrt::bo(xrt_ctx_->get_device(), const_input_size_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));

  c_bo_ = xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));

  b_copy_time_ = 0;
  b_sync_time_ = 0;
  rtp_copy_time_ = 0;
  rtp_sync_time_ = 0;

  // b_bo copy
  int64_t b_copy_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  initialize_const_params(b_bo_map, const_params);
  int64_t b_copy_stop = GET_ELAPSED_TIME_NS();
  // b_bo sync
  int64_t b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t b_sync_stop = GET_ELAPSED_TIME_NS();

  b_copy_time_ = b_copy_stop - b_copy_start;
  b_sync_time_ = b_sync_stop - b_sync_start;

  // rtp_bo copy
  int64_t rtp_copy_start = GET_ELAPSED_TIME_NS();
  uint8_t *rtp_bo_map = rtp_bo_.map<uint8_t *>();
  memcpy((void *)rtp_bo_map, (void *)rtp, RTP_BYTE_SIZE);
  int64_t rtp_copy_stop = GET_ELAPSED_TIME_NS();

  // rtp_bo sync
  int64_t rtp_sync_start = GET_ELAPSED_TIME_NS();
  rtp_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t rtp_sync_stop = GET_ELAPSED_TIME_NS();

  rtp_copy_time_ = rtp_copy_stop - rtp_copy_start;
  rtp_sync_time_ = rtp_sync_stop - rtp_sync_start;
}

template <typename InT, typename WtT, typename OutT>
mladf_softmax<InT, WtT, OutT>::mladf_softmax(const std::string &operand_dtype,
                                             bool load_xrt) {
  operand_dtype_ = operand_dtype;
  operand_dtype_size_ = sizeof(InT);

  txnbin_operand_header = {{"uint16", "a16"}};
  default_shapes_["mladfsoftmax_a16"] = std::vector<std::vector<size_t>>();
  default_shapes_["mladfsoftmax_a16"].push_back({4096, 4096});
  adf_softmax_id_ = adf_softmax_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = getXCLBinName();
  txn_fname_prefix_ =
      "mladfsoftmax_" + txnbin_operand_header.at(operand_dtype_);

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;

  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  num_run_aie_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header = "masked_softmax_id Batch M N Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "Mask_copy_time(ns) Mask_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[MASKEDSOFTMAX] ID: " + std::to_string(adf_softmax_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (operand_dtype, b_dtype, c_dtype): (" + operand_dtype_ +
                    ", " + operand_dtype_ + ", " + operand_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void mladf_softmax<InT, WtT, OutT>::execute(const std::vector<Tensor> &input,
                                            std::vector<Tensor> &output) {

  InT *a = (InT *)input.at(0).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  num_run_aie_ = 0;

  int64_t exec_start = GET_ELAPSED_TIME_NS();

  // a_bo copy
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  memcpy((void *)a_bo_map, (void *)a, operand_size_in_bytes_);
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  // prepare inst_bo and param_bo
  const auto instr_bo_key = get_instr_key(txn_fname_prefix_, kernel_x_shape_);
  const xrt::bo &instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;

  int instr_bo_words = instr_bo.size() / sizeof(int);
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  xrt::run run;
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();

  run = kernel_(2, instr_bo, instr_bo_words,
                rtp_bo_.address() + DDR_AIE_ADDR_OFFSET,
                b_bo_.address() + DDR_AIE_ADDR_OFFSET,
                a_bo_.address() + DDR_AIE_ADDR_OFFSET,
                c_bo_.address() + DDR_AIE_ADDR_OFFSET, 0);
  run.wait2();
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += run_aie_stop - run_aie_start;
  num_run_aie_++;

  // sync output activation to host memory
  int64_t c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int64_t c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += c_sync_stop - c_sync_start;

  // copy c_bo to host memory
  auto aie_out = (OutT *)output.at(0).data;
  int64_t c_copy_start = GET_ELAPSED_TIME_NS();
  OutT *c_bo_map = c_bo_.map<OutT *>();
  memcpy((void *)aie_out, (void *)c_bo_map, operand_size_in_bytes_);
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
  int64_t exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(adf_softmax_id_) + " " +
      std::to_string(kernel_x_shape_[0]) + " " +
      std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(b_copy_time_) + " " + std::to_string(b_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> mladf_softmax<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  DOD_THROW_IF(
      (input.at(0).shape != vector<size_t>{1, 4096, 4096} &&
       input.at(0).shape != vector<size_t>{4096, 4096}),
      OpsFusion::dod_format("Unsupported input sharep for MLADFSOFTMAX\n") +
          OpsFusion::dod_format("(Details : #Now only support shape of (1, "
                                "4096, 4096) or (4096, 4096)."));
  auto a_shape = input.at(0).shape;
  std::vector<size_t> input_1_shape;
  if (a_shape[0] == 1) {
    input_1_shape.assign(a_shape.begin() + 1, a_shape.end());
  } else {
    input_1_shape = a_shape;
  }
  std::string txn_key = get_instr_key(txn_fname_prefix_, input_1_shape);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
mladf_softmax<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  std::vector<uint8_t> params(RTP_BYTE_SIZE);
  memcpy(params.data(), rtp, RTP_BYTE_SIZE);
  return params;
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> mladf_softmax<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  DOD_THROW_IF(
      (input.at(0).shape != vector<size_t>{1, 4096, 4096} &&
       input.at(0).shape != vector<size_t>{4096, 4096}),
      OpsFusion::dod_format("Unsupported input shape for MLADFSOFTMAX\n") +
          OpsFusion::dod_format("(Details : #Now only support shape of (1, "
                                "4096, 4096) or (4096, 4096)."));
  auto input1_shape = input.at(0).shape;
  auto input_1_bo_size =
      std::accumulate(input1_shape.begin(), input1_shape.end(), size_t{1},
                      std::multiplies{}) *
      sizeof(InT);
  size_t const_bo_size =
      (sizeof(float) * lut_array_sz + sizeof(int32_t) * param_array_sz) * 2;
  auto output1_shape = input.at(0).shape;
  auto output_1_bo_size =
      std::accumulate(output1_shape.begin(), output1_shape.end(), size_t{1},
                      std::multiplies{}) *
      sizeof(OutT);
  // each op kernel can only set one CONST_INPUT here
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 0, 5, 0, RTP_BYTE_SIZE},
      {OpArgMap::OpArgType::INPUT, 2, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 3, 6, 0, output_1_bo_size}};
  return arg_map;
}

template <typename InT, typename WtT, typename OutT>
std::once_flag mladf_softmax<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t mladf_softmax<InT, WtT, OutT>::adf_softmax_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag mladf_softmax<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void mladf_softmax<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template class mladf_softmax<uint16_t, uint8_t, uint16_t>;
} // namespace ryzenai
