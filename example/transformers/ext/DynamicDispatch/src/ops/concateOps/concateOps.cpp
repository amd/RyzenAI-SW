/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>

#ifndef _WIN32
#include <cmath>
#endif

#include <iomanip>
#include <iterator>
#include <string>

#include "utils/dpu_mdata.hpp"
#include <ops/concateOps/concateOps.hpp>
#include <ops/conv/conv.hpp>
#include <ops/maxpool/maxpool.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/help_file.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

namespace ryzenai {
/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename OutT>
void concateOps<InT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;

  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key =
        "concatenate_" + get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N);
    instructions.push_back(std::make_pair(key, false));
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
}

/*
 * concateOps class constructor
 */
template <typename InT, typename OutT>
concateOps<InT, OutT>::concateOps(
    const int graphId, const int inChannels, const int outChannels,
    const std::vector<std::map<std::string, std::any>> &attributesVec) {
  concatenate_id_ = concatenate_count++;
  graphId_ = graphId;
  inChannels_ = inChannels;
  outChannels_ = outChannels;

  txn_fname_prefix_ = "concatenate";
  default_shapes_["concatenate"] = std::vector<matrix_shapes>{};
  default_shapes_["concatenate"].emplace_back(320, 8, 16);
  default_shapes_["concatenate"].emplace_back(640, 8, 16);
  default_shapes_["concatenate"].emplace_back(1280, 8, 16);
  default_shapes_["concatenate"].emplace_back(2560, 8, 16);
  run_aie_time_ = 0;

  /* Attribute Parsing */
  for (const auto &attrs : attributesVec) {
    std::string opType = std::any_cast<std::string>(attrs.at("opType"));
    if (opType == "conv") {
      CreateConvOperator(attrs);
    } else if (opType == "maxpool") {
      CreateMaxpoolOperator(attrs);
    } else {
      std::cout << "Error: Concatenate does't support this operator"
                << std::endl;
    }
  }

  std::call_once(logger_flag_, []() {
    std::string header = "concatenate_id Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "IFM_copy_time(ns) IFM_sync_time(ns) "
                         "OFM_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });
}

/* CreateConvOperator private function
Below function creates a conv operator and pushed it's instance in
op_interfaces_ */
template <typename InT, typename OutT>
void concateOps<InT, OutT>::CreateConvOperator(
    const std::map<std::string, std::any> &attrs) {
  std::string opIfmDtype = std::any_cast<std::string>(attrs.at("opIfmDtype"));
  std::string opWtsDtype = std::any_cast<std::string>(attrs.at("opWtsDtype"));
  std::string opOfmDtype = std::any_cast<std::string>(attrs.at("opOfmDtype"));

  std::map<std::string, std::any> attr;
  attr["group"] = attrs.at("group");
  attr["input_shape"] = attrs.at("input_shape");
  attr["output_shape"] = attrs.at("output_shape");
  attr["weight_shape"] = attrs.at("weight_shape");
  attr["zero_point"] = attrs.at("zero_point");
  if (attrs.count("width")) {
    attr["width"] = attrs.at("width");
  }

  /* Sandip TBD: In below datatype should come from the opIfmDtype, opOfmDtype,
   * and opWtsDtype Store the std::unique_ptr<OpInterface> in the private member
   */
  op_interfaces_.push_back(
      std::make_unique<ryzenai::conv<uint16_t, uint16_t, uint16_t>>(
          opIfmDtype, opWtsDtype, opOfmDtype, false, attr));
}

/* CreateMaxpoolOperator private function
Below function creates a conv operator and pushed it's instance in
op_interfaces_ */
template <typename InT, typename OutT>
void concateOps<InT, OutT>::CreateMaxpoolOperator(
    const std::map<std::string, std::any> &attrs) {
  std::string opIfmDtype = std::any_cast<std::string>(attrs.at("opIfmDtype"));
  std::string opWtsDtype = std::any_cast<std::string>(attrs.at("opWtsDtype"));
  std::string opOfmDtype = std::any_cast<std::string>(attrs.at("opOfmDtype"));

  std::map<std::string, std::any> attr;
  attr["group"] = attrs.at("group");
  attr["input_shape"] = attrs.at("input_shape");
  attr["output_shape"] = attrs.at("output_shape");
  attr["weight_shape"] = attrs.at("weight_shape");
  attr["zero_point"] = attrs.at("zero_point");

  /* Sandip TBD: In below datatype should come from the opIfmDtype, opOfmDtype,
   * and opWtsDtype Store the std::unique_ptr<OpInterface> in the private member
   */
  op_interfaces_.push_back(
      std::make_unique<ryzenai::maxpool<uint16_t, uint16_t>>(opIfmDtype,
                                                             opOfmDtype, attr));
}

template <typename InT, typename OutT>
void concateOps<InT, OutT>::set_params(const std::string &modelName,
                                       bool debugFlag) {
  std::string modelNameLowerCase = modelName;
  std::transform(modelNameLowerCase.begin(), modelNameLowerCase.end(),
                 modelNameLowerCase.begin(), ::tolower);
  std::string XCLBIN_FNAME;
  if (modelNameLowerCase == "pso2") {
    XCLBIN_FNAME = OpInterface::get_dod_base_dir() +
                   "\\xclbin\\stx\\4x2_pso2_model_a16w16_qdq.xclbin";
  }
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  debug_ = debugFlag;
}

template <typename InT, typename OutT>
std::string
concateOps<InT, OutT>::get_instr_key(std::string prefix, int64_t graphId,
                                     int64_t inChannels, int64_t outChannels) {
  return prefix + "_" + std::to_string(graphId) + "_" +
         std::to_string(inChannels) + "_" + std::to_string(outChannels);
}

/* Below function is not using input tensor and output tensor. So it is OK to
 * call this functin with dummy input and ouput tensors */
template <typename InT, typename OutT>
std::vector<OpArgMap> concateOps<InT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  size_t const_params_bo_size = 0;
  size_t max_scratch_pad_size = 0;
  size_t ifm_bo_size = 0;
  size_t ofm_bo_size = 0;

  for (size_t i = 0; i < op_interfaces_.size(); ++i) {
    auto &op_interface = op_interfaces_[i];
    auto args_map_list =
        op_interface->get_buffer_reqs(input, output, op_interface->get_attr());

    for (const auto &args_map : args_map_list) {
      if (args_map.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
        const_params_bo_size += args_map.size;
      }
      if (args_map.arg_type == OpArgMap::OpArgType::SCRATCH_PAD) {
        max_scratch_pad_size = max(max_scratch_pad_size, args_map.size);
      }
      if ((i == 0) && (args_map.arg_type == OpArgMap::OpArgType::INPUT)) {
        ifm_bo_size = args_map.size;
      }
      if ((i == (op_interfaces_.size() - 1)) &&
          (args_map.arg_type == OpArgMap::OpArgType::OUTPUT)) {
        ofm_bo_size = args_map.size;
      }
    }
  }

  /* Sandip TBD: Below if else condition is a workaround. The proper fix is the
   * ofm bo should be given xrt id 2 and scratch bo should be given xrt id 3 for
   * all graphs */
  std::vector<OpArgMap> arg_map;
  if ((graphId_ == 1280) || (graphId_ == 2560)) {
    arg_map = {
        {OpArgMap::OpArgType::INPUT, 1, 0, 0, ifm_bo_size},
        {OpArgMap::OpArgType::CONST_INPUT, 0, 1, 0, const_params_bo_size},
        {OpArgMap::OpArgType::OUTPUT, 3, (op_interfaces_.size() + 1), 0,
         ofm_bo_size},
        {OpArgMap::OpArgType::SCRATCH_PAD, 2, 0, 0, max_scratch_pad_size}};
  } else {
    arg_map = {
        {OpArgMap::OpArgType::INPUT, 1, 0, 0, ifm_bo_size},
        {OpArgMap::OpArgType::CONST_INPUT, 0, 1, 0, const_params_bo_size},
        {OpArgMap::OpArgType::OUTPUT, 2, (op_interfaces_.size() + 1), 0,
         ofm_bo_size},
        {OpArgMap::OpArgType::SCRATCH_PAD, 3, 0, 0, max_scratch_pad_size}};
  }
  return arg_map;
}

static std::string GetParamKey(std::string prefix, int64_t graphId,
                               int64_t inChannels, int64_t outChannels) {
  return prefix + "_" + std::to_string(graphId) + "_" +
         std::to_string(inChannels) + "_" + std::to_string(outChannels);
}

template <typename InT, typename OutT>
void concateOps<InT, OutT>::WriteToFile(void *src, uint64_t length) {
  uint8_t *dataPtr = (uint8_t *)src;
  std::string testDataFolder =
      OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" + "GeneratedWeights";
  std::string fileName =
      testDataFolder + "\\" +
      GetParamKey("wtsGenerated", graphId_, inChannels_, outChannels_) + ".txt";
  write32BitHexTxtFile<uint16_t>(fileName, (uint16_t *)src, length);
}

template <typename InT, typename OutT>
void concateOps<InT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  auto storedDest = dest;
  size_t generatedWeightSize = 0;

  for (size_t i = 0; i < op_interfaces_.size(); ++i) {
    auto &op_interface = op_interfaces_[i];
    std::vector<Tensor> sub_const_params = {const_params.at(i)};
    op_interface->initialize_const_params(dest, sub_const_params,
                                          op_interface->get_attr());
    /* Get buffer sizes required for this operators. We are not using input and
     * output tenosrs in get_buffer_req(). So calling with dummy tensors */
    std::vector<Tensor> input;
    std::vector<Tensor> output;
    auto args_map_list =
        op_interface->get_buffer_reqs(input, output, op_interface->get_attr());

    for (const auto &args_map : args_map_list) {
      if (args_map.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
        dest = static_cast<uint8_t *>(dest) + args_map.size;
        generatedWeightSize += args_map.size;
      }
    }
  }
  if (debug_) {
    WriteToFile(storedDest, (generatedWeightSize >> 1));
  }
}

template <typename InT, typename OutT>
void concateOps<InT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("concateOps initialize_const_params ...");

  /* Get buffer sizes required for this operators. We are not using input and
   * output tenosrs in get_buffer_req(). So calling with dummy tensors */
  std::vector<Tensor> input;
  std::vector<Tensor> output;
  size_t CONST_BO_SIZE, IFM_BO_SIZE, OFM_BO_SIZE, SCRATCH_BO_SIZE;
  CONST_BO_SIZE = IFM_BO_SIZE = OFM_BO_SIZE = SCRATCH_BO_SIZE = 0;
  auto args_map_list = this->get_buffer_reqs(input, output, attr);
  for (const auto &args_map : args_map_list) {
    if (args_map.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
      CONST_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::INPUT) {
      IFM_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::OUTPUT) {
      OFM_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::SCRATCH_PAD) {
      SCRATCH_BO_SIZE = args_map.size;
    }
  }

  RYZENAI_LOG_TRACE("Concatenate: IFM_BO_SIZE:" + std::to_string(IFM_BO_SIZE) +
                    " CONST_BO_SIZE:" + std::to_string(CONST_BO_SIZE) +
                    " OFM_BO_SIZE:" + std::to_string(OFM_BO_SIZE));

  constBo_ =
      xrt::bo(xrt_ctx_->get_device(), CONST_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  ifmBo_ = xrt::bo(xrt_ctx_->get_device(), IFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));
  ofmBo_ = xrt::bo(xrt_ctx_->get_device(), OFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));
  scratchBo_ =
      xrt::bo(xrt_ctx_->get_device(), SCRATCH_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));

  /* Sandip TBD : uint16_t should not be hardcoded */
  uint16_t *b_bo_map = constBo_.map<uint16_t *>();
  initialize_const_params(b_bo_map, const_params);
  constBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  RYZENAI_LOG_TRACE("Concatenate initialize_const_params ... DONE");
}

template <typename InT, typename OutT>
const std::vector<uint8_t> concateOps<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  std::string txn_key =
      "concatenate_" + txn_fname_prefix_ + "_" + std::to_string(graphId_) +
      "_" + std::to_string(inChannels_) + "_" + std::to_string(outChannels_);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Instruction fname : {}", txn_key));
  std::vector<uint8_t> txnData((std::istreambuf_iterator<char>(txn_stream)),
                               std::istreambuf_iterator<char>());
  return txnData;
}

template <typename InT, typename OutT>
void concateOps<InT, OutT>::execute(const std::vector<Tensor> &input,
                                    std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("Conv execute ...");

  xrt::bo instr_bo; // = nullptr;
  xrt::bo param_bo;

  auto ifmDtype = input.at(0).dtype;
  size_t ifmDSize;
  if (ifmDtype == "uint16") {
    ifmDSize = 2;
  } else if (ifmDtype == "uint8") {
    ifmDSize = 1;
  }
  size_t ifmDataSize = input.at(0).shape[0] * input.at(0).shape[1] *
                       input.at(0).shape[2] * ifmDSize;
  ifmBo_.write(input.at(0).data, ifmDataSize, 0);
  ifmBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto instr_bo_key =
      "concatenate_" + txn_fname_prefix_ + "_" + std::to_string(graphId_) +
      "_" + std::to_string(inChannels_) + "_" + std::to_string(outChannels_);
  instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;
  int instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();
  xrt::run run;
  // launch the Conv kernel
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  /* kernel call for Conv that supports transaction binary flow */

  /* Sandip TBD: Below if else condition is a workaround. The proper fix is the
   * ofm bo should be given xrt id 2 and scratch bo should be given xrt id 3 for
   * all graphs */
  if ((graphId_ == 1280) || (graphId_ == 2560)) {
    run = kernel_(2, instr_bo, instr_bo_words,
                  constBo_.address() + DDR_AIE_ADDR_OFFSET,
                  ifmBo_.address() + DDR_AIE_ADDR_OFFSET,
                  scratchBo_.address() + DDR_AIE_ADDR_OFFSET,
                  ofmBo_.address() + DDR_AIE_ADDR_OFFSET, 0);
  } else {
    run = kernel_(2, instr_bo, instr_bo_words,
                  constBo_.address() + DDR_AIE_ADDR_OFFSET,
                  ifmBo_.address() + DDR_AIE_ADDR_OFFSET,
                  ofmBo_.address() + DDR_AIE_ADDR_OFFSET,
                  scratchBo_.address() + DDR_AIE_ADDR_OFFSET, 0);
  }
  run.wait2();
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += run_aie_stop - run_aie_start;

  // sync output activation to host memory
  ofmBo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  ofmBo_.read(output.at(0).data);

  RYZENAI_LOG_TRACE("Conv execute ... DONE");
}

template <typename InT, typename OutT>
std::once_flag concateOps<InT, OutT>::logger_flag_;

template <typename InT, typename OutT>
uint64_t concateOps<InT, OutT>::concatenate_count = 0;

template <typename InT, typename OutT>
std::once_flag concateOps<InT, OutT>::instr_reg_flag_;

template class concateOps<uint16_t, uint16_t>;

} // namespace ryzenai
