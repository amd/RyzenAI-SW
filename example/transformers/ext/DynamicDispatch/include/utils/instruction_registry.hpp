/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef DYNAMIC_DISPATCH_UTILS_INSTRUCTION_REGISTRY_H
#define DYNAMIC_DISPATCH_UTILS_INSTRUCTION_REGISTRY_H

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utils/txn_container.hpp>
// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Subroutines to read the transaction binary
#include "op_buf.hpp"
#include "op_types.h"

#include "xaiengine.h"

#include "logging.hpp"
#include <xrt_context/xrt_context.hpp>
namespace ryzenai::dynamic_dispatch {

class instruction_registry {
private:
  Transaction &txn = Transaction::getInstance();

public:
  std::shared_ptr<xrt_context> xrt_ctx_;
  std::map<std::string, std::pair<bool, xrt::bo>> instr_map_;
  std::map<std::string, std::pair<bool, xrt::bo>> params_map_;

  void insert_to_instruction_map(std::pair<std::string, bool> instr) {
    std::string txn_string = txn.get_txn_str(instr.first);
    std::istringstream txn_bin(txn_string, std::ios::binary);
    if (txn_bin.fail()) {
      throw std::runtime_error("Failed to open txn binary file: ");
    }

    std::vector<char> prm_buffer(txn_string.begin(), txn_string.end());
    XAie_TxnHeader *hdr = new XAie_TxnHeader();
    txn_bin.read((char *)hdr, sizeof(XAie_TxnHeader));
    uint8_t *ptr = new uint8_t[hdr->TxnSize];
    std::memcpy(ptr, hdr, sizeof(XAie_TxnHeader));
    uint8_t *txn_ptr = ptr + sizeof(*hdr);
    txn_bin.read((char *)txn_ptr, hdr->TxnSize - sizeof(XAie_TxnHeader));

    aiectrl::op_buf instr_buf;
    instr_buf.addOP(aiectrl::transaction_op(ptr));
    size_t instr_bo_words = instr_buf.ibuf_.size();
    xrt::bo instr_bo =
        xrt::bo(xrt_ctx_->get_context(), instr_bo_words,
                xrt::bo::flags::cacheable, xrt_ctx_->get_kernel().group_id(1));
    instr_bo.write(instr_buf.ibuf_.data());
    instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    RYZENAI_LOG_TRACE("[INSTR_REG] instr_bo created and saved: " + instr.first);

    instr_map_.insert({instr.first, std::make_pair(instr.second, instr_bo)});

    delete[] ptr;
    delete hdr;
  }

  void insert_to_layer_params_map(std::pair<std::string, bool> params) {

    std::string layer_params = txn.get_txn_str(params.first);
    std::vector<char> prm_buffer(layer_params.begin(), layer_params.end());
    size_t prm_size = prm_buffer.size();
    xrt::bo param_bo =
        xrt::bo(xrt_ctx_->get_context(), prm_size, xrt::bo::flags::host_only,
                xrt_ctx_->get_kernel().group_id(8));
    param_bo.write(prm_buffer.data());
    param_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    RYZENAI_LOG_TRACE("[INSTR_REG] instr_bo created and saved: " +
                      params.first);

    params_map_.insert({params.first, std::make_pair(params.second, param_bo)});
  }

  void setup_hw_ctx(std::shared_ptr<xrt_context> ctx) { xrt_ctx_ = ctx; }

  void add_instructions(vector<std::pair<std::string, bool>> instr) {
    for (auto &i : instr) {
      insert_to_instruction_map(i);
    }
  }

  void add_layer_params(vector<std::pair<std::string, bool>> params) {
    for (auto &i : params) {
      insert_to_layer_params_map(i);
    }
  }

  std::pair<bool, xrt::bo> get_instr_bo(std::string key) {
    auto val = instr_map_.find(key);
    if (val == instr_map_.end()) {
      throw runtime_error("Failed to get instruction buffer for key: " + key);
    }
    return val->second;
  }

  std::pair<bool, xrt::bo> get_param_bo(std::string key) {
    auto val = params_map_.find(key);
    if (val == params_map_.end()) {
      throw runtime_error("Failed to get instruction buffer for key: " + key);
    }
    return val->second;
  }
};

} // namespace ryzenai::dynamic_dispatch

#endif /* DYNAMIC_DISPATCH_UTILS_INSTRUCTION_REGISTRY_H */
