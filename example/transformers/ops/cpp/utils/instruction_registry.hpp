/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __INSTRUCTION_REGISTRY_H__
#define __INSTRUCTION_REGISTRY_H__

#include <fstream>
#include <iostream>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Subroutines to read the transaction binary
#include "op_buf.hpp"
#include "op_types.h"

#include "xaiengine.h"

#include "logging.h"
#include "xrt_context.hpp"

namespace ryzenai {
class instruction_registry {
public:
  instruction_registry() {}
  void setup_hw_ctx(xrt_context *ctx) { xrt_ctx_ = ctx; }

  void add_instructions(vector<std::pair<std::string, bool>> instr,
                        std::string dir) {
    for (auto &i : instr) {
      insert_to_instruction_map(i, dir);
    }
  }

  std::pair<bool, xrt::bo> get_instr_bo(std::string key) {
    auto val = instr_map_.find(key);
    if (val == instr_map_.end()) {
      throw runtime_error("Failed to get instruction buffer for key: " + key);
    }
    return val->second;
  }

private:
  xrt_context *xrt_ctx_;
  std::map<std::string, std::pair<bool, xrt::bo>> instr_map_;
  void instruction_registry::insert_to_instruction_map(
      std::pair<std::string, bool> instr, std::string dir) {
    auto txn_bin_fname = instr.first;
    std::ifstream txn_bin(dir + txn_bin_fname, std::ios::binary);
    if (txn_bin.fail()) {
      throw std::runtime_error("Failed to open transaction binary file: " +
                               txn_bin_fname);
    }

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
  };
};

} // namespace ryzenai

#endif /* __INSRTUCTION_REGISTRY_H__ */
