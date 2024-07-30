#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <xaiengine.h>

#include <op_fuser/fuse_types.hpp>
#include <ops/op_interface.hpp>
#include <ps/op_types.h>

namespace utils {

class txn_util {
public:
  txn_util() = default;
  txn_util(std::string file_name);
  txn_util(const std::vector<uint8_t> &txn_vec);
  std::string summarize();
  std::string text_dump();
  void patch(const OpsFusion::Metadata::OpInfo &op_info,
             const OpsFusion::Metadata &meta,
             const std::vector<OpArgMap> &args_map);
  void write_fused_txn_to_file(std::string file_name);
  ~txn_util();
  std::vector<uint8_t> to_vector();
  static std::vector<uint8_t>
  fuse_txns(const std::vector<std::vector<uint8_t>> &txns);
  static void pass_through(uint8_t **ptr);
  uint8_t *txn_ptr_ = nullptr;
  uint8_t *fused_txn_ptr_ = nullptr;

private:
  std::stringstream ss_hdr_;
  std::stringstream ss_ops_;
  std::stringstream ss_summary_;
  uint64_t txn_size_;
  uint64_t num_txn_ops_;
  uint64_t fused_size_;
  uint64_t fused_ops_;

  uint32_t num_w_ops = 0;
  uint32_t num_bw_ops = 0;
  uint32_t num_mw_ops = 0;
  uint32_t num_mp_ops = 0;
  uint32_t num_tct_ops = 0;
  uint32_t num_patch_ops = 0;
  uint32_t num_read_ops = 0;
  uint32_t num_readtimer_ops = 0;
  uint32_t num_mergesync_ops = 0;

  void stringify_txn_ops();
  void stringify_w32(uint8_t **ptr);
  void stringify_bw32(uint8_t **ptr);
  void stringify_mw32(uint8_t **ptr);
  void stringify_mp32(uint8_t **ptr);
  void stringify_tct(uint8_t **ptr);
  void stringify_patchop(uint8_t **ptr);
  void stringify_rdreg(uint8_t **ptr);
  void stringify_rectimer(uint8_t **ptr);
  void stringify_mergesync(uint8_t **ptr);
  void stringify_txn_bin();
  void prepare_summary();
};

} // namespace utils

#endif /* __UTILS_HPP__ */
