#pragma once

#include <string>
#include <utility>
#include <vector>

#include <op_fuser/fuse_ops.hpp>
#include <ops/op_interface.hpp>

namespace OpsFusion {
struct Metadata;

struct DDConfig {
  uint32_t profile =
      0; // pass profile level. 0 - None, 1 - subgraph, 2 - subgraph+PDI
         // partition, 3 - subgraph + PDI partition + ops
  bool pm_swap = false;
  bool optimize_scratch = true;
  // use fused transaction, but run each op serially
  bool eager_mode = false;
};

class FusionRuntime {
public:
  FusionRuntime(const std::string &xclbin,
                const std::string &kernel_name_prefix = "DPU");
  FusionRuntime(xrt::hw_context *ctx,
                const std::string &kernel_name_prefix = "DPU");
  ~FusionRuntime();
  // Do not allow copying or assignment
  // to prevent instruction BO for hw_context to grow
  FusionRuntime(const FusionRuntime &) = delete;
  FusionRuntime &operator=(const FusionRuntime &) = delete;
  void execute(const std::vector<Tensor> &inputs,
               const std::vector<Tensor> &outputs);
  void init(const Metadata &meta, const std::string &base_dir = "",
            const DDConfig &cfg = {});
  std::vector<std::vector<uint8_t>> get_txns();
  const Metadata &get_meta() const;

  // Unpack internal buffers of RT
  // This is useful for debugging after the execution.
  // Its result is meaningful only after execution.
  // TODO : Testing
  std::map<std::string, std::vector<uint8_t>>
  unpack_internal_buffers(const std::string &dir = "");

private:
  void load_const(const Metadata &meta);
  void fill_super_instr(const Metadata &meta);
  void setup_xrt_run(const Metadata &meta);
  void split_outputs(const std::vector<Tensor> &outputs, const Metadata &meta);
  void merge_inputs(const std::vector<Tensor> &inputs, const Metadata &meta);
  std::vector<std::vector<uint8_t>> generate_fused_txns(const Metadata &meta);
  bool check_context_instr_size(
      const std::vector<std::vector<uint8_t>> &fused_instr_vec,
      const size_t limit);
  bool check_partition_instr_size(
      const std::vector<std::vector<uint8_t>> &fused_instr_vec,
      const size_t partition_limit);
  bool
  allocate_instr_bos(const std::vector<std::vector<uint8_t>> &fused_instr_vec);
  void
  populate_instr_bos(const std::vector<std::vector<uint8_t>> &fused_instr_vec);
  void reallocate_data_bos(const Metadata &meta);
  void initialize_inputs(const Metadata &meta);

private:
  static std::once_flag logger_flag_;

  // External Context
  xrt::hw_context ctx_;
  std::vector<xrt::kernel> kernels_;
  std::vector<xrt::run> runs_;

  Metadata meta_;
  std::vector<xrt::bo> instr_bos_;
  xrt::bo input_bo_;
  xrt::bo output_bo_;
  xrt::bo scratch_bo_;
  xrt::bo const_bo_;
  xrt::bo super_instr_bo_;

  // Config
  DDConfig cfg_;

  // TODO : calling .size() on an empty bo throws exception
  size_t instr_bo_sz_{0};
  size_t input_bo_sz_{0};
  size_t output_bo_sz_{0};
  size_t scratch_bo_sz_{0};
  size_t const_bo_sz_{0};
  size_t super_instr_bo_sz_{0};
  // TODO: can we only keep fused_instr_vec_ ??
  std::vector<std::vector<uint8_t>> txns_;
  std::vector<std::vector<uint8_t>> fused_instr_vec_;

  // Timers
  int64_t input_copy_time_{0};
  int64_t input_sync_time_{0};
  int64_t output_copy_time_{0};
  int64_t output_sync_time_{0};
  int64_t xrt_exec_time_{0};

  // make copying to input BO, from output BO thread safe
  std::mutex execute_mutex_;
  // Fallback to dynamically updating instr bo
  bool use_instr_sw_cache_;
};

} // namespace OpsFusion
