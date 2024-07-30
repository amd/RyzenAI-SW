#include <algorithm>
#include <mutex>
#include <op_fuser/fuse_ops.hpp>
#include <op_fuser/fusion_rt.hpp>
#include <ops/op_builder.hpp>

#include <ps/op_buf.hpp>
#include <ps/op_init.hpp>
#include <ps/op_types.h>

#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>

#include "passes/passes.hpp"
#include "txn/txn_utils.hpp"
#include "utils/dpu_mdata.hpp"

#include <experimental/xrt_error.h>

#ifdef SIMNOWLITE_EN
static constexpr uint32_t HOST_BO_GROUP_ID = 8;
#else
static constexpr uint32_t HOST_BO_GROUP_ID = 0;
#endif

static constexpr size_t XRT_BO_MIN_SIZE = 4096; // Bytes
static constexpr size_t XRT_BO_INIT_VALUE = 0;

static constexpr size_t INSTR_XRT_BO_MAX_SIZE = 48ULL * 1024ULL * 1024ULL;
static constexpr size_t INSTR_XRT_BO_STACK_SIZE = 12ULL * 1024ULL * 1024ULL;
static constexpr size_t NUM_STATIC_INSTR_BUFFERS = 2;
static constexpr size_t INSTR_BUFFER_SIZE =
    INSTR_XRT_BO_STACK_SIZE / NUM_STATIC_INSTR_BUFFERS;
static constexpr size_t INSTR_XRT_BO_HEAP_SIZE =
    INSTR_XRT_BO_MAX_SIZE - INSTR_XRT_BO_STACK_SIZE;
static constexpr size_t INSTR_XRT_BO_ALIGNMENT = 32ULL * 1024ULL;

struct XRTBufferState {
  // how much memory is currently used by this context
  size_t heap_total_size = 0;
  // for debug - how many BOs have already been allocated
  size_t num_instr_bos = 0;
  // global instr BO per xrt hw context
  std::vector<xrt::bo> static_instr_bos;
  // actual size of instruction contents
  std::vector<size_t> static_instr_sizes;
};

// NOTE: current access to this is NOT THREAD SAFE!!!
static std::map<xrt_core::hwctx_handle *, XRTBufferState> xrt_instr_state;
static std::mutex instr_state_mutex;

static bool enable_write_internal_bufs = static_cast<bool>(
    std::stol(Utils::get_env_var("DD_WRITE_INTERNAL_BUFS", "0")));

static size_t compute_hash(const void *data, size_t size) {
  std::hash<std::string_view> hash;
  return hash(std::string_view((const char *)data, size));
}

static std::string replace_characters(const std::string &name,
                                      const std::string &pattern,
                                      char replacement) {
  std::string::size_type pos{0};
  std::string res(name);
  while (true) {
    pos = name.find_first_of(pattern, pos);
    if (pos == std::string::npos) {
      break;
    }
    res[pos] = replacement;
    pos++;
  }
  return res;
}

namespace OpsFusion {

// Depad & Copy data from src to dst buffer.
// So src is larger than
static void copy_data(const Tensor &src_tensor, const Tensor &dst_tensor) {
  size_t src_sz =
      std::accumulate(src_tensor.shape.begin(), src_tensor.shape.end(),
                      size_t{1}, std::multiplies{}) *
      Utils::get_size_of_type(src_tensor.dtype);
  memcpy(dst_tensor.data, src_tensor.data, src_sz);
}

static void write_to_bo(xrt::bo &dst_bo, size_t offset, const void *src,
                        size_t size) {
  dst_bo.write(src, size, offset);
  dst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

static void read_from_bo(xrt::bo &src_bo, size_t offset, void *dst,
                         size_t size) {
  src_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  src_bo.read(dst, size, offset);
}

template <typename T> static float sum_bo(xrt::bo &src) {
  T *ptr = src.map<T *>();
  size_t size = src.size() / sizeof(T);
  return std::accumulate(ptr, ptr + size, 0.0f,
                         [](float a, T b) { return a + b; });
}

template <typename T> static int sum_bo_int(xrt::bo &src) {
  T *ptr = src.map<T *>();
  size_t size = src.size() / sizeof(T);
  int sum = 0;
  for (int i = 0; i < size; ++i) {
    sum += (int)ptr[i];
  }
  return sum;
}

FusionRuntime::FusionRuntime(const std::string &xclbin,
                             const std::string &kernel_name_prefix)
    : ctx_(ryzenai::dynamic_dispatch::xrt_context::get_instance(xclbin)
               ->get_context()) {
  std::vector<std::string> kernel_names;

  for (const auto &kernel : ctx_.get_xclbin().get_kernels()) {
    const auto kernel_name = kernel.get_name();

    if (kernel_name.rfind(kernel_name_prefix, 0) == 0) {
      RYZENAI_LOG_TRACE(OpsFusion::dod_format(
          "FusionRuntime : found kernel : {}", kernel_name));
      kernel_names.push_back(kernel_name);
    }
  }

  // In case kernel names arent in order within xclbin
  std::sort(kernel_names.begin(), kernel_names.end());

  for (const auto &kernel_name : kernel_names) {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "FusionRuntime : Creating kernel object : {}", kernel_name));
    xrt::kernel k(ctx_, kernel_name);
    kernels_.push_back(k);
    runs_.emplace_back(k);
  }

  xrt_core::hwctx_handle *handle = static_cast<xrt_core::hwctx_handle *>(ctx_);

  std::call_once(logger_flag_, []() {
    std::string header = "Graph/PDI_Partition, xrt execute time(ns), "
                         "in_copy_time(ns), in_sync_time(ns), "
                         "out_copy_time(ns), out_sync_time(ns), json_path\n";
    RYZENAI_LOG_INFO(header);
  });

  // make inserting into global state here thread safe
  std::lock_guard<std::mutex> guard(instr_state_mutex);

  if (xrt_instr_state.find(handle) == xrt_instr_state.end()) {
    xrt_instr_state[handle] = XRTBufferState{};
    for (size_t i = 0; i < NUM_STATIC_INSTR_BUFFERS; i++) {
      xrt_instr_state.at(handle).static_instr_bos.emplace_back(
          xrt::bo(ctx_, INSTR_BUFFER_SIZE, xrt::bo::flags::cacheable,
                  kernels_[0].group_id(1)));
      xrt_instr_state.at(handle).static_instr_sizes.push_back(0);
    }
    xrt_instr_state.at(handle).num_instr_bos = 2;
  }
}

FusionRuntime::FusionRuntime(xrt::hw_context *ctx,
                             const std::string &kernel_name_prefix)
    : ctx_(*ctx) {
  std::vector<std::string> kernel_names;

  for (const auto &kernel : ctx_.get_xclbin().get_kernels()) {
    const auto kernel_name = kernel.get_name();

    if (kernel_name.rfind(kernel_name_prefix, 0) == 0) {
      RYZENAI_LOG_TRACE(OpsFusion::dod_format(
          "FusionRuntime : found kernel : {}", kernel_name));
      kernel_names.push_back(kernel_name);
    }
  }

  // In case kernel names arent in order within xclbin
  std::sort(kernel_names.begin(), kernel_names.end());

  for (const auto &kernel_name : kernel_names) {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "FusionRuntime : Creating kernel object : {}", kernel_name));
    xrt::kernel k(ctx_, kernel_name);
    kernels_.push_back(k);
    runs_.emplace_back(k);
  }

  xrt_core::hwctx_handle *handle = static_cast<xrt_core::hwctx_handle *>(ctx_);

  std::call_once(logger_flag_, []() {
    std::string header = "Graph/PDI_Partition, xrt execute time(ns), "
                         "in_copy_time(ns), in_sync_time(ns), "
                         "out_copy_time(ns), out_sync_time(ns), json_path\n";
    RYZENAI_LOG_INFO(header);
  });

  // make inserting into global state here thread safe
  std::lock_guard<std::mutex> guard(instr_state_mutex);

  if (xrt_instr_state.find(handle) == xrt_instr_state.end()) {
    xrt_instr_state[handle] = XRTBufferState{};
    for (size_t i = 0; i < NUM_STATIC_INSTR_BUFFERS; i++) {
      xrt_instr_state.at(handle).static_instr_bos.emplace_back(
          xrt::bo(ctx_, INSTR_BUFFER_SIZE, xrt::bo::flags::cacheable,
                  kernels_[0].group_id(1)));
      xrt_instr_state.at(handle).static_instr_sizes.push_back(0);
    }
    xrt_instr_state.at(handle).num_instr_bos = 2;
  }
}

FusionRuntime::~FusionRuntime() {
  xrt_core::hwctx_handle *handle = static_cast<xrt_core::hwctx_handle *>(ctx_);

  // make book-keeping here thread safe
  // heap_total_size is used to determine if we should
  // use static instruction BO or not
  std::lock_guard<std::mutex> guard(instr_state_mutex);
  for (auto &instr_bo : instr_bos_) {
    xrt_instr_state.at(handle).heap_total_size -=
        Utils::align_to_next(instr_bo.size(), INSTR_XRT_BO_ALIGNMENT);
  }
  xrt_instr_state.at(handle).num_instr_bos -= instr_bos_.size();
}

void FusionRuntime::execute(const std::vector<Tensor> &inputs,
                            const std::vector<Tensor> &outputs) {
  // this lock guard serves 2 purposes
  //  - Make copying in inputs and outputs out thread-safe
  //    when there is single FusionRuntime object, but multiple theads call
  //    execute
  //  - For multiple partitions, which would need to be
  //    run together
  std::lock_guard<std::mutex> guard(execute_mutex_);
  const auto &meta = meta_;
  merge_inputs(inputs, meta);

  if (enable_write_internal_bufs) {
    unpack_internal_buffers("tmp/dd_bufs_pre_exec");
  }
  xrt_exec_time_ = 0;

  xrt_core::hwctx_handle *handle = static_cast<xrt_core::hwctx_handle *>(ctx_);
  size_t cache_instr_idx = 0;

  if (use_instr_sw_cache_) {
    // NOTE: this writes to BO and does sync
    std::lock_guard<std::mutex> guard(instr_state_mutex);
    write_to_bo(xrt_instr_state.at(handle).static_instr_bos[cache_instr_idx],
                0, /*offset*/
                fused_instr_vec_.at(0).data(), fused_instr_vec_.at(0).size());
    xrt_instr_state.at(handle).static_instr_sizes[cache_instr_idx] =
        fused_instr_vec_.at(0).size();

    for (size_t i = 0; i < meta.partitions.size(); i++) {
      auto pdi_id = meta.partitions[i].pdi_id;
      auto exec_start = GET_ELAPSED_TIME_NS();
      size_t instr_idx = cache_instr_idx;
      cache_instr_idx = (cache_instr_idx + 1) % NUM_STATIC_INSTR_BUFFERS;
      bool prefetch_instr = (i != (meta.partitions.size() - 1));
      try {
        runs_[pdi_id].set_arg(
            1, xrt_instr_state.at(handle).static_instr_bos[instr_idx]);
        runs_[pdi_id].set_arg(
            2, xrt_instr_state.at(handle).static_instr_sizes[instr_idx] /
                   sizeof(int));
        runs_[pdi_id].start();
        // try to overlap instruction copying with AIE execution
        if (prefetch_instr) {
          write_to_bo(
              xrt_instr_state.at(handle).static_instr_bos[cache_instr_idx],
              0, /*offset*/
              fused_instr_vec_.at(i + 1).data(),
              fused_instr_vec_.at(i + 1).size());
          xrt_instr_state.at(handle).static_instr_sizes[cache_instr_idx] =
              fused_instr_vec_.at(i + 1).size();
        }
        runs_[pdi_id].wait2();
      } catch (const std::exception &e) {

#ifdef RYZENAI_DEBUG
        std::cout << "Running under debug mode...  Hardware context handle = "
                  << ctx_.get_handle() << ", PID = " << _DD_GET_PID()
                  << std::endl;
        std::cout << "Will wait for user input." << std::endl;
        std::cin.get();
#endif

        std::cerr << "ERROR: Kernel partition: " << i
                  << ", pdi_id: " << (std::uint32_t)pdi_id << " timed out!"
                  << std::endl;
        std::cerr << "Details: " << e.what() << std::endl;

        xrt::error err = xrt::error(ctx_.get_device(), XRT_ERROR_CLASS_AIE);
        if (err.get_error_code()) {
          std::string err_message =
              std::string("Error while executing pdi_id: ") +
              std::to_string((std::uint32_t)pdi_id) +
              ", partition: " + std::to_string(i) +
              ", info: " + err.to_string();
          std::cerr << err_message << std::endl;
          RYZENAI_LOG_TRACE(err_message);
        }

        DOD_THROW(OpsFusion::dod_format(
            "Kernel partition: {} pdi_id: {} timeout (Detail : {})", i,
            (std::uint32_t)pdi_id, e.what()));
      }
      auto exec_end = GET_ELAPSED_TIME_NS();
      int64_t partition_exec_time = exec_end - exec_start;
      xrt_exec_time_ += partition_exec_time;
      RYZENAI_LOG_INFO("PDI_Partition " + std::to_string(i) + " , " +
                       std::to_string(partition_exec_time) + ", " +
                       std::to_string(0) + ", " + std::to_string(0) + ", " +
                       std::to_string(0) + ", " + std::to_string(0) + ", " +
                       meta.json_path + "\n");
    }
  } else {
    for (size_t i = 0; i < meta.partitions.size(); i++) {
      auto pdi_id = meta.partitions[i].pdi_id;
      auto exec_start = GET_ELAPSED_TIME_NS();
      size_t instr_idx = i;
      try {

        runs_[pdi_id].set_arg(1, instr_bos_[instr_idx]);
        runs_[pdi_id].set_arg(2, instr_bos_[instr_idx].size() / sizeof(int));

        runs_[pdi_id].start();
        runs_[pdi_id].wait2();
      } catch (const std::exception &e) {

#ifdef RYZENAI_DEBUG
        std::cout << "Running under debug mode...  Hardware context handle = "
                  << ctx_.get_handle() << ", PID = " << _DD_GET_PID()
                  << std::endl;
        std::cout << "Will wait for user input." << std::endl;
        std::cin.get();
#endif

        std::cerr << "ERROR: Kernel partition: " << i
                  << ", pdi_id: " << (std::uint32_t)pdi_id << " timed out!"
                  << std::endl;
        std::cerr << "Details: " << e.what() << std::endl;

        xrt::error err = xrt::error(ctx_.get_device(), XRT_ERROR_CLASS_AIE);
        if (err.get_error_code()) {
          std::string err_message =
              std::string("Error while executing pdi_id: ") +
              std::to_string((std::uint32_t)pdi_id) +
              ", partition: " + std::to_string(i) +
              ", info: " + err.to_string();
          std::cerr << err_message << std::endl;
          RYZENAI_LOG_TRACE(err_message);
        }

        DOD_THROW(OpsFusion::dod_format(
            "Kernel partition: {} pdi_id: {} timeout (Detail : {})", i,
            (std::uint32_t)pdi_id, e.what()));
      }
      auto exec_end = GET_ELAPSED_TIME_NS();
      int64_t partition_exec_time = exec_end - exec_start;
      xrt_exec_time_ += partition_exec_time;
      RYZENAI_LOG_INFO("PDI_Partition " + std::to_string(i) + " , " +
                       std::to_string(partition_exec_time) + ", " +
                       std::to_string(0) + ", " + std::to_string(0) + ", " +
                       std::to_string(0) + ", " + std::to_string(0) + ", " +
                       meta.json_path + "\n");
    }
  }

  if (enable_write_internal_bufs) {
    unpack_internal_buffers("tmp/dd_bufs_post_exec");
  }

  split_outputs(outputs, meta);

  RYZENAI_LOG_INFO("Graph, " + std::to_string(xrt_exec_time_) + ", " +
                   std::to_string(input_copy_time_) + ", " +
                   std::to_string(input_sync_time_) + ", " +
                   std::to_string(output_copy_time_) + ", " +
                   std::to_string(output_sync_time_) + ", " + meta.json_path +
                   "\n");
}

void FusionRuntime::init(const Metadata &meta, const std::string &base_dir,
                         const DDConfig &cfg) {
  // TODO : Need a way to compare if metadata is same as old, and if so skip.
  RYZENAI_LOG_TRACE("FusionRuntime : Init ...");
  meta_ = meta;
  cfg_ = cfg;

  // if env variables are set, update cfg_
  // check if profile option is enabled using env varaibles
  auto profile_level = Utils::get_env_var("DD_ENABLE_PROFILE");
  if (profile_level != "") {
    auto p_lvl = std::stoi(profile_level);
    cfg_.profile = std::min(3, p_lvl);
  }

  RYZENAI_LOG_TRACE(dod_format("Setting profile level to {}", cfg_.profile));

  OpInterface::set_dod_base_dir(base_dir);

  Metadata mdata = meta;

  // partition ops to PDIs
  OpPDIMap op_pdi_map = {{{{"Add", 0},
                           {"DQAdd", 0},
                           {"LayerNorm", 0},
                           {"QLayerNorm", 0},
                           {"QGroupNorm", 0},
                           {"MatMul", 0},
                           {"QMatMul", 0},
                           {"MatMulAdd", 0},
                           {"QMatMulAdd", 0},
                           {"MatMulAddGelu", 0},
                           {"QMatMulAddGelu", 0},
                           {"MladfMatMul", 0},
                           {"MHAGRPB", 0},
                           {"QMHAGRPB", 0},
                           {"QEltWiseAdd", 0},
                           {"QMHAWINDOW", 0},
                           {"QMHACHANNEL", 0},
                           {"QELWEMUL_qdq", 0},
                           {"SILU", 0},
                           {"ELWMUL", 0},
                           {"MLADFADD", 0},
                           {"MLADFRMSNORM", 0},
                           {"MASKEDSOFTMAX", 0},
                           {"MLADFMHAROPE", 0},
                           {"QConv", 1},
                           {"QConcateOPs", 0},
                           {"IConv", 0},
                           {"QReshapeTranspose", 0},
                           {"square", 0},
                           {"cube", 0},
                           {"QMHA", 0},
                           {"QGlobalAvgPool", 1},
                           {"xcom-conv2d", 0},
                           {"QConv2MatMul", 0},
                           {"QMatMulDynamic", 0},
                           {"QMulSoftmax", 0},
                           {"PSRMHA", 0},
                           {"QSilu", 0},
                           {"QSlice", 0},
                           {"QConcat", 0},
                           {"QResize", 0},
                           {"QBroadcastAdd", 0},
                           {"QGelu", 0},
                           {"Mladfsoftmax", 0},
                           {"MLADFMATMULA16A16", 0},
                           {"MLADFMATMULA16W8", 0},
                           {"QLstm", 0},
                           {"Mladfelwadd", 0},
                           {"Mladfelwmul", 0}}},
                         {{{0, "DPU"}, {1, "DPU_1"}}}};

  assign_pdi_id_pass(op_pdi_map, meta_);

  if (cfg_.pm_swap) {
    meta_ = insert_pm_swap_nodes(meta_);
  }

  generate_pdi_partitions_pass(meta_, cfg_.eager_mode);
  if (cfg_.profile) {
    meta_ = insert_record_timer_nodes(meta_, cfg_.profile);
    generate_pdi_partitions_pass(meta_, cfg_.eager_mode);
  }

  analyze_buffer_reqs(meta_);

  if (cfg_.optimize_scratch) {
    optimize_scratch_buffer(meta_);
  }

  fused_instr_vec_ = generate_fused_txns(meta_);
  {
    // this block determines if we should either use "heap" or "stack" for
    // instruction BO since this a global state, add lock guard e.g. trying to
    // read map but another thread does an insertion which cause reallocation
    std::lock_guard<std::mutex> guard(instr_state_mutex);
    bool repartition_instr =
        check_context_instr_size(fused_instr_vec_, INSTR_XRT_BO_HEAP_SIZE);

    bool need_realloc = false;

    do {
      repartition_instr = repartition_instr || need_realloc;

      use_instr_sw_cache_ = repartition_instr;

      while (repartition_instr) {
        // this is to ensure current set of txn binaries fit into
        // the static instr BO's
        bool split = split_max_partition_pass(meta_, fused_instr_vec_,
                                              INSTR_BUFFER_SIZE);
        DOD_THROW_IF(!split,
                     OpsFusion::dod_format("Instruction partition failed!"));
        fused_instr_vec_ = generate_fused_txns(meta_);
        repartition_instr =
            check_partition_instr_size(fused_instr_vec_, INSTR_BUFFER_SIZE);
      }

      need_realloc = allocate_instr_bos(fused_instr_vec_);
    } while (need_realloc);
  }
  // this is no-op if using static instr BO - hence no guard
  populate_instr_bos(fused_instr_vec_);

  const auto &new_meta = meta_;
  reallocate_data_bos(new_meta);
  initialize_inputs(new_meta);
  load_const(new_meta);
  fill_super_instr(new_meta);
  setup_xrt_run(new_meta);

  RYZENAI_LOG_TRACE("FusionRuntime : Init ... DONE");
}

// For every Op, collect all the const data it has.
// Pass everything to the OpInterface and let it copy to
//    the right place.
void FusionRuntime::load_const(const Metadata &meta) {
  RYZENAI_LOG_TRACE("FusionRuntime : Load const ...");
  void *const_bo_ptr = const_bo_.map();

  for (const auto &op_info : meta.op_list) {
    // get all const buffers' name
    std::vector<std::string> tensor_names;
    for (const auto &buf_name : op_info.args) {
      const auto &tensor_info = MAP_AT(meta.tensor_map, buf_name);
      if (tensor_info.parent_name != "const") {
        continue;
      }
      tensor_names.push_back(buf_name);
    }

    // Load the const data from disk
    std::vector<vector<char>> const_data;
    std::vector<Tensor> const_tensors;
    // Read const inputs from files only if tensor_names is not empty.
    // initialize_const_params() is called regardless for each op later.
    // This enabled operators to copy LUTs / other data to AIE. This is required
    // for operators like bf16 Silu/Gelu when ONNX op does not have a constant
    // input.
    if (!tensor_names.empty()) {
      for (const auto &name : tensor_names) {
        const auto &tensor_info = meta.tensor_map.at(name);
        std::vector<char> data = read_bin_file(tensor_info.file_name);
        const_data.push_back(std::move(data));
        const_tensors.push_back(
            {const_data.back().data(), tensor_info.shape, tensor_info.dtype});
      }
    }

    // Get the offset of this op's const buffer in const bo.
    size_t offset = 0;
    // if const_map is empty, call all initilize_const_params for constant
    // initilization, if any.
    if (meta.const_map.find(op_info.name) != meta.const_map.end()) {
      const auto &tensor_info = meta.const_map.at(op_info.name);
      offset = tensor_info.offset;
    }

    auto op = OpBuilder::create(op_info.name, op_info, meta.tensor_map);
    using signature = void(void *, const std::vector<Tensor> &,
                           const std::map<std::string, std::any> &);
    DD_INVOKE_OVERLOADED_OPMETHOD(initialize_const_params, signature, op.get(),
                                  op_info, (char *)const_bo_ptr + offset,
                                  const_tensors, op_info.attr);
  }

  const_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  RYZENAI_LOG_TRACE("FusionRuntime : Load const ... DONE");
}

void FusionRuntime::fill_super_instr(const Metadata &meta) {
  RYZENAI_LOG_TRACE("FusionRuntime : Fill Super Instrns ... ");
  void *super_bo_ptr = super_instr_bo_.map();
  for (const auto &op_info : meta.op_list) {
    auto op = OpBuilder::create(op_info.name, op_info, meta.tensor_map);
    auto offset = MAP_AT(meta.super_instr_map, op_info.name).offset;
    auto const_buffers = MetaUtils::load_op_const_buffers(meta, op_info);
    std::map<std::string, void *> const_buf_ptrs;
    for (auto &[name, buffer] : const_buffers) {
      const_buf_ptrs[name] = buffer.data();
    }
    std::vector<Tensor> tensors =
        MetaUtils::collect_op_tensors(meta, op_info, const_buf_ptrs);

    auto super_instr =
        DD_INVOKE_OPMETHOD(get_super_kernel_params, op.get(), op_info, tensors,
                           tensors, op_info.attr);
    RYZENAI_LOG_TRACE(
        OpsFusion::dod_format("Copying super instr to bo : offset:{}, size:{}",
                              offset, super_instr.size()));
    memcpy((char *)super_bo_ptr + offset, super_instr.data(),
           super_instr.size());

    super_instr_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  RYZENAI_LOG_TRACE("FusionRuntime : Fill Super Instrns ... DONE");
}

void FusionRuntime::setup_xrt_run(const Metadata &meta) {
  // IMPORTANT: this should only be called after instruction
  //            and data BO's have been allocated
  RYZENAI_LOG_TRACE("FusionRuntime : Setup XRT Run objects ...");

  for (auto &run : runs_) {
    run.set_arg(0, OPCODE);
    // instruction BO and instruction size will be updated on the fly
    // since same PDI could be running different potions of transaction binary
    // TODO: should we have 1 run object per partition? current scheme is to
    //       reduce memory footprint for large graphs
    run.set_arg(3, input_bo_.address() + DDR_AIE_ADDR_OFFSET);
    run.set_arg(4, output_bo_.address() + DDR_AIE_ADDR_OFFSET);
    run.set_arg(5, scratch_bo_.address() + DDR_AIE_ADDR_OFFSET);
    run.set_arg(6, const_bo_.address() + DDR_AIE_ADDR_OFFSET);
    run.set_arg(7, super_instr_bo_.address() + DDR_AIE_ADDR_OFFSET);
  }

  RYZENAI_LOG_TRACE("FusionRuntime : Setup XRT Run objects ... DONE");
}

std::vector<std::vector<uint8_t>>
FusionRuntime::generate_fused_txns(const Metadata &meta) {

  std::vector<std::vector<uint8_t>> fused_txns;
  fused_txns.reserve(meta.partitions.size());

  txns_.clear();
  txns_.reserve(meta.partitions.size());

  size_t partition_index = 0;

  for (const auto &partition : meta.partitions) {
    std::vector<uint8_t> txn_vec = generate_fused_ops(meta, partition.op_range);
    txns_.push_back(std::move(txn_vec));
    utils::txn_util txn = utils::txn_util(txns_.back());
    aiectrl::op_buf instr_buf;
    instr_buf.addOP(aiectrl::transaction_op(txn.txn_ptr_));
    RYZENAI_LOG_TRACE(
        OpsFusion::dod_format("Partition {} Fused Txn Summary :\n{}",
                              partition_index, txn.summarize()));

    fused_txns.push_back(std::move(instr_buf.ibuf_));

    partition_index += 1;
  }

  RYZENAI_LOG_TRACE("FusionRuntime : Generate Fused Transactions ... DONE");

  return fused_txns;
}

bool FusionRuntime::check_context_instr_size(
    const std::vector<std::vector<uint8_t>> &fused_instr_vec,
    const size_t limit) {

  xrt_core::hwctx_handle *handle = static_cast<xrt_core::hwctx_handle *>(ctx_);
  size_t curr_instr_size = xrt_instr_state.at(handle).heap_total_size;
  size_t instr_size = 0;

  for (const auto &instr : fused_instr_vec) {
    instr_size += instr.size();
  }

  RYZENAI_LOG_TRACE(OpsFusion::dod_format(
      "FusionRuntime : check_context_instr_size requesting "
      "instruction BO instr_size: {}, "
      "curr_instr_size: {}, limit: {}",
      instr_size, curr_instr_size, limit));

  return ((curr_instr_size + instr_size) > limit);
}

bool FusionRuntime::check_partition_instr_size(
    const std::vector<std::vector<uint8_t>> &fused_instr_vec,
    const size_t partition_limit) {

  bool repartition = false;
  size_t instruction_idx = 0;

  for (const auto &instr : fused_instr_vec) {
    auto instr_size = instr.size();
    if (instr_size > partition_limit) {
      repartition = true;
      RYZENAI_LOG_TRACE(OpsFusion::dod_format(
          "FusionRuntime : Need to repartition instruction: {}, size: {}",
          instruction_idx, instr_size));
    }

    instruction_idx++;
  }

  return repartition;
}

bool FusionRuntime::allocate_instr_bos(
    const std::vector<std::vector<uint8_t>> &fused_instr_vec) {

  // TODO: move this deallocation to beginning of init??
  // Added here since instr_bos get cleared
  xrt_core::hwctx_handle *handle = static_cast<xrt_core::hwctx_handle *>(ctx_);

  for (auto &instr_bo : instr_bos_) {
    xrt_instr_state.at(handle).heap_total_size -=
        Utils::align_to_next(instr_bo.size(), INSTR_XRT_BO_ALIGNMENT);
  }
  xrt_instr_state.at(handle).num_instr_bos -= instr_bos_.size();

  instr_bos_.clear();

  if (use_instr_sw_cache_) {
    RYZENAI_LOG_TRACE(
        OpsFusion::dod_format("FusionRuntime : skip allocate instr_bo"));
    return false;
  }

  instr_bos_.reserve(fused_instr_vec.size());

  bool alloc_failed = false;

  for (const auto &instr : fused_instr_vec) {
    size_t instr_size = instr.size();
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "FusionRuntime : Reallocating instr_bo, new_size:{}", instr_size));
    try {
      instr_bos_.emplace_back(xrt::bo(ctx_, instr_size,
                                      xrt::bo::flags::cacheable,
                                      kernels_[0].group_id(1)));
    } catch (...) {
      RYZENAI_LOG_TRACE(
          OpsFusion::dod_format("FusionRuntime : Reallocating instr BO failed! "
                                "Fallback to static buffers"));
      alloc_failed = true;
      break;
    }
  }

  if (alloc_failed) {
    // For a particular FusionRuntime object, its instruction BOs
    // wiil either be all in "heap" or "stack"
    // in this case, fall back to using the "stack" instruction BOs
    instr_bos_.clear();
    return true;
  }

  // update state if allocation was successful
  for (const auto &instr : fused_instr_vec) {
    size_t instr_size = instr.size();
    xrt_instr_state.at(handle).heap_total_size +=
        Utils::align_to_next(instr_size, INSTR_XRT_BO_ALIGNMENT);
  }

  xrt_instr_state.at(handle).num_instr_bos += fused_instr_vec.size();

  return false;
}

void FusionRuntime::populate_instr_bos(
    const std::vector<std::vector<uint8_t>> &fused_instr_vec) {

  if (use_instr_sw_cache_) {
    RYZENAI_LOG_TRACE(
        OpsFusion::dod_format("FusionRuntime : skip populate instr_bo"));
    return;
  }

  RYZENAI_LOG_TRACE(OpsFusion::dod_format("FusionRuntime : populate instr_bo"));

  size_t instr_index = 0;
  for (const auto &instr : fused_instr_vec) {
    write_to_bo(instr_bos_.at(instr_index), 0, /*offset*/
                fused_instr_vec.at(instr_index).data(),
                fused_instr_vec.at(instr_index).size());

    instr_index += 1;
  }
}

// TODO : Calling .size() on empty xrt::bo crashes.
void FusionRuntime::reallocate_data_bos(const Metadata &meta) {
  RYZENAI_LOG_TRACE("Reallocating Data BOs ...");
  size_t new_size =
      std::max(MAP_AT(meta.fused_tensors, "super_instr").size, XRT_BO_MIN_SIZE);
  if (super_instr_bo_sz_ < new_size) {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "FusionRuntime : Reallocating input bo, curr_size:{}, new_size:{}",
        super_instr_bo_sz_, new_size));
    super_instr_bo_ = xrt::bo(ctx_, new_size, xrt::bo::flags::host_only,
                              kernels_[0].group_id(HOST_BO_GROUP_ID));
    memset(super_instr_bo_.map(), XRT_BO_INIT_VALUE, super_instr_bo_.size());
    super_instr_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  super_instr_bo_sz_ = new_size;

  new_size =
      std::max(MAP_AT(meta.fused_tensors, "const").size, XRT_BO_MIN_SIZE);
  if (const_bo_sz_ < new_size) {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "FusionRuntime : Reallocating const bo, curr_size:{}, new_size:{}",
        const_bo_sz_, new_size));
    const_bo_ = xrt::bo(ctx_, new_size, xrt::bo::flags::host_only,
                        kernels_[0].group_id(HOST_BO_GROUP_ID));
    memset(const_bo_.map(), XRT_BO_INIT_VALUE, const_bo_.size());
    const_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  const_bo_sz_ = new_size;

  new_size = std::max(MAP_AT(meta.fused_tensors, "in").size, XRT_BO_MIN_SIZE);
  if (input_bo_sz_ < new_size) {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "FusionRuntime : Reallocating input bo, curr_size:{}, new_size:{}",
        input_bo_sz_, new_size));
    input_bo_ = xrt::bo(ctx_, new_size, xrt::bo::flags::host_only,
                        kernels_[0].group_id(HOST_BO_GROUP_ID));
    memset(input_bo_.map(), XRT_BO_INIT_VALUE, input_bo_.size());
    input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  input_bo_sz_ = new_size;

  new_size = std::max(MAP_AT(meta.fused_tensors, "out").size, XRT_BO_MIN_SIZE);
  if (output_bo_sz_ < new_size) {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "FusionRuntime : Reallocating output bo, curr_size:{}, new_size:{}",
        output_bo_sz_, new_size));
    output_bo_ = xrt::bo(ctx_, new_size, xrt::bo::flags::host_only,
                         kernels_[0].group_id(HOST_BO_GROUP_ID));
    memset(output_bo_.map(), XRT_BO_INIT_VALUE, output_bo_.size());
    output_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  output_bo_sz_ = new_size;

  new_size = std::max(MAP_AT(meta.fused_tensors, "scratch").size +
                          meta.max_op_scratch_pad_size,
                      XRT_BO_MIN_SIZE);
  if (scratch_bo_sz_ < new_size) {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "FusionRuntime : Reallocating scratch bo, curr_size:{}, new_size:{}",
        scratch_bo_sz_, new_size));
    scratch_bo_ = xrt::bo(ctx_, new_size, xrt::bo::flags::host_only,
                          kernels_[0].group_id(HOST_BO_GROUP_ID));
    memset(scratch_bo_.map(), XRT_BO_INIT_VALUE, scratch_bo_.size());
    scratch_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  scratch_bo_sz_ = new_size;

  RYZENAI_LOG_TRACE("Reallocating Data BOs ... DONE");
  RYZENAI_LOG_TRACE(OpsFusion::dod_format(
      "\ninput bo size : {}\noutput bo size : {}\nconst bo size : "
      "{}\nscratch bo size : {}\nsuper_instr_bo size : {}",
      input_bo_.size(), output_bo_.size(), const_bo_.size(), scratch_bo_.size(),
      super_instr_bo_.size()));
}

std::vector<std::vector<uint8_t>> FusionRuntime::get_txns() { return txns_; }

void FusionRuntime::initialize_inputs(const Metadata &meta) {
  uint8_t *in_ptr = input_bo_.map<uint8_t *>();
  uint8_t *out_ptr = output_bo_.map<uint8_t *>();
  uint8_t *scratch_ptr = scratch_bo_.map<uint8_t *>();
  std::array<uint8_t *, 3> buf_ptrs = {in_ptr, out_ptr, scratch_ptr};

  for (const auto &op_info : meta.op_list) {
    std::vector<Tensor> tensors;
    for (const auto &buf_name : op_info.args) {
      const auto &tensor_info = MAP_AT(meta.tensor_map, buf_name);
      const auto &packed_tensor_name = tensor_info.parent_name;

      // TODO : Unsafe check. Fails if buf name changed in future
      if (!(packed_tensor_name == "in" || packed_tensor_name == "scratch")) {
        break;
      }

      const auto buf_arg_id =
          MAP_AT(meta.fused_tensors, packed_tensor_name).arg_idx;
      uint8_t *ptr = buf_ptrs[buf_arg_id] + tensor_info.offset;
      Tensor tensor = {ptr, tensor_info.shape, tensor_info.dtype};
      tensors.push_back(std::move(tensor));
    }
    auto op = OpBuilder::create(op_info.name, op_info, meta.tensor_map);
    DD_INVOKE_OPMETHOD(initialize_inputs, op.get(), op_info, tensors,
                       op_info.attr);
  }

  input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  scratch_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void FusionRuntime::merge_inputs(const std::vector<Tensor> &inputs,
                                 const Metadata &meta) {
  RYZENAI_LOG_TRACE("Packing Inputs ... ");
  size_t n_meta_inputs = MetaUtils::get_num_inputs(meta);
  DOD_ASSERT(
      inputs.size() == n_meta_inputs,
      OpsFusion::dod_format(
          "Number of inputs ({}) doesn't match with that of metadata ({})",
          inputs.size(), n_meta_inputs));

  auto t1 = GET_ELAPSED_TIME_NS();
  const auto &in_buf_names = MAP_AT(meta.fused_tensors, "in").packed_tensors;
  for (int i = 0; i < in_buf_names.size(); i++) {
    size_t sz = std::accumulate(inputs[i].shape.begin(), inputs[i].shape.end(),
                                size_t{1}, std::multiplies{}) *
                Utils::get_size_of_type(inputs[i].dtype);

    // TODO : Check if input buffer size matches with metadata
    const auto &tensor_info = MAP_AT(meta.tensor_map, in_buf_names[i]);
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "copying input:{} to input bo at offset:{} and size:{}",
        in_buf_names[i], tensor_info.offset, sz));
    char *inp_ptr = (char *)inputs[i].data;
    input_bo_.write(inputs[i].data, sz, tensor_info.offset);
  }
  auto t2 = GET_ELAPSED_TIME_NS();
  input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto t3 = GET_ELAPSED_TIME_NS();

  input_copy_time_ = t2 - t1;
  input_sync_time_ = t3 - t2;
  RYZENAI_LOG_TRACE("Packing Inputs ... DONE");
}

void FusionRuntime::split_outputs(const std::vector<Tensor> &outputs,
                                  const Metadata &meta) {
  RYZENAI_LOG_TRACE("Unpacking Outputs ...");
  size_t n_meta_outputs = MetaUtils::get_num_outputs(meta);
  DOD_ASSERT(outputs.size() == n_meta_outputs,
             OpsFusion::dod_format(
                 "Number of outputs ({}) doesn't match with number of "
                 "metadata outputs ({})",
                 outputs.size(), n_meta_outputs));

  auto t1 = GET_ELAPSED_TIME_NS();
  output_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  void *output_bo_ptr = output_bo_.map();
  auto t2 = GET_ELAPSED_TIME_NS();

  const auto &out_buf_names = meta.fused_tensors.at("out").packed_tensors;
  auto hwout_tensors = MetaUtils::get_output_tensors(meta);
  for (int i = 0; i < out_buf_names.size(); i++) {
    DOD_ASSERT(outputs[i].shape == hwout_tensors[i].shape &&
                   outputs[i].dtype == hwout_tensors[i].dtype,
               dod_format("output tensor shapes/dtype doesn't match with the "
                          "Runtime output tensor shapes/dtype"));

    const auto &tensor_info = MAP_AT(meta.tensor_map, out_buf_names[i]);
    size_t sz =
        std::accumulate(outputs[i].shape.begin(), outputs[i].shape.end(),
                        size_t{1}, std::multiplies{}) *
        Utils::get_size_of_type(outputs[i].dtype);

    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "Reading output:{} from output bo at offset:{} and size:{}",
        out_buf_names[i], tensor_info.offset, sz));

    // TODO : Do the depad here
    hwout_tensors[i].data = (char *)output_bo_ptr + tensor_info.offset;

    copy_data(hwout_tensors[i], outputs[i]);
    // std::ofstream ofs("mat128_" + std::to_string(i) + ".bin");
    // ofs.write((char *)hwout_tensors[i].data,
    //           std::accumulate(hwout_tensors[i].shape.begin(),
    //                           hwout_tensors[i].shape.end(), size_t{1},
    //                           std::multiplies{}) *
    //               Utils::get_size_of_type(hwout_tensors[i].dtype));
    // ofs.close();
    // output_bo_.read(outputs[i].data, sz, tensor_info.offset);
  }
  auto t3 = GET_ELAPSED_TIME_NS();

  output_copy_time_ = t3 - t2;
  output_sync_time_ = t2 - t1;
  RYZENAI_LOG_TRACE("Unpacking Outputs ... DONE");
}

const Metadata &FusionRuntime::get_meta() const { return meta_; }

std::map<std::string, std::vector<uint8_t>>
FusionRuntime::unpack_internal_buffers(const std::string &dir) {
  std::map<std::string, std::vector<uint8_t>> res;
  auto meta = meta_;

  auto unpack_buffer = [&meta, &res](const std::string &fused_buffer,
                                     xrt::bo &bo) {
    const auto &in_buf_names =
        MAP_AT(meta.fused_tensors, fused_buffer).packed_tensors;
    for (const auto &buf_name : in_buf_names) {
      const auto &tensor_info = MAP_AT(meta.tensor_map, buf_name);
      auto sz = tensor_info.size_in_bytes;
      auto offset = tensor_info.offset;

      RYZENAI_LOG_TRACE(OpsFusion::dod_format(
          "Unpacking input:{} from input bo at offset:{} and size:{}", buf_name,
          offset, sz));

      std::vector<uint8_t> vec(sz);
      bo.read(vec.data(), sz, tensor_info.offset);
      res[buf_name] = std::move(vec);
    }
  };

  input_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  output_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  scratch_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  super_instr_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Not a deadcode. Keeping it as a reminder so it won't be repeated
  // "Don't sync instr bo from device, it will hang the board."

  unpack_buffer("in", input_bo_);
  unpack_buffer("scratch", scratch_bo_);
  unpack_buffer("out", output_bo_);

  if (!dir.empty()) {
    std::cout << "[WARNING] Writing DD intrenal buffers to " << dir
              << std::endl;
    std::cout << "[WARNING] This incurs performance overhead. Set env variable "
                 "DD_WRITE_INTERNAL_BUFS=0 to disable it."
              << std::endl;

    std::filesystem::path dir_path{dir};
    std::filesystem::create_directories(dir_path);
    for (const auto &[name, data] : res) {
      auto newname = replace_characters(name, "/:,", '_');
      auto filename = OpsFusion::dod_format("{}/{}.bin", dir, newname);
      std::ofstream ofs(filename, std::ios::binary);
      DOD_ASSERT(
          ofs, OpsFusion::dod_format("Couldn't open {} for writing", filename))
      ofs.write((char *)data.data(), data.size());
      ofs.close();
    }

    auto hash_filename = dir_path / "buffer_hash.txt"s;
    std::ofstream hash_fs(hash_filename);
    for (const auto &[name, data] : res) {
      hash_fs << name << ", " << compute_hash(data.data(), data.size())
              << std::endl;
    }

    // Hash for full bos.
    hash_fs << "input_bo, " << compute_hash(input_bo_.map(), input_bo_.size())
            << std::endl;
    hash_fs << "output_bo, "
            << compute_hash(output_bo_.map(), output_bo_.size()) << std::endl;
    hash_fs << "scratch_bo, "
            << compute_hash(scratch_bo_.map(), scratch_bo_.size()) << std::endl;
    hash_fs << "const_bo, " << compute_hash(const_bo_.map(), const_bo_.size())
            << std::endl;
    hash_fs << "super_kernel_bo, "
            << compute_hash(super_instr_bo_.map(), super_instr_bo_.size())
            << std::endl;
    hash_fs << "instruction_bo, "
            << compute_hash(instr_bos_[0].map(), instr_bos_[0].size())
            << std::endl;
  }

  return res;
}

std::once_flag FusionRuntime::logger_flag_;

} // namespace OpsFusion
