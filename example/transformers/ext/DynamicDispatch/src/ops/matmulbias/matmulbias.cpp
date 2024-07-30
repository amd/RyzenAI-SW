#include <any>
#include <sstream>

#include <ops/matmulbias/matmulbias.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

#include <utils/instruction_registry.hpp>
#include <xrt_context/xrt_context.hpp>

#include "../ops_common/matmul_matrix.hpp"
#include "op_types.h"

// AIE Driver header
#include "xaiengine.h"

#include "utils/dpu_mdata.hpp"
#include <utils/txn_container.hpp>

using namespace matmul_matrix;
namespace ryzenai {

static std::tuple<int, int, int>
extract_MKN(const std::vector<Tensor> &inputs) {
  // inputs[0] --> input
  // inputs[1] --> wts
  int M = inputs.at(0).shape.size() == 3 ? inputs.at(0).shape.at(1)
                                         : inputs.at(0).shape.at(0);
  int K = inputs.at(1).shape.at(0);
  int N = inputs.at(1).shape.at(1);

  return std::make_tuple(M, K, N);
}

template <typename InT, typename WtT, typename OutT>
std::once_flag matmulbias<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t matmulbias<InT, WtT, OutT>::matmulbias_count = 0;

// template <typename InT, typename WtT, typename OutT>
// instruction_registry matmulbias<InT, WtT, OutT>::instr_reg_;

template <typename InT, typename WtT, typename OutT>
std::once_flag matmulbias<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void matmulbias<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::string matmulbias<InT, WtT, OutT>::get_instr_key(std::string prefix, int m,
                                                      int k, int n) {
  auto instr_key = prefix + "_" + std::to_string(m) + "_" + std::to_string(k) +
                   "_" + std::to_string(n);
  //   std::cout << "MatmulBias Txn Key : " << instr_key << std::endl;
  return instr_key;
}

template <typename InT, typename WtT, typename OutT>
void matmulbias<InT, WtT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;

  // GEMM _a8w8
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key =
        "gemm_bias_" + get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N);
    auto param_key =
        "gemm_bias_" + get_instr_key(param_fname_prefix_, mat.M, mat.K, mat.N);
    instructions.push_back(std::make_pair(key, false));
    layer_params.push_back(std::make_pair(param_key, false));
  }

  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
  instr_reg_.add_layer_params(layer_params);
}

template <typename InT, typename WtT, typename OutT>
matmulbias<InT, WtT, OutT>::matmulbias(const std::string &a_dtype,
                                       const std::string &b_dtype,
                                       const std::string &c_dtype,
                                       bool load_xrt) {

  txnbin_a_header = {{"int8", "a8"}};

  txnbin_b_header = {{"int8", "w8"}};

  txnbin_acc_header = {{"int32", "acc32"}, {"int8", "acc8"}};

  xclbin_a_header = {{"int8", "a8"}};

  xclbin_b_header = {{"int8", "w8"}};

  xclbin_acc_header = {{"int32", "acc32"}, {"int8", "acc8"}};

  default_shapes_["a8w8acc8"] = std::vector<matrix_shapes>();
  default_shapes_["a8w8acc8"].emplace_back(512, 1152, 1152);
  default_shapes_["a8w8acc8"].emplace_back(512, 768, 1152);
  default_shapes_["a8w8acc8"].emplace_back(512, 512, 1152);
  default_shapes_["a8w8acc8"].emplace_back(512, 768, 768);
  default_shapes_["a8w8acc8"].emplace_back(512, 3072, 768);
  default_shapes_["a8w8acc8"].emplace_back(512, 768, 3072);
  default_shapes_["a8w8acc8"].emplace_back(512, 768, 128);

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  matmulbias_id_ = matmulbias_count++;

  /*select xclbin based on the input/output types*/
  const std::string XCLBIN_FNAME =
      OpInterface::get_dod_base_dir() + ryzenai::PSF_A8W8_QDQ_XCLBIN_PATH;
  // std::cout << "xclbin fname : " << XCLBIN_FNAME << std::endl;
  txn_fname_prefix_ = txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  param_fname_prefix_ = txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_) + "_param";
  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  KERNEL_M_MAX = 512;
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
        "matmulbias_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[OP] ID: " + std::to_string(matmulbias_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void matmulbias<InT, WtT, OutT>::set_kernel_shapes() {
  // Use largest M dimension as the default
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;
  if (a_dtype_ == "int16" || a_dtype_ == "int8") {
    if ((w_shape_[0] == 1152 && w_shape_[1] == 1152) ||
        (w_shape_[0] == 768 && w_shape_[1] == 1152) ||
        (w_shape_[0] == 512 && w_shape_[1] == 1152) ||
        (w_shape_[0] == 768 && w_shape_[1] == 768) ||
        (w_shape_[0] == 3072 && w_shape_[1] == 768) ||
        (w_shape_[0] == 768 && w_shape_[1] == 3072) ||
        (w_shape_[0] == 768 && w_shape_[1] == 128)) {
      // Update kernel shape to match weight matrix if a
      // supported kernel exists
      kernel_x_shape_[1] = w_shape_[0];
      kernel_y_shape_[0] = w_shape_[0];
      kernel_y_shape_[1] = w_shape_[1];
      kernel_z_shape_[1] = w_shape_[1];
    }
  } else {
    /*Current support is only for bfp16 activation types*/
    throw std::runtime_error(
        "No Kernel exists for the current activation data type");
  }
}

template <typename InT, typename WtT, typename OutT>
void matmulbias<InT, WtT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Matmulbias initialize_const_params(ptr) ...");

  DOD_THROW_IF(
      (const_params.size() != 2) || (const_params.at(0).shape.size() != 2) ||
          (const_params.at(1).shape.size() != 2),
      OpsFusion::dod_format("Unsupported const spec for Matmulbias\n") +
          OpsFusion::dod_format(
              "(Details : #const params == 1 ({}), Const param1 dim == 2 ({}) "
              ", Const param2 dim == 3 ({})",
              const_params.size(), const_params.at(0).shape.size(),
              const_params.at(1).shape.size()));

  const int w_idx = 0, bias_idx = 1;
  // The first data is Weight
  auto weights = (int8_t *)const_params.at(w_idx).data;

  std::vector<size_t> shape = const_params.at(w_idx).shape;

  // Init the BO size
  w_shape_[0] = shape[0]; // K
  w_shape_[1] = shape[1]; // N
  set_kernel_shapes();

  auto bias = (int8_t *)const_params.at(bias_idx).data;

  std::vector<WtT> buf(w_shape_[0] * w_shape_[1]);
  matmul_matrix::WgtMatrix<WtT, Ksubv, Nsubv> W(w_shape_[0], w_shape_[1],
                                                buf.data());
  for (int r = 0; r < w_shape_[0]; ++r) {
    for (int c = 0; c < w_shape_[1]; ++c) {
      W.at(r, c) = weights[(r * w_shape_[1]) + c];
    }
  }
  auto total_size = Ksubv * Nsubv;
  //// WGT + Bias(all zeros)
  { // This section of the code interleaves bias with weights Nsubv of bias
    // with every K x N
    int write_offset = 0;
    for (int N_shard = 0; N_shard < (w_shape_[1]) / (Nsubv); N_shard++) {
      for (int K_shard = 0; K_shard < (w_shape_[0]) / (Ksubv); K_shard++) {
        memcpy((void *)(reinterpret_cast<int8_t *>(dest) + write_offset),
               (void *)&buf[(N_shard * w_shape_[0] * Nsubv) +
                            (K_shard * total_size)],
               (total_size));
        write_offset += total_size;
        memcpy((void *)(reinterpret_cast<int8_t *>(dest) + write_offset),
               (void *)(reinterpret_cast<int8_t *>(bias) + (N_shard * Nsubv)),
               Nsubv);
        write_offset += Nsubv;
      }
    }
  }

  RYZENAI_LOG_TRACE("Matmulbias initialize_const_params(ptr) ... DONE");
}

// For MATMUL bias: weight + bias
template <typename InT, typename WtT, typename OutT>
void matmulbias<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  // Check the number of inputs
  if (const_params.size() != 2) {
    throw std::runtime_error("MATMULbias expect to have two constants.");
  }
  const int w_idx = 0;
  // The first data is Weight
  // auto weights = (int8_t*)const_params.at(w_idx).data;

  std::vector<size_t> shape = const_params.at(w_idx).shape;
  int size_weight = shape[0] * shape[1] * b_dtype_size_;

  // Init the BO size
  w_shape_[0] = shape[0];
  w_shape_[1] = shape[1];
  set_kernel_shapes();

  int size_interleaved_bias =
      w_shape_[0] * w_shape_[1] / matmul_matrix::Ksubv * b_dtype_size_;

  // Create input/output BOs
  const int A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  const int B_BO_SIZE = size_interleaved_bias + size_weight;
  // (kernel_y_shape_[0] * kernel_y_shape_[1] * b_dtype_size_);
  const int C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);
  RYZENAI_LOG_TRACE("MATMULBIAS: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE));
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  // copy b_bo
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  auto b_format_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  initialize_const_params(b_bo_map, const_params);
  auto b_format_stop = GET_ELAPSED_TIME_NS();
  auto b_copy_stop = GET_ELAPSED_TIME_NS();
  b_format_time_ += b_format_stop - b_format_start;
  b_copy_time_ = b_copy_stop - b_copy_start;

  // sync b_bo
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();
  b_sync_time_ = b_sync_stop - b_sync_start;
}

template <typename InT, typename WtT, typename OutT>
void matmulbias<InT, WtT, OutT>::execute(const std::vector<Tensor> &input,
                                         std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 1) {
    throw std::runtime_error("MATMULbias expect to have one input.");
  }
  const int a_idx = 0;
  // The first data is a
  InT *a = (InT *)input.at(a_idx).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;

  int64_t exec_start = GET_ELAPSED_TIME_NS();

  a_shape_[0] = input.at(a_idx).shape.at(0);
  a_shape_[1] = input.at(a_idx).shape.at(1);

  c_shape_[0] = a_shape_[0];
  c_shape_[1] = w_shape_[1];
  if (a_shape_[0] == 512) {
    kernel_x_rows = 512;
  }

  // a_bo copy
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  int a_size = a_shape_[0] * a_shape_[1] * sizeof(InT);
  memcpy((void *)a_bo_map, (void *)a, a_size);
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  // prepare inst_bo and param_bo
  auto instr_bo_key = "gemm_bias" + txn_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]);
  auto param_bo_key = "gemm_bias" + param_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]);

  const xrt::bo &instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;
  const xrt::bo &param_bo = instr_reg_.get_param_bo(param_bo_key).second;
  int instr_bo_words = instr_bo.size() / sizeof(int);
  auto kernel_ = xrt_ctx_->get_kernel();

  // launch the kernel
  xrt::run run;
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  run = kernel_(2, instr_bo, instr_bo_words,
                c_bo_.address() + DDR_AIE_ADDR_OFFSET,
                a_bo_.address() + DDR_AIE_ADDR_OFFSET,
                b_bo_.address() + DDR_AIE_ADDR_OFFSET,
                param_bo.address() + DDR_AIE_ADDR_OFFSET, 0);
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
  memcpy((void *)aie_out, (void *)c_bo_map,
         c_shape_[0] * c_shape_[1] * sizeof(OutT));
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
  int64_t exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(matmulbias_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) + " " +
      std::to_string(kernel_x_rows) + " " + std::to_string(kernel_x_shape_[1]) +
      " " + std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> matmulbias<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [M, K, N] = extract_MKN(input);
  std::string txn_key =
      "gemm_bias_" + get_instr_key(txn_fname_prefix_, M, K, N);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Instruction fname : {}", txn_key));
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());
  return data;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> matmulbias<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [M, K, N] = extract_MKN(input);

  // TODO: Add check to validate tensor shapes
  std::string param_key =
      "gemm_bias_" + get_instr_key(param_fname_prefix_, M, K, N);
  Transaction &txn = Transaction::getInstance();
  std::string param_string = txn.get_txn_str(param_key);
  std::istringstream params_stream(param_string, std::ios::binary);
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Super kernel params name : {}", param_key));
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(params_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> matmulbias<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  // input --> [input, weights, output]
  // Check if IO buffers have batch.
  w_shape_[0] = input.at(1).shape.at(0);
  w_shape_[1] = input.at(1).shape.at(1);
  set_kernel_shapes();

  int size_interleaved_bias =
      w_shape_[0] * w_shape_[1] / matmul_matrix::Ksubv * b_dtype_size_;

  size_t const_params_bo_size =
      (kernel_y_shape_[0] * kernel_y_shape_[1] * b_dtype_size_) +
      size_interleaved_bias;
  size_t input_bo_size =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  size_t output_bo_size =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);
  size_t super_kernel_size = get_super_kernel_params(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 3, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size}};
  // TO_DO: ostringstream compiled fail
  // RYZENAI_LOG_TRACE(OpsFusion::dod_format("Matmulbias Argmap : {}",
  // cvt_to_string(arg_map)));
  return arg_map;
};

template class matmulbias<int8_t, int8_t, int8_t>;
} // namespace ryzenai
