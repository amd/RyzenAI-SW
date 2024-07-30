#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <thread>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/matmul/matmul.hpp>

#include "test_common.hpp"

using namespace matmul_matrix;

struct ConstConfig {
  uint32_t C1;
  uint32_t C2;
  uint8_t SQb;
  uint8_t Sout;
  uint8_t Stdm;
  int64_t *C0_vec;
  int64_t c0;
};

template <typename InT, typename WgT, typename OuT>
static void run_matmul(size_t M, size_t K, size_t N, bool debug,
                       const std::string &a_dtype, const std::string &b_dtype,
                       const std::string &c_dtype, int thread_idx,
                       size_t n_iters, WgT *b_data, const ConstConfig &config,
                       OpsFusion::FusionRuntime &rt,
                       std::vector<int> &err_counts) {

  // TODO: move preparation of input/golden to config too
  ostringstream os;
  os << "[" << thread_idx << "] Running for n_iters: " << n_iters << "\n";
  std::cout << os.str();

  int Msubv_act = 0;
  if (a_dtype == "int16" || a_dtype == "uint16") {
    Msubv_act = 32;
  } else if (a_dtype == "int8" || a_dtype == "uint8") {
    Msubv_act = 64;
  } else {
    throw std::invalid_argument("a_dtype is not supported");
  }
  int N_w = N;
  if (N_w < Nsubv * 2) {
    N_w = Nsubv * 2; // This is the miminum N
  }
  std::vector<size_t> a_shape = {1, M, K};
  std::vector<size_t> aie_out_shape = {1, M, N};

  std::vector<InT> a(M * K);
  std::vector<int32_t> cpu_out(M * N_w);
  std::vector<OuT> cpu_out_qdq(M * N_w);
  std::vector<OuT> aie_out(M * N, garbage_value);

  RowMajorMatrix<InT> X(M, K, a.data());
  RowMajorMatrix<int32_t> cpu_Y(M, N_w, cpu_out.data());
  RowMajorMatrix<OuT> cpu_Y_qdq(M, N_w, cpu_out_qdq.data());
  RowMajorMatrix<OuT> aie_Y(M, N, aie_out.data());

  srand(0xABCD);
  init_random(X, 0, 32);

  const uint32_t C1 = config.C1;
  const uint32_t C2 = config.C2;
  const uint8_t SQb = config.SQb;
  const uint8_t Sout = config.Sout;
  const uint8_t Stdm = config.Stdm;
  int64_t *const C0_vec = config.C0_vec;
  const int64_t c0 = config.c0;
  if (a_dtype == "uint16") {
    srand(0xABCD);
    init_random(X, 0, 128);
  }

  RowMajorMatrix<WgT> W(K, N_w, b_data);
  if (a_dtype == "uint16") {
    cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
               RowMajorMatrix<int32_t>>(X, W, cpu_Y, Stdm, Msubv_act, Ksubv,
                                        Nsubv);
    qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
               RowMajorMatrix<OuT>>(X, cpu_Y, C2, C1, C0_vec, SQb, Sout,
                                    cpu_Y_qdq, "uint16");
  } else {
    cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
               RowMajorMatrix<int32_t>>(X, W, cpu_Y, "int32");
    qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
               RowMajorMatrix<OuT>>(X, cpu_Y, C2, C1, C0_vec, SQb, Sout,
                                    cpu_Y_qdq, "uint8");
  }

  // TODO: also use different input for each iteration ??
  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

  os = std::ostringstream();
  os << "[" << thread_idx << "] Calling execute\n";
  std::cout << os.str();

  for (int i = 0; i < n_iters; i++) {
    rt.execute(input_Tensor, output_Tensor);
  }

  os = std::ostringstream();
  os << "[" << thread_idx << "] Done!\n";
  std::cout << os.str();

  int err_count = check_result(cpu_Y_qdq, aie_Y);

  err_counts.at(thread_idx) = err_count;
}

template <typename InT, typename WgT, typename OuT>
static void
config_matmul(const std::string &meta_json, size_t M, size_t K, size_t N,
              bool debug, const std::string &a_dtype,
              const std::string &b_dtype, const std::string &c_dtype,
              int thread_idx, size_t n_threads_0, size_t n_threads_1,
              size_t n_iters, WgT *b_data, const ConstConfig &config,
              const std::string &xclbin_fname, const OpsFusion::Metadata &meta,
              std::vector<int> &err_counts) {
  OpsFusion::FusionRuntime rt(xclbin_fname);

  rt.init(meta);

  ostringstream os;
  os << "[" << thread_idx << "] Configured meta: " << meta_json
     << ", xclbin: " << xclbin_fname << "\n";
  std::cout << os.str();

  std::vector<std::thread> workers;

  for (int i = 0; i < n_threads_1; i++) {
    workers.emplace_back(run_matmul<InT, WgT, OuT>, M, K, N, debug, a_dtype,
                         b_dtype, c_dtype, thread_idx * n_threads_1 + i,
                         n_iters, b_data, std::cref(config), std::ref(rt),
                         std::ref(err_counts));
  }

  for (auto &worker : workers) {
    worker.join();
  }
}

template <typename InT, typename WgT, typename OuT>
static void
config_and_run_matmul(const std::string &meta_json, size_t M, size_t K,
                      size_t N, bool debug, const std::string &a_dtype,
                      const std::string &b_dtype, const std::string &c_dtype,
                      const std::string &matmul_dir,
                      std::vector<int> &err_counts, int thread_idx,
                      size_t n_threads_0, size_t n_threads_1, size_t n_iters) {

  std::vector<int> model_err_count(n_threads_0 * n_threads_1, 0);

  std::vector<WgT> b(K * N);
  std::vector<int64_t> qdq(1 * N); // c0
  std::vector<int32_t> qdq_params(QDQparam_size);

  srand(0xABCD);
  initialize_random<WgT>(b, K * N, 32, 0);
  initialize_random<int64_t>(qdq, 1 * N, 32, 0);

  uint32_t C1 = 0;
  uint32_t C2 = 10;
  uint8_t SQb = 0;
  uint8_t Sout = 13;
  uint8_t Stdm = 0;
  int64_t *C0_vec = (int64_t *)qdq.data();
  int64_t c0 = 0;
  if (a_dtype == "uint16") {
    srand(0xABCD);
    initialize_random<WgT>(b, K * N, 128, 0);
    initialize_random<int64_t>(qdq, 1 * N, 10, 0);
    c0 = 0;
    C1 = -11;
    SQb = 0;
    Sout = 9;
    Stdm = round(log2(K)) - 8;
    C2 = 2 << Stdm;
  }
  *(int64_t *)(&qdq_params[qdq_c0_idx]) = c0; // qdq_params[0] = c0;
  qdq_params[qdq_c1_idx] = C1;
  qdq_params[qdq_c2_idx] = C2;
  qdq_params[qdq_c3_idx] = 0;
  // qdq_params[qdq_Mv_idx] = Msubv;
  // qdq_params[qdq_Nv_idx] = Nsubv;
  qdq_params[qdq_SQb_idx] = SQb;
  qdq_params[qdq_Sout_idx] = Sout;
  qdq_params[qdq_Stdm_idx] = Stdm;

  struct ConstConfig config;

  config.C1 = C1;
  config.C2 = C2;
  config.SQb = SQb;
  config.Sout = Sout;
  config.Stdm = Stdm;
  config.C0_vec = C0_vec;
  config.c0 = c0;

  {
    // NOTE: constants are generated by CPP test, not model.py
    std::ofstream wts_f(matmul_dir + "/0.const",
                        std::ios::out | std::ios::binary);
    std::ofstream qdq_f(matmul_dir + "/1.const",
                        std::ios::out | std::ios::binary);
    std::ofstream qdq_p_f(matmul_dir + "/2.const",
                          std::ios::out | std::ios::binary);
    confirmOpen(wts_f);
    confirmOpen(qdq_f);
    confirmOpen(qdq_p_f);
    wts_f.write((char *)b.data(), b.size() * sizeof(WgT));
    qdq_f.write((char *)qdq.data(), qdq.size() * sizeof(int64_t));
    qdq_p_f.write((char *)qdq_params.data(),
                  qdq_params.size() * sizeof(int32_t));
  }

  auto meta = OpsFusion::load_meta_json(meta_json);

  std::string xclbin_fname;
  if (a_dtype == "uint16") {
    xclbin_fname = PSJ_A16W8_QDQ_XCLBIN_REL_PATH;
  } else {
    xclbin_fname = PSF_A8W8_QDQ_XCLBIN_REL_PATH;
  }

  std::vector<std::thread> workers;

  for (int i = 0; i < n_threads_0; i++) {
    workers.emplace_back(config_matmul<InT, WgT, OuT>, std::cref(meta_json), M,
                         K, N, debug, a_dtype, b_dtype, c_dtype, i, n_threads_0,
                         n_threads_1, n_iters, b.data(), std::cref(config),
                         std::cref(xclbin_fname), std::cref(meta),
                         std::ref(model_err_count));
  }

  for (auto &worker : workers) {
    worker.join();
  }

  err_counts.at(thread_idx) =
      std::accumulate(model_err_count.begin(), model_err_count.end(), 0);

  ostringstream os_1;
  os_1 << "All iterations done for " << thread_idx << "\n";
  std::cout << os_1.str();
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    std::cout << "Usage : ops_fusion.exe n_threads_0 n_threads_1 n_iters "
                 "<meta0.json> <meta1.json> "
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;
  try {

    // how many threads to launch to create FusionRT object
    size_t n_threads_0 = std::max(1LL, std::atoll(argv[1]));
    // how many threads to launch to call execute for each FusionRT object
    size_t n_threads_1 = std::max(1LL, std::atoll(argv[2]));
    // how many times to call execute within a thread
    size_t n_iters = std::max(1LL, std::atoll(argv[3]));

    // each json will load a different model, do init in their own thread
    std::vector<std::string> meta_jsons;
    for (int i = 4; i < argc; i++) {
      meta_jsons.push_back(std::string(argv[i]));
    }

    std::vector<int> err_counts(meta_jsons.size(), 0);

    std::vector<std::thread> workers;
    int thread_idx = 0;

    for (const auto &meta_json : meta_jsons) {
      // NOTE: entire test config is derived from name currently
      //       could also vary shapes potentially
      if (meta_json.find("a16w8") != string::npos) {
        workers.emplace_back(
            config_and_run_matmul<uint16_t, uint8_t, uint16_t>, meta_json, 128,
            768, 1152, false, "uint16", "uint8", "uint16",
            "test_multi_thread_matmul_a16w8", std::ref(err_counts), thread_idx,
            n_threads_0, n_threads_1, n_iters);
      } else {
        workers.emplace_back(config_and_run_matmul<uint8_t, uint8_t, uint8_t>,
                             meta_json, 512, 768, 1152, false, "uint8", "uint8",
                             "uint8", "test_multi_thread_matmul_a8w8",
                             std::ref(err_counts), thread_idx, n_threads_0,
                             n_threads_1, n_iters);
      }

      thread_idx++;
    }

    for (auto &worker : workers) {
      worker.join();
    }

    std::cout << "All models run\n";

    int tot_err_count =
        std::accumulate(err_counts.begin(), err_counts.end(), 0);

    if (tot_err_count > 0) {
      std::cout << "Matmul test failed with tot_err_count = " << tot_err_count
                << std::endl;
      return EXIT_FAILURE;
    }
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
