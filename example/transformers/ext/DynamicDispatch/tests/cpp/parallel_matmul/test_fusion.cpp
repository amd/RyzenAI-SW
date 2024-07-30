#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/matmul/matmul.hpp>

#include "test_common.hpp"

using namespace matmul_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
static int test_matmul(const std::string &meta_json, size_t M, size_t K,
                       size_t N, bool debug = false,
                       const std::string &a_dtype = "int16",
                       const std::string &b_dtype = "int8",
                       const std::string &c_dtype = "int32") {
  int err_count = 0;
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

  // Second matrix multiplication will inherit types from first
  auto d_dtype = a_dtype;
  auto f_dtype = c_dtype;

  // Shapes for one matrix multiplication
  std::vector<size_t> a_shape = {1, M, K};
  std::vector<size_t> b_shape = {K, N};
  std::vector<size_t> abc_qdq_shape = {N};                    // UNUSED
  std::vector<size_t> abc_qdq_params_shape = {QDQparam_size}; // UNUSED
  std::vector<size_t> c_shape = {1, M, N};

  // Shapes for second matrix multiplication (Same as first one for now)
  std::vector<size_t> d_shape = {1, M, K};
  std::vector<size_t> e_shape = {K, N};
  std::vector<size_t> def_qdq_shape = {N};                    // UNUSED
  std::vector<size_t> def_qdq_params_shape = {QDQparam_size}; // UNUSED
  std::vector<size_t> f_shape = {1, M, N};

  // Buffers for one matrix multiplication
  std::vector<InT> a(M * K);
  std::vector<WgT> b(K * N);
  std::vector<OuT> c(M * N, garbage_value);
  std::vector<int64_t> abc_qdq(1 * N); // c0
  std::vector<int32_t> abc_qdq_params(QDQparam_size);
  std::vector<int32_t> abc_cpu_out(M * N_w);
  std::vector<OuT> abc_cpu_out_qdq(M * N_w);
  RowMajorMatrix<InT> abc_X(M, K, a.data());
  RowMajorMatrix<int32_t> abc_cpu_Y(M, N_w, abc_cpu_out.data());
  RowMajorMatrix<OuT> abc_cpu_Y_qdq(M, N_w, abc_cpu_out_qdq.data());
  RowMajorMatrix<OuT> abc_aie_Y(M, N, c.data());

  // Buffers for second matrix multiplication
  std::vector<InT> d(M * K);
  std::vector<WgT> e(K * N);
  std::vector<OuT> f(M * N, garbage_value);
  std::vector<int64_t> def_qdq(1 * N); // c0
  std::vector<int32_t> def_qdq_params(QDQparam_size);
  std::vector<int32_t> def_cpu_out(M * N_w);
  std::vector<OuT> def_cpu_out_qdq(M * N_w);
  RowMajorMatrix<InT> def_X(M, K, d.data());
  RowMajorMatrix<int32_t> def_cpu_Y(M, N_w, def_cpu_out.data());
  RowMajorMatrix<OuT> def_cpu_Y_qdq(M, N_w, def_cpu_out_qdq.data());
  RowMajorMatrix<OuT> def_aie_Y(M, N, f.data());

  srand(0xABCD);

  // Randomize weights and qdq params for one matrix multiplication
  init_random(abc_X, 0, 32);
  initialize_random<WgT>(b, K * N, 32, 0);
  initialize_random<int64_t>(abc_qdq, 1 * N, 32, 0);

  // Randomize weights and qdq params for second matrix multiplication
  init_random(def_X, 0, 32);
  initialize_random<WgT>(e, K * N, 32, 0);
  initialize_random<int64_t>(def_qdq, 1 * N, 32, 0);

  // Magic numbers
  uint32_t C1 = 0;
  uint32_t C2 = 10;
  uint8_t SQb = 0;
  uint8_t Sout = 13;
  uint8_t Stdm = 0;
  int64_t *C0_vec = (int64_t *)abc_qdq.data();
  int64_t *C0_vec_def = (int64_t *)def_qdq.data(); // Different for two matmul
  int64_t c0 = 0;
  if (a_dtype == "uint16") {
    init_random(abc_X, 0, 128);
    init_random(def_X, 0, 128);
    initialize_random<WgT>(b, K * N, 128, 0);
    initialize_random<WgT>(e, K * N, 128, 0);
    initialize_random<int64_t>(abc_qdq, 1 * N, 10, 0);
    initialize_random<int64_t>(def_qdq, 1 * N, 10, 0);
    c0 = 0;
    C1 = -11;
    SQb = 0;
    Sout = 9;
    Stdm = round(log2(K)) - 8;
    C2 = 2 << Stdm;
  }

  // qdq params for one matmul
  *(int64_t *)(&abc_qdq_params[qdq_c0_idx]) = c0; // qdq_params[0] = c0;
  abc_qdq_params[qdq_c1_idx] = C1;
  abc_qdq_params[qdq_c2_idx] = C2;
  abc_qdq_params[qdq_c3_idx] = 0;
  abc_qdq_params[qdq_SQb_idx] = SQb;
  abc_qdq_params[qdq_Sout_idx] = Sout;
  abc_qdq_params[qdq_Stdm_idx] = Stdm;

  // qdq params for second matmul, use same
  *(int64_t *)(&def_qdq_params[qdq_c0_idx]) = c0; // qdq_params[0] = c0;
  def_qdq_params[qdq_c1_idx] = C1;
  def_qdq_params[qdq_c2_idx] = C2;
  def_qdq_params[qdq_c3_idx] = 0;
  def_qdq_params[qdq_SQb_idx] = SQb;
  def_qdq_params[qdq_Sout_idx] = Sout;
  def_qdq_params[qdq_Stdm_idx] = Stdm;

  {
    std::string matmul_dir = "test_parallel_matmul";
    if (a_dtype == "uint16")
      matmul_dir += "_a16w8";
    else
      matmul_dir += "_a8w8";
    std::ofstream abc_wts_f(matmul_dir + "/0.const",
                            std::ios::out | std::ios::binary);
    std::ofstream abc_qdq_f(matmul_dir + "/1.const",
                            std::ios::out | std::ios::binary);
    std::ofstream abc_qdq_p_f(matmul_dir + "/2.const",
                              std::ios::out | std::ios::binary);
    confirmOpen(abc_wts_f);
    confirmOpen(abc_qdq_f);
    confirmOpen(abc_qdq_p_f);
    abc_wts_f.write((char *)b.data(), b.size() * sizeof(WgT));
    abc_qdq_f.write((char *)abc_qdq.data(), abc_qdq.size() * sizeof(int64_t));
    abc_qdq_p_f.write((char *)abc_qdq_params.data(),
                      abc_qdq_params.size() * sizeof(int32_t));
  }

  {
    std::string matmul_dir = "test_parallel_matmul";
    if (a_dtype == "uint16")
      matmul_dir += "_a16w8";
    else
      matmul_dir += "_a8w8";
    std::ofstream def_wts_f(matmul_dir + "/3.const",
                            std::ios::out | std::ios::binary);
    std::ofstream def_qdq_f(matmul_dir + "/4.const",
                            std::ios::out | std::ios::binary);
    std::ofstream def_qdq_p_f(matmul_dir + "/5.const",
                              std::ios::out | std::ios::binary);
    confirmOpen(def_wts_f);
    confirmOpen(def_qdq_f);
    confirmOpen(def_qdq_p_f);
    def_wts_f.write((char *)e.data(), e.size() * sizeof(WgT));
    def_qdq_f.write((char *)def_qdq.data(), def_qdq.size() * sizeof(int64_t));
    def_qdq_p_f.write((char *)def_qdq_params.data(),
                      def_qdq_params.size() * sizeof(int32_t));
  }

  {
    RowMajorMatrix<WgT> W(K, N_w, b.data());
    if (a_dtype == "uint16") {
      cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
                 RowMajorMatrix<int32_t>>(abc_X, W, abc_cpu_Y, Stdm, Msubv_act,
                                          Ksubv, Nsubv);
      qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
                 RowMajorMatrix<OuT>>(abc_X, abc_cpu_Y, C2, C1, C0_vec, SQb,
                                      Sout, abc_cpu_Y_qdq, "uint16");
    } else {
      cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
                 RowMajorMatrix<int32_t>>(abc_X, W, abc_cpu_Y, "int32");
      qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
                 RowMajorMatrix<OuT>>(abc_X, abc_cpu_Y, C2, C1, C0_vec, SQb,
                                      Sout, abc_cpu_Y_qdq, "uint8");
    }
  }

  {
    RowMajorMatrix<WgT> W(K, N_w, e.data());
    if (d_dtype == "uint16") {
      cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
                 RowMajorMatrix<int32_t>>(def_X, W, def_cpu_Y, Stdm, Msubv_act,
                                          Ksubv, Nsubv);
      qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
                 RowMajorMatrix<OuT>>(def_X, def_cpu_Y, C2, C1, C0_vec_def, SQb,
                                      Sout, def_cpu_Y_qdq, "uint16");
    } else {
      cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
                 RowMajorMatrix<int32_t>>(def_X, W, def_cpu_Y, "int32");
      qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
                 RowMajorMatrix<OuT>>(def_X, def_cpu_Y, C2, C1, C0_vec_def, SQb,
                                      Sout, def_cpu_Y_qdq, "uint8");
    }
  }

  std::string xclbin_fname;
  if (a_dtype == "uint16") {
    xclbin_fname = PSJ_A16W8_QDQ_XCLBIN_REL_PATH;
  } else {
    xclbin_fname = PSF_A8W8_QDQ_XCLBIN_REL_PATH;
  }

  auto meta = OpsFusion::load_meta_json(meta_json);

  auto dod_input_names = meta.fused_tensors.at("in").packed_tensors;

  for (int i = 0; i < dod_input_names.size(); ++i) {
    std::cout << "input " << i << " : " << dod_input_names[i] << std::endl;
  }

  OpsFusion::FusionRuntime rt(xclbin_fname);
  rt.init(meta);

// #define ONE_MATMUL
#ifdef ONE_MATMUL

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{c.data(), c_shape, c_dtype}};

#else

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{a.data(), a_shape, a_dtype}, {d.data(), d_shape, d_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{c.data(), c_shape, c_dtype}, {f.data(), f_shape, f_dtype}};

#endif

  rt.execute(input_Tensors, output_Tensors);

  err_count = check_result(abc_cpu_Y_qdq, abc_aie_Y);
  std::cout << "err_count first matmul: " << err_count << std::endl;

#ifndef ONE_MATMUL
  err_count += check_result(def_cpu_Y_qdq, def_aie_Y);
  std::cout << "err_count second matmul: " << err_count << std::endl;
#endif

  return err_count;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_parallel_matmul.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;
  size_t M = 128;
  size_t K = 768;
  size_t N = 1152;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    if (meta_json.find("a16w8") != string::npos) {
      err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
          meta_json, 128, 768, 1152, false, "uint16", "uint8", "uint16");
    } else {
      err_count = test_matmul<uint8_t, uint8_t, uint8_t>(
          meta_json, 512, 768, 1152, false, "uint8", "uint8", "uint8");
    }
    if (err_count > 0) {
      std::cout << "Matmul test failed with err_count = " << err_count
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
