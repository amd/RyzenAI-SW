#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <ops/mladfrmsnorm/mladfrmsnorm.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "mladfsoftmax_helpers.hpp"
#include "test_common.hpp"

// z = RMSNorm(x)
template <typename InT = uint16_t, typename WtsT = uint16_t,
          typename OuT = uint16_t>
static int test_mladfrmsnorm(const std::string &meta_json, size_t M, size_t K,
                             bool debug = false,
                             const std::string &a_dtype = "bfloat16",
                             const std::string &wts_dtype = "bfloat16",
                             const std::string &c_dtype = "bfloat16") {
  std::vector<size_t> a_shape = {M, K};
  std::vector<size_t> wts_shape = {K};

  std::vector<InT> a(M * K);
  std::vector<InT> wts(K);

  // compute aie
  std::vector<OuT> aie_out(M * K, garbage_value);
  std::vector<OuT> reference_out(M * K);

  // Using golden teste vectors from MLLIB
  // https://gitenterprise.xilinx.com/AIELibs/mllib/tree/716e81ac7bf6fd135c86d54eb51435c6a1a3f403/internal/examples/rmsnorm_2x4x4/data
  std::string data_path_prefix = OpInterface::get_dod_base_dir() + "\\" +
                                 "tests" + "\\" + "cpp" + "\\" + "unit_tests" +
                                 "\\" + "testDataMladf" + "\\" +
                                 "llama2_2x4x4_mladfrmsnorm_2048_4096" + "\\";
  std::string a_bin_path = data_path_prefix + "ifm32.bin";
  std::string wts_bin_path = data_path_prefix + "wts32.bin";
  std::string ofm_bin_path = data_path_prefix + "ofm32.bin";

  mladfsoftmax_helpers::read_bin_to_vector(a_bin_path, a);
  mladfsoftmax_helpers::read_bin_to_vector(wts_bin_path, wts);
  mladfsoftmax_helpers::read_bin_to_vector(ofm_bin_path, reference_out);

  const auto &xclbin_fname =
      LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_REL_PATH;
  OpsFusion::FusionRuntime rt(xclbin_fname);
  auto meta = OpsFusion::load_meta_json(meta_json);
  rt.init(meta);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{a.data(), a_shape, a_dtype},
                   {wts.data(), wts_shape, wts_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{aie_out.data(), a_shape, c_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  return dd::count_errors_bfloat16vsbfloat16(
      reference_out, aie_out, a_shape,
      ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>::EPSILON);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_mladfrmsnorm.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }
  size_t M = 2048;
  size_t N = 4096;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_mladfrmsnorm(meta_json, M, N, false);
    if (err_count > 0) {
      std::cout << "mladfrmsnorm test failed with err_count = " << err_count
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
