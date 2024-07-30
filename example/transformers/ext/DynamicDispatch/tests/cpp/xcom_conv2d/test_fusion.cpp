#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include <fstream>

#include "test_common.hpp"

struct TestConfig {
  size_t H;
  size_t W;
  size_t C_in;
  size_t C_out;
  size_t kernel_size;
  size_t stride;

  friend bool operator<(const TestConfig &lhs, const TestConfig &rhs) {
    return std::tie(lhs.H, lhs.W, lhs.C_in, lhs.C_out, lhs.kernel_size,
                    lhs.stride) < std::tie(rhs.H, rhs.W, rhs.C_in, rhs.C_out,
                                           rhs.kernel_size, rhs.stride);
  }
};

const std::map<std::uint32_t, TestConfig> test_configs = {
    {0XFF, TestConfig{320, 320, 48, 24, 1, 1}}, // initial xint8 conv added
    {0, TestConfig{128, 128, 256, 512, 1, 1}},
    {1, TestConfig{64, 64, 4, 512, 3, 1}},
    {2, TestConfig{64, 64, 512, 512, 3, 1}},
};

template <typename InT, typename WgT, typename BiasT, typename OuT>
int test_xcom_conv2d(const std::string &meta_json, uint32_t case_idx) {

  DOD_ASSERT(case_idx <= 2, "Only support 3 test case at the moment!");

  const auto &test_config = test_configs.at(case_idx);

  const size_t H = test_config.H;
  const size_t W = test_config.W;
  const size_t C_in = test_config.C_in;
  const size_t C_out = test_config.C_out;
  const size_t kernel_size = test_config.kernel_size;
  const size_t stride = test_config.stride;
  const bool bias_en = true;

  std::string a_dtype = "int8";
  std::string b_dtype = "int8";
  std::string bias_dtype = "int8";
  std::string c_dtype = "int8";

  constexpr bool is_qdq = std::is_same_v<InT, std::uint16_t>;

  if constexpr (std::is_same_v<InT, std::uint16_t>) {
    a_dtype = "uint16";
  }

  if constexpr (std::is_same_v<WgT, std::uint8_t>) {
    b_dtype = "uint8";
  }

  if constexpr (std::is_same_v<BiasT, std::int32_t>) {
    bias_dtype = "int32";
  }

  if constexpr (std::is_same_v<OuT, std::uint16_t>) {
    c_dtype = "uint16";
  }

  constexpr size_t a_dtype_size = sizeof(InT);
  constexpr size_t b_dtype_size = sizeof(WgT);
  constexpr size_t bias_dtype_size = sizeof(BiasT);
  constexpr size_t c_dtype_size = sizeof(OuT);

  DOD_ASSERT(H % stride == 0, "H should be divisible by stride");
  DOD_ASSERT(W % stride == 0, "W should be divisible by stride");
  DOD_ASSERT(kernel_size == 1 || kernel_size == 3,
             "Expect kernel size to be 1x1 or 3x3");

  const size_t H_out = H / stride;
  const size_t W_out = W / stride;

  std::cout << "H: " << H << ", W: " << W << ", C_in: " << C_in << std::endl;
  std::cout << "H_out: " << H_out << ", W_out: " << W_out
            << ", C_out: " << C_out << std::endl;

  // should be same as dims in Model.py
  // underlying memory layout can be different
  std::vector<size_t> activations_shape = {1, H, W, C_in};
  std::vector<size_t> weights_shape = {
      C_out,
      C_in,
      kernel_size,
      kernel_size,
  };
  std::vector<size_t> bias_shape = {C_out, 1, 1, 1};

  std::vector<size_t> out_shape = {1, H_out, W_out, C_out};

  std::vector<InT> activations(H * W * C_in);
  // std::vector<WgT> weights(C_out * kernel_size * kernel_size * C_in);
  // std::vector<BiasT> bias(C_out, 0);
  std::vector<OuT> cpu_q_out(H_out * W_out * C_out, 0x3);
  std::vector<OuT> aie_out(H_out * W_out * C_out, 0xC);

  // for (size_t i = 0; i < 32; i++) {
  //   std::cout << "aie_out: " << (std::int32_t)aie_out.at(i)
  //             << ", cpu_q_out: " << (std::int32_t)cpu_q_out.at(i) <<
  //             std::endl;
  // }

  // TODO: initialize activations using random data and calculate golden for
  // this
  srand(0xABCD);
  initialize_random<InT>(activations, activations.size(), 5, -5);

  constexpr bool OVERRIDE_DATA = true;

  if constexpr (OVERRIDE_DATA) {
    std::string folder_name = "/bin/conv_case/";
    std::string activation_path =
        OpInterface::get_dod_base_dir() + folder_name + "/ifm.bin";
    std::vector<std::uint8_t> input_override =
        OpsFusion::read_bin_file<std::uint8_t>(activation_path);

    // Input activations dont get shuffled, just catch if any padding is done
    DOD_ASSERT(activations.size() * a_dtype_size == input_override.size(),
               "Data buffers size mismatch");
    memcpy(activations.data(), input_override.data(), input_override.size());

    // TODO: add golden function here
    std::string golden_path =
        OpInterface::get_dod_base_dir() + folder_name + "/golden_0.bin";
    std::vector<std::uint8_t> golden_override =
        OpsFusion::read_bin_file<std::uint8_t>(golden_path);

    DOD_ASSERT(cpu_q_out.size() * c_dtype_size == golden_override.size(),
               "Golden Data buffers size mismatch");
    memcpy(cpu_q_out.data(), golden_override.data(), golden_override.size());
  }

  std::string xclbin_fname = XCOM_4x4_XCLBIN_REL_PATH;

  if constexpr (is_qdq) {
    xclbin_fname = XCOM_4x4_Q_XCLBIN_REL_PATH;
  }

  auto meta = OpsFusion::load_meta_json(meta_json);

  OpsFusion::FusionRuntime rt(xclbin_fname);
  // TODO: enable scratch pad optimization pass - should co-exists with
  //       ops that need internal scratch pad space
  OpsFusion::DDConfig cfg = {false, false, false};
  rt.init(meta, "", cfg);

  // constants will be loaded through artifacts of model.py (e.g. 0.const etc)

  std::vector<Tensor> input_Tensor;
  struct Tensor act_T = {activations.data(), activations_shape, a_dtype};
  input_Tensor.push_back(act_T);

  std::vector<Tensor> output_Tensor;
  struct Tensor out_T = {aie_out.data(), out_shape, c_dtype};
  output_Tensor.push_back(out_T);

  rt.execute(input_Tensor, output_Tensor);

  int err_status = 0;

  if (0 == case_idx) {
    err_status = memcmp(cpu_q_out.data(), output_Tensor.at(0).data,
                        H_out * W_out * C_out * c_dtype_size);
  }

  constexpr size_t MAX_DIM_PRINT = 8;

  // underlying memory format is NHWC
  for (size_t i = 0; i < std::min(MAX_DIM_PRINT, H_out); i++) {
    for (size_t j = 0; j < std::min(MAX_DIM_PRINT, W_out); j++) {
      for (size_t k = 0; k < std::min(MAX_DIM_PRINT, C_out); k++) {
        size_t index = i * W_out * C_out + j * C_out + k;

        bool mismatch = (std::int32_t)aie_out.at(index) !=
                        (std::int32_t)cpu_q_out.at(index);
        std::cout << "aie_out: " << (std::int32_t)aie_out.at(index);
        std::cout << ", cpu_q_out: " << (std::int32_t)cpu_q_out.at(index);
        if (mismatch) {
          err_status = -1;
          std::cout << "  DIFFERS!" << std::endl;
        } else {
          std::cout << "  EQUALS!" << std::endl;
        }
      }
    }
  }

  std::cout << "err_status : " << err_status << std::endl;

  return err_status;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage : ops_fusion.exe <meta.json> <config>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;

  try {
    std::string meta_json = std::string(argv[1]);
    int config = std::stoi(std::string(argv[2]));

    using A8 = std::int8_t;
    using W8 = std::int8_t;
    using B8 = std::int8_t;
    using OUT8 = std::int8_t;

    using AU16 = std::uint16_t;
    using WU8 = std::uint8_t;
    using OUTU16 = std::uint16_t;
    using B32 = std::int32_t;

    int err_status = 0;
    if (0xFF == config) {
      err_status = test_xcom_conv2d<A8, W8, B8, OUT8>(meta_json, config);
    } else {
      err_status = test_xcom_conv2d<AU16, WU8, B32, OUTU16>(meta_json, config);
    }

    if (err_status) {
      std::cout << "XCOM::CONV2D Test failed!" << std::endl;
      return EXIT_FAILURE;
    } else {
      std::cout << "XCOM::CONV2D Test Passed!" << std::endl;
    }

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
