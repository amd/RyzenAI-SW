#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <ops/mladfmharope/mladfmharope.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "test_common.hpp"

// z = RoPE(x)
template <typename InT = uint16_t, typename TrigT = uint16_t,
          typename OuT = uint16_t>
static int test_mladfmharope(const std::string &meta_json, size_t B, size_t M,
                             size_t K, bool debug = false,
                             const std::string &a_dtype = "bfloat16",
                             const std::string &trig_dtype = "bfloat16",
                             const std::string &c_dtype = "bfloat16") {
  // TODO
  // start of duplicated code from unit test
  std::vector<size_t> a_shape = {B, M, K};
  std::vector<size_t> trig_shape = {2, M, K};

  // simple test vector for functionality
  // ifm = all ones
  // trig = all ones
  std::vector<InT> a(B * M * K, dd::float_to_bfloat16(1.0f));
  std::vector<TrigT> trig(2 * M * K, dd::float_to_bfloat16(1.0f));
  //  ==> Rope = half zeros half two
  // TODO: I believe these should be interleaved but current kernel has
  // contigous K/2 0s and then 2s
  std::vector<float> cpu_float(B * M * K, 0.0f);
  for (int i = 0; i < cpu_float.size(); ++i) {
    if (i % K >= K / 2) {
      cpu_float.at(i) = 2.0f;
    }
  }

  std::vector<OuT> aie_out(B * M * K, garbage_value);

  const auto &xclbin_fname =
      LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_REL_PATH;
  OpsFusion::FusionRuntime rt(xclbin_fname);
  auto meta = OpsFusion::load_meta_json(meta_json);
  rt.init(meta);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{a.data(), a_shape, a_dtype},
                   {trig.data(), trig_shape, trig_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{aie_out.data(), a_shape, c_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  return dd::count_errors_floatvsbfloat16(
      cpu_float, aie_out, a_shape,
      ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>::EPSILON);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_mladfmharope.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }
  size_t B = 32;
  size_t M = 4096;
  size_t N = 128;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_mladfmharope(meta_json, B, M, N, false);
    if (err_count > 0) {
      std::cout << "mladfmharope test failed with err_count = " << err_count
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
