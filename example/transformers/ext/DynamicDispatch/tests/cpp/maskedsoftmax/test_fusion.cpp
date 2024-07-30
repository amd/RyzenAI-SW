#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <ops/maskedsoftmax/maskedsoftmax.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "maskedsoftmax_helpers.hpp"
#include "test_common.hpp"

// z = x * y
template <typename InT = uint16_t, typename MaskT = uint16_t,
          typename OuT = uint16_t>
static int test_maskedsoftmax(const std::string &meta_json, size_t B, size_t M,
                              size_t K, bool debug = false,
                              const std::string &a_dtype = "bfloat16",
                              const std::string &mask_dtype = "bfloat16",
                              const std::string &c_dtype = "bfloat16") {
  // TODO
  // start of duplicated code from unit test
  std::vector<size_t> a_shape = {B, M, K};
  std::vector<size_t> mask_shape = {1, M, K};

  std::vector<InT> a(B * M * K);
  // Range taken from
  // https://gitenterprise.xilinx.com/AIELibs/mllib/blob/dev/internal/models/python/restructured/operators/Transformers/SoftMax.py#L348
  dd::initialize_random_bfloat16(a, 5);

  std::vector<InT> mask(
      M * K, dd::float_to_bfloat16(-std::numeric_limits<float>::infinity()));
  dd::initialize_lowertriangular(mask, M, K, dd::float_to_bfloat16(0.0));

  std::vector<float> cpu_float = maskedsoftmax_helpers::golden_maskedsoftmax(
      {B, M, K}, a, mask,
      ryzenai::masked_softmax<uint16_t, uint16_t,
                              uint16_t>::DEFAULT_PREMASK_SCALE);

  std::vector<OuT> aie_out(B * M * K, garbage_value);

  const auto &xclbin_fname =
      LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_REL_PATH;
  OpsFusion::FusionRuntime rt(xclbin_fname);
  auto meta = OpsFusion::load_meta_json(meta_json);
  rt.init(meta);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{a.data(), a_shape, a_dtype},
                   {mask.data(), mask_shape, mask_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{aie_out.data(), a_shape, c_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  return dd::count_errors_floatvsbfloat16(
      cpu_float, aie_out, a_shape,
      ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>::EPSILON);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_maskedsoftmax.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }
  size_t B = 32;
  size_t M = 2048;
  size_t N = 2048;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_maskedsoftmax(meta_json, B, M, N, false);
    if (err_count > 0) {
      std::cout << "MaskedSoftmax test failed with err_count = " << err_count
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
