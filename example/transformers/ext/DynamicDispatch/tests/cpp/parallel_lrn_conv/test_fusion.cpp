#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "test_common.hpp"

static void test_model(const std::string &meta_json, size_t niters) {
  std::string xclbin_fname =
      "xclbin\\stx\\4x2_psi_integrated_model_a16w8_qdq.xclbin";
  auto meta = OpsFusion::load_meta_json(meta_json);

  auto context =
      ryzenai::dynamic_dispatch::xrt_context::get_instance(xclbin_fname)
          ->get_context();

  OpsFusion::FusionRuntime rt(&context, "DPU");
  OpsFusion::DDConfig cfg = {3, false};
  rt.init(meta, "", cfg);
  auto fops2 = rt.get_txns();

  // Prepare inputs
  std::vector<std::vector<uint8_t>> inputs;
  std::vector<Tensor> in_tensors =
      OpsFusion::MetaUtils::get_input_tensors(meta);
  for (auto &tensor : in_tensors) {
    size_t sz = std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                size_t{1}, std::multiplies{}) *
                Utils::get_size_of_type(tensor.dtype);
    std::vector<uint8_t> in(sz, 1);
    rand_init_int((int8_t *)in.data(), in.size() / sizeof(int8_t));
    inputs.push_back(std::move(in));
    tensor.data = inputs.back().data();
  }

  // Prepare outputs
  std::vector<Tensor> out_tensors =
      OpsFusion::MetaUtils::get_output_tensors(meta);
  std::vector<std::vector<uint8_t>> outputs;
  for (auto &tensor : out_tensors) {
    size_t sz = std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                size_t{1}, std::multiplies{}) *
                Utils::get_size_of_type(tensor.dtype);

    outputs.emplace_back(sz, 1);
    tensor.data = outputs.back().data();
  }

  std::cout << OpsFusion::MetaUtils::get_summary(rt.get_meta()) << std::endl;

  std::cout << "Executing for iterations:" << niters << std::endl;
  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < niters; ++i) {
    rt.execute(in_tensors, out_tensors);
  }
  auto t2 = std::chrono::steady_clock::now();
  std::cout << "Avg. Time (ms) : "
            << std::chrono::duration<float, std::milli>(t2 - t1).count() /
                   niters
            << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : ops_fusion.exe <meta.json> [niters=1]" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    std::string meta_json = std::string(argv[1]);
    size_t niters = (argc > 2) ? std::atoll(argv[2]) : 1;
    test_model(meta_json, niters);

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
