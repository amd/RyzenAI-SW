#include "../include/elemw_add_torch.hpp"

#if __has_include(<ryzenai/dynamic_dispatch/ops/mladfadd/mladfadd.hpp>)
#include <ryzenai/dynamic_dispatch/ops/mladfadd/mladfadd.hpp>
#else
#include <ops/mladfadd/mladfadd.hpp>
#endif

#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#include "utils.h"

aie::elemw_add_torch::elemw_add_torch() {}
aie::elemw_add_torch::~elemw_add_torch() {}

template <typename InOutT = int16_t>
InOutT *run_mladfadd(InOutT *aInput, InOutT *bInput, size_t M, size_t K,
                     bool debug = false,
                     const std::string &a_dtype = "bfloat16",
                     const std::string &b_dtype = "bfloat16",
                     const std::string &c_dtype = "bfloat16") {

  std::vector<size_t> a_shape = {M, K};
  std::vector<size_t> b_shape = a_shape;
  std::vector<size_t> aie_out_shape = a_shape;

  InOutT *aie_out = new InOutT[M * K];

  ryzenai::mladf_add mladfaddKernel =
      ryzenai::mladf_add<InOutT, InOutT, InOutT>(a_dtype, true);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{aInput, a_shape, a_dtype}, {bInput, a_shape, a_dtype}};
  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out, aie_out_shape, c_dtype}};
  mladfaddKernel.execute(input_Tensor, output_Tensor);
  return aie_out;
}

torch::Tensor aie::elemw_add_torch::execute(torch::Tensor x, torch::Tensor y) {
  if ((x.dim() != 2) || (y.dim() != 2)) {
    throw std::runtime_error(
        "MLADFADD expects ONLY rank 2 tensors [M,K] for operands");
  }
  if (x.sizes() != y.sizes()) {
    throw std::runtime_error("MLADFADD expects shame shaped operands");
  }

  size_t M = x.sizes()[0];
  size_t K = x.sizes()[1];

  auto xCasted = static_cast<uint16_t *>(x.data_ptr());
  auto yCasted = static_cast<uint16_t *>(y.data_ptr());
  uint16_t *c;

  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    c = run_mladfadd<uint16_t>(xCasted, yCasted, M, K, false, "bfloat16",
                               "bfloat16", "bfloat16");
  }

  torch::Tensor out =
      torch::from_blob(c, {x.sizes()[0], x.sizes()[1]}, torch::kBFloat16);
  return out;
}
