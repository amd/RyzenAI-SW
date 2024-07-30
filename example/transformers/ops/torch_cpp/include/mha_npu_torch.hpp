#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../include/bmm_torch.hpp"
#include "../include/softmax_torch.hpp"

#define HEAD_DIM 128

#if __has_include(<ryzenai/dynamic_dispatch/ops/bmm/bmm.hpp>)
#include <ryzenai/dynamic_dispatch/ops/bmm/bmm.hpp>
#include <ryzenai/dynamic_dispatch/ops/maskedsoftmax/maskedsoftmax.hpp>
#else
#include <ops/bmm/bmm.hpp>
#include <ops/maskedsoftmax/maskedsoftmax.hpp>
#endif

namespace aie {
class mha_npu_torch {

  ryzenai::bmm<uint16_t, uint16_t, uint16_t> bmm1 =
      ryzenai::bmm<uint16_t, uint16_t, uint16_t>("bfloat16", "bfloat16",
                                                 "bfloat16", false);
  ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t> softmax =
      ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>("bfloat16", true);

  ryzenai::bmm<uint16_t, uint16_t, uint16_t> bmm2 =
      ryzenai::bmm<uint16_t, uint16_t, uint16_t>("bfloat16", "bfloat16",
                                                 "bfloat16", false);

  std::vector<xrt::bo> bmm1_inputs;
  std::vector<xrt::bo> bmm1_outputs;
  std::vector<xrt::bo> bmm2_inputs;
  std::vector<xrt::bo> bmm2_outputs;
  xrt::bo softmax_mask;

public:
  torch::Tensor bmm_scale;
  mha_npu_torch();
  ~mha_npu_torch();
  torch::Tensor execute(torch::Tensor query_states, torch::Tensor key_states,
                        torch::Tensor value_states,
                        torch::Tensor attention_mask);
};
} // namespace aie
