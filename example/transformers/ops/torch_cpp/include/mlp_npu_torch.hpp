#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#if __has_include(<ryzenai/dynamic_dispatch/ops/elwmul/elwmul.hpp>)
#include <ryzenai/dynamic_dispatch/ops/elwmul/elwmul.hpp>
#else
#include <ops/elwmul/elwmul.hpp>
#endif
#if __has_include(<ryzenai/dynamic_dispatch/ops/silu/silu.hpp>)
#include <ryzenai/dynamic_dispatch/ops/silu/silu.hpp>
#else
#include <ops/silu/silu.hpp>
#endif

namespace aie {
class mlp_npu_torch {
  ryzenai::elw_mul<uint16_t, uint16_t, uint16_t> mulKernel =
      ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>("bfloat16", true);
  ryzenai::silu<uint16_t, uint16_t> siluKernel =
      ryzenai::silu<uint16_t, uint16_t>("bfloat16", true);

public:
  torch::Tensor bmm_scale;
  mlp_npu_torch();
  ~mlp_npu_torch();
  void initialize_weights(torch::Tensor data);
  torch::Tensor execute(torch::Tensor x, torch::Tensor y);
};
} // namespace aie
