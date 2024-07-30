#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>
#if __has_include(<ryzenai/dynamic_dispatch/ops/elwmul/elwmul.hpp>)
#include <ryzenai/dynamic_dispatch/ops/elwmul/elwmul.hpp>
#else
#include <ops/elwmul/elwmul.hpp>
#endif
namespace aie {
class elemw_mul_torch {
  ryzenai::elw_mul<uint16_t, uint16_t, uint16_t> mulKernel =
      ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>("bfloat16", true);

public:
  std::string xclbinFileName;
  elemw_mul_torch(size_t size);
  ~elemw_mul_torch();
  torch::Tensor execute(torch::Tensor x, torch::Tensor y);
};
} // namespace aie
