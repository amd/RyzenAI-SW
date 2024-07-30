#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace aie {
class rmsnorm_torch {
public:
  rmsnorm_torch();
  ~rmsnorm_torch();
  torch::Tensor execute(torch::Tensor x);
};
} // namespace aie
