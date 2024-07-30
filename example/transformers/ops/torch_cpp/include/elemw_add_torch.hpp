#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace aie {
class elemw_add_torch {
private:
public:
  elemw_add_torch();
  ~elemw_add_torch();
  torch::Tensor execute(torch::Tensor x, torch::Tensor y);
};
} // namespace aie
