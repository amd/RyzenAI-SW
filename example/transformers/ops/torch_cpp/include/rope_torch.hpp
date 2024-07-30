#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace aie {
class rope_torch {
public:
  rope_torch();
  ~rope_torch();
  torch::Tensor execute(torch::Tensor x);
};
} // namespace aie
