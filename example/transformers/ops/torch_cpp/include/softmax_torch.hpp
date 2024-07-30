#ifndef __SOFTMAX_TORCH_HEADER__
#define __SOFTMAX_TORCH_HEADER__

#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace aie {
class softmax_torch {
public:
  softmax_torch();
  ~softmax_torch();

  void run_softmax(int16_t *aInput, int16_t *bInput, int16_t *cOutput, int B,
                   int M, int K);

  torch::Tensor execute(torch::Tensor x, torch::Tensor y);
};
} // namespace aie

#endif
