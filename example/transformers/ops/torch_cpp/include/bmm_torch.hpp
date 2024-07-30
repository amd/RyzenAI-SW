#ifndef __BMM__TORCH__
#define __BMM__TORCH__
#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#if __has_include(<ryzenai/dynamic_dispatch/ops/bmm/bmm.hpp>)
#include <ryzenai/dynamic_dispatch/ops/bmm/bmm.hpp>
#else
#include <ops/bmm/bmm.hpp>
#endif

namespace aie {
class bmm_torch {
private:
  bool transpose;
  ryzenai::bmm<uint16_t, uint16_t, uint16_t> bmmKernel =
      ryzenai::bmm<uint16_t, uint16_t, uint16_t>("bfloat16", "bfloat16",
                                                 "bfloat16", false);

  void run_bmm(uint16_t *aInput, uint16_t *bInput, uint16_t *aie_out, int M,
               int K, int N, int B);

public:
  bmm_torch(bool tr);
  ~bmm_torch();
  torch::Tensor execute(torch::Tensor x, torch::Tensor y);
};
} // namespace aie
#endif
