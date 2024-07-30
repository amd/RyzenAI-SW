#ifndef __GEMM_TORCH__
#define __GEMM_TORCH__
#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#if __has_include(<ryzenai/dynamic_dispatch/ops/mladfmatmulbias/mladfmatmulbias.hpp>)
#include <ryzenai/dynamic_dispatch/ops/mladfmatmulbias/mladfmatmulbias.hpp>
#else
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>
#endif

namespace aie {
class gemm_torch {
public:
  int m, k, n;
  bool bval;
  torch::Tensor qweight, qzero; // int8 container holding (u)int4 values
  torch::Tensor scales;         // bfloat16
  torch::Tensor scales_float;   // float32
  torch::Tensor bias;           // bfloat16
  torch::Tensor bias_float;     // float32
  int group_size;

  gemm_torch(int mm, int kk, int nn, bool bb);
  ~gemm_torch();
  void initialize_params(torch::Tensor qw, torch::Tensor qz, torch::Tensor sc,
                         torch::Tensor b, int gs);
  torch::Tensor execute(torch::Tensor x);

private:
  ryzenai::mladfmatmulbias<int16_t, int8_t, int16_t> gemm_;
};
} // namespace aie

#endif
