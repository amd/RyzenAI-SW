#ifndef __TORCH_LINEAR_H__
#define __TORCH_LINEAR_H__

#include <tuple>
#include <vector>

// Torch headers
#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <linear.hpp>
#include <logging.h>
using namespace ryzenai;

namespace aie {
class torch_linear {
private:
  ryzenai::linear linbf16_;
  std::tuple<int, int> weight_shape_;

public:
  torch_linear(const std::tuple<int, int> &kernel_x_shape,
               const std::tuple<int, int> &kernel_y_shape);
  void initialize_weights(torch::Tensor w);
  torch::Tensor execute(torch::Tensor a);
};

torch_linear::torch_linear(const std::tuple<int, int> &kernel_x_shape,
                           const std::tuple<int, int> &kernel_y_shape)
    : linbf16_(ryzenai::linear(kernel_x_shape, kernel_y_shape)) {}

void torch_linear::initialize_weights(torch::Tensor w) {
  weight_shape_ = {w.sizes()[0], w.sizes()[1]};

  linbf16_.initialize_weights((bfloat16 *)w.data_ptr(), weight_shape_);
}

torch::Tensor torch_linear::execute(torch::Tensor a) {
  int64_t torch_zero_alloc_start = GET_ELAPSED_TIME_NS();
  torch::Tensor out =
      torch::zeros({a.sizes()[0], std::get<1>(weight_shape_)}, torch::kFloat);
  int64_t torch_zero_alloc_stop = GET_ELAPSED_TIME_NS();

  int64_t torch_exec_start = GET_ELAPSED_TIME_NS();
  std::tuple<int, int> a_shape = {a.sizes()[0], a.sizes()[1]};
  linbf16_.execute((bfloat16 *)a.data_ptr(), a_shape, (float *)out.data_ptr());
  int64_t torch_exec_stop = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(torch_exec_stop - torch_exec_start) + " " +
      std::to_string(torch_zero_alloc_stop - torch_zero_alloc_start) + "\n");

  return out.to(torch::kBFloat16);
}
} // namespace aie

#endif /* __TORCH_LINEAR_H__ */
