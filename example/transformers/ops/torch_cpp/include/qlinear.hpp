#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace cpu {
class qlinear {
public:
  torch::Tensor weights;
  torch::Tensor weights_scale;
  torch::Tensor requantize_in_scale;
  torch::Tensor requantize_out_scale;
  torch::Tensor req_scale;
  qlinear(torch::Tensor w, torch::Tensor w_scale, torch::Tensor req_in_scale,
          torch::Tensor req_out_scale);
  ~qlinear();
  torch::Tensor qmmul(torch::Tensor x);
};

qlinear::qlinear(torch::Tensor w, torch::Tensor w_scale,
                 torch::Tensor req_in_scale, torch::Tensor req_out_scale) {
  weights = at::transpose(w, 0, 1);
  weights_scale = w_scale;
  requantize_in_scale = req_in_scale;
  requantize_out_scale = req_out_scale;
  req_scale = requantize_in_scale / requantize_out_scale;
}

qlinear::~qlinear() {}

torch::Tensor qlinear::qmmul(torch::Tensor x) {
  /*inputs:
      x: fp32 Tensor
      weights: int8 Tensor
    outputs:
      y: fp32 tensor

    1. Quantize activations
    2. Compute GEMM : int16 = int8 @ int8
    3. Dequantize by multiplying with weight scale factor
    4. output is a fp32 tensor
  */
  /*
  // don't do this please : std::cout << weights <<std::endl;
  std::cout << "x.sizes(): " << x.sizes() << std::endl;
  std::cout << "x.scalar_type(): " << x.scalar_type() << std::endl;
  std::cout << "x.numel(): " << x.numel() << std::endl;
  std::cout << "weights.sizes(): " << weights.sizes() << std::endl;
  std::cout << "weights.scalar_type(): " << weights.scalar_type() << std::endl;
  std::cout << "weights.numel(): " << weights.numel() << std::endl;
  std::cout << "w_scale.sizes(): " << w_scale.sizes() << std::endl;
  std::cout << "w_scale.scalar_type(): " << w_scale.scalar_type() << std::endl;
  std::cout << "w_scale.numel(): " << w_scale.numel() << std::endl;
  */
  torch::Tensor x_scale = at::max(at::abs(x)) / 128;
  x = at::round((1 / x_scale) * x); // + zp
  x = x.to(torch::kInt32);
  x = at::clip(x, -128, 128);
  auto res = torch::mm(x, weights);
  res = res * req_scale;
  res = at::clip(res, -32768, 32767);
  // std::cout << req_scale << std::endl;
  // std::cout << requantize_in_scale << std::endl;
  // std::cout << requantize_out_scale << std::endl;
  res =
      res.to(torch::kFloat32) * x_scale * weights_scale * requantize_out_scale;
  return res;
}
} // namespace cpu
