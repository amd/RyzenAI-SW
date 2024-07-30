#include "../include/gemm_torch.hpp"
#include <iostream>

namespace {
// Utility function to get the shape of a tensor as a vector of size_t
std::vector<size_t> getTensorShape(const torch::Tensor &tensor) {
  // Get the shape of the tensor as a vector of int64_t
  std::vector<int64_t> int64_shape = tensor.sizes().vec();

  // Create a vector of size_t to hold the shape
  std::vector<size_t> size_t_shape(int64_shape.size());

  // Transform the vector of int64_t to size_t
  std::transform(int64_shape.begin(), int64_shape.end(), size_t_shape.begin(),
                 [](int64_t val) { return static_cast<size_t>(val); });

  return size_t_shape;
}
} // namespace

aie::gemm_torch::gemm_torch(int mm, int kk, int nn, bool bb)
    : m(mm), k(kk), n(nn), bval(bb),
      gemm_("bfloat16", "uint4", "bfloat16", true) {}

aie::gemm_torch::~gemm_torch() {}

void aie::gemm_torch::initialize_params(torch::Tensor qw, torch::Tensor qz,
                                        torch::Tensor sc, torch::Tensor b,
                                        int gs) {
  group_size = gs;
  qweight = qw.contiguous(); // k x n
  qzero = qz.contiguous();   // k/group_size x n
  scales = sc.contiguous();  // k/group_size x n
  bias = b.contiguous();     // 1 x n

  scales_float = scales.to(torch::kFloat);
  bias_float = bias.to(torch::kFloat);

  // BE WARNED OF MASSIVE HACK
  // GROUP SIZE IS COMMUNICATED TO DD through scale shape

  // DD Tensors
  Tensor weight_tensor = {qweight.data_ptr<int8_t>(), getTensorShape(qweight),
                          "uint4"};
  Tensor bias_tensor = {bias_float.data_ptr<float>(), getTensorShape(bias),
                        "float"};
  Tensor scales_tensor = {
      scales_float.data_ptr<float>(), {(size_t)gs, 0}, "float"};
  Tensor zeros_tensor = {qzero.data_ptr<int8_t>(), getTensorShape(qzero),
                         "uint4"};
  std::vector<Tensor> constant_tensors = {weight_tensor, bias_tensor,
                                          scales_tensor, zeros_tensor};

  gemm_.initialize_const_params(constant_tensors);
}

torch::Tensor aie::gemm_torch::execute(torch::Tensor x) {

  if (!x.is_contiguous()) {
    std::cout << "Warning: gemm_torch was provided a noncontiguous input "
                 "tensor, this will impact performance!"
              << std::endl;
    x = x.contiguous();
  }
  // else {
  //  std::cout << "INFO: gemm_torch was provided a contiguous input" <<
  //  std::endl;
  // }

  auto y = torch::empty({m, n}, torch::dtype(torch::kBFloat16));

  // DD Tensors
  Tensor input_tensor = {(int16_t *)x.data_ptr<torch::BFloat16>(),
                         getTensorShape(x), "bfloat16"};
  Tensor output_tensor = {(int16_t *)y.data_ptr<torch::BFloat16>(),
                          getTensorShape(y), "bfloat16"};

  std::vector<Tensor> input_tensors = {input_tensor};
  std::vector<Tensor> output_tensors = {output_tensor};

  gemm_.execute(input_tensors, output_tensors);

  return y;
}
