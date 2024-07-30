#include "../include/softmax_torch.hpp"
#include "../../cpp/softmax/softmax.hpp"

#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

aie::softmax_torch::softmax_torch() {}
aie::softmax_torch::~softmax_torch() {}

void aie::softmax_torch::run_softmax(int16_t *aInput, int16_t *bInput,
                                     int16_t *cOutput, int B, int M, int K) {

  std::tuple<int, int, int> a_shape = {B, M, K};

  const bool debug = false;
  const std::string &inout_dtype = "bfloat16";

  ryzenai::softmax softmaxKernel =
      ryzenai::softmax<int16_t>(inout_dtype, inout_dtype, inout_dtype);
  softmaxKernel.debug(debug);
  softmaxKernel.execute(aInput, bInput, cOutput, a_shape);
}

torch::Tensor aie::softmax_torch::execute(torch::Tensor x, torch::Tensor y) {
  int B = x.sizes()[0];
  int M = x.sizes()[1];
  int K = x.sizes()[2];

  auto z = torch::empty({B, M, K}).to(torch::kBFloat16);
  auto xCasted = static_cast<int16_t *>(x.data_ptr());
  auto yCasted = static_cast<int16_t *>(y.data_ptr());
  auto zCasted = static_cast<int16_t *>(z.data_ptr());

  aie::softmax_torch::run_softmax(xCasted, yCasted, zCasted, B, M, K);
  return z;
}
