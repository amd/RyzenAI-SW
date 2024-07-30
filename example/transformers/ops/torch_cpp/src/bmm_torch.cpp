#include "../include/bmm_torch.hpp"
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

aie::bmm_torch::bmm_torch(bool tr) {
  transpose = tr;
  aie::bmm_torch::bmmKernel.debug(false);
  int B;
  int M;
  int K;
  if (transpose == true) {
    B = 32;
    M = 2048;
    K = 128;
  } else {
    B = 32;
    M = 2048;
    K = 2048;
  }
  size_t BMs = static_cast<size_t>(B * M);
  size_t Ks = static_cast<size_t>(K);
  std::vector<size_t> a_shape = {BMs, Ks};
  aie::bmm_torch::bmmKernel.set_params("BMM", a_shape);
}

aie::bmm_torch::~bmm_torch() {}

void aie::bmm_torch::run_bmm(uint16_t *aInput, uint16_t *bInput,
                             uint16_t *aie_out, int M, int K, int N, int B) {
  int BM = B * M;
  size_t BMs = static_cast<size_t>(BM);
  size_t Ks = static_cast<size_t>(K);
  size_t BKs = static_cast<size_t>(B * K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {BMs, Ks};
  std::vector<size_t> b_shape = {BKs, Ns};
  std::vector<size_t> aie_out_shape = {BMs, Ns};
  std::vector<Tensor> const_Tensor;
  const_Tensor = {{bInput, b_shape, "bfloat16"}};
  aie::bmm_torch::bmmKernel.initialize_const_params(const_Tensor);
  std::vector<Tensor> input_Tensor;
  input_Tensor = {{aInput, a_shape, "bfloat16"}};
  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out, aie_out_shape, "bfloat16"}};
  aie::bmm_torch::bmmKernel.execute(input_Tensor, output_Tensor);
}

torch::Tensor aie::bmm_torch::execute(torch::Tensor x, torch::Tensor y) {
  int B = x.sizes()[0];
  int M = x.sizes()[1];
  int K = x.sizes()[2];
  int N = y.sizes()[2];
  if (transpose == true) {
    N = y.sizes()[1];
  }
  auto z = torch::empty({B, M, N}).to(torch::kBFloat16);
  auto xCasted = static_cast<uint16_t *>(x.data_ptr());
  auto yCasted = static_cast<uint16_t *>(y.data_ptr());
  auto zCasted = static_cast<uint16_t *>(z.data_ptr());
  run_bmm(xCasted, yCasted, zCasted, M, K, N, B);
  return z;
}
