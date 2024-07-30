#include "../include/mlp_npu_torch.hpp"

#include "logging.h"
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#ifdef RYZENAI_PERF
using namespace ryzenai;
#endif

aie::mlp_npu_torch::mlp_npu_torch() {
  siluKernel.debug(false);
  mulKernel.debug(false);
}

aie::mlp_npu_torch::~mlp_npu_torch() {}

void aie::mlp_npu_torch::initialize_weights(torch::Tensor data) {}

torch::Tensor aie::mlp_npu_torch::execute(torch::Tensor x, torch::Tensor y) {

  int K = 1;
  int M = 1;
  int N = 11008;
  if (x.dim() == 2) {
    M = x.sizes()[0];
    N = x.sizes()[1];
  }
  if (x.dim() == 3) {
    K = x.sizes()[0];
    M = x.sizes()[1];
    N = x.sizes()[2];
  }

  auto xCasted = static_cast<uint16_t *>(x.data_ptr());
  auto yCasted = static_cast<uint16_t *>(y.data_ptr());
  std::vector<size_t> a_shape = {static_cast<size_t>(K * M),
                                 static_cast<size_t>(N)};

  torch::Tensor out;
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape_silu = {Ms, Ns};
  siluKernel.set_kernel_shape(a_shape_silu);
  auto inputs = siluKernel.get_inputs();
  auto outputs = siluKernel.get_outputs();
  uint16_t *a_bo_map = inputs[0].map<uint16_t *>();
  memcpy((void *)a_bo_map, (void *)xCasted, Ms * Ns * sizeof(uint16_t));
  inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  siluKernel.execute(inputs, outputs);

  auto inputs_mul = mulKernel.get_inputs();
  auto outputs_mul = mulKernel.get_outputs();
  mulKernel.set_kernel_shape(a_shape);
  uint16_t *b_bo_map = inputs_mul[1].map<uint16_t *>();
  memcpy((void *)b_bo_map, (void *)yCasted, M * K * N * sizeof(uint16_t));
  inputs_mul[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  outputs_mul[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  std::vector<xrt::bo> inputs_2 = {outputs[0], inputs_mul[1]};
  mulKernel.execute(inputs_2, outputs_mul);
  uint16_t *c_bo_map = outputs_mul[0].map<uint16_t *>();

  out = torch::from_blob((void *)c_bo_map, x.sizes(), torch::kBFloat16);

  return out;
}
