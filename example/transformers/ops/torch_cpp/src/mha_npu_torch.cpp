#include "../include/mha_npu_torch.hpp"

#include "logging.h"
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#ifdef RYZENAI_PERF
using namespace ryzenai;
#endif

aie::mha_npu_torch::mha_npu_torch() {
  const float sc[] = {(float)HEAD_DIM};
  bmm_scale = torch::from_blob((void *)sc, {1, 1});
  bmm_scale = 1 / torch::sqrt(bmm_scale);
  bmm_scale = bmm_scale.to(torch::kBFloat16);

  bmm1.debug(false);
  bmm2.debug(false);
  softmax.debug(false);

  std::vector<size_t> a_shape_1 = {32 * 2048, 128};
  std::vector<size_t> a_shape_2 = {32 * 2048, 2048};

  bmm1.set_params("BMM", a_shape_1);
  bmm2.set_params("BMM", a_shape_2);
  bmm1_inputs = bmm1.allocate_inputs();
  bmm1_outputs = bmm1.allocate_outputs();
  bmm2_inputs = bmm2.allocate_inputs();
  bmm2_outputs = bmm2.allocate_outputs();
  softmax_mask = softmax.get_inputs()[1];
}

aie::mha_npu_torch::~mha_npu_torch() {}

torch::Tensor aie::mha_npu_torch::execute(torch::Tensor query_states,
                                          torch::Tensor key_states,
                                          torch::Tensor value_states,
                                          torch::Tensor attention_mask) {
  int64_t exec_start = 0, exec_end = 0, time0 = 0, time1 = 0, time2 = 0,
          time3 = 0;
  int B = query_states.sizes()[0];
  int M = query_states.sizes()[1];
  int K = query_states.sizes()[2];
  int N = key_states.sizes()[1];
  exec_start = GET_ELAPSED_TIME_NS();

  auto xCasted = static_cast<uint16_t *>(query_states.data_ptr());
  auto yCasted = static_cast<uint16_t *>(key_states.data_ptr());
  auto mCasted = static_cast<uint16_t *>(attention_mask.data_ptr());
  auto y2Casted = static_cast<uint16_t *>(value_states.data_ptr());
  uint16_t *a_bo_map = bmm1_inputs[0].map<uint16_t *>();
  memcpy((void *)a_bo_map, (void *)xCasted, B * M * K * sizeof(uint16_t));
  uint16_t *b_bo_map = bmm1_inputs[1].map<uint16_t *>();
  memcpy((void *)b_bo_map, (void *)yCasted, B * K * N * sizeof(uint16_t));

  uint16_t *out = bmm2_outputs[0].map<uint16_t *>();

  bmm1_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm1_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm1_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  exec_end = GET_ELAPSED_TIME_NS();
  time0 = exec_end - exec_start;

  exec_start = GET_ELAPSED_TIME_NS();
  bmm1.execute(bmm1_inputs, bmm1_outputs);

  exec_end = GET_ELAPSED_TIME_NS();
  time1 = exec_end - exec_start;

  exec_start = GET_ELAPSED_TIME_NS();
  uint16_t *mask_bo_map = softmax_mask.map<uint16_t *>();
  memcpy((void *)mask_bo_map, (void *)mCasted, M * N * sizeof(uint16_t));
  softmax_mask.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm2_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<xrt::bo> inputs = {bmm1_outputs[0], softmax_mask};
  std::vector<xrt::bo> outputs = {bmm2_inputs[0]};
  softmax.execute(inputs, outputs);

  exec_end = GET_ELAPSED_TIME_NS();
  time2 = exec_end - exec_start;

  exec_start = GET_ELAPSED_TIME_NS();
  uint16_t *value_bo_map = bmm2_inputs[1].map<uint16_t *>();
  memcpy((void *)value_bo_map, (void *)y2Casted, B * N * K * sizeof(uint16_t));
  bmm2_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm2_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm2.execute(bmm2_inputs, bmm2_outputs);
  bmm2_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  exec_end = GET_ELAPSED_TIME_NS();
  time3 = exec_end - exec_start;
  auto bmm2_out = torch::from_blob((void *)out, {B, M, K}, torch::kBFloat16);
  RYZENAI_LOG_INFO(
      std::string("aie::mha_npu_torch::execute  bmm1 ") +
      std::string(" prepare ") + std::to_string(time0) std::to_string(time1) +
      std::string(" softmax ") + std::to_string(time2) std::string(" bmm2 ") +
      std::to_string(time3) + std::string("\n"));
  return bmm2_out;
}
