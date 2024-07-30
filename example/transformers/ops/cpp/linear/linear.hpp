/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __LINEAR_H__
#define __LINEAR_H__

#include <fstream>
#include <iostream>
#include <tuple>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include "dpu_kernel_metadata.hpp"

// Headers to create Txn binary
#include "op_buf.hpp"
#include "op_types.h"

#include "xrt_context.hpp"

// AIE Driver header
#include "xaiengine.h"

#include "logging.h"
#include "utils.h"

using bfloat16 = int16_t;

namespace ryzenai {
class linear {
private:
  int64_t kernel_x_shape_[2];
  int64_t kernel_y_shape_[2];
  int64_t kernel_z_shape_[2];
  int64_t a_shape_[2];
  int64_t c_shape_[2];
  int64_t w_shape_[2];
  int64_t w_padded_shape_[2];
  xrt_context *xrt_ctx_;
  xrt::bo a_bo_;
  xrt::bo c_bo_;
  bfloat16 *a_map_;
  std::vector<bfloat16> c_tmp_buf;
  std::vector<xrt::bo> weights_bo_;
  xrt::bo instr_bo_;
  int64_t a_copy_time_;
  int64_t a_sync_time_;
  int64_t c_copy_time_;
  int64_t c_sync_time_;
  int64_t run_aie_time_;
  int64_t cpu_acc_time_;
  int64_t num_run_aie_;

  void linear::run_aie(bfloat16 *a, xrt::bo &w_bo, bfloat16 *c,
                       int64_t *a_shape);
  void linear::initialize_instructions(std::string &txn_fname);
  void linear::execute_tiled(bfloat16 *a, float *c, int64_t *a_shape);

public:
  linear(const std::tuple<int, int> &kernel_x_shape,
         const std::tuple<int, int> &kernel_y_shape);
  void linear::initialize_weights(bfloat16 *weights,
                                  const std::tuple<int, int> &w_shape);
  void linear::execute(bfloat16 *a, const std::tuple<int, int> &a_shape,
                       float *c);
};

void linear::initialize_weights(bfloat16 *weights,
                                const std::tuple<int, int> &w_shape) {
  w_shape_[0] = std::get<0>(w_shape);
  w_shape_[1] = std::get<1>(w_shape);

  w_padded_shape_[0] = Utils::ceil_for_me(w_shape_[0], kernel_y_shape_[0]);
  w_padded_shape_[1] = Utils::ceil_for_me(w_shape_[1], kernel_y_shape_[1]);
  std::vector<bfloat16> weights_padded;

  auto w_ptr = weights;
  if ((w_padded_shape_[0] != w_shape_[0]) ||
      (w_padded_shape_[1] != w_shape_[1])) {
    weights_padded.resize(w_padded_shape_[1] * w_padded_shape_[0]);
    memset(weights_padded.data(), 0,
           w_padded_shape_[0] * w_padded_shape_[1] * sizeof(bfloat16));
    Utils::_copy_pad_data<bfloat16>((bfloat16 *)weights, weights_padded.data(),
                                    &w_shape_[0], &w_padded_shape_[0]);
    w_ptr = weights_padded.data();
  }

  auto device_ = xrt_ctx_->get_device();
  auto kernel_ = xrt_ctx_->get_kernel();

  for (int64_t rb = 0; rb < w_padded_shape_[0]; rb += kernel_y_shape_[0]) {
    for (int64_t cb = 0; cb < w_padded_shape_[1]; cb += kernel_y_shape_[1]) {
      auto offset = cb + rb * w_padded_shape_[1];
      // create BO, copy tile to BO
      auto bo =
          xrt::bo(xrt_ctx_->get_context(),
                  kernel_y_shape_[0] * kernel_y_shape_[1] * sizeof(bfloat16),
                  xrt::bo::flags::host_only, kernel_.group_id(0));
      auto bo_map = bo.map<bfloat16 *>();
      Utils::_copy_tile<bfloat16>(w_ptr + offset, bo_map, &kernel_y_shape_[0],
                                  &w_padded_shape_[1]);
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      weights_bo_.push_back(bo);
    }
  }
}

void linear::initialize_instructions(std::string &txn_fname) {
  /* Create intruction buffer with transaction binary*/
  ifstream txn_bin(txn_fname, ios::binary);
  XAie_TxnHeader hdr;
  txn_bin.read((char *)&hdr, sizeof(XAie_TxnHeader));
  printf("Header version %d.%d\n", hdr.Major, hdr.Minor);
  printf("Device Generation: %d\n", hdr.DevGen);
  printf("Cols, Rows, NumMemRows : (%d, %d, %d)\n", hdr.NumCols, hdr.NumRows,
         hdr.NumMemTileRows);
  printf("TransactionSize: %u\n", hdr.TxnSize);
  printf("NumOps: %u\n", hdr.NumOps);
  std::vector<uint8_t> txn(hdr.TxnSize);
  uint8_t *ptr = txn.data();
  std::memcpy(ptr, &hdr, sizeof(XAie_TxnHeader));
  ptr = ptr + sizeof(XAie_TxnHeader);
  txn_bin.read((char *)ptr, hdr.TxnSize - sizeof(XAie_TxnHeader));

  aiectrl::op_buf instr_buf;
  instr_buf.addOP(aiectrl::transaction_op(txn.data()));

  auto device_ = xrt_ctx_->get_device();
  auto kernel_ = xrt_ctx_->get_kernel();
  instr_bo_ = xrt::bo(xrt_ctx_->get_context(), instr_buf.ibuf_.size(),
                      xrt::bo::flags::cacheable, kernel_.group_id(1));
  instr_bo_.write(instr_buf.ibuf_.data());
  instr_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

linear::linear(const std::tuple<int, int> &kernel_x_shape,
               const std::tuple<int, int> &kernel_y_shape) {
  kernel_x_shape_[0] = std::get<0>(kernel_x_shape);
  kernel_x_shape_[1] = std::get<1>(kernel_x_shape);
  kernel_y_shape_[0] = std::get<0>(kernel_y_shape);
  kernel_y_shape_[1] = std::get<1>(kernel_y_shape);
  kernel_z_shape_[0] = std::get<0>(kernel_x_shape);
  kernel_z_shape_[1] = std::get<1>(kernel_y_shape);

  std::string xclbin_fname = "bfloat16_" + std::to_string(kernel_x_shape_[0]) +
                             "x" + std::to_string(kernel_x_shape_[1]) + "_" +
                             std::to_string(kernel_y_shape_[0]) + "x" +
                             std::to_string(kernel_y_shape_[1]) + ".xclbin";
  std::string txn_fname = "bfloat16_" + std::to_string(kernel_x_shape_[0]) +
                          "x" + std::to_string(kernel_x_shape_[1]) + "_" +
                          std::to_string(kernel_y_shape_[0]) + "x" +
                          std::to_string(kernel_y_shape_[1]) + "_txn.bin";

  std::string xclbin = std::string(std::getenv("PYTORCH_AIE_PATH")) +
                       "\\xclbin\\phx\\" + xclbin_fname;
  std::string txn_bin = std::string(std::getenv("PYTORCH_AIE_PATH")) +
                        "\\xclbin\\phx\\" + txn_fname;

  xrt_ctx_ = &xrt_context::get_instance(xclbin);

  auto device_ = xrt_ctx_->get_device();
  auto kernel_ = xrt_ctx_->get_kernel();

  a_bo_ = xrt::bo(xrt_ctx_->get_context(),
                  kernel_x_shape_[0] * kernel_x_shape_[1] * sizeof(bfloat16),
                  xrt::bo::flags::host_only, kernel_.group_id(0));
  c_bo_ = xrt::bo(xrt_ctx_->get_context(),
                  kernel_z_shape_[0] * kernel_z_shape_[1] * sizeof(bfloat16),
                  xrt::bo::flags::host_only, kernel_.group_id(0));
  a_map_ = a_bo_.map<bfloat16 *>();

  c_tmp_buf = std::vector<bfloat16>(kernel_x_shape_[0] * kernel_y_shape_[1]);

  initialize_instructions(txn_bin);

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;
}

bfloat16 toBfloat16(float f) {
  bfloat16 bf;
  std::memcpy(&bf, (uint8_t *)&f + 2, 2);
  return bf;
}

float toFloat(bfloat16 bf) {
  float f = 0;
  std::memcpy((uint8_t *)&f + 2, &bf, 2);
  return f;
}

template <typename T>
void _cpu_acc(float *acc, T *in, int64_t *tile_shape, int64_t *acc_stride) {
  for (int i = 0; i < tile_shape[0]; i++) {
    for (int j = 0; j < tile_shape[1]; j++) {
      float a = toFloat(in[i * tile_shape[1] + j]);
      acc[i * acc_stride[0] + j] += a;
    }
  }
}

// run with strided copy support
void linear::run_aie(bfloat16 *a, xrt::bo &w_bo, bfloat16 *c,
                     int64_t *a_shape) {
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  Utils::_copy_tile<bfloat16>(a, a_map_, &kernel_x_shape_[0], &a_shape[1]);
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();

  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  std::vector<u64> kargv(5, 0);
  auto kernel_ = xrt_ctx_->get_kernel();
  auto run = kernel_(OPCODE, instr_bo_, instr_bo_.size() / sizeof(int),
                     c_bo_.address() + DDR_AIE_ADDR_OFFSET,
                     a_bo_.address() + DDR_AIE_ADDR_OFFSET,
                     w_bo.address() + DDR_AIE_ADDR_OFFSET, kargv[3], kargv[4]);
  run.wait();
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();

  int64_t c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int64_t c_sync_stop = GET_ELAPSED_TIME_NS();

  int64_t c_copy_start = GET_ELAPSED_TIME_NS();
  c_bo_.read(c);
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ += a_copy_stop - a_copy_start;
  a_sync_time_ += a_sync_stop - a_sync_start;
  c_copy_time_ += c_copy_stop - c_copy_start;
  c_sync_time_ += c_sync_stop - c_sync_start;
  run_aie_time_ += run_aie_stop - run_aie_start;
  num_run_aie_++;
}

void linear::execute(bfloat16 *a, const std::tuple<int, int> &a_shape,
                     float *c) {
  int64_t exec_start = GET_ELAPSED_TIME_NS();
  a_shape_[0] = std::get<0>(a_shape);
  a_shape_[1] = std::get<1>(a_shape);
  c_shape_[0] = std::get<0>(a_shape);
  c_shape_[1] = w_shape_[1];

  int64_t a_new_shape[2] = {
      Utils::ceil_for_me(a_shape_[0], kernel_x_shape_[0]),
      Utils::ceil_for_me(a_shape_[1], kernel_x_shape_[1])};
  int64_t c_new_shape[2] = {a_new_shape[0], w_padded_shape_[1]};

  bfloat16 *a_ptr = (bfloat16 *)a;
  float *c_ptr = (float *)c;

  bfloat16 *a_compute = a_ptr;
  float *c_result = c_ptr;
  bool a_pad = false, c_pad = false;

  std::vector<bfloat16> a_copy(a_new_shape[0] * a_new_shape[1]);
  std::vector<float> c_copy(c_new_shape[0] * c_new_shape[1]);

  int64_t a_pad_start = GET_ELAPSED_TIME_NS();
  if ((a_new_shape[0] != a_shape_[0]) || (a_new_shape[1] != a_shape_[1])) {
    // padding is required for A
    memset(a_copy.data(), 0,
           sizeof(bfloat16) * a_new_shape[0] * a_new_shape[1]);
    Utils::_copy_pad_data<bfloat16>(a_ptr, a_copy.data(), &a_shape_[0],
                                    &a_new_shape[0]);
    a_compute = a_copy.data();
    a_pad = true;
  }
  int64_t a_pad_stop = GET_ELAPSED_TIME_NS();

  int64_t c_pad_start = GET_ELAPSED_TIME_NS();
  if ((c_new_shape[0] != c_shape_[0]) || (c_new_shape[1] != c_shape_[1])) {
    // padding is required for C
    memset(c_copy.data(), 0, sizeof(float) * c_new_shape[0] * c_new_shape[1]);
    c_result = c_copy.data();
    c_pad = true;
  }
  int64_t c_pad_stop = GET_ELAPSED_TIME_NS();

  // execute tiled matmul
  execute_tiled(a_compute, c_result, a_new_shape);

  int64_t cpu_depad_start = GET_ELAPSED_TIME_NS();
  if (c_pad) {
    Utils::_copy_depad_data<float>(c_result, c_ptr, &c_new_shape[0],
                                   &c_shape_[0]);
  }
  int64_t cpu_depad_stop = GET_ELAPSED_TIME_NS();

  int64_t exec_end = GET_ELAPSED_TIME_NS();
  int64_t a_pad_time = a_pad_stop - a_pad_start;
  int64_t c_pad_time = c_pad_stop - c_pad_start;
  int64_t cpu_depad_time = cpu_depad_stop - cpu_depad_start;

  RYZENAI_LOG_INFO(
      std::to_string(a_shape_[0]) + " " + std::to_string(a_shape_[1]) + " " +
      std::to_string(w_shape_[1]) + " " + std::to_string(kernel_x_shape_[0]) +
      " " + std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_pad_time) + " " + std::to_string(c_pad_time) + " " +
      std::to_string(cpu_depad_time) + " " + std::to_string(a_copy_time_) +
      " " + std::to_string(a_sync_time_) + " " + std::to_string(c_copy_time_) +
      " " + std::to_string(c_sync_time_) + " " + std::to_string(cpu_acc_time_) +
      " ");
}

void linear::execute_tiled(bfloat16 *a, float *c, int64_t *a_shape) {
  for (int64_t ra = 0; ra < a_shape[0]; ra += kernel_x_shape_[0]) {
    for (int64_t cb = 0; cb < w_padded_shape_[1]; cb += kernel_y_shape_[1]) {
      for (int64_t ca = 0; ca < a_shape[1]; ca += kernel_x_shape_[1]) {
        auto rb = ca;
        auto num_b_tiled_cols = w_padded_shape_[1] / kernel_y_shape_[1];
        auto cb_num = cb / kernel_y_shape_[1];
        auto rb_num = rb / kernel_y_shape_[0];
        auto b_tile_idx = cb_num + rb_num * num_b_tiled_cols;
        auto c_offset = ra * w_padded_shape_[1] + cb;
        auto a_offset = ra * a_shape[1] + ca;

        run_aie(a + a_offset, weights_bo_.data()[b_tile_idx], c_tmp_buf.data(),
                a_shape);

        int64_t cpu_accum_start = GET_ELAPSED_TIME_NS();
        _cpu_acc(c + c_offset, c_tmp_buf.data(), &kernel_z_shape_[0],
                 &w_padded_shape_[1]);
        int64_t cpu_accum_stop = GET_ELAPSED_TIME_NS();
        cpu_acc_time_ += cpu_accum_stop - cpu_accum_start;
      }
    }
  }
}

} // namespace ryzenai

#endif /* __LINEAR_H__ */
