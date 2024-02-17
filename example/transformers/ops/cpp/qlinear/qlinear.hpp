/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __QLINEAR_H__
#define __QLINEAR_H__

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>

#include <nlohmann/json.hpp>

// tvm headers
#include <dlpack/dlpack.h>
#include <tvm/runtime/contrib/aie_api.h>
#include <tvm/runtime/module.h>

// xrt headers
#include <maize/xrt_context.h>
#include <xrt/xrt_bo.h>

#include "logging.h"
#include "threadpool.h"
#include "utils.h"

#ifdef RYZENAI_LOGGING
#define PROFILE
#endif

#ifdef PROFILE
#define USE_TIMER(timer) timer
#else
#define USE_TIMER(timer)
#endif

using NDArray = tvm::runtime::NDArray;
using cache_key = int64_t;

using json = nlohmann::json;

namespace ryzenai {

template <typename InT, typename WtT, typename OutT> class qlinear_impl {
private:
  std::vector<tvm::runtime::PackedFunc> _set_input;
  std::vector<tvm::runtime::PackedFunc> _set_input_zc;
  std::vector<tvm::runtime::PackedFunc> _set_output_zc;
  std::vector<tvm::runtime::PackedFunc> _get_output;
  std::vector<tvm::runtime::PackedFunc> _tvm_run;
  tvm::runtime::PackedFunc cpu_set_input;
  tvm::runtime::PackedFunc cpu_get_output;
  tvm::runtime::PackedFunc cpu_tvm_run;
  DLDevice _dev;

  // Shape in terms of primitive kernel capability
  int64_t _kernel_x_shape[2];
  int64_t _kernel_y_shape[2];
  int64_t _kernel_z_shape[2];

  // Shape of padded weights
  int64_t _y_pad_shape[2];

  // Number of weight tiles in row & col direction.
  int64_t _y_tiles_count[2];

  // Shape of the actual computation
  int64_t _a_shape[2];
  int64_t _b_shape[2];
  int64_t _c_shape[2];
  void _run_aie2(void *a, void *b, void *c, int64_t *a_stride,
                 int64_t *b_stride, int64_t *c_stride, int tid);
  void _cpu_acc(OutT *acc, OutT *in, int64_t *tile_shape, int64_t *acc_stride);
  void _cpu_acc_bo(OutT *acc, Utils::AlignedBO *in, int64_t *tile_shape,
                   int64_t *acc_stride);
  void _reset_timers();
  void _reset_counters();
  void _execute_tiled_gemm_copy_opt(InT *a, OutT *c, int64_t *a_shape,
                                    int64_t *b_shape);
  void _execute_tiled_gemm_copy_opt_pipelined(InT *a, OutT *c, int64_t *a_shape,
                                              int64_t *b_shape);
  void _compute_C_partial_tile(InT *a, OutT *c, Utils::AlignedBO *a_tile,
                               Utils::AlignedBO *c_tile, int a_offset,
                               int b_offset, int c_offset, int64_t *xr,
                               int64_t *yr, int64_t *zr, int64_t *a_str,
                               int64_t *b_str, int64_t *c_str, cache_key key,
                               int worker_id);
  int64_t _aie_run_count;
  int64_t _a_tile_copy_count{0};
  int64_t _b_tile_copy_count{0};

  // Timing data;
#ifdef PROFILE
  int64_t _aie_total, _a_copy_total, _cpu_acc_total;
  int64_t _b_bo_create_total;
  int64_t _a_sync_total, _out_sync_total;
  int64_t _num_execute;
#endif
  std::string _log_file;

  static std::once_flag _logger_flag;
  static std::mutex _mtx;
  bool _enable_pipelining = false;
  size_t _nworkers = 1;

  // Buffer Management

  // XRT tile buffers. Currently, they act as ref to actual buffers allocated
  // statically.
  std::vector<Utils::AlignedBO> a_tile_bo;
  std::vector<Utils::AlignedBO> b_tile_bo;
  std::vector<Utils::AlignedBO> c_tile_bo;

  // Tiles cache for weights
  std::vector<Utils::AlignedBO> b_tile_cache;

  uint64_t _idx;
  static uint64_t qlinear_count;
  std::string _xclbin;

public:
  qlinear_impl(const std::string &shared_lib,
               const std::tuple<int, int> &x_shape,
               const std::tuple<int, int> &y_shape, size_t nworkers);
  qlinear_impl(const std::string &shared_lib,
               const std::tuple<int, int> &x_shape,
               const std::tuple<int, int> &y_shape, size_t nworkers,
               const std::string &log_file);
  ~qlinear_impl();
  void execute(InT *x, const std::tuple<int, int> &x_shape, OutT *y);
  void register_weights_cache(const std::vector<Utils::AlignedBO> &wts_cache,
                              std::tuple<int, int> b_shape);
};

template <typename InT, typename WtT, typename OutT>
uint64_t qlinear_impl<InT, WtT, OutT>::qlinear_count = 0;

template <typename InT, typename WtT, typename OutT>
std::mutex qlinear_impl<InT, WtT, OutT>::_mtx;

template <typename InT, typename WtT, typename OutT>
std::once_flag qlinear_impl<InT, WtT, OutT>::_logger_flag;

template <typename InT, typename WtT, typename OutT>
qlinear_impl<InT, WtT, OutT>::qlinear_impl(const std::string &shared_lib,
                                           const std::tuple<int, int> &x_shape,
                                           const std::tuple<int, int> &y_shape,
                                           size_t nworkers)
    : qlinear_impl(shared_lib, x_shape, y_shape, nworkers,
                   "log_qlinear_cpp_profile.csv") {}

template <typename InT, typename WtT, typename OutT>
qlinear_impl<InT, WtT, OutT>::qlinear_impl(const std::string &shared_lib,
                                           const std::tuple<int, int> &x_shape,
                                           const std::tuple<int, int> &y_shape,
                                           size_t nworkers,
                                           const std::string &log_file) {
  _idx = qlinear_count++;
  _dev = {kDLCPU, 0};
  _set_input.reserve(nworkers);
  _set_input_zc.reserve(nworkers);
  _get_output.reserve(nworkers);
  _tvm_run.reserve(nworkers);

  for (size_t i = 0; i < nworkers; ++i) {
    auto mod_factory = tvm::runtime::Module::LoadFromFile(shared_lib);
    auto config = tvm::runtime::aie::AieExecutorConfig(true);
    tvm::runtime::Module m_aie =
        mod_factory.GetFunction("default_with_config")(config, _dev);
    _set_input.push_back(m_aie.GetFunction("set_input"));
    _set_input_zc.push_back(m_aie.GetFunction("set_input_zero_copy"));
    _set_output_zc.push_back(m_aie.GetFunction("set_output_zero_copy"));
    _get_output.push_back(m_aie.GetFunction("get_output"));
    _tvm_run.push_back(m_aie.GetFunction("run"));
  }

  // Get XCLBIN path
  _xclbin = Utils::get_xclbin_path();

  _kernel_x_shape[0] = std::get<0>(x_shape);
  _kernel_x_shape[1] = std::get<1>(x_shape);
  _kernel_y_shape[0] = std::get<0>(y_shape);
  _kernel_y_shape[1] = std::get<1>(y_shape);
  _kernel_z_shape[0] = std::get<0>(x_shape);
  _kernel_z_shape[1] = std::get<1>(y_shape);

  auto [bo_group, newly_created] =
      Utils::AlignedBOCache::getInstance(shared_lib);

  if (newly_created) {
    bo_group->a_tile_bo.resize(nworkers);
    bo_group->b_tile_bo.resize(nworkers);
    bo_group->c_tile_bo.resize(nworkers);

    auto &xrt_context_ = amd::maize::XrtContext::GetInstance(&_xclbin);

    xrt_context_.GetOrCreateNewContext(0, &_xclbin);

    for (size_t i = 0; i < nworkers; ++i) {
      bo_group->a_tile_bo[i].bo = xrt::bo(
          xrt_context_.GetDevice(),
          _kernel_x_shape[0] * _kernel_x_shape[1] * sizeof(InT),
          xrt::bo::flags::host_only, xrt_context_.GetKernel(0).group_id(0));
      bo_group->b_tile_bo[i].bo = xrt::bo(
          xrt_context_.GetDevice(),
          _kernel_y_shape[0] * _kernel_y_shape[1] * sizeof(WtT),
          xrt::bo::flags::host_only, xrt_context_.GetKernel(0).group_id(0));
      bo_group->c_tile_bo[i].bo = xrt::bo(
          xrt_context_.GetDevice(),
          _kernel_z_shape[0] * _kernel_z_shape[1] * sizeof(OutT),
          xrt::bo::flags::host_only, xrt_context_.GetKernel(0).group_id(0));

      bo_group->a_tile_bo[i].bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_group->b_tile_bo[i].bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_group->c_tile_bo[i].bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      if (((uint64_t)(bo_group->a_tile_bo[i].data()) % BUFFER_ALIGNMENT != 0) ||
          ((uint64_t)(bo_group->b_tile_bo[i].data()) % BUFFER_ALIGNMENT != 0) ||
          ((uint64_t)(bo_group->c_tile_bo[i].data()) % BUFFER_ALIGNMENT != 0)) {
        throw std::runtime_error("Mapped host buffers from xrt::bo doesn't "
                                 "meet alignment reqs from TVM");
      }
    }
  }

  a_tile_bo.resize(nworkers);
  b_tile_bo.resize(nworkers);
  c_tile_bo.resize(nworkers);
  for (size_t i = 0; i < nworkers; ++i) {
    a_tile_bo[i].bo = bo_group->a_tile_bo[i].bo;
    b_tile_bo[i].bo = bo_group->b_tile_bo[i].bo;
    c_tile_bo[i].bo = bo_group->c_tile_bo[i].bo;

    // Caution : Initialize the Maize.runner via a dummy execute() call
    // Transactions are generated lazily only once at the first execute() call.
    // But transaction generation is not thread-safe currently.
    // So in pipeline mode, multiple threads invoke execute().
    // This causes multiple transaction generation in parallel causing random
    // crash
    _run_aie2(a_tile_bo[i].data(), b_tile_bo[i].data(), c_tile_bo[i].data(),
              /*a_stride*/ NULL, /*b_stride*/ NULL, /*c_stride*/ NULL,
              /*tid*/ (int)i);
  }

  _aie_run_count = 0;
  _enable_pipelining = nworkers > 1;
  if (_enable_pipelining) {
    auto &tps = ThreadPoolSingleton::getInstance(); // mere pre-init
    (void)tps;
  }
  _nworkers = nworkers;

  if (_nworkers > Utils::get_qlinear_num_threads()) {
    throw std::runtime_error(
        "Number of qlinear instance workers can't be more than the available "
        "workers.\n"
        "Increase number of available workers by setting qlinear_NUM_THREADS "
        "env variable to a higher value.\n"
        "Or decrease the number of workers of qlinear Instance.");
  }
  _log_file = log_file;
#ifdef PROFILE
  _aie_total = 0;
  _a_copy_total = 0;
  _cpu_acc_total = 0;
  _num_execute = 0;
  std::call_once(_logger_flag, []() {
    std::string header =
        "qlinear id, M,K,N,kernel_m,kernel_k,kernel_n,Execute "
        "time(us),Num Executes,Pad alloc time (ns), A Pad time (ns),B Pad time "
        "(ns),C "
        "Pad time (ns),B bo creation (ns), B tile copy(ns), B tile sync (ns), "
        "A tile copy (ns), "
        "A tile sync time (ns),AIE run time (ns), out sync time (ns), cpu "
        "accumulation time (ns),C Depad time (ns), Pad Dealloc time "
        "(ns),Number of "
        "AIE(tvm) runs,Average time per AIE(tvm) run(ns),A_tile copy "
        "count, B_tile copy count\n";
    RYZENAI_LOG_INFO(header);
  });

#endif
}

template <typename InT, typename WtT, typename OutT>
qlinear_impl<InT, WtT, OutT>::~qlinear_impl() {}

template <typename OutT>
static void test_for_acc_overflow(OutT *acc_base, int offset,
                                  Utils::AlignedBO *in_bo, int64_t *tile_shape,
                                  int64_t *acc_stride) {
  static const float max_value_fp32 =
      static_cast<float>(std::numeric_limits<OutT>::max());
  static const float min_value_fp32 =
      static_cast<float>(std::numeric_limits<OutT>::min());
  OutT *acc = acc_base + offset;
  xrt::bo &bo = in_bo->bo;
  OutT *in = bo.map<OutT *>();

  for (int i = 0; i < tile_shape[0]; i++) {
    for (int j = 0; j < tile_shape[1]; j++) {
      auto af = static_cast<float>(acc[i * acc_stride[0] + j]);
      auto bf = static_cast<float>(in[i * tile_shape[1] + j]);
      auto cf = af + bf;
      if (cf > max_value_fp32) {
        int loc_i = (offset / acc_stride[0]) + i;
        int loc_j = offset % acc_stride[0] + j;
        std::cout << "Overflow detected @ C[" << loc_i << ", " << loc_j
                  << "] : " << af << "+" << bf << "=" << cf << " > "
                  << max_value_fp32 << std::endl;
      }

      if (cf < min_value_fp32) {
        int loc_i = (offset / acc_stride[0]) + i;
        int loc_j = offset % acc_stride[0] + j;
        std::cout << "Underflow detected @ C[" << loc_i << ", " << loc_j
                  << "] : " << af << "+" << bf << "=" << cf << " < "
                  << min_value_fp32 << std::endl;
      }
    }
  }
}

template <typename InT, typename WtT, typename OutT>
void qlinear_impl<InT, WtT, OutT>::_reset_timers() {
#ifdef PROFILE
  _aie_total = 0;
  _a_copy_total = 0;
  _cpu_acc_total = 0;
  _b_bo_create_total = 0;
  _a_sync_total = 0;
  _out_sync_total = 0;
#endif
}

template <typename InT, typename WtT, typename OutT>
void qlinear_impl<InT, WtT, OutT>::_reset_counters() {
  _aie_run_count = 0;
  _a_tile_copy_count = 0;
  _b_tile_copy_count = 0;
}

// DLTensor can take explicit stride to do strided data access.
// If stride=NULL, it means data is compact row-major format.
// Currently, qlinear doesn't support strided data.
// So _run_aie2 is always called with NULL for strides.
template <typename InT, typename WtT, typename OutT>
void qlinear_impl<InT, WtT, OutT>::_run_aie2(void *a, void *b, void *c,
                                             int64_t *a_stride,
                                             int64_t *b_stride,
                                             int64_t *c_stride, int tid) {
  DLTensor a_dl = {a,
                   _dev,
                   2,
                   DLDataType{kDLInt, 8, 1},
                   (int64_t *)&_kernel_x_shape[0],
                   &a_stride[0],
                   0};
  DLTensor b_dl = {b,
                   _dev,
                   2,
                   DLDataType{kDLInt, 8, 1},
                   (int64_t *)&_kernel_y_shape[0],
                   &b_stride[0],
                   0};
  DLTensor c_dl = {
      c,    _dev, 2, DLDataType{kDLInt, 32, 1}, (int64_t *)&_kernel_z_shape[0],
      NULL, 0};
  DLManagedTensor a_dlmt = {a_dl, NULL, NULL};
  DLManagedTensor b_dlmt = {b_dl, NULL, NULL};
  DLManagedTensor c_dlmt = {c_dl, NULL, NULL};
  auto a_nd = NDArray::FromDLPack(&a_dlmt);
  auto b_nd = NDArray::FromDLPack(&b_dlmt);
  auto c_nd = NDArray::FromDLPack(&c_dlmt);

  _set_input_zc[tid]("x", (a_nd));
  _set_input_zc[tid]("y", (b_nd));
  _set_output_zc[tid](0, c_nd);
  _tvm_run[tid]();
}

template <typename InT, typename WtT, typename OutT>
void qlinear_impl<InT, WtT, OutT>::_cpu_acc(OutT *acc, OutT *in,
                                            int64_t *tile_shape,
                                            int64_t *acc_stride) {
  for (int i = 0; i < tile_shape[0]; i++) {
    for (int j = 0; j < tile_shape[1]; j++) {
      acc[i * acc_stride[0] + j] += in[i * tile_shape[1] + j];
    }
  }
}

template <typename InT, typename WtT, typename OutT>
void qlinear_impl<InT, WtT, OutT>::_cpu_acc_bo(OutT *acc,
                                               Utils::AlignedBO *in_bo,
                                               int64_t *tile_shape,
                                               int64_t *acc_stride) {
  xrt::bo &bo = in_bo->bo;
  OutT *in = bo.map<OutT *>();
  _cpu_acc(acc, in, tile_shape, acc_stride);
}

template <typename InT, typename WtT, typename OutT>
void qlinear_impl<InT, WtT, OutT>::_compute_C_partial_tile(
    InT *a, OutT *c, Utils::AlignedBO *a_tile, Utils::AlignedBO *c_tile,
    int a_offset, int b_offset, int c_offset, int64_t *xr, int64_t *yr,
    int64_t *zr, int64_t *a_str, int64_t *b_str, int64_t *c_str, cache_key key,
    int worker_id) {
  Utils::_copy_tile<InT>(a + a_offset, a_tile, xr, a_str);
  a_tile->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  void *aie_a_ptr = a_tile;

  auto &cached_b_tile = b_tile_cache.at(key);
  void *aie_b_ptr = cached_b_tile.data();

  _run_aie2(aie_a_ptr, aie_b_ptr, c_tile, /*a_stride*/ NULL, /*b_stride*/ NULL,
            /*c_stride*/ NULL, worker_id);

  std::unique_lock<std::mutex> lk(_mtx);
  c_tile->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  _cpu_acc_bo(c + c_offset, c_tile, zr, c_str);
}

template <typename InT, typename WtT, typename OutT>
inline void qlinear_impl<InT, WtT, OutT>::_execute_tiled_gemm_copy_opt(
    InT *a, OutT *c, int64_t *a_shape, int64_t *b_shape) {

  void *aie_a_ptr, *aie_b_ptr;
  for (int64_t ra = 0; ra < a_shape[0]; ra += _kernel_x_shape[0]) {
    for (int64_t cb = 0; cb < b_shape[1]; cb += _kernel_y_shape[1]) {
      for (int64_t ca = 0; ca < a_shape[1]; ca += _kernel_x_shape[1]) {
        auto rb = ca;
        auto b_offset = cb + rb * b_shape[1];
        auto c_offset = ra * b_shape[1] + cb;
        auto a_offset = ra * a_shape[1] + ca;
        cache_key key = (rb / _kernel_y_shape[0]) +
                        (cb / _kernel_y_shape[1]) * _y_tiles_count[0];
        // std::cout << "Extraction : " << key << std::endl;
        USE_TIMER(auto b_bo_create_start = GET_ELAPSED_TIME_NS());
        auto &cached_b_tile = b_tile_cache.at(key);
        aie_b_ptr = cached_b_tile.data();
        assert(!newly_created);
        USE_TIMER(auto b_bo_create_stop = GET_ELAPSED_TIME_NS());
        USE_TIMER(_b_bo_create_total += (b_bo_create_stop - b_bo_create_start));

        USE_TIMER(auto a_copy_start = GET_ELAPSED_TIME_NS());
        Utils::_copy_tile<InT>(a + a_offset, a_tile_bo[0].data(),
                               &_kernel_x_shape[0], &a_shape[1]);
        USE_TIMER(auto a_copy_stop = GET_ELAPSED_TIME_NS());
        USE_TIMER(_a_copy_total += (a_copy_stop - a_copy_start));
        USE_TIMER(_a_tile_copy_count++);

        USE_TIMER(auto a_sync_start = GET_ELAPSED_TIME_NS());
        a_tile_bo[0].bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        aie_a_ptr = a_tile_bo[0].data();
        USE_TIMER(auto a_sync_stop = GET_ELAPSED_TIME_NS());
        USE_TIMER(_a_sync_total += (a_sync_stop - a_sync_start));

        USE_TIMER(auto aie_start = GET_ELAPSED_TIME_NS());
        _run_aie2(aie_a_ptr, aie_b_ptr, c_tile_bo[0].data(), /*a_stride*/ NULL,
                  /*b_stride*/ NULL, /*c_stride*/ NULL, /*tid*/ 0);
        USE_TIMER(auto aie_stop = GET_ELAPSED_TIME_NS());
        USE_TIMER(_aie_total += (aie_stop - aie_start));

        a_tile_bo[0].need_sync = false;
        cached_b_tile.need_sync = false;

        USE_TIMER(auto out_sync_start = GET_ELAPSED_TIME_NS());
        c_tile_bo[0].bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        USE_TIMER(auto out_sync_stop = GET_ELAPSED_TIME_NS());
        USE_TIMER(_out_sync_total += (out_sync_stop - out_sync_start));

        USE_TIMER(auto cpu_acc_start = GET_ELAPSED_TIME_NS());
        // test_for_acc_overflow(c, c_offset, c_tile_bo[0].data(),
        // &_kernel_z_shape[0], &b_shape[1]);
        _cpu_acc_bo(c + c_offset, c_tile_bo[0].data(), &_kernel_z_shape[0],
                    &b_shape[1]);
        USE_TIMER(auto cpu_acc_stop = GET_ELAPSED_TIME_NS());
        USE_TIMER(_cpu_acc_total += (cpu_acc_stop - cpu_acc_start));

        _aie_run_count++;
      }
    }
  }
}

template <typename InT, typename WtT, typename OutT>
void qlinear_impl<InT, WtT, OutT>::_execute_tiled_gemm_copy_opt_pipelined(
    InT *a, OutT *c, int64_t *a_shape, int64_t *b_shape) {

  std::vector<std::future<void>> futs;
  int ntiles = (int)((a_shape[0] / _kernel_x_shape[0]) *
                     (b_shape[1] / _kernel_y_shape[1]) *
                     (a_shape[1] / _kernel_x_shape[1]));
  futs.reserve(ntiles);
  auto &Pool = ThreadPoolSingleton::getInstance().pool;
  for (int64_t ra = 0; ra < a_shape[0]; ra += _kernel_x_shape[0]) {
    for (int64_t cb = 0; cb < b_shape[1]; cb += _kernel_y_shape[1]) {
      for (int64_t ca = 0; ca < a_shape[1]; ca += _kernel_x_shape[1]) {
        auto rb = ca;
        auto a_offset = ra * a_shape[1] + ca;
        auto b_offset = cb + rb * b_shape[1];
        auto c_offset = ra * b_shape[1] + cb;
        cache_key key = (rb / _kernel_y_shape[0]) +
                        (cb / _kernel_y_shape[1]) * _y_tiles_count[0];
        // std::cout << "Extraction key : " << key << std::endl;

        int tid = _aie_run_count % _nworkers;
        // TODO : Reduce the args.
        auto ret =
            Pool.enqueue(tid, [&, a_offset, b_offset, c_offset, tid, key]() {
              _compute_C_partial_tile(
                  a, c, a_tile_bo[tid].data(), c_tile_bo[tid].data(),
                  (int)a_offset, (int)b_offset, (int)c_offset,
                  &_kernel_x_shape[0], &_kernel_y_shape[0], &_kernel_z_shape[0],
                  &a_shape[1], &b_shape[1], &b_shape[1], key, tid);
            });

        futs.push_back(std::move(ret));
        _aie_run_count++;
      }
    }
  }
  for (auto &fut : futs) {
    fut.get();
  }
}

static int64_t ceil_for_me(int64_t x, int64_t y) {
  return static_cast<int64_t>(y * std::ceil(x * 1.0 / y));
}

template <typename InT, typename WtT, typename OutT>
void qlinear_impl<InT, WtT, OutT>::execute(InT *a,
                                           const std::tuple<int, int> &a_shape,
                                           OutT *c) {
  USE_TIMER(_reset_timers());
  USE_TIMER(_reset_counters());
  USE_TIMER(auto exec_start = GET_ELAPSED_TIME_NS());

  _a_shape[0] = std::get<0>(a_shape);
  _a_shape[1] = std::get<1>(a_shape);

  _c_shape[0] = std::get<0>(a_shape);
  _c_shape[1] = _b_shape[1];

  int64_t a_new_shape[2] = {ceil_for_me(_a_shape[0], _kernel_x_shape[0]),
                            ceil_for_me(_a_shape[1], _kernel_x_shape[1])};
  int64_t b_new_shape[2] = {_y_pad_shape[0], _y_pad_shape[1]};
  int64_t c_new_shape[2] = {a_new_shape[0], b_new_shape[1]};

  auto a_ptr = a;
  auto c_ptr = c;
  InT *a_compute = a_ptr;
  OutT *c_result = c_ptr;

  OutT *c_copy;
  InT *a_copy;

  bool a_pad = false, b_pad = false, c_pad = true;
  // stride in numpy is represented in number of bytes. But DLTensor expects
  // this to be in number of elements.
  int64_t a_new_stride[2] = {a_new_shape[1], a_new_shape[0]};
  int64_t b_new_stride[2] = {b_new_shape[1], b_new_shape[0]};
  int64_t c_new_stride[2] = {c_new_shape[1], c_new_shape[0]};

  // PAD for all buffers
  USE_TIMER(auto pad_alloc_start = GET_ELAPSED_TIME_NS());
  a_copy = new InT[a_new_shape[0] * a_new_shape[1]];
  c_copy = new OutT[c_new_shape[0] * c_new_shape[1]];
  USE_TIMER(auto pad_alloc_stop = GET_ELAPSED_TIME_NS());

  USE_TIMER(auto a_pad_start = GET_ELAPSED_TIME_NS());
  if ((a_new_shape[0] != _a_shape[0]) || (a_new_shape[1] != _a_shape[1])) {
    // padding is required for A
    memset(a_copy, 0, sizeof(InT) * a_new_shape[0] * a_new_shape[1]);
    Utils::_copy_pad_data<InT>(a_ptr, a_copy, &_a_shape[0], &a_new_shape[0]);
    a_compute = a_copy;
    a_pad = true;
  }
  USE_TIMER(auto a_pad_stop = GET_ELAPSED_TIME_NS());

  USE_TIMER(auto c_pad_start = GET_ELAPSED_TIME_NS());
  if ((c_new_shape[0] != _c_shape[0]) || (c_new_shape[1] != _c_shape[1])) {
    // padding is required for C
    memset(c_copy, 0, sizeof(OutT) * c_new_shape[0] * c_new_shape[1]);
    c_result = c_copy;
    c_pad = true;
  }
  USE_TIMER(auto c_pad_stop = GET_ELAPSED_TIME_NS());

  if (_enable_pipelining)
    _execute_tiled_gemm_copy_opt_pipelined(a_compute, c_result, &a_new_shape[0],
                                           &b_new_shape[0]);
  else
    _execute_tiled_gemm_copy_opt(a_compute, c_result, &a_new_shape[0],
                                 &b_new_shape[0]);

  USE_TIMER(auto c_depad_start = GET_ELAPSED_TIME_NS());
  if (c_pad) {
    Utils::_copy_depad_data<OutT>(c_result, c_ptr, &c_new_shape[0],
                                  &_c_shape[0]);
  }
  USE_TIMER(auto c_depad_stop = GET_ELAPSED_TIME_NS());

  USE_TIMER(auto pad_dealloc_start = GET_ELAPSED_TIME_NS());
  delete[] a_copy;
  delete[] c_copy;
  USE_TIMER(auto pad_dealloc_stop = GET_ELAPSED_TIME_NS());

  USE_TIMER(auto exec_stop = GET_ELAPSED_TIME_NS());
  USE_TIMER(_num_execute++);
  USE_TIMER(auto a_pad_total = (a_pad_stop - a_pad_start));
  USE_TIMER(auto c_pad_total = (c_pad_stop - c_pad_start));
  USE_TIMER(auto c_depad_total = (c_depad_stop - c_depad_start));
  USE_TIMER(auto pad_alloc = (pad_alloc_stop - pad_alloc_stop));
  USE_TIMER(auto pad_dealloc = (pad_dealloc_stop - pad_dealloc_stop));
  USE_TIMER([=]() {
    std::stringstream csv_out;
    csv_out << _idx << ",";
    csv_out << _a_shape[0] << "," << _a_shape[1] << "," << _b_shape[1] << ",";
    csv_out << _kernel_x_shape[0] << "," << _kernel_x_shape[1] << ","
            << _kernel_y_shape[1] << ",";
    csv_out << (exec_stop - exec_start) << ",";
    csv_out << _num_execute << ",";
    csv_out << pad_alloc << ",";
    csv_out << a_pad_total << ",";
    csv_out << /*b_pad_total*/ 0 << ",";
    csv_out << c_pad_total << ",";
    csv_out << (_b_bo_create_total) << ",";
    csv_out << /*b_copy_total*/ 0 << ",";
    csv_out << /*_b_sync_total*/ 0 << ",";
    csv_out << (_a_copy_total) << ",";
    csv_out << (_a_sync_total) << ",";
    csv_out << (_aie_total) << ",";
    csv_out << (_out_sync_total) << ",";
    csv_out << (_cpu_acc_total) << ",";
    csv_out << (c_depad_total) << ",";
    csv_out << pad_dealloc << ",";
    csv_out << _aie_run_count << ",";
    csv_out << (double)_aie_total / _aie_run_count << ", ";
    csv_out << _a_tile_copy_count << ", " << _b_tile_copy_count << "\n";
    RYZENAI_LOG_INFO(csv_out.str());
  }());
}

template <typename InT, typename WtT, typename OutT>
void qlinear_impl<InT, WtT, OutT>::register_weights_cache(
    const std::vector<Utils::AlignedBO> &wts_cache,
    std::tuple<int, int> b_shape) {
  b_tile_cache = wts_cache;
  _b_shape[0] = std::get<0>(b_shape);
  _b_shape[1] = std::get<1>(b_shape);

  _y_pad_shape[0] = ceil_for_me(_b_shape[0], _kernel_y_shape[0]);
  _y_pad_shape[1] = ceil_for_me(_b_shape[1], _kernel_y_shape[1]);

  _y_tiles_count[0] = _y_pad_shape[0] / _kernel_y_shape[0];
  _y_tiles_count[1] = _y_pad_shape[1] / _kernel_y_shape[1];
}

template <typename InT, typename WtT, typename OutT> class qlinear {
public:
  qlinear(const std::vector<std::string> &shared_libs,
          const std::vector<std::tuple<int, int>> &x_shapes,
          const std::tuple<int, int> &y_shape, size_t nworkers, int num_dlls,
          int dll_switch_limit, bool pack_weights);
  qlinear(const std::vector<std::string> &shared_libs,
          const std::vector<std::tuple<int, int>> &x_shapes,
          const std::tuple<int, int> &y_shape, size_t nworkers, int num_dlls,
          int dll_switch_limit, bool pack_weights, const std::string &log_file);

  // Legacy constructors for bwd compatibility.
  // num_dlls = 1, limit = 4 by default
  qlinear(const std::string &shared_lib, const std::tuple<int, int> &x_shape,
          const std::tuple<int, int> &y_shape, size_t nworkers);
  qlinear(const std::string &shared_lib, const std::tuple<int, int> &x_shape,
          const std::tuple<int, int> &y_shape, size_t nworkers,
          const std::string &log_file);

  ~qlinear() = default;
  void execute(InT *x, const std::tuple<int, int> &x_shape, OutT *y);
  void initialize_weights(WtT *weights, std::tuple<int, int> &wt_shape);
  void initialize_weights(const std::string &weights_json);

private:
  std::vector<WtT> _pad_weights(WtT *weights, int64_t b_shape[2],
                                int64_t y_pad_shape[2]);
  void _tile_weights(const std::vector<WtT> &padded_weights,
                     int64_t y_pad_shape[2], int64_t y_tiles_count[2]);
  void _copy_tile_packed(const WtT *src, Utils::AlignedBO *dst,
                         int64_t *tile_shape, int64_t *src_strides);
  void _pack_tile(const WtT *src, WtT *dest, int64_t *tile_shape,
                  int64_t *stride);
  void _parse_overlay_properties();

  std::vector<qlinear_impl<InT, WtT, OutT>>
      qlinear_objs_; // maintain multiple qlinear objs based on num_dlls.
  int num_dlls_{0};
  int dll_switch_limit_{0};
  int64_t kernel_y_shape_[2];
  bool pack_weights_;
  // Block size for weights packing
  int64_t _block_shape[2];
  // number of aie cores on which matmul wil be executed
  int64_t _num_cores;
  std::string _overlay_prop;

  std::vector<Utils::AlignedBO> b_tile_cache;

  inline static double weights_size_ = 0;
  inline static double padded_weights_size_ = 0;
};

template <typename InT, typename WtT, typename OutT>
qlinear<InT, WtT, OutT>::qlinear(
    const std::vector<std::string> &shared_libs,
    const std::vector<std::tuple<int, int>> &x_shapes,
    const std::tuple<int, int> &y_shape, size_t nworkers, int num_dlls,
    int dll_switch_limit, bool pack_weights, const std::string &log_file)
    : num_dlls_(num_dlls), dll_switch_limit_(dll_switch_limit),
      pack_weights_(pack_weights) {
  if (shared_libs.empty()) {
    throw std::runtime_error("ERROR : Now tvm dlls passed");
  }
  if ((shared_libs.size() != x_shapes.size()) ||
      (shared_libs.size() != num_dlls_)) {
    throw std::runtime_error(
        "ERROR : # shared dlls, # x_shapes & num_dlls should match");
  }

  kernel_y_shape_[0] = std::get<0>(y_shape);
  kernel_y_shape_[1] = std::get<1>(y_shape);
  _parse_overlay_properties();

  qlinear_objs_.reserve(num_dlls);
  for (int i = 0; i < num_dlls_; ++i) {
    qlinear_objs_.emplace_back(shared_libs.at(i), x_shapes.at(i), y_shape,
                               nworkers, log_file);
  }
}

template <typename InT, typename WtT, typename OutT>
qlinear<InT, WtT, OutT>::qlinear(
    const std::vector<std::string> &shared_libs,
    const std::vector<std::tuple<int, int>> &x_shapes,
    const std::tuple<int, int> &y_shape, size_t nworkers, int num_dlls,
    int dll_switch_limit, bool pack_weights)
    : qlinear(shared_libs, x_shapes, y_shape, nworkers, num_dlls,
              dll_switch_limit, pack_weights, "log_qlinear_cpp_profile.csv") {}

template <typename InT, typename WtT, typename OutT>
qlinear<InT, WtT, OutT>::qlinear(const std::string &shared_lib,
                                 const std::tuple<int, int> &x_shape,
                                 const std::tuple<int, int> &y_shape,
                                 size_t nworkers, const std::string &log_file)
    : qlinear({shared_lib}, {x_shape}, y_shape, nworkers, /*num_dlls*/ 1,
              /*dll_switch_limit*/ 4, true, log_file) {}

template <typename InT, typename WtT, typename OutT>
qlinear<InT, WtT, OutT>::qlinear(const std::string &shared_lib,
                                 const std::tuple<int, int> &x_shape,
                                 const std::tuple<int, int> &y_shape,
                                 size_t nworkers)
    : qlinear(shared_lib, x_shape, y_shape, nworkers,
              "log_qlinear_cpp_profile.csv") {}

template <typename InT, typename WtT, typename OutT>
void qlinear<InT, WtT, OutT>::execute(InT *x,
                                      const std::tuple<int, int> &x_shape,
                                      OutT *y) {
  const int M = std::get<0>(x_shape);
  if (M > dll_switch_limit_ && num_dlls_ == 2) {
    // std::cout << "Qlinear execute : Using second dll" << std::endl;
    qlinear_objs_.at(1).execute(x, x_shape, y);
    return;
  }

  // std::cout << "Qlinear execute : Using first dll" << std::endl;
  qlinear_objs_.at(0).execute(x, x_shape, y);
}

template <typename InT, typename WtT, typename OutT>
void qlinear<InT, WtT, OutT>::initialize_weights(
    WtT *weights, std::tuple<int, int> &wt_shape) {
  int64_t b_shape[2];
  int64_t y_pad_shape[2];
  int64_t y_tiles_count[2];

  b_shape[0] = std::get<0>(wt_shape);
  b_shape[1] = std::get<1>(wt_shape);

  y_pad_shape[0] = ceil_for_me(b_shape[0], kernel_y_shape_[0]);
  y_pad_shape[1] = ceil_for_me(b_shape[1], kernel_y_shape_[1]);

  y_tiles_count[0] = y_pad_shape[0] / kernel_y_shape_[0];
  y_tiles_count[1] = y_pad_shape[1] / kernel_y_shape_[1];

  std::vector<WtT> padded_weights = _pad_weights(weights, b_shape, y_pad_shape);
  _tile_weights(padded_weights, y_pad_shape, y_tiles_count);

  for (auto &qlinear_obj : qlinear_objs_) {
    qlinear_obj.register_weights_cache(b_tile_cache, wt_shape);
  }

  const double gb = (1 << 30);
  weights_size_ += (b_shape[0] * b_shape[1]) / gb;
  padded_weights_size_ += (y_pad_shape[0] * y_pad_shape[1]) / gb;

  // std::cout << "weights shape : " << b_shape[0] << "x" << b_shape[1] <<
  // std::endl; std::cout << "padded weights shape : " << y_pad_shape[0] << "x"
  // << y_pad_shape[1] << std::endl; std::cout << "Total weights size (GB) : "
  // << weights_size_ << std::endl; std::cout << "Total padded weights size (GB)
  // : " << padded_weights_size_ << std::endl;
}

template <typename InT, typename WtT, typename OutT>
void qlinear<InT, WtT, OutT>::initialize_weights(
    const std::string &weights_json) {
  throw std::runtime_error(
      "No impl available for qlinear::initialize_weights(const std::string "
      "&weights_json)");
}

template <typename InT, typename WtT, typename OutT>
std::vector<WtT> qlinear<InT, WtT, OutT>::_pad_weights(WtT *weights,
                                                       int64_t b_shape[2],
                                                       int64_t y_pad_shape[2]) {
  std::vector<WtT> padded_weights;
  if ((y_pad_shape[0] != b_shape[0]) || (y_pad_shape[1] != b_shape[1])) {
    padded_weights = std::vector<WtT>(y_pad_shape[0] * y_pad_shape[1], 0);
    Utils::_copy_pad_data<InT>(weights, padded_weights.data(), &b_shape[0],
                               &y_pad_shape[0]);
  } else {
    int64_t nelems = y_pad_shape[0] * y_pad_shape[1];
    padded_weights = std::vector<WtT>(weights, weights + nelems);
  }
  return padded_weights;
}

template <typename InT, typename WtT, typename OutT>
void qlinear<InT, WtT, OutT>::_pack_tile(const WtT *src, WtT *dest,
                                         int64_t *tile_shape, int64_t *stride) {
  int64_t num_tiles_k = tile_shape[0] / _block_shape[0];

  auto dest_idx = 0;
  for (int64_t i = 0; i < _num_cores; i++) {
    for (int64_t t = 0; t < tile_shape[1] / _block_shape[1]; t += _num_cores) {
      for (int64_t k = 0; k < num_tiles_k; k++) {
        auto src_offset = i * _block_shape[1] + t * _block_shape[0] +
                          k * _block_shape[0] * stride[0];
        for (int64_t ii = 0; ii < _block_shape[0]; ii++) {
          for (int64_t jj = 0; jj < _block_shape[1]; jj++) {
            auto src_idx = src_offset + ii * stride[0] + jj;
            dest[dest_idx] = src[src_idx];
            dest_idx++;
          }
        }
      }
    }
  }
}

template <typename InT, typename WtT, typename OutT>
void qlinear<InT, WtT, OutT>::_copy_tile_packed(const WtT *src,
                                                Utils::AlignedBO *dst,
                                                int64_t *tile_shape,
                                                int64_t *src_strides) {
  WtT *dest = dst->bo.map<WtT *>();
  _pack_tile(src, dest, tile_shape, src_strides);
  dst->need_sync = true;
}

// The padded weight matrix is split into tiles and stored in a vector in
// col-row order. 1  |  2  |  3
// ---+-----+-----
// 4  |  5  |  6
// These tiles are stored in the cache in this order :  {1, 4, 2, 5, 3, 6}
// Why ? Because in MNK loop-structure, weight tiles are accessed in col-major
// order
template <typename InT, typename WtT, typename OutT>
void qlinear<InT, WtT, OutT>::_tile_weights(
    const std::vector<WtT> &padded_weights, int64_t y_pad_shape[2],
    int64_t y_tiles_count[2]) {
  std::string _xclbin = Utils::get_xclbin_path();
  b_tile_cache.clear();
  b_tile_cache.resize(y_tiles_count[0] * y_tiles_count[1]);
  auto &xrt_context_ = amd::maize::XrtContext::GetInstance(&_xclbin);
  xrt_context_.GetOrCreateNewContext(0, &_xclbin);
  for (int j = 0; j < y_tiles_count[1]; ++j) {
    for (int i = 0; i < y_tiles_count[0]; ++i) {
      int rb = (int)(i * kernel_y_shape_[0]);
      int cb = (int)(j * kernel_y_shape_[1]);
      auto b_offset = cb + rb * y_pad_shape[1];
      cache_key key = j * y_tiles_count[0] + i;
      // std::cout << "Filling : " << key << std::endl;

      Utils::AlignedBO &cached_b_tile = b_tile_cache.at(key);
      cached_b_tile.bo = xrt::bo(
          xrt_context_.GetDevice(),
          kernel_y_shape_[0] * kernel_y_shape_[1] * sizeof(WtT),
          xrt::bo::flags::host_only, xrt_context_.GetKernel(0).group_id(0));
      if (pack_weights_) {
        _copy_tile_packed(padded_weights.data() + b_offset,
                          cached_b_tile.data(), &kernel_y_shape_[0],
                          &y_pad_shape[1]);
      } else {
        Utils::_copy_tile<InT>(padded_weights.data() + b_offset,
                               cached_b_tile.data(), &kernel_y_shape_[0],
                               &y_pad_shape[1]);
      }
      cached_b_tile.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      cached_b_tile.need_sync = false;
    }
  }
}

template <typename InT, typename WtT, typename OutT>
void qlinear<InT, WtT, OutT>::_parse_overlay_properties() {
  // Get overlay properties
  _overlay_prop = Utils::get_overlay_props();

  std::ifstream f(_overlay_prop);
  json _prop = json::parse(f);

  auto ops = _prop.find("OPS");
  auto gemm_prop = ops->find("gemm");
  auto gemm_attr = gemm_prop->find("ATTRS");
  if (gemm_attr == gemm_prop->end()) {
    throw std::runtime_error("Could not find gemm attributes");
  }

  _num_cores = (int)gemm_attr.value()["PN"] * (int)gemm_attr.value()["QN"];
  auto tn = gemm_attr.value()["TN"];
  auto n0 = gemm_attr.value()["N0"];
  auto tk = gemm_attr.value()["TK"];
  auto k0 = gemm_attr.value()["K0"];

  _block_shape[0] = (int)tk * (int)k0;
  _block_shape[1] = (int)tn * (int)n0;
}

} // namespace ryzenai

#endif /* __QLIENAR_H__ */

/*
    Notes for Devs
    --------------------
    1.  In eager mode execution, only one linear node is supposed to be executed
   at any point in time. 1.a.    So, the tile buffers for A, B, C are kept
   static so that they are shared between multiple linear objects. 1.b.    Shape
   of A & C tile buffers will change based on dll used. So for each dll,
   separate sets of BOs are created. 1.c.    There is also separate sets of tile
   BOs for each worker thread in pipeline mode.
    2.  Transaction Generation is not thread-safe in underlying stack. So it is
   generated sequentially by a dummy _run_aie2 call in ctr. Please see the note
   in ctr for more details 3.
*/
