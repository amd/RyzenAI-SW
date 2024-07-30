/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#pragma once

#ifdef _WIN32
#ifdef DYNAMIC_DISPATCH_BUILD_SHARED
#ifdef DYNAMIC_DISPATCH_EXPORT
#define DYNAMIC_DISPATCH_API __declspec(dllexport)
#else
#define DYNAMIC_DISPATCH_API __declspec(dllimport)
#endif
#endif
#endif

#ifdef __GNUC__
#ifdef DYNAMIC_DISPATCH_BUILD_SHARED
#ifdef DYNAMIC_DISPATCH_EXPORT
#define DYNAMIC_DISPATCH_API __attribute__((visibility("default")))
#else
#define DYNAMIC_DISPATCH_API
#endif
#endif
#endif

#ifndef DYNAMIC_DISPATCH_API
#define DYNAMIC_DISPATCH_API
#endif

#include <memory>
#include <mutex>
#include <unordered_map>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <utils/logging.hpp>

// dpu kernel metadata
constexpr auto NPU_KERNEL_NAME = "DPU";

namespace ryzenai {
namespace dynamic_dispatch {
class xrt_context {
private:
  static DYNAMIC_DISPATCH_API
      std::unordered_map<std::string, std::shared_ptr<xrt_context>>
          ctx_map_;
  static DYNAMIC_DISPATCH_API std::mutex xrt_ctx_mutex_;
  xrt::device device_;
  xrt::xclbin xclbin_;
  xrt::hw_context context_;
  xrt::kernel kernel_;
  xrt_context(const std::string &xclbin_fname) {
    unsigned int device_index = 0;
    device_ = xrt::device(device_index);
    xclbin_ = xrt::xclbin(xclbin_fname);
    device_.register_xclbin(xclbin_);
    RYZENAI_LOG_TRACE("Creating new context with xclbin: " + xclbin_fname);
    context_ = xrt::hw_context(device_, xclbin_.get_uuid());
    kernel_ = xrt::kernel(context_, NPU_KERNEL_NAME);
  }

public:
  xrt_context() {}
  static std::shared_ptr<xrt_context> get_instance(const std::string &xclbin) {
    RYZENAI_LOG_TRACE("Getting context with xclbin: " + xclbin);
    std::lock_guard<std::mutex> guard(xrt_ctx_mutex_);
    if (ctx_map_.find(xclbin) != ctx_map_.end()) {
      RYZENAI_LOG_TRACE("Context found in map");
      auto ctx = ctx_map_[xclbin];
      return ctx;
    } else {
      RYZENAI_LOG_TRACE("Context not found in map, creating new one");
      auto ctx = std::make_shared<xrt_context>(xrt_context(xclbin));
      ctx_map_[xclbin] = ctx;
      return ctx;
    }
  }

  xrt::device &get_device() { return device_; }
  xrt::hw_context &get_context() { return context_; }
  xrt::kernel &get_kernel() { return kernel_; }
  xrt::xclbin &get_xclbin() { return xclbin_; }
};

} // namespace dynamic_dispatch

} // namespace ryzenai
