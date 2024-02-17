/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __XRT_CONTEXT_H__
#define __XRT_CONTEXT_H__

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include "dpu_kernel_metadata.hpp"

namespace ryzenai {
class xrt_context {
protected:
  xrt::device device_;
  xrt::xclbin xclbin_;
  xrt::kernel kernel_;

protected:
  xrt_context(const std::string &xclbin_fname) {
    unsigned int device_index = 0;
    device_ = xrt::device(device_index);
    xclbin_ = xrt::xclbin(xclbin_fname);

    device_.register_xclbin(xclbin_);
    xrt::hw_context context(device_, xclbin_.get_uuid());
    kernel_ = xrt::kernel(context, KERNEL_NAME);
  }

public:
  static xrt_context &get_instance(const std::string &xclbin) {
    static xrt_context ctx_(xclbin);
    return ctx_;
  }

  xrt_context(const xrt_context &) = delete;
  xrt_context(const xrt_context &&) = delete;
  xrt_context &operator=(const xrt_context &) = delete;
  xrt_context &operator=(const xrt_context &&) = delete;

  xrt::device &get_device() { return device_; }

  xrt::kernel &get_kernel() { return kernel_; }
};

} // namespace ryzenai

#endif /* __XRT_CONTEXT_H__ */
