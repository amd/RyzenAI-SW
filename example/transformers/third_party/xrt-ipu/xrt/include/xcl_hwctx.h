// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XCL_HWCTX_H_
#define XCL_HWCTX_H_

// Definitions related to HW context shared between user space XRT and
// Linux kernel driver.  The header file is exported as underlying
// types are exposed in xrt::hw_context::qos definition.

#ifdef __cplusplus
# include <cstdint>
extern "C" {
#else
# if defined(__KERNEL__)
#  include <linux/types.h>
# else
#  include <stdint.h>
# endif
#endif

// Underlying representation of a hardware context handle.
//
// The context handle is among other things used with / encoded in
// buffer object flags.
typedef uint32_t xcl_hwctx_handle;
#define XRT_NULL_HWCTX 0xffffffff

#ifdef __cplusplus
}
#endif

#endif
