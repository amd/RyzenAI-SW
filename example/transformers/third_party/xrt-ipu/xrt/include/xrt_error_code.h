/*
 *  SPDX-License-Identifier: Apache-2.0
 *  Copyright (C) 2020-2021, Xilinx Inc
 *  Copyright (C) 2022, Advanced Micro Devices, Inc.  All rights reserved.
 */
#ifndef XRT_ERROR_CODE_H_
#define XRT_ERROR_CODE_H_

#if defined(__KERNEL__)
# include <linux/types.h>
#elif defined(__cplusplus)
# include <cstdint>
#else
# include <stdint.h>
#endif

/**
 * xrtErrorCode layout
 *
 * This layout is internal to XRT (akin to a POSIX error code).
 *
 * The error code is populated by driver and consumed by XRT
 * implementation where it is translated into an actual error / info /
 * warning that is propagated to the end user.
 *
 * 63 - 48  47 - 40   39 - 32   31 - 24   23 - 16    15 - 0
 * --------------------------------------------------------
 * |    |    |    |    |    |    |    |    |    |    |----| xrtErrorNum
 * |    |    |    |    |    |    |    |    |----|---------- xrtErrorDriver
 * |    |    |    |    |    |    |----|-------------------- xrtErrorSeverity
 * |    |    |    |    |----|------------------------------ xrtErrorModule
 * |    |    |----|---------------------------------------- xrtErrorClass
 * |----|-------------------------------------------------- reserved
 *
 */
typedef uint64_t xrtErrorCode;
typedef uint64_t xrtErrorTime;

#define XRT_ERROR_NUM_MASK		0xFFFFUL
#define XRT_ERROR_NUM_SHIFT		0
#define XRT_ERROR_DRIVER_MASK		0xFUL
#define XRT_ERROR_DRIVER_SHIFT		16
#define XRT_ERROR_SEVERITY_MASK		0xFUL
#define XRT_ERROR_SEVERITY_SHIFT	24
#define XRT_ERROR_MODULE_MASK		0xFUL
#define XRT_ERROR_MODULE_SHIFT		32
#define XRT_ERROR_CLASS_MASK		0xFUL
#define XRT_ERROR_CLASS_SHIFT		40

#define	XRT_ERROR_CODE_BUILD(num, driver, severity, module, eclass) \
	((((num) & XRT_ERROR_NUM_MASK) << XRT_ERROR_NUM_SHIFT) | \
	(((driver) & XRT_ERROR_DRIVER_MASK) << XRT_ERROR_DRIVER_SHIFT) | \
	(((severity) & XRT_ERROR_SEVERITY_MASK) << XRT_ERROR_SEVERITY_SHIFT) | \
	(((module) & XRT_ERROR_MODULE_MASK) << XRT_ERROR_MODULE_SHIFT) | \
	(((eclass) & XRT_ERROR_CLASS_MASK) << XRT_ERROR_CLASS_SHIFT))

#define XRT_ERROR_NUM(code) (((code) >> XRT_ERROR_NUM_SHIFT) & XRT_ERROR_NUM_MASK)
#define XRT_ERROR_DRIVER(code) (((code) >> XRT_ERROR_DRIVER_SHIFT) & XRT_ERROR_DRIVER_MASK)
#define XRT_ERROR_SEVERITY(code) (((code) >> XRT_ERROR_SEVERITY_SHIFT) & XRT_ERROR_SEVERITY_MASK)
#define XRT_ERROR_MODULE(code) (((code) >> XRT_ERROR_MODULE_SHIFT) & XRT_ERROR_MODULE_MASK)
#define XRT_ERROR_CLASS(code) (((code) >> XRT_ERROR_CLASS_SHIFT) & XRT_ERROR_CLASS_MASK)

/**
 * xrt_error_num - XRT specific error numbers
 */
enum xrtErrorNum {
  XRT_ERROR_NUM_FIRWWALL_TRIP = 1,
  XRT_ERROR_NUM_TEMP_HIGH,
  XRT_ERROR_NUM_AIE_SATURATION,
  XRT_ERROR_NUM_AIE_FP,
  XRT_ERROR_NUM_AIE_STREAM,
  XRT_ERROR_NUM_AIE_ACCESS,
  XRT_ERROR_NUM_AIE_BUS,
  XRT_ERROR_NUM_AIE_INSTRUCTION,
  XRT_ERROR_NUM_AIE_ECC,
  XRT_ERROR_NUM_AIE_LOCK,
  XRT_ERROR_NUM_AIE_DMA,
  XRT_ERROR_NUM_AIE_MEM_PARITY,
  XRT_ERROR_NUM_KDS_CU,
  XRT_ERROR_NUM_KDS_EXEC,
  XRT_ERROR_NUM_UNKNOWN
};

enum xrtErrorDriver {
  XRT_ERROR_DRIVER_XOCL = 1,
  XRT_ERROR_DRIVER_XCLMGMT,
  XRT_ERROR_DRIVER_ZOCL,
  XRT_ERROR_DRIVER_AIE,
  XRT_ERROR_DRIVER_UNKNOWN
};

enum xrtErrorSeverity {
  XRT_ERROR_SEVERITY_EMERGENCY = 1,
  XRT_ERROR_SEVERITY_ALERT,
  XRT_ERROR_SEVERITY_CRITICAL,
  XRT_ERROR_SEVERITY_ERROR,
  XRT_ERROR_SEVERITY_WARNING,
  XRT_ERROR_SEVERITY_NOTICE,
  XRT_ERROR_SEVERITY_INFO,
  XRT_ERROR_SEVERITY_DEBUG,
  XRT_ERROR_SEVERITY_UNKNOWN
};

enum xrtErrorModule {
  XRT_ERROR_MODULE_FIREWALL = 1,
  XRT_ERROR_MODULE_CMC,
  XRT_ERROR_MODULE_AIE_CORE,
  XRT_ERROR_MODULE_AIE_MEMORY,
  XRT_ERROR_MODULE_AIE_SHIM,
  XRT_ERROR_MODULE_AIE_NOC,
  XRT_ERROR_MODULE_AIE_PL,
  XRT_ERROR_MODULE_UNKNOWN
};

enum xrtErrorClass {
  XRT_ERROR_CLASS_FIRST_ENTRY = 1,
  XRT_ERROR_CLASS_SYSTEM = XRT_ERROR_CLASS_FIRST_ENTRY,
  XRT_ERROR_CLASS_AIE,
  XRT_ERROR_CLASS_HARDWARE,
  XRT_ERROR_CLASS_UNKNOWN,
  XRT_ERROR_CLASS_LAST_ENTRY = XRT_ERROR_CLASS_UNKNOWN
};

#endif
