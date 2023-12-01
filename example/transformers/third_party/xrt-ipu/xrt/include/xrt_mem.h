/*
 *  SPDX-License-Identifier: Apache-2.0
 *  Copyright (C) 2019-2022, Xilinx Inc
 *  Copyright (C) 2022, Advanced Micro Devices, Inc.  All rights reserved.
 */
#ifndef _XRT_MEM_H_
#define _XRT_MEM_H_

#ifdef _WIN32
# pragma warning( push )
# pragma warning( disable : 4201 )
#endif

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

/**
 * Encoding of flags passed to xcl buffer allocation APIs
 */
struct xcl_bo_flags
{
  union {
    uint32_t flags;
    struct {
      uint16_t bank;       // [15-0]
      uint8_t  slot;       // [16-23]
      uint8_t  boflags;    // [24-31]
    };
  };
};

/**
 * XCL BO Flags bits layout
 *
 * bits  0 ~ 15: DDR BANK index
 * bits 24 ~ 31: BO flags
 */
#define XRT_BO_FLAGS_MEMIDX_MASK	(0xFFFFFFUL)
#define	XCL_BO_FLAGS_NONE		(0)
#define	XCL_BO_FLAGS_CACHEABLE		(1U << 24)
#define	XCL_BO_FLAGS_KERNBUF		(1U << 25)
#define	XCL_BO_FLAGS_SGL		(1U << 26)
#define	XCL_BO_FLAGS_SVM		(1U << 27)
#define	XCL_BO_FLAGS_DEV_ONLY		(1U << 28)
#define	XCL_BO_FLAGS_HOST_ONLY		(1U << 29)
#define	XCL_BO_FLAGS_P2P		(1U << 30)
#define	XCL_BO_FLAGS_EXECBUF		(1U << 31)

/**
 * XRT Native BO flags
 *
 * These flags are simple aliases for use with XRT native BO APIs.
 */
#define XRT_BO_FLAGS_NONE      XCL_BO_FLAGS_NONE
#define XRT_BO_FLAGS_CACHEABLE XCL_BO_FLAGS_CACHEABLE
#define XRT_BO_FLAGS_DEV_ONLY  XCL_BO_FLAGS_DEV_ONLY
#define XRT_BO_FLAGS_HOST_ONLY XCL_BO_FLAGS_HOST_ONLY
#define XRT_BO_FLAGS_P2P       XCL_BO_FLAGS_P2P
#define XRT_BO_FLAGS_SVM       XCL_BO_FLAGS_SVM

/**
 * This is the legacy usage of XCL DDR Flags.
 *
 * byte-0 lower 4 bits for DDR Flags are one-hot encoded
 */
enum xclDDRFlags {
    XCL_DEVICE_RAM_BANK0 = 0x00000000,
    XCL_DEVICE_RAM_BANK1 = 0x00000002,
    XCL_DEVICE_RAM_BANK2 = 0x00000004,
    XCL_DEVICE_RAM_BANK3 = 0x00000008,
};

#ifdef __cplusplus
}
#endif

#ifdef _WIN32
# pragma warning( pop )
#endif

#endif
