/*
 * Copyright (C) 2021-2022 Xilinx, Inc
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef xrt_system_h_
#define xrt_system_h_

#include "xrt.h"

#ifdef __cplusplus

/*!
 * @namespace xrt::system
 *
 * @brief
 * APIs for system level queries and control.
 */
namespace xrt { namespace system {

/**
 * enumerate_devices() - Enumerate devices found in the system
 *
 * @return
 *  Number of devices in the system recognized by XRT
 */
XCL_DRIVER_DLLESPEC
unsigned int
enumerate_devices();

}}

#endif // __cplusplus

#endif
