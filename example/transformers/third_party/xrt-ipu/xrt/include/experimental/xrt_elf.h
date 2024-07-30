// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef XRT_ELF_H_
#define XRT_ELF_H_

#include "xrt/detail/config.h"
#include "xrt/detail/pimpl.h"
#include "xrt/xrt_uuid.h"

#ifdef __cplusplus
# include <string>
#endif

#ifdef __cplusplus
namespace xrt {

/*!
 * @class elf
 *
 * @brief
 * An elf contains instructions for functions to execute in some
 * pre-configured hardware.  The xrt::elf class provides APIs to mine
 * the elf itself for relevant data.
 *
 * An xclbin is used to configure the hardware and an elf object is
 * always associated with exactly one xclbin, meaning the instructions
 * are for a specific hardware configuration.
 */
class elf_impl;
class elf : public detail::pimpl<elf_impl>
{
public:
  XRT_API_EXPORT
  elf(const std::string& fnm);
  
  XRT_API_EXPORT
  xrt::uuid
  get_cfg_uuid() const;
};

} // namespace xrt

#else
# error xrt::elf is only implemented for C++
#endif // __cplusplus

#endif
