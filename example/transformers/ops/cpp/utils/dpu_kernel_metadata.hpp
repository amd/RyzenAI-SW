/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __DPU_KERNEL_MDATA_H__
#define __DPU_KERNEL_MDATA_H__

#include <cstdint>

constexpr auto KERNEL_NAME = "DPU";
constexpr std::uint64_t DDR_AIE_ADDR_OFFSET = std::uint64_t{0x80000000};
constexpr std::uint64_t OPCODE = std::uint64_t{2};

#endif /* __DPU_KERNEL_MDATA_H__ */
