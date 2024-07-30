/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __DPU_KERNEL_MDATA_H__
#define __DPU_KERNEL_MDATA_H__

#include <cstdint>
// TODO(varunsh): this is needed on Linux to prevent compiler errors
// when including "op_buf.hpp" from aie_controller. Many files include
// that header but also include this one so this header is here for now
#include <cstring>

constexpr std::uint64_t DDR_AIE_ADDR_OFFSET = std::uint64_t{0x80000000};
constexpr std::uint64_t OPCODE = std::uint64_t{2};

#endif /* __DPU_KERNEL_MDATA_H__ */
