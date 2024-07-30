/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __INTERPRETER_H__
#define __INTERPRETER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <xaiengine.h>

#include "interpreter_exec_impl.h"
#include "interpreter_op_impl.h"

int32_t RunDPUInstTransaction(XAie_DevInst *DevInst, u8 *instr_pc, unsigned sz, u8 start_col_idx, const u8 *args);

#ifdef __cplusplus
}
#endif


#endif
