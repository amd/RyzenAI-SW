/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */


#include "interpreter.h"
#include "op_defs.h"
#include "op_base.h"
#include "op_types.h"

//TODO: Refactor to use ErrCodes.h in dpufw
#define ERR_TXN_FW_SUCCESS (0)
#define ERR_TXN_FW_UNKNOWN_FW_OP_CODE (1 << 1)

#define GENERATE_FUNC_INSTRNAME(INSTRNAME) op_##INSTRNAME##_func,

int (*op_func_ptrs[]) (XAie_DevInst*, op_base *, u8, const u8 *) = {
    OP_LIST(GENERATE_FUNC_INSTRNAME)
};

int32_t RunDPUInstTransaction(XAie_DevInst *DevInst, u8 *instr_pc, unsigned sz, u8 start_col_idx, const u8 *args) {
  const u8* limit = instr_pc + sz;
  while (instr_pc < limit ) {
    op_base * ibuf = (op_base*)instr_pc;

    if (ibuf->type != e_TRANSACTION_OP) {
      return ERR_TXN_FW_UNKNOWN_FW_OP_CODE;
    }

    op_TRANSACTION_OP_func( DevInst, ibuf, start_col_idx, args );
    instr_pc += ibuf->size_in_bytes;
  }

  return ERR_TXN_FW_SUCCESS;
}
