/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __OP_DEFS_H__
#define __OP_DEFS_H__

#define OP_LIST(OP) \
        OP(TRANSACTION_OP) \
        OP(WAIT_OP) \
        OP(PENDINGBDCOUNT_OP) \
        OP(DBGPRINT_OP) \
        OP(PATCHBD_OP)

#include "op_base.h"
#endif
