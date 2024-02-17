/******************************************************************************
* Copyright (C) 2022 AMD.  All rights reserved.
* Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All Rights Reserved.  *
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_util_status.h
* @{
*
* Header to include function prototypes for AIE status utilities
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   dsteger 07/25/2022  Initial creation
* </pre>
*
******************************************************************************/
#ifndef XAIE_UTIL_STATUS_H
#define XAIE_UTIL_STATUS_H

/***************************** Include Files *********************************/

#include "xaie_feature_config.h"
#include "xaiegbl_defs.h"
#include "xaiegbl.h"

#ifdef XAIE_FEATURE_UTIL_STATUS_ENABLE

/**************************** Variable Definitions *******************************/

enum {
    XAIE_CORE_STATUS_ENABLE_BIT = 0U,
    XAIE_CORE_STATUS_RESET_BIT,
    XAIE_CORE_STATUS_MEM_STALL_S_BIT,
    XAIE_CORE_STATUS_MEM_STALL_W_BIT,
    XAIE_CORE_STATUS_MEM_STALL_N_BIT,
    XAIE_CORE_STATUS_MEM_STALL_E_BIT,
    XAIE_CORE_STATUS_LOCK_STALL_S_BIT,
    XAIE_CORE_STATUS_LOCK_STALL_W_BIT,
    XAIE_CORE_STATUS_LOCK_STALL_N_BIT,
    XAIE_CORE_STATUS_LOCK_STALL_E_BIT,
    XAIE_CORE_STATUS_STREAM_STALL_SS0_BIT,
    XAIE_CORE_STATUS_STREAM_STALL_MS0_BIT = 12,
    XAIE_CORE_STATUS_CASCADE_STALL_SCD_BIT = 14,
    XAIE_CORE_STATUS_CASCADE_STALL_MCD_BIT,
    XAIE_CORE_STATUS_DEBUG_HALT_BIT,
    XAIE_CORE_STATUS_ECC_ERROR_STALL_BIT,
    XAIE_CORE_STATUS_ECC_SCRUBBING_STALL_BIT,
    XAIE_CORE_STATUS_ERROR_HALT_BIT,
    XAIE_CORE_STATUS_DONE_BIT,
    XAIE_CORE_STATUS_PROCESSOR_BUS_STALL_BIT,
    XAIE_CORE_STATUS_MAX_BIT
};

enum DmaStatus_S2MM_enum{
    XAIE_DMA_STATUS_S2MM_STATUS = 0U,
    XAIE_DMA_STATUS_S2MM_STALLED_LOCK_ACK = 2U,
    XAIE_DMA_STATUS_S2MM_STALLED_LOCK_REL,
    XAIE_DMA_STATUS_S2MM_STALLED_STREAM_STARVATION,
    XAIE_DMA_STATUS_S2MM_STALLED_TCT_OR_COUNT_FIFO_FULL,
    XAIE_DMA_STATUS_S2MM_ERROR_LOCK_ACCESS_TO_UNAVAIL = 8U, // Specific only to MEM Tile
    XAIE_DMA_STATUS_S2MM_ERROR_DM_ACCESS_TO_UNAVAIL,       // Specific only to MEM Tile
    XAIE_DMA_STATUS_S2MM_ERROR_BD_UNAVAIL = 10U,
    XAIE_DMA_STATUS_S2MM_ERROR_BD_INVALID,
    XAIE_DMA_STATUS_S2MM_ERROR_FOT_LENGTH,
    XAIE_DMA_STATUS_S2MM_ERROR_FOT_BDS_PER_TASK,
    XAIE_DMA_STATUS_S2MM_AXI_MM_DECODE_ERROR = 16U,
    XAIE_DMA_STATUS_S2MM_AXI_MM_SLAVE_ERROR  = 17U,
    XAIE_DMA_STATUS_S2MM_TASK_QUEUE_OVERFLOW = 18U,
    XAIE_DMA_STATUS_S2MM_CHANNEL_RUNNING,
    XAIE_DMA_STATUS_S2MM_TASK_QUEUE_SIZE,
    XAIE_DMA_STATUS_S2MM_CURRENT_BD = 24U,
    XAIE_DMA_STATUS_S2MM_MAX
};

enum DmaStatus_MM2S_enum{
    XAIE_DMA_STATUS_MM2S_STATUS = 0U,
    XAIE_DMA_STATUS_MM2S_STALLED_LOCK_ACK = 2U,
    XAIE_DMA_STATUS_MM2S_STALLED_LOCK_REL,
    XAIE_DMA_STATUS_MM2S_STALLED_STREAM_BACKPRESSURE,
    XAIE_DMA_STATUS_MM2S_STALLED_TCT,
    XAIE_DMA_STATUS_MM2S_ERROR_LOCK_ACCESS_TO_UNAVAIL = 8U, // Specific only to MEM Tile
    XAIE_DMA_STATUS_MM2S_ERROR_DM_ACCESS_TO_UNAVAIL,        // Specific only to MEM Tile
    XAIE_DMA_STATUS_MM2S_ERROR_BD_UNAVAIL,
    XAIE_DMA_STATUS_MM2S_ERROR_BD_INVALID = 11U,
    XAIE_DMA_STATUS_MM2S_AXI_MM_DECODE_ERROR = 16U,
    XAIE_DMA_STATUS_MM2S_AXI_MM_SLAVE_ERROR  = 17U,
    XAIE_DMA_STATUS_MM2S_TASK_QUEUE_OVERFLOW = 18U,
    XAIE_DMA_STATUS_MM2S_CHANNEL_RUNNING,
    XAIE_DMA_STATUS_MM2S_TASK_QUEUE_SIZE,
    XAIE_DMA_STATUS_MM2S_CURRENT_BD = 24U,
    XAIE_DMA_STATUS_MM2S_MAX
};

/**************************** Function Prototypes *******************************/
#define TRUE   1u
#define FALSE  0u

int XAie_CoreStatus_CSV(u32 Reg, char *Buf);
int XAie_DmaS2MMStatus_CSV(u32 Reg, char *Buf, u8 );
int XAie_DmaMM2SStatus_CSV(u32 Reg, char *Buf, u8 );

#define MAX_CHAR_ARRAY_SIZE   400
#endif  /* XAIE_FEATURE_UTIL_STATUS_ENABLE */

#endif	/* end of protection macro */

/** @} */
