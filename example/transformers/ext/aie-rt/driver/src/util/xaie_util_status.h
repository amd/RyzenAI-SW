/******************************************************************************
* Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All Rights Reserved.
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

/*Core Status register lookup*/
static const char* XAie_CoreStatus_Strings[] = {
    [XAIE_CORE_STATUS_ENABLE_BIT]              = "Enable",
    [XAIE_CORE_STATUS_RESET_BIT]               = "Reset",
    [XAIE_CORE_STATUS_MEM_STALL_S_BIT]         = "Memory_Stall_S",
    [XAIE_CORE_STATUS_MEM_STALL_W_BIT]         = "Memory_Stall_W",
    [XAIE_CORE_STATUS_MEM_STALL_N_BIT]         = "Memory_Stall_N",
    [XAIE_CORE_STATUS_MEM_STALL_E_BIT]         = "Memory_Stall_E",
    [XAIE_CORE_STATUS_LOCK_STALL_S_BIT]        = "Lock_Stall_S",
    [XAIE_CORE_STATUS_LOCK_STALL_W_BIT]        = "Lock_Stall_W",
    [XAIE_CORE_STATUS_LOCK_STALL_N_BIT]        = "Lock_Stall_N",
    [XAIE_CORE_STATUS_LOCK_STALL_E_BIT]        = "Lock_Stall_E",
    [XAIE_CORE_STATUS_STREAM_STALL_SS0_BIT]    = "Stream_Stall_SSO",
    [XAIE_CORE_STATUS_STREAM_STALL_MS0_BIT]    = "Stream_Stall_MSO",
    [XAIE_CORE_STATUS_CASCADE_STALL_SCD_BIT]   = "Cascade_Stall_SCD",
    [XAIE_CORE_STATUS_CASCADE_STALL_MCD_BIT]   = "Cascade_Stall_MCD",
    [XAIE_CORE_STATUS_DEBUG_HALT_BIT]          = "Debug_Halt",
    [XAIE_CORE_STATUS_ECC_ERROR_STALL_BIT]     = "ECC_Error_Stall",
    [XAIE_CORE_STATUS_ECC_SCRUBBING_STALL_BIT] = "ECC_Scrubbing_Stall",
    [XAIE_CORE_STATUS_ERROR_HALT_BIT]          = "Error_Halt",
    [XAIE_CORE_STATUS_DONE_BIT]                = "Core_Done",
    [XAIE_CORE_STATUS_PROCESSOR_BUS_STALL_BIT] = "Core_Proc_Bus_Stall",
};

/*DMA S2MM Status Register lookup*/
static const char* XAie_DmaS2MMStatus_Strings[] = {
    [XAIE_DMA_STATUS_S2MM_STATUS]                         = "Status",
    [XAIE_DMA_STATUS_S2MM_STALLED_LOCK_ACK]               = "Stalled_Lock_Acq",
    [XAIE_DMA_STATUS_S2MM_STALLED_LOCK_REL]               = "Stalled_Lock_Rel",
    [XAIE_DMA_STATUS_S2MM_STALLED_STREAM_STARVATION]      = "Stalled_Stream_Starvation",
    [XAIE_DMA_STATUS_S2MM_STALLED_TCT_OR_COUNT_FIFO_FULL] = "Stalled_TCT_Or_Count_FIFO_Full",
    [XAIE_DMA_STATUS_S2MM_ERROR_LOCK_ACCESS_TO_UNAVAIL]   = "Error_Lock_Access_Unavail",
    [XAIE_DMA_STATUS_S2MM_ERROR_DM_ACCESS_TO_UNAVAIL]     = "Error_DM_Access_Unavail",
    [XAIE_DMA_STATUS_S2MM_ERROR_BD_UNAVAIL]               = "Error_BD_Unavail",
    [XAIE_DMA_STATUS_S2MM_ERROR_BD_INVALID]               = "Error_BD_Invalid",
    [XAIE_DMA_STATUS_S2MM_ERROR_FOT_LENGTH]               = "Error_FoT_Length",
    [XAIE_DMA_STATUS_S2MM_ERROR_FOT_BDS_PER_TASK]         = "Error_Fot_BDs",
    [XAIE_DMA_STATUS_S2MM_AXI_MM_DECODE_ERROR]            = "AXI-MM_decode_error",
    [XAIE_DMA_STATUS_S2MM_AXI_MM_SLAVE_ERROR]             = "AXI-MM_slave_error",
    [XAIE_DMA_STATUS_S2MM_TASK_QUEUE_OVERFLOW]            = "Task_Queue_Overflow",
    [XAIE_DMA_STATUS_S2MM_CHANNEL_RUNNING]                = "Channel_Running",
    [XAIE_DMA_STATUS_S2MM_TASK_QUEUE_SIZE]                = "Task_Queue_Size",
    [XAIE_DMA_STATUS_S2MM_CURRENT_BD]                     = "Cur_BD",
};

/*DMA MM2S Status register lookup*/
static const char* XAie_DmaMM2SStatus_Strings[] = {
    [XAIE_DMA_STATUS_MM2S_STATUS]                       = "Status",
    [XAIE_DMA_STATUS_MM2S_STALLED_LOCK_ACK]             = "Stalled_Lock_Acq",
    [XAIE_DMA_STATUS_MM2S_STALLED_LOCK_REL]             = "Stalled_Lock_Rel",
    [XAIE_DMA_STATUS_MM2S_STALLED_STREAM_BACKPRESSURE]  = "Stalled_Stream_Back_Pressure",
    [XAIE_DMA_STATUS_MM2S_STALLED_TCT]                  = "Stalled_TCT",
    [XAIE_DMA_STATUS_MM2S_ERROR_LOCK_ACCESS_TO_UNAVAIL] = "Error_Lock_Access_Unavail",
    [XAIE_DMA_STATUS_MM2S_ERROR_DM_ACCESS_TO_UNAVAIL]   = "Error_DM_Access_Unavail",
    [XAIE_DMA_STATUS_MM2S_ERROR_BD_UNAVAIL]             = "Error_BD_Unavail",
    [XAIE_DMA_STATUS_MM2S_ERROR_BD_INVALID]             = "Error_BD_Invalid",
    [XAIE_DMA_STATUS_MM2S_AXI_MM_DECODE_ERROR]          = "AXI-MM_decode_error",
    [XAIE_DMA_STATUS_MM2S_AXI_MM_SLAVE_ERROR]           = "AXI-MM_slave_error",
    [XAIE_DMA_STATUS_MM2S_TASK_QUEUE_OVERFLOW]          = "Task_Queue_Overflow",
    [XAIE_DMA_STATUS_MM2S_CHANNEL_RUNNING]              = "Channel_Running",
    [XAIE_DMA_STATUS_MM2S_TASK_QUEUE_SIZE]              = "Task_Queue_Size",
    [XAIE_DMA_STATUS_MM2S_CURRENT_BD]                   = "Cur_BD",
};



/**************************** Function Prototypes *******************************/
#define TRUE   1u
#define FALSE  0u

int XAie_CoreStatus_CSV(u32 Reg, char *Buf, u32 BufSize);
int XAie_DmaS2MMStatus_CSV(u32 Reg, char *Buf, u32 BufSize, u8 );
int XAie_DmaMM2SStatus_CSV(u32 Reg, char *Buf, u32 BufSize, u8 );

#define MAX_CHAR_ARRAY_SIZE   400
#define MAX_EVENT_CHAR_ARRAY_SIZE  2000
#endif  /* XAIE_FEATURE_UTIL_STATUS_ENABLE */

/** @} */
