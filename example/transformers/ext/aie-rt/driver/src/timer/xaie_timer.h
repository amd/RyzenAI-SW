/******************************************************************************
* Copyright (C) 2019 - 2020 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_timer.h
* @{
*
* Header file for timer implementation.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who      Date     Changes
* ----- ------   -------- -----------------------------------------------------
* 1.0   Dishita  04/05/2020  Initial creation
* </pre>
*
******************************************************************************/
#ifndef XAIETIMER_H
#define XAIETIMER_H

/***************************** Include Files *********************************/
#include "xaie_events.h"
#include "xaiegbl.h"

/************************** Enum *********************************************/

/************************** Function Prototypes  *****************************/
XAIE_AIG_EXPORT AieRC XAie_SetTimerTrigEventVal(XAie_DevInst *DevInst, XAie_LocType Loc,
		XAie_ModuleType Module, u32 LowEventValue, u32 HighEventValue);
XAIE_AIG_EXPORT AieRC XAie_ResetTimer(XAie_DevInst *DevInst, XAie_LocType Loc,
		XAie_ModuleType Module);
XAIE_AIG_EXPORT AieRC XAie_SetTimerResetEvent(XAie_DevInst *DevInst, XAie_LocType Loc,
		XAie_ModuleType Module, XAie_Events Event,
		XAie_Reset Reset);
XAIE_AIG_EXPORT AieRC XAie_ReadTimer(XAie_DevInst *DevInst, XAie_LocType Loc,
		XAie_ModuleType Module, u64 *TimerVal);
XAIE_AIG_EXPORT AieRC XAie_WaitCycles(XAie_DevInst *DevInst, XAie_LocType Loc,
		XAie_ModuleType Module, u64 CycleCnt);
XAIE_AIG_EXPORT AieRC XAie_SyncTimer(XAie_DevInst *DevInst, u8 BcastChannelId);

#endif		/* end of protection macro */
