/******************************************************************************
* Copyright (C) 2019 - 2020 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_core.h
* @{
*
* Header file for core control and wait functions
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Tejus   09/24/2019  Initial creation
* 1.1   Tejus   03/20/2020  Remove range apis
* 1.2   Tejus   06/01/2020  Add core debug halt apis
* 1.3   Tejus   06/01/2020  Add api to read core done bit.
* 1.4   Tejus   06/05/2020  Add api to reset/unreset aie cores.
* </pre>
*
******************************************************************************/
#ifndef XAIECORE_H
#define XAIECORE_H

/***************************** Include Files *********************************/
#include "xaiegbl.h"
#include "xaiegbl_defs.h"
#include "xaie_helper.h"

/************************** Constant Definitions *****************************/
#define XAIE_CORE_DEBUG_STATUS_ANY_HALT			(1U << 0U)
#define XAIE_CORE_DEBUG_STATUS_PC_EVENT_HALT		(1U << 1U)
#define XAIE_CORE_DEBUG_STATUS_MEM_STALL_HALT		(1U << 2U)
#define XAIE_CORE_DEBUG_STATUS_LOCK_STALL_HALT		(1U << 3U)
#define XAIE_CORE_DEBUG_STATUS_STREAM_STALL_HALT 	(1U << 4U)
#define XAIE_CORE_DEBUG_STATUS_EVENT0_STALL_HALT 	(1U << 5U)
#define XAIE_CORE_DEBUG_STATUS_EVENT1_STALL_HALT 	(1U << 6U)

#define XAIE_CORE_STATUS_ENABLE				(1U <<  0U)
#define XAIE_CORE_STATUS_RESET     			(1U <<  1U)
#define XAIE_CORE_STATUS_MEM_STALL_S			(1U <<  2U)
#define XAIE_CORE_STATUS_MEM_STALL_W			(1U <<  3U)
#define XAIE_CORE_STATUS_MEM_STALL_N			(1U <<  4U)
#define XAIE_CORE_STATUS_MEM_STALL_E			(1U <<  5U)
#define XAIE_CORE_STATUS_LOCK_STALL_S			(1U <<  6U)
#define XAIE_CORE_STATUS_LOCK_STALL_W			(1U <<  7U)
#define XAIE_CORE_STATUS_LOCK_STALL_N			(1U <<  8U)
#define XAIE_CORE_STATUS_LOCK_STALL_E			(1U <<  9U)
#define XAIE_CORE_STATUS_STREAM_STALL_SS0  		(1U << 10U)
#define XAIE_CORE_STATUS_STREAM_STALL_SS1  		(1U << 11U)
#define XAIE_CORE_STATUS_STREAM_STALL_MS0  		(1U << 12U)
#define XAIE_CORE_STATUS_STREAM_STALL_MS1  		(1U << 13U)
#define XAIE_CORE_STATUS_CASCADE_STALL_SCD 		(1U << 14U)
#define XAIE_CORE_STATUS_CASCADE_STALL_MCD 		(1U << 15U)
#define XAIE_CORE_STATUS_DEBUG_HALT        		(1U << 16U)
#define XAIE_CORE_STATUS_ECC_ERROR_STALL        	(1U << 17U)
#define XAIE_CORE_STATUS_ECC_SCRUBBING_STALL      	(1U << 18U)
#define XAIE_CORE_STATUS_ERROR_HALT        		(1U << 19U)
#define XAIE_CORE_STATUS_DONE              		(1U << 20U)
#define XAIE_CORE_STATUS_PROCESSOR_BUS_STALL      	(1U << 21U)

/************************** Function Prototypes  *****************************/
/*****************************************************************************/
/*
*
* This API checks for the event due to which AIE was debug halted.
*
* @param	DebugStatus: Value of debug status register.
* @param	DebugEventMask: Debug Event Mask.
*		Any of XAIE_CORE_DEBUG_STATUS_*_HALT macros.
* @return	1 on success. 0 of failure.
*
* @note		None.
*
******************************************************************************/
static inline u32 XAie_CheckDebugHaltStatus(u32 DebugStatus, u32 DebugEventMask)
{
	return (DebugStatus & DebugEventMask) ? 1U : 0U;
}

/*****************************************************************************/
/*
*
* This API checks for the state of the indiviual bits in core status register.
*
* @param	CoreStatus: Value of core status register.
* @param	CoreStatMask: Core Event Mask.
*		Any of XAIE_CORE_STATUS_* macros.
* @return	1 if the bit is set. 0 otherwise.
*
* @note		None.
*
******************************************************************************/
static inline u32 XAie_CheckCoreStatus(u32 CoreStatus, u32 CoreStatMask)
{
	return (CoreStatus & CoreStatMask) ? 1U : 0U;
}

AieRC XAie_CoreDisable(XAie_DevInst *DevInst, XAie_LocType Loc);
AieRC XAie_CoreEnable(XAie_DevInst *DevInst, XAie_LocType Loc);
AieRC XAie_CoreWaitForDone(XAie_DevInst *DevInst, XAie_LocType Loc,
		u32 TimeOut);
AieRC XAie_CoreWaitForDisable(XAie_DevInst *DevInst, XAie_LocType Loc,
		u32 TimeOut);
AieRC XAie_CoreDebugHalt(XAie_DevInst *DevInst, XAie_LocType Loc);
AieRC XAie_CoreDebugUnhalt(XAie_DevInst *DevInst, XAie_LocType Loc);
AieRC XAie_CoreGetDebugHaltStatus(XAie_DevInst *DevInst, XAie_LocType Loc,
		u32 *DebugStatus);
AieRC XAie_CoreGetPCValue(XAie_DevInst *DevInst, XAie_LocType Loc,
		u32 *PCValue);
AieRC XAie_CoreReadDoneBit(XAie_DevInst *DevInst, XAie_LocType Loc,
		u8 *DoneBit);
AieRC XAie_CoreGetStatus(XAie_DevInst *DevInst, XAie_LocType Loc,
		u32 *CoreStatus);
AieRC XAie_CoreReset(XAie_DevInst *DevInst, XAie_LocType Loc);
AieRC XAie_CoreUnreset(XAie_DevInst *DevInst, XAie_LocType Loc);
AieRC XAie_CoreConfigDebugControl1(XAie_DevInst *DevInst, XAie_LocType Loc,
		XAie_Events Event0, XAie_Events Event1,
		XAie_Events SingleStepEvent, XAie_Events ResumeCoreEvent);
AieRC XAie_CoreClearDebugControl1(XAie_DevInst *DevInst, XAie_LocType Loc);
AieRC XAie_CoreConfigureEnableEvent(XAie_DevInst *DevInst, XAie_LocType Loc,
		XAie_Events Event);
AieRC XAie_CoreConfigureDone(XAie_DevInst *DevInst, XAie_LocType Loc);
AieRC XAie_CoreConfigAccumulatorControl(XAie_DevInst *DevInst,
		XAie_LocType Loc, StrmSwPortType InDir, StrmSwPortType OutDir);
AieRC XAie_ClearCoreDisableEventOccurred(XAie_DevInst *DevInst,
		XAie_LocType Loc);
AieRC XAie_CoreProcessorBusEnable(XAie_DevInst *DevInst, XAie_LocType Loc);
AieRC XAie_CoreProcessorBusDisable(XAie_DevInst *DevInst, XAie_LocType Loc);

#endif		/* end of protection macro */
/** @} */
