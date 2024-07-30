/******************************************************************************
* Copyright (C) 2023 AMD, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_interrupt_aie2ps.c
* @{
*
* This file contains AIE2PS specific interrupt routines which are not exposed
* to the user.
*
******************************************************************************/
/***************************** Include Files *********************************/
#include "xaie_feature_config.h"
#include "xaie_helper.h"
#include "xaie_interrupt_aie2ps.h"

#ifdef XAIE_FEATURE_INTR_INIT_ENABLE
/************************** Constant Definitions *****************************/
/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
* This API computes first level IRQ broadcast ID.
*
* @param	DevInst: Device Instance
* @param	Loc: Location of AIE Tile
* @param	Switch: Switch in the given module. For a shim tile, value
* 			could be XAIE_EVENT_SWITCH_A or XAIE_EVENT_SWITCH_B.
*
* @return	IrqId: IRQ broadcast ID.
*
* @note		AIE2PS has NOC tile in every SHIM tile. Each switch in L1 ctrl
* 		maps to 0 or 1.
*
*		Internal Only.
******************************************************************************/
u8 _XAie2ps_IntrCtrlL1IrqId(XAie_DevInst *DevInst, XAie_LocType Loc,
		XAie_BroadcastSw Switch)
{
	(void)DevInst;
	(void)Loc;
	return Switch;
}

#endif /* XAIE_FEATURE_INTR_INIT_ENABLE */

/** @} */
