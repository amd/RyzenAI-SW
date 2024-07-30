/******************************************************************************
* Copyright (C) 2021 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_interrupt_aie2ipu.c
* @{
*
* This file contains AIE2 IPU specific interrupt routines which are not exposed
* to the user.
*
******************************************************************************/
/***************************** Include Files *********************************/
#include "xaie_feature_config.h"
#include "xaie_helper.h"
#include "xaie_interrupt_aie2ipu.h"

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
* @note		IRQ ID for each switch block starts from 0, every block on the
*		left will increase by 1 until it reaches the first Shim NoC
*		column. The IRQ ID restarts from 0 on the switch A of the
*		second shim NoC column. For the shim PL columns after the
*		second Shim NoC, if there is no shim NoC further right, the
*		column will use the shim NoC on the left. That is the L1 IRQ
*		broadcast ID pattern,
*		For column from 0 to 1 is: 0 1 2 3
*		For column from 2 to 4 is: 0 1
*
*		Here we do not use Loc.Col + DevInst->start for the folling
*		reasons:
*
*		The cdo generated for IPU needs to be relocatable across other
*		columns. For building the cdo, aiecompiler uses DevInst->StartCol = 1
*		and Loc here will have Loc.Col = 0;
*		So we essentially configure 0, 1 for all columns (when cdo is reused).
*
*		Todo:
*		Call to XAie_BacktrackErrorInterrupts() needs to have correct
*		DevInst->StartCol. Currently it has 0. The IrqId is used to
*		determine the L1 column from L2 col, L2 mask. DevInst->StartCol
*		Should also be used to determine L1 col, Switch. This way, we can
*		make cdo relocatable.
*
*		Internal Only.
******************************************************************************/
u8 _XAie2Ipu_IntrCtrlL1IrqId(XAie_DevInst *DevInst, XAie_LocType Loc,
		XAie_BroadcastSw Switch)
{
	u8 TileType = DevInst->DevOps->GetTTypefromLoc(DevInst, Loc);

	if (TileType != XAIEGBL_TILE_TYPE_SHIMNOC) {
		return (u8)Switch;
	} else {
		if (Loc.Col  == 1) {
			/* Shim PL on the left */
			return 2 + (u8)Switch;
		} else {
			return (u8)Switch;
		}
	}
}

#endif /* XAIE_FEATURE_INTR_INIT_ENABLE */

/** @} */
