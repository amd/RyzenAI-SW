/******************************************************************************
* Copyright (C) 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_lite_shim_aie.h
* @{
*
* This header file defines a lite shim interface for AIE type devices.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Nishad   06/23/2022  Initial creation
* </pre>
*
******************************************************************************/
#ifndef XAIE_LITE_SHIM_AIE_H
#define XAIE_LITE_SHIM_AIE_H

/***************************** Include Files *********************************/
#include "xaie_lite_hwcfg.h"
#include "xaiegbl_defs.h"
#include "xaiegbl.h"

/************************** Constant Definitions *****************************/
#define UPDT_NEXT_NOC_TILE_LOC(Loc)			\
({							\
	if ((Loc).Col <= 1)				\
		(Loc).Col = 2;				\
	else if ((((Loc).Col + 1) % 4) / 2)		\
		(Loc).Col += 1;				\
	else if ((((Loc).Col + 2) % 4) / 2)		\
		(Loc).Col += 2;				\
	else						\
		(Loc).Col += 3;				\
})

#define IS_TILE_NOC_TILE(Loc)				\
({							\
	 ((Loc).Col % 4) / 2 ? 1: 0;			\
})

#define XAIE_MAX_NUM_NOC_INTR				3U

/************************** Function Prototypes  *****************************/
/*****************************************************************************/
/**
*
* This is API returns the shim tile type for a given device instance and tile
* location.
*
* @param	DevInst: Device Instance
* @param	Loc: Location of the AIE tile.
*
* @return	TileType SHIMPL/SHIMNOC on success.
*
* @note		Internal only.
*
******************************************************************************/
static inline u8 _XAie_LGetShimTTypefromLoc(XAie_DevInst *DevInst,
			XAie_LocType Loc)
{
	u8 ColType = (DevInst->StartCol + Loc.Col) % 4U;

	if((ColType == 0U) || (ColType == 1U))
		return XAIEGBL_TILE_TYPE_SHIMPL;

	return XAIEGBL_TILE_TYPE_SHIMNOC;
}

/*****************************************************************************/
/**
*
* This API maps L2 status bit to its L1 switch.
*
* @param	DevInst: Device Instance.
* @param	Index: Set bit position in L2 status.
* @param	L2Col: Location of L2 column.
* @param	L1Col: Mapped value of L1 column.
* @param	Switch: Broadcast switch.
*
* @return	None.
*
* @note		Internal only.
*
******************************************************************************/
static inline void _XAie_MapL2MaskToL1(XAie_DevInst *DevInst, u32 Index,
			u8 L2Col, u8 *L1Col, XAie_BroadcastSw *Switch)
{
	if (L2Col + 3 >=  DevInst->NumCols) {
	        *L1Col = L2Col + (Index % 6) / 2;
	        *Switch = (Index % 6) % 2;
	} else if ((L2Col) % 2 == 0) {
	        /* Set bit position could be 0 - 5 */
	        *L1Col = L2Col - (2 - (Index % 6) / 2);
	        *Switch = (Index % 6) % 2;
	} else {
	        /* Set bit position could be 0 - 1 */
	        *L1Col = L2Col;
	        *Switch= Index;
	}
}

/*****************************************************************************/
/**
*
* This is API returns the range of columns programmed to generate interrupt on
* the given IRQ channel.
*
* @param	IrqId: L2 IRQ ID.
*
* @return	Range of columns.
*
* @note		Internal only.
*
******************************************************************************/
static inline XAie_Range _XAie_MapIrqIdToCols(u8 IrqId)
{
	XAie_Range _MapIrqIdToCols[] = {
		{.Start = 0, .Num = 16},
		{.Start = 16, .Num = 16},
		{.Start = 32, .Num = 18},
	};

	return _MapIrqIdToCols[IrqId];
}

/*****************************************************************************/
/**
*
* This is API returns the L2 IRQ ID for a given column.
*
* @param	DevInst: Device Instance
* @param	Loc: Location of the AIE tile.
*
* @return	L2 IRQ ID.
*
* @note		Internal only.
*
******************************************************************************/
static inline u8 _XAie_MapColToIrqId(XAie_DevInst *DevInst, XAie_LocType Loc)
{
	u8 IrqId, AbsCol;

	AbsCol = DevInst->StartCol + Loc.Col;
	IrqId = AbsCol / (XAIE_NUM_COLS / XAIE_MAX_NUM_NOC_INTR) +
		XAIE_NUM_NOC_INTR_OFFSET;

	if (AbsCol + 3 > XAIE_NUM_COLS) {
		IrqId -= 1;
	}

	return IrqId;
}

#endif		/* end of protection macro */
/** @} */
