/******************************************************************************
* Copyright (C) 2021 - 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_lite_shim_aie2ipu.h
* @{
*
* This header file defines a lite shim interface for AIE2IPU type devices.
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
#ifndef XAIE_LITE_SHIM_AIE2IPU_H
#define XAIE_LITE_SHIM_AIE2IPU_H

/***************************** Include Files *********************************/
#include "xaie_lite_hwcfg.h"
#include "xaiegbl_defs.h"
#include "xaiegbl.h"

/************************** Constant Definitions *****************************/
#define UPDT_NEXT_NOC_TILE_LOC(Loc)	\
({					\
	if ((Loc).Col == 0)		\
		(Loc).Col = 1;		\
	else				\
		(Loc).Col++;		\
})

#define IS_TILE_NOC_TILE(Loc)		\
({					\
	(Loc).Col ? 1: 0;		\
})

#define XAIE_MAX_NUM_NOC_INTR		4U

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
* @return	TileType SHIMPL/SHIMNOC.
*
* @note		Internal only.
*
******************************************************************************/
static inline u8 _XAie_LGetShimTTypefromLoc(XAie_DevInst *DevInst,
			XAie_LocType Loc)
{
	if((DevInst->StartCol + Loc.Col) == 0U)
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
* 		For IPU, Col < 2, Index 0 to 3 is valid.
* 		For Col >= 2, Index 0 to 1 is valid.
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
	(void) DevInst;

	if (L2Col < 2) {
		*L1Col = Index / 2;
		*Switch = Index % 2;
	} else {
	        *L1Col = L2Col;
	        *Switch= Index % 2;
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
		{.Start = 0, .Num = 2},
		{.Start = 2, .Num = 1},
		{.Start = 3, .Num = 1},
		{.Start = 4, .Num = 1},
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
	u8 AbsCol = DevInst->StartCol + Loc.Col;

	if (AbsCol < 2) {
		return 0;
	} else {
		return AbsCol - 1;
	}
}

#endif		/* end of protection macro */
/** @} */
