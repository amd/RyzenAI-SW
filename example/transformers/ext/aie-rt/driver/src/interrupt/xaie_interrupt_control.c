/******************************************************************************
* Copyright (C) 2021 - 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_interrupt_control.c
* @{
*
* This file implements routine for disabling AIE interrupts.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Nishad  08/20/2021  Initial creation
* </pre>
*
******************************************************************************/

/***************************** Include Files *********************************/
#include "xaie_feature_config.h"
#include "xaie_interrupt.h"
#include "xaie_lite.h"
#include "xaie_lite_io.h"

#if defined(XAIE_FEATURE_INTR_CTRL_ENABLE) && defined(XAIE_FEATURE_LITE)

/************************** Constant Definitions *****************************/
/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
*
* This API returns the status of second-level interrupt controller.
*
* @param	Loc: Location of AIE tile.
*
* @return	Status: Status second-level interrupt controller.
*
* @note		Internal only.
*
******************************************************************************/
static inline u32 _XAie_LIntrCtrlL2Status(XAie_LocType Loc)
{
	u64 RegAddr = _XAie_LGetTileAddr(Loc.Row, Loc.Col) +
				XAIE_NOC_MOD_INTR_L2_STATUS;
	return _XAie_LRead32(RegAddr);
}

/*****************************************************************************/
/**
*
* This API clears the status of interrupts in the second-level interrupt
* controller.
*
* @param	Loc: Location of AIE tile.
* @param	ChannelBitMap: Bitmap of channels to be acknowledged. Writing a
*				value of 1 to the register field clears the
*				corresponding interrupt channel.
*
* @return	None.
*
* @note		Internal only.
*
******************************************************************************/
static inline void _XAie_LIntrCtrlL2Ack(XAie_LocType Loc, u32 ChannelBitMap)
{
	u64 RegAddr = _XAie_LGetTileAddr(Loc.Row, Loc.Col) +
				XAIE_NOC_MOD_INTR_L2_STATUS;
	_XAie_LWrite32(RegAddr, ChannelBitMap);
}

/*****************************************************************************/
/**
*
* This API disables interrupts to second level interrupt controller.
*
* @param	Loc: Location of AIE Tile
* @param	ChannelBitMap: Interrupt Bitmap.
*
* @return	None.
*
* @note		Internal Only.
*
******************************************************************************/
static inline void _XAie_LIntrCtrlL2Disable(XAie_LocType Loc, u32 ChannelBitMap)
{
	u64 RegAddr = _XAie_LGetTileAddr(Loc.Row, Loc.Col) +
				XAIE_NOC_MOD_INTR_L2_DISABLE;
	_XAie_LWrite32(RegAddr, ChannelBitMap);
}

/*****************************************************************************/
/**
*
* This API disables all second-level interrupt controllers reporting errors.
*
* @param	IrqId: Zero indexed IRQ ID. Valid values corresponds to the
*		       number of AIE IRQs mapped to the processor.
*
* @return	None.
*
* @note		None.
*
******************************************************************************/
void XAie_DisableErrorInterrupts(u8 IrqId)
{
	XAie_Range Cols;

	XAie_MapIrqIdToCols(IrqId, &Cols);

	XAie_LocType Loc = XAie_TileLoc(Cols.Start, XAIE_SHIM_ROW);

	if (!IS_TILE_NOC_TILE(Loc)) {
		UPDT_NEXT_NOC_TILE_LOC(Loc);
	}

	while (Loc.Col < Cols.Start + Cols.Num) {
		u32 Status;

		Status = _XAie_LIntrCtrlL2Status(Loc);

		/* Only disable L2s that are reporting errors. */
		if (Status) {
			_XAie_LIntrCtrlL2Disable(Loc, Status);
			_XAie_LIntrCtrlL2Ack(Loc, Status);
		}

		UPDT_NEXT_NOC_TILE_LOC(Loc);
	}
}

#endif /* XAIE_FEATURE_INTR_CTRL_ENABLE */

/** @} */
