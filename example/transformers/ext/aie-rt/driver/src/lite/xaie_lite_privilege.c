/******************************************************************************
* Copyright (C) 2021 - 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_lite_privilege.c
* @{
*
* This file contains lite version of AI engine partition management operations.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date        Changes
* ----- ------  --------    ---------------------------------------------------
* 1.0   Wendy 05/17/2021  Initial creation
*
* </pre>
*
******************************************************************************/
/***************************** Include Files *********************************/

#include "xaie_feature_config.h"

#if defined(XAIE_FEATURE_PRIVILEGED_ENABLE) && defined(XAIE_FEATURE_LITE)

#include "xaie_lite.h"
#include "xaie_lite_io.h"
#include "xaie_lite_internal.h"
#include "xaiegbl_defs.h"
#include "xaiegbl.h"
#include "xaie_helper.h"
#include "xaie_lite_internal.h"

/***************************** Macro Definitions *****************************/
#define XAIE_ISOLATE_EAST_MASK	(1U << 3)
#define XAIE_ISOLATE_NORTH_MASK	(1U << 2)
#define XAIE_ISOLATE_WEST_MASK	(1U << 1)
#define XAIE_ISOLATE_SOUTH_MASK	(1U << 0)
#define XAIE_ISOLATE_ALL_MASK	((1U << 4) - 1)

/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
* This API sets the column clock control register. Its configuration affects
* (enable or disable) all tile's clock above the Shim tile.
*
* @param        DevInst: Device Instance
* @param        Loc: Location of AIE SHIM tile
* @param        Enable: XAIE_ENABLE to enable column global clock buffer,
*                       XAIE_DISABLE to disable.

* @note         It is internal function to this file
*
******************************************************************************/
static void _XAie_PrivilegeSetColClkBuf(XAie_DevInst *DevInst,
		XAie_LocType Loc, u8 Enable)
{
	u64 RegAddr;
	u32 FldVal;

	RegAddr = _XAie_LGetTileAddr(Loc.Row, Loc.Col) +
		XAIE_PL_MOD_COL_CLKCNTR_REGOFF;
	FldVal = XAie_SetField(Enable,
			XAIE_PL_MOD_COL_CLKCNTR_CLKBUF_ENABLE_LSB,
			XAIE_PL_MOD_COL_CLKCNTR_CLKBUF_ENABLE_MASK);

	_XAie_LPartMaskWrite32(DevInst, RegAddr,
		XAIE_PL_MOD_COL_CLKCNTR_CLKBUF_ENABLE_MASK, FldVal);
}

/*****************************************************************************/
/**
*
* This API set the tile columns clock buffer for every column in the partition
*
* @param	DevInst: Device Instance
* @param	Enable: XAIE_ENABLE to enable clock buffers, XAIE_DISABLE to
*			  disable clock buffers.
*
* @note		This function is internal to this file.
*
******************************************************************************/
static void _XAie_PrivilegeSetPartColClkBuf(XAie_DevInst *DevInst,
		u8 Enable)
{
	for(u32 C = 0; C < DevInst->NumCols; C++) {
		XAie_LocType Loc = XAie_TileLoc(C, 0);

		_XAie_PrivilegeSetColClkBuf(DevInst, Loc, Enable);
	}
}

/*****************************************************************************/
/**
*
*  This API modifies column clock and shim clk control registers for the requested columns.
*  Caller may request subset of columns from a partition.
*
* @param        DevInst: Device Instance
* @param        StartCol: Starting column
* @param        NumCols: Number of columns
* @param        Enable: Enable/Disable
*
* @return       XAIE_OK
******************************************************************************/
AieRC XAie_SetColumnClk(XAie_DevInst *DevInst, u32 StartCol, u32 NumCols, u8 Enable)
{
	u32 PartEndCol = DevInst->StartCol + DevInst->NumCols -1;

	if((StartCol < DevInst->StartCol) || (StartCol > PartEndCol) ||
	   ((StartCol + NumCols - 1) > PartEndCol) ) {

		XAIE_ERROR_MSG("Invalid Start Column/Numcols \n");
		return XAIE_ERR;
	   }

	_XAie_LNpiSetPartProtectedReg(DevInst, XAIE_ENABLE);
	for(u32 C = StartCol; C < (StartCol + NumCols); C++) {
		XAie_LocType Loc = XAie_TileLoc(C, 0);

		//This modifies column clk which affects all the aie and mem tile in the column.
		_XAie_PrivilegeSetColClkBuf(DevInst, Loc, Enable);
		//This modifies the clk in the shim tile
		_XAie_PrivilegeSetShimClk(DevInst, Loc, Enable);
	}
	_XAie_LNpiSetPartProtectedReg(DevInst, XAIE_DISABLE);

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This API set the tile column reset
*
* @param	DevInst: Device Instance
* @param	Loc: Location of AIE SHIM tile
* @param	RstEnable: XAIE_ENABLE to assert reset, XAIE_DISABLE to
*			   deassert reset.
*
* @note		This function is internal to this file.
*
******************************************************************************/
static void _XAie_PrivilegeSetColReset(XAie_DevInst *DevInst,
		XAie_LocType Loc, u8 RstEnable)
{
	u32 FldVal;
	u64 RegAddr;

	RegAddr = _XAie_LGetTileAddr(Loc.Row, Loc.Col) +
		XAIE_PL_MOD_COL_RST_REGOFF;
	FldVal = XAie_SetField(RstEnable, XAIE_PL_MOD_COL_RST_LSB,
			XAIE_PL_MOD_COL_RST_MASK);

	_XAie_LPartWrite32(DevInst, RegAddr, FldVal);
}

/*****************************************************************************/
/**
*
* This API set the tile columns reset for every column in the partition
*
* @param	DevInst: Device Instance
* @param	RstEnable: XAIE_ENABLE to assert reset, XAIE_DISABLE to
*			   deassert reset.
*
* @note		This function is internal to this file.
*
******************************************************************************/
static void _XAie_PrivilegeSetPartColRst(XAie_DevInst *DevInst, u8 RstEnable)
{
	for(u32 C = 0; C < DevInst->NumCols; C++) {
		XAie_LocType Loc = XAie_TileLoc(C, 0);

		_XAie_PrivilegeSetColReset(DevInst, Loc, RstEnable);
	}
}

/*****************************************************************************/
/**
*
* This API reset all SHIMs in the AI engine partition
*
* @param	DevInst: Device Instance
*
* @return	XAIE_OK for success, and error value for failure
*
* @note		This function asserts reset, and then deassert it.
*		This function is internal to this file.
*
******************************************************************************/
static void _XAie_PrivilegeRstPartShims(XAie_DevInst *DevInst)
{
	for(u32 C = 0; C < DevInst->NumCols; C++) {
		XAie_LocType Loc = XAie_TileLoc(C, 0);

		_XAie_LSetPartColShimReset(DevInst, Loc, XAIE_ENABLE);
	}

	_XAie_LNpiSetShimReset(XAIE_ENABLE);

	_XAie_LNpiSetShimReset(XAIE_DISABLE);

	for(u32 C = 0; C < DevInst->NumCols; C++) {
		XAie_LocType Loc = XAie_TileLoc(C, 0);

		_XAie_LSetPartColShimReset(DevInst, Loc, XAIE_DISABLE);
	}
}

/*****************************************************************************/
/**
*
* This API sets to block NSU AXI MM slave error and decode error based on user
* inputs. If NSU errors is blocked, it will only generate error events.
*
* @param	DevInst: Device Instance
* @param	Loc: Location of AIE SHIM tile
* @param	BlockSlvEnable: XAIE_ENABLE to block NSU AXI MM Slave errors,
*				or XAIE_DISABLE to unblock.
* @param	BlockDecEnable: XAIE_ENABLE to block NSU AXI MM Decode errors,
*				or XAIE_DISABLE to unblock.
*
* @note		This function is internal to this file.
*
******************************************************************************/
static void _XAie_PrivilegeSetBlockAxiMmNsuErr(XAie_DevInst *DevInst,
		XAie_LocType Loc, u8 BlockSlvEnable, u8 BlockDecEnable)
{
	u32 FldVal;
	u64 RegAddr;

	RegAddr = XAIE_NOC_AXIMM_CONF_REGOFF +
		_XAie_LGetTileAddr(Loc.Row, Loc.Col);
	FldVal = XAie_SetField(BlockSlvEnable,
			XAIE_NOC_AXIMM_CONF_SLVERR_BLOCK_LSB,
			XAIE_NOC_AXIMM_CONF_SLVERR_BLOCK_MASK);
	FldVal |= XAie_SetField(BlockDecEnable,
			XAIE_NOC_AXIMM_CONF_DECERR_BLOCK_LSB,
			XAIE_NOC_AXIMM_CONF_DECERR_BLOCK_MASK);

	_XAie_LPartWrite32(DevInst, RegAddr, FldVal);
}

/*****************************************************************************/
/**
*
* This API sets to block the NSU AXI MM slave error and decode error config
* for all the SHIM NOCs in the full partition based on user inputs.
*
* @param	DevInst: Device Instance
* @param	BlockSlvEnable: XAIE_ENABLE to block NSU AXI MM Slave errors,
*				or XAIE_DISABLE to unblock.
* @param	BlockDecEnable: XAIE_ENABLE to block NSU AXI MM Decode errors,
*				or XAIE_DISABLE to unblock.
*
* @note		This function will do the following steps:
*		 * set AXI MM registers NSU errors fields in all SHIM NOC tiles
*		This function is internal to this file.
*
******************************************************************************/
static void _XAie_PrivilegeSetPartBlockAxiMmNsuErr(XAie_DevInst *DevInst,
		u8 BlockSlvEnable, u8 BlockDecEnable)
{
	XAie_LocType Loc = XAie_LPartGetNextNocTile(DevInst,
			XAie_TileLoc(0, 0));

	for(; Loc.Col < DevInst->NumCols;
		Loc = XAie_LPartGetNextNocTile(DevInst, Loc)) {
		_XAie_PrivilegeSetBlockAxiMmNsuErr(DevInst, Loc,
				BlockSlvEnable, BlockDecEnable);
	}
}

/*****************************************************************************/
/**
*
* This API sets NoC interrupt ID to which the interrupt from second level
* interrupt controller shall be driven to.
*
* @param	DevInst: Device Instance
* @param	Loc: Location of AIE Tile
* @param	NoCIrqId: NoC IRQ index on which the interrupt shall be
*			  driven.
*
* @note		None.
*
******************************************************************************/
static void _XAie_PrivilegeSetL2IrqId(XAie_DevInst *DevInst, XAie_LocType Loc,
		u8 NoCIrqId)
{
	u64 RegAddr;

	RegAddr = _XAie_LGetTileAddr(Loc.Row, Loc.Col) +
		XAIE_NOC_MOD_INTR_L2_IRQ;
	_XAie_LPartWrite32(DevInst, RegAddr, NoCIrqId);
}

/*****************************************************************************/
/**
*
* This API sets NoC interrupt ID to which the error interrupts from second
* level interrupt controller shall be driven to. All the second level interrupt
* controllers with a given partition are configured.
*
* @param	DevInst: Device Instance
*
* @note		None.
*
******************************************************************************/
static void _XAie_PrivilegeSetL2ErrIrq(XAie_DevInst *DevInst)
{
	XAie_LocType Rloc = {0, 0};

	for (;Rloc.Col < DevInst->NumCols; Rloc.Col++) {
		u8 TileType = _XAie_LGetShimTTypefromLoc(DevInst, Rloc);

		if (TileType != XAIEGBL_TILE_TYPE_SHIMNOC)
			continue;
		_XAie_PrivilegeSetL2IrqId(DevInst, Rloc,
				_XAie_MapColToIrqId(DevInst, Rloc));
	}
}

/*****************************************************************************/
/**
* This API initializes the AI engine partition
*
* @param	DevInst: AI engine partition device instance pointer
* @param	Opts: Initialization options
*
* @return       XAIE_OK
*
* @note		This operation does the following steps to initialize an AI
*		engine partition:
*		- Clock gate all columns
*		- Reset Columns
*		- Ungate all Columns
*		- Remove columns reset
*		- Reset shims
*		- Setup AXI MM not to return errors for AXI decode or slave
*		  errors, raise events instead.
*		- ungate all columns
*		- Setup partition isolation.
*		- zeroize memory if it is requested
*
*******************************************************************************/
AieRC XAie_PartitionInitialize(XAie_DevInst *DevInst, XAie_PartInitOpts *Opts)
{
	u32 OptFlags;

	XAIE_ERROR_RETURN((DevInst == NULL || DevInst->NumCols > XAIE_NUM_COLS),
		XAIE_INVALID_ARGS,
		XAIE_ERROR_MSG("Partition initialization failed, invalid partition instance\n"));

	if(Opts != NULL) {
		OptFlags = Opts->InitOpts;
	} else {
		OptFlags = XAIE_PART_INIT_OPT_DEFAULT;
	}

	_XAie_LNpiSetPartProtectedReg(DevInst, XAIE_ENABLE);

	if((OptFlags & XAIE_PART_INIT_OPT_COLUMN_RST) != 0) {
		/* Always gate all tiles before resetting columns */
		_XAie_PrivilegeSetPartColClkBuf(DevInst, XAIE_DISABLE);
		_XAie_PrivilegeSetPartColRst(DevInst, XAIE_ENABLE);
		_XAie_PrivilegeSetPartColClkBuf(DevInst, XAIE_ENABLE);
		_XAie_PrivilegeSetPartColRst(DevInst, XAIE_DISABLE);
	}

	if((OptFlags & XAIE_PART_INIT_OPT_SHIM_RST) != 0) {
		_XAie_PrivilegeRstPartShims(DevInst);
	}

	if((OptFlags & XAIE_PART_INIT_OPT_BLOCK_NOCAXIMMERR) != 0) {
		_XAie_PrivilegeSetPartBlockAxiMmNsuErr(DevInst,
			XAIE_ENABLE, XAIE_ENABLE);
	}

	_XAie_PrivilegeSetPartColClkBuf(DevInst, XAIE_ENABLE);

	if ((OptFlags & XAIE_PART_INIT_OPT_ISOLATE) != 0) {
		_XAie_LSetPartIsolationAfterRst(DevInst);
	}

	if ((OptFlags & XAIE_PART_INIT_OPT_ZEROIZEMEM) != 0) {
		_XAie_LPartMemZeroInit(DevInst);
	}

	/* Gate all the tiles and ungate the requested tiles*/
	if(Opts != NULL)
	{
		/* Disbale all the column clock and enable only the requested column clock */
		_XAie_PrivilegeSetPartColClkBuf(DevInst, XAIE_DISABLE);

		/* Ungate the tiles that is requested */
		for(u32 i = 0; i < Opts->NumUseTiles; i++)
		{

			if(Opts->Locs[i].Col >= DevInst->NumCols || Opts->Locs[i].Row >= DevInst->NumRows) {
				XAIE_ERROR("Invalid Tile Location\n");
				return XAIE_INVALID_TILE;
			}

			if(Opts->Locs[i].Row == 0) {
				continue;
			}
			/*
			* Check if column clock buffer is already enabled and continue
			*/
			_XAie_PrivilegeSetColClkBuf(DevInst, Opts->Locs[i], XAIE_ENABLE);

		}
	}

	_XAie_PrivilegeSetL2ErrIrq(DevInst);

	_XAie_DisableTlast(DevInst);
	_XAie_LNpiSetPartProtectedReg(DevInst, XAIE_DISABLE);

	return XAIE_OK;
}

/*****************************************************************************/
/**
* This API tears down the AI engine partition
*
* @param	DevInst: AI engine partition device instance pointer
* @param	Opts: Initialization options
*
* @return       XAIE_OK on success, error code on failure
*
* @note		This operation does the following steps to initialize an AI
*		engine partition:
*		- Clock gate all columns
*		- Reset Columns
*		- Ungate all columns
*		- Reset shims
*		- Ungate all columns
*		- Zeroize memories
*		- Clock gate all columns
*
*******************************************************************************/
AieRC XAie_PartitionTeardown(XAie_DevInst *DevInst)
{

	XAIE_ERROR_RETURN((DevInst == NULL || DevInst->NumCols > XAIE_NUM_COLS),
		XAIE_INVALID_ARGS,
		XAIE_ERROR_MSG("Partition teardown failed, invalid partition instance\n"));

	_XAie_LNpiSetPartProtectedReg(DevInst, XAIE_ENABLE);

	_XAie_PrivilegeSetPartColClkBuf(DevInst, XAIE_DISABLE);

	_XAie_PrivilegeSetPartColRst(DevInst, XAIE_ENABLE);

	_XAie_PrivilegeSetPartColClkBuf(DevInst, XAIE_ENABLE);

	_XAie_PrivilegeSetPartColRst(DevInst, XAIE_DISABLE);

	_XAie_PrivilegeRstPartShims(DevInst);

	_XAie_PrivilegeSetPartColClkBuf(DevInst, XAIE_ENABLE);

	_XAie_LPartMemZeroInit(DevInst);

	_XAie_LCertMemZeroInit(DevInst);

	_XAie_PrivilegeSetPartColClkBuf(DevInst, XAIE_DISABLE);

	_XAie_LNpiSetPartProtectedReg(DevInst, XAIE_DISABLE);

	return XAIE_OK;
}

/*****************************************************************************/
/**
* This API checks if all the shim dmas in the paritition are idle.
*
* @param	DevInst: AI engine partition device instance pointer
*
* @return       XAIE_OK on success, error code on failure
*
* @note		Internal only.
*
*******************************************************************************/
static inline AieRC _XAie_LPartIsShimDmaIdle(XAie_DevInst *DevInst)
{
	for(u8 C = 0U; C < DevInst->NumCols; C++) {
		AieRC RC;
		u8 TType;

		TType = _XAie_LGetTTypefromLoc(DevInst, XAie_TileLoc(C, 0));
		if(TType == XAIEGBL_TILE_TYPE_SHIMNOC) {
			RC = _XAie_LIsShimDmaIdle(DevInst, XAie_TileLoc(C, 0));
			if(RC != XAIE_OK)
				return RC;
		}
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
* This API check if all the resources in the partition are idle. Currently,
* we check only for dma status.
*
* @param	DevInst: AI engine partition device instance pointer
*
* @return       XAIE_OK on success, error code on failure
*
* @note		None.
*
*******************************************************************************/
AieRC XAie_IsPartitionIdle(XAie_DevInst *DevInst)
{
	AieRC RC = XAIE_OK;

	XAIE_ERROR_RETURN((DevInst == NULL || DevInst->NumCols > XAIE_NUM_COLS),
		XAIE_INVALID_ARGS,
		"Partition idle check failed, invalid partition instance\n");

	RC = _XAie_LPartIsDmaIdle(DevInst);
	XAIE_ERROR_RETURN(RC == XAIE_OK, RC, "DMAs are not idle\n");

	RC = _XAie_LPartIsShimDmaIdle(DevInst);
	XAIE_ERROR_RETURN(RC == XAIE_OK, RC, "Shim DMA is not idle\n");

	return RC;
}

/*****************************************************************************/
/**
* This API resets all modules, clears registers needed for context switching.
*
* @param	DevInst: AI engine partition device instance pointer
*
* @return       XAIE_OK on success, error code on failure
*
* @note		None.
*
*******************************************************************************/
AieRC XAie_ClearPartitionContext(XAie_DevInst *DevInst)
{
	AieRC RC;

	_XAie_LNpiSetPartProtectedReg(DevInst, XAIE_ENABLE);

	_XAie_PrivilegeSetPartColClkBuf(DevInst, XAIE_DISABLE);
	_XAie_PrivilegeSetPartColRst(DevInst, XAIE_ENABLE);
	_XAie_PrivilegeSetPartColClkBuf(DevInst, XAIE_ENABLE);
	_XAie_PrivilegeSetPartColRst(DevInst, XAIE_DISABLE);

	_XAie_PrivilegeRstPartShims(DevInst);

	_XAie_PrivilegeSetPartBlockAxiMmNsuErr(DevInst, XAIE_ENABLE,
			XAIE_ENABLE);

	_XAie_PrivilegeSetPartColClkBuf(DevInst, XAIE_ENABLE);

	_XAie_LSetPartIsolationAfterRst(DevInst);

	RC = _XAie_LPartDataMemZeroInit(DevInst);
	if (RC != XAIE_OK)
		return RC;

	_XAie_PrivilegeSetL2ErrIrq(DevInst);

	_XAie_LNpiSetPartProtectedReg(DevInst, XAIE_DISABLE);

	return XAIE_OK;
}

#endif /* XAIE_FEATURE_PRIVILEGED_ENABLE && XAIE_FEATURE_LITE */
/** @} */
