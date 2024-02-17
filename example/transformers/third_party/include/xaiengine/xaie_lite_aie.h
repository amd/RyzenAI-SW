/******************************************************************************
* Copyright (C) 2021 - 2022 Xilinx, Inc.  All rights reserved.
* Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_lite_aie.h
* @{
*
* This header file defines a lightweight version of AIE specific register
* operations.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Wendy   09/06/2021  Initial creation
* </pre>
*
******************************************************************************/
#ifndef XAIE_LITE_AIE_H
#define XAIE_LITE_AIE_H

/***************************** Include Files *********************************/
#include "xaie_lite_hwcfg.h"
#include "xaie_lite_io.h"
#include "xaie_lite_npi.h"
#include "xaie_lite_regdef_aie.h"
#include "xaiegbl_defs.h"
#include "xaiegbl.h"
#include "xaie_lite_util.h"

/************************** Constant Definitions *****************************/
/************************** Function Prototypes  *****************************/
/*****************************************************************************/
/**
*
* This API checks if an AI engine tile is in use.
*
* @param	DevInst: Device Instance.
* @param	Loc: Tile location.
*
* @return	XAIE_ENABLE if a tile is in use, otherwise XAIE_DISABLE.
*
* @note		Internal only.
*
******************************************************************************/
static inline u8 _XAie_LPmIsTileRequested(XAie_DevInst *DevInst,
		XAie_LocType Loc)
{
	(void) DevInst;
	(void) Loc.Col;
	(void) Loc.Row;

	/* TODO: Implement lite API to scan AIE array and update bitmap */
	return XAIE_ENABLE;
}

/*****************************************************************************/
/**
*
* This API set SHIM reset in the AI engine partition
*
* @param	DevInst: Device Instance
* @param	Loc: SHIM tile location
* @param	Reset: XAIE_ENABLE to enable reset,
*			XAIE_DISABLE to disable reset
*
* @return	XAIE_OK for success, and error value for failure
*
* @note		This function is internal.
*
******************************************************************************/
static inline void _XAie_LSetPartColShimReset(XAie_DevInst *DevInst,
		XAie_LocType Loc, u8 Reset)
{
	u64 RegAddr;
	u32 FldVal;

	RegAddr = _XAie_LGetTileAddr(0, Loc.Row) +
		XAIE_PL_MOD_SHIM_RST_ENA_REGOFF;
	FldVal = XAie_SetField(Reset, XAIE_PL_MOD_SHIM_RST_ENA_LSB,
			XAIE_PL_MOD_SHIM_RST_ENA_MASK);
	_XAie_LPartWrite32(DevInst, RegAddr, FldVal);
}

/*****************************************************************************/
/**
*
* This API sets isolation boundry of an AI engine partition after reset
*
* @param	DevInst: Device Instance
*
* @note		Internal API only.
*
******************************************************************************/
static inline void _XAie_LSetPartIsolationAfterRst(XAie_DevInst *DevInst)
{
	for(u8 C = 0; C < DevInst->NumCols; C++) {
		u64 RegAddr;
		u32 RegVal = 0;

		if(C == 0) {
			RegVal = XAIE_TILE_CNTR_ISOLATE_WEST_MASK;
		} else if(C == (u8)(DevInst->NumCols - 1)) {
			RegVal = XAIE_TILE_CNTR_ISOLATE_EAST_MASK;
		} else {
			/* No isolation for tiles by default for AIE */
			continue;
		}

		/* Isolate boundrary of SHIM tiles */
		RegAddr = _XAie_LGetTileAddr(0, C) +
			XAIE_PL_MOD_TILE_CNTR_REGOFF;
		_XAie_LPartWrite32(DevInst, RegAddr, RegVal);

		/* Isolate boundrary of CORE tiles */
		for (u8 R = XAIE_AIE_TILE_ROW_START; R < XAIE_NUM_ROWS; R++) {
			RegAddr = _XAie_LGetTileAddr(R, C) +
				XAIE_CORE_MOD_TILE_CNTR_REGOFF;
			_XAie_LPartWrite32(DevInst, RegAddr, RegVal);
		}
	}
}

/*****************************************************************************/
/**
*
* This API initialize the memories of the partition to zero.
*
* @param	DevInst: Device Instance
*
* @return       XAIE_OK on success, error code on failure
*
* @note		Internal API only.
*
******************************************************************************/
static inline void  _XAie_LPartMemZeroInit(XAie_DevInst *DevInst)
{
	for(u8 C = 0; C < DevInst->NumCols; C++) {
		/* Isolate boundrary of CORE tiles */
		for (u8 R = XAIE_AIE_TILE_ROW_START;
			R < XAIE_NUM_ROWS; R++) {
			u64 RegAddr;

			RegAddr = _XAie_LGetTileAddr(R, C) +
				XAIE_CORE_MOD_PMEM_START_ADDR;
			_XAie_LPartBlockSet32(DevInst, RegAddr, 0,
				XAIE_CORE_MOD_PMEM_SIZE);

			RegAddr = _XAie_LGetTileAddr(R, C) +
				XAIE_MEM_MOD_DMEM_START_ADDR;
			_XAie_LPartBlockSet32(DevInst, RegAddr, 0,
				XAIE_MEM_MOD_DMEM_SIZE);
		}
	}
}

/*****************************************************************************/
/**
*
* This API checks if all the Tile DMA channels in a partition are idle.
*
* @param	DevInst: Device Instance
*
* @return       XAIE_OK if all channels are idle, XAIE_ERR otherwise.
*
* @note		Internal API only.
*
******************************************************************************/
static inline AieRC _XAie_LPartIsDmaIdle(XAie_DevInst *DevInst)
{
	for(u8 C = 0; C < DevInst->NumCols; C++) {
		u64 RegAddr;
		u32 RegVal;

		/* AIE TILE DMAs */
		for(u8 R = XAIE_AIE_TILE_ROW_START; R < XAIE_NUM_ROWS; R++) {
			/* S2MM Channel */
			RegAddr = _XAie_LGetTileAddr(R, C) +
				XAIE_TILE_DMA_S2MM_CHANNEL_STATUS_REGOFF;
			RegVal = _XAie_LPartRead32(DevInst, RegAddr);
			if(RegVal & (XAIE_TILE_DMA_S2MM_CHANNEL_STATUS_0_MASK |
						XAIE_TILE_DMA_S2MM_CHANNEL_STATUS_1_MASK))
				return XAIE_ERR;

			/* MM2S Channel */
			RegAddr = _XAie_LGetTileAddr(R, C) +
				XAIE_TILE_DMA_MM2S_CHANNEL_STATUS_REGOFF;
			RegVal = _XAie_LPartRead32(DevInst, RegAddr);
			if(RegVal & (XAIE_TILE_DMA_MM2S_CHANNEL_STATUS_0_MASK |
						XAIE_TILE_DMA_MM2S_CHANNEL_STATUS_1_MASK))
				return XAIE_ERR;

		}
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This API checks if all the DMA channels in a SHIM NOC tile are idle.
*
* @param	DevInst: Device Instance
* @param	Loc: ShimDma location
*
* @return       XAIE_OK if all channels are idle, XAIE_ERR otherwise.
*
* @note		Internal API only. Checks for AIE Tile DMAs and Mem Tile DMAs
*
******************************************************************************/
static inline AieRC _XAie_LIsShimDmaIdle(XAie_DevInst *DevInst,
		XAie_LocType Loc)
{
	u64 RegAddr;
	u32 RegVal;

	/* S2MM Channel */
	RegAddr = _XAie_LGetTileAddr(0, Loc.Col) +
		XAIE_SHIM_DMA_S2MM_CHANNEL_STATUS_REGOFF;
	RegVal = _XAie_LPartRead32(DevInst, RegAddr);
	if(RegVal & (XAIE_SHIM_DMA_S2MM_CHANNEL_STATUS_0_MASK |
				XAIE_SHIM_DMA_S2MM_CHANNEL_STATUS_1_MASK))
		return XAIE_ERR;

	/* MM2S Channel */
	RegAddr = _XAie_LGetTileAddr(0, Loc.Col) +
		XAIE_SHIM_DMA_MM2S_CHANNEL_STATUS_REGOFF;
	RegVal = _XAie_LPartRead32(DevInst, RegAddr);
	if(RegVal & (XAIE_SHIM_DMA_MM2S_CHANNEL_STATUS_0_MASK |
				XAIE_SHIM_DMA_MM2S_CHANNEL_STATUS_1_MASK))
		return XAIE_ERR;

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is function to setup the protected register configuration value.
*
* @param	DevInst : AI engine partition device pointer
* @param	Enable: Enable partition
*
* @note		None
*
*******************************************************************************/
static inline void _XAie_LNpiSetPartProtectedReg(XAie_DevInst *DevInst,
		u8 Enable)
{
	u32 RegVal;

	(void)DevInst;
	RegVal = XAie_SetField(Enable, XAIE_NPI_PROT_REG_CNTR_EN_LSB,
			       XAIE_NPI_PROT_REG_CNTR_EN_MSK);

	_XAie_LNpiSetLock(XAIE_DISABLE);
	_XAie_LNpiWriteCheck32(XAIE_NPI_PROT_REG_CNTR_REG, RegVal);
	_XAie_LNpiSetLock(XAIE_ENABLE);
}

/*****************************************************************************/
/**
*
* This API initialize the data memory to zero.
*
* @param	DevInst: Device Instance
*
* @return	XAIE_OK on success, error code on failure
*
* @note		None
*
******************************************************************************/
static inline AieRC _XAie_LPartDataMemZeroInit(XAie_DevInst *DevInst)
{
	for(u8 C = 0; C < DevInst->NumCols; C++) {
		/* Isolate boundrary of CORE tiles */
		for (u8 R = XAIE_AIE_TILE_ROW_START;
			R < XAIE_NUM_ROWS; R++) {
			u64 RegAddr;

			RegAddr = _XAie_LGetTileAddr(R, C) +
				XAIE_MEM_MOD_DMEM_START_ADDR;
			_XAie_LPartBlockSet32(DevInst, RegAddr, 0,
				XAIE_MEM_MOD_DMEM_SIZE);
		}
	}
	return XAIE_OK;
}

#endif		/* end of protection macro */
/** @} */
