/******************************************************************************
* Copyright (C) 2019 - 2020 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_clock.c
* @{
*
* This file contains routines for AIE timers
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date        Changes
* ----- ------  --------    ---------------------------------------------------
* 1.0   Dishita 06/26/2020  Initial creation
*
* </pre>
*
******************************************************************************/
/***************************** Include Files *********************************/
#include "xaie_clock.h"
#include "xaie_reset_aie.h"
#include "xaie_feature_config.h"
#include "xaie_helper.h"

#ifdef XAIE_FEATURE_PRIVILEGED_ENABLE

/*****************************************************************************/
/***************************** Macro Definitions *****************************/
/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
* This API enables clock for all the tiles passed as argument to this API.
*
* @param	DevInst: Device Instance
* @param	Loc: Location of AIE tile
* @param	NumTiles: Number of tiles requested for use.
*
* @return	XAIE_OK on success.
*
* @note		As all the tiles are gated when the system boots, this function
*		needs to be called after device instance is initialized and
*		before any other AI engine operations. Otherwise, the other
*		AI engine functions may access gated tiles.
*
*******************************************************************************/
AieRC XAie_PmRequestTiles(XAie_DevInst *DevInst, XAie_LocType *Loc,
		u32 NumTiles)
{
	XAie_BackendTilesArray TilesArray;

	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	if(NumTiles > (u32)(DevInst->NumRows * DevInst->NumCols)) {
		XAIE_ERROR("Invalid NumTiles\n");
		return XAIE_INVALID_ARGS;
	}

	if (NumTiles != 0U && Loc == NULL) {
		XAIE_ERROR("NumTiles is not 0, but Location array is empty.\n");
		return XAIE_INVALID_ARGS;
	}

	/* Check validity of all tiles in the list passed to this API */
	for(u32 j = 0; j < NumTiles; j++) {
		if(Loc[j].Row >= DevInst->NumRows ||
			Loc[j].Col >= DevInst->NumCols) {
			XAIE_ERROR("Invalid Loc Col:%d Row:%d\n", Loc[j].Col, Loc[j].Row);
			return XAIE_INVALID_ARGS;
		}
	}

	TilesArray.NumTiles = NumTiles;
	TilesArray.Locs = Loc;

	return XAie_RunOp(DevInst, XAIE_BACKEND_OP_REQUEST_TILES,
			(void *)&TilesArray);
}

/*****************************************************************************/
/**
*
* This API checks if an AI engine tile is in use.
*
* @param	DevInst: Device Instance
* @param	Loc: tile location
*
* @return	XAIE_ENABLE if a tile is in use, otherwise XAIE_DISABLE.
*
* @note		This API is supposed to be called internal only.
*******************************************************************************/
u8 _XAie_PmIsTileRequested(XAie_DevInst *DevInst, XAie_LocType Loc)
{
	u8 TileType;
	u32 TileBit;

	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_DISABLE;
	}

	TileType = DevInst->DevOps->GetTTypefromLoc(DevInst, Loc);
	if (TileType == XAIEGBL_TILE_TYPE_MAX) {
		return XAIE_DISABLE;
	}

	if (TileType == XAIEGBL_TILE_TYPE_SHIMNOC ||
	    TileType == XAIEGBL_TILE_TYPE_SHIMPL) {
		return XAIE_ENABLE;
	}

	TileBit = (u32)(Loc.Col * (DevInst->NumRows - 1U) + Loc.Row - 1U);
	if (CheckBit(DevInst->DevOps->TilesInUse, TileBit)) {
		return XAIE_ENABLE;
	}

	return XAIE_DISABLE;
}

/*****************************************************************************/
/**
* This API enables column clock and module clock control register for requested
* tiles passed as argument to this API.
*
* @param	DevInst: Device Instance
* @param	Loc: Location of AIE tile
* @param	NumTiles: Number of tiles requested for use.
*
* @return	XAIE_OK on success.
*
* @note 	This API enables the clock on runtime
*
*******************************************************************************/
AieRC XAie_PmSetColumnClk(XAie_DevInst *DevInst, u32 StartCol, u32 NumCols,
		u8 Enable)
{
	XAie_BackendColumnReq  ColumnReq;

	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	u32 PartEndCol = (u32)(DevInst->StartCol + DevInst->NumCols - 1U);

	if((StartCol < DevInst->StartCol) || (StartCol > PartEndCol) ||
	   ((StartCol + NumCols - 1U) > PartEndCol) ) {

		XAIE_ERROR("Invalid Start Column/Numcols \n");
		return XAIE_INVALID_ARGS;
	}

	ColumnReq.StartCol = StartCol;
	ColumnReq.NumCols = NumCols;
	ColumnReq.Enable = Enable;

	return XAie_RunOp(DevInst, XAIE_BACKEND_OP_SET_COLUMN_CLOCK,
			(void *)&ColumnReq);
}

#endif /* XAIE_FEATURE_PRIVILEGED_ENABLE */
/** @} */
