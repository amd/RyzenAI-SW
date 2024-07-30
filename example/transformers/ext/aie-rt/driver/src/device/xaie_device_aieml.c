/******************************************************************************
* Copyright (C) 2021 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_device_aieml.c
* @{
*
* This file contains the apis for device specific operations of aieml.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Tejus   05/03/2021  Initial creation
* </pre>
*
******************************************************************************/
/***************************** Include Files *********************************/
#include "xaie_feature_config.h"
#include "xaie_helper.h"
#include "xaie_clock.h"
#include "xaie_reset_aie.h"
#include "xaie_tilectrl.h"
#include "xaiemlgbl_params.h"
#ifdef XAIE_FEATURE_PRIVILEGED_ENABLE
/***************************** Macro Definitions *****************************/
/* set timeout to 1000us. */
#define XAIEML_MEMZERO_POLL_TIMEOUT		1000

/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
*
* This is the function used to get the tile type for a given device instance
* and tile location.
*
* @param	DevInst: Device Instance
* @param	Loc: Location of the AIE tile.
* @return	TileType (AIETILE/MEMTILE/SHIMPL/SHIMNOC on success and MAX on
*		error)
*
* @note		Internal API only. This API returns tile type based on
*		SHIMPL-SHIMPL-SHIMNOC-SHIMNOC pattern
*
******************************************************************************/
u8 _XAieMl_GetTTypefromLoc(XAie_DevInst *DevInst, XAie_LocType Loc)
{
	u8 ColType;

	if(Loc.Col >= DevInst->NumCols) {
		XAIE_ERROR("Invalid column: %d\n", Loc.Col);
		return XAIEGBL_TILE_TYPE_MAX;
	}

	if(Loc.Row == 0U) {
		ColType = (DevInst->StartCol + Loc.Col) % 4U;
		if((ColType == 0U) || (ColType == 1U)) {
			return XAIEGBL_TILE_TYPE_SHIMPL;
		}

		return XAIEGBL_TILE_TYPE_SHIMNOC;

	} else if(Loc.Row >= DevInst->MemTileRowStart &&
			(Loc.Row < (DevInst->MemTileRowStart +
				     DevInst->MemTileNumRows))) {
		return XAIEGBL_TILE_TYPE_MEMTILE;
	} else if (Loc.Row >= DevInst->AieTileRowStart &&
			(Loc.Row < (DevInst->AieTileRowStart +
				     DevInst->AieTileNumRows))) {
		return XAIEGBL_TILE_TYPE_AIETILE;
	}

	XAIE_ERROR("Cannot find Tile Type\n");

	return XAIEGBL_TILE_TYPE_MAX;
}

/*****************************************************************************/
/**
*
* This API sets the reset bit of SHIM for the specified partition.
*
* @param	DevInst: Device Instance
* @param	Enable: Indicate if to enable SHIM reset or disable SHIM reset
*			XAIE_ENABLE to enable SHIM reset, XAIE_DISABLE to
*			disable SHIM reset.
*
* @return	XAIE_OK
*
* @note		Internal API only. Always returns XAIE_OK, as there is nothing
*		need to be done for aieml device.
*
******************************************************************************/
AieRC _XAieMl_SetPartColShimReset(XAie_DevInst *DevInst, u8 Enable)
{
	(void)DevInst;
	(void)Enable;
	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This API sets column clock buffers after SHIM is reset.
*
* @param	DevInst: Device Instance
* @param	Enable: Indicate if to enable clock buffers or disable them.
*			XAIE_ENABLE to enable clock buffers, XAIE_DISABLE to
*			disable.
*
* @return	XAIE_OK for success, and error code for failure
*
* @note		Internal API only.
*
******************************************************************************/
AieRC _XAieMl_SetPartColClockAfterRst(XAie_DevInst *DevInst, u8 Enable)
{
	AieRC RC;

	if(Enable == XAIE_DISABLE) {
		/* Column  clocks are disable by default for aieml device */
		return XAIE_OK;
	}

	RC = _XAie_PmSetPartitionClock(DevInst, XAIE_ENABLE);
	if(RC != XAIE_OK) {
		XAIE_ERROR("Failed to enable clock buffers.\n");
	}

	return RC;
}

/*****************************************************************************/
/**
*
* This API sets isolation boundry of an AI engine partition after reset
*
* @param	DevInst: Device Instance
*
* @return       XAIE_OK on success, error code on failure
*
* @note		It is not required to check the DevInst as the caller function
*		should provide the correct value.
*		Internal API only.
*
******************************************************************************/
AieRC _XAieMl_SetPartIsolationAfterRst(XAie_DevInst *DevInst)
{
	AieRC RC = XAIE_OK;

	for(u8 C = 0; C < DevInst->NumCols; C++) {
		u8 Dir = 0;

		if(C == 0U) {
			Dir = XAIE_ISOLATE_WEST_MASK;
		} else if(C == (u8)(DevInst->NumCols - 1U)) {
			Dir = XAIE_ISOLATE_EAST_MASK;
		}

		for(u8 R = 0; R < DevInst->NumRows; R++) {
			RC = _XAie_TileCtrlSetIsolation(DevInst,
					XAie_TileLoc(C, R), Dir);
			if(RC != XAIE_OK) {
				XAIE_ERROR("Failed to set partition isolation.\n");
				return RC;
			}
		}
	}

	return RC;
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
* @note		It is not required to check the DevInst as the caller function
*		should provide the correct value.
*		Internal API only.
*
******************************************************************************/
AieRC _XAieMl_PartMemZeroInit(XAie_DevInst *DevInst)
{
	AieRC RC = XAIE_OK;
	const XAie_MemCtrlMod *MCtrlMod;
	u64 RegAddr;
	XAie_LocType Loc;

	for(u8 C = 0; C < DevInst->NumCols; C++) {
		for(u8 R = 1; R < DevInst->NumRows; R++) {
			u32 FldVal;
			u8 TileType, NumMods;

			Loc = XAie_TileLoc(C, R);
			TileType = DevInst->DevOps->GetTTypefromLoc(DevInst,
					Loc);
			NumMods = DevInst->DevProp.DevMod[TileType].NumModules;
			MCtrlMod = DevInst->DevProp.DevMod[TileType].MemCtrlMod;
			for (u8 M = 0; M < NumMods; M++) {
				RegAddr = MCtrlMod[M].MemCtrlRegOff +
					_XAie_GetTileAddr(DevInst, R, C);
				FldVal = XAie_SetField(XAIE_ENABLE,
					MCtrlMod[M].MemZeroisation.Lsb,
					MCtrlMod[M].MemZeroisation.Mask);
				RC = XAie_MaskWrite32(DevInst, RegAddr,
					MCtrlMod[M].MemZeroisation.Mask,
					FldVal);
				if(RC != XAIE_OK) {
					XAIE_ERROR("Failed to zeroize partition mems.\n");
					return RC;
				}

				if((C == DevInst->NumCols - 1U) &&
						(R == DevInst->NumRows - 1U) &&
						(M == NumMods - 1U)) {
					RegAddr = MCtrlMod[M].MemCtrlRegOff +
						_XAie_GetTileAddr(DevInst,
								Loc.Row,
								Loc.Col);
					return XAie_MaskPoll(DevInst, RegAddr,
							MCtrlMod[M].MemZeroisation.Mask,
							0, XAIEML_MEMZERO_POLL_TIMEOUT);
				}
			}
		}
	}

	/* Code should never reach here. */
	return XAIE_ERR;
}

/*****************************************************************************/
/**
* This API sets the column clock control register. Its configuration affects
* (enable or disable) all tile's clock above the Shim tile.
*
* @param        DevInst: Device Instance
* @param        Loc: Location of AIE SHIM tile
* @param        Enable: XAIE_ENABLE to enable column global clock buffer,
*                       XAIE_DISABLE to disable.
*
* @return       XAIE_OK for success, and error code for failure.
*
* @note         It is not required to check the DevInst and the Loc tile type
*               as the caller function should provide the correct value.
*               It is internal function to this file
*
******************************************************************************/
static AieRC _XAieMl_PmSetColumnClockBuffer(XAie_DevInst *DevInst,
		XAie_LocType Loc, u8 Enable)
{
	u8 TileType;
	u32 FldVal;
	u64 RegAddr;
	XAie_LocType ShimLoc = XAie_TileLoc(Loc.Col, 0U);
	const XAie_PlIfMod *PlIfMod;
	const XAie_ShimClkBufCntr *ClkBufCntr;

	TileType = DevInst->DevOps->GetTTypefromLoc(DevInst, ShimLoc);
	PlIfMod = DevInst->DevProp.DevMod[TileType].PlIfMod;
	ClkBufCntr = PlIfMod->ClkBufCntr;

	RegAddr = ClkBufCntr->RegOff +
			_XAie_GetTileAddr(DevInst, 0U, Loc.Col);
	FldVal = XAie_SetField(Enable, ClkBufCntr->ClkBufEnable.Lsb,
			ClkBufCntr->ClkBufEnable.Mask);

	return XAie_MaskWrite32(DevInst, RegAddr, ClkBufCntr->ClkBufEnable.Mask,
			FldVal);
}

/*****************************************************************************/
/**
* This API enables clock for all the tiles passed as argument to this API.
*
* @param	DevInst: AI engine partition device instance pointer
* @param	Args: Backend tile args
*
* @return       XAIE_OK on success, error code on failure
*
* @note		Internal only.
*
*******************************************************************************/
AieRC _XAieMl_RequestTiles(XAie_DevInst *DevInst, XAie_BackendTilesArray *Args)
{
	AieRC RC;
	u32 SetTileStatus;


	if(Args->Locs == NULL) {
		u32 NumTiles;

		XAie_LocType TileLoc = XAie_TileLoc(0, 1);
		NumTiles = (u32)((DevInst->NumRows - 1U) * (DevInst->NumCols));

		SetTileStatus = _XAie_GetTileBitPosFromLoc(DevInst, TileLoc);
		_XAie_SetBitInBitmap(DevInst->DevOps->TilesInUse, SetTileStatus,
				NumTiles);

		return DevInst->DevOps->SetPartColClockAfterRst(DevInst,
				XAIE_ENABLE);
	}

	/* Disbale all the column clock and enable only the requested column clock */
	RC = _XAie_PmSetPartitionClock(DevInst, XAIE_DISABLE);
	if(RC != XAIE_OK) {
		XAIE_ERROR("Failed to set partition clock buffers.\n");
		return RC;
	}

	/* Clear the TilesInuse bitmap to reflect the current status */
	for(u32 C = 0; C < DevInst->NumCols; C++) {
		XAie_LocType Loc;
		u32 ColClockStatus;

		Loc = XAie_TileLoc((u8)C, 1U);
		ColClockStatus = _XAie_GetTileBitPosFromLoc(DevInst, Loc);

		_XAie_ClrBitInBitmap(DevInst->DevOps->TilesInUse,
				ColClockStatus, (u32)(DevInst->NumRows - 1U));
	}

	for(u32 i = 0; i < Args->NumTiles; i++) {
		u32 ColClockStatus;

		if(Args->Locs[i].Col >= DevInst->NumCols || Args->Locs[i].Row >= DevInst->NumRows) {
			XAIE_ERROR("Invalid Tile Location \n");
			return XAIE_INVALID_TILE;
		}

		/*
		 * Shim rows are enabled by default, skip shim row
		 */
		if (Args->Locs[i].Row == 0U) {
			continue;
		}

		/*
		 * Check if column clock buffer is already enabled and continue
		 * Get bitmap position from first row after shim
		 */
		ColClockStatus = _XAie_GetTileBitPosFromLoc(DevInst,
				XAie_TileLoc(Args->Locs[i].Col, 1));
		if (CheckBit(DevInst->DevOps->TilesInUse, ColClockStatus)) {
			continue;
		}

		RC = _XAieMl_PmSetColumnClockBuffer(DevInst, Args->Locs[i],
				XAIE_ENABLE);
		if(RC != XAIE_OK) {
			XAIE_ERROR("Failed to enable clock for column: %d\n",
					Args->Locs[i].Col);
			return RC;
		}

		/*
		 * Set bitmap for entire column, row 1 to last row.
		 * Shim row is already set, so use NumRows-1
		 */
		_XAie_SetBitInBitmap(DevInst->DevOps->TilesInUse,
				ColClockStatus, (u32)(DevInst->NumRows - 1U));
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
* This API enable/disable the module clock control
*
* @param        DevInst: Device Instance
* @param        Loc: Location of AIE SHIM tile
* @param        Enable: XAIE_ENABLE to enable shim clock buffer,
*                       XAIE_DISABLE to disable.
*
* @return       XAIE_OK for success, and error code for failure.
*
* @note         It is not required to check the DevInst and the Loc tile type
*               as the caller function should provide the correct value.
*               It is internal function to this file
*
******************************************************************************/
static AieRC _XAieMl_PmSetShimClk(XAie_DevInst *DevInst,
		XAie_LocType Loc, u8 Enable)
{
	u8 TileType;
	u32 FldVal;
	u64 RegAddr;
	AieRC RC;
	XAie_LocType ShimLoc = XAie_TileLoc(Loc.Col, 0U);
	const XAie_PlIfMod *PlIfMod;
	const XAie_ShimModClkCntr0 *ModClkCntr0;
	const XAie_ShimModClkCntr1 *ModClkCntr1;

	TileType = DevInst->DevOps->GetTTypefromLoc(DevInst, ShimLoc);
	PlIfMod = DevInst->DevProp.DevMod[TileType].PlIfMod;
	ModClkCntr0 = PlIfMod->ModClkCntr0;
	ModClkCntr1 = PlIfMod->ModClkCntr1;

	RegAddr = ModClkCntr0->RegOff +
			_XAie_GetTileAddr(DevInst, 0U, Loc.Col);
	FldVal = XAie_SetField(Enable, ModClkCntr0->StrmSwClkEnable.Lsb,
			ModClkCntr0->StrmSwClkEnable.Mask);
	FldVal |= XAie_SetField(Enable, ModClkCntr0->PlIntClkEnable.Lsb,
			ModClkCntr0->PlIntClkEnable.Mask);
	FldVal |= XAie_SetField(Enable, ModClkCntr0->CteClkEnable.Lsb,
			ModClkCntr0->CteClkEnable.Mask);

	RC = XAie_MaskWrite32(DevInst, RegAddr, XAIEMLGBL_PL_MODULE_MODULE_CLOCK_CONTROL_0_MASK,
			FldVal);
	if(RC != XAIE_OK) {
		XAIE_ERROR("Failed to enable module clock control 0\n");
		return RC;
	}

	RegAddr = ModClkCntr1->RegOff +
			_XAie_GetTileAddr(DevInst, 0U, Loc.Col);
	FldVal = XAie_SetField(Enable, ModClkCntr1->NocModClkEnable.Lsb,
			ModClkCntr1->NocModClkEnable.Mask);

	RC = XAie_MaskWrite32(DevInst, RegAddr, XAIEMLGBL_PL_MODULE_MODULE_CLOCK_CONTROL_1_MASK,
			FldVal);
	if(RC != XAIE_OK) {
		XAIE_ERROR("Failed to enable module clock control 1\n");
		return RC;
	}

	return XAIE_OK;

}

/*****************************************************************************/
/**
* This API enables column clock and module clock control register for the
* requested tiles passed as argument to this API.
*
* @param	DevInst: AI engine partition device instance pointer
* @param	Args: Backend column args
*
* @return       XAIE_OK on success, error code on failure
*
* @note		Internal only.
*
*******************************************************************************/
AieRC _XAieMl_SetColumnClk(XAie_DevInst *DevInst, XAie_BackendColumnReq *Args)
{
	AieRC RC;

	u32 StartBit, EndBit;
	u32 PartEndCol = (u32)(DevInst->StartCol + DevInst->NumCols - 1U);

	if((Args->StartCol < DevInst->StartCol) || (Args->StartCol > PartEndCol) ||
	   ((Args->StartCol + Args->NumCols - 1U) > PartEndCol) ) {
		XAIE_ERROR("Invalid Start Column/Numcols \n");
		return XAIE_ERR;
	}

	/*Enable the clock control register for shims*/
	for(u32 C = Args->StartCol; C < (Args->StartCol + Args->NumCols); C++) {
		XAie_LocType TileLoc = XAie_TileLoc((u8)C, 1U);

		RC = _XAieMl_PmSetColumnClockBuffer(DevInst, TileLoc,
				Args->Enable);
		if(RC != XAIE_OK) {
			XAIE_ERROR("Failed to enable clock for column: %d\n",
					TileLoc.Col);
			return RC;
		}

		RC = _XAieMl_PmSetShimClk(DevInst, TileLoc, Args->Enable);
		if(RC != XAIE_OK) {
			XAIE_ERROR("Failed to set module clock control.\n");
			return RC;
		}
	}

	StartBit = _XAie_GetTileBitPosFromLoc(DevInst,
			 XAie_TileLoc((u8)Args->StartCol, 0));
	EndBit = _XAie_GetTileBitPosFromLoc(DevInst,
			 XAie_TileLoc((u8)(Args->StartCol + Args->NumCols), 0));

	if(Args->Enable) {
		/*
		 * Set bitmap from start column to Start+Number of columns
		 */
		_XAie_SetBitInBitmap(DevInst->DevOps->TilesInUse,
					StartBit, EndBit);
	} else {
		_XAie_ClrBitInBitmap(DevInst->DevOps->TilesInUse,
				StartBit, EndBit);
	}

	return XAIE_OK;
}
#endif /* XAIE_FEATURE_PRIVILEGED_ENABLE */
/** @} */
