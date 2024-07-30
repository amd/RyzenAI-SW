/******************************************************************************
* Copyright (C) 2023 AMD, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_lite_shim_aieml.h
* @{
*
* This header file defines a lite shim interface for AIE2 type devices.
*
******************************************************************************/
#ifndef _XAIE_LITE_SHIM_AIEML_H_
#define _XAIE_LITE_SHIM_AIEML_H_

/***************************** Include Files *********************************/
#include "xaie_lite_hwcfg.h"
#include "xaiegbl_defs.h"
#include "xaiegbl.h"

/************************** Function Prototypes  *****************************/
/*****************************************************************************/
/**
* This API modifies(enable or disable) the clock control register for given shim.
*
* @param        DevInst: Device Instance
* @param        Loc: Location of AIE SHIM tile
* @param        Enable: XAIE_ENABLE to enable shim clock buffer,
*                       XAIE_DISABLE to disable.

* @note         It is internal function to this file
*
******************************************************************************/
static inline void _XAie_PrivilegeSetShimClk(XAie_DevInst *DevInst,
					     XAie_LocType Loc, u8 Enable)
{
	u64 RegAddr;
	u32 FldVal;

	RegAddr = _XAie_LGetTileAddr(Loc.Row, Loc.Col) +
		XAIE_SHIM_TILE_MOD_CLOCK_CONTROL_0_REGOFF;
	FldVal = XAie_SetField(Enable,
			XAIE_SHIM_TILE_MOD_CLOCK_CONTROL_0_CTE_CLOCK_ENABLE_LSB,
			XAIE_SHIM_TILE_MOD_CLOCK_CONTROL_0_CTE_CLOCK_ENABLE_MASK);
	FldVal |= XAie_SetField(Enable,
			XAIE_SHIM_TILE_MOD_CLOCK_CONTROL_0_PL_INTERFACE_CLOCK_ENABLE_LSB,
			XAIE_SHIM_TILE_MOD_CLOCK_CONTROL_0_PL_INTERFACE_CLOCK_ENABLE_MASK);
	FldVal |= XAie_SetField(Enable,
			XAIE_SHIM_TILE_MOD_CLOCK_CONTROL_0_STREAM_SWITCH_CLOCK_ENABLE_LSB,
			XAIE_SHIM_TILE_MOD_CLOCK_CONTROL_0_STREAM_SWITCH_CLOCK_ENABLE_MASK);



	_XAie_LPartMaskWrite32(DevInst, RegAddr,
		XAIE_SHIM_TILE_MOD_CLOCK_CONTROL_0_MASK, FldVal);

	RegAddr = _XAie_LGetTileAddr(Loc.Row, Loc.Col) +
		XAIE_SHIM_TILE_MOD_CLOCK_CONTROL_1_REGOFF;
	FldVal = XAie_SetField(Enable,
			XAIE_SHIM_TILE_NOC_MOD_CLOCK_CONTROL_1_CLOCK_ENABLE_LSB,
			XAIE_SHIM_TILE_NOC_MOD_CLOCK_CONTROL_1_CLOCK_ENABLE_MASK);

	_XAie_LPartMaskWrite32(DevInst, RegAddr,
		XAIE_SHIM_TILE_MOD_CLOCK_CONTROL_1_MASK, FldVal);

}


#endif /* _XAIE_LITE_SHIM_AIEML_H_ */
/** @} */
