/******************************************************************************
* Copyright (C) 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

/*****************************************************************************/
/**
* @file xaie_lite.c
* @{
*
* This file contains lite routines.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date        Changes
* ----- ------  --------    ---------------------------------------------------
* 1.0   Nishad  06/23/2022  Initial creation
*
* </pre>
*
******************************************************************************/
/***************************** Include Files *********************************/

#include "xaie_feature_config.h"

#if defined(XAIE_FEATURE_PRIVILEGED_ENABLE) && defined(XAIE_FEATURE_LITE)

#include "xaie_lite.h"
#include "xaiegbl_defs.h"
#include "xaiegbl.h"

/***************************** Macro Definitions *****************************/
/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
*
* This API maps given IRQ ID to a range of columns it is programmed to receive
* interrupts from.
*
* @param	IrqId:
* @param	Range: Pointer to return column range mapping.
*
* @return	XAIE_OK on success, error code on failure.
*
* @note		None
*
******************************************************************************/
AieRC XAie_MapIrqIdToCols(u8 IrqId, XAie_Range *Range)
{
	XAIE_ERROR_RETURN(IrqId >= XAIE_MAX_NUM_NOC_INTR, XAIE_INVALID_ARGS,
			XAIE_ERROR_MSG("Invalid AIE IRQ ID\n"));

	XAie_Range Temp = _XAie_MapIrqIdToCols(IrqId);
	Range->Start = Temp.Start;
	Range->Num = Temp.Num;

	return XAIE_OK;
}

#endif /* XAIE_FEATURE_PRIVILEGED_ENABLE && XAIE_FEATURE_LITE */
/** @} */
