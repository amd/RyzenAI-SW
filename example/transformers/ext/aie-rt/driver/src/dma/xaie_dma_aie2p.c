/******************************************************************************
* Copyright (C) 2024 AMD, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_dma_aie2p.c
* @{
*
* This file contains routines for AIE2P DMA configuration and controls. This
* header file is not exposed to the user.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who        Date        Changes
* ----- ------     --------    -----------------------------------------------------
* 1.0   jbaniset   16/02/2024  Initial creation
* </pre>
*
******************************************************************************/
/***************************** Include Files *********************************/
#include "xaiegbl.h"

/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
*
* This API checks for correct Burst length.
*
* @param	BurstLen: Burst length to check if it has correct value or not.
*
* @return	XAIE_OK on success, Error code on failure.
*
* @note		Internal only.
*
******************************************************************************/
AieRC _XAie2P_AxiBurstLenCheck(u8 BurstLen)
{
	switch (BurstLen) {
	case 4:
	case 8:
	case 16:
	case 32:
		return XAIE_OK;
	default:
		return XAIE_INVALID_BURST_LENGTH;
	}
}
