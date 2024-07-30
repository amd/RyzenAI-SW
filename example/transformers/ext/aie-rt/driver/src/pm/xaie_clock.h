/******************************************************************************
* Copyright (C) 2019 - 2020 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_clock.h
* @{
*
* Header file for timer implementation.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who      Date        Changes
* ----- ------   --------    --------------------------------------------------
* 1.0   Dishita  06/26/2020  Initial creation
* </pre>
*
******************************************************************************/
#ifndef XAIE_CLOCK_H
#define XAIE_CLOCK_H

/***************************** Include Files *********************************/
#include "xaiegbl.h"

/************************** Enum *********************************************/

/************************** Function Prototypes  *****************************/
XAIE_AIG_EXPORT AieRC XAie_PmRequestTiles(XAie_DevInst *DevInst, XAie_LocType *Loc,
		u32 NumTiles);
XAIE_AIG_EXPORT AieRC XAie_PmSetColumnClk(XAie_DevInst *DevInst, u32 StartCol,
		u32 NumCols, u8 Enable);
#endif		/* end of protection macro */
