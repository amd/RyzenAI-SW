/******************************************************************************
* Copyright (C) 2020 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_ss_aie2ps.h
* @{
*
* This file contains internal api implementations for AIE stream switch.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who         Date        Changes
* ----- ---------   ----------  -----------------------------------------------
* 1.0   Sankarji   06/15/2023  Initial creation
* </pre>
*
******************************************************************************/
#ifndef XAIE_SS_AIE2PS_H
#define XAIE_SS_AIE2PS_H

/***************************** Include Files *********************************/
#include "xaie_helper.h"

/************************** Constant Definitions *****************************/

/************************** Function Prototypes  *****************************/

AieRC _XAie2PS_AieTile_StrmSwCheckPortValidity(StrmSwPortType Slave,
		u8 SlvPortNum, StrmSwPortType Master, u8 MstrPortNum);
AieRC _XAie2PS_MemTile_StrmSwCheckPortValidity(StrmSwPortType Slave,
                u8 SlvPortNum, StrmSwPortType Master, u8 MstrPortNum);
AieRC _XAie2PS_ShimTile_StrmSwCheckPortValidity(StrmSwPortType Slave,
                u8 SlvPortNum, StrmSwPortType Master, u8 MstrPortNum);

#endif /* XAIE_SS_AIE2PS_H */
/** @} */
