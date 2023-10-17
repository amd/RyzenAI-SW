/******************************************************************************
* Copyright (C) 2019 - 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_uc.h
* @{
*
* Header file for core uc loader functions
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Kishan  12/23/2022  Initial creation
* </pre>
*
******************************************************************************/
#ifndef XAIEUCLOADER_H
#define XAIEUCLOADER_H

#include "xaie_feature_config.h"
#ifdef XAIE_FEATURE_UC_ENABLE

/***************************** Include Files *********************************/
#include "xaiegbl.h"
#include "xaiegbl_regdef.h"

/************************** Function Prototypes  *****************************/

AieRC XAie_LoadUc(XAie_DevInst *DevInst, XAie_LocType Loc, const char *ElfPtr);
AieRC XAie_LoadUcMem(XAie_DevInst *DevInst, XAie_LocType Loc,
		const unsigned char* ElfMem);
AieRC _XAie_UcCoreWakeup(XAie_DevInst *DevInst, XAie_LocType Loc,
		const struct XAie_UcMod *UcMod);
AieRC _XAie_UcCoreSleep(XAie_DevInst *DevInst, XAie_LocType Loc,
		const struct XAie_UcMod *UcMod);
AieRC _XAie_UcCoreGetStatus(XAie_DevInst *DevInst, XAie_LocType Loc,
		u32 *CoreStatus, const struct XAie_UcMod *UcMod);

#endif /* XAIE_FEATURE_UC_ENABLE */

#endif		/* end of protection macro */
/** @} */
