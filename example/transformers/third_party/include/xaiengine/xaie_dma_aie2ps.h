/******************************************************************************
* Copyright (C) 2019 - 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_dma_aie2ps.h
* @{
*
* This file contains routines for AIEML DMA configuration and controls. This
* header file is not exposed to the user.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who        Date        Changes
* ----- ------     --------    -----------------------------------------------------
* 1.0   Sanakrji   10/04/2022  Initial creation
* </pre>
*
******************************************************************************/
#ifndef XAIE_DMA_AIE2PS_H
#define XAIE_DMA_AIE2PS_H

/***************************** Include Files *********************************/
#include "xaiegbl.h"

/************************** Function Prototypes  *****************************/
AieRC _XAie2PS_ShimDmaWriteBd(XAie_DevInst *DevInst , XAie_DmaDesc *DmaDesc,
		XAie_LocType Loc, u8 BdNum);

AieRC _XAie2PS_ShimDmaUpdateBdAddr(XAie_DevInst *DevInst,
		const XAie_DmaMod *DmaMod, XAie_LocType Loc, u64 Addr,
		u8 BdNum);

#endif /* XAIE_DMA_AIE2PS_H */
/** @} */
