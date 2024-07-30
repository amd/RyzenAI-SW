/******************************************************************************
* Copyright (C) 2024 AMD, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_dma_aie2p.h
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
#ifndef XAIE_DMA_AIE2P_H
#define XAIE_DMA_AIE2P_H

/***************************** Include Files *********************************/
#include "xaiegbl.h"

/************************** Function Prototypes  *****************************/
AieRC _XAie2P_AxiBurstLenCheck(u8 BurstLen);

#endif /* XAIE_DMA_AIE2P_H */
/** @} */
