/******************************************************************************
* Copyright (C) 2023 AMD, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_interrupt_aie2ps.h
* @{
*
* Internal header file to capture interrupt APIs specific AIE2PS.
*
******************************************************************************/
#ifndef XAIE_INTERRUPT_AIE2PS_H
#define XAIE_INTERRUPT_AIE2PS_H

/***************************** Include Files *********************************/
/**************************** Type Definitions *******************************/
/************************** Function Prototypes  *****************************/
u8 _XAie2ps_IntrCtrlL1IrqId(XAie_DevInst *DevInst, XAie_LocType Loc,
		XAie_BroadcastSw Switch);

#endif		/* end of protection macro */
