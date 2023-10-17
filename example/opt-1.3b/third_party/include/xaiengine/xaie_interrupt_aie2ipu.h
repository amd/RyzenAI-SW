/******************************************************************************
* Copyright (C) 2021 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_interrupt_aie2ipu.h
* @{
*
* Internal header file to capture interrupt APIs specific AIE2 IPU.
*
******************************************************************************/
#ifndef XAIE_INTERRUPT_AIE2IPU_H
#define XAIE_INTERRUPT_AIE2IPU_H

/***************************** Include Files *********************************/
/**************************** Type Definitions *******************************/
/************************** Function Prototypes  *****************************/
u8 _XAie2Ipu_IntrCtrlL1IrqId(XAie_DevInst *DevInst, XAie_LocType Loc,
		XAie_BroadcastSw Switch);

#endif		/* end of protection macro */
