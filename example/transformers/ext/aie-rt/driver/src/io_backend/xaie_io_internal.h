/******************************************************************************
* Copyright (C) 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_io_internal.c
* @{
*
* This file contains the data structures and routines for low level IO
* operations.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Ssatpute   27/06/2020 Initial creation.
* </pre>
*
******************************************************************************/
#ifndef XAIE_IO_INTERNAL_H
#define XAIE_IO_INTERNAL_H

/***************************** Include Files *********************************/
#include "xaie_rsc.h"
#include "xaiegbl.h"

/****************************** Type Definitions *****************************/
/************************** Function Prototypes  *****************************/
const XAie_Backend* _XAie_GetBackendPtr(XAie_BackendType Backend);

/*****************************************************************************/
/**
*
* Set the NPI write request arguments
*
* @param	RegOff : NPI register offset
* @param	RegVal : Register Value
* @return	NPI write request
*
* @note		Internal API only.
*
******************************************************************************/
static inline XAie_BackendNpiWrReq
_XAie_SetBackendNpiWrReq(u32 RegOff, u32 RegVal)
{
	XAie_BackendNpiWrReq Req;

	Req.NpiRegOff = RegOff;
	Req.Val = RegVal;

	return Req;
}

/*****************************************************************************/
/**
*
* Set the NPI mask poll request arguments
*
* @param	RegOff : NPI register offset
* @param	Mask   : Mask Value
* @param	RegVal : Register Value
* @param	TimeOutUs : Time Out Value in us.
* @return	NPI mask poll request
*
* @note		Internal API only.
*
******************************************************************************/
static inline XAie_BackendNpiMaskPollReq
_XAie_SetBackendNpiMaskPollReq(u32 RegOff, u32 Mask, u32 RegVal, u32 TimeOutUs)
{
	XAie_BackendNpiMaskPollReq Req;

	Req.NpiRegOff = RegOff;
	Req.Val = RegVal;
	Req.Mask = Mask;
	Req.TimeOutUs = TimeOutUs;

	return Req;
}

#endif	/* End of protection macro */
