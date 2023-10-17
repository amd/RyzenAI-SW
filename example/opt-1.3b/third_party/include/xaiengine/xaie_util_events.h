/******************************************************************************
* Copyright (C) 2022 Xilinx, Inc.  All rights reserved.
* Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All Rights Reserved.  *
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_util_events.h
* @{
*
* Header to include function prototypes for AIE utilities
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Gregory 03/31/2022  Initial creation
* </pre>
*
******************************************************************************/
#ifndef XAIE_UTIL_EVENTS_H
#define XAIE_UTIL_EVENTS_H

#include "xaie_feature_config.h"
#ifdef XAIE_FEATURE_UTIL_ENABLE

/***************************** Include Files *********************************/
#include "xaie_events.h"

/**************************** Function Prototypes *******************************/
const char* XAie_EventGetString(XAie_Events Event);
#endif  /* XAIE_FEATURE_UTIL_ENABLE */

#endif	/* end of protection macro */

/** @} */
