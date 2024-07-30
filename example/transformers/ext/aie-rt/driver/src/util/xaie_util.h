/******************************************************************************
* Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All Rights Reserved.  *
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_util.h
* @{
*
* This file contains function implementations for AIE utilities
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Keerthanna 03/10/2023  Initial creation
* </pre>
*
******************************************************************************/

/***************************** Include Files *********************************/
#include "xaie_feature_config.h"

#ifdef XAIE_FEATURE_UTIL_STATUS_ENABLE

/**************************** Function Definitions *******************************/
/*****************************************************************************/
/**
*
* This is a helper function to calculate the number of characters in the string
* buffer
*
*
* @param        XAie_EvntStrings: Pointer to the string to calculate the length.
*
* @return	The number of characters in XAie_EvntStrings.
*
* @note		Internal only.
*
*******************************************************************************/
static inline int _XAie_Length(const char* XAie_RegStrings) {
	int XAie_RegStrSize = 0;
	while(*XAie_RegStrings != '\0') {
		XAie_RegStrSize++;
		XAie_RegStrings++;
	}
	return XAie_RegStrSize;
}


/*****************************************************************************/
/**
*
* This is a helper function to implement string copy.
*
*
* @param	Destination: Destination char pointer.
* @param	Source: Source string.
* @param        CommaNeeded: u8; if true a comma will be appended at the
*               end after copying.
*
* @return	The number of characters copied successfully from source to
*               destination. XAIE_ERROR if a failure occured.
*
* @note		Internal only.
*
*******************************************************************************/
static int _XAie_strcpy(char* Destination, u32 DestSize, const char* Source,
		u8 CommaNeeded)
{

    int len = 0;
    const char* Ptr = Source;
    u32 RegStrSize = (u32)_XAie_Length(Ptr);

    RegStrSize += (CommaNeeded != 0U) ? 1U: 0U;

    if (Destination == NULL || Source == NULL || DestSize < RegStrSize) {
        return -1;
    }
    while (*Source != '\0')
    {
	*Destination = *Source;
        Destination++;
        Source++;
	len++;
    }

    if (CommaNeeded)
    {
        *Destination = ',';
        len++;
    }
    return len;
}

#endif /* XAIE_FEATURE_UTIL_STATUS_ENABLE */

/** @} */
