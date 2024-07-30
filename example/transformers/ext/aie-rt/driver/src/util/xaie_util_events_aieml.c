/******************************************************************************
* Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All Rights Reserved.  *
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_util_events_aieml.c
* @{
*
* This file contains function implementations for AIE utilities
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Keerthanna 03/06/2023  Initial creation
* </pre>
*
******************************************************************************/

/***************************** Include Files *********************************/
#include "xaie_feature_config.h"
#include "xaiegbl_regdef.h"
#include "xaie_util_events_aieml.h"
#include "xaie_util.h"

#ifdef XAIE_FEATURE_UTIL_STATUS_ENABLE

/**************************** Function Definitions *******************************/
/*****************************************************************************/
/**
*
* This API maps the event status bits to its coresponding string.  If more
* than one bit is set, all the corresponding strings are separated by a comma
* and concatenated.
*
* @param        Reg: DMA S2MM status raw register value.
* @param        Buf: Pointer to the buffer which the string will be written to.
* @param        TType: Tile type used to distinguish the tile type, Core,
*               Memory or Shim.
* @param	Mod: Type of Module in the core tile.
*
* @return       The total number of characters filled up in the Buffer
*               argument parameter.
*
* @note         None.
*
******************************************************************************/
int XAie_EventStatus_CSV(XAie_DevInst* DevInst, u32 Reg, char* Buf, u32 BufSize,
		u8 TType, XAie_ModuleType Mod, u8 RegNum) {
	int CharsWritten = 0, Ret;
	XAie_Events Flag = 0;
	u32 FlagVal;
	u8 MappedEvent, CommaNeeded;

	const XAie_EvntMod *EvntMod;
	const char** XAie_EvntStrings = NULL;
	CommaNeeded = 0U;
	FlagVal = (u32)Flag;

	if((TType == XAIEGBL_TILE_TYPE_SHIMNOC) ||
			(TType == XAIEGBL_TILE_TYPE_SHIMPL)) {
		XAie_EvntStrings = XAie_EventNoCStrings;
		EvntMod = &DevInst->DevProp.DevMod[TType].EvntMod[0U];
	}
	else {
		if(TType == XAIEGBL_TILE_TYPE_MEMTILE) {
			XAie_EvntStrings = XAie_EventMemTileStrings;
		}
		else if(TType == XAIEGBL_TILE_TYPE_AIETILE) {
			XAie_EvntStrings = XAie_EventCoreModStrings[Mod];
		}
		EvntMod = &DevInst->DevProp.DevMod[TType].EvntMod[Mod];
	}

	for(FlagVal = EvntMod->EventMin; FlagVal <= EvntMod->EventMax;
					FlagVal++) {
		XAie_Events Evnt = FlagVal - EvntMod->EventMin;
		MappedEvent = EvntMod->XAie_EventNumber[Evnt];
		/*
		 * 1. Events no. 13 - 62, 67 - 75 are NoC only events skipping
		 *    it for PL Event registers.
		 * 2. Mapping the events depending on the number of the event register.
		 */
		if(((TType == XAIEGBL_TILE_TYPE_SHIMPL) &&
		  (((MappedEvent > 12U) && (MappedEvent < 63U))
		  || ((MappedEvent > 66U) && (MappedEvent < 76U))))
		  || ((MappedEvent < ((32U*RegNum)))
		  ||(MappedEvent >= (32U * (RegNum + 1U))))) {
			continue;
		}
		if(XAie_EvntStrings[MappedEvent] != NULL) {
			CommaNeeded = 0x1U;
			u32 Val = (Reg >> MappedEvent) & 0x1U;
			if(Val) {
				Ret = _XAie_strcpy(&Buf[CharsWritten],
						BufSize-(u32)CharsWritten,
						XAie_EvntStrings[MappedEvent],
						CommaNeeded);
				if(Ret == -1) {
					return -1;
				}
				else {
					CharsWritten += Ret;
				}
			}
		}

	}

	if(CommaNeeded) {
		CharsWritten--;
	}
	Buf[CharsWritten]='\0';
	return CharsWritten;
}

#endif /* XAIE_FEATURE_UTIL_ENABLE */

/** @} */
