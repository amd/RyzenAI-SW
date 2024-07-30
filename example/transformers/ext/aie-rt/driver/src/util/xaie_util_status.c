/******************************************************************************
* Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_util_status.c
* @{
*
* This file contains function implementations for AIE status utilities
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   dsteger 07/25/2022  Initial creation
* </pre>
*
******************************************************************************/

/***************************** Include Files *********************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xaie_util_status.h"
#include "xaie_util.h"

#ifdef XAIE_FEATURE_UTIL_STATUS_ENABLE

/**************************** Function Definitions *******************************/
/*****************************************************************************/
/**
*
* This is a helper function to implement integer to string conversion.
*
* @param	num: Integer value to be converted to String.
* @param	str: Char pointer to copy the integer to.
*
* @return	XAIE_OK if the string was successfully copied. XAIE_ERROR if
*               a failure occured.
*
* @note		Internal only.
*
******************************************************************************/
static void _XAie_ToString(char str[], int num)
{
    int i, rem, len = 0, n;

    n = num;

    if(num == 0)
    {
	str[num] = '0';
        str[1]   = '\0';
	return;
    }
    while (n != 0)
    {
        len++;
        n /= 10;
    }
    for (i = 0; i < len; i++)
    {
        rem = num % 10;
        num = num / 10;
        str[len - (i + 1)] = rem + '0';
    }
    str[len] = '\0';
}

/*****************************************************************************/
/**
*
* This API maps the Core status bits to its corresponding string.  If more
* than one bit is set, all the corresponding strings are separated by a comma
* and concatenated.
*
* @param        Reg: Core Status raw register value.
* @param        Buf: Pointer to the buffer which the string will be written to.
*
* @return       The total number of characters filled up in the Buffer
*               argument parameter.
*
* @note         None.
*
******************************************************************************/
int XAie_CoreStatus_CSV(u32 Reg, char *Buf, u32 BufSize) {

    int CharsWritten = 0, Ret;        // characters written
    u8  Shift 	= 0, CommaNeeded = 1U;
    u32 Val 	= 0;
    u32 TempReg = Reg;
    if( (TempReg & 0x01U) ) {// if bit 0 is set then Enabled state.
	CommaNeeded = 0x1U;
        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
			XAie_CoreStatus_Strings[Shift], CommaNeeded);
	if(Ret == -1) {
		return -1;
	}
	else {
		CharsWritten += Ret;
	}

    }

    Shift++;
    TempReg     = TempReg >> 1;
    Val = TempReg & 0x01U;
    if(Val) {
	    // if bit 1 is set then Reset state.
	Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
			XAie_CoreStatus_Strings[Shift], CommaNeeded);
	if(Ret == -1) {
		return -1;
	}
	else {
		CharsWritten += Ret;
	}

    }
    if((Reg&0x01U) || (Reg&0x02U) == 0U) {
	    // if neither bit 0 nor bit  1 is set, Disabled is output.
        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
			"Disabled", CommaNeeded);
	if(Ret == -1) {
		return -1;
	}
	else {
		CharsWritten += Ret;
	}

    }

    TempReg     = TempReg >> 1;

    while(TempReg!=0U){
	Shift++;
	Val 	= TempReg & 0x1U;
	if (Val){
            Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
			    XAie_CoreStatus_Strings[Shift], CommaNeeded);
		if(Ret == -1) {
			return -1;
		}
		else {
			CharsWritten += Ret;
		}

	}
	TempReg >>= 1;
    }
    if(CommaNeeded) {
	CharsWritten--;   // The last call added a comma , which is not needed at the end.
    }
    Buf[CharsWritten]='\0';
    return CharsWritten;
}

/*****************************************************************************/
/**
*
* This API maps the DMA S2MM status bits to its coresponding string.  If more
* than one bit is set, all the corresponding strings are separated by a comma
* and concatenated.
*
* @param        Reg: DMA S2MM status raw register value.
* @param        Buf: Pointer to the buffer which the string will be written to.
* @param        TType: Tile type used to distinguish the tile type, Core,
*               Memory or Shim for which this function is called.
*
* @return       The total number of characters filled up in the Buffer
*               argument parameter.
*
* @note         None.
*
******************************************************************************/
int XAie_DmaS2MMStatus_CSV(u32 Reg, char *Buf, u32 BufSize,  u8 TType) {

    int CharsWritten = 0, Ret;
    enum DmaStatus_S2MM_enum Flag = 0;
    u32 FlagVal;
    u8 CommaNeeded = 0U;
    FlagVal = (u32)Flag;

    for(FlagVal = (u32)XAIE_DMA_STATUS_S2MM_STATUS; FlagVal <= (u32)XAIE_DMA_STATUS_S2MM_MAX;
		    FlagVal++) {
	Ret = 0;
	CommaNeeded = 1U;
        // Below is for bits  8, 9 in DMAS2MM for mem tile
        if( (TType != XAIEGBL_TILE_TYPE_MEMTILE) && \
			((FlagVal == (u32)XAIE_DMA_STATUS_S2MM_ERROR_LOCK_ACCESS_TO_UNAVAIL) || \
			 (FlagVal == (u32)XAIE_DMA_STATUS_S2MM_ERROR_DM_ACCESS_TO_UNAVAIL))
	  ) {
	   continue;
	}

        // Below is for bits 16,17 in DMAS2MM for shim tile
        if( (TType != XAIEGBL_TILE_TYPE_SHIMNOC) && ((FlagVal == (u32)XAIE_DMA_STATUS_S2MM_AXI_MM_DECODE_ERROR) || \
				   (FlagVal == (u32)XAIE_DMA_STATUS_S2MM_AXI_MM_SLAVE_ERROR)) ) {
	   continue;
	}

	if(XAie_DmaS2MMStatus_Strings[FlagVal] != NULL)
        {
            u32 Val = (Reg >> FlagVal);
	    char TempString[4];
            switch (FlagVal) {
		case (u32)XAIE_DMA_STATUS_S2MM_STATUS:
			CommaNeeded = 0U;
		    Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten ,
				    "Channel_status:", CommaNeeded);
		    if(Ret == -1) {
			    return -1;
		    }
		    else {
			CharsWritten += Ret;
		    }

		    Val &= 0x3U;
		    CommaNeeded = 1U;
		    if (Val == 0U) {
		        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					"Idle", CommaNeeded);
		    }
		    else if (Val == 1U) {
		        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					"Starting", CommaNeeded);
		    }
		    else if (Val == 2U) {
		        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					"Running", CommaNeeded);
		    }
		    else {
		        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					"Invalid_State", CommaNeeded);
		    }
		    break;
		case (u32)XAIE_DMA_STATUS_S2MM_TASK_QUEUE_OVERFLOW:
		    Val &= 0x01U;
		    CharsWritten--; // to overwrite the comma in the previous write
		    if (Val == 0U) {
		        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					";Queue_status:okay", CommaNeeded);
		    }
		    else {
		        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					";Queue_status:channel_overflow",
					CommaNeeded);
		    }
		    break;
		case (u32)XAIE_DMA_STATUS_S2MM_CHANNEL_RUNNING:
		    Val &= 0x1U;
		    CharsWritten--; // to overwrite the comma in the previous write
		    if (Val == 0U) {
		        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					";Queue_empty", CommaNeeded);
		    }
		    else {
		        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					";Queue_not_empty", CommaNeeded);
		    }
		    break;
		case (u32)XAIE_DMA_STATUS_S2MM_TASK_QUEUE_SIZE:
		    Val &= 0x7U;
		    CommaNeeded = 0U;
		    CharsWritten--; // to overwrite the comma in the previous write
		    _XAie_ToString(TempString, (int)Val);
		    Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
				    ";Tasks_in_queue:", CommaNeeded);
		    if(Ret == -1) {
			return -1;
		    }
		    else {
			CharsWritten += Ret;
                    }
		    CommaNeeded = 1U;
		    Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
				    TempString, CommaNeeded);
		    break;
		case (u32)XAIE_DMA_STATUS_S2MM_CURRENT_BD:
		    if(TType == XAIEGBL_TILE_TYPE_MEMTILE) {
			    Val &= 0x3FU;
		    }
		    else {
			    Val &= 0x0FU;
		    }
		    CommaNeeded = 0U;
		    CharsWritten--; // to overwrite the comma in the previous write
		    _XAie_ToString(TempString,(int)Val);
		    Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
				    ";Current_bd:", CommaNeeded);
		    if(Ret == -1) {
			return -1;
		    }
		    else {
			CharsWritten += Ret;
                    }
		    CommaNeeded = 1U;
		    Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
				    TempString, CommaNeeded);
		    break;
		default:
		    Val &= 0x1U;
		    if (Val) {
			Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					XAie_DmaS2MMStatus_Strings[FlagVal],
					CommaNeeded);
		    }
		    break;
		};
		if(Ret == -1) {
			return -1;
		}
		else {
			CharsWritten += Ret;
		}
        }
    }

    if(CommaNeeded) {
	CharsWritten--;   // The last call added a comma , which is not needed at the end.
    }
    Buf[CharsWritten]='\0';
    return CharsWritten;
}   // end of XAie_DmaS2MMStatus_CSV

/*****************************************************************************/
/**
*
* This API maps the DMA MM2S status bits to its corresponding string. If more
* than one bit is set, all the corresponding strings are separated by a comma
* and concatenated.
*
* @param        Reg: DMA MM2S status raw register value.
* @param        Buf: Pointer to the buffer to where the string will be written.
* @param        TType: Tile type used to distinguish the tile type, Core,
*               Memory or Shim for which this function is called.
*
* @return       The total number of characters filled up in the Buffer.
*
* @note         None.
*
******************************************************************************/
int XAie_DmaMM2SStatus_CSV(u32 Reg, char *Buf, u32 BufSize, u8 TType) {

    int CharsWritten = 0, Ret;
    enum DmaStatus_MM2S_enum Flag = 0;
    u32 FlagVal;
    u8 CommaNeeded = 0U;
    FlagVal = (u32)Flag;

    for(FlagVal = (u32)XAIE_DMA_STATUS_MM2S_STATUS; FlagVal <= (u32)XAIE_DMA_STATUS_MM2S_MAX;
			 FlagVal++) {
	Ret = 0;
	CommaNeeded = 1U;
        // Below is for bits  8, 9, 10 in DMA_MM2S for mem tile
        if( (TType != XAIEGBL_TILE_TYPE_MEMTILE) && \
			((FlagVal == (u32)XAIE_DMA_STATUS_MM2S_ERROR_LOCK_ACCESS_TO_UNAVAIL) || \
                         (FlagVal == (u32)XAIE_DMA_STATUS_MM2S_ERROR_DM_ACCESS_TO_UNAVAIL)   || \
			 (FlagVal == (u32)XAIE_DMA_STATUS_MM2S_ERROR_BD_UNAVAIL))
	  ) {
           continue;
	}

	// Below is for bits 16, 17 in DMA_MM2S for shim tile
	if( (TType != XAIEGBL_TILE_TYPE_SHIMNOC) && \
			((FlagVal == (u32)XAIE_DMA_STATUS_MM2S_AXI_MM_DECODE_ERROR) || \
			 (FlagVal == (u32)XAIE_DMA_STATUS_MM2S_AXI_MM_SLAVE_ERROR))
	  ) {
	   continue;
	}

	if(XAie_DmaMM2SStatus_Strings[FlagVal] != NULL)
        {
            u32 Val = (Reg >> FlagVal);
            char TempString[4];
            switch (FlagVal) {
                case (u32)XAIE_DMA_STATUS_MM2S_STATUS:
		    CommaNeeded = 0U;
		    Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
				    "Channel_status:", CommaNeeded);
		    if(Ret == -1) {
			return -1;
		    }
		    else {
			CharsWritten += Ret;
		    }
		    CommaNeeded = 1U;
                    Val &= 0x3U;
                    if (Val == 0U) {
                        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					"Idle", CommaNeeded);
                    }
                    else if (Val == 1U) {
                        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					"Starting", CommaNeeded);
                    }
                    else if (Val == 2U) {
                        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					"Running", CommaNeeded);
                    }
                    else {
                        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					"Invalid_State", CommaNeeded);
                    }
                    break;
		case (u32)XAIE_DMA_STATUS_MM2S_TASK_QUEUE_OVERFLOW:
		    CharsWritten--; // to overwrite the comma in the previous write
		    Val &= 0x01U;
		    if (Val == 0U) {
		        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					";Queue_status:okay", CommaNeeded);
		    }
		    else {
		        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					";Queue_status:channel_overflow", CommaNeeded);
		    }
		    break;
                case (u32)XAIE_DMA_STATUS_MM2S_CHANNEL_RUNNING:
		    CharsWritten--; // to overwrite the comma in the previous write
                    Val &= 0x1U;
                    if (Val == 0U) {
                        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					";Queue_empty", CommaNeeded);
		    }
                    else {
                        Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
					";Queue_not_empty", CommaNeeded);
		    }
                    break;
                case (u32)XAIE_DMA_STATUS_MM2S_TASK_QUEUE_SIZE:
		    CommaNeeded = 0U;
		    CharsWritten--; // to overwrite the comma in the previous write
                    Val &= 0x7U;
                    _XAie_ToString(TempString, (int)Val);
		    Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
				    ";Tasks_in_queue:", CommaNeeded);
		    if(Ret == -1) {
			    return -1;
		    }
		    else {
			CharsWritten += Ret;
		    }

		    CommaNeeded = 1U;
                    Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
				    TempString, CommaNeeded);
                    break;
                case (u32)XAIE_DMA_STATUS_MM2S_CURRENT_BD:
		    CommaNeeded = 0U;
		    CharsWritten--; // to overwrite the comma in the previous write
		    if(TType == XAIEGBL_TILE_TYPE_MEMTILE) {
			    Val &= 0x3FU;
		    }
		    else {
			    Val &= 0x0FU;
		    }

                    _XAie_ToString(TempString, (int)Val);
                    Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
				    ";Current_bd:", CommaNeeded);
		    if(Ret == -1) {
			    return -1;
		    }
		    else {
			CharsWritten += Ret;
		    }
		    CommaNeeded = 1U;
                    Ret = _XAie_strcpy(&Buf[CharsWritten], BufSize-(u32)CharsWritten,
				    TempString, CommaNeeded);
                    break;
                default:
                    Val &= 0x1U;
                    if (Val) {
                        Ret = _XAie_strcpy(&Buf[CharsWritten],
					BufSize-(u32)CharsWritten,
					XAie_DmaMM2SStatus_Strings[FlagVal],
					CommaNeeded);
                    }
                    break;
                };
		if(Ret == -1) {
		    return -1;
		}
		else {
		    CharsWritten += Ret;
		}
        }
    }
    if(CommaNeeded) {
	CharsWritten--;   // The last call added a comma , which is not needed at the end.
    }
    Buf[CharsWritten]='\0';
    return CharsWritten;
}   // end of XAie_DmaMM2SStatus_CSV

#endif /* XAIE_FEATURE_UTIL_STATUS_ENABLE */
/** @} */
