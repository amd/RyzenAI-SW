/******************************************************************************
* Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_txn.h
* @{
*
* This file contains data structure for TxN flow
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Keyur   08/25/2023  Initial creation
* </pre>
*
******************************************************************************/
#ifndef XAIETXN_H
#define XAIETXN_H

/***************************** Include Files *********************************/
/* All New custom Ops should be added above XAIE_IO_CUSTOM_OP_NEXT
 * To support backward compatibility existing enums should not be
 * modified. */
typedef enum {
	XAIE_IO_WRITE,
	XAIE_IO_BLOCKWRITE,
	XAIE_IO_BLOCKSET,
	XAIE_IO_MASKWRITE,
	XAIE_IO_MASKPOLL,
	XAIE_CONFIG_SHIMDMA_BD,
	XAIE_CONFIG_SHIMDMA_DMABUF_BD,
	XAIE_IO_CUSTOM_OP_BEGIN = 1U<<7U,
	XAIE_IO_CUSTOM_OP_TCT = XAIE_IO_CUSTOM_OP_BEGIN,
	XAIE_IO_CUSTOM_OP_DDR_PATCH, // Previously this was XAIE_IO_CUSTOM_OP_BEGIN + 1
	XAIE_IO_CUSTOM_OP_READ_REGS, // Previously this was XAIE_IO_CUSTOM_OP_BEGIN + 2
	XAIE_IO_CUSTOM_OP_RECORD_TIMER, // Previously this was XAIE_IO_CUSTOM_OP_BEGIN + 3
	XAIE_IO_CUSTOM_OP_MERGE_SYNC, // Previously this was XAIE_IO_CUSTOM_OP_BEGIN + 4
	XAIE_IO_CUSTOM_OP_NEXT,
	XAIE_IO_CUSTOM_OP_MAX = UCHAR_MAX,
} XAie_TxnOpcode;

struct XAie_TxnCmd {
	XAie_TxnOpcode Opcode;
	u32 Mask;
	u64 RegOff;
	u32 Value;
	u64 DataPtr;
	u32 Size;
};

#endif
