/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "CppUTest/TestHarness.h"
#include <hw_config.h>

static XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
		XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
		XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
		XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
		XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);

static XAie_InstDeclare(DevInst, &ConfigPtr);

TEST_GROUP(MultiPartsApis)
{
	void setup()
	{
		AieRC RC;

		RC = XAie_CfgInitialize(&(DevInst), &ConfigPtr);
		CHECK_EQUAL(XAIE_OK, RC);

		RC = XAie_PartitionInitialize(&(DevInst), NULL);
		CHECK_EQUAL(XAIE_OK, RC);
	}
	void teardown()
	{
		AieRC RC;

		RC = XAie_PartitionTeardown(&(DevInst));
		CHECK_EQUAL(XAIE_OK, RC);

		XAie_Finish(&DevInst);
	}
};

TEST(MultiPartsApis, MultiPartsBasic)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;
	u64 AiePartBaseAddr = 0x40000000;
	u8 PartStartCol = 2;
	u8 PartNumCols = 3;

	XAie_LocType TileLoc0 = XAie_TileLoc(0, 0);
	XAie_LocType TileLoc1 = XAie_TileLoc(2, 0);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc0);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc1);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_SetupPartitionConfig(&DevInst, AiePartBaseAddr, PartStartCol,
			PartNumCols);
	CHECK_EQUAL(RC, XAIE_INVALID_DEVICE);

	XAie_Finish(&DevInst);

	RC = XAie_SetupPartitionConfig(&DevInst, AiePartBaseAddr, PartStartCol,
			XAIE_NUM_COLS);
	RC = XAie_CfgInitialize(&(DevInst), &ConfigPtr);
	CHECK_EQUAL(RC, XAIE_INVALID_DEVICE);

	RC = XAie_SetupPartitionConfig(&DevInst, AiePartBaseAddr, PartStartCol,
			PartNumCols);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_CfgInitialize(&(DevInst), &ConfigPtr);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc0);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc1);
#if AIE_GEN == 3
	CHECK_EQUAL(RC, XAIE_OK);
#else
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);
#endif

	CHECK_EQUAL(DevInst.BaseAddr, AiePartBaseAddr);
}
