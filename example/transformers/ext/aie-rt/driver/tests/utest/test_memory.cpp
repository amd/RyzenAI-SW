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

TEST_GROUP(MemoryApis)
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

TEST(MemoryApis, MemWrWord)
{
	AieRC RC;

	RC = XAie_DataMemWrWord(&DevInst, XAie_TileLoc(8, 3), 0x4000,
			0xDEADBEEF);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(MemoryApis, MemRdWord)
{
	AieRC RC;
	u32 Data;

	XAie_LocType TileLoc;

	RC = XAie_DataMemRdWord(&DevInst, XAie_TileLoc(8, 3), 0x4000, &Data);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(MemoryApis, MemBlck)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	u32 tempStorage = 75;
	RC = XAie_DataMemBlockWrite(&DevInst, TileLoc, 0xFFF,
			&tempStorage, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_DataMemBlockRead(&DevInst, TileLoc, 0xFFF,
			&tempStorage, 2);
	CHECK_EQUAL(XAIE_OK, RC);
}


TEST(MemoryApis, MemRdWordNegativePlTile)
{
	AieRC RC;
	u32 Data;

	XAie_LocType TileLoc;

	RC = XAie_DataMemRdWord(&DevInst, XAie_TileLoc(8, 0), 0x4000, &Data);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}

TEST(MemoryApis, MemRdWordNegativeTest)
{
	AieRC RC;
	u32 Data;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_DataMemRdWord(NULL, TileLoc, 0x4000, &Data);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_DataMemWrWord(&DevInst, TileLoc, 0x400000, Data);
	CHECK_EQUAL(XAIE_INVALID_DATA_MEM_ADDR, RC);

	RC = XAie_DataMemWrWord(NULL, TileLoc, 0x4000, Data);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_DataMemRdWord(&DevInst, TileLoc, 0x90000, &Data);
	CHECK_EQUAL(XAIE_INVALID_DATA_MEM_ADDR, RC);

	TileLoc = XAie_TileLoc(8, 0);

	RC = XAie_DataMemWrWord(&DevInst, TileLoc, 0x4000, Data);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	u32 tempStorage = 75;

	RC = XAie_DataMemBlockWrite(&DevInst, TileLoc, 0x0, &tempStorage, 2);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_DataMemBlockRead(&DevInst, TileLoc, 0x0,
			&tempStorage, 2);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_DataMemBlockWrite(NULL, TileLoc, 0xFFF,
			&tempStorage, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_DataMemBlockWrite(&DevInst, TileLoc, 0xFFFF,
			&tempStorage, 2);
	CHECK_EQUAL(XAIE_ERR_OUTOFBOUND, RC);

	RC = XAie_DataMemBlockRead(NULL, TileLoc, 0xFFF,
			&tempStorage, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_DataMemBlockRead(&DevInst, TileLoc, 0xFFFFFF,
			&tempStorage, 2);
	CHECK_EQUAL(XAIE_ERR_OUTOFBOUND, RC);

	/* Branch Coverage */
	RC = XAie_DataMemRdWord(&DevInst, TileLoc, 0x4000, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_DataMemBlockWrite(&DevInst, TileLoc, 0xFFF,
			NULL, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_DataMemBlockRead(&DevInst, TileLoc, 0xFFF,
			NULL, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 0;

	RC = XAie_DataMemWrWord(&DevInst, TileLoc, 0x4000,
			0xDEADBEEF);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_DataMemRdWord(&DevInst, TileLoc, 0x4000, &Data);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_DataMemBlockWrite(&DevInst, TileLoc, 0xFFF,
			&tempStorage, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_DataMemBlockRead(&DevInst, TileLoc, 0xFFF,
			&tempStorage, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 1;
}
