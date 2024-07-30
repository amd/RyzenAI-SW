/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "CppUTest/TestHarness.h"
#include <hw_config.h>

#if AIE_GEN != 4
static XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
		XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
		XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
		XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
		XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);

static XAie_InstDeclare(DevInst, &ConfigPtr);

TEST_GROUP(PmApis)
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

TEST(PmApis, PmAllTiles)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(1, 0);

	XAie_LocType TilesToRequest[4];
	TilesToRequest[0].Col = 3;
	TilesToRequest[0].Row = 5;
	TilesToRequest[1].Col = 5;
	TilesToRequest[1].Row = 3;

	RC = XAie_PmRequestTiles(&DevInst, TilesToRequest, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	
	RC = XAie_ResetPartition(&DevInst);
	CHECK_EQUAL(XAIE_OK, RC);
	

	RC = XAie_ClearPartitionMems(&DevInst);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PmRequestTiles(&DevInst, &TileLoc, 1);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PmRequestTiles(&DevInst, NULL, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_LocType Tiles[4];
	Tiles[0].Col = 3;
	Tiles[0].Row = 3;
	Tiles[1].Col = 3;
	Tiles[1].Row = 4;
	Tiles[2].Col = 2;
	Tiles[2].Row = 4;
	Tiles[3].Col = 2;
	Tiles[3].Row = 3;

	RC = XAie_PmRequestTiles(&DevInst, Tiles, 4);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_ClearPartitionMems(&DevInst);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST_GROUP(PmNegs)
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

TEST(PmNegs, InvalidArgs)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_PmRequestTiles(NULL, NULL, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PmRequestTiles(&DevInst, &TileLoc, 100022);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	XAie_LocType Tiles[4];
	Tiles[0].Col = 3;
	Tiles[0].Row = 3;
	Tiles[1].Col = 3;
	Tiles[1].Row = XAIE_NUM_ROWS+3;
	Tiles[2].Col = XAIE_NUM_COLS+2;
	Tiles[2].Row = 4;
	Tiles[3].Col = 2;
	Tiles[3].Row = 3;

	RC = XAie_PmRequestTiles(&DevInst, Tiles, 4);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	Tiles[1].Row = 3;
	Tiles[1].Col = XAIE_NUM_COLS + 2;

	RC = XAie_PmRequestTiles(&DevInst, Tiles, 4);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_ClearPartitionMems(NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_ResetPartition(NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 0;

	RC = XAie_ClearPartitionMems(&DevInst);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_ResetPartition(&DevInst);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PmRequestTiles(&DevInst, Tiles, 4);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 1;
}

#endif
