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

TEST_GROUP(LockApis)
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

#if AIE_GEN != 3
TEST(LockApis, ReleaseLockShimPlTile)
{
	AieRC RC;

	RC = XAie_LockRelease(&DevInst, XAie_TileLoc(1, 0), XAie_LockInit(5, 1),
			0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}
#endif

#if AIE_GEN >= 2
TEST(LockApis, LockSetValue)
{
	AieRC RC;

	RC = XAie_LockSetValue(&DevInst, XAie_TileLoc(8, 4),
			XAie_LockInit(5, 1));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_LockSetValue(&DevInst, XAie_TileLoc(8, 1),
			XAie_LockInit(11, 5));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_LockSetValue(&DevInst, XAie_TileLoc(2, 0),
			XAie_LockInit(3, 0));
	CHECK_EQUAL(RC, XAIE_OK);

}
#endif

TEST(LockApis, Negatives)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_LockAcquire(NULL, TileLoc, XAie_LockInit(5, 1), 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_LockAcquire(&DevInst, TileLoc, XAie_LockInit(25, 1), 0);
	CHECK_EQUAL(XAIE_INVALID_LOCK_ID, RC);

	RC = XAie_LockAcquire(&DevInst, TileLoc, XAie_LockInit(5, 150), 0);
	CHECK_EQUAL(XAIE_INVALID_LOCK_VALUE, RC);

	RC = XAie_LockRelease(NULL, TileLoc, XAie_LockInit(5, 1), 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_LockRelease(&DevInst, TileLoc, XAie_LockInit(25, 1), 0);
	CHECK_EQUAL(XAIE_INVALID_LOCK_ID, RC);

	RC = XAie_LockRelease(&DevInst, TileLoc, XAie_LockInit(5, 150), 0);
	CHECK_EQUAL(XAIE_INVALID_LOCK_VALUE, RC);

#if AIE_GEN != 3
	TileLoc = XAie_TileLoc(8,0);
	RC = XAie_LockRelease(&DevInst, TileLoc, XAie_LockInit(5, 1), 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_LockAcquire(&DevInst, TileLoc, XAie_LockInit(5, 1), 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
#endif

	DevInst.IsReady = 0;
	RC = XAie_LockAcquire(&DevInst, TileLoc, XAie_LockInit(5, 1), 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_LockRelease(&DevInst, TileLoc, XAie_LockInit(5, 1), 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 1;
}
