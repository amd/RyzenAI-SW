/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include  "CppUTest/TestHarness.h"
#include <hw_config.h>

static XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
		XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
		XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
		XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
		XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);

static XAie_InstDeclare(DevInst, &ConfigPtr);

TEST_GROUP(PerfCountApis)
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

TEST(PerfCountApis, PerfCountPlMod)
{

	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(1, 0);

	u32 CounterVal;

	RC = XAie_PerfCounterGet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, &CounterVal);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, XAIE_EVENT_PERF_CNT_0_PL, XAIE_EVENT_PERF_CNT_1_PL);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterResetControlSet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterSet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, 200);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterEventValueSet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, 1);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterEventValueReset(&DevInst, TileLoc, XAIE_PL_MOD,
			0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterReset(&DevInst, TileLoc, XAIE_PL_MOD, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterResetControlReset(&DevInst, TileLoc, XAIE_PL_MOD,
			0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterControlReset(&DevInst, TileLoc, XAIE_PL_MOD, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_Events StartEvent, StopEvent, ResetEvent;
	RC = XAie_PerfCounterGetControlConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			0, &StartEvent, &StopEvent, &ResetEvent);
	CHECK_EQUAL(XAIE_OK, RC);
}


TEST(PerfCountApis, PerfCountCoreMod)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(5, 4);

	u32 CounterVal;

	RC = XAie_PerfCounterGet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, &CounterVal);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, TileLoc, XAIE_CORE_MOD,0,
			XAIE_EVENT_PERF_CNT_0_CORE,
			XAIE_EVENT_PERF_CNT_1_CORE);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterResetControlSet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterSet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, 200);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterEventValueSet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, 1);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_Events StartEvent, StopEvent, ResetEvent;
	RC = XAie_PerfCounterGetControlConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, &StartEvent, &StopEvent, &ResetEvent);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(PerfCountApis, PerfCountMemMod)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(4, 5);

	u32 CounterVal;

	RC = XAie_PerfCounterGet(&DevInst, TileLoc, XAIE_MEM_MOD,
			0, &CounterVal);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, TileLoc, XAIE_MEM_MOD, 0,
			XAIE_EVENT_PERF_CNT_0_MEM,
			XAIE_EVENT_PERF_CNT_1_MEM);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterResetControlSet(&DevInst, TileLoc, XAIE_MEM_MOD,
			0, XAIE_EVENT_TRUE_MEM);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterSet(&DevInst, TileLoc, XAIE_MEM_MOD,
			0, 200);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterEventValueSet(&DevInst, TileLoc, XAIE_MEM_MOD,
			0, 1);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterEventValueReset(&DevInst, TileLoc, XAIE_CORE_MOD,
			0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterReset(&DevInst, TileLoc, XAIE_CORE_MOD, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterResetControlReset(&DevInst, TileLoc, XAIE_CORE_MOD,
			0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterControlReset(&DevInst, TileLoc, XAIE_CORE_MOD, 0);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(PerfCountApis, PerfCountMemTile)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 1);

	u32 CounterVal;

	RC = XAie_PerfCounterGet(&DevInst, TileLoc, XAIE_MEM_MOD,
			0, &CounterVal);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterSet(&DevInst, TileLoc, XAIE_MEM_MOD,
			0, 200);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterEventValueSet(&DevInst, TileLoc, XAIE_MEM_MOD,
			0, 1);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterEventValueReset(&DevInst, TileLoc, XAIE_MEM_MOD,
			0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterReset(&DevInst, TileLoc, XAIE_MEM_MOD, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterResetControlReset(&DevInst, TileLoc, XAIE_MEM_MOD,
			0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_PerfCounterControlReset(&DevInst, TileLoc, XAIE_MEM_MOD, 0);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST_GROUP(PerfcntNegs)
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


TEST(PerfcntNegs, InvalidArgs)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	u32 CounterVal;

	RC = XAie_PerfCounterGet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, &CounterVal);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, TileLoc, XAIE_PL_MOD, 0,
			XAIE_EVENT_PERF_CNT_0_PL,
			XAIE_EVENT_PERF_CNT_1_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterResetControlSet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterSet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, 200);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterEventValueSet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterGet(NULL, TileLoc, XAIE_PL_MOD,
			0, &CounterVal);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterGet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlSet(NULL, TileLoc, XAIE_PL_MOD, 0,
			XAIE_EVENT_PERF_CNT_0_PL,
			XAIE_EVENT_PERF_CNT_1_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterResetControlSet(NULL, TileLoc, XAIE_PL_MOD,
			0, XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterSet(NULL, TileLoc, XAIE_PL_MOD,
			0, 200);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterEventValueSet(NULL, TileLoc, XAIE_PL_MOD,
			0, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterResetControlReset(NULL, TileLoc, XAIE_PL_MOD,
			0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlReset(NULL, TileLoc, XAIE_PL_MOD, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	TileLoc = XAie_TileLoc(1, 0);

	RC = XAie_PerfCounterGet(&DevInst, TileLoc, XAIE_PL_MOD,
			16, &CounterVal);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, TileLoc, XAIE_PL_MOD, 16,
			XAIE_EVENT_PERF_CNT_0_PL,
			XAIE_EVENT_PERF_CNT_1_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterResetControlSet(&DevInst, TileLoc, XAIE_PL_MOD,
			16, XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterSet(&DevInst, TileLoc, XAIE_PL_MOD,
			16, 200);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterEventValueSet(&DevInst, TileLoc, XAIE_PL_MOD,
			16, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, TileLoc, XAIE_PL_MOD, 0,
			XAIE_EVENT_PERF_CNT_0_MEM,
			XAIE_EVENT_PERF_CNT_1_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, TileLoc, XAIE_PL_MOD, 0,
			XAIE_EVENT_PERF_CNT_1_PL,
			XAIE_EVENT_PERF_CNT_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, XAie_TileLoc(8, 3),
			XAIE_MEM_MOD, 0, XAIE_EVENT_PERF_CNT_1_PL,
			XAIE_EVENT_PERF_CNT_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, XAie_TileLoc(8, 3),
			XAIE_MEM_MOD, 0, XAIE_EVENT_PERF_CNT_0_MEM,
			XAIE_EVENT_PERF_CNT_1_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	/*Testing on Core Tile*/
	TileLoc = XAie_TileLoc(5, 4);
	RC = XAie_PerfCounterResetControlReset(&DevInst, TileLoc, XAIE_PL_MOD,
			0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlReset(&DevInst, TileLoc, XAIE_PL_MOD, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterResetControlSet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, XAIE_EVENT_PERF_CNT_0_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterResetControlSet(&DevInst, XAie_TileLoc(1, 0),
			XAIE_PL_MOD, 0, XAIE_EVENT_PERF_CNT_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterResetControlSet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, XAIE_EVENT_GROUP_0_MEM, XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, XAIE_EVENT_NONE_CORE, XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	XAie_Events StartEvent, StopEvent, ResetEvent;
	RC = XAie_PerfCounterGetControlConfig(NULL, TileLoc, XAIE_CORE_MOD,
			0, &StartEvent, &StopEvent, &ResetEvent);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterGetControlConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			0, &StartEvent, &StopEvent, &ResetEvent);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterGetControlConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			5, &StartEvent, &StopEvent, &ResetEvent);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 0;
	RC = XAie_PerfCounterGet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, &CounterVal);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, XAIE_EVENT_PERF_CNT_0_CORE,
			XAIE_EVENT_PERF_CNT_1_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterResetControlSet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterSet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, 200);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterEventValueSet(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterEventValueReset(&DevInst, TileLoc, XAIE_CORE_MOD,
			0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterReset(&DevInst, TileLoc, XAIE_CORE_MOD, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterResetControlReset(&DevInst, TileLoc, XAIE_CORE_MOD,
			0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterControlReset(&DevInst, TileLoc, XAIE_CORE_MOD, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PerfCounterGetControlConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			0, &StartEvent, &StopEvent, &ResetEvent);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 1;
}

TEST(PerfcntNegs, InvalidTile)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(XAIE_NUM_COLS + 5,
			XAIE_NUM_ROWS + 5);

	u32 CounterVal;

	RC = XAie_PerfCounterGet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, &CounterVal);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_PerfCounterControlSet(&DevInst, TileLoc, XAIE_PL_MOD,	0,
			XAIE_EVENT_PERF_CNT_0_PL,
			XAIE_EVENT_PERF_CNT_1_PL);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_PerfCounterResetControlSet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_PerfCounterSet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, 200);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_PerfCounterEventValueSet(&DevInst, TileLoc, XAIE_PL_MOD,
			0, 1);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_PerfCounterResetControlReset(&DevInst, TileLoc, XAIE_PL_MOD,
			0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_PerfCounterControlReset(&DevInst, TileLoc, XAIE_PL_MOD, 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	XAie_Events StartEvent, StopEvent, ResetEvent;
	RC = XAie_PerfCounterGetControlConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			0, &StartEvent, &StopEvent, &ResetEvent);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}
