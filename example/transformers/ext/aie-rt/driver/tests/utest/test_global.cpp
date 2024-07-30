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

TEST_GROUP(GlobalApis)
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

TEST(GlobalApis, PositiveTests)
{
	AieRC RC;
	XAie_MemInst* TestMemory;

	CHECK_EQUAL(DevInst.IsReady, XAIE_COMPONENT_IS_READY);

	RC = XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_MAX);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);
	
	TestMemory = XAie_MemAllocate(&DevInst, 5,
			XAIE_MEM_NONCACHEABLE);
	CHECK(TestMemory != NULL);

	RC = XAie_MemSyncForCPU(TestMemory);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_MemSyncForDev(TestMemory);
	CHECK_EQUAL(RC, XAIE_OK);

#if AIE_GEN < 3
	RC = XAie_TurnEccOff(&DevInst);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_TurnEccOn(&DevInst);
	CHECK_EQUAL(RC, XAIE_OK);
#endif
	
	RC = XAie_MemAttach(&DevInst, TestMemory,
			(u64)XAie_MemGetVAddr(TestMemory),
			XAie_MemGetDevAddr(TestMemory), 4,
			XAIE_MEM_CACHEABLE, 0);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_MemDetach(TestMemory);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_MemFree(TestMemory);
	CHECK_EQUAL(RC, XAIE_OK);
}

TEST(GlobalApis, InvalidDevInst){
	AieRC RC;

	XAie_SetupConfig(TestConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT,
			XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
			XAIE_SHIM_ROW, XAIE_MEM_TILE_ROW_START,
			XAIE_MEM_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
			XAIE_AIE_TILE_NUM_ROWS);

	XAie_InstDeclare(TestDevInst, &TestConfigPtr);

	RC = XAie_CfgInitialize(NULL, &TestConfigPtr);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CfgInitialize(&TestDevInst, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	XAie_SetupConfig(TestConfigPtr2, 10, XAIE_BASE_ADDR, XAIE_COL_SHIFT,
			XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
			XAIE_SHIM_ROW, XAIE_MEM_TILE_ROW_START,
			XAIE_MEM_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
			XAIE_AIE_TILE_NUM_ROWS);
	XAie_InstDeclare(TestDevInst2, &TestConfigPtr2);

	RC = XAie_CfgInitialize(&(TestDevInst2), &TestConfigPtr2);
	CHECK_EQUAL(XAIE_INVALID_DEVICE, RC);
}

TEST(GlobalApis, InvalidApis){
	AieRC RC;

	RC = XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_MAX);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_SetIOBackend(NULL, XAIE_IO_BACKEND_DEBUG);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	XAie_MemInst* TestMemory = XAie_MemAllocate(NULL, 5,
			XAIE_MEM_NONCACHEABLE);
	CHECK(TestMemory == NULL);

	RC = XAie_MemAttach(&DevInst, NULL,
			(u64)XAie_MemGetVAddr(TestMemory),
			XAie_MemGetDevAddr(TestMemory), 4,
			XAIE_MEM_CACHEABLE, 0);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_MemDetach(NULL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_MemSyncForCPU(NULL);
	CHECK_EQUAL(RC, XAIE_ERR);

	RC = XAie_MemFree(NULL);
	CHECK_EQUAL(RC, XAIE_ERR);

	RC = XAie_MemSyncForDev(NULL);
	CHECK_EQUAL(RC, XAIE_ERR);

	RC = XAie_Finish(NULL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_TurnEccOff(NULL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

#if AIE_GEN != 4
	RC = XAie_TurnEccOn(NULL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);
#endif
	
	TestMemory = XAie_MemAllocate(&DevInst, 5, XAIE_MEM_NONCACHEABLE);
	RC = XAie_MemAttach(NULL, TestMemory, 0x4000, 0x6000, 4,
			XAIE_MEM_CACHEABLE, 0);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	TestMemory->DevInst->IsReady = 0;
	RC = XAie_MemDetach(TestMemory);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);
	TestMemory->DevInst->IsReady = 1;

	DevInst.IsReady = 0;

	RC = XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_SIM);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	TestMemory = XAie_MemAllocate(&DevInst, 5,
			XAIE_MEM_NONCACHEABLE);
	CHECK(TestMemory == NULL);

#if AIE_GEN != 4
	RC = XAie_TurnEccOff(&DevInst);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_TurnEccOn(&DevInst);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);
#endif

	RC = XAie_Finish(&DevInst);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	DevInst.IsReady = 1;
}
