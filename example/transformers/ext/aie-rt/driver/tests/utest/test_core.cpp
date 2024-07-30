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

TEST_GROUP(CoreControlApis)
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

TEST(CoreControlApis, CoreDisable)
{
	AieRC RC;

	RC = XAie_CoreDisable(&DevInst, XAie_TileLoc(8, 3));
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_CoreDisable(&DevInst, XAie_TileLoc(2, 0));
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

}


TEST(CoreControlApis, CoreEnable)
{
	AieRC RC;

	RC = XAie_CoreEnable(&DevInst, XAie_TileLoc(8, 3));
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(CoreControlApis, CoreEnableShimPL)
{
	AieRC RC;

	RC = XAie_CoreEnable(&DevInst, XAie_TileLoc(8, 0));
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}

#if AIE_GEN >= 2
TEST(CoreControlApis, CoreBusArrayTile)
{
	AieRC RC;

	RC = XAie_CoreProcessorBusEnable(&DevInst, XAie_TileLoc(8, 3));
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_CoreProcessorBusDisable(&DevInst, XAie_TileLoc(8, 3));
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(CoreControlApis, CoreBusShimTile)
{
	AieRC RC;

	RC = XAie_CoreProcessorBusEnable(&DevInst, XAie_TileLoc(8, 0));
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_CoreProcessorBusDisable(&DevInst, XAie_TileLoc(8, 0));
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}

TEST(CoreControlApis, CoreBusMemTile)
{
	AieRC RC;

	RC = XAie_CoreProcessorBusEnable(&DevInst, XAie_TileLoc(8, 1));
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_CoreProcessorBusDisable(&DevInst, XAie_TileLoc(8, 1));
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}

TEST(CoreControlApis, CoreEnableMemTile)
{
	AieRC RC;

	RC = XAie_CoreEnable(&DevInst, XAie_TileLoc(8, 1));
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}
#endif

#if AIE_GEN == 1
TEST(CoreControlApis, CoreBusAie)
{
	AieRC RC;

	RC = XAie_CoreProcessorBusEnable(&DevInst, XAie_TileLoc(8, 1));
	CHECK_EQUAL(XAIE_FEATURE_NOT_SUPPORTED, RC);

	RC = XAie_CoreProcessorBusDisable(&DevInst, XAie_TileLoc(8, 1));
	CHECK_EQUAL(XAIE_FEATURE_NOT_SUPPORTED, RC);
}
#endif

TEST(CoreControlApis, CoreWaitForDone)
{
	AieRC RC;

	RC = XAie_CoreWaitForDone(&DevInst, XAie_TileLoc(8, 3), 0);
	/*
	 * This test case will always fail as the core is not executing during
	 * the execution of this unit test case.
	 */
	CHECK_EQUAL(XAIE_CORE_STATUS_TIMEOUT, RC);

	RC = XAie_CoreWaitForDone(&DevInst, XAie_TileLoc(8, 3), 1);
	/*
	 * This test case will always fail as the core is not executing during
	 * the execution of this unit test case.
	 */
	CHECK_EQUAL(XAIE_CORE_STATUS_TIMEOUT, RC);
}

TEST(CoreControlApis, CoreWaitForDisable)
{
	AieRC RC;

	RC = XAie_CoreWaitForDisable(&DevInst, XAie_TileLoc(8, 3), 0);
	/*
	 * This test case will always as we are in reset state
	 */
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(CoreControlApis, ConfigEnableEvent)
{
	AieRC RC;

	RC = XAie_CoreConfigureEnableEvent(&DevInst, XAie_TileLoc(8, 3),
			XAIE_EVENT_USER_EVENT_0_CORE);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST_GROUP(CoreDebugApis)
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

TEST(CoreDebugApis, DebugHalt)
{
	AieRC RC;

	RC = XAie_CoreDebugHalt(&DevInst, XAie_TileLoc(8, 3));
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(CoreDebugApis, DebugUnhalt)
{
	AieRC RC;

	RC = XAie_CoreDebugUnhalt(&DevInst, XAie_TileLoc(8, 3));
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(CoreDebugApis, DebugHaltNegTests)
{
	AieRC RC;

	RC = XAie_CoreDebugHalt(NULL, XAie_TileLoc(8, 3));
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreDebugUnhalt(&DevInst, XAie_TileLoc(8, 0));
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}

TEST(CoreControlApis, ReadCoreDoneBit)
{
	AieRC RC;
	u8 DoneBit;

	RC = XAie_CoreReadDoneBit(&DevInst, XAie_TileLoc(8,3), &DoneBit);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(CoreControlApis, ReadCoreDoneBitNeg)
{
	AieRC RC;
	u8 DoneBit;

	RC = XAie_CoreReadDoneBit(NULL, XAie_TileLoc(8, 3), &DoneBit);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreReadDoneBit(&DevInst, XAie_TileLoc(8, 3), NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreReadDoneBit(&DevInst, XAie_TileLoc(8, 0), &DoneBit);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}

TEST(CoreControlApis, GetCoreStatusRegister)
{
	AieRC RC;
	u32 CoreStatus;

	RC = XAie_CoreGetStatus(&DevInst, XAie_TileLoc(8,3), &CoreStatus);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(CoreControlApis, GetCoreStatusRegisterNeg)
{
	AieRC RC;
	u32 CoreStatus;

	RC = XAie_CoreGetStatus(NULL, XAie_TileLoc(8, 3), &CoreStatus);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreGetStatus(&DevInst, XAie_TileLoc(8, 3), NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreGetStatus(&DevInst, XAie_TileLoc(8, 0), &CoreStatus);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}


TEST(CoreControlApis, CoreReset)
{
	AieRC RC;

	RC = XAie_CoreReset(&DevInst, XAie_TileLoc(8, 3));
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_CoreReset(&DevInst, XAie_TileLoc(4, 0));
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}

TEST(CoreControlApis, CoreUnreset)
{
	AieRC RC;

	RC = XAie_CoreUnreset(&DevInst, XAie_TileLoc(8, 3));
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_CoreUnreset(&DevInst, XAie_TileLoc(4, 0));
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}

TEST(CoreControlApis, CoreDebugStatus)
{
	AieRC RC;
	u32 DebugStatus;

	RC = XAie_CoreGetDebugHaltStatus(&DevInst, XAie_TileLoc(8, 3),
			&DebugStatus);
	CHECK_EQUAL(XAIE_OK, RC);

	DebugStatus = XAie_CheckDebugHaltStatus(DebugStatus,
			XAIE_CORE_DEBUG_STATUS_ANY_HALT);
	CHECK_EQUAL(0, DebugStatus);
}

TEST(CoreControlApis, CoreGetPCValue)
{
	AieRC RC;
	u32 PCValue;

	RC = XAie_CoreGetPCValue(&DevInst, XAie_TileLoc(8, 3), &PCValue);
	CHECK_EQUAL(XAIE_OK, RC);
}

#if AIE_GEN >= 2
TEST(CoreControlApis, CoreAccumCtrl)
{
	AieRC RC;

	RC = XAie_CoreConfigAccumulatorControl(&DevInst, XAie_TileLoc(6, 3), NORTH, SOUTH);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_CoreConfigAccumulatorControl(&DevInst, XAie_TileLoc(6, 3), WEST, EAST);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_CoreConfigAccumulatorControl(&DevInst, XAie_TileLoc(6, 3), NORTH, EAST);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_CoreConfigAccumulatorControl(&DevInst, XAie_TileLoc(6, 3), WEST, SOUTH);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_CoreConfigAccumulatorControl(&DevInst, XAie_TileLoc(6, 3), EAST, SOUTH);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreConfigAccumulatorControl(&DevInst, XAie_TileLoc(6, 3), NORTH, WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreConfigAccumulatorControl(&DevInst, XAie_TileLoc(6, 0), NORTH, SOUTH);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_CoreConfigAccumulatorControl(&DevInst, XAie_TileLoc(6, 1), NORTH, SOUTH);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_CoreConfigAccumulatorControl(NULL, XAie_TileLoc(6, 2), NORTH, SOUTH);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);
}
#endif

TEST(CoreControlApis, CoreControlNegs)
{
	AieRC RC;
	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_CoreDisable(NULL, TileLoc);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreEnable(NULL, TileLoc);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreReset(NULL, TileLoc);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreUnreset(NULL, TileLoc);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreWaitForDone(NULL, TileLoc, 0U);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreWaitForDisable(NULL, TileLoc, 0U);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreWaitForDone(&DevInst, XAie_TileLoc(8, 0), 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_CoreWaitForDisable(&DevInst, XAie_TileLoc(8, 0), 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_CoreConfigureEnableEvent(&DevInst, TileLoc,
			XAIE_EVENT_USER_EVENT_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreConfigureEnableEvent(&DevInst, XAie_TileLoc(8, 0),
			XAIE_EVENT_USER_EVENT_0_CORE);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_CoreConfigureEnableEvent(NULL, TileLoc,
			XAIE_EVENT_USER_EVENT_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 0;

	RC = XAie_CoreDisable(&DevInst, TileLoc);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreEnable(&DevInst, TileLoc);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreReset(&DevInst, TileLoc);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreUnreset(&DevInst, TileLoc);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreWaitForDone(&DevInst, TileLoc, 0U);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreWaitForDisable(&DevInst, TileLoc, 0U);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreDebugUnhalt(&DevInst, TileLoc);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreReadDoneBit(&DevInst, TileLoc, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	u8 DoneBit;

	TileLoc = XAie_TileLoc(8,3);
	RC = XAie_CoreReadDoneBit(&DevInst, TileLoc, &DoneBit);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_CoreGetStatus(&DevInst, TileLoc, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	u32 CoreStatus;

	TileLoc = XAie_TileLoc(8,3);
	RC = XAie_CoreGetStatus(&DevInst, TileLoc, &CoreStatus);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 1;
}

TEST_GROUP(ElfApis)
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

TEST(ElfApis, ElfLoad){
	AieRC RC;
	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

#if AIE_GEN == 1
	RC = XAie_LoadElf(&DevInst, TileLoc, "elf_files/aie_elfs/passthrough",
			1);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_LoadElf(&DevInst, TileLoc, "elf_files/aie_elfs/mmul", 1);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_LoadElf(&DevInst, TileLoc, "elf_files/aie_elfs/overlap_elf",
			1);
	CHECK_EQUAL(XAIE_OK, RC);

	DevInst.EccStatus = XAIE_DISABLE;

	RC = XAie_LoadElf(&DevInst, TileLoc, "elf_files/aie_elfs/passthrough",
			1);
	CHECK_EQUAL(XAIE_OK, RC);
	RC = XAie_LoadElfPartial(&DevInst, TileLoc,
			"elf_files/aie_elfs/large_elf",
			XAIE_LOAD_ELF_BSS | XAIE_LOAD_ELF_DATA);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_LoadElfPartial(&DevInst, TileLoc,
			"elf_files/aie_elfs/large_elf", XAIE_LOAD_ELF_TXT);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_LoadElfPartial(&DevInst, TileLoc,
			"elf_files/aie_elfs/large_elf", XAIE_LOAD_ELF_ALL);
	CHECK_EQUAL(XAIE_OK, RC);

#elif AIE_GEN == 2
	RC = XAie_LoadElf(&DevInst, TileLoc, "elf_files/aieml_elfs/passthrough",
			1);
	CHECK_EQUAL(XAIE_OK, RC);

#endif

	/*Testing invalid variants of APIS*/
	RC = XAie_LoadElf(NULL, TileLoc, "elf_files/aie_elfs/passthrough", 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_LoadElf(&DevInst, XAie_TileLoc(2, 0),
			"elf_files/aie_elfs/passthrough", 1);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_LoadElf(&DevInst, TileLoc, "elf_files/not_existing_file", 1);
	CHECK_EQUAL(XAIE_INVALID_ELF, RC);

	RC = XAie_LoadElf(&DevInst, TileLoc, "elf_files/emptyFile",
			1);
	CHECK_EQUAL(XAIE_ERR, RC);

	RC = XAie_LoadElf(&DevInst, TileLoc, NULL, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_LoadElfMem(&DevInst, XAie_TileLoc(2, 0),
			(unsigned char*)"path");
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_LoadElfMem(NULL, TileLoc, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_LoadElfMem(&DevInst, TileLoc, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 0;

	RC = XAie_LoadElf(&DevInst, TileLoc, "elf_files/aie_elfs/passthrough",
			1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_LoadElfMem(&DevInst, TileLoc,
			(unsigned char*)"elf_files/aie_elfs/passthrough");
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 1;
}
