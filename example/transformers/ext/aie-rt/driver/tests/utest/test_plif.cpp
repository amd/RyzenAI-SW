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

TEST_GROUP(PlIfApis)
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

TEST(PlIfApis, BliBypassEnable)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		if(i == 3) continue;
		RC = XAie_PlIfBliBypassEnable(&DevInst, XAie_TileLoc(8, 0), i);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, BliBypassEnableInvalidPortNum)
{
	AieRC RC;

	RC = XAie_PlIfBliBypassEnable(&DevInst, XAie_TileLoc(8, 0), 3);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);
}

TEST(PlIfApis, BliBypassDisable)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		if(i == 3) continue;
		RC = XAie_PlIfBliBypassDisable(&DevInst, XAie_TileLoc(8, 0), i);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, BliBypassDisableInvalidPortNum)
{
	AieRC RC;

	RC = XAie_PlIfBliBypassDisable(&DevInst, XAie_TileLoc(8, 0), 3);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);
}

TEST(PlIfApis, DownSzrEnable)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_PlIfDownSzrEnable(&DevInst, XAie_TileLoc(8, 0), i);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, DownSzrDisable)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_PlIfDownSzrDisable(&DevInst, XAie_TileLoc(8, 0), i);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, PlToAieIntfEnableWidth32)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_PlToAieIntfEnable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_32);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, PlToAieIntfEnableWidth64)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_PlToAieIntfEnable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_64);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, PlToAieIntfEnableWidth128)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_PlToAieIntfEnable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_128);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, PlToAieIntfDisableWidth32)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_PlToAieIntfDisable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_32);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, PlToAieIntfDisableWidth64)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_PlToAieIntfDisable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_64);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, PlToAieIntfDisableWidth128)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_PlToAieIntfDisable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_128);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, AieToPlIntfEnableWidth32)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_AieToPlIntfEnable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_32);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, AieToPlIntfEnableWidth64)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_AieToPlIntfEnable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_64);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, AieToPlIntfEnableWidth128)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_AieToPlIntfEnable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_128);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, AieToPlIntfDisableWidth32)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_AieToPlIntfDisable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_32);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, AieToPlIntfDisableWidth64)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_AieToPlIntfDisable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_64);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, AieToPlIntfDisableWidth128)
{
	AieRC RC;

	for(uint8_t i = 0; i < 7; i++) {
		RC = XAie_AieToPlIntfDisable(&DevInst, XAie_TileLoc(8, 0), i,
				PLIF_WIDTH_128);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, EnableShimDmaToAieStrmPort)
{
	AieRC RC;

	RC = XAie_EnableShimDmaToAieStrmPort(&DevInst, XAie_TileLoc(3, 0), 3);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(PlIfApis, EnableShimDmaToAieStrmPort7)
{
	AieRC RC;

	RC = XAie_EnableShimDmaToAieStrmPort(&DevInst, XAie_TileLoc(3, 0), 7);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(PlIfApis, EnableShimDmaToAieStrmPort2)
{
	AieRC RC;

	RC = XAie_EnableShimDmaToAieStrmPort(&DevInst, XAie_TileLoc(3, 0), 2);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);
}

TEST(PlIfApis, EnableAieToShimDmaStrmPort)
{
	AieRC RC;

	RC = XAie_EnableAieToShimDmaStrmPort(&DevInst, XAie_TileLoc(3, 0), 3);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(PlIfApis, EnableAieToShimDmaStrmPort2)
{
	AieRC RC;

	RC = XAie_EnableAieToShimDmaStrmPort(&DevInst, XAie_TileLoc(3, 0), 2);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(PlIfApis, EnableAieToShimDmaStrmPort7)
{
	AieRC RC;

	RC = XAie_EnableAieToShimDmaStrmPort(&DevInst, XAie_TileLoc(3, 0), 7);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);
}

TEST(PlIfApis, AieToNoc)
{
	AieRC RC;

	for(uint8_t i = 2; i < 6; i++) {
		RC = XAie_EnableAieToNoCStrmPort(&DevInst, XAie_TileLoc(3, 0),
				i);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, NocToAie)
{
	AieRC RC;

	for(uint8_t i = 2; i < 8; i++) {
		if((i == 4) || (i == 5)) continue;
		RC = XAie_EnableNoCToAieStrmPort(&DevInst, XAie_TileLoc(3, 0),
				i);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, AieToPl)
{
	AieRC RC;

	for(uint8_t i = 2; i < 6; i++) {
		RC = XAie_EnableAieToPlStrmPort(&DevInst, XAie_TileLoc(3, 0),
				i);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, PlToAie)
{
	AieRC RC;

	for(uint8_t i = 2; i < 8; i++) {
		if((i == 4) || (i == 5)) continue;
		RC = XAie_EnablePlToAieStrmPort(&DevInst, XAie_TileLoc(3, 0),
				i);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(PlIfApis, EnableAieToShimDmaStrmNegs)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(3, 0);

	RC = XAie_EnableAieToNoCStrmPort(&DevInst, TileLoc, 10);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_EnableNoCToAieStrmPort(&DevInst, TileLoc, 10);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	TileLoc = XAie_TileLoc(3, 5);
	RC = XAie_EnableAieToShimDmaStrmPort(&DevInst, TileLoc, 2);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EnableShimDmaToAieStrmPort(&DevInst, TileLoc, 3);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_PlToAieIntfEnable(&DevInst, TileLoc, 0, PLIF_WIDTH_64);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	TileLoc = XAie_TileLoc(3, 0);
	RC = XAie_PlToAieIntfEnable(NULL, TileLoc, 0, PLIF_WIDTH_64);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PlToAieIntfEnable(&DevInst, TileLoc, 10, PLIF_WIDTH_64);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);
}

TEST(PlIfApis, OtherNegs){
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(5, 4);

	RC = XAie_PlIfBliBypassEnable(NULL, TileLoc, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PlIfDownSzrEnable(NULL, TileLoc, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_AieToPlIntfEnable(NULL, TileLoc, 0, PLIF_WIDTH_32);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PlIfBliBypassEnable(&DevInst, TileLoc, 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_PlIfDownSzrEnable(&DevInst, TileLoc, 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_AieToPlIntfEnable(&DevInst, TileLoc, 0, PLIF_WIDTH_32);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	TileLoc = XAie_TileLoc(3, 0);

	RC = XAie_PlIfDownSzrEnable(&DevInst, TileLoc, 9);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_AieToPlIntfEnable(&DevInst, TileLoc, 9, PLIF_WIDTH_32);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_AieToPlIntfEnable(&DevInst, TileLoc, 0, (XAie_PlIfWidth)3);
	CHECK_EQUAL(XAIE_INVALID_PLIF_WIDTH, RC);

	RC = XAie_PlToAieIntfEnable(&DevInst, TileLoc, 0, (XAie_PlIfWidth)3);
	CHECK_EQUAL(XAIE_INVALID_PLIF_WIDTH, RC);

	RC = XAie_PlIfBliBypassEnable(&DevInst, TileLoc, 3);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_PlIfBliBypassEnable(&DevInst, TileLoc, 7);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	/* Dev Inst isn't ready */
	DevInst.IsReady = 0;

	RC = XAie_PlIfBliBypassEnable(&DevInst, TileLoc, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PlIfDownSzrEnable(&DevInst, TileLoc, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_AieToPlIntfEnable(&DevInst, TileLoc, 0, PLIF_WIDTH_32);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_PlToAieIntfEnable(&DevInst, TileLoc, 0, PLIF_WIDTH_32);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EnableShimDmaToAieStrmPort(&DevInst, TileLoc, 3);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EnableAieToNoCStrmPort(&DevInst, TileLoc, 10);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 1;
}
