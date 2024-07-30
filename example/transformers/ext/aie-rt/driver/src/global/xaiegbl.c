/******************************************************************************
* Copyright (C) 2019 - 2020 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaiegbl.c
* @{
*
* This file contains the global initialization functions for the Tile.
* This is applicable for both the AIE tiles and Shim tiles.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Tejus   09/24/2019  Initial creation
* 1.1   Tejus   10/22/2019  Enable AIE initilization
* 1.2   Tejus   06/09/2020  Call IO init api from XAie_CfgInitialize
* 1.3   Tejus   06/10/2020  Add api to change backend at runtime.
* 1.4   Dishita 07/28/2020  Add api to turn ECC On and Off.
* 1.5   Nishad  09/15/2020  Add check to validate XAie_MemCacheProp value in
*			    XAie_MemAllocate().
* </pre>
* @addtogroup AIEAPI AI Engine Software APIs
* @{
*
******************************************************************************/

/***************************** Include Files *********************************/
#include <string.h>
#include <stdlib.h>

#include "xaie_helper.h"
#include "xaie_io.h"
#include "xaie_io_internal.h"
#include "xaie_rsc_internal.h"
#include "xaiegbl.h"
#include "xaiegbl_defs.h"
#include "xaiegbl_regdef.h"

/************************** Constant Definitions *****************************/

/**************************** Type Definitions *******************************/

/**************************** Macro Definitions ******************************/
#define XAIE_ECC_BROADCAST_ID		6U

/************************** Variable Definitions *****************************/
extern XAie_TileMod AieMod[XAIEGBL_TILE_TYPE_MAX];
extern XAie_TileMod AieMlMod[XAIEGBL_TILE_TYPE_MAX];
extern XAie_TileMod Aie2IpuMod[XAIEGBL_TILE_TYPE_MAX];
extern XAie_TileMod Aie2PMod[XAIEGBL_TILE_TYPE_MAX];
extern XAie_TileMod Aie2PSMod[XAIEGBL_TILE_TYPE_MAX];

extern XAie_DeviceOps AieDevOps;
extern XAie_DeviceOps AieMlDevOps;
extern XAie_DeviceOps Aie2IpuDevOps;
extern XAie_DeviceOps Aie2PDevOps;
extern XAie_DeviceOps Aie2PSDevOps;

extern u8 XAieDevType;

#if XAIE_DEV_SINGLE_GEN == XAIE_DEV_GEN_AIE2IPU
#define XAIE_DEV_SINGLE_MOD Aie2IpuMod
#define XAIE_DEV_SINGLE_DEVOPS Aie2IpuDevOps
#elif XAIE_DEV_SINGLE_GEN == XAIE_DEV_GEN_AIEML
#define XAIE_DEV_SINGLE_MOD AieMlMod
#define XAIE_DEV_SINGLE_DEVOPS AieMlDevOps
#elif ((XAIE_DEV_SINGLE_GEN == XAIE_DEV_GEN_AIE2P) ||       \
		(XAIE_DEV_SINGLE_GEN == XAIE_DEV_GEN_AIE2P_STRIX_A0) || \
		(XAIE_DEV_SINGLE_GEN == XAIE_DEV_GEN_AIE2P_STRIX_B0))
#define XAIE_DEV_SINGLE_MOD Aie2PMod
#define XAIE_DEV_SINGLE_DEVOPS Aie2PDevOps
#elif XAIE_DEV_SINGLE_GEN == XAIE_DEV_GEN_AIE2PS
#define XAIE_DEV_SINGLE_MOD Aie2PSMod
#define XAIE_DEV_SINGLE_DEVOPS Aie2PSDevOps
#elif XAIE_DEV_SINGLE_GEN == XAIE_DEV_GEN_AIE
#define XAIE_DEV_SINGLE_MOD AieMod
#define XAIE_DEV_SINGLE_DEVOPS AieDevOps
#else
#ifdef XAIE_DEV_SINGLE_GEN
#error "Unsupported device defined."
#endif
#endif

/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
*
* This API is to set up AI engine partition instance location and size.
*
* @param	Inst: Pointer of AI engine partition instance
* @param	PartBaseAddr: Partition base address
* @param	PartStartCol: Absolute partition start column
* @param	PartNumCols: Number of columns of the partition
*
* @return	XAIE_OK.
*
* @note		If this API is not called, the AI engine partition instance
*		uses the base address and the number of columns from the device
*		config (XAie_Config), and the start column is 0 by default.
*		This function do not verify the PartBaseAddr, PartStartCol,
*		or the PartNumcols. The PartStartCol and the PartNumCols will
*		be validated in XAie_CfgInitialize().
*		This functions is supposed to be called before
*		XAie_CfgInitialize().
*
*******************************************************************************/
AieRC XAie_SetupPartitionConfig(XAie_DevInst *DevInst,
		u64 PartBaseAddr, u8 PartStartCol, u8 PartNumCols)
{
	if (DevInst == XAIE_NULL || (DevInst->IsReady != 0U)) {
		XAIE_ERROR("Invalid Device instance to set part config.\n");
		return XAIE_INVALID_DEVICE;
	}

	DevInst->BaseAddr = PartBaseAddr;
	DevInst->StartCol = PartStartCol;
	DevInst->NumCols = PartNumCols;

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the global initialization function for all the tiles of the AIE array
* The function sets up the Device Instance pointer with the appropriate values
* from the ConfigPtr.
*
* @param	InstPtr: Global AIE instance structure.
* @param	ConfigPtr: Global AIE configuration pointer.
*
* @return	XAIE_OK on success and error code on failure
*
* @note		This function needs to be called before calling any other AI
*		engine functions. After this function, as all tiles are gated
*		after system boots, XAie_PmRequestTiles() needs to be called
*		before calling other functions, otherwise, other functions
*		may access gated tiles.
*
******************************************************************************/
AieRC XAie_CfgInitialize(XAie_DevInst *InstPtr, XAie_Config *ConfigPtr)
{
	AieRC RC;

	if((InstPtr == XAIE_NULL) || (ConfigPtr == XAIE_NULL)) {
		XAIE_ERROR("Invalid input arguments\n",
				XAIE_INVALID_ARGS);
		return XAIE_INVALID_ARGS;
	}

	if (InstPtr->IsReady) {
		return XAIE_OK;
	}

	/* Initialize device property according to Device Type */
#ifdef XAIE_DEV_SINGLE_GEN
	if (ConfigPtr->AieGen == XAIE_DEV_SINGLE_GEN) {
		InstPtr->DevProp.DevMod = XAIE_DEV_SINGLE_MOD;
		InstPtr->DevProp.DevGen = XAIE_DEV_SINGLE_GEN;
		InstPtr->DevOps = &XAIE_DEV_SINGLE_DEVOPS;
#else
	if(ConfigPtr->AieGen == XAIE_DEV_GEN_AIEML) {
		InstPtr->DevProp.DevMod = AieMlMod;
		InstPtr->DevProp.DevGen = XAIE_DEV_GEN_AIEML;
		InstPtr->DevOps = &AieMlDevOps;
	} else if((ConfigPtr->AieGen == XAIE_DEV_GEN_AIE) ||
			(ConfigPtr->AieGen == XAIE_DEV_GEN_S100) ||
			(ConfigPtr->AieGen == XAIE_DEV_GEN_S200)) {
		InstPtr->DevProp.DevMod = AieMod;
		InstPtr->DevProp.DevGen = XAIE_DEV_GEN_AIE;
		InstPtr->DevOps = &AieDevOps;
		XAieDevType = (u8)ConfigPtr->AieGen;
	} else if(ConfigPtr->AieGen == XAIE_DEV_GEN_AIE2IPU) {
		InstPtr->DevProp.DevMod = Aie2IpuMod;
		InstPtr->DevProp.DevGen = XAIE_DEV_GEN_AIE2IPU;
		InstPtr->DevOps = &Aie2IpuDevOps;
	} else if(ConfigPtr->AieGen == XAIE_DEV_GEN_AIE2P) {
		InstPtr->DevProp.DevMod = Aie2PMod;
		InstPtr->DevProp.DevGen = XAIE_DEV_GEN_AIE2P;
		InstPtr->DevOps = &Aie2PDevOps;
	} else if(ConfigPtr->AieGen == XAIE_DEV_GEN_AIE2P_STRIX_A0){
		InstPtr->DevProp.DevMod = Aie2PMod;
		InstPtr->DevProp.DevGen = XAIE_DEV_GEN_AIE2P_STRIX_A0;
		InstPtr->DevOps = &Aie2PDevOps;
	} else if(ConfigPtr->AieGen == XAIE_DEV_GEN_AIE2P_STRIX_B0){
		InstPtr->DevProp.DevMod = Aie2PMod;
		InstPtr->DevProp.DevGen = XAIE_DEV_GEN_AIE2P_STRIX_B0;
		InstPtr->DevOps = &Aie2PDevOps;
	} else if(ConfigPtr->AieGen == XAIE_DEV_GEN_AIE2PS) {
		InstPtr->DevProp.DevMod = Aie2PSMod;
		InstPtr->DevProp.DevGen = XAIE_DEV_GEN_AIE2PS;
		InstPtr->DevOps = &Aie2PSDevOps;
#endif
	} else {
		XAIE_ERROR("Invalid device\n",
				XAIE_INVALID_DEVICE);
		return XAIE_INVALID_DEVICE;
	}

	if(InstPtr->NumCols == 0U) {
		InstPtr->BaseAddr = ConfigPtr->BaseAddr;
		InstPtr->StartCol = 0;
		InstPtr->NumCols = ConfigPtr->NumCols;
	} else if((u32)InstPtr->StartCol + (u32)InstPtr->NumCols >
			(u32)ConfigPtr->NumCols) {
		XAIE_ERROR("Invalid Partition location or size.\n");
		return XAIE_INVALID_DEVICE;
	}

	InstPtr->IsReady = XAIE_COMPONENT_IS_READY;
	InstPtr->DevProp.RowShift = ConfigPtr->RowShift;
	InstPtr->DevProp.ColShift = ConfigPtr->ColShift;
	InstPtr->NumRows = ConfigPtr->NumRows;
	InstPtr->ShimRow = ConfigPtr->ShimRowNum;
	InstPtr->MemTileRowStart = ConfigPtr->MemTileRowStart;
	InstPtr->MemTileNumRows = ConfigPtr->MemTileNumRows;
	InstPtr->AieTileRowStart = ConfigPtr->AieTileRowStart;
	InstPtr->AieTileNumRows = ConfigPtr->AieTileNumRows;
	InstPtr->TxnList.Next = NULL;

	if ((InstPtr->DevProp.DevGen == XAIE_DEV_GEN_AIE2IPU) ||
		(InstPtr->DevProp.DevGen == XAIE_DEV_GEN_AIE2P_STRIX_A0) ||
		(InstPtr->DevProp.DevGen == XAIE_DEV_GEN_AIE2P_STRIX_B0)) {
		InstPtr->EccStatus = XAIE_DISABLE;

	} else {
		InstPtr->EccStatus = XAIE_ENABLE;
	}

	RC = _XAie_RscMgrInit(InstPtr);
	if(RC != XAIE_OK) {
		return RC;
	}

	memcpy(&InstPtr->PartProp, &ConfigPtr->PartProp,
		sizeof(ConfigPtr->PartProp));

	RC = XAie_IOInit(InstPtr);
	if(RC != XAIE_OK) {
		return RC;
	}

	return XAIE_OK;
}

#if !defined(XAIE_FEATURE_LITE) && defined(XAIE_FEATURE_PRIVILEGED_ENABLE)
/*****************************************************************************/
/**
*
* This is the API to initialize the AI engine partition. It will initialize the
* AI engine partition hardware.
*
* @param	DevInst: Global AIE device instance pointer.
* @param	Opts: AI engine partition initialization options.
*			If @Opts is NULL, it will do the default options without
*			clock gating. The default options will:
*			* reset columns,
*			* reset shims,
*			* set to block NOC AXI MM decode and slave errors
*			* setup isolation
*			If @Opts is not NULL, it will follow the set bits of the
*			InitOpts field, the available options are as follows:
*			* XAIE_PART_INIT_OPT_DEFAULT
*			* XAIE_PART_INIT_OPT_COLUMN_RST
*			* XAIE_PART_INIT_OPT_SHIM_RST
*			* XAIE_PART_INIT_OPT_BLOCK_NOCAXIMMERR
*			* XAIE_PART_INIT_OPT_ISOLATE
*			* XAIE_PART_INIT_OPT_ZEROIZEMEM (not on by default)
*
* @return	XAIE_OK on success and error code on failure.
*
******************************************************************************/
AieRC XAie_PartitionInitialize(XAie_DevInst *DevInst, XAie_PartInitOpts *Opts)
{
	AieRC RC;

	if((DevInst == XAIE_NULL) ||
			(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	RC = XAie_RunOp(DevInst, XAIE_BACKEND_OP_PARTITION_INITIALIZE,
			(void *)Opts);
	if (RC != XAIE_OK) {
		XAIE_ERROR("Failed to initialize partition.\n");
		return RC;
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the API to teardown the AI engine partition. It will initialize
* the AI engine partition hardware.
*
* @param	DevInst: Global AIE device instance pointer.
*
* @return	XAIE_OK on success and error code on failure.
*
******************************************************************************/
AieRC XAie_PartitionTeardown(XAie_DevInst *DevInst)
{
	AieRC RC;

	if((DevInst == XAIE_NULL) ||
			(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	RC = XAie_RunOp(DevInst, XAIE_BACKEND_OP_PARTITION_TEARDOWN,
			NULL);
	if (RC != XAIE_OK) {
		XAIE_ERROR("Failed to teardown partition.\n");
		return RC;
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* The API clears partition context
*
* @param        DevInst: Global AIE device instance pointer.
*
* @return       XAIE_OK on success and error code on failure.
*
******************************************************************************/
AieRC XAie_ClearPartitionContext(XAie_DevInst *DevInst)
{
	AieRC RC;

	if((DevInst == XAIE_NULL) ||
			(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	RC = XAie_RunOp(DevInst, XAIE_BACKEND_OP_PARTITION_CLEAR_CONTEXT,
			NULL);
	if (RC != XAIE_OK) {
		XAIE_ERROR("Failed to clear context of partition\n");
		return RC;
	}

	return XAIE_OK;
}
#endif /* !XAIE_FEATURE_LITE && XAIE_FEATURE_PRIVILEGED_ENABLE */

/*****************************************************************************/
/**
*
* This is the API to setup the AI engine partition intialization. It is
* supposed to be called after the partition resets.
*
* @param	DevInst: Global AIE device instance pointer.
*
* @return	XAIE_OK on success and error code on failure.
*
* @note		This is a temporary wrapper. Users should migrate to use
* 		partition initialize function instead. This function will
* 		be deprecated in the future.
*
******************************************************************************/
AieRC _XAie_PartitionIsolationInitialize(XAie_DevInst *DevInst)
{
	if((DevInst == XAIE_NULL) ||
			(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	return DevInst->DevOps->SetPartIsolationAfterRst(DevInst);

}

/*****************************************************************************/
/**
*
* This is the API to finish the AI enigne partition. It will release
* the occupied AI engine resources.
*
* @param	DevInst: Global AIE device instance pointer.
*
* @return	XAIE_OK on success and error code on failure.
*
******************************************************************************/
AieRC XAie_Finish(XAie_DevInst *DevInst)
{
	const XAie_Backend *CurrBackend;
	AieRC RC;

	if((DevInst == XAIE_NULL) ||
			(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	/* Free transaction mode resources, if any */
	_XAie_TxnResourceCleanup(DevInst);

	CurrBackend = DevInst->Backend;
	RC = CurrBackend->Ops.Finish(DevInst->IOInst);
	if (RC != XAIE_OK) {
		XAIE_ERROR("Failed to close backend instance.\n");
		return RC;
	}

	RC = _XAie_RscMgrFinish(DevInst);
	if(RC != XAIE_OK) {
		return RC;
	}

	DevInst->IsReady = 0;

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the API to set the I/O backend of the driver at runtime.
*
* @param	DevInst: Global AIE device instance pointer.
* @param	Backend: Backend I/O type to switch to.
*
* @return	XAIE_OK on success and error code on failure.
*
******************************************************************************/
AieRC XAie_SetIOBackend(XAie_DevInst *DevInst, XAie_BackendType Backend)
{
	AieRC RC;
	const XAie_Backend *CurrBackend, *NewBackend;

	if((DevInst == XAIE_NULL) ||
			(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	if(Backend >= XAIE_IO_BACKEND_MAX ||
		_XAie_GetBackendPtr(Backend) == NULL) {
		XAIE_ERROR("Invalid backend request \n");
		return XAIE_INVALID_ARGS;
	}

	/* Release resources for current backend */
	CurrBackend = DevInst->Backend;
	RC = CurrBackend->Ops.Finish((void *)(DevInst->IOInst));
	if(RC != XAIE_OK) {
		XAIE_ERROR("Failed to close backend instance."
				"Falling back to backend %d\n",
				CurrBackend->Type);
		return RC;
	}

	/* Get new backend and initialize the backend */
	NewBackend = _XAie_GetBackendPtr(Backend);
	RC = NewBackend->Ops.Init(DevInst);
	if(RC != XAIE_OK) {
		XAIE_ERROR("Failed to initialize backend %d\n",
				Backend);
		return RC;
	}

	XAIE_DBG("Switching backend to %d\n", Backend);
	DevInst->Backend = NewBackend;

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the memory function to allocate a memory
*
* @param	DevInst: Device Instance
* @param	Size: Size of the memory
* @param	Cache: Buffer to be cacheable or not
*
* @return	Pointer to the allocated memory instance.
*
*******************************************************************************/
XAie_MemInst* XAie_MemAllocate(XAie_DevInst *DevInst, u64 Size,
		XAie_MemCacheProp Cache)
{
	const XAie_Backend *Backend;

	if((DevInst == XAIE_NULL) ||
			(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return NULL;
	}

	if(Cache > XAIE_MEM_NONCACHEABLE) {
		XAIE_ERROR("Invalid cache property\n");
		return NULL;
	}

	Backend = DevInst->Backend;

	return Backend->Ops.MemAllocate(DevInst, Size, Cache);
}

/*****************************************************************************/
/**
*
* This is the memory function to free the memory
*
* @param	MemInst: Memory instance pointer.
*
* @return	XAIE_OK on success, Error code on failure.
*
*******************************************************************************/
AieRC XAie_MemFree(XAie_MemInst *MemInst)
{
	const XAie_Backend *Backend;

	if(MemInst == XAIE_NULL) {
		XAIE_ERROR("Invalid memory instance\n");
		return XAIE_ERR;
	}

	Backend = MemInst->DevInst->Backend;

	return Backend->Ops.MemFree(MemInst);
}

/*****************************************************************************/
/**
*
* This is the memory function to sync the memory for CPU
*
* @param	MemInst: Memory instance pointer.
*
* @return	XAIE_OK on success, Error code on failure.
*
*******************************************************************************/
AieRC XAie_MemSyncForCPU(XAie_MemInst *MemInst)
{
	const XAie_Backend *Backend;

	if(MemInst == XAIE_NULL) {
		XAIE_ERROR("Invalid memory instance\n");
		return XAIE_ERR;
	}

	Backend = MemInst->DevInst->Backend;

	return Backend->Ops.MemSyncForCPU(MemInst);
}

/*****************************************************************************/
/**
*
* This is the memory function to sync the memory for device
*
* @param	MemInst: Memory instance pointer.
*
* @return	XAIE_OK on success, Error code on failure.
*
*******************************************************************************/
AieRC XAie_MemSyncForDev(XAie_MemInst *MemInst)
{
	const XAie_Backend *Backend;

	if(MemInst == XAIE_NULL) {
		XAIE_ERROR("Invalid memory instance\n");
		return XAIE_ERR;
	}

	Backend = MemInst->DevInst->Backend;

	return Backend->Ops.MemSyncForDev(MemInst);
}

/*****************************************************************************/
/**
*
* This is the memory function to return the virtual address of the memory
* instance
*
* @param	MemInst: Memory instance pointer.
*
* @return	Mapped virtual address of the memory instance.
*
*******************************************************************************/
void* XAie_MemGetVAddr(XAie_MemInst *MemInst)
{
	if(MemInst == XAIE_NULL) {
		XAIE_ERROR("Invalid memory instance\n");
		return NULL;
	}

	return MemInst->VAddr;
}

/*****************************************************************************/
/**
*
* This is the memory function to return the physical address of the memory
* instance
*
* @param	MemInst: Memory instance pointer.
*
* @return	Physical address of the memory instance.
*
*******************************************************************************/
u64 XAie_MemGetDevAddr(XAie_MemInst *MemInst)
{
	if(MemInst == XAIE_NULL) {
		XAIE_ERROR("Invalid memory instance\n");
		return 1U;
	}

	return MemInst->DevAddr;
}

/*****************************************************************************/
/**
*
* This is the memory function to attach user allocated memory to the AI engine
* partition device instance.
*
* @param	DevInst: Device Instance
* @param	MemInst: Pointer to memory instance which will be filled with
*			 attached AI engine memory instance information by
*			 this function.
* @param	DevAddr: Device address of the allocated memory. It is usually
*			 the physical address of the memory. For Linux dmabuf
*			 memory, it is the offset to the start of the dmabuf.
* @param	VAddr: Virtual address of the allocated memory. For Linux
*		       dmabuf memory, it is not required, it can be NULL.
*		Cache: Buffer is cacheable or not.
*		MemHandle: Handle of the allocated memory. It is ignored for
*			   other backends except Linux backend. For Linux
*			   backend, it is the file descriptor of a dmabuf.
*
* @return	XAIE_OK for success, or error code for failure.
*
*******************************************************************************/
AieRC XAie_MemAttach(XAie_DevInst *DevInst, XAie_MemInst *MemInst, u64 DevAddr,
		u64 VAddr, u64 Size, XAie_MemCacheProp Cache, u64 MemHandle)
{
	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY) ||
		(DevInst->Backend == XAIE_NULL)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	if(MemInst == XAIE_NULL) {
		XAIE_ERROR("Invalid memory instance\n");
		return XAIE_INVALID_ARGS;
	}

	if(Cache > XAIE_MEM_NONCACHEABLE) {
		XAIE_ERROR("Invalid cache property\n");
		return XAIE_INVALID_ARGS;
	}

	MemInst->DevInst = DevInst;
	MemInst->VAddr = (void *)(uintptr_t)VAddr;
	MemInst->DevAddr = DevAddr;
	MemInst->Size = Size;
	MemInst->Cache = Cache;

	return DevInst->Backend->Ops.MemAttach(MemInst, MemHandle);
}

/*****************************************************************************/
/**
*
* This is the memory function to dettach user allocated memory from the AI engine
* partition device instance.
*
* @param	MemInst: Memory Instance
*
* @return	XAIE_OK for success, and error code for failure.
*
*******************************************************************************/
AieRC XAie_MemDetach(XAie_MemInst *MemInst)
{
	XAie_DevInst *DevInst;

	if(MemInst == XAIE_NULL) {
		XAIE_ERROR("Invalid memory instance\n");
		return XAIE_INVALID_ARGS;
	}

	DevInst = MemInst->DevInst;
	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY) ||
		(DevInst->Backend == XAIE_NULL)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	return DevInst->Backend->Ops.MemDetach(MemInst);
}

/*****************************************************************************/
/*
* This API disables the ECC flag in the Device Instance of the partition. It
* should be called before calling elf loader to disable ECC. ECC configuration
* is done from elf loader.
*
* @param        DevInst: Device Instance
*
* @return       XAIE_OK on success
*
*******************************************************************************/
AieRC XAie_TurnEccOff(XAie_DevInst *DevInst)
{
	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	DevInst->EccStatus = XAIE_DISABLE;

	return XAIE_OK;
}

/*****************************************************************************/
/*
* This API enables the ECC flag in the Device Instance of the partition. ECC
* configuration is done from elf loader.
*
* @param        DevInst: Device Instance
*
* @return       XAIE_OK on success
*
*******************************************************************************/
AieRC XAie_TurnEccOn(XAie_DevInst *DevInst)
{
	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	if ((DevInst->DevProp.DevGen == XAIE_DEV_GEN_AIE2IPU) ||
		(DevInst->DevProp.DevGen == XAIE_DEV_GEN_AIE2P_STRIX_A0) ||
		(DevInst->DevProp.DevGen == XAIE_DEV_GEN_AIE2P_STRIX_B0)) {
		XAIE_ERROR("ECC feature not supported\n");
		return XAIE_FEATURE_NOT_SUPPORTED;
	}

	DevInst->EccStatus = XAIE_ENABLE;

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This API starts the execution of the driver in transaction mode. All the
* resulting I/O operations are stored in an internally managed buffer. The user
* has to explicitly submit the transaction for the driver to execute the I/O
* operations to configure the device. The transaction instance allocated by this
* API is tied to the thread ID of the executing context. SubmitTransaction API
* must be called from the same context with NULL for the TxnInst parameter.
*
* @param	DevInst: Device instance pointer.
* @param	Flags: Flags passed by the user.
*			XAIE_TRANSACTION_ENABLE/DISBALE_AUTO_FLUSH
*
* @return	XAIE_OK on success and error code on failure.
*
* @note		If the ENABLE_AUTO_FLUSH flag is set, the driver will
*		automatically flush the transaction buffer when an API results
*		in Read/MaskPoll/BlockWrite/BlockSet/CmdWrite/RunOp operation.
*		If the DISABLE_AUTO_FLUSH flag is set, the driver will return an
*		error when an API results in Read/MaskPoll/CmdWrite/RunOp
*		operation. In both cases, the user has to call
*		XAie_SubmitTransaction API to flush all the pending I/O
*		operations stored in the command buffer.
*
******************************************************************************/
AieRC XAie_StartTransaction(XAie_DevInst *DevInst, u32 Flags)
{
	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid arguments\n");
		return XAIE_INVALID_ARGS;
	}

	return _XAie_Txn_Start(DevInst, Flags);
}

/*****************************************************************************/
/**
*
* This API executes all the pending I/O operations stored in the command buffer.
* The transaction instance returned by the StartTransaction API is tied to the
* thread ID of the executing context. If TxnInst is NULL, the transaction
* instance is automatically fetched using the thread ID of the current context.
* If the TxnInst is not NULL, the transaction instance passed by the user is
* executed.
*
* @param	DevInst: Device instance pointer.
* @param	TxnInst: Transaction instance pointer.
*
* @return	XAIE_OK on success and Error code or failure.
*
******************************************************************************/
AieRC XAie_SubmitTransaction(XAie_DevInst *DevInst, XAie_TxnInst *TxnInst)
{
	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid arguments\n");
		return XAIE_INVALID_ARGS;
	}

	return _XAie_Txn_Submit(DevInst, TxnInst);
}

/*****************************************************************************/
/**
*
* This API copies an existing transaction instance and returns a copy of the
* instance with all the commands for users to save the commands and use them
* at a later point.
*
* @param	DevInst: Device instance pointer.
*
* @return	Pointer to copy of transaction instance on success and NULL
*		on error.
*
* @note		The copy of the transaction instance must be explicitly freed
*		using the XAie_FreeTransactionInstance API. If Auto flush was
*		enabled during the creating of the initial transaction, the
*		instance returned by this API will not have the commands that
*		are already flushed. The transaction instance must be exported
*		before it is submitted as the XAie_SubmitTransaction API will
*		free all the resources associated with it.
*
******************************************************************************/
XAie_TxnInst* XAie_ExportTransactionInstance(XAie_DevInst *DevInst)
{
	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid arguments\n");
		return NULL;
	}

	return _XAie_TxnExport(DevInst);
}

/*****************************************************************************/
/**
*
* This API copies an existing transaction instance and returns a copy of the
* instance with all the commands for users to save the commands and use them
* at a later point.
*
* @param	DevInst: Device instance pointer.
* @param	NumConsumers: Number of consumers for the generated
*		transactions (Unused for now)
* @param	Flags: Flags (Unused for now)
*
* @return	Pointer to copy of transaction instance on success and NULL
*		on error.
*
******************************************************************************/
u8* XAie_ExportSerializedTransaction_opt(XAie_DevInst *DevInst,
		u8 NumConsumers, u32 Flags)
{
	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid arguments\n");
		return NULL;
	}

    return _XAie_TxnExportSerialized_opt(DevInst, NumConsumers, Flags);
}


/*****************************************************************************/
/**
*
* This API releases the memory resources used by exported transaction instance.
*
* @param	TxnInst: Existing Transaction instance
*
* @return	XAIE_OK on success or error code on failure.
*
******************************************************************************/
AieRC XAie_FreeTransactionInstance(XAie_TxnInst *TxnInst)
{
	if(TxnInst == NULL) {
		XAIE_ERROR("Invalid arguments\n");
		return XAIE_INVALID_ARGS;
	}

	return _XAie_TxnFree(TxnInst);
}

/*****************************************************************************/
/**
*
* This is the API to return if the device generation has checkboarded tiles for
* broadcast ungating
*
* @param	DevInst: Global AIE device instance pointer.
* @param	IsCheckerBoard: Pointer to be set if tile is checkerboarded
* @return	XAIE_OK on success
*
******************************************************************************/
AieRC XAie_IsDeviceCheckerboard(XAie_DevInst *DevInst, u8 *IsCheckerBoard)
{
	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY) ||
		(IsCheckerBoard == NULL)) {
		XAIE_ERROR("Invalid arguments\n");
		return XAIE_INVALID_ARGS;
	}

	*IsCheckerBoard = DevInst->DevOps->IsCheckerBoard;

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the API updates the base NPI address.
*
* @param	DevInst: Global AIE device instance pointer.
* @param	NpiAddr: Base NPI address
* @return	XAIE_OK on success error code on failure.
*
******************************************************************************/
AieRC XAie_UpdateNpiAddr(XAie_DevInst *DevInst, u64 NpiAddr)
{
	if((DevInst == NULL) || (DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	return XAie_RunOp(DevInst, XAIE_BACKEND_OP_UPDATE_NPI_ADDR,
			(void *)&NpiAddr);
}

/*****************************************************************************/
/**
*
* This API copies an existing transaction instance and returns a copy of the
* instance with all the commands for users to save the commands and use them
* at a later point.
*
* @param	DevInst: Device instance pointer.
* @param	NumConsumers: Number of consumers for the generated
*		transactions (Unused for now)
* @param	Flags: Flags (Unused for now)
*
* @return	Pointer to copy of transaction instance on success and NULL
*		on error.
*
******************************************************************************/
u8* XAie_ExportSerializedTransaction(XAie_DevInst *DevInst,
		u8 NumConsumers, u32 Flags)
{
	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid arguments\n");
		return NULL;
	}

	return _XAie_TxnExportSerialized(DevInst, NumConsumers, Flags);
}

/*****************************************************************************/
/**
*
* This API deallocates the memory allocated for the serialized transaction
* buffer.
*
* @param	Ptr: Pointer to the transaction buffer.
*
******************************************************************************/
void XAie_FreeSerializedTransaction(void *Ptr)
{
	if (Ptr == NULL) {
		XAIE_ERROR("Invalid argument\n");
		return;
	}
	_XAie_FreeTxnPtr(Ptr);
}

AieRC XAie_ClearTransaction(XAie_DevInst* DevInst)
{
	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid arguments\n");
		return XAIE_INVALID_ARGS;
	}

	return _XAie_ClearTransaction(DevInst);
}

/*****************************************************************************/
/**
* This function configures the attribute for the backend.
*
* @param	DevInst: Device instance pointer.
* @param	AttrType: Backend attribute type.
* @param	AttrVal: Backend attribute value.
*
* @return	XAIE_OK on success and error code on failure
*
******************************************************************************/
AieRC XAie_ConfigBackendAttr(XAie_DevInst *DevInst,
		XAie_BackendAttrType AttrType, u64 AttrVal)
{
	if((DevInst == XAIE_NULL) || (DevInst->Backend->Ops.SetAttr == XAIE_NULL)) {
		return XAIE_INVALID_ARGS;
	}
	return DevInst->Backend->Ops.SetAttr(DevInst->IOInst, AttrType, AttrVal);
}

/*****************************************************************************/
/**
* This function captures kernel utilization of the core tiles mentioned in the
* columns in range in PerfInst.
*
* @param	DevInst: Device instance pointer.
* @param	PerfInst: Performance instance pointer.
* @return	XAIE_OK on success and error code on failure.
*
* @note		If Range in PerfInst is NULL, all the columns in the partition
*		will be scaned to gather all the core tiles in the partition.
*
******************************************************************************/
AieRC XAie_PerfUtilization(XAie_DevInst *DevInst, XAie_PerfInst *PerfInst)
{

	AieRC RC = XAIE_OK;
	XAie_Range PartRange;
	u32 Size, NumTiles;

	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid arguments\n");
		return XAIE_INVALID_ARGS;
	}

	if(PerfInst == XAIE_NULL) {
		XAIE_ERROR("Invalid arguments\n");
		return XAIE_INVALID_ARGS;
	}

	if(PerfInst->Range == XAIE_NULL) {
		PartRange.Start = DevInst->StartCol;
		PartRange.Num = DevInst->NumCols;
		XAIE_DBG("Start Col: %d\tnum: %d\n",
				PartRange.Start, PartRange.Num);
		PerfInst->Range = &PartRange;
	} else if (PerfInst->Range->Num <= 0U ||
			PerfInst->Range->Num > DevInst->NumCols) {
		XAIE_ERROR("Invalid range!\n");
		return XAIE_INVALID_ARGS;
	} else if(PerfInst->Range->Start >= DevInst->NumCols) {
		XAIE_ERROR("Invalid range!\n");
		return XAIE_INVALID_ARGS;
	}

	Size = (u32)sizeof(XAie_Occupancy);
	NumTiles = (DevInst->NumCols) * DevInst->AieTileNumRows;
	if((PerfInst->UtilSize)	< (NumTiles * Size)) {
		XAIE_ERROR("Insufficient Buffer Size!\n");
		return XAIE_INSUFFICIENT_BUFFER_SIZE;
	}

	/*
	 * PerfInst->UtilSize will contain the number of elements hereforth.
	 */
	PerfInst->UtilSize =(u32)(PerfInst->UtilSize/sizeof(XAie_Occupancy));

	/*
	 * By default kernel utilization is captured over a time interval of
	 * 1 ms.
	 */
	if(PerfInst->TimeInterval_ms == 0U) {
		XAIE_WARN("Capturing for 1ms as minimum time interval is 1ms!\n");
		PerfInst->TimeInterval_ms = 1;
	} else if(PerfInst->TimeInterval_ms > 3000U) {
		XAIE_WARN("Capturing for 3000ms as maximum time interval is 3000ms!\n");
		PerfInst->TimeInterval_ms = 3000;
	}

	RC = XAie_RunOp(DevInst, XAIE_BACKEND_OP_PERFORMANCE_UTILIZATION,
			(void *)PerfInst);
	if(RC != XAIE_OK) {
		XAIE_ERROR("Failed to capture kernel utilization.\n");
		return RC;
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
 *
 * This API is to enable/disable memory interleaving mode in all MemTiles of AI
 * engine partition.
 *
 * @param	DevInst - Global AIE device instance pointer.
 * @param	Locs - Pointer to tiles locatations
 * @param	NumTiles - Number of tiles
 * @param	Enable - 0/1 to Disable/Enable memory interleaving.
 *
 * @return	XAIE_OK on success and error code on failure.
 *
 * @note		None.
 *
 ******************************************************************************/
AieRC XAie_ConfigMemTilesMemInterleaving(XAie_DevInst *DevInst,
		XAie_LocType *Locs, u32 NumTiles, u8 Enable)
{
	AieRC RC;
	XAie_BackendTilesEnableArray Tiles;
	u32 i;

	if((DevInst == XAIE_NULL) ||
	   (DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid Device Instance\n");
		return XAIE_INVALID_ARGS;
	}

	/* Verify Locations */
	for (i = 0; i < NumTiles; i++) {
		if ((Locs[i].Row < DevInst->MemTileRowStart) ||
		    (Locs[i].Row > (DevInst->MemTileRowStart + DevInst->MemTileNumRows)) ||
		    (Locs[i].Col > DevInst->NumCols)) {
			XAIE_ERROR("Wrong Location of tile Loc (%d, %d)\n",
					Locs[i].Row, Locs[i].Col);
			return XAIE_INVALID_TILE;
		}
	}

	Tiles.Locs = Locs;
	Tiles.NumTiles = NumTiles;
	Tiles.Enable = Enable;

	RC = XAie_RunOp(DevInst, XAIE_BACKEND_OP_CONFIG_MEM_INTRLVNG,
			(void *)&Tiles);
	if (RC != XAIE_OK) {
		XAIE_ERROR("Failed to configure memory interleaving.\n");
		return RC;
	}

	return XAIE_OK;
}

/** @}@} */
