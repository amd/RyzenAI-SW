/******************************************************************************
* Copyright (C) 2019 - 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_dma_aie2ps.c
* @{
*
* This file contains routines for AIE2PS DMA configuration and controls.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who        Date     Changes
* ----- ------     -------- -----------------------------------------------------
* 1.0   Sankarji   10/04/2022  Initial creation
** </pre>
*
******************************************************************************/
/***************************** Include Files *********************************/
#include "xaie_feature_config.h"
#include "xaie_helper.h"
#include "xaie_io.h"
#include "xaiegbl_regdef.h"

#ifdef XAIE_FEATURE_DMA_ENABLE

/************************** Constant Definitions *****************************/
#define XAIE2PS_TILEDMA_NUM_BD_WORDS			6U
#define XAIE2PS_SHIMDMA_NUM_BD_WORDS			9U
#define XAIE2PS_MEMTILEDMA_NUM_BD_WORDS			8U
#define XAIE2PS_DMA_STEPSIZE_DEFAULT			1U
#define XAIE2PS_DMA_ITERWRAP_DEFAULT			1U
#define XAIE2PS_DMA_PAD_NUM_DIMS				3U

#define XAIE2PS_DMA_STATUS_IDLE					0x0U
#define XAIE2PS_DMA_STATUS_CHNUM_OFFSET			0x4U

/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
*
* This API writes a Dma Descriptor which is initialized and setup by other APIs
* into the corresponding registers and register fields in the hardware. This API
* is specific to AIE2PS Shim Tiles only.
*
* @param	DevInst: Device Instance
* @param	DmaDesc: Initialized Dma Descriptor.
* @param	Loc: Location of AIE Tile
* @param	BdNum: Hardware BD number to be written to.
*
* @return	XAIE_OK on success, Error code on failure.
*
* @note		Internal only. For AIE2PS Shim Tiles only.
*
******************************************************************************/
AieRC _XAie2PS_ShimDmaWriteBd(XAie_DevInst *DevInst , XAie_DmaDesc *DmaDesc,
		XAie_LocType Loc, u8 BdNum)
{
	u64 Addr;
	u64 BdBaseAddr;
	u32 BdWord[XAIE2PS_SHIMDMA_NUM_BD_WORDS];
	XAie_ShimDmaBdArgs Args;
	const XAie_DmaMod *DmaMod;
	const XAie_DmaBdProp *BdProp;

	DmaMod = DevInst->DevProp.DevMod[DmaDesc->TileType].DmaMod;
	BdProp = DmaMod->BdProp;

	BdBaseAddr = DmaMod->BaseAddr + BdNum * DmaMod->IdxOffset;

	/* Setup BdWord with the right values from DmaDesc */
	BdWord[0U] = XAie_SetField(DmaDesc->AddrDesc.Length,
			BdProp->BufferLen.Lsb,
			BdProp->BufferLen.Mask);

	BdWord[1U] = XAie_SetField(DmaDesc->AddrDesc.Address >>
				BdProp->Buffer->ShimDmaBuff.AddrLow.Lsb,
			BdProp->Buffer->ShimDmaBuff.AddrLow.Lsb,
			BdProp->Buffer->ShimDmaBuff.AddrLow.Mask);

	BdWord[2U] = XAie_SetField((DmaDesc->AddrDesc.Address >> 32U),
			BdProp->Buffer->ShimDmaBuff.AddrHigh.Lsb,
			BdProp->Buffer->ShimDmaBuff.AddrHigh.Mask) |
		XAie_SetField(DmaDesc->PktDesc.PktEn,
				BdProp->Pkt->EnPkt.Lsb,
				BdProp->Pkt->EnPkt.Mask) |
		XAie_SetField(DmaDesc->PktDesc.PktType,
				BdProp->Pkt->PktType.Lsb,
				BdProp->Pkt->PktType.Mask) |
		XAie_SetField(DmaDesc->PktDesc.PktId, BdProp->Pkt->PktId.Lsb,
				BdProp->Pkt->PktId.Mask) |
		XAie_SetField(DmaDesc->BdEnDesc.OutofOrderBdId,
				BdProp->BdEn->OutofOrderBdId.Lsb,
				BdProp->BdEn->OutofOrderBdId.Mask);

	BdWord[3U] = XAie_SetField(DmaDesc->MultiDimDesc.AieMlMultiDimDesc.DimDesc[0U].Wrap,
			BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Lsb,
			BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Mask) |
		XAie_SetField((DmaDesc->MultiDimDesc.AieMlMultiDimDesc.DimDesc[0U].StepSize - 1U),
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Mask) |
		XAie_SetField(DmaDesc->AxiDesc.SecureAccess,
				BdProp->SysProp->SecureAccess.Lsb,
				BdProp->SysProp->SecureAccess.Mask);
	BdWord[4U] = XAie_SetField(DmaDesc->MultiDimDesc.AieMlMultiDimDesc.DimDesc[1U].Wrap,
			BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Lsb,
			BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Mask) |
		XAie_SetField((DmaDesc->MultiDimDesc.AieMlMultiDimDesc.DimDesc[1U].StepSize - 1U),
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Mask) |
		XAie_SetField(DmaDesc->AxiDesc.BurstLen,
				BdProp->SysProp->BurstLen.Lsb,
				BdProp->SysProp->BurstLen.Mask);

	BdWord[5U] = XAie_SetField(DmaDesc->AxiDesc.SMID,
			BdProp->SysProp->SMID.Lsb, BdProp->SysProp->SMID.Mask) |
		XAie_SetField(DmaDesc->AxiDesc.AxQos,
				BdProp->SysProp->AxQos.Lsb,
				BdProp->SysProp->AxQos.Mask) |
		XAie_SetField(DmaDesc->AxiDesc.AxCache,
				BdProp->SysProp->AxCache.Lsb,
				BdProp->SysProp->AxCache.Mask) |
		XAie_SetField((DmaDesc->MultiDimDesc.AieMlMultiDimDesc.DimDesc[2U].StepSize - 1U),
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Mask);

	BdWord[6U] = XAie_SetField(DmaDesc->MultiDimDesc.AieMlMultiDimDesc.IterCurr,
			BdProp->AddrMode->AieMlMultiDimAddr.IterCurr.Lsb,
			BdProp->AddrMode->AieMlMultiDimAddr.IterCurr.Mask) |
		XAie_SetField((DmaDesc->MultiDimDesc.AieMlMultiDimDesc.IterDesc.Wrap - 1U),
				BdProp->AddrMode->AieMlMultiDimAddr.Iter.Wrap.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.Iter.Wrap.Mask) |
		XAie_SetField((DmaDesc->MultiDimDesc.AieMlMultiDimDesc.IterDesc.StepSize - 1U),
				BdProp->AddrMode->AieMlMultiDimAddr.Iter.StepSize.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.Iter.StepSize.Mask);

	BdWord[7U] = XAie_SetField(DmaDesc->BdEnDesc.ValidBd,
			BdProp->BdEn->ValidBd.Lsb, BdProp->BdEn->ValidBd.Mask) |
		XAie_SetField(DmaDesc->LockDesc.LockRelVal,
				BdProp->Lock->AieMlDmaLock.LckRelVal.Lsb,
				BdProp->Lock->AieMlDmaLock.LckRelVal.Mask) |
		XAie_SetField(DmaDesc->LockDesc.LockRelId,
				BdProp->Lock->AieMlDmaLock.LckRelId.Lsb,
				BdProp->Lock->AieMlDmaLock.LckRelId.Mask) |
		XAie_SetField(DmaDesc->LockDesc.LockAcqId,
				BdProp->Lock->AieMlDmaLock.LckAcqId.Lsb,
				BdProp->Lock->AieMlDmaLock.LckAcqId.Mask) |
		XAie_SetField(DmaDesc->LockDesc.LockAcqVal,
				BdProp->Lock->AieMlDmaLock.LckAcqVal.Lsb,
				BdProp->Lock->AieMlDmaLock.LckAcqVal.Mask) |
		XAie_SetField(DmaDesc->LockDesc.LockAcqEn,
				BdProp->Lock->AieMlDmaLock.LckAcqEn.Lsb,
				BdProp->Lock->AieMlDmaLock.LckAcqEn.Mask) |
		XAie_SetField(DmaDesc->BdEnDesc.UseNxtBd,
				BdProp->BdEn->UseNxtBd.Lsb,
				BdProp->BdEn->UseNxtBd.Mask) |
		XAie_SetField(DmaDesc->BdEnDesc.NxtBd, BdProp->BdEn->NxtBd.Lsb,
				BdProp->BdEn->NxtBd.Mask) |
		XAie_SetField(DmaDesc->TlastSuppress,
				BdProp->BdEn->TlastSuppress.Lsb,
				BdProp->BdEn->TlastSuppress.Mask);

	BdWord[8U] = XAie_SetField(DmaDesc->AddrDesc.Address >>
			BdProp->Buffer->ShimDmaBuff.AddrExtHigh.Lsb,
			BdProp->Buffer->ShimDmaBuff.AddrExtHigh.Lsb,
			BdProp->Buffer->ShimDmaBuff.AddrExtHigh.Mask);

	Addr = BdBaseAddr + _XAie_GetTileAddr(DevInst, Loc.Row, Loc.Col);

	Args.NumBdWords = XAIE2PS_SHIMDMA_NUM_BD_WORDS;
	Args.BdWords = &BdWord[0U];
	Args.Loc = Loc;
	Args.VAddr = DmaDesc->AddrDesc.Address;
	Args.BdNum = BdNum;
	Args.Addr = Addr;
	Args.MemInst = DmaDesc->MemInst;

	return XAie_RunOp(DevInst, XAIE_BACKEND_OP_CONFIG_SHIMDMABD, (void *)&Args);
}

/*****************************************************************************/
/**
*
* This API reads a the data from the buffer descriptor registers to fill the
* DmaDesc structure. This API is meant for AIE2PS Shim Tiles only.
*
* @param	DevInst: Device Instance
* @param	DmaDesc: Dma Descriptor to be filled.
* @param	Loc: Location of AIE Tile
* @param	BdNum: Hardware BD number to be read from.
*
* @return	XAIE_OK on success, Error code on failure.
*
* @note		Internal only. For AIE2PS Shim Tiles only.
*
******************************************************************************/
AieRC _XAie2PS_ShimDmaReadBd(XAie_DevInst *DevInst , XAie_DmaDesc *DmaDesc,
		XAie_LocType Loc, u8 BdNum)
{
	AieRC RC;
	u64 Addr;
	u64 BdBaseAddr;
	u32 BdWord[XAIE2PS_SHIMDMA_NUM_BD_WORDS];
	const XAie_DmaBdProp *BdProp;

	BdProp = DmaDesc->DmaMod->BdProp;
	BdBaseAddr = DmaDesc->DmaMod->BaseAddr +
			BdNum * DmaDesc->DmaMod->IdxOffset;
	Addr = BdBaseAddr + _XAie_GetTileAddr(DevInst, Loc.Row, Loc.Col);

	/* Setup DmaDesc with values read from bd registers */
	for(u8 i = 0; i < XAIE2PS_SHIMDMA_NUM_BD_WORDS; i++) {
		RC = XAie_Read32(DevInst, Addr, &BdWord[i]);
		if (RC != XAIE_OK) {
			return RC;
		}
		Addr += 4U;
	}

	DmaDesc->AddrDesc.Length = XAie_GetField(BdWord[0U],
				BdProp->BufferLen.Lsb,
				BdProp->BufferLen.Mask);

	DmaDesc->AddrDesc.Address |= (u64)XAie_GetField(BdWord[1U],
				BdProp->Buffer->ShimDmaBuff.AddrLow.Lsb,
				BdProp->Buffer->ShimDmaBuff.AddrLow.Mask) <<
				BdProp->Buffer->ShimDmaBuff.AddrLow.Lsb;

	DmaDesc->AddrDesc.Address |= (u64)XAie_GetField(BdWord[2U],
				BdProp->Buffer->ShimDmaBuff.AddrHigh.Lsb,
				BdProp->Buffer->ShimDmaBuff.AddrHigh.Mask) << 32U;
	DmaDesc->PktDesc.PktEn = XAie_GetField(BdWord[2U],
				BdProp->Pkt->EnPkt.Lsb,
				BdProp->Pkt->EnPkt.Mask);
	DmaDesc->PktDesc.PktType = XAie_GetField(BdWord[2U],
				BdProp->Pkt->PktType.Lsb,
				BdProp->Pkt->PktType.Mask);
	DmaDesc->PktDesc.PktId = XAie_GetField(BdWord[2U],
				BdProp->Pkt->PktId.Lsb,
				BdProp->Pkt->PktId.Mask);
	DmaDesc->BdEnDesc.OutofOrderBdId = XAie_GetField(BdWord[2U],
				BdProp->BdEn->OutofOrderBdId.Lsb,
				BdProp->BdEn->OutofOrderBdId.Mask);

	DmaDesc->MultiDimDesc.AieMlMultiDimDesc.DimDesc[0U].Wrap =
			XAie_GetField(BdWord[3U],
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Mask);
	DmaDesc->MultiDimDesc.AieMlMultiDimDesc.DimDesc[0U].StepSize = 1U +
			XAie_GetField(BdWord[3U],
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Mask);
	DmaDesc->AxiDesc.SecureAccess = XAie_GetField(BdWord[3U],
				BdProp->SysProp->SecureAccess.Lsb,
				BdProp->SysProp->SecureAccess.Mask);

	DmaDesc->MultiDimDesc.AieMlMultiDimDesc.DimDesc[1U].Wrap =
			XAie_GetField(BdWord[4U],
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Mask);
	DmaDesc->MultiDimDesc.AieMlMultiDimDesc.DimDesc[1U].StepSize = 1U +
			XAie_GetField(BdWord[4U],
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Mask);
	DmaDesc->AxiDesc.BurstLen = XAie_GetField(BdWord[4U],
				BdProp->SysProp->BurstLen.Lsb,
				BdProp->SysProp->BurstLen.Mask);

	DmaDesc->AxiDesc.SMID = XAie_GetField(BdWord[5U],
				BdProp->SysProp->SMID.Lsb,
				BdProp->SysProp->SMID.Mask);
	DmaDesc->AxiDesc.AxQos = XAie_GetField(BdWord[5U],
				BdProp->SysProp->AxQos.Lsb,
				BdProp->SysProp->AxQos.Mask);
	DmaDesc->AxiDesc.AxCache = XAie_GetField(BdWord[5U],
				BdProp->SysProp->AxCache.Lsb,
				BdProp->SysProp->AxCache.Mask);
	DmaDesc->MultiDimDesc.AieMlMultiDimDesc.DimDesc[2U].StepSize = 1U +
			XAie_GetField(BdWord[5U],
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Mask);

	DmaDesc->MultiDimDesc.AieMlMultiDimDesc.IterCurr =
			XAie_GetField(BdWord[6U],
				BdProp->AddrMode->AieMlMultiDimAddr.IterCurr.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.IterCurr.Mask);
	DmaDesc->MultiDimDesc.AieMlMultiDimDesc.IterDesc.Wrap = 1U +
			XAie_GetField(BdWord[6U],
				BdProp->AddrMode->AieMlMultiDimAddr.Iter.Wrap.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.Iter.Wrap.Mask);
	DmaDesc->MultiDimDesc.AieMlMultiDimDesc.IterDesc.StepSize = 1U +
			XAie_GetField(BdWord[6U],
				BdProp->AddrMode->AieMlMultiDimAddr.Iter.StepSize.Lsb,
				BdProp->AddrMode->AieMlMultiDimAddr.Iter.StepSize.Mask);

	DmaDesc->BdEnDesc.ValidBd = XAie_GetField(BdWord[7U],
				BdProp->BdEn->ValidBd.Lsb,
				BdProp->BdEn->ValidBd.Mask);
	DmaDesc->LockDesc.LockRelVal = XAie_GetField(BdWord[7U],
				BdProp->Lock->AieMlDmaLock.LckRelVal.Lsb,
				BdProp->Lock->AieMlDmaLock.LckRelVal.Mask);
	DmaDesc->LockDesc.LockRelId = XAie_GetField(BdWord[7U],
				BdProp->Lock->AieMlDmaLock.LckRelId.Lsb,
				BdProp->Lock->AieMlDmaLock.LckRelId.Mask);
	DmaDesc->LockDesc.LockAcqId = XAie_GetField(BdWord[7U],
				BdProp->Lock->AieMlDmaLock.LckAcqId.Lsb,
				BdProp->Lock->AieMlDmaLock.LckAcqId.Mask);
	DmaDesc->LockDesc.LockAcqVal = XAie_GetField(BdWord[7U],
				BdProp->Lock->AieMlDmaLock.LckAcqVal.Lsb,
				BdProp->Lock->AieMlDmaLock.LckAcqVal.Mask);
	DmaDesc->LockDesc.LockAcqEn = XAie_GetField(BdWord[7U],
				BdProp->Lock->AieMlDmaLock.LckAcqEn.Lsb,
				BdProp->Lock->AieMlDmaLock.LckAcqEn.Mask);
	DmaDesc->BdEnDesc.UseNxtBd = XAie_GetField(BdWord[7U],
				BdProp->BdEn->UseNxtBd.Lsb,
				BdProp->BdEn->UseNxtBd.Mask);
	DmaDesc->BdEnDesc.NxtBd = XAie_GetField(BdWord[7U],
				BdProp->BdEn->NxtBd.Lsb,
				BdProp->BdEn->NxtBd.Mask);
	DmaDesc->TlastSuppress = XAie_GetField(BdWord[7U],
				BdProp->BdEn->TlastSuppress.Lsb,
				BdProp->BdEn->TlastSuppress.Mask);

	DmaDesc->AddrDesc.Address |= XAie_GetField(BdWord[8U],
				BdProp->Buffer->ShimDmaBuff.AddrExtHigh.Lsb,
				BdProp->Buffer->ShimDmaBuff.AddrExtHigh.Mask) <<
				BdProp->Buffer->ShimDmaBuff.AddrExtHigh.Lsb;
	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This API updates the length of the buffer descriptor in the dma module.
*
* @param	DevInst: Device Instance.
* @param	DmaMod: Dma module pointer
* @param	Loc: Location of AIE Tile
* @param	Addr: Buffer address
* @param	BdNum: Hardware BD number to be written to.
*
* @return	XAIE_OK on success, Error code on failure.
*
* @note		Internal only. This API accesses the hardware directly and does
*		not operate on software descriptor.
******************************************************************************/
AieRC _XAie2PS_ShimDmaUpdateBdAddr(XAie_DevInst *DevInst,
		const XAie_DmaMod *DmaMod, XAie_LocType Loc, u64 Addr, u8 BdNum)
{
	AieRC RC;
	u64 RegAddr;
	u32 RegVal, Mask;

	RegAddr = DmaMod->BaseAddr + BdNum * DmaMod->IdxOffset +
		_XAie_GetTileAddr(DevInst, Loc.Row, Loc.Col) +
		DmaMod->BdProp->Buffer->ShimDmaBuff.AddrLow.Idx * 4U;

	Mask = DmaMod->BdProp->Buffer->ShimDmaBuff.AddrLow.Mask;
	RegVal = XAie_SetField(Addr,
			DmaMod->BdProp->Buffer->ShimDmaBuff.AddrLow.Lsb, Mask);

	/* Addrlow maps to a single register without other fields. */
	RC =  XAie_Write32(DevInst, RegAddr, RegVal);
	if(RC != XAIE_OK) {
		XAIE_ERROR("Failed to update lower 32 bits of address\n");
		return RC;
	}

	RegAddr = DmaMod->BaseAddr + BdNum * DmaMod->IdxOffset +
		_XAie_GetTileAddr(DevInst, Loc.Row, Loc.Col) +
		DmaMod->BdProp->Buffer->ShimDmaBuff.AddrHigh.Idx * 4U;

	Mask = DmaMod->BdProp->Buffer->ShimDmaBuff.AddrHigh.Mask;
	RegVal = XAie_SetField(Addr,
			DmaMod->BdProp->Buffer->ShimDmaBuff.AddrHigh.Lsb, Mask);

	/* Addrhigh bite 30_46 mapping. */
	RC = XAie_MaskWrite32(DevInst, RegAddr, Mask, RegVal);

	if(RC != XAIE_OK) {
		XAIE_ERROR("Failed to update 30_46 bits of address\n");
		return RC;
	}

	RegAddr = DmaMod->BaseAddr + BdNum * DmaMod->IdxOffset +
		_XAie_GetTileAddr(DevInst, Loc.Row, Loc.Col) +
		DmaMod->BdProp->Buffer->ShimDmaBuff.AddrExtHigh.Idx * 4U;

	Mask = DmaMod->BdProp->Buffer->ShimDmaBuff.AddrExtHigh.Mask;
	RegVal = XAie_SetField(Addr,
			DmaMod->BdProp->Buffer->ShimDmaBuff.AddrExtHigh.Lsb, Mask);

	return XAie_MaskWrite32(DevInst, RegAddr, Mask, RegVal);
}

#endif /* XAIE_FEATURE_DMA_ENABLE */

/** @} */
