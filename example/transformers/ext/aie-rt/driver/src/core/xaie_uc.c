/******************************************************************************
* Copyright (C) 2019 - 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_uc.c
* @{
*
* The file has implementations of routines for uC loading.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Kishan  12/23/2022  Initial creation
* </pre>
*
******************************************************************************/
/***************************** Include Files *********************************/
#include <errno.h>

#include "xaie_uc.h"
#include "xaie_elfloader.h"
#include "xaie_core_aie.h"
#include "xaie_feature_config.h"
#include "xaie_ecc.h"
#include "xaie_mem.h"

#ifdef XAIE_FEATURE_UC_ENABLE
/*****************************************************************************/
/**
*
* This function is used to get the target tile location from Host's perspective
* based on the physical address of the data memory from the device's
* perspective.
*
* @param	DevInst: Device Instance.
* @param	Loc: Location specified by the user.
* @param	Addr: Physical Address from Device's perspective.
* @param	TgtLoc: Pointer to the target location based on physical
*		address.
*
* @return	XAIE_OK on success and error code for failure.
*
* @note		Internal API only.
*
*******************************************************************************/
static AieRC _XAie_GetTargetTileLoc(XAie_DevInst *DevInst, XAie_LocType Loc,
		u32 Addr, XAie_LocType *TgtLoc)
{
	u8 CardDir;
	u8 RowParity;
	u8 TileType;
	const XAie_UcMod *UcMod;

	UcMod = DevInst->DevProp.DevMod[XAIEGBL_TILE_TYPE_SHIMNOC].UcMod;

	/*
	 * Find the cardinal direction and get tile address.
	 * CardDir can have values of 4, 5, 6 or 7 for valid data memory
	 * addresses..
	 */
	CardDir = (u8)(Addr / UcMod->PrivDataMemSize);

	RowParity = Loc.Row % 2U;
	/*
	 * Checkerboard architecture is valid for AIE. Force RowParity to 1
	 * otherwise.
	 */
	if(UcMod->IsCheckerBoard == 0U) {
		RowParity = 1U;
	}

	switch(CardDir) {
	case 4U:
		/* South */
		Loc.Row -= 1U;
		break;
	case 5U:
		/*
		 * West - West I/F could be the same tile or adjacent
		 * tile based on the row number
		 */
		if(RowParity == 1U) {
			/* Adjacent tile */
			Loc.Col -= 1U;
		}
		break;
	case 6U:
		/* North */
		Loc.Row += 1U;
		break;
	case 7U:
		/*
		 * East - East I/F could be the same tile or adjacent
		 * tile based on the row number
		 */
		if(RowParity == 0U) {
			/* Adjacent tile */
			Loc.Col += 1U;
		}
		break;
	default:
		/* Invalid CardDir */
		XAIE_ERROR("Invalid address - 0x%x\n", Addr);
		return XAIE_ERR;
	}

	/* Return errors if modified rows and cols are invalid */
	if(Loc.Row >= DevInst->NumRows || Loc.Col >= DevInst->NumCols) {
		XAIE_ERROR("Target row/col out of range\n");
		return XAIE_ERR;
	}

	TileType = DevInst->DevOps->GetTTypefromLoc(DevInst, Loc);
	if(TileType != XAIEGBL_TILE_TYPE_SHIMNOC) {
		XAIE_ERROR("Invalid tile type for address\n");
		return XAIE_ERR;
	}

	TgtLoc->Row = Loc.Row;
	TgtLoc->Col = Loc.Col;

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This routine is used to write the loadable sections of the elf belonging to
* the program memory of ai engines.
*
* @param	DevInst: Device Instance.
* @param	Loc: Starting location of the section.
* @param	SectionPtr: Pointer to the program section entry in the ELF
*		buffer.
* @param	Phdr: Pointer to the program header.
*
* @return	XAIE_OK on success and error code for failure.
*
* @note		Internal API only.
*
*******************************************************************************/
static AieRC _XAie_LoadProgMemSection(XAie_DevInst *DevInst, XAie_LocType Loc,
		const unsigned char *SectionPtr, const Elf32_Phdr *Phdr)
{
	u64 Addr;
	const XAie_UcMod *UcMod;

	UcMod = DevInst->DevProp.DevMod[XAIEGBL_TILE_TYPE_SHIMNOC].UcMod;

	/* Write to Program Memory */
	if((Phdr->p_paddr + Phdr->p_memsz) > UcMod->ProgMemSize) {
		XAIE_ERROR("Overflow of program memory\n");
		return XAIE_INVALID_ELF;
	}

	Addr = (u64)(UcMod->ProgMemHostOffset + Phdr->p_paddr) +
		_XAie_GetTileAddr(DevInst, Loc.Row, Loc.Col);

	/*
	 * The program memory sections in the elf can end at 32bit
	 * unaligned addresses. To factor this, we round up the number
	 * of 32-bit words that has to be written to the program
	 * memory. Since the elf has footers at the end, accessing
	 * memory out of Progsec will not result in a segmentation
	 * fault.
	 */
	return XAie_BlockWrite32(DevInst, Addr, (u32 *)SectionPtr,
			(Phdr->p_memsz + 4U - 1U) / 4U);
}

/*****************************************************************************/
/**
*
* This routine is used to write the loadable sections of the elf belonging to
* the data memory of ai engines.
*
* @param	DevInst: Device Instance.
* @param	Loc: Starting location of the section.
* @param	SectionPtr: Pointer to the program section entry in the ELF
*		buffer.
* @param	Phdr: Pointer to the program header.
*
* @return	XAIE_OK on success and error code for failure.
*
* @note		Internal API only.
*
*******************************************************************************/
static AieRC _XAie_LoadDataMemSection(XAie_DevInst *DevInst, XAie_LocType Loc,
		const unsigned char *SectionPtr, const Elf32_Phdr *Phdr)
{
	AieRC RC;
	u32 OverFlowBytes;
	u32 BytesToWrite;
	u32 SectionAddr;
	u32 SectionSize;
	u32 AddrMask;
	u64 Addr;
	XAie_LocType TgtLoc;
	const unsigned char *Buffer = SectionPtr;
	unsigned char *Tmp = XAIE_NULL;
	const XAie_UcMod *UcMod;

	UcMod = DevInst->DevProp.DevMod[XAIEGBL_TILE_TYPE_SHIMNOC].UcMod;

	/* Check if section can access out of bound memory location on device */
	if(((Phdr->p_paddr > UcMod->ProgMemSize) &&
			(Phdr->p_paddr < UcMod->PrivDataMemAddr)) ||
			((Phdr->p_paddr + Phdr->p_memsz) >
			 (UcMod->PrivDataMemAddr + UcMod->PrivDataMemSize * 4U))) {
		XAIE_ERROR("Invalid section starting at 0x%x\n",
				Phdr->p_paddr);
		return XAIE_INVALID_ELF;
	}

	/* Write initialized section to data memory */
	SectionSize = Phdr->p_memsz;
	SectionAddr = Phdr->p_paddr;
	AddrMask = UcMod->PrivDataMemSize - 1U;
	/* Check if file size is 0. If yes, allocate memory and init to 0 */
	if(Phdr->p_filesz == 0U) {
		Buffer = (const unsigned char *)calloc(Phdr->p_memsz,
				sizeof(char));
		if(Buffer == XAIE_NULL) {
			XAIE_ERROR("Memory allocation failed for buffer\n");
			return XAIE_ERR;
		}
		/* Copy pointer to free allocated memory in case of error. */
		Tmp = (unsigned char *)Buffer;
	}

	while(SectionSize > 0U) {
		RC = _XAie_GetTargetTileLoc(DevInst, Loc, SectionAddr, &TgtLoc);
		if(RC != XAIE_OK) {
			XAIE_ERROR("Failed to get target "\
					"location for p_paddr 0x%x\n",
					SectionAddr);
			if(Phdr->p_filesz == 0U) {
				free(Tmp);
			}
			return RC;
		}

		/*Bytes to write in this section */
		OverFlowBytes = 0U;
		if((SectionAddr & AddrMask) + SectionSize >
				UcMod->PrivDataMemSize) {
			OverFlowBytes = (SectionAddr & AddrMask) + SectionSize -
				UcMod->PrivDataMemSize;
		}

		BytesToWrite = SectionSize - OverFlowBytes;
		Addr = (u64)(SectionAddr & AddrMask);

		/* Turn ECC On if EccStatus flag is set. */
		if(DevInst->EccStatus) {
			RC = _XAie_EccOnDM(DevInst, TgtLoc);
			if(RC != XAIE_OK) {
				XAIE_ERROR("Unable to turn ECC On for Data Memory\n");
				if(Phdr->p_filesz == 0U) {
					free(Tmp);
				}
				return RC;
			}
		}

		RC = XAie_DataMemBlockWrite(DevInst, TgtLoc, (u32)Addr,
				(const void*)Buffer, BytesToWrite);
		if(RC != XAIE_OK) {
			XAIE_ERROR("Write to data memory failed\n");
			if(Phdr->p_filesz == 0U) {
				free(Tmp);
			}
			return RC;
		}

		SectionSize -= BytesToWrite;
		SectionAddr += BytesToWrite;
		Buffer += BytesToWrite;
	}

	if(Phdr->p_filesz == 0U) {
		free(Tmp);
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This routine is used to write to the specified program section by reading the
* corresponding data from the ELF buffer.
*
* @param	DevInst: Device Instance.
* @param	Loc: Starting location of the section.
* @param	ProgSec: Pointer to the program section entry in the ELF buffer.
* @param	Phdr: Pointer to the program header.
*
* @return	XAIE_OK on success and error code for failure.
*
* @note		Internal API only.
*
*******************************************************************************/
static AieRC _XAie_WriteProgramSection(XAie_DevInst *DevInst, XAie_LocType Loc,
		const unsigned char *ProgSec, const Elf32_Phdr *Phdr)
{
	const XAie_UcMod *UcMod;

	UcMod = DevInst->DevProp.DevMod[XAIEGBL_TILE_TYPE_SHIMNOC].UcMod;

	/* Write to Program Memory */
	if(Phdr->p_paddr < UcMod->ProgMemSize) {
		return _XAie_LoadProgMemSection(DevInst, Loc, ProgSec, Phdr);
	}

	if((Phdr->p_filesz == 0U) || (Phdr->p_filesz != 0U)) {
		return _XAie_LoadDataMemSection(DevInst, Loc, ProgSec, Phdr);
	} else  {
		XAIE_WARN("Mismatch in program header to data memory loadable section. Skipping this program section.\n");
		return XAIE_OK;
	}
}

static AieRC _XAie_LoadElfFromMem(XAie_DevInst *DevInst, XAie_LocType Loc, const unsigned char* ElfMem)
{
	AieRC RC;
	const Elf32_Ehdr *Ehdr;
	const Elf32_Phdr *Phdr;
	const unsigned char *SectionPtr;

	Ehdr = (const Elf32_Ehdr *) ElfMem;
	_XAie_PrintElfHdr(Ehdr);

	for(u32 phnum = 0U; phnum < Ehdr->e_phnum; phnum++) {
		Phdr = (Elf32_Phdr*) (ElfMem + sizeof(*Ehdr) +
				phnum * sizeof(*Phdr));
		_XAie_PrintProgSectHdr(Phdr);
		if(Phdr->p_type == (u32)PT_LOAD) {
			SectionPtr = ElfMem + Phdr->p_offset;
			RC = _XAie_WriteProgramSection(DevInst, Loc, SectionPtr, Phdr);
			if(RC != XAIE_OK) {
				return RC;
			}
		}
	}

	/* Turn ECC On after program memory load */
	if(DevInst->EccStatus) {
		RC = _XAie_EccOnPM(DevInst, Loc);
		if(RC != XAIE_OK) {
			XAIE_ERROR("Unable to turn ECC On for Program Memory\n");
			return RC;
		}
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This function loads the elf from memory to the uC. The function writes
* 0 for the uninitialized data section.
*
* @param	DevInst: Device Instance.
* @param	Loc: Location of AIE Tile.
* @param	ElfMem: Pointer to the Elf contents in memory.
*
* @return	XAIE_OK on success and error code for failure.
*
* @note		None.
*
*******************************************************************************/
AieRC XAie_LoadUcMem(XAie_DevInst *DevInst, XAie_LocType Loc,
		const unsigned char* ElfMem)
{
	u8 TileType;

	if((DevInst == XAIE_NULL) || (ElfMem == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid arguments\n");
		return XAIE_INVALID_ARGS;
	}

	TileType = DevInst->DevOps->GetTTypefromLoc(DevInst, Loc);
	if(TileType != XAIEGBL_TILE_TYPE_SHIMNOC) {
		XAIE_ERROR("Invalid tile type\n");
		return XAIE_INVALID_TILE;
	}

	if (ElfMem == XAIE_NULL) {
		XAIE_ERROR("Invalid ElfMem\n");
		return XAIE_INVALID_ARGS;
	}

	return _XAie_LoadElfFromMem(DevInst, Loc, ElfMem);
}

/*****************************************************************************/
/**
*
* This function loads the elf from file to the uC. The function writes
* 0 for the unitialized data section.
*
* @param	DevInst: Device Instance.
* @param	Loc: Location of AIE Tile.
* @param	ElfPtr: Path to the elf file.
*
* @return	XAIE_OK on success and error code for failure.
*
* @note		None.
*
*******************************************************************************/
AieRC XAie_LoadUc(XAie_DevInst *DevInst, XAie_LocType Loc, const char *ElfPtr)
{
	u8 TileType;
	FILE *Fd;
	int Ret;
	unsigned char *ElfMem;
	u64 ElfSz;
	AieRC RC;


	if((DevInst == XAIE_NULL) ||
		(DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
		XAIE_ERROR("Invalid device instance\n");
		return XAIE_INVALID_ARGS;
	}

	TileType = DevInst->DevOps->GetTTypefromLoc(DevInst, Loc);
	if(TileType != XAIEGBL_TILE_TYPE_SHIMNOC) {
		XAIE_ERROR("Invalid tile type\n");
		return XAIE_INVALID_TILE;
	}

	if (ElfPtr == XAIE_NULL) {
		XAIE_ERROR("Invalid ElfPtr\n");
		return XAIE_INVALID_ARGS;
	}

	Fd = fopen(ElfPtr, "r");
	if(Fd == XAIE_NULL) {
		XAIE_ERROR("Unable to open elf file, %d: %s\n",
			errno, strerror(errno));
		return XAIE_INVALID_ELF;
	}

	/* Get the file size of the elf */
	Ret = fseek(Fd, 0L, SEEK_END);
	if(Ret != 0) {
		XAIE_ERROR("Failed to get end of file, %d: %s\n",
			errno, strerror(errno));
		fclose(Fd);
		return XAIE_INVALID_ELF;
	}

	ElfSz = (u64)ftell(Fd);
	rewind(Fd);
	XAIE_DBG("Elf size is %ld bytes\n", ElfSz);

	/* Read entire elf file into memory */
	ElfMem = (unsigned char*) malloc(ElfSz);
	if(ElfMem == NULL) {
		fclose(Fd);
		XAIE_ERROR("Memory allocation failed\n");
		return XAIE_ERR;
	}

	Ret = (int)fread((void*)ElfMem, ElfSz, 1U, Fd);
	if(Ret == 0) {
		fclose(Fd);
		free(ElfMem);
		XAIE_ERROR("Failed to read Elf into memory\n");
		return XAIE_ERR;
	}

	fclose(Fd);

	RC = _XAie_LoadElfFromMem(DevInst, Loc, ElfMem);
	free(ElfMem);

	return RC;
}

/*****************************************************************************/
/*
*
* This API writes to the Core control register of a uC to wakeup the core.
*
* @param	DevInst: Device Instance
* @param	Loc: Location of the Shim tile.
*
* @return	XAIE_OK on success, Error code on failure.
*
* @note		Internal only.
*
******************************************************************************/
AieRC _XAie_UcCoreWakeup(XAie_DevInst *DevInst, XAie_LocType Loc,
		const struct XAie_UcMod *UcMod)
{
	u32 Mask, Value;
	u64 RegAddr;

	Mask = UcMod->CoreCtrl->CtrlWakeup.Mask;
	Value = (u32)(1U << UcMod->CoreCtrl->CtrlWakeup.Lsb);
	RegAddr = UcMod->CoreCtrl->RegOff +
		_XAie_GetTileAddr(DevInst, Loc.Row, Loc.Col);

	return XAie_MaskWrite32(DevInst, RegAddr, Mask, Value);
}

/*****************************************************************************/
/*
*
* This API writes to the Core control register of a uC to sleep the core.
*
* @param	DevInst: Device Instance
* @param	Loc: Location of the Shim tile.
*
* @return	XAIE_OK on success, Error code on failure.
*
* @note		Internal only.
*
******************************************************************************/
AieRC _XAie_UcCoreSleep(XAie_DevInst *DevInst, XAie_LocType Loc,
		const struct XAie_UcMod *UcMod)
{
	u32 Mask, Value;
	u64 RegAddr;

	Mask = UcMod->CoreCtrl->CtrlSleep.Mask;
	Value = (u32)(1U << UcMod->CoreCtrl->CtrlSleep.Lsb);
	RegAddr = UcMod->CoreCtrl->RegOff +
		_XAie_GetTileAddr(DevInst, Loc.Row, Loc.Col);

	return XAie_MaskWrite32(DevInst, RegAddr, Mask, Value);
}

/*****************************************************************************/
/*
*
* This API reads the uC core status register value.
*
* @param	DevInst: Device Instance
* @param	Loc: Location of the AIE tile.
* @param	CoreStatus: Pointer to store the core status register value.
* @param	UcMod: Pointer to the uC module data structure.
*
* @return	XAIE_OK on success, Error code on failure.
*
* @note		Internal only.
*
******************************************************************************/
AieRC _XAie_UcCoreGetStatus(XAie_DevInst *DevInst, XAie_LocType Loc,
		u32 *CoreStatus, const struct XAie_UcMod *UcMod)
{
	AieRC RC;
	u64 RegAddr;
	u32 RegVal;

	/* Read core status register */
	RegAddr = UcMod->CoreSts->RegOff +
		_XAie_GetTileAddr(DevInst, Loc.Row, Loc.Col);
	RC = XAie_Read32(DevInst, RegAddr, &RegVal);
	if(RC != XAIE_OK) {
		return RC;
	}

	*CoreStatus = XAie_GetField(RegVal, 0U, UcMod->CoreSts->Mask);

	return XAIE_OK;
}

#endif /* XAIE_FEATURE_UC_ENABLE */
/** @} */
