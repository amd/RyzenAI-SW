/******************************************************************************
* Copyright (C) 2019 - 2020 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_elfloader.h
* @{
*
* Header file for core elf loader functions
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Tejus   09/24/2019  Initial creation
* 1.1   Tejus   03/20/2020  Remove range apis
* 1.2   Tejus   05/26/2020  Add API to load elf from memory.
* </pre>
*
******************************************************************************/
#ifndef XAIELOADER_H
#define XAIELOADER_H

#include "xaie_feature_config.h"
#ifdef XAIE_FEATURE_ELF_ENABLE

/***************************** Include Files *********************************/
#ifndef _WIN32
#include <elf.h>
#endif

#include <stdlib.h>
#include <string.h>
#include "xaie_helper.h"
#include "xaiegbl.h"
#include "xaiegbl_defs.h"

/************************** Constant Definitions *****************************/
#define XAIE_LOAD_ELF_TXT	(1U << 0U)
#define XAIE_LOAD_ELF_BSS	(1U << 1U)
#define XAIE_LOAD_ELF_DATA	(1U << 2U)
#define XAIE_LOAD_ELF_ALL	(XAIE_LOAD_ELF_TXT | XAIE_LOAD_ELF_BSS | \
					XAIE_LOAD_ELF_DATA)

/************************** Variable Definitions *****************************/
typedef struct {
	u32 start;	/**< Stack start address */
	u32 end;	/**< Stack end address */
} XAieSim_StackSz;

#ifdef _WIN32
/*
 *  * Typedef for ELF related struct, Taken from GNU version of elf.h
	* https://github.com/lattera/glibc/blob/master/elf/elf.h
 *  */
/* These constants are for the segment types stored in the image headers */
#define PT_NULL    0
#define PT_LOAD    1
#define PT_DYNAMIC 2
#define PT_INTERP  3
#define PT_NOTE    4
#define PT_SHLIB   5
#define PT_PHDR    6
#define EI_NIDENT       16
typedef u16     Elf32_Half;
typedef u32     Elf32_Word;
typedef u32     Elf32_Addr;
typedef u32     Elf32_Off;
typedef struct elf32_hdr {
	unsigned char	e_ident[EI_NIDENT];
	Elf32_Half	e_type;
	Elf32_Half	e_machine;
	Elf32_Word	e_version;
	Elf32_Addr	e_entry;  /* Entry point */
	Elf32_Off	e_phoff;
	Elf32_Off	e_shoff;
	Elf32_Word	e_flags;
	Elf32_Half	e_ehsize;
	Elf32_Half	e_phentsize;
	Elf32_Half	e_phnum;
	Elf32_Half	e_shentsize;
	Elf32_Half	e_shnum;
	Elf32_Half	e_shstrndx;
} Elf32_Ehdr;

typedef struct elf32_phdr {
	Elf32_Word	p_type;
	Elf32_Off	p_offset;
	Elf32_Addr	p_vaddr;
	Elf32_Addr	p_paddr;
	Elf32_Word	p_filesz;
	Elf32_Word	p_memsz;
	Elf32_Word	p_flags;
	Elf32_Word	p_align;
} Elf32_Phdr;
#endif /* WIN32 */

/************************** Function Prototypes  *****************************/

XAIE_AIG_EXPORT AieRC XAie_LoadElf(XAie_DevInst *DevInst, XAie_LocType Loc, const char *ElfPtr,
		u8 LoadSym);
XAIE_AIG_EXPORT AieRC XAie_LoadElfMem(XAie_DevInst *DevInst, XAie_LocType Loc,
		const unsigned char* ElfMem);
XAIE_AIG_EXPORT AieRC XAie_LoadElfSection(XAie_DevInst *DevInst, XAie_LocType Loc,
		const unsigned char *SectionPtr, const Elf32_Phdr *Phdr);
XAIE_AIG_EXPORT AieRC XAie_LoadElfSectionBlock(XAie_DevInst *DevInst, XAie_LocType Loc,
		const unsigned char* SectionPtr, u64 TgtAddr, u32 Size);
XAIE_AIG_EXPORT AieRC XAie_LoadElfPartial(XAie_DevInst *DevInst, XAie_LocType Loc,
		const char* ElfPtr, u8 Sections);
void _XAie_PrintElfHdr(const Elf32_Ehdr *Ehdr);
void _XAie_PrintProgSectHdr(const Elf32_Phdr *Phdr);

#endif /* XAIE_FEATURE_ELF_ENABLE */
#endif		/* end of protection macro */
/** @} */
