/*
 *  SPDX-License-Identifier: Apache-2.0
 *  Copyright (C) 2015-2018, Xilinx Inc
 *  Copyright (C) 2022, Advanced Micro Devices, Inc.  All rights reserved.
 */
#ifndef xclfeatures_h_
#define xclfeatures_h_

#define FEATURE_ROM_MAJOR_VERSION 10
#define FEATURE_ROM_MINOR_VERSION 1

//Layout: At address 0xB0000, we will have the FeatureRomHeader that comprises:
//
//1. First have FeatureRomHeader: 152 bytes of information followed by
//2. Then, as a part of FeatureRomHeader we have the PRRegion struct(s).
//	The number of such structs will be same as OCLRegionCount.
//3. After this the freq scaling table is laid out.
//

//#include <stdint.h>

struct PartialRegion {
	uint16_t clk[4];
	int8_t XPR; //0 : non-xpt, 1: xpr
};

// Each entry represents one row in freq scaling table.
struct FreqScalingTableRow {
	short config0;
	short freq;
	short config2;
};

enum PROMType  {
	BPI	= 0
	, SPI	= 1
   //room for 6 more types of flash devices.
};

enum DebugType	{
	DT_NIFD	 = 0x01,
	DT_FIREWALL  = 0x02
  //There is room for future expansion upto 8 IPs
};

// This bit mask is used with the FeatureBitMap to calculate 64 bool features
//
// To test if a feature is provided:
//   FeatureRomHeader header;
//   if (FeatureBitMask::FBM_IS_UNIFIED & header.FeatureBitMap)
//     // it is supported
//   else
//     // it is not supported
//
// To set if a feature is provided:
//   header.FeatureBitMap = 0;
//   header.FeatureBitMap |= FeatureBitMask::FBM_IS_UNIFIED;
//
enum FeatureBitMask {
	UNIFIED_PLATFORM       =   0x0000000000000001	     /* bit 1 : Unified platform */
	, XARE_ENBLD	       =   0x0000000000000002	    /* bit 2 : Aurora link enabled DSA */
	, BOARD_MGMT_ENBLD     =   0x0000000000000004	    /* bit 3 : Has MB based power monitoring */
	, MB_SCHEDULER	       =   0x0000000000000008	    /* bit 4:  Has MB based scheduler */
	, PROM_MASK	       =   0x0000000000000070	    /* bits 5,6 &7  : 3 bits for PROMType */
	/**  ------ Bit 8 unused **/
	, DEBUG_MASK	       =   0x000000000000FF00	    /* bits 9 through 16  : 8 bits for DebugType */
	, PEER_TO_PEER	       =   0x0000000000010000	    /* bits 17	: Bar 2 is a peer to peer bar*/
	, FBM_UUID		       =   0x0000000000020000	    /* bits 18	: UUID enabled. uuid[16] field is valid*/
	, HBM		       =   0x0000000000040000	    /* bits 19	: Device has HBM's. HBMCount/Size are valid*/
	, CDMA		       =   0x0000000000080000	    /* bits 21	: Device has CDMA*/
	, QDMA		       =   0x0000000000100000	    /* bits 20	: Device has QDMA*/
	, RUNTIME_CLK_SCALE    =   0x0000000000200000	    /* bit 22 : Device has RUNTIME CLOCK SCALING feature*/
	, PASSTHROUGH_VIRTUALIZATION    =   0x0000000000400000	/* bit 23 : Device has Passthrough Virtualization feature*/

   //....more
};



// In the following data structures, the EntryPointString, MajorVersion, and MinorVersion
// values are all used in the Runtime to identify if the ROM is producing valid data, and
// to pick the schema to read the rest of the data; Ergo, these values shall not change.

/*
 * Struct used for >  2017.2_sdx
 * This struct should be used for version (==) 10.0 (Major: 10, Minor: 0)
 */
struct FeatureRomHeader {
	unsigned char EntryPointString[4];  // This is "xlnx"
	uint8_t MajorVersion;		    // Feature ROM's major version eg 1
	uint8_t MinorVersion;		    // minor version eg 2.
	// -- DO NOT CHANGE THE TYPES ABOVE THIS LINE --
	uint32_t VivadoBuildID;		    // Vivado Software Build (e.g., 1761098 ). From ./vivado --version
	uint32_t IPBuildID;		    // IP Build (e.g., 1759159 from abve)
	uint64_t TimeSinceEpoch;	    // linux time(NULL) call, at write_dsa_rom invocation
	unsigned char FPGAPartName[64];	    // The hardware FPGA part. Null termninated
	unsigned char VBNVName[64];	    // eg : xilinx:xil-accel-rd-ku115:4ddr-xpr:3.4: null terminated
	uint8_t DDRChannelCount;	    // 4 for TUL
	uint8_t DDRChannelSize;		    // 4 (in GB)
	uint64_t DRBaseAddress;		    // The Dynamic Range's (AppPF/CL/Userspace) Base Address
	uint64_t FeatureBitMap;		    // Feature Bit Map, 64 different bool features, maps to enum FeatureBitMask
	unsigned char uuid[16];		    // UUID of the DSA.
	uint8_t HBMCount;		    // Number of HBMs
	uint8_t HBMSize;		    // Size of (each) HBM in GB
	uint32_t CDMABaseAddress[4];	    // CDMA base addresses
};

#endif // xclfeatures_h_
