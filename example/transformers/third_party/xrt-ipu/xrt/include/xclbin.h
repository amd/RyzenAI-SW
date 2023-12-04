/*
 *  SPDX-License-Identifier: Apache-2.0
 *  Copyright (C) 2015-2022, Xilinx Inc
 *  Copyright (C) 2022, Advanced Micro Devices, Inc.  All rights reserved.
 */

#ifndef _XCLBIN_H_
#define _XCLBIN_H_

#ifdef _WIN32
  #include <cstdint>
  #include <algorithm>
  #include "windows/uuid.h"
#else
  #if defined(__KERNEL__)
    #include <linux/types.h>
    #include <linux/uuid.h>
    #include <linux/version.h>
  #elif defined(__cplusplus)
    #include <cstdlib>
    #include <cstdint>
    #include <algorithm>
    #include <uuid/uuid.h>
  #else
    #include <stdlib.h>
    #include <stdint.h>
    #include <uuid/uuid.h>
  #endif

  #if !defined(__KERNEL__)
    typedef uuid_t xuid_t;
  #else //(__KERNEL__)
    #if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 12, 0)
      typedef uuid_t xuid_t;
    #elif defined(RHEL_RELEASE_CODE)
      #if RHEL_RELEASE_CODE > RHEL_RELEASE_VERSION(7,4)
        typedef uuid_t xuid_t;
      #else
        typedef uuid_le xuid_t;
      #endif
    #else
      typedef uuid_le xuid_t;
    #endif
  #endif
#endif

// ----------------- Custom Assert Macro -------------------------
// The xclbin.h header file is used both by the driver and host
// applications/libraries.  For C++ static_assert() would be used, but
// for the the drivers (e.g., linux) a more custom solution is needed
// e.g., BUILD_BUG_ON_ZERO()).  Instead of creating a wrapper around the
// various OS builds, a custom assert (based on the linux assert) is used.

#ifdef __cplusplus
  // Used C++ static assert
  #define XCLBIN_STATIC_ASSERT(e,m) \
    static_assert (e, m)
#else
  // Use our "custom" kernel compilation assert
  #define XLCBIN_ASSERT_CONCAT_(a, b) a##b
  #define XLCBIN_ASSERT_CONCAT(a, b) XLCBIN_ASSERT_CONCAT_(a, b)

  // Create an artifitial assertion via a bad divide by zero assertion.
  #define XCLBIN_STATIC_ASSERT(e,m) \
    enum { XLCBIN_ASSERT_CONCAT(assert_line_, __LINE__) = 1/(int)(!!(e)) }
#endif

// *** Helper macro ***
// Reports a size of structure via an error
#define SIZE_OF_STRUCT(s) \
   char (*__fail)[sizeof(struct s)] = 1

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Container format for Xilinx bitstreams, metadata and other
     * binary blobs.
     * Every segment must be aligned at 8 byte boundary with null byte padding
     * between adjacent segments if required.
     * For segements which are not present both offset and length must be 0 in
     * the header.
     * Currently only xclbin0\0 is recognized as file magic. In future if/when file
     * format is updated the magic string will be changed to xclbin1\0 and so on.
     */
    enum XCLBIN_MODE {
        XCLBIN_FLAT,
        XCLBIN_PR,
        XCLBIN_TANDEM_STAGE2,
        XCLBIN_TANDEM_STAGE2_WITH_PR,
        XCLBIN_HW_EMU,
        XCLBIN_SW_EMU,
        XCLBIN_HW_EMU_PR,
        XCLBIN_MODE_MAX
    };

    /*
     *  AXLF LAYOUT
     *  -----------
     *
     *  -----------------------------------------
     *  | Magic                                 |
     *  -----------------------------------------
     *  | Header                                |
     *  -----------------------------------------
     *  | One or more section headers           |
     *  -----------------------------------------
     *  | Matching number of sections with data |
     *  -----------------------------------------
     *
     */

    enum axlf_section_kind {
        BITSTREAM              =  0,
        CLEARING_BITSTREAM     =  1,
        EMBEDDED_METADATA      =  2,
        FIRMWARE               =  3,
        DEBUG_DATA             =  4,
        SCHED_FIRMWARE         =  5,
        MEM_TOPOLOGY           =  6,
        CONNECTIVITY           =  7,
        IP_LAYOUT              =  8,
        DEBUG_IP_LAYOUT        =  9,
        DESIGN_CHECK_POINT     = 10,
        CLOCK_FREQ_TOPOLOGY    = 11,
        MCS                    = 12,
        BMC                    = 13,
        BUILD_METADATA         = 14,
        KEYVALUE_METADATA      = 15,
        USER_METADATA          = 16,
        DNA_CERTIFICATE        = 17,
        PDI                    = 18,
        BITSTREAM_PARTIAL_PDI  = 19,
        PARTITION_METADATA     = 20,
        EMULATION_DATA         = 21,
        SYSTEM_METADATA        = 22,
        SOFT_KERNEL            = 23,
        ASK_FLASH              = 24,
        AIE_METADATA           = 25,
        ASK_GROUP_TOPOLOGY     = 26,
        ASK_GROUP_CONNECTIVITY = 27,
        SMARTNIC               = 28,
        AIE_RESOURCES          = 29,
        OVERLAY                = 30,
        VENDER_METADATA        = 31,
        AIE_PARTITION          = 32,
    };

    enum MEM_TYPE {
        MEM_DDR3,
        MEM_DDR4,
        MEM_DRAM,
        MEM_STREAMING,
        MEM_PREALLOCATED_GLOB,
        MEM_ARE, //Aurora
        MEM_HBM,
        MEM_BRAM,
        MEM_URAM,
        MEM_STREAMING_CONNECTION,
        MEM_HOST,
        MEM_PS_KERNEL,
    };

    enum IP_TYPE {
        IP_MB = 0,
        IP_KERNEL, //kernel instance
        IP_DNASC,
        IP_DDR4_CONTROLLER,
        IP_MEM_DDR4,
        IP_MEM_HBM,
        IP_MEM_HBM_ECC,
        IP_PS_KERNEL,
    };

    enum ACTION_MASK {
      AM_LOAD_AIE = 0x1                     /* Indicates to the driver to load the AIE PID section */
    };

    struct axlf_section_header {
        uint32_t m_sectionKind;             /* Section type */
        char m_sectionName[16];             /* Examples: "stage2", "clear1", "clear2", "ocl1", "ocl2, "ublaze", "sched" */
        uint64_t m_sectionOffset;           /* File offset of section data */
        uint64_t m_sectionSize;             /* Size of section data */
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct axlf_section_header) == 40, "axlf_section_header structure no longer is 40 bytes in size");

    struct axlf_header {
        uint64_t m_length;                  /* Total size of the xclbin file */
        uint64_t m_timeStamp;               /* Number of seconds since epoch when xclbin was created */
        uint64_t m_featureRomTimeStamp;     /* TimeSinceEpoch of the featureRom */
        uint16_t m_versionPatch;            /* Patch Version */
        uint8_t m_versionMajor;             /* Major Version - Version: 2.1.0*/
        uint8_t m_versionMinor;             /* Minor Version */
        uint16_t m_mode;                    /* XCLBIN_MODE */
        uint16_t m_actionMask;              /* Bit Mask */
	union {
	    struct {
		uint64_t m_platformId;      /* 64 bit platform ID: vendor-device-subvendor-subdev */
		uint64_t m_featureId;       /* 64 bit feature id */
	    } rom;
	    unsigned char rom_uuid[16];     /* feature ROM UUID for which this xclbin was generated */
	};
        unsigned char m_platformVBNV[64];   /* e.g. xilinx:xil-accel-rd-ku115:4ddr-xpr:3.4: null terminated */
	union {
	    char m_next_axlf[16];           /* Name of next xclbin file in the daisy chain */
	    xuid_t uuid;                    /* uuid of this xclbin*/
	};
        char m_debug_bin[16];               /* Name of binary with debug information */
        uint32_t m_numSections;             /* Number of section headers */
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct axlf_header) == 152, "axlf_header structure no longer is 152 bytes in size");
    #define XCLBIN_MAX_NUM_SECTION    0x10000

    struct axlf {
        char m_magic[8];                            /* Should be "xclbin2\0"  */
        int32_t m_signature_length;                 /* Length of the signature. -1 indicates no signature */
        unsigned char reserved[28];                 /* Note: Initialized to 0xFFs */

        unsigned char m_keyBlock[256];              /* Signature for validation of binary */
        uint64_t m_uniqueId;                        /* axlf's uniqueId, use it to skip redownload etc */
        struct axlf_header m_header;                /* Inline header */
        struct axlf_section_header m_sections[1];   /* One or more section headers follow */
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct axlf) == 496, "axlf structure no longer is 496 bytes in size");

    typedef struct axlf xclBin;

    /**** BEGIN : Xilinx internal section *****/

    /* bitstream information */
    struct xlnx_bitstream {
        uint8_t m_freq[8];
        char bits[1];
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct xlnx_bitstream) == 9, "xlnx_bitstream structure no longer is 9 bytes in size");


    /****   MEMORY TOPOLOGY SECTION ****/
    struct mem_data {
        uint8_t m_type;          // enum corresponding to mem_type.
        uint8_t m_used;          // if 0 this bank is not present
        uint8_t padding[6];      // 8 Byte alignment padding (initialized to zeros)
        union {
            uint64_t m_size;     // if mem_type DDR, then size in KB;
            uint64_t route_id;   // if streaming then "route_id"
        };
        union {
            uint64_t m_base_address; // if DDR then the base address;
            uint64_t flow_id;        // if streaming then "flow id"
        };
        unsigned char m_tag[16]; // DDR: BANK0,1,2,3, has to be null terminated; if streaming then stream0, 1 etc
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct mem_data) == 40, "mem_data structure no longer is 40 bytes in size");

    struct mem_topology {
        int32_t m_count; //Number of mem_data
        struct mem_data m_mem_data[1]; //Should be sorted on mem_type
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct mem_topology) == 48, "mem_topology structure no longer is 48 bytes in size");


    /****   CONNECTIVITY SECTION ****/
    /* Connectivity of each argument of Kernel. It will be in terms of argument
     * index associated. For associating kernel instances with arguments and
     * banks, start at the connectivity section. Using the m_ip_layout_index
     * access the ip_data.m_name. Now we can associate this kernel instance
     * with its original kernel name and get the connectivity as well. This
     * enables us to form related groups of kernel instances.
     */

    struct connection {
        int32_t arg_index; //From 0 to n, may not be contiguous as scalars skipped
        int32_t m_ip_layout_index; //index into the ip_layout section. ip_layout.m_ip_data[index].m_type == IP_KERNEL
        int32_t mem_data_index; //index of the m_mem_data . Flag error is m_used false.
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct connection) == 12, "connection structure no longer is 12 bytes in size");

    struct connectivity {
        int32_t m_count;
        struct connection m_connection[1];
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct connectivity) == 16, "connectivity structure no longer is 16 bytes in size");

    /****   IP_LAYOUT SECTION ****/

    // IP Kernel
    #define IP_INT_ENABLE_MASK    0x0001
    #define IP_INTERRUPT_ID_MASK  0x00FE
    #define IP_INTERRUPT_ID_SHIFT 0x1

    enum IP_CONTROL {
        AP_CTRL_HS = 0,
        AP_CTRL_CHAIN = 1,
        AP_CTRL_NONE = 2,
        AP_CTRL_ME = 3,
        ACCEL_ADAPTER = 4,
        FAST_ADAPTER = 5
    };

    #define IP_CONTROL_MASK  0xFF00
    #define IP_CONTROL_SHIFT 0x8

    /* IPs on AXI lite - their types, names, and base addresses.*/
    struct ip_data {
        uint32_t m_type; //map to IP_TYPE enum
        union {
            uint32_t properties; // Default: 32-bits to indicate ip specific property.
                                 // m_type: IP_KERNEL
                                 //         m_int_enable   : Bit  - 0x0000_0001;
                                 //         m_interrupt_id : Bits - 0x0000_00FE;
                                 //         m_ip_control   : Bits = 0x0000_FF00;
            struct {             // m_type: IP_MEM_*
               uint16_t m_index;
               uint8_t m_pc_index;
               uint8_t unused;
            } indices;
        };
        uint64_t m_base_address;
        uint8_t m_name[64]; //eg Kernel name corresponding to KERNEL instance, can embed CU name in future.
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct ip_data) == 80, "ip_data structure no longer is 80 bytes in size");

    struct ip_layout {
        int32_t m_count;
        struct ip_data m_ip_data[1]; //All the ip_data needs to be sorted by m_base_address.
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct ip_layout) == 88, "ip_layout structure no longer is 88 bytes in size");

    /*** Debug IP section layout ****/
    enum DEBUG_IP_TYPE {
        UNDEFINED = 0,
        LAPC,
        ILA,
        AXI_MM_MONITOR,
        AXI_TRACE_FUNNEL,
        AXI_MONITOR_FIFO_LITE,
        AXI_MONITOR_FIFO_FULL,
        ACCEL_MONITOR,
        AXI_STREAM_MONITOR,
        AXI_STREAM_PROTOCOL_CHECKER,
        TRACE_S2MM,
        AXI_DMA,
        TRACE_S2MM_FULL,
        AXI_NOC,
        ACCEL_DEADLOCK_DETECTOR,
        HSDP_TRACE,
        DEBUG_IP_TYPE_MAX
    };

    struct debug_ip_data {
        uint8_t m_type; // type of enum DEBUG_IP_TYPE
        uint8_t m_index_lowbyte;
        uint8_t m_properties;
        uint8_t m_major;
        uint8_t m_minor;
        uint8_t m_index_highbyte;
        uint8_t m_reserved[2];
        uint64_t m_base_address;
        char    m_name[128];
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct debug_ip_data) == 144, "debug_ip_data structure no longer is 144 bytes in size");

    struct debug_ip_layout {
        uint16_t m_count;
        struct debug_ip_data m_debug_ip_data[1];
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct debug_ip_layout) == 152, "debug_ip_layout structure no longer is 152 bytes in size");

    enum CLOCK_TYPE {                      /* Supported clock frequency types */
        CT_UNUSED = 0,                     /* Initialized value */
        CT_DATA   = 1,                     /* Data clock */
        CT_KERNEL = 2,                     /* Kernel clock */
        CT_SYSTEM = 3                      /* System Clock */
    };

    struct clock_freq {                    /* Clock Frequency Entry */
        uint16_t m_freq_Mhz;               /* Frequency in MHz */
        uint8_t m_type;                    /* Clock type (enum CLOCK_TYPE) */
        uint8_t m_unused[5];               /* Not used - padding */
        char m_name[128];                  /* Clock Name */
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct clock_freq) == 136, "clock_freq structure no longer is 136 bytes in size");

    struct clock_freq_topology {           /* Clock frequency section */
        int16_t m_count;                   /* Number of entries */
        struct clock_freq m_clock_freq[1]; /* Clock array */
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct clock_freq_topology) == 138, "clock_freq_topology structure no longer is 138 bytes in size");

    enum MCS_TYPE {                        /* Supported MCS file types */
        MCS_UNKNOWN = 0,                   /* Initialized value */
        MCS_PRIMARY = 1,                   /* The primary mcs file data */
        MCS_SECONDARY = 2,                 /* The secondary mcs file data */
    };

    struct mcs_chunk {                     /* One chunk of MCS data */
        uint8_t m_type;                    /* MCS data type */
        uint8_t m_unused[7];               /* padding */
        uint64_t m_offset;                 /* data offset from the start of the section */
        uint64_t m_size;                   /* data size */
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct mcs_chunk) == 24, "mcs_chunk structure no longer is 24 bytes in size");

    struct mcs {                           /* MCS data section */
        int8_t m_count;                    /* Number of chunks */
        int8_t m_unused[7];                /* padding */
        struct mcs_chunk m_chunk[1];       /* MCS chunks followed by data */
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct mcs) == 32, "mcs structure no longer is 32 bytes in size");

    struct bmc {                           /* bmc data section  */
        uint64_t m_offset;                 /* data offset from the start of the section */
        uint64_t m_size;                   /* data size (bytes)*/
        char m_image_name[64];             /* Name of the image (e.g., MSP432P401R) */
        char m_device_name[64];            /* Device ID         (e.g., VCU1525)  */
        char m_version[64];
        char m_md5value[33];               /* MD5 Expected Value(e.g., 56027182079c0bd621761b7dab5a27ca)*/
        char m_padding[7];                 /* Padding */
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct bmc) == 248, "bmc structure no longer is 248 bytes in size");

    struct soft_kernel {                   /* soft kernel data section  */
        // Prefix Syntax:
        //   mpo - member, pointer, offset
        //     This variable represents a zero terminated string
        //     that is offseted from the beginning of the section.
        //
        //     The pointer to access the string is initialized as follows:
        //     char * pCharString = (address_of_section) + (mpo value)
        uint32_t mpo_name;         // Name of the soft kernel
        uint32_t m_image_offset;   // Image offset
        uint32_t m_image_size;     // Image size
        uint32_t mpo_version;      // Version
        uint32_t mpo_md5_value;    // MD5 checksum
        uint32_t mpo_symbol_name;  // Symbol name
        uint32_t m_num_instances;  // Number of instances
        uint8_t padding[36];       // Reserved for future use
        uint8_t reservedExt[16];   // Reserved for future extended data
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct soft_kernel) == 80, "soft_kernel structure no longer is 80 bytes in size");

    enum FLASH_TYPE
    {
        FLT_UNKNOWN = 0,
        FLT_BIN_PRIMARY
    };

    struct flash {                 /* flash data section  */
        // Prefix Syntax:
        //   mpo - member, pointer, offset
        //     This variable represents a zero terminated string
        //     that is offseted from the beginning of the section.
        //
        //     The pointer to access the string is initialized as follows:
        //     char * pCharString = (address_of_section) + (mpo value)
        uint16_t m_flash_type;     // ENUM FLASH_TYPE
        uint8_t padding[2];        // Alignment buffer
        uint32_t m_image_offset;   // Image offset
        uint32_t m_image_size;     // Image size
        uint32_t mpo_name;         // Name of the flash image
        uint32_t mpo_version;      // Version
        uint32_t mpo_md5_value;    // MD5 checksum
        uint8_t reserved[32];      // Reserved for future use
        uint8_t reservedExt[16];   // Reserved for future extended data
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct flash) == 72, "flash structure no longer is 72 bytes in size");

    enum CHECKSUM_TYPE
    {
        CST_UNKNOWN = 0,
        CST_SDBM = 1,
        CST_LAST
    };

    struct vender_metadata {                   /* vender metadata section  */
        // Prefix Syntax:
        //   mpo - member, pointer, offset
        //     This variable represents a zero terminated string
        //     that is offseted from the beginning of the section.
        //
        //     The pointer to access the string is initialized as follows:
        //     char * pCharString = (address_of_section) + (mpo value)
        uint32_t mpo_name;         // Name of the the vender metadata section
        uint32_t m_image_offset;   // Image offset
        uint32_t m_image_size;     // Image size
        uint8_t padding[36];       // Reserved for future use
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct vender_metadata) == 48, "vender metadata kernel structure no longer is 48 bytes in size");


    struct array_offset {
        uint32_t size;             // Number of elements in the array
        uint32_t offset;           // Array offset from the start of the section
    };

    XCLBIN_STATIC_ASSERT(sizeof(struct array_offset) == 8, "array_offset structure no longer is 8 bytes in size");

    enum CDO_Type {
        CT_UNKNOWN = 0,
        CT_PRIMARY = 1,
        CT_LITE    = 2,
        CT_PREPOST = 3
    };

    struct cdo_group {
        uint32_t mpo_name;                  // Name of the CDO group (Null terminated string)
        uint8_t cdo_type;                   // CDO group type (CDO_Type)
        uint8_t padding[3];
        uint64_t pdi_id;                    // PDI ID
        struct array_offset dpu_kernel_ids; // Array of dpu_kernel_ids (uint64_t)
        struct array_offset pre_cdo_groups; // Array of Pre CDO Group IDs (uint32_t)
        uint8_t reserved[64];               // Reserved
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct cdo_group) == 96, "cdo_group structure no longer is 96 bytes in size");
    XCLBIN_STATIC_ASSERT(sizeof(struct cdo_group) % sizeof(uint64_t) == 0, "cdo_group structure needs to be 64-bit word aligned");


    struct aie_pdi {
        xuid_t uuid;                        // PDI container UUID (16 bytes)
        struct array_offset pdi_image;      // PDI Image (uint8_t)
        struct array_offset cdo_groups;     // Array of cdo_groups (cdo_group)
        uint8_t reserved[64];               // Reserved
    };

    XCLBIN_STATIC_ASSERT(sizeof(struct aie_pdi) == 96, "aie_pdi structure no longer is 96 bytes in size");
    XCLBIN_STATIC_ASSERT(sizeof(struct aie_pdi) % sizeof(uint64_t) == 0, "aie_pdi structure needs to be 64-bit word aligned");

    struct aie_partition_info {
        uint16_t column_width;              // Width of the partition
        uint8_t padding[6];                 // Byte alignment
        struct array_offset start_columns;  // Array of start column identifiers (uint16_t)
        uint8_t reserved[72];               // Reserved
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct aie_partition_info) == 88, "partition_info structure no longer is 88 bytes in size");

    struct aie_partition {
        uint8_t schema_version;             // Group schema version (default 0)
        uint8_t padding0[3];                // Byte alignment
        uint32_t mpo_name;                  // Name of the aie_partition
        uint32_t operations_per_cycle;      // Operations per cycle. Used later to create TOPS (operations_per_cycle * <AIE Clock Frequency>)
        uint8_t padding[4];
        uint64_t inference_fingerprint;     // The unique hash value of the inference function
        uint64_t pre_post_fingerprint;      // The unique hash value of pre post 
        struct aie_partition_info info;     // Partition information
        struct array_offset aie_pdi;        // PDI Array (aie_partition_info)
        uint8_t reserved[54];               // Reserved
    };
    XCLBIN_STATIC_ASSERT(sizeof(struct aie_partition) == 184, "aie_partition structure no longer is 184 bytes in size");
    XCLBIN_STATIC_ASSERT(sizeof(struct aie_partition) % sizeof(uint64_t) == 0, "aie_partition structure needs to be 64-bit word aligned");

    /**** END : Xilinx internal section *****/

# ifdef __cplusplus
    namespace xclbin {
      inline const axlf_section_header*
      get_axlf_section(const axlf* top, axlf_section_kind kind)
      {
        auto begin = top->m_sections;
        auto end = begin + top->m_header.m_numSections;
        auto itr = std::find_if(begin,end,[kind](const axlf_section_header& sec) { return sec.m_sectionKind == static_cast<uint32_t>(kind); });
        return (itr!=end) ? &(*itr) : nullptr;
      }

      // Helper C++ section iteration
      // To keep with with the current "coding" them, the function get_axlf_section_next() was
      // introduced find 'next' common section names.
      //
      // Future TODO: Create a custom iterator and refactor the code base to use it.
      //
      // Example on how this function may be used:
      //
      // const axlf_section_header * pSection;
      // const axlf* top = <xclbin image in memory>;
      // for (pSection = xclbin::get_axlf_section( top, SOFT_KERNEL);
      //      pSection != nullptr;
      //      pSection = xclbin::get_axlf_section_next( top, pSection, SOFT_KERNEL)) {
      //   <code to do work>
      // }
      inline const axlf_section_header*
      get_axlf_section_next(const axlf* top, const axlf_section_header* current_section, axlf_section_kind kind)
      {
        if (top == nullptr) { return nullptr; }
        if (current_section == nullptr) { return nullptr; }

        auto end = top->m_sections + top->m_header.m_numSections;

        auto begin = current_section + 1;        // Point to the next section
        if (begin == end) { return nullptr; }

        auto itr = std::find_if(begin, end, [kind](const axlf_section_header &sec) {return sec.m_sectionKind == static_cast<uint32_t>(kind); });
        return (itr!=end) ? &(*itr) : nullptr;
      }
    }
# endif

#ifdef __cplusplus
}
#endif

#endif
