/*
 *  SPDX-License-Identifier: Apache-2.0
 *  Copyright (C) 2021-2022, Xilinx Inc
 *  Copyright (C) 2022, Advanced Micro Devices, Inc.  All rights reserved.
 */
#ifndef XGQ_CMD_COMMON_H
#define XGQ_CMD_COMMON_H

/* !!! This header file is for internal project use only and it is subject to removal without notice !!! */
#if defined(__KERNEL__)
# include <linux/types.h>
#else
# include <stdint.h>
# include <stddef.h>
# include <errno.h>
#endif

/**
 * This header file could be used both in driver and user space applications.
 * It could be used cross platform as well. Need assert macro.
 */
#ifdef __cplusplus
  // Used C++ static assert
  #define XGQ_STATIC_ASSERT(e,m) \
    static_assert (e, m)
#else
  // Use our "custom" kernel compilation assert
  #define XGQ_ASSERT_CONCAT_(a, b) a##b
  #define XGQ_ASSERT_CONCAT(a, b) XGQ_ASSERT_CONCAT_(a, b)

  // Create an artifitial assertion via a bad divide by zero assertion.
  #define XGQ_STATIC_ASSERT(e,m) \
    enum { XGQ_ASSERT_CONCAT(xgq_assert_line_, __LINE__) = 1/(int)(!!(e)) }
#endif

#define XGQ_SUB_Q1_SLOT_SIZE	512	// NOLINT
#define XGQ_QUEUE1_SLOT_NUM	4	// NOLINT
#define XGQ_QUEUE1_SLOT_MASK	(XGQ_QUEUE1_SLOT_NUM - 1)	// NOLINT

#define XGQ_Q1_SUB_SIZE		(XGQ_SUB_Q1_SLOT_SIZE * XGQ_QUEUE1_SLOT_NUM) // NOLINT
#define XGQ_Q1_COM_SIZE		(XGQ_COM_Q1_SLOT_SIZE * XGQ_QUEUE1_SLOT_NUM) // NOLINT

/* Opcode encoding rules:
 * | 15 ------ 11 | 10 ----- 8 | 7 ----- 0 |
 * +--------------+------------+-----------+
 * |   Reserved   |    Type    |  OP's ID  |
 * +--------------+------------+-----------+
 */
enum xgq_cmd_opcode {
	/* Management command type */
	XGQ_CMD_OP_LOAD_XCLBIN		= 0x0,
	XGQ_CMD_OP_CONFIGURE		= 0x1,

	XGQ_CMD_OP_GET_LOG_PAGE		= 0x8,

	XGQ_CMD_OP_DOWNLOAD_PDI		= 0xa,
	XGQ_CMD_OP_CLOCK		= 0xb,
	XGQ_CMD_OP_SENSOR		= 0xc,
	XGQ_CMD_OP_LOAD_APUBIN		= 0xd,
	XGQ_CMD_OP_VMR_CONTROL		= 0xe,
	XGQ_CMD_OP_PROGRAM_SCFW		= 0xf,

	/* User command type */
	XGQ_CMD_OP_START_CUIDX	        = 0x100,
	XGQ_CMD_OP_START_CUIDX_INDIR	= 0x101,
	XGQ_CMD_OP_START_CUIDX_KV	= 0x102,
	XGQ_CMD_OP_START_CUIDX_KV_INDIR	= 0x103,
	XGQ_CMD_OP_INIT_CUIDX	        = 0x104,
	XGQ_CMD_OP_INIT_CUIDX_INDIR     = 0x105,
	XGQ_CMD_OP_INIT_CUIDX_KV	= 0x106,
	XGQ_CMD_OP_INIT_CUIDX_KV_INDIR	= 0x107,
	XGQ_CMD_OP_CFG_START	        = 0x108,
	XGQ_CMD_OP_CFG_END	        = 0x109,
	XGQ_CMD_OP_CFG_CU	        = 0x10a,
	XGQ_CMD_OP_QUERY_CU	        = 0x10b,
	XGQ_CMD_OP_CLOCK_CALIB     	= 0x10c,
	XGQ_CMD_OP_ACCESS_VALID     	= 0x10d,
	XGQ_CMD_OP_DATA_INTEGRITY   	= 0x10e,
	XGQ_CMD_OP_EXIT             	= 0x10f,

	/* Common command type */
	XGQ_CMD_OP_BARRIER		= 0x200,
	XGQ_CMD_OP_EXIT_ERT		= 0x201,
	XGQ_CMD_OP_IDENTIFY		= 0x202,
};

enum xgq_cmd_addr_type {
	XGQ_CMD_ADD_TYPE_DEVICE		= 0x0,
	XGQ_CMD_ADD_TYPE_SLAVEBRIDGE	= 0x1,
	XGQ_CMD_ADD_TYPE_HOST_MEM	= 0x2,
	XGQ_CMD_ADD_TYPE_AP_OFFSET	= 0x3,
};

#define XGQ_SQ_CMD_NEW 1

enum xgq_cmd_state {
	XGQ_CMD_STATE_COMPLETED		= 0x0,
	XGQ_CMD_STATE_ABORTED		= 0x1,
	XGQ_CMD_STATE_TIMEOUT		= 0x2,
	XGQ_CMD_STATE_INVALID		= 0x3,
	XGQ_CMD_STATE_CONFLICT_ID	= 0x4,
};

enum xgq_cmd_page_id {
	XGQ_CMD_PAGE_ID_HEALTH		= 0x0,
	XGQ_CMD_PAGE_ID_ERROR_INFO	= 0x1,
	XGQ_CMD_PAGE_ID_PROFIL		= 0x2,
	XGQ_CMD_PAGE_ID_DEBUG		= 0x3,
	XGQ_CMD_PAGE_ID_SENSOR		= 0x4,
};

/*
 * On some platforms, XGQ IP and XGQ ring buffer can be located
 * on different hardware location, e.g. separate PCIe BARs. So
 * updating doorbell register can be faster than the ring buffer.
 * We have a special flag in Submission queue and Completion queue
 * entry to indicate this entry is a new one. This flag is located
 * at the first Word MSB in both Submission queue and Completion
 * entry. After receiving doorbell update interrupt, consumer will
 * need to check this flag as well to make sure this is a new entry.
 * After the check, consumer also needs to clear it so that it won't
 * be confused by the stale data next time when it comes to this slot.
 * To handle this case, XGQ_OUT_OF_ORDER_WRITE macro shall be defined.
 *
 * Note: for same reason, the producer will make sure to write to
 *       Word0 as the last update of the Submission and Completion
 *       queue entry before writing to the doorbell register.
 */
#define XGQ_ENTRY_NEW_FLAG_MASK		0x80000000

/**
 * struct xgq_cmd_sq_hdr: XGQ submission queue entry header format
 *
 * @opcode:	[15-0]	command opcode identifying specific command
 * @count:	[30-16]	number of bytes representing packet payload
 * @state:	[31]	flag indicates this is a new entry
 * @cid:		unique command id
 * @rsvd:	        reserved for future use
 * @cu_domain:	[3-0]	CU domain for certain start CU op codes
 * @cu_idx:	[11-0]	CU index for certain start CU op codes
 *
 * Any command in XGQ submission queue shares same command header.
 * An command ID is used to identify the command. When the command
 * is completed, the same command ID is put into the completion
 * queue entry.
 *
 * Please declare this struct at the begin of a submission queue entry
 */
struct xgq_cmd_sq_hdr {
	union {
		struct {
			uint32_t opcode:16; /* [15-0]   */
			uint32_t count:15;  /* [30-16] */
			uint32_t state:1;   /* [31] */
			uint16_t cid;
			union {
				uint16_t rsvd;
				struct {
					uint16_t cu_idx:12;
					uint16_t cu_domain:4;
				};
			};
		};
		uint32_t header[2]; // NOLINT
	};
};
XGQ_STATIC_ASSERT(sizeof(struct xgq_cmd_sq_hdr) == 8, "xgq_cmd_sq_hdr structure no longer is 8 bytes in size");

/**
 * struct xgq_cmd_cq_hdr: XGQ completion queue entry header format
 *
 * @cid:        unique command id
 * @cstate:     command state
 * @specific:	flag indicates there is command specific info in result
 * @state:      flag indicates this is a new entry
 *
 * This is the header of the completion queue entry. A generic command
 * state is put into cstate. The command is identified by cid which
 * matches the cid in submission queue.
 *
 * Please declare this struct at the begin of a completion queue entry
 */
struct xgq_cmd_cq_hdr {
	union {
		struct {
			uint16_t cid;
			uint16_t cstate:14;
			uint16_t specific:1;
			uint16_t state:1;
		};
		uint32_t header[1]; // NOLINT
	};
};
XGQ_STATIC_ASSERT(sizeof(struct xgq_cmd_cq_hdr) == 4, "xgq_cmd_cq_hdr structure no longer is 4 bytes in size");

/**
 * struct xgq_sub_queue_entry: XGQ submission queue entry
 *
 * @hdr: header of the entry
 * @data: payload of the entry
 *
 * XGQ submission command is variable length command.
 * This is very useful when a XGQ entity needs to access command payload,
 * but it doesn't need to know the detail of the payload.
 */
struct xgq_sub_queue_entry {
	struct xgq_cmd_sq_hdr  hdr;
	uint32_t data[1]; // NOLINT
};

/**
 * struct xgq_com_queue_entry: XGQ completion queue entry format
 *
 * @hdr: header of the entry
 * @result: command specific result
 * @resvd: reserved
 * @rcode: POSIX error return code
 *
 * When a command is completed, a completion entry is put into completion
 * queue. A generic command state is put into cstate. The command is
 * identified by cid which matches the cid in submission queue.
 * More command specific result is put into result field. POSIX error code
 * can be put into rcode. This is useful for some case like PS kernel.
 *
 * All completion queue entries have a same fixed size of 4 words.
 */
struct xgq_com_queue_entry {
	union {
		struct {
			struct xgq_cmd_cq_hdr hdr;
			uint32_t result;
			uint32_t resvd;
			uint32_t rcode;
		};
		uint32_t data[4]; // NOLINT
	};
};
XGQ_STATIC_ASSERT(sizeof(struct xgq_com_queue_entry) == 16, "xgq_com_queue_entry structure no longer is 16 bytes in size");

#define XGQ_SUB_HEADER_SIZE	(sizeof(struct xgq_cmd_sq_hdr)) // NOLINT
#define XGQ_COM_Q1_SLOT_SIZE	(sizeof(struct xgq_com_queue_entry)) // NOLINT

/**
 * struct xgq_cmd_load_xclbin: load XCLBIN command
 *
 * @hdr: header of the command
 * @address:	XCLBIN address
 * @size:	XCLBIN size in Byte
 * @addr_type:	Address type
 *
 * This command is used to load XCLBIN to device through XGQ.
 * This is an indirect command that XCLBIN blob's address is
 * embedded.
 */
struct xgq_cmd_load_xclbin {
	struct xgq_cmd_sq_hdr  hdr;
	uint64_t address;
	uint32_t size;
	uint32_t addr_type:4;
	uint32_t rsvd1:28;
};

struct xgq_cmd_configure {
	struct xgq_cmd_sq_hdr  hdr;
	uint32_t data[1]; // NOLINT
};

/**
 * struct xgq_cmd_indentify: identify command
 *
 * @hdr: header of the command
 *
 * This command is used to get XGQ command set version.
 * After that XGQ client XGQ server can support
 */
struct xgq_cmd_identify {
	struct xgq_cmd_sq_hdr  hdr;
};

/**
 * struct xgq_cmd_resp_indentify: identify command response
 *
 * @minor: minor version of the XGQ command set
 * @major: major version of the XGQ command set
 *
 * This command is used to get XGQ command set version information.
 */
struct xgq_cmd_resp_identify {
	struct xgq_cmd_cq_hdr  hdr;
	union {
		struct {
			uint16_t minor;
			uint16_t major;
		};
		uint32_t result;
	};
	uint32_t resvd;
	uint32_t rcode;
};

struct xgq_cmd_exit_ert {
	struct xgq_cmd_sq_hdr  hdr;
};

#endif
