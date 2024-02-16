/*
 *  SPDX-License-Identifier: Apache-2.0
 *  Copyright (C) 2021-2022, Xilinx Inc
 *  Copyright (C) 2022, Advanced Micro Devices, Inc.  All rights reserved.
 */
#ifndef XGQ_CMD_ERT_H
#define XGQ_CMD_ERT_H

/* !!! This header file is for internal project use only and it is subject to removal without notice !!! */
#include "xgq_cmd_common.h"

/* This header file defines data structure format for user opcodes. */

/**
 * struct xgq_cmd_start_cuidx: start CU by index command
 *
 * @hdr:	CU command header
 * @data:	CU parameters
 *
 * This command is used to start a specific CU with its index. And the
 * CU parameters are embedded in the command payload.
 */
struct xgq_cmd_start_cuidx {
	struct xgq_cmd_sq_hdr hdr;
	uint32_t data[1]; // NOLINT
};

/**
 * struct xgq_cmd_start_cuidx_kv: start CU with offset-value pairs by index command
 *
 * @hdr:	CU command header
 * @data:	cu parameters in a [offset:value] list. The offset is in bytes.
 *
 * This command is used to start a specific CU with its index. And the
 * [offset:value] list are embedded in the command payload.
 */
struct xgq_cmd_start_cuidx_kv {
	struct xgq_cmd_sq_hdr hdr;
	/*
	 * even index (0, 2, 4, ...) are offsets
	 * odd index (1, 3, 5, ...) are values
	 */
	uint32_t data[1]; // NOLINT
};

/**
 * struct xgq_cmd_init_cuidx: init CU by index command
 *
 * @hdr:	CU command header
 * @offset:	Offset to start initialize CU parameters
 * @data:	CU parameters
 *
 * This command is used to initial a specific CU with its index. And the
 * CU parameters are embedded in the command payload.
 *
 * This command would NOT kick off the CU.
 */
struct xgq_cmd_init_cuidx {
	struct xgq_cmd_sq_hdr hdr;
	uint32_t offset;
	uint32_t data[1]; // NOLINT
};

/**
 * struct xgq_cmd_init_cuidx_kv: init CU with offset-value pairs by index command
 *
 * @hdr:	CU command header
 * @data:	CU parameters in a [offset:value] list. The offset is in bytes.
 *
 * This command is used to initial a specific CU with its index. And the
 * [offset:value] list are embedded in the command payload.
 *
 * This command would NOT kick off the CU.
 */
struct xgq_cmd_init_cuidx_kv {
	struct xgq_cmd_sq_hdr hdr;
	/*
	 * even index (0, 2, 4, ...) are offsets
	 * odd index (1, 3, 5, ...) are values
	 */
	uint32_t data[1]; // NOLINT
};

/**
 * struct xgq_cmd_config_start: configure start command
 *
 * @num_cus: number of CUs
 * @i2h: ERT interrupt to host enable
 * @i2e: Host interrupt to ERT enable
 * @cui: CU interrupt to ERT enable
 *
 * This command would let ERT goes into configure state
 */
struct xgq_cmd_config_start {
	struct xgq_cmd_sq_hdr hdr;

	/* word 2 */
	uint32_t num_cus:13;
	uint32_t i2h:1;
	uint32_t i2e:1;
	uint32_t cui:1;
	uint32_t mode:2;
	uint32_t echo:1;
	uint32_t verbose:1;
	uint32_t resvd:12;

	/* word 3 */
	uint32_t num_scus:32;
};

/**
 * struct xgq_cmd_clock_calib: xgq clock counter command
 *
 * This command would let ERT read the clock counter
 */

struct xgq_cmd_clock_calib {
	struct xgq_cmd_sq_hdr hdr;
};

/**
 * struct xgq_cmd_access_valid: ERT performance measurement
 *
 * This command would measure the performance number of ERT accessing peripherals
 */

struct xgq_cmd_access_valid {
	struct xgq_cmd_sq_hdr hdr;
};

/**
 * struct xgq_cmd_data_integrity: queue data integrity test
 *
 * @rw_count: the number of write operation remaining
 * @draft:    the offset of read/write operation
 *
 * This command would let ERT test host/device accessibility
 */

struct xgq_cmd_data_integrity {
	struct xgq_cmd_sq_hdr hdr;

	uint32_t rw_count;
	uint32_t draft;
};
/**
 * struct xgq_cmd_resp_config_start: configure start command response
 *
 * @i2h: ERT interrupt to host enabled
 * @i2e: Host interrupt to ERT enable
 * @cui: CU interrupt to ERT enable
 * @ob:  device supports out of band memory
 *
 * The response of start configure command.
 */
struct xgq_cmd_resp_config_start {
	struct xgq_cmd_cq_hdr hdr;

	uint32_t i2h:1;
	uint32_t i2e:1;
	uint32_t cui:1;
	uint32_t ob:1;
	uint32_t rsvd:28;
	uint32_t resvd;
	uint32_t rcode;
};

/**
 * struct xgq_cmd_config_end: configure end command
 *
 * This command has no payload. Once ERT received this command it knowns configure is done.
 */
struct xgq_cmd_config_end {
	struct xgq_cmd_sq_hdr hdr;
};

/**
 * struct xgq_cmd_config_cu: configure CU command
 *
 * @cu_idx: cu index to configure
 * @ip_ctrl: IP control protocol
 * @map_size: use this size to map CU if apply
 * @laddr: lower 32 bits of the CU address
 * @haddr: higher 32 bits of the CU address
 * @payload_size: CU XGQ slot payload size
 * @name: name of the CU
 * @uuid: UUID of the XCLBIN of the CU
 *
 * Configure PL/PS CUs.
 */
struct xgq_cmd_config_cu {
	struct xgq_cmd_sq_hdr hdr;

	/* word 2 */
	uint32_t cu_idx:12;
	uint32_t cu_domain:4;
	uint32_t ip_ctrl:8;
	uint32_t rsvd2:8;

	uint32_t map_size;
	uint32_t laddr;
	uint32_t haddr;
	uint32_t payload_size;
	char name[64];
	unsigned char uuid[16];
};

/**
 * struct xgq_cmd_query_cu: query CU command
 *
 * @cu_idx: cu index to query
 * @type: type of the query
 *
 */
enum xgq_cmd_query_cu_type {
	XGQ_CMD_QUERY_CU_CONFIG =	0x0,
	XGQ_CMD_QUERY_CU_STATUS =	0x1,
};

struct xgq_cmd_query_cu {
	struct xgq_cmd_sq_hdr hdr;

	/* word 2 */
	uint32_t cu_idx:12;
	uint32_t cu_domain:4;
	uint32_t type:4;
	uint32_t rsvd2:8;
};

/**
 * struct xgq_cmd_resp_query_cu: query CU command response
 *
 * @status: status of the CU
 *
 * @xgq_id: XGQ ID
 * @type: type of the queue
 * @offset: ring buffer offset
 *
 */
enum xgq_cmd_resp_query_cu_type {
	XGQ_CMD_RESP_QUERY_XGQ =	0x0,
	XGQ_CMD_RESP_QUERY_OOO =	0x1,
};

struct xgq_cmd_resp_query_cu {
	struct xgq_cmd_cq_hdr hdr;

	union {
		struct {
			uint16_t status;
			uint16_t rsvd1;
			uint32_t resvd;
		};
		struct {
			uint32_t xgq_id:12;
			uint32_t rsvd2:4;
			uint32_t size:15;
			uint32_t type:1;
			uint32_t offset;
		};
	};
	uint32_t rcode;
};

/**
 * struct xgq_cmd_resp_clock_calib: query clock counter response
 *
 * @timestamp: the number of clock counter
 *
 */

struct xgq_cmd_resp_clock_calib {
	struct xgq_cmd_cq_hdr hdr;

	union {
		struct {
			uint32_t resvd;
			uint32_t timestamp;
		};
	};
	uint32_t rcode;
};

/**
 * struct xgq_cmd_resp_access_valid: query performance measurement response
 *
 * @cq_read_single: the cycle number of single queue read
 * @cq_write_single: the cycle number of single queue write
 * @cu_read_single: the cycle number of single CU read
 * @cu_write_single: the cycle number of single CU write
 *
 */

struct xgq_cmd_resp_access_valid {
	struct xgq_cmd_cq_hdr hdr;

	union {
		struct {
			uint16_t status;
			uint16_t rsvd1;
			uint32_t resvd;
		};
		struct {
			uint8_t cq_read_single;
			uint8_t cq_write_single;
			uint8_t cu_read_single;
			uint8_t cu_write_single;
		};
	};
	uint32_t rcode;
};

/**
 * struct xgq_cmd_resp_data_integrity: device accessibility response
 *
 * @h2d_access:     the result of host to device accessibility
 * @d2d_access:     the result of device accessibility
 * @d2cu_access:    the result of device to cu accessibility
 * @data_integrity: the result of stress test
 *
 */

struct xgq_cmd_resp_data_integrity {
	struct xgq_cmd_cq_hdr hdr;

	union {
		struct {
			uint32_t h2d_access:1;
			uint32_t d2d_access:1;
			uint32_t d2cu_access:1;
			uint32_t data_integrity:1;
			uint32_t resvd:28;
		};
	};
	uint32_t rcode;
};

/* Helper functions */
static inline uint32_t xgq_cmd_get_cu_payload_size(struct xgq_cmd_sq_hdr *hdr)
{
	uint32_t ret = hdr->count;

	if (hdr->opcode == XGQ_CMD_OP_INIT_CUIDX)
		ret -= sizeof(uint32_t);
	return ret;
}

static inline uint32_t xgq_cmd_is_cu_kv(struct xgq_cmd_sq_hdr *hdr)
{
	return (hdr->opcode == XGQ_CMD_OP_START_CUIDX_KV ||
		hdr->opcode == XGQ_CMD_OP_START_CUIDX_KV) ? 1 : 0;
}

#endif // XGQ_CMD_ERT_H
