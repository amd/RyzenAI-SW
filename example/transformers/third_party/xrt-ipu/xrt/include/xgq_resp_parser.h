/*
 *  SPDX-License-Identifier: Apache-2.0
 *  Copyright (C) 2021-2022, Xilinx Inc
 *  Copyright (C) 2022, Advanced Micro Devices, Inc.  All rights reserved.
 */
#ifndef XGQ_RESP_PARSER_H
#define XGQ_RESP_PARSER_H

#define SDR_NULL_BYTE	0x00
#define SDR_LENGTH_MASK	0x3F
#define SDR_TYPE_MASK	0x03
#define SDR_TYPE_POS	6

#define SDR_COMPLETE_IDX  0
#define SDR_REPO_IDX      1
#define SDR_REPO_VER_IDX  2
#define SDR_NUM_REC_IDX   3
#define SDR_NUM_BYTES_IDX 4
#define SDR_HEADER_SIZE   5

#define SDR_EOR_BYTES     3

#define THRESHOLD_UPPER_WARNING_MASK	(0x1 << 0)
#define THRESHOLD_UPPER_CRITICAL_MASK	(0x1 << 1)
#define THRESHOLD_UPPER_FATAL_MASK		(0x1 << 2)
#define THRESHOLD_LOWER_WARNING_MASK	(0x1 << 3)
#define THRESHOLD_LOWER_CRITICAL_MASK	(0x1 << 4)
#define THRESHOLD_LOWER_FATAL_MASK		(0x1 << 5)
#define THRESHOLD_SENSOR_AVG_MASK		(0x1 << 6)
#define THRESHOLD_SENSOR_MAX_MASK		(0x1 << 7)

#define SENSOR_IDS_MAX				256

enum xgq_sdr_repo_type {
    SDR_TYPE_GET_SIZE     = 0x00,
    SDR_TYPE_BDINFO       = 0xC0,
    SDR_TYPE_TEMP         = 0xC1,
    SDR_TYPE_VOLTAGE      = 0xC2,
    SDR_TYPE_CURRENT      = 0xC3,
    SDR_TYPE_POWER        = 0xC4,
    SDR_TYPE_QSFP         = 0xC5,
    SDR_TYPE_VPD_PCIE     = 0xD0,
    SDR_TYPE_IPMIFRU      = 0xD1,
    SDR_TYPE_CSDR_LOGDATA = 0xE0,
    SDR_TYPE_VMC_LOGDATA  = 0xE1,
    SDR_TYPE_MAX	  = 11,//increment if new entry added in this enum
};

enum xgq_sdr_completion_code {
    SDR_CODE_NOT_AVAILABLE            = 0x00,
    SDR_CODE_OP_SUCCESS               = 0x01,
    SDR_CODE_OP_FAILED                = 0x02,
    SDR_CODE_FLOW_CONTROL_READ_STALE  = 0x03,
    SDR_CODE_FLOW_CONTROL_WRITE_ERROR = 0x04,
    SDR_CODE_INVALID_SENSOR_ID        = 0x05,
};

enum sensor_record_fields {
    SENSOR_ID = 0,
    SENSOR_NAME_TL,
    SENSOR_NAME,
    SENSOR_VALUE_TL,
    SENSOR_VALUE,
    SENSOR_BASEUNIT_TL,
    SENSOR_BASEUNIT,
    SENSOR_UNIT_MODIFIER,
    SENSOR_THRESHOLD_SUPPORT,
    SENSOR_LOWER_FATAL,
    SENSOR_LOWER_CRITICAL,
    SENSOR_LOWER_WARNING,
    SENSOR_UPPER_FATAL,
    SENSOR_UPPER_CRITICAL,
    SENSOR_UPPER_WARNING,
    SENSOR_STATUS,
    SENSOR_MAX_VAL,
    SENSOR_AVG_VAL,
};

enum sensor_status {
    SENSOR_NOT_PRESENT          = 0x00,
    SENSOR_PRESENT_AND_VALID    = 0x01,
    DATA_NOT_AVAILABLE          = 0x02,
    SENSOR_STATUS_NOT_AVAILABLE = 0x7F,
};

#endif
