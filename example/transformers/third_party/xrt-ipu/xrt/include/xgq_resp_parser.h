/*
 *  Copyright (C) 2021-2022, Xilinx Inc
 *
 *  This file is dual licensed.  It may be redistributed and/or modified
 *  under the terms of the Apache 2.0 License OR version 2 of the GNU
 *  General Public License.
 *
 *  Apache License Verbiage
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  GPL license Verbiage:
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as
 *  published by the Free Software Foundation; either version 2 of the
 *  License, or (at your option) any later version.  This program is
 *  distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 *  License for more details.  You should have received a copy of the
 *  GNU General Public License along with this program; if not, write
 *  to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
 *  Boston, MA 02111-1307 USA
 *
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
