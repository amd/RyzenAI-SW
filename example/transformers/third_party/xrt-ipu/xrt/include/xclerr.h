/*
 *  SPDX-License-Identifier: Apache-2.0
 *  Copyright (C) 2017, Xilinx Inc
 *  Copyright (C) 2022, Advanced Micro Devices, Inc.  All rights reserved.
 */

/**
 * DOC: Device error status related structs and defines
 * This file is used by both userspace and xclmgmt kernel driver.
 */

#ifndef XCLERR_H_
#define XCLERR_H_

/**
 * enum xclFirewallID - AXI Firewall IDs used to identify individual AXI Firewalls
 *
 * @XCL_FW_MGMT_CONTROL:  MGMT BAR AXI-Lite BAR access protection
 * @XCL_FW_USER_CONTROL:  USER BAR AXI-Lite BAR access protection
 * @XCL_FW_DATAPATH:      DMA data path protection
 */
enum xclFirewallID {
        XCL_FW_MGMT_CONTROL = 0,
        XCL_FW_USER_CONTROL,
        XCL_FW_DATAPATH,
        XCL_FW_MAX_LEVEL // always the last one
};

/**
 * struct xclAXIErrorStatus - Record used to capture specific error
 *
 * @mErrFirewallTime:    Timestamp of when Firewall tripped
 * @mErrFirewallStatus:  Error code obtained from the Firewall
 * @mErrFirewallID:      Firewall ID
 */
struct xclAXIErrorStatus {
        unsigned long       mErrFirewallTime;
        unsigned            mErrFirewallStatus;
        enum xclFirewallID  mErrFirewallID;
};

struct xclPCIErrorStatus {
        unsigned mDeviceStatus;
        unsigned mUncorrErrStatus;
        unsigned mCorrErrStatus;
        unsigned rsvd1;
        unsigned rsvd2;
};

/**
 * struct xclErrorStatus - Container for all error records
 *
 * @mNumFirewalls:    Count of Firewalls in the record (max is 8)
 * @mAXIErrorStatus:  Records holding Firewall information
 * @mPCIErrorStatus:  Unused
 */
struct xclErrorStatus {
        unsigned  mNumFirewalls;
        struct xclAXIErrorStatus mAXIErrorStatus[8];
        struct xclPCIErrorStatus mPCIErrorStatus;
        unsigned mFirewallLevel;
};

#endif /* XCLERR_H_ */
