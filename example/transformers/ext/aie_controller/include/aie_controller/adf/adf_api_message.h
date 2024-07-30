/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
/**
* Copyright (C) 2021 Xilinx, Inc
*
*/

#ifndef __ADF_API_MESSAGE_H__
#define __ADF_API_MESSAGE_H__

#include "errno.h"
#include <string>

namespace aiectrl
{
enum class err_code : int
{
    ok = 0,
    user_error = EINVAL,
    internal_error = ENOTSUP,
    aie_driver_error = EIO,
    resource_unavailable = EAGAIN

};

}

#endif