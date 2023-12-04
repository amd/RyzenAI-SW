/**
 * Copyright (C) 2020 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

/**
 * This file contains publically exported xclbin utilities.
 */

#ifndef xclbin_util_h_
#define xclbin_util_h_

#include "xclbin.h"

static inline const axlf*
xclbin_axlf_handle(const void *xclbin)
{
  const axlf* top = (const axlf*) xclbin;
  return (strncmp(top->m_magic,"xclbin2",7)) ? NULL : top;
}

/**
 * xclbin_uuid() - Get the uuid of an xclbin
 *
 * @xclbin:  Raw pointer the enture xclbin file content
 * @out:     A uuid to populate with the xclbin uuid
 * Return:   0 on success, errcode otherwise
 */
static inline int
xclbin_uuid(const void *xclbin, xuid_t out)
{
  const axlf* top = xclbin_axlf_handle(xclbin);
  return (top) ? uuid_copy(out, top->m_header.uuid) , 0 : -EINVAL;
}

#endif
