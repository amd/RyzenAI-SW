/**
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (C) 2019-2021 Xilinx, Inc. All rights reserved.
 * Copyright (C) 2023 Advanced Micro Device, Inc. All rights reserved.
 */

#ifndef _XRT_VERSION_H_
#define _XRT_VERSION_H_

static const char xrt_build_version[] = "2.17.0";

static const char xrt_build_version_branch[] = "HEAD";

static const char xrt_build_version_hash[] = "a395e702b2e79b3ec23c9cdc3ab4ad31a0d84eab";

static const char xrt_build_version_hash_date[] = "Fri, 9 Feb 2024 19:02:43 -0800";

static const char xrt_build_version_date_rfc[] = "Fri, 29 Mar 2024 15:08:40 -0400";

static const char xrt_build_version_date[] = "2024-03-29 15:08:40";

static const char xrt_modified_files[] = "";

#define XRT_DRIVER_VERSION "2.17.0,a395e702b2e79b3ec23c9cdc3ab4ad31a0d84eab"

#define XRT_VERSION(major, minor) ((major << 16) + (minor))
#define XRT_VERSION_CODE XRT_VERSION(2, 17)
#define XRT_MAJOR(code) ((code >> 16))
#define XRT_MINOR(code) (code - ((code >> 16) << 16))
#define XRT_PATCH 0
#define XRT_HEAD_COMMITS 7476
#define XRT_BRANCH_COMMITS 0

#ifdef __cplusplus
#include <iostream>
#include <string>

namespace xrt::version {

inline void
print(std::ostream & output)
{
  output << "       XRT Build Version: " << xrt_build_version << std::endl;
  output << "    Build Version Branch: " << xrt_build_version_branch << std::endl;
  output << "      Build Version Hash: " << xrt_build_version_hash << std::endl;
  output << " Build Version Hash Date: " << xrt_build_version_hash_date << std::endl;
  output << "      Build Version Date: " << xrt_build_version_date_rfc << std::endl;

  std::string modified_files(xrt_modified_files);
  if (modified_files.empty())
    return;

  const std::string& delimiters = ",";      // Our delimiter
  std::string::size_type last_pos = 0;
  int running_index = 1;
  while (last_pos < modified_files.length() + 1) {
    if (running_index == 1)
      output << "  Current Modified Files: ";
    else 
      output << "                          ";

    output << running_index++ << ") ";

    auto pos = modified_files.find_first_of(delimiters, last_pos);

    if (pos == std::string::npos)
      pos = modified_files.length();

    output << modified_files.substr(last_pos, pos - last_pos) << std::endl;
    
    last_pos = pos + 1;
  }
}

} // namespace xrt::version 
#endif // __cplusplus

#endif

