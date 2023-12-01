/**
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (C) 2019-2021 Xilinx, Inc. All rights reserved.
 */

#ifndef _XRT_VERSION_H_
#define _XRT_VERSION_H_

static const char xrt_build_version[] = "2.14.0";

static const char xrt_build_version_branch[] = "HEAD";

static const char xrt_build_version_hash[] = "6c6a8bddd007847d8125213c91c2f13ae61cf4ca";

static const char xrt_build_version_hash_date[] = "Tue, 15 Aug 2023 10:41:12 -0700";

static const char xrt_build_version_date_rfc[] = "Wed, 16 Aug 2023 12:14:06 -0400";

static const char xrt_build_version_date[] = "2023-08-16 12:14:06";

static const char xrt_modified_files[] = "";

#define XRT_DRIVER_VERSION "2.14.0,6c6a8bddd007847d8125213c91c2f13ae61cf4ca"

#define XRT_VERSION(major, minor) ((major << 16) + (minor))
#define XRT_VERSION_CODE XRT_VERSION(2, 14)
#define XRT_MAJOR(code) ((code >> 16))
#define XRT_MINOR(code) (code - ((code >> 16) << 16))

# ifdef __cplusplus
#include <iostream>
#include <string>

namespace xrt {

class version {
 public:
  static void print(std::ostream & output)
  {
     output << "       XRT Build Version: " << xrt_build_version << std::endl;
     output << "    Build Version Branch: " << xrt_build_version_branch << std::endl;
     output << "      Build Version Hash: " << xrt_build_version_hash << std::endl;
     output << " Build Version Hash Date: " << xrt_build_version_hash_date << std::endl;
     output << "      Build Version Date: " << xrt_build_version_date_rfc << std::endl;

     std::string modifiedFiles(xrt_modified_files);
     if ( !modifiedFiles.empty() ) {
        const std::string& delimiters = ",";      // Our delimiter
        std::string::size_type lastPos = 0;
        int runningIndex = 1;
        while(lastPos < modifiedFiles.length() + 1) {
          if (runningIndex == 1) {
             output << "  Current Modified Files: ";
          } else {
             output << "                          ";
          }
          output << runningIndex++ << ") ";

          std::string::size_type pos = modifiedFiles.find_first_of(delimiters, lastPos);

          if (pos == std::string::npos) {
            pos = modifiedFiles.length();
          }

          output << modifiedFiles.substr(lastPos, pos-lastPos) << std::endl;

          lastPos = pos + 1;
        }
     }
  }
};
}
#endif

#endif
