// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_DETAIL_CONFIG_H
#define XRT_DETAIL_CONFIG_H

//------------------Enable dynamic linking on windows-------------------------//

#ifdef _WIN32
# ifdef XRT_API_SOURCE
#  define XRT_API_EXPORT __declspec(dllexport)
# else
#  define XRT_API_EXPORT __declspec(dllimport)
# endif
#endif
#ifdef __GNUC__
# ifdef XRT_API_SOURCE
#  define XRT_API_EXPORT __attribute__ ((visibility("default")))
# else
#  define XRT_API_EXPORT
# endif
#endif

#ifndef XRT_API_EXPORT
# define XRT_API_EXPORT
#endif

#endif
