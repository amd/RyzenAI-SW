/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#ifndef xaie_dynlink_h
#define xaie_dynlink_h
#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif
#ifdef RDI_BUILD
# ifdef _MSC_VER
#  ifdef XAIE_AIG_SOURCE
#   define XAIE_AIG_EXPORT __declspec(dllexport)
#  else
#   define XAIE_AIG_EXPORT __declspec(dllimport)
#  endif
# endif
# ifdef __GNUC__
#  ifdef XAIE_AIG_SOURCE
#   define XAIE_AIG_EXPORT __attribute__ ((visibility("default")))
#  else
#   define XAIE_AIG_EXPORT
#  endif
# endif
#endif
#ifndef XAIE_AIG_EXPORT
# define XAIE_AIG_EXPORT
#endif
#endif // #ifndef xaie_dynlink_h
