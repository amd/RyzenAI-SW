//
// Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_API_COMMON_H_
#define EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_API_COMMON_H_

#include <inttypes.h>

#ifndef WIN32
#define CVML_SDK_EXPORT
#define CVML_SDK_EXPORT_CORE
#define CVML_SDK_NO_EXPORT
#define CVML_SDK_DEPRECATED
#define CVML_SDK_DEPRECATED_EXPORT
#define CVML_SDK_DEPRECATED_NO_EXPORT
#else

#ifdef CVML_SDK_STATIC_DEFINE
#define CVML_SDK_EXPORT
#define CVML_SDK_EXPORT_CORE
#define CVML_SDK_NO_EXPORT
#else

#ifndef CVML_SDK_EXPORT_CORE
#ifdef cvml_sdk_EXPORTS_0
/* We are building this core library */
#define CVML_SDK_EXPORT_CORE __declspec(dllexport)
#else
/* We are using this core library */
#define CVML_SDK_EXPORT_CORE __declspec(dllimport)
#endif
#endif

#ifndef CVML_SDK_EXPORT
#ifdef cvml_sdk_EXPORTS
/* We are building this library */
#define CVML_SDK_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define CVML_SDK_EXPORT __declspec(dllimport)
#endif
#endif

#ifndef CVML_SDK_NO_EXPORT
#define CVML_SDK_NO_EXPORT
#endif
#endif

#ifndef CVML_SDK_DEPRECATED
#define CVML_SDK_DEPRECATED __attribute__((__deprecated__))
#endif

#ifndef CVML_SDK_DEPRECATED_EXPORT
#define CVML_SDK_DEPRECATED_EXPORT CVML_SDK_EXPORT CVML_SDK_DEPRECATED
#endif

#ifndef CVML_SDK_DEPRECATED_NO_EXPORT
#define CVML_SDK_DEPRECATED_NO_EXPORT CVML_SDK_NO_EXPORT CVML_SDK_DEPRECATED
#endif

#endif

#define AMD_CVML_INTERFACE(TypeName)             \
 public:                                         \
  virtual ~TypeName();                           \
                                                 \
 protected:                                      \
  TypeName();                                    \
  TypeName(const TypeName&) = delete;            \
  TypeName& operator=(const TypeName&) = delete; \
  TypeName(TypeName&&) noexcept = delete;        \
  TypeName& operator=(TypeName&&) noexcept = delete;

#endif  // EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_API_COMMON_H_
