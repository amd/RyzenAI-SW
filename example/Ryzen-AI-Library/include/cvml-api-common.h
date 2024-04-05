//
// Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_API_COMMON_H_
#define EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_API_COMMON_H_

#include <inttypes.h>

#ifdef CVML_SDK_STATIC_DEFINE
#define CVML_SDK_EXPORT
#define CVML_SDK_NO_EXPORT
#else
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

namespace amd {
namespace cvml {

/**
 * Encapsulates success or failure of an API call.
 */
template <typename R, typename F>
struct Result {
  R result; /**< result of running or building models*/
  F error;  /**< error code*/
  /**
   * Implementation of operator bool for Result
   */
  explicit operator bool() const { return error == F::kSuccess; }
  /**
   * Implementation of operator-> for Result
   */
  R operator->() const { return result; }
  /**
   * Implementation of operator() for Result
   */
  R operator()() const { return result; }
};

/**
 * Result output due to success
 * @param r result to be returned from an successful operation
 * @return result object of success
 */
template <typename R, typename F>
const Result<R, F> Success(const R& r) {
  Result<R, F> ret;
  ret.result = r;
  ret.error = F::kSuccess;
  return ret;
}

/**
 * Result output due to failure
 * @param f error code from an failed operation
 * @return result object of failure
 */
template <typename R, typename F>
const Result<R, F> Error(F f) {
  Result<R, F> ret;
  ret.result = {};
  ret.error = f;
  return ret;
}

}  // namespace cvml
}  // namespace amd

#endif  // EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_API_COMMON_H_
