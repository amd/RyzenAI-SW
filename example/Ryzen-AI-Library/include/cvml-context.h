/*!
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * @file
 *
 * Definitions for CVML SDK Contexts and associated structures/functions.
 */

#ifndef EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_CONTEXT_H_
#define EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_CONTEXT_H_

#include "cvml-api-common.h"
#include "cvml-logger.h"

namespace amd {
namespace cvml {

/**
 * Maximal number of different platforms CVML SDK can support
 */
static const uint32_t MAX_SUPPORTED_PLATFORMS = 10;

/**
 * Information of platforms supported by CVML SDK
 * @see \a amd::cvml::Context
 */
typedef struct SupportedPlatformInformation {
  struct SupportedPlatform {
    /// Device ID of supported AMD APU
    int64_t device_id;

    /// Required mininal vulkan driver version on supported AMD APU
    int64_t required_gpu_minimal_vulkan_driver_version;
  } platform[MAX_SUPPORTED_PLATFORMS];

  /// Total number of supported AMD APU platforms.
  /// @see \a amd::cvml::MAX_SUPPORTED_PLATFORMS
  uint32_t supported_platform_count;

  /// Whether supported platform checking is enforced.
  bool checking_enforced;
} SupportedPlatformInformation;

/**
 * Represents a context of a CVML SDK feature
 * Can be shared by multiple features of the CVML SDK.
 */
class CVML_SDK_EXPORT Context {
  AMD_CVML_INTERFACE(Context);

 public:
  /**
   * Releases the memory allocated by the context.
   */
  virtual void Release() = 0;

  /**
   * Sets the verbosity of the log.
   *
   * @param level CVML SDK feature log level
   */
  virtual void SetLogLevel(CvmlLogger::LogLevels level) = 0;

  /**
   * Gets the pointer to the cvml logger
   */
  virtual CvmlLogger* GetLogger() const = 0;

  /**
   * Get the Supported Platform Information object
   *
   * @param info Pointer to structure for receiving platform information
   * @return True on success
   * @return False on failure
   */
  static bool GetSupportedPlatformInformation(amd::cvml::SupportedPlatformInformation* info);

  /**
   * Defines the inference backends that can be supported by the CVML SDK.
   *
   * These are provided to the \a SetInferenceBackend API function.
   */
  enum InferenceBackend {
    AUTO,  ///< Allow the CVML SDK to select the hardware for inference operations
    GPU,   ///< Use GPU hardware for inference operations
    NPU,   ///< Use NPU hardware for inference operations
    CPU,   ///< Use CPU hardware for inference operations
    ONNX,  ///< Use NPU hardware for ONNX inference operations
  };

  /**
   * Define input source streaming mode that can be supported by CVMLSDK
   *
   */
  enum StreamingMode {
    ONE_SHOT,          ///< Input source is image
    ONLINE_STREAMING,  ///< Input source is video/audio file, or camera stream
    OFFLINE_STREAMING  ///< Input source is image playback
  };

  /**
   * Specifies the inference backend for subsequently created features.
   *
   * This function does not affect any CVML features that were instantiated
   * via the context before its call. If a CVML feature is unable to support
   * a specified inference backend, it will refuse to construct and an
   * exception will be thrown instead.
   *
   * @param inference_backend Desired hardware inference backend
   */
  virtual void SetInferenceBackend(InferenceBackend inference_backend) = 0;

  /**
   * Returns the inference backend selection strategy for newly created features.
   *
   * @return Current hardware inference backend selection
   */
  virtual InferenceBackend GetInferenceBackend(void) = 0;

  /**
   * Returns StreamingMode of input source, an enum class
   *
   */
  virtual StreamingMode GetStreamingMode(void) = 0;

  /**
   * Set input source type
   *  0: one-shot image
   *  1: online streaming mode (e.g. streaming video/audio, camera)
   *  2: offline streaming model (e.g. image loop playback)
   */
  virtual void SetStreamingMode(StreamingMode mode) = 0;
};

/**
 * API to Create CVML Context.
 *
 * @param log_level  Sets the log level. Default value is kINFO
 * @param logger External logger for cvml context. Default value is nullptr
 * @see \a amd::cvml::CvmlLogger
 * @return Pointer to the created Context
 */
CVML_SDK_EXPORT amd::cvml::Context* CreateContext(
    CvmlLogger::LogLevels log_level = CvmlLogger::LogLevels::kINFO, CvmlLogger* logger = nullptr);

}  // namespace cvml
}  // namespace amd

#endif  // EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_CONTEXT_H_
