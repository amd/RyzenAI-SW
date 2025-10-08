/*!
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * @file
 *
 * Definitions for SDK contexts and associated structures/functions.
 */

#ifndef EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_CONTEXT_H_
#define EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_CONTEXT_H_

#include "cvml-api-common.h"
#include "cvml-logger.h"

namespace amd {
namespace cvml {

/**
 * Maximum number of different platforms the SDK can support.
 */
static const uint32_t MAX_SUPPORTED_PLATFORMS = 10;

/**
 * Structure of platforms supported by the SDK.
 *
 * @see \a amd::cvml::Context
 */
struct SupportedPlatformInformation {
  /**
   * Structure describing a single supported platform.
   */
  struct SupportedPlatform {
    /// Device ID of supported AMD APU
    /// @deprecated Always returns -1
    int64_t device_id;

    /// Required minimum Vulkan driver version on supported AMD APU
    int64_t required_gpu_minimal_vulkan_driver_version;
  } platform[MAX_SUPPORTED_PLATFORMS];  ///< Array of supported platforms.

  /// Total number of supported AMD APU platforms.
  /// @deprecated Always returns amd::cvml::MAX_SUPPORTED_PLATFORMS
  /// @see \a amd::cvml::MAX_SUPPORTED_PLATFORMS
  uint32_t supported_platform_count;

  /// Whether supported platform checking is enforced.
  bool checking_enforced;
};

/**
 * Execution context for Ryzen AI CVML library features.
 *
 * An appropriate context must be created by calling,
 *
 *     amd::cvml::CreateContext()
 *
 * before using any features in the Ryzen AI CVML library and provided
 * to the feature constructor(s).
 *
 * The context can be shared by multiple features of the SDK.
 */
class CVML_SDK_EXPORT_CORE Context {
  AMD_CVML_INTERFACE(Context);

 public:
  /**
   * Releases all resources for the context and destroys it.
   */
  virtual void Release() = 0;

  /**
   * Sets the verbosity of the log.
   *
   * @param level SDK feature log level
   */
  virtual void SetLogLevel(Logger::LogLevels level) = 0;

  /**
   * Gets the pointer to the logger object.
   *
   * @return Pointer to logger object
   */
  virtual Logger* GetLogger() const = 0;

  /**
   * Get the Supported Platform Information object.
   *
   * @param info Pointer to structure for receiving platform information
   * @return true on success, false on failure
   */
  static bool GetSupportedPlatformInformation(amd::cvml::SupportedPlatformInformation* info);

  /**
   * Defines the inference backends that can be supported by the SDK.
   *
   * These are provided to the \a SetInferenceBackend API function.
   */
  enum InferenceBackend {
    AUTO,  ///< Allow the SDK to select the hardware for inference operations
    GPU,   ///< Use GPU hardware for inference operations
    NPU,   ///< Use NPU hardware for inference operations
    CPU,   ///< Use CPU hardware for inference operations
    dGPU   ///< Use discrete GPU hardware, if available, for inference operations
  };

  /**
   * Defines the source streaming mode for feature processing.
   */
  enum StreamingMode {
    ONE_SHOT,          ///< Features should expect to process independent images.
    ONLINE_STREAMING,  ///< Input images are part of real-time streaming content.
    OFFLINE_STREAMING  ///< Features are intended to process offline streaming content.
  };

  /**
   * Specifies the inference backend for subsequently created features.
   *
   * This function does not affect any features that were instantiated
   * via the context before its call. If a feature is unable to support
   * a specified inference backend, it will refuse to construct and an
   * exception will be thrown instead.
   *
   * @param inference_backend Desired hardware inference backend
   * @return true if backend updated
   */
  bool SetInferenceBackend(InferenceBackend inference_backend);

  /**
   * Returns the inference backend selection strategy for newly created features.
   *
   * @return Current hardware inference backend selection
   */
  InferenceBackend GetInferenceBackend() const;

  /**
   * Returns the current streaming mode.
   *
   * See \a amd::cvml::Context::SetStreamingMode for more details.
   *
   * @return Currently configured streaming mode.
   */
  StreamingMode GetStreamingMode() const;

  /**
   * Set the streaming mode for the context.
   *
   * The requested streaming mode is used to configure new features
   * that are constructed against the context. Any features that
   * were created before are not affected by changing streaming
   * mode changes.
   *
   * See \a amd::cvml::Context::StreamingMode
   *
   * @param mode Desired streaming mode.
   */
  void SetStreamingMode(StreamingMode mode);

  /**
   * Get current CVML 'nice' setting.
   *
   * See \a amd::cvml::Context::SetNiceMode() for more details.
   *
   * @return true if 'nice' request is currently enabled
   */
  bool GetNiceMode();

  /**
   * Set current CVML 'nice' settings.
   *
   * If enabled, the CVML 'nice' mode directs underlying inference
   * engines to run in a lower scheduling priority or more power
   * efficient mode if possible. The setting is applied to all
   * CVML features that were instantiated with the current context
   * and may be changed at any time.
   *
   * This setting provides a hint to the underlying inference
   * execution framework but does not guarantee lower priority
   * execution or more power efficient inference. Applications may
   * enable this setting of the use case is tolerant to occasionally
   * longer inference latencies as a tradeoff for potentially reducing
   * power consumption.
   *
   * @param nice_mode Set to true to enable 'niceness' for subsequent features
   */
  void SetNiceMode(bool nice_mode);

  /**
   * Return if NPU is available on platform
   *
   * @return true if NPU available
   */
  static bool IsNPUAvailable();

  /**
   * Return if iGPU is available on platform
   *
   * @return true if iGPU available
   */
  static bool IsiGPUAvailable();

  /**
   * Return if dGPU is available on platform
   *
   * @return true if dGPU available
   */
  static bool IsdGPUAvailable();

  /**
   * Get detected NPU driver version.
   * On Linux, NPU driver will return 1 if legacy driver detected
   *
   * @return NPU driver version, or 0 if not detected
   */
  uint32_t GetNPUDriverVersion();

 public:
  class Impl;
  Impl* impl_;  ///< Pointer to context implementation
};

/**
 * Create a Ryzen AI context.
 *
 * @param log_level  Sets the log level. Default value is kINFO
 * @param logger External logger for the context. Default value is nullptr
 * @see \a amd::cvml::Logger
 * @return Pointer to the created Context
 */
CVML_SDK_EXPORT_CORE amd::cvml::Context* CreateContext(
    Logger::LogLevels log_level = Logger::LogLevels::kINFO, Logger* logger = nullptr);

}  // namespace cvml
}  // namespace amd

#endif  // EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_CONTEXT_H_
