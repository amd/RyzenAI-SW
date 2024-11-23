/*
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_LOGGER_H_
#define EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_LOGGER_H_

#include <string>

#include "cvml-api-common.h"

namespace amd {
namespace cvml {

/**
 * Base class for capturing log messages from the SDK.
 *
 * To customize the target of log messages from the SDK, create a new C++
 * class derived from amd::cvml::Logger and implement its \a LogStr member
 * function to direct formatted log messages to the target of choice. For
 * example, a derived Logger class may choose to capture all log messages
 * to a file on the file system or send them to another process or device.
 */
class CVML_SDK_EXPORT Logger {
  AMD_CVML_INTERFACE(Logger);

 public:
  /**
   * Log levels to set the log output verbosity.
   * Logger will print all the log messages if the log level is greater than
   * or equal to the level which is already set.
   */
  enum LogLevels {
    kVERBOSE = 0,   ///< To print all types of log messages
    kDEBUG = 1,     ///< To print debug type messages and the levels above kDEBUG
    kINFO = 2,      ///< To print information type messages and the levels above kINFO
    kWARNING = 3,   ///< To print warning messages and the levels above kWARNING
    kERROR = 4,     ///< To print error messages only
    kDISABLED = 5,  ///< To disable logging
  };

 public:
  /**
   * Set the required log level
   *
   * @param level A valid value from Logger::LogLevels
   */
  void SetLogLevel(Logger::LogLevels level) { level_ = level; }

  /**
   * Get the log level
   *
   * @return level A valid value from Logger::LogLevels
   */
  Logger::LogLevels GetLogLevel() { return level_; }

  /**
   * Write an entry into the log with a std::string message as input
   *
   * @param log_level Type of the log message
   * @param msg Message of std::string type that needs to be logged
   */
  void Log(amd::cvml::Logger::LogLevels log_level, const std::string& msg);

  /**
   * Write an entry into the log with a C type string as input
   *
   * @param log_level Type of the log message
   * @param msg C type string message that needs to be logged
   */
  void Log(amd::cvml::Logger::LogLevels log_level, const char* msg);

  /**
   * Output the actual log message.
   *
   * This capability is must be implemented by a derived class.
   *
   * @param msg C type string message to be logged
   */
  virtual void LogStr(const char* msg) = 0;

 protected:
  /// Currently configured log level
  Logger::LogLevels level_ = Logger::LogLevels::kINFO;
};

// \deprecated
// This definition is retained for backward compatibility only.
using ICvmlLogger = Logger;

// \deprecated
// This definition is retained for backward compatibility only.
using CvmlLogger = Logger;

}  // namespace cvml
}  // namespace amd

#endif  // EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_LOGGER_H_
