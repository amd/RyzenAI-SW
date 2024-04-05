/*
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_LOGGER_H_
#define EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_LOGGER_H_

#include <ctime>
#include <string>

#include "cvml-api-common.h"

using std::time_t;

namespace amd {
namespace cvml {

/**
 * Interface to handle logging in cvml sdk
 */
class CVML_SDK_EXPORT CvmlLogger {
  AMD_CVML_INTERFACE(CvmlLogger);

 public:
  /**
   * Log levels to set the log output verbosity.
   * Logger will print all the log messages if the log level is greater than
   * or equal to the level which is already set
   */
  enum class LogLevels {
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
   * @param level A valid value from CvmlLogger::LogLevels
   */
  void SetLogLevel(CvmlLogger::LogLevels level) { level_ = level; }

  /**
   * Get the log level
   *
   * @return level A valid value from CvmlLogger::LogLevels
   */
  CvmlLogger::LogLevels GetLogLevel() { return level_; }

  /**
   * Write an entry into the log with a std::string message as input
   *
   * @param log_level Type of the log message
   * @param msg Message of std::string type that needs to be logged
   */
  void Log(amd::cvml::CvmlLogger::LogLevels log_level, const std::string& msg);

  /**
   * Write an entry into the log with a C type string as input
   *
   * @param log_level Type of the log message
   * @param msg C type string message that needs to be logged
   */
  void Log(amd::cvml::CvmlLogger::LogLevels log_level, const char* msg);

  /**
   * Print the actual log message.Implemented by the respective child class
   *
   * @param msg C type string message that needs to be logged
   */
  virtual void LogStr(const char* msg) = 0;

 protected:
  CvmlLogger::LogLevels level_ = CvmlLogger::LogLevels::kINFO;
};

// \deprecated
// Below typedef is retained for backward compatibility
typedef CvmlLogger ICvmlLogger;

}  // namespace cvml
}  // namespace amd

#endif  // EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_LOGGER_H_
