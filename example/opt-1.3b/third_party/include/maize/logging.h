// Copyright 2023 AMD, Inc
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or Implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file
 * @brief Defines logging
 */

#pragma once

#include <memory>  // for shared_ptr
#include <string>  // for string

#ifdef MAIZE_ENABLE_LOGGING

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO

#include <spdlog/spdlog.h>
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

namespace amd {
namespace maize {

class LoggerSingleton {
public:
  static LoggerSingleton& GetInstance() {
    static LoggerSingleton instance;
    return instance;
  }
  spdlog::logger* get() const { return logger_ptr_.get(); }
private:
  LoggerSingleton() {
    auto *env = std::getenv("MAIZE_LOG_MODE");
    if (env) {
      std::string envstr(env);
      if (envstr == "FILE") {
        logger_ptr_ = spdlog::basic_logger_mt("console", "maize-log.txt", true);
      }
      else {
        logger_ptr_ = spdlog::stdout_color_mt("console");
      }
    }
    else {
      logger_ptr_ = spdlog::stdout_color_mt("console");
    }
  }

  std::shared_ptr<spdlog::logger> logger_ptr_;
};

}  // namespace maize
}  // namespace amd

#define MAIZE_LOG_TRACE(message) \
  SPDLOG_LOGGER_TRACE(LoggerSingleton::GetInstance().get(), message)
#define MAIZE_LOG_DEBUG(message) \
  SPDLOG_LOGGER_DEBUG(LoggerSingleton::GetInstance().get(), message)
#define MAIZE_LOG_INFO(message) \
  SPDLOG_LOGGER_INFO(LoggerSingleton::GetInstance().get(), message)
#define MAIZE_LOG_WARN(message) \
  SPDLOG_LOGGER_WARN(LoggerSingleton::GetInstance().get(), message)
#define MAIZE_LOG_ERROR(message) \
  SPDLOG_LOGGER_ERROR(LoggerSingleton::GetInstance().get(), message)

#define MAIZE_IF_LOGGING(args) args
#else
#define MAIZE_LOG_TRACE(message)
#define MAIZE_LOG_DEBUG(message)
#define MAIZE_LOG_INFO(message)
#define MAIZE_LOG_WARN(message)
#define MAIZE_LOG_ERROR(message)

#define MAIZE_IF_LOGGING(args)
#endif
