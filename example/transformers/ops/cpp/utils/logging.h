/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __LOGGING_H__
#define __LOGGING_H__

#include <chrono>

// spdlog headers
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

// #define RYZENAI_LOGGING
#ifdef RYZENAI_LOGGING

namespace ryzenai {
static const std::string logger_name = "ryzenai_logger";
static const std::string log_file = "logs/ryzenai_ops.log";

class logger {
public:
  static logger &get_instance() {
    static logger logger_;
    return logger_;
  }

  spdlog::logger *get() const { return logger_ptr_.get(); }
  int64_t get_elapsed() { return sw.elapsed() / std::chrono::nanoseconds(1); }

private:
  std::shared_ptr<spdlog::logger> logger_ptr_;
  spdlog::stopwatch sw;
  logger() {
    logger_ptr_ = spdlog::create<spdlog::sinks::basic_file_sink_mt>(
        logger_name, log_file, true);
    logger_ptr_->set_formatter(
        std::unique_ptr<spdlog::formatter>(new spdlog::pattern_formatter(
            "%v", spdlog::pattern_time_type::local, "")));
  }
};
} /* namespace ryzenai */

#define GET_ELAPSED_TIME_NS() (ryzenai::logger::get_instance().get_elapsed())
#define RYZENAI_LOG_INFO(message)                                              \
  SPDLOG_LOGGER_INFO(ryzenai::logger::get_instance().get(), message)
#else
#define GET_ELAPSED_TIME_NS() 0
#define RYZENAI_LOG_INFO(message)
#endif /* RYZENAI_LOGGING */

#endif /* __LOGGING_H__ */