/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __LOGGING_H__
#define __LOGGING_H__

#include <chrono>

// #define RYZENAI_PERF  // enable macro to log perf metrics
// #define RYZENAI_TRACE // enable macro to log trace points

#if defined(RYZENAI_PERF) || defined(RYZENAI_TRACE)
#define RYZENAI_LOGGING
#endif

#ifdef RYZENAI_LOGGING

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

// spdlog headers
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

namespace ryzenai {
static const std::string perf_logger_name = "ryzenai_perf_logger";
static const std::string perf_logger_fname = "logs/ryzenai_ops.log";
static const std::string trace_logger_name = "ryzenai_trace_logger";
static const std::string trace_logger_fname = "logs/ryzenai_trace.log";

class logger {
public:
  enum log_levels {
    PERF,
    TRACE,
  };
  static logger &get_instance() {
    static logger logger_;
    return logger_;
  }

  spdlog::logger *get(log_levels lvl) const {
    if (lvl == PERF) {
      return perf_logger_.get();
    } else if (lvl == TRACE) {
      return trace_logger_.get();
    } else {
      throw std::runtime_error("Invalid logger option");
    }
  }
  int64_t get_elapsed() { return sw.elapsed() / std::chrono::nanoseconds(1); }

private:
  std::shared_ptr<spdlog::logger> perf_logger_;
  std::shared_ptr<spdlog::logger> trace_logger_;
  spdlog::stopwatch sw;
  logger() {
    perf_logger_ = spdlog::create<spdlog::sinks::basic_file_sink_mt>(
        perf_logger_name, perf_logger_fname, true);
    perf_logger_->set_formatter(
        std::unique_ptr<spdlog::formatter>(new spdlog::pattern_formatter(
            "%v", spdlog::pattern_time_type::local, "")));

    trace_logger_ = spdlog::create<spdlog::sinks::basic_file_sink_mt>(
        trace_logger_name, trace_logger_fname, true);
    trace_logger_->set_level(spdlog::level::trace);
  }
};
} /* namespace ryzenai */

#ifdef RYZENAI_PERF
#define GET_ELAPSED_TIME_NS() (ryzenai::logger::get_instance().get_elapsed())
#define RYZENAI_LOG_INFO(message)                                              \
  SPDLOG_LOGGER_INFO(                                                          \
      ryzenai::logger::get_instance().get(logger::log_levels::PERF), message)
#endif /* RYZENAI_PERF */

#ifdef RYZENAI_TRACE
#define RYZENAI_LOG_TRACE(message)                                             \
  SPDLOG_LOGGER_TRACE(                                                         \
      ryzenai::logger::get_instance().get(logger::log_levels::TRACE), message)
#endif /*RYZENAI_TRACE */

#endif /* RYZENAI_LOGGING */

#ifndef RYZENAI_PERF
#define GET_ELAPSED_TIME_NS() 0
#define RYZENAI_LOG_INFO(message)
#endif

#ifndef RYZENAI_TRACE
#define RYZENAI_LOG_TRACE(message)
#endif

#endif /* __LOGGING_H__ */
