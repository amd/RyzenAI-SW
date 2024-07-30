/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */
#pragma once

#ifdef UNIT_TEST_PERF
// Enable average latency collection/display
#include <chrono>
// Number of iterations
#define NITER 100
// Use this to profile any function call
#define PROFILE_THIS(function_call)                                            \
  auto start = std::chrono::steady_clock::now();                               \
  for (auto i = 0; i < NITER; ++i)                                             \
    function_call;                                                             \
  auto end = std::chrono::steady_clock::now();                                 \
  double elapsed =                                                             \
      std::chrono::duration<double, std::milli>(end - start).count();          \
  std::cout << "\33[1;36m[----------] Average latency (ms): "                  \
            << (elapsed / NITER) << "\33[0m" << std::endl;

#endif // UNIT_TEST_PERF

#define LOG_THIS(message_string)                                               \
  std::cout << "\33[1;36m[----------] " << message_string << "\33[0m"          \
            << std::endl;
