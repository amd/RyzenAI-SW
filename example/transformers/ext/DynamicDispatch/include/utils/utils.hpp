/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __UTILS_H_
#define __UTILS_H_

#include <cmath>
#include <fstream>
#include <map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define _DD_GET_PID() GetCurrentProcessId()
#else
#include <unistd.h>
#define _DD_GET_PID() getpid()
#endif

namespace Utils {

std::string get_env_var(const std::string &var,
                        const std::string &default_val = {});
int64_t ceil_for_me(int64_t x, int64_t y);
size_t get_size_of_type(const std::string &type);

// Align the 'n' to the multiple of 'A'
// new_n >= n
// new_n % A = 0
size_t align_to_next(size_t n, size_t alignment);

template <typename T>
static void write_buffer_to_file(T *buf, size_t buf_size, std::string fname) {
  std::ofstream ofs;
  ofs.open("./logs/" + fname);
  for (size_t i = 0; i < buf_size; i++) {
    ofs << std::to_string(buf[i]) << "\n";
  }
  ofs.close();
}

/// @brief Splits a string into tokens on delimiter.
/// eg: "A,B,C" --> ["A", "B", "C"]
std::vector<std::string> split_string(const std::string &msg,
                                      const std::string &delim = ",");

/// @brief Remove all whitespaces in a string
std::string remove_whitespaces(std::string x);

} // namespace Utils

#endif // __UTILS_H_
