/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <algorithm>
#include <unordered_map>

#include <utils/utils.hpp>

namespace Utils {
std::string get_env_var(const std::string &var,
                        const std::string &default_val) {
#ifdef _WIN32
  char *value = nullptr;
  size_t size = 0;
  errno_t err = _dupenv_s(&value, &size, var.c_str());
  std::string result =
      (!err && (value != nullptr)) ? std::string{value} : default_val;
  free(value);
#else
  const char *value = std::getenv(var.c_str());
  std::string result = (value != nullptr) ? std::string{value} : default_val;
#endif
  return result;
}

int64_t ceil_for_me(int64_t x, int64_t y) {
  return int64_t(y * std::ceil(x * 1.0 / y));
}

// Align the 'n' to a multiple of 'A'
// new_n >= n
// new_n % A = 0
size_t align_to_next(size_t n, size_t alignment) {
  return ((n + alignment - 1) / alignment) * alignment;
}

size_t get_size_of_type(const std::string &type) {
  static const std::unordered_map<std::string, size_t> elem_size{
      {"int8", 1},     {"uint8", 1},   {"int16", 2}, {"uint16", 2},
      {"int32", 4},    {"uint32", 4},  {"int64", 8}, {"uint64", 8},
      {"bfloat16", 2}, {"float32", 4}, {"float", 4}, {"double", 8},
      {"float64", 8}};
  if (elem_size.find(type) == elem_size.end()) {
    throw std::runtime_error("get_size_of_type - Invalid type : " + type);
  }
  auto sz = elem_size.at(type);
  return sz;
}

std::vector<std::string> split_string(const std::string &msg,
                                      const std::string &delim) {
  auto start = 0;
  auto end = std::string::npos;
  std::vector<std::string> tokens;
  while (true) {
    end = msg.find(delim, start);
    if (end == std::string::npos) {
      auto sub = msg.substr(start, end - start);
      tokens.push_back(sub);
      break;
    }
    auto sub = msg.substr(start, end - start);
    tokens.push_back(sub);
    start = end + delim.size();
  }
  return tokens;
}

std::string remove_whitespaces(std::string x) {
  auto iter = std::remove(x.begin(), x.end(), ' ');
  x.erase(iter, x.end());
  return x;
}

} // namespace Utils
