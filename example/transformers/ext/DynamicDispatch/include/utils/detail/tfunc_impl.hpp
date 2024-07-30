#pragma once

#include <array>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace OpsFusion {
namespace detail {

static std::string combine_file_line(const std::string &file, size_t line) {
  return "[" + file + ":" + std::to_string(line) + "]";
}

template <typename T>
static T &ptr_get_at(T *arr, size_t sz, size_t idx, const char *name,
                     const char *file, size_t line) {

  if (idx >= sz) {
    std::ostringstream oss;
    oss << file << ":" << line << " [ERROR] array out-of-bound access"
        << "\n"
        << "Details - name: " << name << ", size: " << sz << ", idx: " << idx
        << std::endl;
    throw std::runtime_error(oss.str());
  }
  return arr[idx];
}

template <typename T>
static T &vector_get_value_at(std::vector<T> &vec, size_t idx, const char *name,
                              const char *file, size_t line) {
  return ptr_get_at(vec.data(), vec.size(), idx, name, file, line);
}

template <typename T>
static const T &vector_get_value_at(const std::vector<T> &vec, size_t idx,
                                    const char *name, const char *file,
                                    size_t line) {
  return ptr_get_at(vec.data(), vec.size(), idx, name, file, line);
}

template <typename T, size_t N>
static T &vector_get_value_at(std::array<T, N> &vec, size_t idx,
                              const char *name, const char *file, size_t line) {
  return ptr_get_at(vec.data(), N, idx, name, file, line);
}

template <typename T, size_t N>
static const T &vector_get_value_at(const std::array<T, N> &vec, size_t idx,
                                    const char *name, const char *file,
                                    size_t line) {
  return ptr_get_at(vec.data(), N, idx, name, file, line);
}

template <typename T, size_t N>
static T &vector_get_value_at(T (&vec)[N], size_t idx, const char *name,
                              const char *file, size_t line) {
  return ptr_get_at(&(vec[0]), N, idx, name, file, line);
}

template <typename T, size_t N>
static const T &vector_get_value_at(const T (&vec)[N], size_t idx,
                                    const char *name, const char *file,
                                    size_t line) {
  return ptr_get_at(&(vec[0]), N, idx, name, file, line);
}

template <typename T>
static T &c_array_get_value_at(T vec[], size_t N, size_t idx, const char *name,
                               const char *file, size_t line) {
  return ptr_get_at(&vec[0], N, idx, name, file, line);
}

template <typename T>
static const T &c_array_get_value_at(const T vec[], size_t N, size_t idx,
                                     const char *name, const char *file,
                                     size_t line) {
  return ptr_get_at(&vec[0], N, idx, name, file, line);
}

template <typename Container, typename K = typename Container::key_type,
          typename V = typename Container::mapped_type>
static V &map_get_value_at(Container &container, const K &key, const char *name,
                           const char *file, size_t line) {
  auto iter = container.find(key);
  if (iter == container.end()) {
    std::ostringstream oss;
    oss << combine_file_line(file, line) << " [ERROR] Invalid Key Access "
        << "(Container: " << name << ", Key: " << key
        << ", Size: " << container.size() << ")\n";
    throw std::runtime_error(oss.str());
  }
  return iter->second;
}

template <typename Container, typename K = typename Container::key_type,
          typename V = typename Container::mapped_type>
static const V &map_get_value_at(const Container &container, const K &key,
                                 const char *name, const char *file,
                                 size_t line) {
  auto iter = container.find(key);
  if (iter == container.end()) {
    std::ostringstream oss;
    oss << combine_file_line(file, line) << " [ERROR] Invalid Key Access "
        << "(Name: " << name << ", Key: " << key
        << ", Size: " << container.size() << ")\n";
    throw std::runtime_error(oss.str());
  }
  return iter->second;
}

static std::string cvt_to_string(const std::string &str) { return str; }
static std::string cvt_to_string(const char *str) { return str; }
template <typename T> static std::string cvt_to_string(T num) {
  std::ostringstream oss;
  oss << num;
  return oss.str();
}

template <typename... Args>
static std::string dod_format_impl(const std::string &msg, Args &&...args) {
  constexpr size_t sz = sizeof...(args);
  if constexpr (sz == 0) {
    return msg;
  } else {
    static_assert(sz > 0, "Size > 0");
    std::string sargs[] = {cvt_to_string(args)...};

    auto start = 0;
    auto end = std::string::npos;
    std::vector<std::string> tokens;
    while (true) {
      end = msg.find("{}", start);
      if (end == std::string::npos) {
        auto sub = msg.substr(start, end - start);
        tokens.push_back(sub);
        break;
      }
      auto sub = msg.substr(start, end - start);
      tokens.push_back(sub);
      tokens.push_back("{}");
      start = end + 2;
    }

    std::string res;
    size_t arg_idx = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (tokens[i] == "{}") {
        auto sub = arg_idx < sz ? sargs[arg_idx] : tokens[i];
        res += sub;
        arg_idx++;
      } else {
        res += tokens[i];
      }
    }
    return res;
  }
}

static void throw_loc(const std::string &msg, const char *srcfile,
                      size_t line_no) {
  throw std::runtime_error(
      dod_format_impl("[{}:{}] [ERROR] {}", srcfile, line_no, msg));
}

} // namespace detail
} // namespace OpsFusion
