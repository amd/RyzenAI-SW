#pragma once

#include "detail/tfunc_impl.hpp"
#include <fstream>
#include <utils/logging.hpp>

namespace OpsFusion {

template <typename... Args>
static std::string dod_format(const std::string &msg, Args &&...args) {
  return detail::dod_format_impl(msg, std::forward<Args>(args)...);
}

// A RAII-Wrapper to mark begin & end of life of anything.
struct LifeTracer {
  LifeTracer(std::string msg) : msg_(std::move(msg)) {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format("{} ... START", msg_));
  }
  ~LifeTracer() {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format("{} ... END", msg_));
  }

private:
  std::string msg_;
};

// Open a binary file and throw if unsuccessful
static std::ifstream open_bin_file(const std::string &file_name) {
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Opening file: {} ...", file_name));
  std::ifstream ifs(file_name, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error(OpsFusion::dod_format(
        "Couldn't open file for reading : {}", file_name));
  }
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Opening file: {} ... DONE", file_name));
  return ifs;
}

// Read binary file to a vector
template <typename T = char>
static std::vector<T> read_bin_file(const std::string &filename) {
  std::ifstream ifs = open_bin_file(filename);
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Loading data from {}...", filename));
  std::vector<T> dst;
  try {
    ifs.seekg(0, ifs.end);
    auto size = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    dst.resize(size / sizeof(T));
    ifs.read((char *)dst.data(), size);
  } catch (std::exception &e) {
    throw std::runtime_error(OpsFusion::dod_format(
        "Failed to read contents from file {}, error: {}", filename, e.what()));
  }
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Loading data from {} ... DONE", filename));
  return dst;
}

template <typename srcT, typename Func>
auto for_each(const std::vector<srcT> &src, Func &&f) {
  using dstT = decltype(f(srcT{}));
  std::vector<dstT> res;
  res.reserve(src.size());
  for (const auto &item : src) {
    res.push_back(f(item));
  }
  return res;
}

template <typename Func, typename... Args>
static auto dd_invoke_impl(const std::string &func_name, const char *srcfile,
                           size_t line_no, Func &&f, Args &&...args) {
  LifeTracer lt(dod_format("Invoking {}", func_name));

  try {
    return f(std::forward<Args>(args)...);
  } catch (std::exception &e) {
    throw std::runtime_error(
        dod_format("[{}:{}] Invoking {}() failed with error: {}", srcfile,
                   line_no, func_name, e.what()));
  } catch (...) {
    throw std::runtime_error(
        dod_format("[{}:{}] Invoking {}() failed with Unknown Exception",
                   srcfile, line_no, func_name));
  }
}

} // namespace OpsFusion

// Following helper macros can be used to access elements from containers
// If access throws exception, it prints more details with the exception for
// debugging

// Equivalent to .at() method of std::vector/std::array
#define ARRAY_AT(x, idx)                                                       \
  OpsFusion::detail::vector_get_value_at(x, idx, #x, __FILE__, __LINE__)

// Equivalent to index access of a new/malloc buffer
#define PTR_AT(ptr, sz, idx)                                                   \
  OpsFusion::detail::ptr_get_at(ptr, sz, idx, #ptr, __FILE__, __LINE__)

// Equivalent to .at() method of std::map/std::unordered_map
#define MAP_AT(x, key)                                                         \
  OpsFusion::detail::map_get_value_at(x, key, #x, __FILE__, __LINE__)

// Throw with source location
#define DOD_THROW(msg) OpsFusion::detail::throw_loc(msg, __FILE__, __LINE__)

// Throw is condition fails
#define DOD_ASSERT(cond, msg)                                                  \
  if (!(cond)) {                                                               \
    DOD_THROW(msg);                                                            \
  }

// Throw is condition fails
#define DOD_THROW_IF(cond, msg)                                                \
  if ((cond)) {                                                                \
    DOD_THROW(msg);                                                            \
  }

// Invoke an external function with exception check
// Eg : auto res = DD_INVOKE(add_func, 2, 3);
#define DD_INVOKE(func, ...) dd_invoke_impl(#func, &func, __VA_ARGS__)

// Invoke an external class member func with exception check
// Eg : auto res = DD_INVOKE_MEMFN(classA::methodB, objA, args1, args2)
#define DD_INVOKE_MEMFN(func, ...)                                             \
  dd_invoke_impl(#func, std::mem_fn(&func), __VA_ARGS__)
