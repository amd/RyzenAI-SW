/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __UTILS_H_
#define __UTILS_H_

#include "xrt/xrt_bo.h"

static constexpr int BUFFER_ALIGNMENT = 128;

namespace Utils {

template <typename T> struct AlignedBuffer {
  AlignedBuffer() = default;
  AlignedBuffer(size_t nelems, size_t byte_alignment = alignof(T)) {
    if (!(((byte_alignment & (byte_alignment - 1)) == 0) &&
          (byte_alignment >= alignof(T)))) {
      throw std::runtime_error(
          "Alignment should be power of 2 and higher than default "
          "alignment of the type !!!");
    }
    vec_ = std::vector<T>(nelems + (byte_alignment / sizeof(T)) - 1, 0);
    auto ptr = reinterpret_cast<uint64_t>(vec_.data());
    size_t aligned_ptr =
        ((ptr + byte_alignment - 1) / byte_alignment) * byte_alignment;
    aligned_data_ = reinterpret_cast<T *>(aligned_ptr);
  }

  T *data() { return aligned_data_; }

private:
  std::vector<T> vec_;
  T *aligned_data_{nullptr};
};

template <size_t N> struct alignas(N) AlignedBO_ {
  // A tag to identify xrt::bo from host buffers at maize level
  static constexpr uint32_t TAG = 0xBC15ADB0; // Buffer Content IS AligneD BO;
  uint32_t tag = TAG;
  xrt::bo bo;
  bool need_sync = true;
  AlignedBO_<N> *data() { return this; }
};

using AlignedBO = AlignedBO_<BUFFER_ALIGNMENT>;

struct AlignedBOGroup {
  std::vector<Utils::AlignedBO> a_tile_bo;
  std::vector<Utils::AlignedBO> b_tile_bo;
  std::vector<Utils::AlignedBO> c_tile_bo;
};

struct AlignedBOCache {
private:
  AlignedBOCache() = default;

public:
  static std::pair<AlignedBOGroup *, bool>
  getInstance(const std::string &dllname) {
    static std::map<std::string, AlignedBOGroup> tile_cache;
    auto [iter, newly_created] = tile_cache.emplace(dllname, AlignedBOGroup{});
    return std::make_pair(&(iter->second), newly_created);
  }
};

static std::string get_env_var(const std::string &var,
                               const std::string &default_val = {}) {
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

static std::string get_xclbin_path(void) {
  auto dev = Utils::get_env_var("DEVICE");
  if (dev == "phx") {
    auto target = Utils::get_env_var("TARGET_DESIGN");
    if (target == "ASR4x2") {
      return Utils::get_env_var("PYTORCH_AIE_PATH") +
             std::string("/xclbin/phx/aieml_gemm_asr.xclbin");
    } else {
      return Utils::get_env_var("PYTORCH_AIE_PATH") +
             std::string("/xclbin/phx/aieml_gemm_vm_phx_4x4.xclbin");
    }
  } else {
    return Utils::get_env_var("PYTORCH_AIE_PATH") +
           std::string("/xclbin/stx/aie2p_gemm_vm_strix_4x4.xclbin");
  }
}

static std::string get_overlay_props(void) {
  auto dev = Utils::get_env_var("DEVICE");
  if (dev == "phx") {
    auto target = Utils::get_env_var("TARGET_DESIGN");
    if (target == "ASR4x2") {
      return Utils::get_env_var("THIRD_PARTY") +
             std::string("/.tvm/aie/"
                         "aieml_gemm_asr.json");
    } else {
      return Utils::get_env_var("THIRD_PARTY") +
             std::string("/.tvm/aie/"
                         "aieml_gemm_vm_phx_4x4.json");
    }
  } else {
    return Utils::get_env_var("THIRD_PARTY") +
           std::string("/.tvm/aie/"
                       "aie2p-gemm-vm-strix-4x4.json");
  }
}

static constexpr size_t qlinear_MAX_NTHREADS = 2;

static size_t get_qlinear_num_threads() {
  static const size_t qlinear_nthreads = std::stoi(
      get_env_var("qlinear_NUM_THREADS", std::to_string(qlinear_MAX_NTHREADS)));
  return qlinear_nthreads;
}

template <typename T>
static inline void _copy_tile(const T *src, T *dest, int64_t *tile_shape,
                              int64_t *src_strides) {
  for (int i = 0; i < tile_shape[0]; i++) {
    memcpy((void *)&dest[i * tile_shape[1]], (void *)&src[i * src_strides[0]],
           tile_shape[1] * sizeof(T));
  }
}

template <typename T>
static inline void _copy_tile(const T *src, Utils::AlignedBO *dst,
                              int64_t *tile_shape, int64_t *src_strides) {
  T *dest = dst->bo.map<T *>();
  _copy_tile(src, dest, tile_shape, src_strides);
  dst->need_sync = true;
}

template <typename T>
static inline void _copy_pad_data(T *src, T *dest, int64_t *src_shape,
                                  int64_t *dest_shape) {
  for (int i = 0; i < src_shape[0]; i++) {
    memcpy((void *)&dest[i * dest_shape[1]], (void *)&src[i * src_shape[1]],
           sizeof(T) * src_shape[1]);
  }
}

template <typename T>
static inline void _copy_depad_data(T *src, T *dest, int64_t *src_shape,
                                    int64_t *dest_shape) {
  for (int i = 0; i < dest_shape[0]; i++) {
    memcpy((void *)&dest[i * dest_shape[1]], (void *)&src[i * src_shape[1]],
           sizeof(T) * dest_shape[1]);
  }
}

static inline int64_t ceil_for_me(int64_t x, int64_t y) {
  return int64_t(y * std::ceil(x * 1.0 / y));
}

template <typename T> static inline T abs_max(const T *arr, int size) {
  T max = abs(arr[0]);
  for (int i = 1; i < size; i++) {
    if (abs(arr[i]) > max) {
      max = abs(arr[i]);
    }
  }
  return max;
}

static std::string get_dll_path() {
  return Utils::get_env_var("PYTORCH_AIE_PATH") + "/dll/" +
         Utils::get_env_var("DEVICE") + "/qlinear/";
}

template <typename T>
void write_buffer_to_file(T *buf, size_t buf_size, std::string fname) {
  std::ofstream ofs;
  ofs.open("./logs/" + fname);
  for (auto i = 0; i < buf_size; i++) {
    ofs << std::to_string(buf[i]) << "\n";
  }
  ofs.close();
}

} // namespace Utils

#endif // __UTILS_H_
