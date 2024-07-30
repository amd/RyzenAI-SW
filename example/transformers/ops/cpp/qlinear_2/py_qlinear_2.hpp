/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __PY_QLINEAR_2_H__
#define __PY_QLINEAR_2_H__

// nanobind headers
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <optional>
namespace nb = nanobind;

#include "qlinear_2.hpp"

namespace ryzenai {
template <typename InT, typename WtT, typename AccT, typename OutT = AccT>
class py_qlinear_2 : private qlinear_2<InT, WtT, AccT, OutT> {
public:
  py_qlinear_2(const std::string &a_dtype, const std::string &b_dtype,
               const std::string &c_dtype);
  void py_execute(nb::ndarray<InT, nb::c_contig> &a,
                  nb::ndarray<OutT, nb::c_contig> &c);
  void py_initialize_weights(nb::ndarray<WtT, nb::c_contig> &wts,
                             std::optional<int> group_size);
  void py_qlinear_2<InT, WtT, AccT, OutT>::py_initialize_weights_int4(
      nb::ndarray<WtT, nb::c_contig> &wts,
      nb::ndarray<WtT, nb::c_contig> &zeros,
      nb::ndarray<AccT, nb::c_contig> &scales,
      nb::ndarray<AccT, nb::c_contig> &bias, int group_size);
  void py_qlinear_2<InT, WtT, AccT, OutT>::py_initialize_weights_int4_mladf(
      nb::ndarray<WtT, nb::c_contig> &wts,
      nb::ndarray<WtT, nb::c_contig> &zeros,
      nb::ndarray<float, nb::c_contig> &scales,
      nb::ndarray<float, nb::c_contig> &bias, int group_size);
  void py_qlinear_2<InT, WtT, AccT, OutT>::py_debug(bool enable);
};

template <typename InT, typename WtT, typename AccT, typename OutT>
py_qlinear_2<InT, WtT, AccT, OutT>::py_qlinear_2(const std::string &a_dtype,
                                                 const std::string &b_dtype,
                                                 const std::string &c_dtype)
    : qlinear_2<InT, WtT, AccT, OutT>(a_dtype, b_dtype, c_dtype) {}

template <typename InT, typename WtT, typename AccT, typename OutT>
void py_qlinear_2<InT, WtT, AccT, OutT>::py_initialize_weights(
    nb::ndarray<WtT, nb::c_contig> &wts, std::optional<int> group_size) {
  std::tuple<int, int> wts_shape = {wts.shape(0), wts.shape(1)};
  auto wts_ptr = static_cast<WtT *>(wts.data());
  if (!group_size.has_value()) {
    initialize_weights(wts_ptr, wts_shape);
  } else {
    initialize_weights(wts_ptr, wts_shape, group_size.value());
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void py_qlinear_2<InT, WtT, AccT, OutT>::py_initialize_weights_int4(
    nb::ndarray<WtT, nb::c_contig> &wts, nb::ndarray<WtT, nb::c_contig> &zeros,
    nb::ndarray<AccT, nb::c_contig> &scales,
    nb::ndarray<AccT, nb::c_contig> &bias, int group_size) {

  std::tuple<int, int> wts_shape = {wts.shape(0), wts.shape(1)};
  auto wts_ptr = static_cast<WtT *>(wts.data());
  auto zeros_ptr = static_cast<WtT *>(zeros.data());
  auto scales_ptr = static_cast<AccT *>(scales.data());
  auto bias_ptr = static_cast<AccT *>(bias.data());

  initialize_weights_int4(wts_ptr, zeros_ptr, scales_ptr, bias_ptr, wts_shape,
                          group_size);
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void py_qlinear_2<InT, WtT, AccT, OutT>::py_initialize_weights_int4_mladf(
    nb::ndarray<WtT, nb::c_contig> &wts, nb::ndarray<WtT, nb::c_contig> &zeros,
    nb::ndarray<float, nb::c_contig> &scales,
    nb::ndarray<float, nb::c_contig> &bias, int group_size) {

  std::tuple<int, int> wts_shape = {wts.shape(0), wts.shape(1)};
  auto wts_ptr = static_cast<WtT *>(wts.data());
  auto zeros_ptr = static_cast<WtT *>(zeros.data());
  auto scales_ptr = static_cast<float *>(scales.data());
  auto bias_ptr = static_cast<float *>(bias.data());

  initialize_weights_int4_mladf(wts_ptr, zeros_ptr, scales_ptr, bias_ptr,
                                wts_shape, group_size);
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void py_qlinear_2<InT, WtT, AccT, OutT>::py_debug(bool enable) {
  debug(enable);
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void py_qlinear_2<InT, WtT, AccT, OutT>::py_execute(
    nb::ndarray<InT, nb::c_contig> &a, nb::ndarray<OutT, nb::c_contig> &c) {
  std::tuple<int, int> a_shape = {a.shape(0), a.shape(1)};
  auto a_ptr = static_cast<InT *>(a.data());
  auto c_ptr = static_cast<OutT *>(c.data());
  execute(a_ptr, a_shape, c_ptr);
}

} // namespace ryzenai

#endif /* __PY_QLINEAR_2_H__ */
