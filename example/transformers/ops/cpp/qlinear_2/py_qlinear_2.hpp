/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __PY_QLINEAR_2_H__
#define __PY_QLINEAR_2_H__

// pybind11 headers
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "qlinear_2.hpp"

namespace ryzenai {
template <typename InT, typename WtT, typename OutT>
class py_qlinear_2 : private qlinear_2<InT, WtT, OutT> {
public:
  py_qlinear_2(const std::string &a_dtype, const std::string &b_dtype,
               const std::string &c_dtype);
  void py_execute(py::array_t<InT, py::array::c_style> &a,
                  py::array_t<OutT, py::array::c_style> &c);
  void py_initialize_weights(py::array_t<WtT, py::array::c_style> &wts,
                             std::optional<int> group_size);
  void py_qlinear_2<InT, WtT, OutT>::py_initialize_weights_int4(
      py::array_t<WtT, py::array::c_style> &wts,
      py::array_t<WtT, py::array::c_style> &zeros,
      py::array_t<OutT, py::array::c_style> &scales,
      py::array_t<OutT, py::array::c_style> &bias, int group_size);
  void py_qlinear_2<InT, WtT, OutT>::py_debug(bool enable);
};

template <typename InT, typename WtT, typename OutT>
py_qlinear_2<InT, WtT, OutT>::py_qlinear_2(const std::string &a_dtype,
                                           const std::string &b_dtype,
                                           const std::string &c_dtype)
    : qlinear_2<InT, WtT, OutT>(a_dtype, b_dtype, c_dtype) {}

template <typename InT, typename WtT, typename OutT>
void py_qlinear_2<InT, WtT, OutT>::py_initialize_weights(
    py::array_t<WtT, py::array::c_style> &wts, std::optional<int> group_size) {
  std::tuple<int, int> wts_shape = {wts.shape(0), wts.shape(1)};
  py::buffer_info wts_buf = wts.request();
  auto wts_ptr = static_cast<WtT *>(wts_buf.ptr);
  if (!group_size.has_value()) {
    initialize_weights(wts_ptr, wts_shape);
  } else {
    initialize_weights(wts_ptr, wts_shape, group_size.value());
  }
}

template <typename InT, typename WtT, typename OutT>
void py_qlinear_2<InT, WtT, OutT>::py_initialize_weights_int4(
    py::array_t<WtT, py::array::c_style> &wts,
    py::array_t<WtT, py::array::c_style> &zeros,
    py::array_t<OutT, py::array::c_style> &scales,
    py::array_t<OutT, py::array::c_style> &bias, int group_size) {

  std::tuple<int, int> wts_shape = {wts.shape(0), wts.shape(1)};
  py::buffer_info wts_buf = wts.request();
  py::buffer_info zeros_buf = zeros.request();
  py::buffer_info scales_buf = scales.request();
  py::buffer_info bias_buf = bias.request();
  auto wts_ptr = static_cast<WtT *>(wts_buf.ptr);
  auto zeros_ptr = static_cast<WtT *>(zeros_buf.ptr);
  auto scales_ptr = static_cast<OutT *>(scales_buf.ptr);
  auto bias_ptr = static_cast<OutT *>(bias_buf.ptr);

  initialize_weights_int4(wts_ptr, zeros_ptr, scales_ptr, bias_ptr, wts_shape,
                     group_size);
}

template <typename InT, typename WtT, typename OutT>
void py_qlinear_2<InT, WtT, OutT>::py_debug(bool enable) {
  debug(enable);
}

template <typename InT, typename WtT, typename OutT>
void py_qlinear_2<InT, WtT, OutT>::py_execute(
    py::array_t<InT, py::array::c_style> &a,
    py::array_t<OutT, py::array::c_style> &c) {
  std::tuple<int, int> a_shape = {a.shape(0), a.shape(1)};
  py::buffer_info a_buf = a.request();
  py::buffer_info c_buf = c.request();
  auto a_ptr = static_cast<InT *>(a_buf.ptr);
  auto c_ptr = static_cast<OutT *>(c_buf.ptr);

  execute(a_ptr, a_shape, c_ptr);
}

} // namespace ryzenai

#endif /* __PY_QLINEAR_2_H__ */
