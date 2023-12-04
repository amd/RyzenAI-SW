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
  py_qlinear_2(const std::tuple<int, int> &kernel_x_shape,
               const std::tuple<int, int> &kernel_y_shape);
  void py_execute(py::array_t<InT, py::array::c_style> &a,
                  py::array_t<OutT, py::array::c_style> &c);
  void py_initialize_weights(py::array_t<InT, py::array::c_style> &wts);
  void py_qlinear_2<InT, WtT, OutT>::py_debug(bool enable);
};

template <typename InT, typename WtT, typename OutT>
py_qlinear_2<InT, WtT, OutT>::py_qlinear_2(
    const std::tuple<int, int> &kernel_x_shape,
    const std::tuple<int, int> &kernel_y_shape)
    : qlinear_2<InT, WtT, OutT>(kernel_x_shape, kernel_y_shape) {}

template <typename InT, typename WtT, typename OutT>
void py_qlinear_2<InT, WtT, OutT>::py_initialize_weights(
    py::array_t<InT, py::array::c_style> &wts) {
  std::tuple<int, int> wts_shape = {wts.shape(0), wts.shape(1)};
  py::buffer_info wts_buf = wts.request();
  auto wts_ptr = static_cast<WtT *>(wts_buf.ptr);
  initialize_weights(wts_ptr, wts_shape);
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
