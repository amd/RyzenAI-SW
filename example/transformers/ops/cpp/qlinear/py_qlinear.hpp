/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
 */

#ifndef __PY_QLINEAR_H__
#define __PY_QLINEAR_H__

// pybind11 headers
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "qlinear.hpp"

namespace ryzenai {
template <typename InT, typename WtT, typename OutT>
class py_qlinear : private qlinear<InT, WtT, OutT> {
public:
  py_qlinear(const py::list &shared_libs, const py::list &x_shapes,
             const std::tuple<int, int> &y_shape, size_t nworkers, int num_dlls,
             int dll_switch_limit, bool pack_weights);
  py_qlinear(const py::list &shared_libs, const py::list &x_shapes,
             const std::tuple<int, int> &y_shape, size_t nworkers, int num_dlls,
             int dll_switch_limit, bool pack_weights,
             const std::string &log_file);
  py_qlinear(const std::string &shared_lib, const std::tuple<int, int> &x_shape,
             const std::tuple<int, int> &y_shape, size_t nworkers);
  py_qlinear(const std::string &shared_lib, const std::tuple<int, int> &x_shape,
             const std::tuple<int, int> &y_shape, size_t nworkers,
             const std::string &log_file);
  void py_execute(py::array_t<InT, py::array::c_style> &a,
                  py::array_t<OutT, py::array::c_style> &c);
  void
  py_initialize_weights_from_buffer(py::array_t<InT, py::array::c_style> &wts);
  void py_initialize_weights_from_json(const std::string &json);
};

template <typename InT, typename WtT, typename OutT>
py_qlinear<InT, WtT, OutT>::py_qlinear(const py::list &shared_libs,
                                       const py::list &x_shapes,
                                       const std::tuple<int, int> &y_shape,
                                       size_t nworkers, int num_dlls,
                                       int dll_switch_limit, bool pack_weights)
    : qlinear<InT, WtT, OutT>(
          shared_libs.cast<std::vector<std::string>>(),
          x_shapes.cast<std::vector<std::tuple<int, int>>>(), y_shape, nworkers,
          num_dlls, dll_switch_limit, pack_weights) {}

template <typename InT, typename WtT, typename OutT>
py_qlinear<InT, WtT, OutT>::py_qlinear(const py::list &shared_libs,
                                       const py::list &x_shapes,
                                       const std::tuple<int, int> &y_shape,
                                       size_t nworkers, int num_dlls,
                                       int dll_switch_limit, bool pack_weights,
                                       const std::string &log_file)
    : qlinear<InT, WtT, OutT>(
          shared_libs.cast<std::vector<std::string>>(),
          x_shapes.cast<std::vector<std::tuple<int, int>>>(), y_shape, nworkers,
          num_dlls, dll_switch_limit, pack_weights, log_file) {}

template <typename InT, typename WtT, typename OutT>
py_qlinear<InT, WtT, OutT>::py_qlinear(const std::string &shared_lib,
                                       const std::tuple<int, int> &x_shape,
                                       const std::tuple<int, int> &y_shape,
                                       size_t nworkers)
    : qlinear<InT, WtT, OutT>(shared_lib, x_shape, y_shape, nworkers) {}

template <typename InT, typename WtT, typename OutT>
py_qlinear<InT, WtT, OutT>::py_qlinear(const std::string &shared_lib,
                                       const std::tuple<int, int> &x_shape,
                                       const std::tuple<int, int> &y_shape,
                                       size_t nworkers,
                                       const std::string &log_file)
    : qlinear<InT, WtT, OutT>(shared_lib, x_shape, y_shape, nworkers,
                              log_file) {}

template <typename InT, typename WtT, typename OutT>
void py_qlinear<InT, WtT, OutT>::py_execute(
    py::array_t<InT, py::array::c_style> &a,
    py::array_t<OutT, py::array::c_style> &c) {
  std::tuple<int, int> a_shape = {a.shape(0), a.shape(1)};
  py::buffer_info a_buf = a.request();
  py::buffer_info c_buf = c.request();
  auto a_ptr = static_cast<InT *>(a_buf.ptr);
  auto c_ptr = static_cast<OutT *>(c_buf.ptr);

  execute(a_ptr, a_shape, c_ptr);
}

template <typename InT, typename WtT, typename OutT>
void py_qlinear<InT, WtT, OutT>::py_initialize_weights_from_buffer(
    py::array_t<InT, py::array::c_style> &wts) {
  std::tuple<int, int> wts_shape = {wts.shape(0), wts.shape(1)};
  py::buffer_info wts_buf = wts.request();
  auto wts_ptr = static_cast<WtT *>(wts_buf.ptr);
  initialize_weights(wts_ptr, wts_shape);
}

template <typename InT, typename WtT, typename OutT>
void py_qlinear<InT, WtT, OutT>::py_initialize_weights_from_json(
    const std::string &json) {
  initialize_weights(json);
}

} // namespace ryzenai

#endif /* __PY_QLINEAR_H__ */
