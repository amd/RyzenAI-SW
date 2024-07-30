/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "../qlinear_2/py_qlinear_2.hpp"
#include "../utils/stats.hpp"

namespace nb = nanobind;

NB_MODULE(_RyzenAI, m) {
  nb::class_<ryzenai::py_qlinear_2<int8_t, int8_t, int32_t>>(
      m, "qlinear_2_a8w8acc32")
      .def(nb::init<const std::string &, const std::string &,
                    const std::string &>())
      .def("execute",
           &ryzenai::py_qlinear_2<int8_t, int8_t, int32_t>::py_execute,
           "Function to execute matmul on aie with cpu tiling")
      .def("debug", &ryzenai::py_qlinear_2<int8_t, int8_t, int32_t>::py_debug,
           "Function to enable debug flag that writes matrices to files")
      .def("initialize_weights",
           &ryzenai::py_qlinear_2<int8_t, int8_t,
                                  int32_t>::py_initialize_weights,
           "Register weights from numpy array", nb::arg("wts"),
           nb::arg("group_size") = nb::none());

  nb::class_<ryzenai::py_qlinear_2<int16_t, int8_t, int64_t>>(
      m, "qlinear_2_a16w8acc64")
      .def(nb::init<const std::string &, const std::string &,
                    const std::string &>())
      .def("execute",
           &ryzenai::py_qlinear_2<int16_t, int8_t, int64_t>::py_execute,
           "Function to execute matmul on aie with cpu tiling")
      .def("debug", &ryzenai::py_qlinear_2<int16_t, int8_t, int64_t>::py_debug,
           "Function to enable debug flag that writes matrices to files")
      .def("initialize_weights",
           &ryzenai::py_qlinear_2<int16_t, int8_t,
                                  int64_t>::py_initialize_weights,
           "Register weights from numpy array", nb::arg("wts"),
           nb::arg("group_size") = nb::none());

  nb::class_<ryzenai::py_qlinear_2<int16_t, int8_t, float>>(
      m, "qlinear_2_a16fw4acc32f")
      .def(nb::init<const std::string &, const std::string &,
                    const std::string &>())
      .def("execute",
           &ryzenai::py_qlinear_2<int16_t, int8_t, float>::py_execute,
           "Function to execute matmul on aie with cpu tiling")
      .def("debug", &ryzenai::py_qlinear_2<int16_t, int8_t, float>::py_debug,
           "Function to enable debug flag that writes matrices to files")
      .def("initialize_weights",
           &ryzenai::py_qlinear_2<int16_t, int8_t,
                                  float>::py_initialize_weights_int4,
           "Register weights from numpy array");

  nb::class_<ryzenai::py_qlinear_2<int16_t, int8_t, float, int16_t>>(
      m, "qlinear_2_a16fw4acc32fo16f")
      .def(nb::init<const std::string &, const std::string &,
                    const std::string &>())
      .def("execute",
           &ryzenai::py_qlinear_2<int16_t, int8_t, float, int16_t>::py_execute,
           "Function to execute matmul on aie with cpu tiling")
      .def("debug",
           &ryzenai::py_qlinear_2<int16_t, int8_t, float, int16_t>::py_debug,
           "Function to enable debug flag that writes matrices to files")
      .def("initialize_weights",
           &ryzenai::py_qlinear_2<int16_t, int8_t, float,
                                  int16_t>::py_initialize_weights_int4,
           "Register weights from numpy array");

  nb::class_<ryzenai::py_qlinear_2<int16_t, int8_t, int16_t>>(
      m, "qlinear_2_a16fw4acc16f")
      .def(nb::init<const std::string &, const std::string &,
                    const std::string &>())
      .def("execute",
           &ryzenai::py_qlinear_2<int16_t, int8_t, int16_t>::py_execute,
           "Function to execute matmul on aie with cpu tiling")
      .def("debug", &ryzenai::py_qlinear_2<int16_t, int8_t, int16_t>::py_debug,
           "Function to enable debug flag that writes matrices to files")
      .def("initialize_weights",
           &ryzenai::py_qlinear_2<int16_t, int8_t,
                                  int16_t>::py_initialize_weights_int4_mladf,
           "Register weights from numpy array");

  nb::class_<ryzenai::stats::MemInfo>(m, "MemInfo")
      .def_rw("commit_memory", &ryzenai::stats::MemInfo::commit_memory);

  nb::class_<ryzenai::stats::ProcTimeInfo>(m, "ProcTimeInfo")
      .def_rw("sys_kernel_time", &ryzenai::stats::ProcTimeInfo::sys_kernel_time)
      .def_rw("sys_user_time", &ryzenai::stats::ProcTimeInfo::sys_user_time)
      .def_rw("proc_kernel_time",
              &ryzenai::stats::ProcTimeInfo::proc_kernel_time)
      .def_rw("proc_user_time", &ryzenai::stats::ProcTimeInfo::proc_user_time);

  m.def("get_sys_commit_mem", &ryzenai::stats::get_sys_commit_mem,
        "Get current system commit memory in bytes");

  nb::class_<ryzenai::stats::CPULoad>(m, "CPULoad")
      .def(nb::init<int>())
      .def("get_cpu_load", &ryzenai::stats::CPULoad::get_cpu_load,
           "Get the avg. cpu load from the last time this method was invoked.")
      .def("get_pid", &ryzenai::stats::CPULoad::get_pid,
           "Get the process ID of the given CPULoad object")
      .def("__getstate__",
           [](const ryzenai::stats::CPULoad &a) {
             return std::make_tuple(a.get_pid());
           })
      .def("__setstate__",
           [](ryzenai::stats::CPULoad &a, const std::tuple<int> &state) {
             new (&a) ryzenai::stats::CPULoad(std::get<0>(state));
           });
}
