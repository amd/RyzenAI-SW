/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../qlinear/py_qlinear.hpp"
#include "../qlinear_2/py_qlinear_2.hpp"
#include "../utils/stats.hpp"

namespace py = pybind11;

PYBIND11_MODULE(RyzenAI, m) {
  py::class_<ryzenai::py_qlinear<int8_t, int8_t, int32_t>>(m, "qlinear")
      .def(py::init<const std::string &, const std::tuple<int, int> &,
                    const std::tuple<int, int> &, /*nworkers*/ size_t>())
      .def(py::init<const std::string &, const std::tuple<int, int> &,
                    const std::tuple<int, int> &, /*nworkers*/ size_t,
                    const std::string &>())
      .def(py::init<const py::list &, const py::list &,
                    const std::tuple<int, int> &,
                    /*nworkers*/ size_t, /*num_dlls*/ int,
                    /*dll_switch_limit*/ int, /*pack_weights*/ bool>())
      .def(py::init<const py::list &, const py::list &,
                    const std::tuple<int, int> &,
                    /*nworkers*/ size_t, /*num_dlls*/ int,
                    /*dll_switch_limit*/ int, /*pack_weights*/ bool,
                    const std::string &>())
      .def("execute", &ryzenai::py_qlinear<int8_t, int8_t, int32_t>::py_execute,
           "Function to execute matmul on aie with cpu tiling")
      .def("initialize_weights",
           &ryzenai::py_qlinear<int8_t, int8_t,
                                int32_t>::py_initialize_weights_from_buffer,
           "Register weights from numpy array")
      .def("initialize_weights",
           &ryzenai::py_qlinear<int8_t, int8_t,
                                int32_t>::py_initialize_weights_from_json,
           "Register weights weights from JSON file");

  py::class_<ryzenai::py_qlinear_2<int8_t, int8_t, int32_t>>(
      m, "qlinear_2_a8w8acc32")
      .def(py::init<const std::string &, std::string &, std::string &>())
      .def("execute",
           &ryzenai::py_qlinear_2<int8_t, int8_t, int32_t>::py_execute,
           "Function to execute matmul on aie with cpu tiling")
      .def("debug", &ryzenai::py_qlinear_2<int8_t, int8_t, int32_t>::py_debug,
           "Function to enable debug flag that writes matricies to files")
      .def("initialize_weights",
           &ryzenai::py_qlinear_2<int8_t, int8_t,
                                  int32_t>::py_initialize_weights,
           "Register weights from numpy array", py::arg("wts"),
           py::arg("group_size") = py::none());

  py::class_<ryzenai::py_qlinear_2<int16_t, int8_t, int64_t>>(
      m, "qlinear_2_a16w8acc64")
      .def(py::init<const std::string &, std::string &, std::string &>())
      .def("execute",
           &ryzenai::py_qlinear_2<int16_t, int8_t, int64_t>::py_execute,
           "Function to execute matmul on aie with cpu tiling")
      .def("debug", &ryzenai::py_qlinear_2<int16_t, int8_t, int64_t>::py_debug,
           "Function to enable debug flag that writes matricies to files")
      .def("initialize_weights",
           &ryzenai::py_qlinear_2<int16_t, int8_t,
                                  int64_t>::py_initialize_weights,
           "Register weights from numpy array", py::arg("wts"),
           py::arg("group_size") = py::none());

  py::class_<ryzenai::py_qlinear_2<int16_t, int8_t, float>>(
      m, "qlinear_2_a16fw4acc32f")
      .def(py::init<const std::string &, std::string &, std::string &>())
      .def("execute",
           &ryzenai::py_qlinear_2<int16_t, int8_t, float>::py_execute,
           "Function to execute matmul on aie with cpu tiling")
      .def("debug", &ryzenai::py_qlinear_2<int16_t, int8_t, float>::py_debug,
           "Function to enable debug flag that writes matricies to files")
      .def("initialize_weights",
           &ryzenai::py_qlinear_2<int16_t, int8_t,
                                  float>::py_initialize_weights_int4,
           "Register weights from numpy array");

  py::class_<ryzenai::stats::MemInfo>(m, "MemInfo")
      .def_readwrite("commit_memory", &ryzenai::stats::MemInfo::commit_memory);

  py::class_<ryzenai::stats::ProcTimeInfo>(m, "ProcTimeInfo")
      .def_readwrite("sys_kernel_time",
                     &ryzenai::stats::ProcTimeInfo::sys_kernel_time)
      .def_readwrite("sys_user_time",
                     &ryzenai::stats::ProcTimeInfo::sys_user_time)
      .def_readwrite("proc_kernel_time",
                     &ryzenai::stats::ProcTimeInfo::proc_kernel_time)
      .def_readwrite("proc_user_time",
                     &ryzenai::stats::ProcTimeInfo::proc_user_time);

  m.def("get_sys_commit_mem", &ryzenai::stats::get_sys_commit_mem,
        "Get current system commit memory in bytes");

  py::class_<ryzenai::stats::CPULoad>(m, "CPULoad")
      .def(py::init<int>())
      .def("get_cpu_load", &ryzenai::stats::CPULoad::get_cpu_load,
           "Get the avg. cpu load from the last time this method was invoked.")
      .def("get_pid", &ryzenai::stats::CPULoad::get_pid,
           "Get the process ID of the given CPULoad object")
      .def(py::pickle(
          /* save */ [](const ryzenai::stats::CPULoad
                            &a) { return py::make_tuple(a.get_pid()); },
          /* load */
          [](py::tuple t) {
            return ryzenai::stats::CPULoad{t[0].cast<int>()};
          }));
}
