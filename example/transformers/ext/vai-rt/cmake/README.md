# CMake Flow
This folder is intended for building Vitis-AI Runtime source repositories together in a single CMake Flow

# Dependencies
All VART dependencies except for XRT are built here.  
In a perfect world we should also build XRT.  
For now, XRT must be installed somewhere in the system, or copied into the PREFIX that this repo is creating.

# Build
1. Verify the desired URLs and Git Tags for the source repositories in [SourceVersions.cmake](cmake/SourceVersions.cmake)
2. Review the build options in [Options.cmake](cmake/Options.cmake)
3. Choose a directory `INSTALL_DIRECTORY` as to where you would like to install vai-rt libraries and dependencies
4. Decide whether you want to build vai-rt libraries are shared or static. Use `-DBUILD_SHARED_LIBS=ON` for shared libraries.
5. Install xrt into `INSTALL_DIRECTORY`
6. Run CMake with your desired options and `-DCMAKE_INSTALL_PREFIX=INSTALL_DIRECTORY` and `-DCMAKE_PREFIX_PATH=INSTALL_DIRECTORY`

See [build.sh](build.sh) for an example linux build script.  
See [build.bat](build.bat) for an example windows build script.

# Example Artifacts
```
VitisAI/artifacts  $ tree 
.
├── bin
│   ├── dump_op_weights
│   ├── dump_op_weights_2
│   ├── mem_read
│   ├── mem_save
│   ├── mem_write
│   ├── show_dpu
│   ├── test_xrt_device_handle
│   ├── vart_version
│   ├── xir
│   └── xrt_read_register
├── include
│   ├── vairt
│   │   └── vairt.hpp
│   ├── vart
│   │   ├── assistant
│   │   │   ├── batch_tensor_buffer.hpp
│   │   │   ├── tensor_buffer_allocator.hpp
│   │   │   └── xrt_bo_tensor_buffer.hpp
│   │   ├── dpu
│   │   │   └── vitis_dpu_runner_factory.hpp
│   │   ├── experimental
│   │   │   └── runner_helper.hpp
│   │   ├── mm
│   │   │   └── host_flat_tensor_buffer.hpp
│   │   ├── runner_ext.hpp
│   │   ├── runner.hpp
│   │   ├── tensor_buffer.hpp
│   │   ├── tensor_buffer_unowned_device.hpp
│   │   ├── tensor_mirror_attrs.hpp
│   │   ├── trace
│   │   │   ├── common.hpp
│   │   │   ├── event.hpp
│   │   │   ├── fmt.hpp
│   │   │   ├── payload.hpp
│   │   │   ├── ringbuf.hpp
│   │   │   ├── traceclass.hpp
│   │   │   ├── trace.hpp
│   │   │   └── vaitrace_dbg.hpp
│   │   ├── util_4bit.hpp
│   │   ├── util_export.hpp
│   │   ├── xir_helper.hpp
│   │   └── zero_copy_helper.hpp
│   ├── vitis
│   │   └── ai
│   │       ├── bounded_queue.hpp
│   │       ├── c++14.hpp
│   │       ├── collection_helper.hpp
│   │       ├── dim_calc.hpp
│   │       ├── dpu_runner.hpp
│   │       ├── env_config.hpp
│   │       ├── erl_msg_box.hpp
│   │       ├── library
│   │       │   └── tensor.hpp
│   │       ├── linked_list_queue.hpp
│   │       ├── lock.hpp
│   │       ├── nocopy_bounded_queue.hpp
│   │       ├── parse_value.hpp
│   │       ├── performance_test.hpp
│   │       ├── plugin.hpp
│   │       ├── profiling.hpp
│   │       ├── ring_queue.hpp
│   │       ├── runner.hpp
│   │       ├── shared_queue.hpp
│   │       ├── simple_config.hpp
│   │       ├── sorted_queue.hpp
│   │       ├── tensor_buffer.hpp
│   │       ├── tensor.hpp
│   │       ├── thread_pool.hpp
│   │       ├── time_measure.hpp
│   │       ├── tracelogging.hpp
│   │       ├── util_export.hpp
│   │       ├── variable_bit.hpp
│   │       ├── weak.hpp
│   │       ├── with_injection.hpp
│   │       └── xxd.hpp
│   └── xir
│       ├── attrs
│       │   ├── attr_def.hpp
│       │   ├── attr_expander.hpp
│       │   └── attrs.hpp
│       ├── buffer_object.hpp
│       ├── device_memory.hpp
│       ├── dpu_controller.hpp
│       ├── graph
│       │   ├── graph.hpp
│       │   ├── graph_template.hpp
│       │   └── subgraph.hpp
│       ├── op
│       │   ├── op_def.hpp
│       │   └── op.hpp
│       ├── sfm_controller.hpp
│       ├── tensor
│       │   └── tensor.hpp
│       ├── util
│       │   ├── any.hpp
│       │   ├── data_type.hpp
│       │   └── tool_function.hpp
│       ├── XirExport.hpp
│       └── xrt_device_handle.hpp
├── lib
│   ├── libtrace-logging.a
│   ├── libvairt.so
│   ├── libvart-async-runner.a
│   ├── libvart-buffer-object.a
│   ├── libvart-dpu-controller.a
│   ├── libvart-dpu-runner.a
│   ├── libvart-dummy-runner.a
│   ├── libvart-mem-manager.a
│   ├── libvart-runner.a
│   ├── libvart-runner-assistant.a
│   ├── libvart-trace.a
│   ├── libvart-util.a
│   ├── libvart-xrt-device-handle.a
│   └── libxir.a
└── share
    ├── cmake
    │   ├── vairt
    │   │   ├── vairt-config.cmake
    │   │   ├── vairt-config-version.cmake
    │   │   ├── vairt-targets.cmake
    │   │   └── vairt-targets-release.cmake
    │   ├── vart
    │   │   ├── buffer-object-targets.cmake
    │   │   ├── buffer-object-targets-release.cmake
    │   │   ├── dpu-controller-targets.cmake
    │   │   ├── dpu-controller-targets-release.cmake
    │   │   ├── dpu-runner-targets.cmake
    │   │   ├── dpu-runner-targets-release.cmake
    │   │   ├── dummy-runner-targets.cmake
    │   │   ├── dummy-runner-targets-release.cmake
    │   │   ├── mem-manager-targets.cmake
    │   │   ├── mem-manager-targets-release.cmake
    │   │   ├── runner-assistant-targets.cmake
    │   │   ├── runner-assistant-targets-release.cmake
    │   │   ├── runner-targets.cmake
    │   │   ├── runner-targets-release.cmake
    │   │   ├── trace-logging-targets.cmake
    │   │   ├── trace-logging-targets-release.cmake
    │   │   ├── trace-targets.cmake
    │   │   ├── trace-targets-release.cmake
    │   │   ├── util-targets.cmake
    │   │   ├── util-targets-release.cmake
    │   │   ├── vart-config.cmake
    │   │   ├── vart-config-version.cmake
    │   │   ├── xrt-device-handle-targets.cmake
    │   │   └── xrt-device-handle-targets-release.cmake
    │   └── xir
    │       ├── xir-config.cmake
    │       ├── xir-config-version.cmake
    │       ├── xir-targets.cmake
    │       └── xir-targets-release.cmake
    └── vart
        └── reg.conf

25 directories, 129 files
```
