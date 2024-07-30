get_filename_component(target ${CMAKE_CURRENT_SOURCE_DIR} NAME)

list(APPEND CMAKE_ARGS "-DProtobuf_DEBUG=ON")
list(APPEND CMAKE_ARGS "-DENABLE_CPU_RUNNER=ON")
list(APPEND CMAKE_ARGS "-DENABLE_DPU_RUNNER=ON")
list(APPEND CMAKE_ARGS "-DBUILD_PYTHON=ON")
list(APPEND CMAKE_ARGS "-DINSTALL_HOME=ON") # Determines where to install the python module

list(APPEND deps target_factory xir)

BuildExternalGitProject(
  ${target}
  ${${target}_URL}
  ${${target}_TAG}
  "${CMAKE_ARGS}"
  "${deps}"
  ${SKIP_DOWNLOAD}
  OFF
  )
