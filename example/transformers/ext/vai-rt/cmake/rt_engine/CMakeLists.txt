get_filename_component(target ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if(NOT MSVC)
  list(APPEND CMAKE_ARGS "-DXRT_DIR=/opt/xilinx/xrt/share/cmake/XRT")
endif()

list(APPEND deps vart)

BuildExternalGitProject(
  ${target}
  ${${target}_URL}
  ${${target}_TAG}
  "${CMAKE_ARGS}"
  "${deps}"
  ${SKIP_DOWNLOAD}
  OFF
  )