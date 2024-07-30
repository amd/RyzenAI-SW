get_filename_component(target ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if(ENABLE_IPU)
  list(APPEND CMAKE_ARGS "-DENABLE_IPU=ON")
endif()

list(APPEND CMAKE_ARGS "-DProtobuf_DEBUG=ON")

list(APPEND deps boost protobuf unilog)

BuildExternalGitProject(
  ${target}
  ${${target}_URL}
  ${${target}_TAG}
  "${CMAKE_ARGS}"
  "${deps}"
  ${SKIP_DOWNLOAD}
  OFF
  )
