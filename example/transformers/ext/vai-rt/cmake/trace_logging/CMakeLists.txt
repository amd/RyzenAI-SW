get_filename_component(target ${CMAKE_CURRENT_SOURCE_DIR} NAME)

list(APPEND deps "")

BuildExternalGitProject(
  ${target}
  ${${target}_URL}
  ${${target}_TAG}
  "${CMAKE_ARGS}"
  "${deps}"
  ${SKIP_DOWNLOAD}
  OFF
  )
