get_filename_component(target ${CMAKE_CURRENT_SOURCE_DIR} NAME)

list(APPEND deps vart googletest vairt)

if(BUILD_SINGLE_LIBRARY)
  list(APPEND deps vairt)
elseif(BUILD_RT_ENGINE)
  list(APPEND deps rt_engine)
elseif(BUILD_GRAPH_ENGINE)
  list(APPEND deps graph_engine)
elseif(BUILD_TVM_ENGINE)
  list(APPEND deps tvm_engine)
endif()

BuildExternalGitProject(
  ${target}
  ${${target}_URL}
  ${${target}_TAG}
  "${CMAKE_ARGS}"
  "${deps}"
  ${SKIP_DOWNLOAD}
  OFF
  )
