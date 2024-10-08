###############################################################################
# Copyright (C) 2021 Xilinx, Inc.  All rights reserved.
# SPDX-License-Identifier: MIT
#
###############################################################################

collector_list  (_list PROJECT_INC_DIRS)
include_directories (${_list} ${CMAKE_CURRENT_SOURCE_DIR})

collector_list  (_list PROJECT_LIB_DIRS)
link_directories (${_list})

collector_list (_deps PROJECT_LIB_DEPS)

get_property (_ec_flags GLOBAL PROPERTY "PROJECT_EC_FLAGS")
message ("_ec_flags = ${_ec_flags}")

set (_apps profile-aie)
list(APPEND _apps profile-io-aie)
list(APPEND _apps trace-pc-aie)

foreach (_app ${_apps})
  set (_src ${CMAKE_CURRENT_SOURCE_DIR}/${_app}.cpp)
  if (WITH_SHARED_LIB)
    add_executable (${_app}-share ${_src})
    set_target_properties(${_app}-share PROPERTIES CXX_STANDARD 11)
    target_compile_options (${_app}-share PUBLIC ${_ec_flags})
    target_link_libraries (${_app}-share ${_ec_flags} ${_deps})
    install (TARGETS ${_app}-share RUNTIME DESTINATION bin)
  endif (WITH_SHARED_LIB)

  if (WITH_STATIC_LIB)
    if (${PROJECT_SYSTEM} STREQUAL "linux")
        add_executable (${_app}-static ${_src})
    set_target_properties(${_app}-static PROPERTIES CXX_STANDARD 11)
        target_compile_options (${_app}-static PUBLIC ${_ec_flags})
        target_link_libraries (${_app}-static ${_ec_flags} ${_deps})
        install (TARGETS ${_app}-static RUNTIME DESTINATION bin)
    endif (${PROJECT_SYSTEM} STREQUAL "linux")
  endif (WITH_STATIC_LIB)
endforeach (_app)
