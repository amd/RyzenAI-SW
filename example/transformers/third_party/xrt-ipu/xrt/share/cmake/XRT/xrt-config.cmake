# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2019-2021 Xilinx, Inc. All rights reserved.

# --------------------
# xrt-config.cmake
# --------------------
#
# XRT cmake module.
# This module sets the following variables in your project::
#
#   XRT_FOUND              - true if XRT and all required components found on the system
#   XRT_VERSION            - VERSION of this package in x.y.z format
#   XRT_CMAKE_DIR          - Directory where this cmake module was found
#   XRT_INCLUDE_DIRS       - Directory where XRT headers are located.
#   XRT_LINK_DIRS          - Directory where XRT link libraries are located.
#   XRT_CORE_LIBRARIES     - libraries to link against.
#   XRT_COREUTIL_LIBRARIES - libraries to link against.
#   XRT_OPENCL_LIBRARIES   - libraries to link against.
#   XRT_SWEMU_LIBRARIES    - libraries to link against.
#   XRT_HWEMU_LIBRARIES    - libraries to link against.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was xrt.fp.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(XRT_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/xrt/include")
set(XRT_LINK_DIRS "${PACKAGE_PREFIX_DIR}/xrt/lib")

set(XRT_VERSION 2.14.0)

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/xrt-build-tree-targets.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/xrt-build-tree-targets.cmake")
else()
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/xrt-targets.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/xrt-targets.cmake")
  endif()
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/xrt-dev-targets.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/xrt-dev-targets.cmake")
  endif()
endif()

get_filename_component(XRT_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

set(XRT_CORE_LIBRARIES XRT::xrt_core XRT::xrt_coreutil)
set(XRT_COREUTIL_LIBRARIES XRT::xrt_coreutil)
set(XRT_OPENCL_LIBRARIES XRT::xilinxopencl XRT::xrt++)
set(XRT_SWEMU_LIBRARIES XRT::xrt_swemu)
set(XRT_HWEMU_LIBRARIES XRT::xrt_hwemu)

set(XRT_FOUND True)

message(STATUS "Found XRT: ${XRT_CMAKE_DIR} (found version \"${XRT_VERSION}\")")
