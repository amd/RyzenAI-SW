#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "XRT::xilinxopencl" for configuration "Release"
set_property(TARGET XRT::xilinxopencl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XRT::xilinxopencl PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/xrt/lib/xilinxopencl.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "XRT::xrt++;XRT::xrt_coreutil"
  )

list(APPEND _IMPORT_CHECK_TARGETS XRT::xilinxopencl )
list(APPEND _IMPORT_CHECK_FILES_FOR_XRT::xilinxopencl "${_IMPORT_PREFIX}/xrt/lib/xilinxopencl.lib" )

# Import target "XRT::xilinxopencl_static" for configuration "Release"
set_property(TARGET XRT::xilinxopencl_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XRT::xilinxopencl_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/xrt/lib/xilinxopencl_static.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS XRT::xilinxopencl_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_XRT::xilinxopencl_static "${_IMPORT_PREFIX}/xrt/lib/xilinxopencl_static.lib" )

# Import target "XRT::xrt++" for configuration "Release"
set_property(TARGET XRT::xrt++ APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XRT::xrt++ PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/xrt/lib/xrt++.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "XRT::xrt_coreutil"
  )

list(APPEND _IMPORT_CHECK_TARGETS XRT::xrt++ )
list(APPEND _IMPORT_CHECK_FILES_FOR_XRT::xrt++ "${_IMPORT_PREFIX}/xrt/lib/xrt++.lib" )

# Import target "XRT::xrt++_static" for configuration "Release"
set_property(TARGET XRT::xrt++_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XRT::xrt++_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/xrt/lib/xrt++_static.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS XRT::xrt++_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_XRT::xrt++_static "${_IMPORT_PREFIX}/xrt/lib/xrt++_static.lib" )

# Import target "XRT::xrt_coreutil" for configuration "Release"
set_property(TARGET XRT::xrt_coreutil APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XRT::xrt_coreutil PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/xrt/lib/xrt_coreutil.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS XRT::xrt_coreutil )
list(APPEND _IMPORT_CHECK_FILES_FOR_XRT::xrt_coreutil "${_IMPORT_PREFIX}/xrt/lib/xrt_coreutil.lib" )

# Import target "XRT::xrt_coreutil_static" for configuration "Release"
set_property(TARGET XRT::xrt_coreutil_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XRT::xrt_coreutil_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/xrt/lib/xrt_coreutil_static.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS XRT::xrt_coreutil_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_XRT::xrt_coreutil_static "${_IMPORT_PREFIX}/xrt/lib/xrt_coreutil_static.lib" )

# Import target "XRT::xrt_phxcore" for configuration "Release"
set_property(TARGET XRT::xrt_phxcore APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XRT::xrt_phxcore PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/xrt/lib/xrt_phxcore.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "XRT::xrt_coreutil"
  )

list(APPEND _IMPORT_CHECK_TARGETS XRT::xrt_phxcore )
list(APPEND _IMPORT_CHECK_FILES_FOR_XRT::xrt_phxcore "${_IMPORT_PREFIX}/xrt/lib/xrt_phxcore.lib" )

# Import target "XRT::xrt_phxcore_static" for configuration "Release"
set_property(TARGET XRT::xrt_phxcore_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XRT::xrt_phxcore_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/xrt/lib/xrt_phxcore_static.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS XRT::xrt_phxcore_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_XRT::xrt_phxcore_static "${_IMPORT_PREFIX}/xrt/lib/xrt_phxcore_static.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
