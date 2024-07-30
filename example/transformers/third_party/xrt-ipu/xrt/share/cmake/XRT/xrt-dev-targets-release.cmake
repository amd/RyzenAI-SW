#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

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

# Import target "XRT::xrt_core" for configuration "Release"
set_property(TARGET XRT::xrt_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XRT::xrt_core PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/xrt/lib/xrt_core.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "XRT::xrt_coreutil"
  )

list(APPEND _IMPORT_CHECK_TARGETS XRT::xrt_core )
list(APPEND _IMPORT_CHECK_FILES_FOR_XRT::xrt_core "${_IMPORT_PREFIX}/xrt/lib/xrt_core.lib" )

# Import target "XRT::xrt_core_static" for configuration "Release"
set_property(TARGET XRT::xrt_core_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XRT::xrt_core_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/xrt/lib/xrt_core_static.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS XRT::xrt_core_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_XRT::xrt_core_static "${_IMPORT_PREFIX}/xrt/lib/xrt_core_static.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
