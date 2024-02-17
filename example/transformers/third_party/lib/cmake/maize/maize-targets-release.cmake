#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "maize::maize" for configuration "Release"
set_property(TARGET maize::maize APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(maize::maize PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/maize.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "XRT::xrt_coreutil"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/maize.dll"
  )

list(APPEND _cmake_import_check_targets maize::maize )
list(APPEND _cmake_import_check_files_for_maize::maize "${_IMPORT_PREFIX}/lib/maize.lib" "${_IMPORT_PREFIX}/bin/maize.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
