#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ucode_gen::ucode_gen" for configuration "Release"
set_property(TARGET ucode_gen::ucode_gen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ucode_gen::ucode_gen PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/ucode_gen.lib"
  )

list(APPEND _cmake_import_check_targets ucode_gen::ucode_gen )
list(APPEND _cmake_import_check_files_for_ucode_gen::ucode_gen "${_IMPORT_PREFIX}/lib/ucode_gen.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
