#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tvm_ipu::tvm" for configuration "Release"
set_property(TARGET tvm_ipu::tvm APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tvm_ipu::tvm PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/tvm_ipu.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "maize::maize;XRT::xrt_coreutil"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/tvm_ipu.dll"
  )

list(APPEND _cmake_import_check_targets tvm_ipu::tvm )
list(APPEND _cmake_import_check_files_for_tvm_ipu::tvm "${_IMPORT_PREFIX}/lib/tvm_ipu.lib" "${_IMPORT_PREFIX}/lib/tvm_ipu.dll" )

# Import target "tvm_ipu::tvm_runtime" for configuration "Release"
set_property(TARGET tvm_ipu::tvm_runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tvm_ipu::tvm_runtime PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/tvm_ipu_runtime.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "maize::maize;XRT::xrt_coreutil"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/tvm_ipu_runtime.dll"
  )

list(APPEND _cmake_import_check_targets tvm_ipu::tvm_runtime )
list(APPEND _cmake_import_check_files_for_tvm_ipu::tvm_runtime "${_IMPORT_PREFIX}/lib/tvm_ipu_runtime.lib" "${_IMPORT_PREFIX}/lib/tvm_ipu_runtime.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
