# Copyright © 2022-2024 Advanced Micro Devices, Inc. All rights reserved.

# Parse the VERSION file in the current directory and extract it into variables.
# The VERSION file should be a file containing the semantic version of the
# project on a single line of the form x.y.z[-prerelease]
#
# This function sets the following variables in the parent scope:
#   - version_full: x.y.z-prerelease
#   - version_core: x.y.z
#   - version_major: x
#   - version_minor: y
#   - version_patch: z
#   - version_prerelease: prerelease
# If no prerelease is specified, it's empty.
#
# Example usage: dd_parse_version(${CMAKE_CURRENT_SOURCE_DIR}/VERSION)
function(dd_parse_version version_file)
  file(READ "${version_file}" ver)
  string(REPLACE "\n" "" ver ${ver})
  string(REGEX MATCHALL "([0-9]+)|-(.*)\\+" result ${ver})
  list(GET result 0 ver_major)
  list(GET result 1 ver_minor)
  list(GET result 2 ver_patch)
  list(LENGTH result result_len)
  if(result_len EQUAL "4")
    list(GET result 3 ver_prerelease)
  else()
    set(ver_prerelease "")
  endif()
  message(STATUS "Building version ${ver}")

  set(version_full ${ver} PARENT_SCOPE)
  set(version_core ${ver_major}.${ver_minor}.${ver_patch} PARENT_SCOPE)
  set(version_major ${ver_major} PARENT_SCOPE)
  set(version_minor ${ver_minor} PARENT_SCOPE)
  set(version_patch ${ver_patch} PARENT_SCOPE)
  set(version_prerelease ${ver_prerelease} PARENT_SCOPE)
endfunction()
