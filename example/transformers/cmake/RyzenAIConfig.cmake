# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

set(RyzenAI_FOUND True)
include(CMakeFindDependencyMacro)
find_dependency(XRT REQUIRED)
find_dependency(spdlog REQUIRED)
find_dependency(Eigen3 REQUIRED)
find_dependency(aie_controller REQUIRED)
include("${CMAKE_CURRENT_LIST_DIR}/RyzenaiConfigTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/RyzenAIConfigVersion.cmake")
get_filename_component(RyzenAI_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
message(
  STATUS "Found RyzenAI: ${RyzenAI_CMAKE_DIR} Version: ${PACKAGE_VERSION}"
)

# XRT library target is unaware of its associated include files
# Get the existing interface include directories
get_target_property(
  include_dirs ryzenai::qlinear_2 INTERFACE_INCLUDE_DIRECTORIES
)

# Append the new include directory
list(APPEND include_dirs "${XRT_INCLUDE_DIRS}")

# Set the modified include directories back on the target
set_target_properties(
  ryzenai::qlinear_2 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${include_dirs}"
)
