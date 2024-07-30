# Copyright © 2021-2024 Advanced Micro Devices, Inc. All rights reserved.

find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
  message(STATUS "Using GTest from FetchContent")
  # While we can use an installed version of GTest and use find_package to get it,
  # GTest's official guide recommends this approach of building the library with
  # the same compile options as the executable being tested instead of linking
  # against a precompiled library.
  FetchContent_Declare(
    googletest GIT_REPOSITORY "https://github.com/google/googletest"
    GIT_TAG "v1.14.0"
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE INTERNAL "")

  FetchContent_MakeAvailable(googletest)

  # move all include directories to system directories
  list(APPEND gtest_targets gtest gtest_main gmock)
  foreach(target ${gtest_targets})
    get_target_property(INCLUDE_DIRS ${target} INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(
      ${target} PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                           "${INCLUDE_DIRS}"
    )
  endforeach()
endif()
