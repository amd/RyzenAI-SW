# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
set(ZLIB_USE_STATIC_LIBS ON)
find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
  message(STATUS "Using ZLIB from FetchContent")
  set(ZLIB_BUILD_EXAMPLES OFF CACHE INTERNAL "")
  FetchContent_Declare(
    ZLIB GIT_REPOSITORY "https://github.com/madler/zlib.git" GIT_TAG v1.3.1
  )
  FetchContent_MakeAvailable(ZLIB)
  if(NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
  endif()
  target_include_directories(
    zlibstatic PUBLIC ${zlib_BINARY_DIR} ${zlib_SOURCE_DIR}
  ) # weird bug
  set(ZLIB_FOUND TRUE)
endif()
