# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#

# get absolute library path
get_filename_component(PREFIX ${RyzenAILibrary_DIR}/.. ABSOLUTE)

# set default PLATFORM to windows if not specified
if (NOT DEFINED PLATFORM)
if(WIN32)
    set(PLATFORM windows)
else()
    set(PLATFORM linux)
endif()
endif()

# update include directories
set(RyzenAILibrary_INCLUDE_DIRS ${PREFIX}/include)
include_directories(${RyzenAILibrary_INCLUDE_DIRS})

# collect all available library files
link_directories(${PREFIX}/${PLATFORM})
if (${PLATFORM} MATCHES "windows")
    file(GLOB RyzenAILibrary_LIBS ${PREFIX}/${PLATFORM}/*.lib)
else()
    file(GLOB RyzenAILibrary_LIBS ${PREFIX}/${PLATFORM}/*.so)
    list(FILTER RyzenAILibrary_LIBS EXCLUDE REGEX ".*graphlib.so")
endif()

# generate output variables for find_package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RyzenAILibrary DEFAULT_MSG RyzenAILibrary_LIBS RyzenAILibrary_INCLUDE_DIRS)
