###############################################################################
# Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################

cmake_minimum_required(VERSION 3.16.3)
project(xaiengine LANGUAGES C VERSION 3.5)

set(CMAKE_C_STANDARD 11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

# Define a custom option to control whether the library is built as a shared or static library
option(XAIENGINE_BUILD_SHARED "Build xaiengine as a shared library" ON)

file(
  GLOB_RECURSE
  SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/*/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/*/*/*.c"
  )

file(
  GLOB_RECURSE
  HDRS
  "${CMAKE_CURRENT_SOURCE_DIR}/*/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*/*/*.h"
  )

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/xaiengine.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include/)

file(COPY ${HDRS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME})

if(XAIENGINE_BUILD_SHARED)
  # Build the library as a shared library
  add_library(
    ${PROJECT_NAME}
    SHARED
    ${SRCS}
  )
else()
  # Build the library as a static library
  add_library(
    ${PROJECT_NAME}
    STATIC
    ${SRCS}
  )
endif()

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  )
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
  VERSION
  ${PROJECT_VERSION}
  SOVERSION
  ${PROJECT_VERSION_MAJOR}
  )

if (SOCKET_BACKEND)
	list(APPEND IO_BACKEND -D__AIESOCKET__)
endif (SOCKET_BACKEND)
if (DEBUG_BACKEND)
	list(APPEND IO_BACKEND -D__AIEDEBUG__)
endif (DEBUG_BACKEND)

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE
   /wd4018 /wd4024 /wd4047 /wd4061 /wd4100 /wd4206 /wd4242 /wd4244  /wd4267 /wd4388 /wd4431 /wd4668 /wd4701 /wd4710 /wd4711 /wd4716
   /wd4820 /wd5045
   )
  target_compile_options(${PROJECT_NAME} PRIVATE /Wall -DXAIE_FEATURE_MSVC ${IO_BACKEND})
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra ${IO_BACKEND})
endif()

include(GNUInstallDirs)

install(
  FILES
  ${HDRS}
  DESTINATION
  ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
  )

install(
  FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/xaiengine.h
  DESTINATION
  ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  TARGETS
  ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(
  EXPORT ${PROJECT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT runtime
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

# Support Find Package
include(CMakePackageConfigHelpers)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake @ONLY
  )

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion
  )

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
        COMPONENT runtime DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
