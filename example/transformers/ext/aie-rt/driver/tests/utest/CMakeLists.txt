###############################################################################
# Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################

if (CPPUTEST_DIR)
  find_path(CPPUTEST_INC_DIR  NAMES "CppUTest/Utest.h" PATHS ${CPPUTEST_DIR}/include)
  find_library(CPPUTEST_FOUND_LIB NAMES CppUTest PATHS ${CPPUTEST_DIR}/src/CppUTest)
  if (NOT CPPUTEST_FOUND_LIB)
    message (FATAL_ERROR "No CppUtest Library is found in ${CPPUTEST_DIR}")
  endif (NOT CPPUTEST_FOUND_LIB)
  get_filename_component(CPPUTEST_LIB_DIR ${CPPUTEST_FOUND_LIB} DIRECTORY)
  message ("CPPUTEST_INC_DIR=${CPPUTEST_INC_DIR}")
else()
  message (FATAL_ERROR "No CppUtest directory is found in ${CPPUTEST_DIR}")
endif (CPPUTEST_DIR)

file(COPY elf_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

collector_list (_list PROJECT_INCLUDE_DIRS)
list (APPEND _list ${CPPUTEST_INC_DIR})
list (APPEND _list ${CMAKE_CURRENT_SOURCE_DIR}/common)
list (APPEND _list ${CMAKE_BINARY_DIR}/driver-src/include)
include_directories (${_list})

collector_list (_list PROJECT_LIB_DIRS)
list (APPEND _list ${CPPUTEST_LIB_DIR})
list (APPEND _list ${CMAKE_BINARY_DIR}/driver-src})
link_directories (${_list})

collector_list (_deps PROJECT_LIB_DEPS)
list (APPEND _deps "CppUTest")
list (APPEND _deps "xaiengine")

file(GLOB _sources *.cpp)
set (EXEPREX "run-test")
set (EXETEST "${EXEPREX}-aie")
add_executable (${EXETEST} ${_sources})
set_target_properties(${EXETEST} PROPERTIES CXX_STANDARD 11)
#if(TEST_HARDWARE)
#  set (_test_cflag -DTEST_HARDWARE)
#endif (TEST_HARDWARE)
if (AIE_GEN)
  set (_test_cflag ${_test_cflag} -DAIE_GEN=${AIE_GEN})
else(AIE_GEN)
  set (_test_cflag ${_test_cflag} -DAIE_GEN=1)
endif(AIE_GEN)
target_compile_options (${EXETEST} PUBLIC ${_test_cflag})
target_link_libraries (${EXETEST} ${_deps})
