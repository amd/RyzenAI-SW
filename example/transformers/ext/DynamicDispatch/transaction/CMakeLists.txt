# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

set(LIBRARY_NAME transaction)

set(generate_cmd
    ${Python3_EXECUTABLE}
    "${PROJECT_SOURCE_DIR}/tools/parse_transaction_binary.py"
    "--root"
    "${PROJECT_SOURCE_DIR}"
    "--out-dir"
    "${CMAKE_CURRENT_BINARY_DIR}"
    "--quiet"
)

execute_process(
  COMMAND ${generate_cmd} "--list" OUTPUT_VARIABLE transaction_package_files
)
string(REPLACE "\n" ";" transaction_package_files_list
               ${transaction_package_files}
)

add_custom_command(
  OUTPUT ${transaction_package_files_list} COMMAND ${generate_cmd}
  DEPENDS ${PROJECT_SOURCE_DIR}/tools/parse_transaction_binary.py
)

add_library(${LIBRARY_NAME} ${transaction_package_files_list})
target_link_libraries(${LIBRARY_NAME} PRIVATE ZLIB::ZLIB)
target_include_directories(
  ${LIBRARY_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include
)

include(GNUInstallDirs)

install(
  TARGETS ${LIBRARY_NAME}
  EXPORT ${PROJECT_NAME}-targets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ryzenai/dynamic_dispatch
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
