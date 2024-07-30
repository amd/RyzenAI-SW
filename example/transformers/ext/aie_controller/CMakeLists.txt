cmake_minimum_required(VERSION 3.12)

# Project name
project(aie_controller VERSION 1.4.0)

find_package(xaiengine REQUIRED)

# Define the interface library
add_library(${PROJECT_NAME} INTERFACE)

# Include directory
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/ps>
    $<INSTALL_INTERFACE:include/ryzenai/${PROJECT_NAME}>
    $<INSTALL_INTERFACE:include/ryzenai/${PROJECT_NAME}/ps>
)
target_link_libraries(${PROJECT_NAME} INTERFACE xaiengine::xaiengine)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Install the package version file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# Install the CMake target in export set *ConfigTargets
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}ConfigTargets
    INCLUDES DESTINATION include
    RUNTIME  DESTINATION bin
    ARCHIVE  DESTINATION lib
    LIBRARY  DESTINATION lib
)

# Install header files
install(DIRECTORY include/
    DESTINATION include/ryzenai
)

# Save the export set to a file
install(EXPORT ${PROJECT_NAME}ConfigTargets
    FILE ${PROJECT_NAME}ConfigTargets.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# Generate the package configuration file
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

# Install the package configuration file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
)
