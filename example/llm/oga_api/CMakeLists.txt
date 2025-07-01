cmake_minimum_required(VERSION 3.15)
project(oga_api)

set(CMAKE_CXX_STANDARD 17)

# Output directory setup
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add library search path
link_directories(${CMAKE_SOURCE_DIR}/libs)

# Executable source
add_executable(example main.cpp)

# Link the .lib file (drop the extension)
target_link_libraries(example onnxruntime-genai)

# Copy DLLs after build
file(GLOB DLLS "${CMAKE_SOURCE_DIR}/libs/*.dll")
add_custom_command(TARGET example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${DLLS}
    $<TARGET_FILE_DIR:example>)
