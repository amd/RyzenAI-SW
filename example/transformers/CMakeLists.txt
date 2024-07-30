cmake_minimum_required(VERSION 3.21)

# Define a project name and a version to resolve warnings
# This could be useful in packaging the deliverables in the future
project(RyzenAI-Transformers VERSION 0.0.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_subdirectory(ops/cpp)
add_subdirectory(ops/torch_cpp)
add_subdirectory(tests)
