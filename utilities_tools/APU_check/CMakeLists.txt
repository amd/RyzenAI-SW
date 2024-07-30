cmake_minimum_required(VERSION 3.13)

# For Windows driver build using DK_ROOT Tool chain
if(DEFINED ENV{DK_ROOT})
  set (CMAKE_SYSTEM_VERSION 10.0 CACHE TYPE INTERNAL FORCE)
endif()

project("APU-check" VERSION 1.0.0.0)

set(CMAKE_DEBUG_POSTFIX "-d")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(
    /Qspectre
    /ZH:SHA_256
    /guard:cf
    /W3
    /Zi
    /Zc:__cplusplus
)
add_link_options(
    /CETCOMPAT
    /DEBUG
)

add_subdirectory(src)
