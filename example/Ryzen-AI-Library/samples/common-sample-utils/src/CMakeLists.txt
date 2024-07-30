# Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.

find_package(OpenCV REQUIRED PATHS ${OPENCV_INSTALL_ROOT})

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${AMD_CVML_SDK_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/../include ${OpenCV_INCLUDE_DIRS})

add_library(common-sample-utils STATIC ${SOURCES})

target_include_directories(common-sample-utils
PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(common-sample-utils
    ${OpenCV_LIBS}
)

install(TARGETS common-sample-utils DESTINATION ${AMD_CVML_SDK_ROOT}/${PLATFORM})
