
# Forward Select Super Project CMake Variables To External Projects

list(APPEND CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")

# Special handling for PREFIX PATH, because it can be a list
string(REPLACE ";" "|" CMAKE_PREFIX_PATH_ALT_SEP "${CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_ARGS -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH_ALT_SEP}")

if(BUILD_SHARED_LIBS)
  list(APPEND CMAKE_ARGS "-DBUILD_SHARED_LIBS=ON")
else()
  list(APPEND CMAKE_ARGS "-DBUILD_SHARED_LIBS=OFF")
endif()

list(APPEND CMAKE_ARGS "-DCMAKE_POSITION_INDEPENDENT_CODE=ON")

message(STATUS "Default CMake Arguments:")
foreach(arg ${CMAKE_ARGS})
  message(STATUS "-- ${arg}")
endforeach()