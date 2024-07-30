
# This is how to build Boost with CMake
# Unfortunately this is not the standard way of building Boost
# It seems to cause some issues on Windows
#ExternalProject_Add(
#  boost
#  GIT_REPOSITORY "https://github.com/boostorg/boost"
#  GIT_TAG "${boost_TAG}"
#  UPDATE_COMMAND ""
#  PATCH_COMMAND ""
#  SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/boost"
#  CMAKE_ARGS ${CMAKE_ARGS}
#  TEST_COMMAND ""
#  LIST_SEPARATOR |
#)

if(MSVC)
ExternalProject_Add(
  boost
  URL ${boost_URL}
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/boost"
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND bootstrap.bat
  BUILD_COMMAND
    b2 install
      --prefix=${CMAKE_INSTALL_PREFIX}
      --build-type=complete
      address-model=64
      architecture=x86
      link=static
      threading=multi
      --with-filesystem
      --with-program_options
      --with-system
  INSTALL_COMMAND ""
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  )
else()
ExternalProject_Add(
  boost
  URL ${boost_URL}
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/boost"
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ./bootstrap.sh
  BUILD_COMMAND
    ./b2 install
      --prefix=${CMAKE_INSTALL_PREFIX}
      --with-filesystem
      --with-program_options
      --with-system
  INSTALL_COMMAND ""
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  )
endif()
