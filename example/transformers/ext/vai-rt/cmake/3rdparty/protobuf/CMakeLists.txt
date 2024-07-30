
if(MSVC)
  list(APPEND CMAKE_ARGS "-Dprotobuf_MSVC_STATIC_RUNTIME=OFF")
endif()

list(APPEND CMAKE_ARGS "-Dprotobuf_BUILD_SHARED_LIBS=OFF")

ExternalProject_Add(
  protobuf
  GIT_REPOSITORY "${protobuf_URL}"
  GIT_TAG "${protobuf_TAG}"
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/protobuf"
  SOURCE_SUBDIR cmake
  CMAKE_ARGS ${CMAKE_ARGS}
  TEST_COMMAND ""
  LIST_SEPARATOR |
)
