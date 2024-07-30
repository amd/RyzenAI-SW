
list(APPEND CMAKE_ARGS "-Dgtest_force_shared_crt=ON")

ExternalProject_Add(
  googletest
  GIT_REPOSITORY "${googletest_URL}"
  GIT_TAG "${googletest_TAG}"
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/googletest"
  CMAKE_ARGS ${CMAKE_ARGS}
  TEST_COMMAND ""
  LIST_SEPARATOR |
)
