
include(ExternalProject)

function(BuildExternalGitProject target url tag args deps skipdownload cpack)

message(STATUS "Configuring External Project ${target}")
message(STATUS "Target ${target} depends on targets ${deps}")
message(STATUS "Target ${target} built with arguments ${args}")

if(${skipdownload})
  message(STATUS "Will skip download step for ${target}")
  ExternalProject_Add(
    ${target}
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/${target}"
    CMAKE_ARGS ${args}
    TEST_COMMAND ""
    LIST_SEPARATOR |
    LOG_BUILD ON
  )
else()
    message(STATUS "${target} will be downloaded")
    message(STATUS "${target}_URL = ${url}")
    message(STATUS "${target}_TAG = ${tag}")
    ExternalProject_Add(
    ${target}
	GIT_REPOSITORY "${url}"
    GIT_TAG "${tag}"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/${target}"
    CMAKE_ARGS ${args}
    TEST_COMMAND ""
    LIST_SEPARATOR |
    LOG_BUILD ON
  )
endif()

ExternalProject_Add_StepDependencies(${target} build ${deps})

if(cpack)
# Add a custom step to the external project's build process
ExternalProject_Add_Step(
  ${target}
  cpack
  COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> --target package
  COMMENT "Running CPack for ${target}"
  DEPENDEES build
)
endif()

endfunction()