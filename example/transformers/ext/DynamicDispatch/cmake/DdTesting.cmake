# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

# This function is used to build all test executables
function(dd_configure_test target use_gtest)
  find_package(XRT REQUIRED)
  set(INCL_DIRS ${PROJECT_SOURCE_DIR}/include ${XRT_INCLUDE_DIRS}
                ${PROJECT_SOURCE_DIR}/tests/cpp/include ${DD_SRC_INCLUDE_DIRS}
  )

  set(LINK_LIBS
      ${XRT_COREUTIL_LIBRARIES} nlohmann_json::nlohmann_json spdlog::spdlog
      xaiengine::xaiengine dyn_dispatch_core aie_controller
  )

  if(ENABLE_SIMNOWLITE_BUILD)
    set(LINK_LIBS ${XRT_HWEMU_LIBRARIES} ${LINK_LIBS})
  endif()

  if(${use_gtest})
    list(APPEND LINK_LIBS GTest::gtest_main)
  endif()

  target_include_directories(${target} PRIVATE ${INCL_DIRS})
  target_link_libraries(${target} PRIVATE ${LINK_LIBS})
  target_compile_options(${target} PRIVATE ${DD_DEFAULT_COMPILE_OPTIONS})

  install(TARGETS ${target} DESTINATION tests)

  # The following is removed to avoid build errors
  # The build error happen because the cpp_tests.exe is called
  # during the build, but it crashes do to incomplete PATH settings
  # The consequence is that ctest will not be able to
  # run gtest testcases
  # In the future we may want ctest integration
  # So leaving this here commented out
  #if(${use_gtest})
  #  gtest_discover_tests(${target} DISCOVERY_TIMEOUT 120)
  #endif()
endfunction()
