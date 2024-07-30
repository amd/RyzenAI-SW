
add_executable(apu_check
    main.cpp)

target_compile_definitions(apu_check PUBLIC
    -DIPU_DRIVER_NAME="${IPU_DRIVER_NAME}")
	
if (MSVC)
target_compile_options(apu_check PRIVATE /Qspectre /ZH:SHA_256 /guard:cf /W3 /Zi /Zc:__cplusplus)
target_link_options(apu_check PRIVATE "/DEBUG" "/CETCOMPAT")
endif()


target_include_directories(apu_check PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(apu_check PRIVATE
    Setupapi.lib
)

if(DEFINED ENV{DK_ROOT})
    set(INSTALL_FILES_LIST ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/apu_check.exe)
else()
    set(INSTALL_FILES_LIST
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/apu_check.exe)
endif()


install(FILES
    ${INSTALL_FILES_LIST}
    CONFIGURATIONS ${CMAKE_BUILD_TYPE}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}
)

install(FILES 
    $<TARGET_PDB_FILE:apu_check> 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE})
		 