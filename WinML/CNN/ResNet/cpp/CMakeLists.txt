cmake_minimum_required(VERSION 3.23)
cmake_policy(SET CMP0135 NEW)

project(CppResnetBuildDemo VERSION 1.0)

if(NOT WIN32 OR NOT MSVC)
  message(FATAL_ERROR "Unsupported platform. This project requires Windows + MSVC.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---------------------------------------------------------------------------
# NuGet package version pins. Bump here, then delete build/ to force a re-fetch.
# ---------------------------------------------------------------------------
set(MML_ML_VERSION         "2.1.74")       # Microsoft.WindowsAppSDK.ML (auto-init source)
set(MML_WAML_VERSION       "2.1.74")       # Microsoft.Windows.AI.MachineLearning (ORT C++ headers + WinML metadata + libs/dlls)
set(MML_FOUNDATION_VERSION "2.1.0")        # Microsoft.WindowsAppSDK.Foundation (bootstrap lib/dll)
set(MML_WIL_VERSION        "1.0.250325.1") # Microsoft.Windows.ImplementationLibrary (WIL headers)

# ---------------------------------------------------------------------------
# Fetch the packages. A .nupkg is a plain zip, so FetchContent downloads and
# extracts each one directly from nuget.org. FetchContent does not resolve
# transitive deps, so every package the build consumes is listed explicitly.
# ---------------------------------------------------------------------------
include(FetchContent)

# Fetch a NuGet package by id + version; sets <out_var> to its extracted root.
function(fetch_nuget_package name id version out_var)
  string(TOLOWER "${id}" _id_lower)
  FetchContent_Declare(${name}
    URL "https://api.nuget.org/v3-flatcontainer/${_id_lower}/${version}/${_id_lower}.${version}.nupkg")
  FetchContent_MakeAvailable(${name})
  set(${out_var} "${${name}_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

fetch_nuget_package(wasdkml         Microsoft.WindowsAppSDK.ML              "${MML_ML_VERSION}"         WASDKML_PACKAGE_ROOT)
fetch_nuget_package(waml            Microsoft.Windows.AI.MachineLearning    "${MML_WAML_VERSION}"       WAML_PACKAGE_ROOT)
fetch_nuget_package(wasdkfoundation Microsoft.WindowsAppSDK.Foundation      "${MML_FOUNDATION_VERSION}" WASDK_FOUNDATION_ROOT)
fetch_nuget_package(wil             Microsoft.Windows.ImplementationLibrary "${MML_WIL_VERSION}"        WIL_PACKAGE_ROOT)

set(WAML_METADATA_DIR "${WAML_PACKAGE_ROOT}/metadata")
set(WAML_INCLUDE_DIR  "${WAML_PACKAGE_ROOT}/include")
set(WIL_INCLUDE_DIR   "${WIL_PACKAGE_ROOT}/include")

# ---------------------------------------------------------------------------
# Resolve Windows SDK UnionMetadata (Windows.winmd) for the cppwinrt projection.
# ---------------------------------------------------------------------------
set(WINSDK_DIR "$ENV{WindowsSdkDir}")
set(WINSDK_VER "$ENV{WindowsSDKVersion}")
if(WINSDK_DIR STREQUAL "")
  set(WINSDK_DIR "C:/Program Files (x86)/Windows Kits/10")
endif()
if(NOT WINSDK_VER STREQUAL "")
  string(REGEX REPLACE "\\\\$" "" WINSDK_VER "${WINSDK_VER}")
  set(WSDK_UNION_METADATA_DIR "${WINSDK_DIR}/UnionMetadata/${WINSDK_VER}")
else()
  file(GLOB WINSDK_UM_DIRS "${WINSDK_DIR}/UnionMetadata/[0-9]*")
  if(NOT WINSDK_UM_DIRS)
    message(FATAL_ERROR "Windows SDK UnionMetadata not found under ${WINSDK_DIR}. Install the Windows 10/11 SDK.")
  endif()
  list(SORT WINSDK_UM_DIRS)
  list(REVERSE WINSDK_UM_DIRS)
  list(GET WINSDK_UM_DIRS 0 WSDK_UNION_METADATA_DIR)
endif()
set(WINDOWS_WINMD "${WSDK_UNION_METADATA_DIR}/Windows.winmd")
if(NOT EXISTS "${WINDOWS_WINMD}")
  message(FATAL_ERROR "Windows.winmd not found in ${WSDK_UNION_METADATA_DIR}. Ensure the Windows SDK is installed.")
endif()
set(WINDOWS_FACADE_WINMD "${WSDK_UNION_METADATA_DIR}/Facade/windows.winmd")
set(CPPWINRT_REF_ARGS -reference "${WINDOWS_WINMD}")
if(EXISTS "${WINDOWS_FACADE_WINMD}")
  list(APPEND CPPWINRT_REF_ARGS -reference "${WINDOWS_FACADE_WINMD}")
endif()

# ---------------------------------------------------------------------------
# Generate C++/WinRT projection headers.
# One projection tree (in the build dir) covers both the Windows SDK namespaces
# the app uses (Windows.Graphics.Imaging, Windows.Foundation, Windows.Storage)
# and the Microsoft.Windows.AI.MachineLearning WinRT types.
# ---------------------------------------------------------------------------
set(CPPWINRT_PROJECTION_DIR "${CMAKE_CURRENT_BINARY_DIR}/cppwinrt")
file(GLOB _winsdk_bin_dirs "${WINSDK_DIR}/bin/[0-9]*/x64" "${WINSDK_DIR}/bin/[0-9]*/x86")
find_program(CPPWINRT_EXECUTABLE NAMES cppwinrt HINTS ${_winsdk_bin_dirs})
if(NOT CPPWINRT_EXECUTABLE)
  message(FATAL_ERROR "cppwinrt tool not found in PATH or ${WINSDK_DIR}/bin. Install C++/WinRT tooling (Windows SDK / VS).")
endif()
message(STATUS "Using cppwinrt: ${CPPWINRT_EXECUTABLE}")
add_custom_target(generate_winml_projection
    COMMAND ${CPPWINRT_EXECUTABLE}
      -in "${WSDK_UNION_METADATA_DIR}"
      -in "${WAML_METADATA_DIR}"
      ${CPPWINRT_REF_ARGS}
      -out "${CPPWINRT_PROJECTION_DIR}"
    DEPENDS "${WAML_METADATA_DIR}/Microsoft.Windows.AI.MachineLearning.winmd"
    COMMENT "Generating C++/WinRT projection headers (Windows SDK + Microsoft.Windows.AI.MachineLearning)"
)

# ---------------------------------------------------------------------------
# Package lib/dll dirs (NuGet native layout, x64).
# ---------------------------------------------------------------------------
set(WASDK_FOUNDATION_INCLUDE_DIR "${WASDK_FOUNDATION_ROOT}/include")
set(WASDK_FOUNDATION_LIB_DIR     "${WASDK_FOUNDATION_ROOT}/lib/native/x64")
set(WASDK_FOUNDATION_DLL_DIR     "${WASDK_FOUNDATION_ROOT}/runtimes/win-x64/native")

file(GLOB WAML_LIBS "${WAML_PACKAGE_ROOT}/lib/native/x64/*.lib")
file(GLOB WAML_DLLS "${WAML_PACKAGE_ROOT}/runtimes/win-x64/native/*.dll")

# ---------------------------------------------------------------------------
# The executable.
# ---------------------------------------------------------------------------
add_executable(${PROJECT_NAME}
    src/CppResnetBuildDemo.cpp
    src/ResnetModelHelper.cpp
)
add_dependencies(${PROJECT_NAME} generate_winml_projection)

# WindowsML self-contained auto-initializer: provides OrtGetApiBase and handles
# onnxruntime.dll loading at runtime.
set(WINML_AUTOINIT_SRC "${WASDKML_PACKAGE_ROOT}/include/WindowsMLAutoInitializer.SelfContained.cpp")
if(NOT EXISTS "${WINML_AUTOINIT_SRC}")
  message(FATAL_ERROR "WindowsML auto-initializer not found at ${WINML_AUTOINIT_SRC}. Delete build/ to re-fetch.")
endif()
target_sources(${PROJECT_NAME} PRIVATE "${WINML_AUTOINIT_SRC}")

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CPPWINRT_PROJECTION_DIR}"
    "${WAML_INCLUDE_DIR}"
    "${WASDK_FOUNDATION_INCLUDE_DIR}"
    "${WIL_INCLUDE_DIR}"
    "${WAML_METADATA_DIR}"
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_GLOBAL_WindowsPackageType "None"
    VS_GLOBAL_WindowsAppSDKSelfContained "true"
    VS_GLOBAL_WindowsAppSdkBootstrapInitialize "true"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${WAML_LIBS}
    "${WASDK_FOUNDATION_LIB_DIR}/Microsoft.WindowsAppRuntime.Bootstrap.lib"
    windowsapp
    runtimeobject
)

# ---------------------------------------------------------------------------
# Stage runtime DLLs and resources next to the exe.
# ---------------------------------------------------------------------------
if(WAML_DLLS)
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${WAML_DLLS} "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
endif()

if(EXISTS "${WASDK_FOUNDATION_DLL_DIR}/Microsoft.WindowsAppRuntime.Bootstrap.dll")
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${WASDK_FOUNDATION_DLL_DIR}/Microsoft.WindowsAppRuntime.Bootstrap.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
endif()

# ResNet class labels are read from beside the executable at runtime.
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/src/ResNet50Labels.txt"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>")

# Default sample image, so the exe works with no --image_path argument.
set(_default_image "${CMAKE_CURRENT_SOURCE_DIR}/../images/dog.jpg")
if(EXISTS "${_default_image}")
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_default_image}"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
endif()
