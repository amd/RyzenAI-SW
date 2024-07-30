@echo off
setlocal

REM Check if %CONDA_PREFIX% is set
if "%CONDA_PREFIX%"=="" (
    echo Error: CONDA_PREFIX is not set.
    exit /b 1
)

REM Set the root directory for the dependency
set "CWD=%~dp0"
set "AIERT_CMAKE_PATH=%CWD%\ext\aie-rt\driver\src"
set "AIECTRL_CMAKE_PATH=%CWD%\ext\aie_controller"
set "DD_CMAKE_PATH=%CWD%\ext\DynamicDispatch"
set "XRT_DIR=%CWD%\third_party\xrt-ipu"

REM Check if the directory exists
if not exist "%AIERT_CMAKE_PATH%" (
    echo Error: Directory %AIERT_CMAKE_PATH% does not exist.
    exit /b 1
)

REM Check if the directory exists
if not exist "%AIECTRL_CMAKE_PATH%" (
    echo Error: Directory %AIECTRL_CMAKE_PATH% does not exist.
    exit /b 1
)

REM Check if the directory exists
if not exist "%DD_CMAKE_PATH%" (
    echo Error: Directory %DD_CMAKE_PATH% does not exist.
    exit /b 1
)

REM Check if the directory exists
if not exist "%XRT_DIR%" (
    echo Error: Directory %XRT_DIR% does not exist.
    exit /b 1
)

REM Invoke cmake to build and install the dependency
cmake -S %AIERT_CMAKE_PATH% -B build_aiert -DXAIENGINE_BUILD_SHARED=OFF -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX%
if errorlevel 1 (
    echo Error: cmake configuration failed.
    exit /b 1
)

cmake --build build_aiert --target install --config Release
if errorlevel 1 (
    echo Error: cmake build or installation failed.
    exit /b 1
)

REM Invoke cmake to build and install the dependency
cmake -S %AIECTRL_CMAKE_PATH% -B build_aiectrl -DCMAKE_PREFIX_PATH=%CONDA_PREFIX% -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX%
if errorlevel 1 (
    echo Error: cmake configuration failed.
    exit /b 1
)

cmake --build build_aiectrl --target install --config Release
if errorlevel 1 (
    echo Error: cmake build or installation failed.
    exit /b 1
)

REM Invoke cmake to build and install the dependency
cmake -S %DD_CMAKE_PATH% -B build_dd -DCMAKE_PREFIX_PATH=%CONDA_PREFIX% -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX%
if errorlevel 1 (
    echo Error: cmake configuration failed.
    exit /b 1
)

cmake --build build_dd --target install --config Release
if errorlevel 1 (
    echo Error: cmake build or installation failed.
    exit /b 1
)

echo Build and installation completed successfully.
endlocal
