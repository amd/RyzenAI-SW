@REM Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
@echo off

if "%IPU_BUILD_VARIANT%" == "" (
    set IPU_BUILD_VARIANT=Release
)

:continue
echo Building NPU/APU platform check application...

cmake -S .\ -B .\build_out -DCMAKE_INSTALL_PREFIX=.\build_out\install ^
                           -DCMAKE_BUILD_TYPE=%IPU_BUILD_VARIANT%

if %ERRORLEVEL% NEQ 0 echo "cmake config failed..." & goto :error
cmake --build .\build_out --config %IPU_BUILD_VARIANT% --target ALL_BUILD
if %ERRORLEVEL% NEQ 0 echo "cmake build failed..." & goto :error
cmake --build .\build_out --config %IPU_BUILD_VARIANT% --target INSTALL
if %ERRORLEVEL% NEQ 0 echo "cmake install failed..." & goto :error

:Error
exit /b %ERRORLEVEL%