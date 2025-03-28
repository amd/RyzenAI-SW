@REM # Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.

@REM Build release versions the sample apps

set AMD_CVML_SDK_ROOT="%CD%\.."
set BUILD_DIR="%CD%\build-samples"
if exist %BUILD_DIR% (
    rmdir /Q /S %BUILD_DIR%
)

cmake -S %CD% -B %BUILD_DIR% || goto :error
cmake --build %BUILD_DIR% --config Release --target ALL_BUILD || goto :error

:error
exit /b %errorlevel%