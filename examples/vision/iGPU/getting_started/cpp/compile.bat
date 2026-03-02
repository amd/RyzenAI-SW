@echo off

if "%RYZEN_AI_INSTALLATION_PATH%" == "" echo RYZEN_AI_INSTALLATION_PATH not set. This script requires the RYZEN_AI_INSTALLATION_PATH env var to be set to the RyzenAI 1.2 installation folder. & goto :error

REM Check if the first argument is provided
if "%1"=="" (
    echo Usage: %0 [OpenCV_DIR]
    exit /b 1
)

set "OpenCV_DIR=%~1"

echo OpenCV_DIR is set to: %OpenCV_DIR%

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -DCMAKE_INSTALL_PREFIX=. -DCMAKE_PREFIX_PATH=. -B build -S resnet50 -DOpenCV_DIR="%OpenCV_DIR%" -G "Visual Studio 17 2022"

cmake --build .\build --config Release --target ALL_BUILD

:error
exit /b %errorlevel%