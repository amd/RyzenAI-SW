@echo off

if "%RYZEN_AI_INSTALLATION_PATH%" == "" echo RYZEN_AI_INSTALLATION_PATH not set. This script requires the RYZEN_AI_INSTALLATION_PATH env var to be set to the RyzenAI 1.2 installation folder. & goto :error

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -DCMAKE_INSTALL_PREFIX=. -DCMAKE_PREFIX_PATH=. -B build -S resnet50 -DOpenCV_DIR="C:/opencv/build" -G "Visual Studio 17 2022"

cmake --build .\build --config Release --target ALL_BUILD

:error
exit /b %errorlevel%