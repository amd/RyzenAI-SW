@echo off

if "%CONDA_PREFIX%" == "" echo CONDA_PREFIX not set. This script must be executed from within the RyzenAI conda environment. & goto :error
if "%RYZEN_AI_INSTALLATION_PATH%" == "" echo RYZEN_AI_INSTALLATION_PATH not set. This script requires the RYZEN_AI_INSTALLATION_PATH env var to be set to the RyzenAI installation folder. & goto :error

cmake -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -B build -S . -G "Visual Studio 17 2022"

cmake --build .\build --config Release --target ALL_BUILD

echo.
echo Copying ONNX models, compiled models and vitisai_config.json file
xcopy /Y /I /S ..\models build\Release\models > nul
xcopy /Y /I /S ..\my_cache_dir build\Release\my_cache_dir > nul
xcopy /Y /-I ..\vitisai_config.json build\Release > nul
xcopy /Y /E /I test_images build\Release\test_images > nul

:error
exit /b %errorlevel%