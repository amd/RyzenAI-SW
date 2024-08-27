set CMAKE_INSTALL_PREFIX=%cd%\..
set CMAKE_PREFIX_PATH=%cd%\..

:: --------------------------------------------------------------
:: Please reset the env below to your Ryzen AI installation path
:: --------------------------------------------------------------
set RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\1.2.0

set ONNXRUNTIME_ROOTDIR=%RYZEN_AI_INSTALLATION_PATH%/onnxruntime

:: --------------------------------------------------------------
:: Please reset the env below to your opencv installation path
:: --------------------------------------------------------------
set OpenCV_DIR=C:\Users\fanz\Downloads\NPU\dependencies\opencv\build

set buildType=Release
set src=%cd%
set dst=%cd%\build
set "defaultArg=-DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON"
set "buildTypeOption=-DCMAKE_CONFIGURATION_TYPES=%buildType%"
set "generatorOption=-A x64 -T host=x64  "Visual Studio 17 2022""
set "prefix=-DCMAKE_INSTALL_PREFIX="%CMAKE_INSTALL_PREFIX%" -DCMAKE_PREFIX_PATH="%CMAKE_PREFIX_PATH%" -DONNXRUNTIME_ROOTDIR="%ONNXRUNTIME_ROOTDIR%""
cmake %defaultArg% %buildTypeOption% %generatorOption% %prefix% %customConfigOption% -B %dst% -S %src%

cmake --build %dst% --config %buildType% --clean-first -- /p:CL_MPcount=%buildParallelCount% /nodeReuse:False

cmake --install  %dst% --config %buildType%
pause