set XLNX_VART_FIRMWARE=%cd%\..\1x4.xclbin
set PATH=%cd%\..\bin;%cd%\..;%PATH%
for /f "tokens=*" %%i in ('where python') do set PYTHONPATH=%%i
set PYTHONPATH=%PYTHON_PATH%;%PATHONPATH%
echo %PATHONPATH%
set DEBUG_ONNX_TASK=0
set DEBUG_DEMO=0
set NUM_OF_DPU_RUNNERS=4
set XLNX_ENABLE_GRAPH_ENGINE_PAD=1
set XLNX_ENABLE_GRAPH_ENGINE_DEPAD=1
%cd%\..\bin\ipu_multi_models.exe %cd%\config\modelx4.json
