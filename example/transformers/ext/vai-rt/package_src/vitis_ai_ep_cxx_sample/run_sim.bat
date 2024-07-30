set XLNX_VART_FIRMWARE=%cd%\..\1x4.xclbin

set PATH=%cd%\..\bin;%cd%\..\python;%cd%\..;%PATH%
set USE_CPU_RUNNER=1
set VAIP_COMPILE_RESERVE_CONST_DATA=1
::the following two paths does not work in embedded python. To edit path for embbed python, change .pth file in the python folder
@REM set PYTHONHOME=
@REM set PYTHONPATH=
%cd%\..\bin\resnet50_pt.exe %cd%\..\models\resnet50_pt.onnx %cd%\resnet50\resnet50.jpg
pause
