set XLNX_VART_FIRMWARE=%cd%\..\1x4.xclbin

set PATH=%cd%\..\bin;%cd%\..\python;%cd%\..;%PATH%
set VITISAI_EP_JSON_CONFIG=%cd%\..\vaip_config.json
::the following two paths does not work in embedded python. To edit path for embbed python, change .pth file in the python folder
@REM set PYTHONHOME=
@REM set PYTHONPATH=

test_onnx_runner.exe ..\models\resnet50_pt.onnx
pause 
