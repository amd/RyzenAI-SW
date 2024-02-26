set XLNX_VART_FIRMWARE=%cd%\..\1x4.xclbin
set VITISAI_EP_JSON_CONFIG=%cd%\..\vaip_config.json
:: please setup your conda env path below. For example, C:\Users\AMD\anaconda3\envs\ryzen_ai
set ENV_PATH=<YOUR-CONDA-ENV-PATH>

set PATH=%cd%\..\lib;%ENV_PATH%;%PATH%

::the following two paths does not work in embedded python. To edit path for embbed python, change .pth file in the python folder
set PATHONHOME=%ENV_PATH%
set PYTHONPATH=%ENV_PATH%\Lib;%ENV_PATH%\DLLs

%cd%\..\bin\test_jpeg_yolov8.exe .\DetectionModel_int.onnx .\sample_yolov8.jpg
