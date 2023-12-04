set XLNX_VART_FIRMWARE=%cd%\..\1x4.xclbin
set VITISAI_EP_JSON_CONFIG=%cd%\..\vaip_config.json
:: please setup your conda env path below. For example, C:\Users\AMD\anaconda3\envs\ryzen_ai
set ENV_PATH=<YOUR-CONDA-ENV-PATH>

set PATH=%cd%\..\lib;%ENV_PATH%;%PATH%;C:\Program Files\gflags\bin;C:\Program Files\glog\bin

::the following two paths does not work in embedded python. To edit path for embbed python, change .pth file in the python folder
set PATHONHOME=%ENV_PATH%
set PYTHONPATH=%ENV_PATH%\Lib;%ENV_PATH%\DLLs

set NUM_OF_DPU_RUNNERS=4
%cd%\..\bin\camera_yolov8_nx1x4  -c 5 -x 1 -y 1 -s 0 -D -R 1920x1080 -r 2560x1440 %1
