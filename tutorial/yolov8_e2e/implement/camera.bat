set XLNX_VART_FIRMWARE=%cd%\..\1x4.xclbin
set VITISAI_EP_JSON_CONFIG=%cd%\..\vaip_config.json
@REM set XILINX_XRT=%cd%\..\xrt

set PATH=%cd%\..\lib;C:\Users\AMD\anaconda3\envs\auto_rc2;%PATH%;C:\Program Files\gflags\bin;C:\Program Files\glog\bin

::the following two paths does not work in embedded python. To edit path for embbed python, change .pth file in the python folder
set PATHONHOME=C:\Users\AMD\anaconda3\envs\auto_rc2
set PYTHONPATH=C:\Users\AMD\anaconda3\envs\auto_rc2\Lib;C:\Users\AMD\anaconda3\envs\auto_rc2\DLLs

set NUM_OF_DPU_RUNNERS=4
%cd%\..\bin\camera_yolov8_nx1x4  -c 5 -x 1 -y 1 -s 0 -D -R 1920x1080 -r 2560x1440 %1
