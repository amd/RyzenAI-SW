set XLNX_VART_FIRMWARE=%cd%\..\1x4.xclbin
@REM set XILINX_XRT=%cd%\..\xrt

set PATH=%cd%\..\bin;%cd%\..\python;%cd%\..;%PATH%;C:\Program Files\gflags\bin;C:\Program Files\glog\bin
::the following two paths does not work in embedded python. To edit path for embbed python, change .pth file in the python folder
set NUM_OF_DPU_RUNNERS=4
%cd%\..\bin\camera_yolov8_nx1x4  -c 5 -x 1 -y 1 -s 0 -D -R 1920x1080 -r 2560x1440 %1
