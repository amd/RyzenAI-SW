set XLNX_VART_FIRMWARE=%cd%\..\1x4.xclbin
@REM set XILINX_XRT=%cd%\..\xrt

set PATH=%cd%\..\bin;%cd%\..\python;%cd%\..;%PATH%;C:\Program Files\gflags\bin;C:\Program Files\glog\bin
::the following two paths does not work in embedded python. To edit path for embbed python, change .pth file in the python folder
%cd%\..\bin\test_jpeg_yolov8.exe %1 %2
