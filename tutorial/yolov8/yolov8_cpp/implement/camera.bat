set XLNX_VART_FIRMWARE=%RYZEN_AI_INSTALLATION_PATH%/voe-4.0-win_amd64/xclbins/phoenix/1x4.xclbin

set PATH=%ONNXRUNTIME_ROOTDIR%/bin;%cd%\..;%PATH%;
::the following two paths does not work in embedded python. To edit path for embbed python, change .pth file in the python folder
%cd%\..\bin\camera_yolov8_nx1x4.exe -c 5 -x 1 -y 1 -s 0 -D -R 1920x1080 -r 2560x1440 %1

