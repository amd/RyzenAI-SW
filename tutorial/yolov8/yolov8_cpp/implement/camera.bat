set PATH=%ONNXRUNTIME_ROOTDIR%/bin;%cd%\..;%PATH%;

::the following two paths does not work in embedded python. To edit path for embbed python, change .pth file in the python folder
%cd%\..\bin\camera_yolov8_nx1x4.exe -c 5 -x 1 -y 1 -s 0 -D -R 1920x1080 -r 2560x1440 %1

