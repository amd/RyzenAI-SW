set XLNX_VART_FIRMWARE=%RYZEN_AI_INSTALLATION_PATH%/voe-4.0-win_amd64/xclbins/phoenix/1x4.xclbin
set PATH=%ONNXRUNTIME_ROOTDIR%/bin;%cd%\..;%PATH%

%cd%\..\bin\test_jpeg_yolov8.exe ./DetectionModel_int.onnx ./sample_yolov8.jpg