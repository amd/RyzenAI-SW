set PATH=%ONNXRUNTIME_ROOTDIR%/bin;%cd%\..;%PATH%
set XLNX_VART_FIRMWARE=%RYZEN_AI_INSTALLATION_PATH%\voe-4.0-win_amd64\xclbins\strix\AMD_AIE2P_Nx4_Overlay.xclbin

%cd%\..\bin\test_jpeg_yolov8.exe ./DetectionModel_int.onnx ./sample_yolov8.jpg