set RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\1.2.0
set XLNX_VART_FIRMWARE=%RYZEN_AI_INSTALLATION_PATH%/voe-4.0-win_amd64/xclbins/phoenix/1x4.xclbin
set PATH=%ONNXRUNTIME_ROOTDIR%/bin;%cd%\..;%PATH%
set DEBUG_ONNX_TASK=0
set DEBUG_DEMO=0
set NUM_OF_DPU_RUNNERS=4
set XLNX_ENABLE_GRAPH_ENGINE_PAD=1
set XLNX_ENABLE_GRAPH_ENGINE_DEPAD=1
%cd%\..\bin\npu_multi_models.exe %cd%\config\modelx4_with_camera_on.json
