set XLNX_VART_FIRMWARE=%cd%\..\1x4.xclbin
set PATH=%cd%\..\bin;%cd%\..\python;%cd%\..;%PATH%
set DEBUG_ONNX_TASK=0
set DEBUG_DEMO=0
set NUM_OF_DPU_RUNNERS=4
set XLNX_ENABLE_GRAPH_ENGINE_PAD=1
set XLNX_ENABLE_GRAPH_ENGINE_DEPAD=1
%cd%\..\bin\ipu_multi_models.exe %cd%\config\yolovx.json
