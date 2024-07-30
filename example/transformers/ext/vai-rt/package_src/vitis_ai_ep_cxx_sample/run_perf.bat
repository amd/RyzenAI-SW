set XLNX_VART_FIRMWARE=%cd%\..\1x4.xclbin

set PATH=%cd%\..\bin;%cd%\..\python;%cd%\..;%PATH%

set perf_target=vitisai

set NUM_OF_DPU_RUNNERS=1
onnxruntime_perf_test.exe -e %perf_target% -i "config_file|..\vaip_config.json" -t 60 -c 1

@REM the following commands start with four runners
@REM set NUM_OF_DPU_RUNNERS=16
@REM onnxruntime_perf_test.exe -e %perf_target% -i "config_file|..\vaip_config.json" -t 60 -c 16
pause
