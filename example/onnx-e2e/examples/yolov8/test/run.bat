::
:: Copyright 2022-2023 Advanced Micro Devices Inc.
::
:: Licensed under the Apache License, Version 2.0 (the "License"); you may not
:: use this file except in compliance with the License. You may obtain a copy of
:: the License at
::
:: http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
:: WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
:: License for the specific language governing permissions and limitations under
:: the License.
::
@ECHO OFF
setlocal

@REM Save intermediate results from model compilation
set XLNX_ENABLE_DUMP_XIR_MODEL=0
set XLNX_ENABLE_DUMP_ONNX_MODEL=0
set ENABLE_SAVE_ONNX_MODEL=0

@REM Enable debug messages from few passes and executors
set DEBUG_FUSE_DEVICE_SUBGRAPH=0
set DEBUG_VAIP_PY_EXT=0
set DEBUG_GRAPH_RUNNER=0
set XLNX_ONNX_EP_VERBOSE=0

@REM Enable kernel profiling
set DEEPHI_PROFILING=0

@REM Remove fingerprint check
set XLNX_VART_SKIP_FP_CHK=TRUE

@REM Enable PP Kernels
set XLNX_USE_SHARED_CONTEXT=1

@REM Enable debug messages from VAIP custom ops
set DEBUG_DECODE_FITLER_BOX_CUSTOM_OP=0
set DEBUG_RESIZE_NORM_CUSTOM_OP=0
set DEBUG_DPU_CUSTOM_OP=0
set NUM_OF_DPU_RUNNERS=1

@REM Init
set "__BAT_FILE=%~0"
set "__BAT_PATH=%~dp0"
set "__BAT_NAME=%~nx0"
call :setESC
set ITERATIONS= 1
:: Get args
:GETOPTS
  if /I "%~1" == "--help" (
    set HELP=--help
    goto HELPER
  ) else if /I "%~1" == "--e2e" (
    set E2E=--e2e-model
    shift
  ) else if /I "%~1" == "--vai-ep" (
    set EXEC_PROVIDER=--vai-ep
    shift
  ) else if /I "%~1" == "--op-profile" (
    set OP_PROFILE= --operator-profile
    shift 
  ) else if /I "%~1" == "--e2e-profile" (
    set E2E_PROFILE= --e2e-profile
    shift 
  ) else if /I "%~1" == "--run-both-ep" (
    set EXEC_PROVIDER= --run-both
    shift 
  ) else if /I "%~1" == "--power-profile" (
    set POWER_PROFILE= --power-profile
    shift 
  ) else if /I "%~1" == "--img" (
    set "IMAGE=%~2"
    shift & shift
  ) else if /I "%~1" == "--iterations" (
    set "ITERATIONS=%~2"
    shift & shift
  ) else if /I "%~1" == "--voe-path" (
    set "PKG=%~2"
    shift & shift
  ) else if /I "%~1" == "" (
    goto RUN
  ) else (
    echo %ESC%[1;31m
    echo - ERROR: Unknown command line argument "%~1"
    echo %ESC%[0m
    goto HELPER
  )
if not (%1)==() goto GETOPTS

@REM XCLBIN and CONFIG paths
set VAIP_CONFIG=%PKG%\vaip_config.json
set XLNX_VART_FIRMWARE=%PKG%\1x4.xclbin

:: Run Detection
:RUN 
    if exist %CONDA_PREFIX%\Lib\site-packages\voe (
      echo - Running with Anaconda env python
      if defined EXEC_PROVIDER (
          if not defined PKG (
            echo - Provide the voe package path
          ) else (
             python run.py %HELP% %E2E% %EXEC_PROVIDER% %OP_PROFILE% %POWER_PROFILE% %E2E_PROFILE% --config %VAIP_CONFIG% --img %IMAGE% --iterations %ITERATIONS% 
          ) 
      ) else (
          python run.py %HELP% %E2E% %OP_PROFILE% %POWER_PROFILE% %E2E_PROFILE%  --img %IMAGE% --iterations %ITERATIONS% 
      )
    )
    goto :eof

:HELPER
    echo %ESC%[1;97m
    echo USAGE:
    echo   %__BAT_NAME% [Options]
    echo   ----------------------------------------------------------------------------------------
    echo     --help                    Shows this help
    echo     --voe-path                Path to voe package
    echo     --vai-ep                  Enable VitisAIExecutionProvider (Default: CPUExecutionProvider)
    echo     --e2e                     Run e2e model (Default: non-e2e model)
    echo     --op-profile              Enable Operator profiling(Default: False)
    echo     --e2e-profile             Enable e2e profiling (Default :False)
    echo     --run-both-ep             Run model on VitisAIExecutionProvider,CPUExecutionProvider 
    echo     --img                     Input image file path
    echo     --iterations              Number of iterations
    @REM echo     --power-profile           Enable power profiling (Default :False)
    echo   ----------------------------------------------------------------------------------------
    echo %ESC%[0m
    goto :eof

:setESC
    for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
    set ESC=%%b
    exit /B 0
    )
