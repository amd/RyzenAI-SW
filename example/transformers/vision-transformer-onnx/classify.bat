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
@echo OFF
setlocal
call :setESC

set "__NAME=%~n0"
set "__VERSION=1.0"
set "__YEAR=2023"
set "__BAT_FILE=%~0"
set "__BAT_PATH=%~dp0"
set "__BAT_NAME=%~nx0"

@REM Disable Cache and always compile
set XLNX_ENABLE_CACHE=0

@REM Use CPU runner
set USE_CPU_RUNNER=0
set VAIP_COMPILE_RESERVE_CONST_DATA=0

@REM Enable Verbosity
set XLNX_ONNX_EP_VERBOSE=0
set XLNX_ENABLE_DUMP_XIR_MODEL=0
set XLNX_ENABLE_DUMP_ONNX_MODEL=0
set ENABLE_SAVE_ONNX_MODEL=0

set DEBUG_DPU_CUSTOM_OP=0
set DEBUG_GEMM_CUSTOM_OP=0
set DEBUG_GRAPH_RUNNER=0

@REM Enable Multi-DPU flow
set XLNX_USE_SHARED_CONTEXT=1

@REM Enable DPU profiling
set DEEPHI_PROFILING=0

@REM Repo paths
set TRANSFORMERS_REPO=%PYTORCH_AIE_PATH%
set XLNX_VART_FIRMWARE=%TRANSFORMERS_REPO%\xclbin/phx/1x4.xclbin
set XLNX_VART_SKIP_FP_CHK=TRUE
set TVM_MODULE_PATH=%TRANSFORMERS_REPO%\dll\phx\qlinear\asr\libGemmQnnAie_8x2048_2048x2048.dll

set EP=vai
set RUNNER=dpu
set ITERS=1

:: Get args
:GETOPTS
  if /I "%~1" == "" (
    goto :CHECK
  ) else if /I "%~1" == "--help" (
    goto :HELPER
  ) else if /I "%~1" == "--model" (
    set "MODEL=%~2" & shift & shift
  ) else if /I "%~1" == "--img" (
    set "IMG=%~2" & shift & shift
  ) else if /I "%~1" == "--config" (
    set "CONFIG=--config %~2" & shift & shift
  ) else if /I "%~1" == "--ep" (
    set "EP=%~2" & shift & shift
  ) else if /I "%~1" == "--en-cache" (
    set "XLNX_ENABLE_CACHE=1"
    shift
  ) else if /I "%~1" == "--cpu-runner" (
    set "USE_CPU_RUNNER=1"
    set "VAIP_COMPILE_RESERVE_CONST_DATA=1"
    set "RUNNER=cpu"
    shift
  ) else if /I "%~1" == "--log-to-file" (
    set "LOG_TO_FILE=Yes" & shift
  ) else if /I "%~1" == "--iters" (
    set "ITERS=%~2" & shift & shift
  ) else (
    echo %ESC%[1;31m
    echo - ERROR: Unknown command line argument "%~1"
    echo %ESC%[0m
    goto :eof
  )
  if not (%1)==() goto GETOPTS

@REM Log directory, create if it doesn't exists
:CREATE_LOG_DIR
  set "LOGDIR=console_logs"
  if not exist %LOGDIR%\ (
    mkdir %LOGDIR%
  )

@REM Check for arg validity
:CHECK
  if not defined IMG (
    echo %ESC%[1;31m
    echo - ERROR: Missing command line argument "--img"
    echo %ESC%[0m
    call :HELPER
    goto :eof
  )
  if not defined MODEL (
    echo %ESC%[1;31m
    echo - ERROR: Missing command line argument "--model"
    echo %ESC%[0m
    call :HELPER
    goto :eof
  )

echo -- TRANSFORMERS_REPO = %TRANSFORMERS_REPO%

:RUN
  for %%F in (%MODEL%) do (
    if defined LOG_TO_FILE (
      if "%EP%" == "vai" (
        echo   -- VAI-EP Console Log file: %LOGDIR%\%%~nxF.%EP%.%RUNNER%
        python classification\test.py --model %MODEL% --ep %EP% %CONFIG% ^
        --img %IMG% --iters %ITERS% > "%LOGDIR%\%%~nxF.%EP%.%RUNNER%" 2>&1
      ) else (
        echo   -- CPU-EP Console Log file: %LOGDIR%\%%~nxF.%EP%
        python classification\test.py --model %MODEL% --ep %EP% %CONFIG% ^
        --img %IMG% --iters %ITERS% > "%LOGDIR%\%%~nxF.%EP%" 2>&1
      )
    ) else (
      python classification\test.py --model %MODEL% --ep %EP% %CONFIG% ^
      --img %IMG% --iters %ITERS%
    )
  )
  goto :eof

:header
  echo %ESC%[3;96m%__NAME% - Version: %__VERSION%, %__YEAR% %ESC%[0m
  goto :eof

:HELPER
  @REM Script file name/path
  call :header
  echo USAGE:
  echo %ESC%[3;96m  %__BAT_NAME% [Options] %ESC%[0m
  echo  -----------------------------------------------------------------------
  echo  --help        Shows this help
  echo  --model       Specify onnx model path
  echo  --img         Specify input image file path
  echo  --config      Specify config file path (Optional)
  echo  --log-to-file Enable console log dump to file (Optional)
  echo  --en-cache    Enable cache directory, reuse cached data (Optional)
  echo  --cpu-runner  Run CPU runner instead of DPU runner in VAI EP (Optional)
  echo  --iters       Number of iterations for session run (Default=1)
  echo  -----------------------------------------------------------------------
  goto :eof

:setESC
    for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
    set ESC=%%b
    exit /B 0
    )
