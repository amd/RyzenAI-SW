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
set "__YEAR=2024"
set "__BAT_FILE=%~0"
set "__BAT_PATH=%~dp0"
set "__BAT_NAME=%~nx0"

@REM Disable Cache and always compile
set XLNX_ENABLE_CACHE=0

@REM Enable Verbosity
set XLNX_ONNX_EP_VERBOSE=0
set XLNX_ENABLE_DUMP_ONNX_MODEL=0
set ENABLE_SAVE_ONNX_MODEL=0

@REM Enable shared context
set XLNX_USE_SHARED_CONTEXT=0

@REM Defaults
set "DEVICE=stx"
set "EP=cpu"
set "ITERS=1"
set "SAVE_AS_NPZ="

:: Get args
:GETOPTS
  if /I "%~1" == "" (
    goto :CHECK
  ) else if /I "%~1" == "--help" (
    goto :HELPER
  ) else if /I "%~1" == "--model" (
    set "MODEL=%~2" & shift & shift
  ) else if /I "%~1" == "--in-data-dir" (
    set "IN_DATA_DIR=%~2" & shift & shift
  ) else if /I "%~1" == "--out-data-dir" (
    set "OUT_DATA_DIR=--out-data-dir %~2" & shift & shift
  ) else if /I "%~1" == "--ep" (
    set "EP=%~2" & shift & shift
  ) else if /I "%~1" == "--config" (
    set "CONFIG=--config %~2" & shift & shift
  ) else if /I "%~1" == "--xclbin" (
    set "XCLBIN=--xclbin %~2" & shift & shift
  ) else if /I "%~1" == "--en-cache" (
    set "XLNX_ENABLE_CACHE=1"
    shift
  ) else if /I "%~1" == "--npz" (
    set "SAVE_AS_NPZ=--npz"
    shift
  ) else if /I "%~1" == "--iters" (
    set "ITERS=%~2" & shift & shift
  ) else (
    echo %ESC%[1;31m
    echo - ERROR: Unknown command line argument "%~1"
    echo %ESC%[0m
    goto :eof
  )
  if not (%1)==() goto GETOPTS

@REM Check for arg validity
:CHECK
  if not defined IN_DATA_DIR (
    echo %ESC%[1;31m
    echo - ERROR: Missing command line argument "--in-data-dir"
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

:RUN
  for %%F in (%MODEL%) do (
      python infer.py --model %MODEL% --in-data-dir %IN_DATA_DIR% ^
      --ep %EP% %CONFIG% %XCLBIN% %OUT_DATA_DIR% ^
      --iters %ITERS% %SAVE_AS_NPZ%
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
  echo  --help          Shows this help
  echo  --model         Specify onnx model path
  echo  --in-data-dir   Specify input data directory
  echo  --ep            Specify execution provider (Default: cpu) (Optional)
  echo  --config        Specify config file path (needed to vai-ep)
  echo  --xclbin        Specify XCLBIN file path (needed to vai-ep)
  echo  --out-data-dir  Specify output data directory
  echo  --npz           Save output files as .npz (Default: .raw) (Optional)
  echo  --en-cache      Enable cache directory, reuse cached data (Optional)
  echo  --iters         Number of iterations for session run (Default=1) (Optional)
  echo  -----------------------------------------------------------------------
  goto :eof

:setESC
    for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
    set ESC=%%b
    exit /B 0
    )
