@echo off
setlocal

REM SPDX-License-Identifier: Apache-2.0
REM Copyright (C) 2019-2021 Xilinx, Inc. All rights reserved.

REM Working Variables
set XRT_EXEC=""
set XRT_UNWRAPPED_DIR=%~dp0

REM -- Examine the options
REM Warning: Do not put any echo statements in the "IF" blocks.  Doing so
REM          will result in expansion issues. e.g. the variable will not be set
set XRT_LOADER_ARGS=
  :parseArgs
    if [%1] == [] (
      goto argsParsed
    ) else (
      REM exec option
      if [%1] == [-exec] (
        set XRT_EXEC=%2
        shift
        shift
      ) else (
      if [%1] == [--exec] (
        set XRT_EXEC=%2
        shift
        shift
      ) else (
        REM Unknown option, must be associated with the program
        set XRT_LOADER_ARGS=%XRT_LOADER_ARGS% %1
        shift
      )
    ))
    goto parseArgs
  :argsParsed

REM -- Check to see if the given executable exists
if [%XRT_EXEC%] == [] (
  echo ERROR: The -exec option was not specified.
  GOTO:EOF
)

set XRT_PROG_UNWRAPPED=%XRT_UNWRAPPED_DIR%%XRT_EXEC%.exe
if not exist %XRT_PROG_UNWRAPPED% (
  echo ERROR: Could not find -exec program: %XRT_EXEC%
  echo ERROR: %XRT_PROG_UNWRAPPED% does not exist"
  GOTO:EOF
)

REM -- Find the setup script and configure environment
set XRT_SETUP_SCRIPT=%XRT_UNWRAPPED_DIR%..\setup.bat

if not exist %XRT_SETUP_SCRIPT% (
  echo ERROR: Could not find XRT setup script.
  echo ERROR: %XRT_SETUP_SCRIPT% does not exist.
  GOTO:EOF
)

call %XRT_SETUP_SCRIPT% > NUL

REM -- Execute the wrapped program
call %XRT_PROG_UNWRAPPED% %XRT_LOADER_ARGS%
