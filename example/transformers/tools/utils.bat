@REM Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.

@REM This file contains common helper functions that are used across many
@REM batch files in this repository.

@echo off
CALL %*
GOTO :EOF

:_AddPathToVar
@REM Add a path-like string to an environment variable if it's not already
@REM present in the variable. By default, values are added to the front unless
@REM the appendVar variable is set to true prior to calling this.
@REM
@REM This function is not meant to be used directly. Use AppendPathToVar or
@REM PrependPathToVar as needed.

IF not defined appendVar SET "appendVar=false"
IF not "%appendVar%"=="true" SET "appendVar=false"
CALL echo %%%~2%%%> tmp.txt
@REM work around Batch files limit around 1021 chars in set \P
for %%A in (tmp.txt) do for /f "usebackq delims=" %%B in ("%%A") do (
  SET "VAR=%%B"
  GOTO :var_set
)
:var_set

DEL /Q tmp.txt
IF not "%VAR%" == "ECHO is off." GOTO exists

path|find /i "%~1" >nul || SET "%~2=%~1"
Exit /b 0

:exists
IF "%appendVar%"=="true" GOTO _append
path|find /i "%~1" >nul || SET "%~2=%~1;%VAR%"
Exit /b 0

:_append
path|find /i "%~1" >nul || SET "%~2=%VAR%;%~1"
Exit /b 0

:AppendPathToVar
@REM Add a path-like string to an environment variable if it's not already
@REM present in the variable. Values are added to the front.
@REM
@REM Example usage:
@REM    - Add %PWD%ops\python to the front of PYTHONPATH
@REM        call ./tools/utils.bat :AppendPathToVar %PWD%ops\python PYTHONPATH
SET "appendVar=true"
GOTO _AddPathToVar

:PrependPathToVar
@REM Add a path-like string to an environment variable if it's not already
@REM present in the variable. Values are added to the end.
@REM
@REM Example usage:
@REM    - Add %PWD%ops\python to the end of PYTHONPATH
@REM        call ./tools/utils.bat :PrependPathToVar %PWD%ops\python PYTHONPATH
SET "appendVar=false"
GOTO _AddPathToVar

:ExecuteStep
@REM Execute a command and track the error code to exit appropriately.
@REM
@REM Example usage:
@REM    - call <path to utils.bat> :ExecuteStep "<command>"
@REM        call ./tools/utils.bat :ExecuteStep "pip install black"
  %~1
  IF errorlevel 1 (echo %~1 failed! & exit /B 1 %errorlevel)
  EXIT /B 0

:EchoExecuteStep
@REM Execute a command and track the error code to exit appropriately. Also
@REM print the command before execution.
@REM
@REM Example usage:
@REM    - call <path to utils.bat> :EchoExecuteStep "<command>"
@REM        call ./tools/utils.bat :EchoExecuteStep "pip install black"
  echo Executing %~1
  GOTO :ExecuteStep %~1
