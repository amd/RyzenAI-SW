@echo off
setlocal

REM Working variables
set XRT_PROG=xclbinutil

REM -- Find and call the loader
set XRT_LOADER=%~dp0unwrapped/loader.bat

REM -- Examine the options
set XRTWRAP_PROG_ARGS=
  :parseArgs
    if [%1] == [] (
      goto argsParsed
    ) else (
      REM Future option(s) example syntax
      REM  if [%1] == [-futureOption] ( 
      REM    shift
      REM  ) else (
      REM    if [%1] == [--futureOption] ( 
      REM    shift
      REM    ) else (

      REM Unknown option, must be associated with the program
      set XRTWRAP_PROG_ARGS=%XRTWRAP_PROG_ARGS% %1
      shift
    )
    goto parseArgs
  :argsParsed

if not exist %XRT_LOADER%  (
  echo ERROR: Could not find 64-bit loader executable.
  echo ERROR: %XRT_LOADER% does not exist.
  GOTO:EOF
)

%XRT_LOADER% -exec %XRT_PROG% %XRTWRAP_PROG_ARGS%

