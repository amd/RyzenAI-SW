@echo off
setlocal

REM Working variables
set XRT_PROG=xbutil

REM -- Examine the options
set XRTWRAP_PROG_ARGS=
  :parseArgs
    if [%1] == [] (
      goto argsParsed
    ) else (
      REM New option
      if [%1] == [-new] ( 
        echo INFO: The 'new' option is only valid for the linux version of xbmgmt
        shift
      ) else (
      if [%1] == [--new] ( 
        echo INFO: The 'new' option is only valid for the linux version of xbmgmt
        shift
      ) else (
        REM Unknown option, must be associated with the program
        set XRTWRAP_PROG_ARGS=%XRTWRAP_PROG_ARGS% %1
        shift
      )
    ))
    goto parseArgs
  :argsParsed


REM -- Find the loader from the current directory. If it exists.
set XRT_LOADER=%~dp0unwrapped\loader.bat

REM -- Find loader from the PATH. If it exists.
FOR /F "tokens=* USEBACKQ" %%F IN (`where xbutil`) DO (
set XBUTIL_PATH=%%~dpF
)

REM -- If the loader is not found in the current directory use the PATH.
if not exist %XRT_LOADER%  (
  set XRT_LOADER=%XBUTIL_PATH%unwrapped\loader.bat
)

REM -- Loader is not within the current directory or PATH. All hope is lost.
if not exist %XRT_LOADER%  (
  echo ERROR: Could not find 64-bit loader executable.
  echo ERROR: %XRT_LOADER% does not exist.
  GOTO:EOF
)

%XRT_LOADER% -exec %XRT_PROG% %XRTWRAP_PROG_ARGS%

