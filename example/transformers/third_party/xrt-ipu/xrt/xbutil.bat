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


REM -- Find and call the loader
set XRT_LOADER=%~dp0unwrapped/loader.bat

if not exist %XRT_LOADER%  (
  echo ERROR: Could not find 64-bit loader executable.
  echo ERROR: %XRT_LOADER% does not exist.
  GOTO:EOF
)

%XRT_LOADER% -exec %XRT_PROG% %XRTWRAP_PROG_ARGS%

