@REM Build script for Dynamic Dispatch
@echo off
setlocal

call :setESC
goto :init

@REM Error out when the build type argument is missing
:missing_argument
    echo %ESC%[31m
    echo -- Error: Missing argument value for "--build"
    echo %ESC%[0m
    call :usage & goto :eof

@REM check if DOD Root exists
:missing_root
    echo %ESC%[31m
    echo -- Error: DOD Root path not set. To set, run 'setup.bat' OR 'setup.ps1'.
    echo %ESC%[0m
    call :usage & goto :eof

@REM check if XRT exists
:missing_xrt
    echo %ESC%[31m
    echo -- Error: XRT path not set.
    echo -- On CMD, set "XRT_DIR=<path/to/XRT>"
    echo -- On PowerShell, $env:XRT_DIR="<path/to/XRT>"
    echo %ESC%[0m
    call :usage & goto :eof

@REM Initialize scripts and options
:init
    set "__NAME=%~n0"
    set "__VERSION=1.0"
    set "__YEAR=2024"

    set "__BAT_FILE=%~0"
    set "__BAT_PATH=%~dp0"
    set "__BAT_NAME=%~nx0"

    set "OptHelp="
    set "OptVerbose="

    set "UnNamedArgument="
    set "BuildDir=build"
    set "BoostDir="
    set "CleanFirst="
    set "EnableTests="
    set "PerfProfile="
    set "ConfigureFresh="
    set "BuildType=Release"
    set "DODRoot=%DOD_ROOT%"
    set "XRT_DIR=%XRT_DIR%"

@REM Parse command line args
:parse
    @REM Parse all the command line arguments
    if "%~1"=="" goto :validate

    if /i "%~1"=="/?"                ( call :usage "%~2" & goto :end
    ) else if /i "%~1"=="-?"         ( call :usage "%~2" & goto :end
    ) else if /i "%~1"=="--help"     ( call :usage "%~2" & goto :end
    ) else if /i "%~1"=="/v"         ( set "OptVerbose=--verbose" & shift & goto :parse
    ) else if /i "%~1"=="-v"         ( set "OptVerbose=--verbose" & shift & goto :parse
    ) else if /i "%~1"=="--verbose"  ( set "OptVerbose=--verbose" & shift & goto :parse
    ) else if /i "%~1"=="/c"         ( set "CleanFirst=--clean-first" & shift & goto :parse
    ) else if /i "%~1"=="-c"         ( set "CleanFirst=--clean-first" & shift & goto :parse
    ) else if /i "%~1"=="--clean"    ( set "CleanFirst=--clean-first" & shift & goto :parse
    ) else if /i "%~1"=="/b"         ( set "BuildType=%~2" & shift & shift & goto :parse
    ) else if /i "%~1"=="-b"         ( set "BuildType=%~2" & shift & shift & goto :parse
    ) else if /i "%~1"=="--build"    ( set "BuildType=%~2" & shift & shift & goto :parse
    ) else if /i "%~1"=="/t"         ( set "EnableTests=-DENABLE_DD_TESTS=ON" & shift & goto :parse
    ) else if /i "%~1"=="-t"         ( set "EnableTests=-DENABLE_DD_TESTS=ON" & shift & goto :parse
    ) else if /i "%~1"=="--en-tests" ( set "EnableTests=-DENABLE_DD_TESTS=ON" & shift & goto :parse
    ) else if /i "%~1"=="/p"         ( set "PerfProfile=-DUNIT_TEST_PERF_EN=ON" & shift & goto :parse
    ) else if /i "%~1"=="-p"         ( set "PerfProfile=-DUNIT_TEST_PERF_EN=ON" & shift & goto :parse
    ) else if /i "%~1"=="--en-perf"  ( set "PerfProfile=-DUNIT_TEST_PERF_EN=ON" & shift & goto :parse
    ) else if /i "%~1"=="/f"         ( set "ConfigureFresh=--fresh" & shift & goto :parse
    ) else if /i "%~1"=="-f"         ( set "ConfigureFresh=--fresh" & shift & goto :parse
    ) else if /i "%~1"=="--fresh"    ( set "ConfigureFresh=--fresh" & shift & goto :parse
    ) else (
      echo %ESC%[31m
      echo -- Error: Invalid option "%~1"
      echo %ESC%[0m
      goto usage & goto end
    )
    shift
    goto :parse

@REM Validate required arguments
:validate
    if not defined BuildType call :missing_argument & goto :end
    if not defined DODRoot call :missing_root & goto :end
    if not defined XRT_DIR call :missing_xrt & goto :end
    if /i "%BuildType%" NEQ "Debug" (
        if /i "%BuildType%" NEQ "Release" (
            echo %ESC%[31m
            echo -- Error: Invalid Build Type: %BuildType%
            echo %ESC%[0m
            goto :usage
        )
    )

@REM Call CMake with all the options
:main
    echo %ESC%[1;32m
    if defined OptVerbose echo -- Verbosity: "On"
    if not defined OptVerbose echo -- Verbosity: "Off"
    if defined CleanFirst echo -- CleanFirst: "Yes"
    if not defined CleanFirst echo -- CleanFirst: "No"
    if defined ConfigureFresh echo -- Configure Fresh: "Yes"
    if not defined ConfigureFresh echo -- Configure Fresh: "No"
    echo -- BuildType: "%BuildType%"
    echo -- Configuring ...
    echo %ESC%[0m

    @REM Configure
    cmake -S . -B %BuildDir% -G "Visual Studio 17 2022" ^
      -DCMAKE_INSTALL_PREFIX=%BuildDir%\Release ^
      %PerfProfile% %EnableTests% %ConfigureFresh%

    if %ERRORLEVEL% NEQ 0 (
        echo %ESC%[31m
        echo -- Config failed: code %ERRORLEVEL%
        echo %ESC%[0m
        cd ..
    )

    @REM Build
    echo %ESC%[1;32m
    echo -- Building ...
    echo %ESC%[0m
    cmake --build %BuildDir% %CleanFirst% %OptVerbose% --config %BuildType% --target install
    if %ERRORLEVEL% NEQ 0 (
        echo %ESC%[31m
        echo -- Build failed: code %ERRORLEVEL%
        echo %ESC%[0m
        cd ..
    )

@REM Clean all things
:end
    call :cleanup
    exit /B

@REM Compiler name and version
:header
    echo - Build Dynamic Dispatch : %__VERSION%
    goto :eof

@REM Script Usage
:usage
    echo %ESC%[1;34m
    call :header
    echo.
    echo USAGE:
    echo   %__BAT_NAME% [Options]
    echo.
    echo   -- [Options] ------------------------------------------------------------
    echo.  /?, -?, --help       Shows this help
    echo.  /v, -v, --verbose    Shows detailed compile log
    echo.  /f, -f, --fresh      Reconfigure build, Must use if build configs has changed
    echo.  /b, -b, --build      Build type [Release, Debug] (Default: Debug)
    echo.  /c, -c, --clean      Clean first then build
    echo.  /t, -t, --en-tests   Enable tests
    echo.  /p, -p, --en-perf    Enable unit test performance profiling
    echo   -------------------------------------------------------------------------
    echo %ESC%[0m
    goto :eof

@REM The cleanup function is only really necessary if you
@REM are _not_ using SETLOCAL.
:cleanup
    set "__NAME="
    set "__VERSION="
    set "__YEAR="

    set "__BAT_FILE="
    set "__BAT_PATH="
    set "__BAT_NAME="

    set "OptHelp="
    set "OptVerbose="

    set "CleanFirst="
    set "BuildType="

    goto :eof

:setESC
    for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
    set ESC=%%b
    exit /B 0
    )
