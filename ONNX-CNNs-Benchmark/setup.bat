@echo off

setlocal enabledelayedexpansion

REM Check if Conda environment is activated
if "%CONDA_DEFAULT_ENV%" == "" (
    echo Conda environment is not activated. Please activate the ryzenai-release conda environment
    exit /b 1
)

REM Check if XLNX_VART_FIRMWARE environment variable is set
if not defined XLNX_VART_FIRMWARE (
    echo XLNX_VART_FIRMWARE is not set. Please ensure that you run the `install.bat` located in the ... \ryzen-ai-sw-1.0.1\ directory before proceeding.
    exit /b 1
)

REM Extract the directory path from XLNX_VART_FIRMWARE
set "source_dir=!XLNX_VART_FIRMWARE!\.."

REM Check if source directory exists
if not exist "!source_dir!\" (
    echo Source directory does not exist.
    exit /b 1
)

REM Create the destination directory .\onnxrt if it doesn't exist
if not exist ".\onnxrt\" mkdir ".\onnxrt"

REM Copy all .xclbin files to .\onnxrt\
copy "!source_dir!\*.xclbin" ".\onnxrt\"

REM add additional packages
pip install pandas matplotlib importlib-metadata psutil keyboard

echo setup operation completed.
exit /b 0
