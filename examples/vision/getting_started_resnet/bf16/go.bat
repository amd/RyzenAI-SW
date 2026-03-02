::
:: This script runs the entire example end-to-end:
::  - Exports and compiles the ONNX model
::  - Tests the compiled model using a Python script
::  - Builds the C++ application
::  - Runs the C++ application with the precompiled model
::

@echo off

cd %~dp0

if "%CONDA_PREFIX%" == "" echo CONDA_PREFIX not set. This script must be executed from within the RyzenAI conda environment. & goto :error
pip install -r requirements.txt

:: Export and compile the ONNX model
call precompile_model.bat 
:: Test the compiled model using a Python script
call test_model.bat 

cd app
:: Build the C++ application
call compile.bat
:: Run the C++ application with the precompiled model
call run.bat

cd %~dp0


