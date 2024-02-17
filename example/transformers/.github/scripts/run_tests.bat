REM SPDX-License-Identifier: Apache-2.0
@echo off
setlocal enabledelayedexpansion

set CONDA_PATH=%LOCALAPPDATA%\anaconda3
REM Build transformers
set TRANSFORMERS_CONDA_ENV_PATH=%CONDA_PATH%\envs\%TRANSFORMERS_CONDA_ENV_NAME%
call %CONDA_PATH%\Scripts\activate.bat %TRANSFORMERS_CONDA_ENV_PATH%

set PATH=%XRT_PATH%\xrt;%PATH%
echo PATH %PATH%

cd %REPO_PATH%\tests\python
echo Run setup.bat
call setup.bat

pytest --num_dlls 1 --num_workers 1 test_qlinear.py
RET1=%errorlevel%
pytest --num_dlls 2 --num_workers 1 test_qlinear.py
RET2=%errorlevel%
pytest --num_dlls 1 --num_workers 2 test_qlinear.py
RET3=%errorlevel%
pytest --num_dlls 2 --num_workers 2 test_qlinear.py
RET4=%errorlevel%

@REM C++ File needs to be prebuilt and placed in this directory
cd %REPO_PATH%\tests\cpp\Release
cpp_tests.exe
RET5=%errorlevel%


echo "Return status: %RET1% %RET2% %RET3% %RET4% %RET5%"
if %RET1% neq 0 (exit /b %RET1%)
if %RET2% neq 0 (exit /b %RET2%)
if %RET3% neq 0 (exit /b %RET3%)
if %RET4% neq 0 (exit /b %RET4%)
if %RET5% neq 0 (exit /b %RET5%)
echo "All tests passed!"


