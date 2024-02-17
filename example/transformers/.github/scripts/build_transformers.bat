REM SPDX-License-Identifier: Apache-2.0
@echo off
setlocal enabledelayedexpansion
set WORKSPACE=%REPO_PATH%

set CONDA_PATH=%LOCALAPPDATA%\anaconda3
REM Build transformers
set TRANSFORMERS_CONDA_ENV_PATH=%CONDA_PATH%\envs\%TRANSFORMERS_CONDA_ENV_NAME%
call %CONDA_PATH%\Scripts\activate.bat %TRANSFORMERS_CONDA_ENV_PATH%
pip uninstall -y RyzenAI ryzenai-torch-cpp

cd %WORKSPACE%
set TRANSFORMERS_BUILD_DIR=build
mkdir %TRANSFORMERS_BUILD_DIR%
set PATH=%XRT_PATH%\xrt;%PATH%
echo PATH %PATH%
echo Run setup.bat
call setup.bat
mkdir %WORKSPACE%\tests_release\
pushd %TRANSFORMERS_BUILD_DIR%
REM Build dependancy libraries and c++ tests
cmake ..
cmake --build . --config=Release
copy tests\cpp\Release\cpp_tests.exe %WORKSPACE%\tests_release\
popd
REM Build wheels
pushd ops\cpp
python setup.py bdist_wheel
IF errorlevel 1 (exit /B %errorlevel%)
copy dist\*.whl %WORKSPACE%
IF errorlevel 1 (exit /B %errorlevel%)
popd
pushd ops\torch_cpp
python setup.py bdist_wheel
IF errorlevel 1 (exit /B %errorlevel%)
copy dist\*.whl %WORKSPACE%
IF errorlevel 1 (exit /B %errorlevel%)
popd
REM Loop through each .whl file and install
for %%f in (*.whl) do (
    set "wheel_file=%%~nf"
    echo Installing !wheel_file!...
    pip install "!wheel_file!.whl"
    IF errorlevel 1 (exit /B %errorlevel%)
)

@REM DLLs are checked into the release_* branch, no need to build.
@REM pushd %WORKSPACE%\tools
@REM python generate_dll.py
@REM IF errorlevel 1 (exit /B %errorlevel%)
@REM popd
