@REM Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.

@echo off
call %*
goto :EOF

:BuildAndInstallWheels
@REM Build wheels into the working directory and install them
@REM
@REM Example usage:
@REM    - call <path to steps.bat> :BuildAndInstallWheels
  del *.whl
  call tools\utils.bat :EchoExecuteStep "pip wheel ./ops/cpp --no-deps"
  if errorlevel 1 (exit /B 1 %errorlevel%)
  call tools\utils.bat :EchoExecuteStep "pip wheel ./ops/torch_cpp --no-deps"
  if errorlevel 1 (exit /B 1 %errorlevel%)
  echo All custom packages built successfully!
  for %%f in (*.whl) do (
    pip install "%%f"
    if errorlevel 1 (exit /B %errorlevel%)
  )
  echo All custom packages installed successfully!
  exit /B 0

:BuildTransformers
@REM Build the transformers repository with CMake, deleting the existing build
@REM directory, if it exists.
@REM
@REM Example usage:
@REM    - call <path to steps.bat> :BuildTransformers
  echo Building Transformers
  rmdir /S /Q "%cd%\build"
  call tools\utils.bat :EchoExecuteStep "cmake -B build"
  if errorlevel 1 (exit /B 1 %errorlevel%)
  call tools\utils.bat :EchoExecuteStep "cmake --build build --config Release"
  if errorlevel 1 (exit /B 1 %errorlevel%)
  echo Transformers compiled successfully!
  exit /B 0
