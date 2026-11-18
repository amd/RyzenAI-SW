@echo off
rem Quantize convnext_fp32.onnx with the Entropy recipe (no fine-tuning) -> output\ryzen_model\model.onnx
rem output\ryzen_model\ holds the pre-built model. Pass --force (or set FORCE=1) to rebuild; the old output\ is moved to output_prev\.
setlocal
cd /d "%~dp0"
if not exist convnext_fp32.onnx (
    echo convnext_fp32.onnx not found. Run "python export_fp32.py" first.
    exit /b 1
)
if /i "%~1"=="--force" set "FORCE=1"
if exist output\ryzen_model\model.onnx if not "%FORCE%"=="1" (
    echo output\ryzen_model\model.onnx already exists: this is the pre-built quantized model.
    echo To rebuild anyway, run "run_quant.bat --force". The existing output\ is moved to output_prev\.
    exit /b 1
)
if not exist calibration_dataset\calib_list.txt (
    echo calibration_dataset\calib_list.txt not found.
    exit /b 1
)
set "CALIB_COUNT=0"
if exist calibration_dataset\images (
    for /f %%c in ('dir /b /a-d "calibration_dataset\images\*.png" 2^>nul ^| find /c /v ""') do set "CALIB_COUNT=%%c"
)
if not "%CALIB_COUNT%"=="200" (
    echo ConvNext calibration dataset is missing or incomplete. Generating 200 images...
    python calibration_dataset\prepare_calibration_dataset.py
    if errorlevel 1 exit /b 1
)
if exist output (
    if exist output_prev rmdir /s /q output_prev
    move output output_prev >nul
)
for %%d in (cache tmp_olive_custom) do if exist %%d rmdir /s /q %%d
set "PYTHONPATH=%~dp0olive_ryzen;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"
olive run --config olive_ryzen_config.json --package-config olive_ryzen\olive_ryzen_config.json
exit /b %ERRORLEVEL%
