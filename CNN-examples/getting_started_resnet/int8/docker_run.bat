@echo off
REM ==============================================================================
REM AMD Quark Quantization Pipeline Runner (Windows Docker)
REM ==============================================================================
setlocal enabledelayedexpansion

echo ==================================================================
echo  Building Docker image: ryzenai-resnet-quark
echo ==================================================================
docker build -t ryzenai-resnet-quark:latest -f Dockerfile .
if %errorlevel% neq 0 (
    echo Error: Docker build failed!
    exit /b %errorlevel%
)

echo ==================================================================
echo  Running Quantization Pipeline in Container
echo ==================================================================
if not exist "models" mkdir models
if not exist "data" mkdir data

docker run --rm ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\data:/app/data" ^
    ryzenai-resnet-quark:latest

if %errorlevel% neq 0 (
    echo Error: Quantization failed!
    exit /b %errorlevel%
)

echo ==================================================================
echo  SUCCESS!
echo  Quantized model generated at: %cd%\models\resnet_quantized.onnx
echo  You can now run predict.py in your ryzen-ai conda environment.
echo ==================================================================

