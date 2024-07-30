@echo OFF
REM # ENV for directing the install
setlocal

set VAI_RT=%~dp0\vai-rt-build
set VAI_RT_WORKSPACE=%VAI_RT%\workspace
set VAI_RT_PREFIX=%VAI_RT%\install
set VAI_RT_BUILD_DIR=%VAI_RT%\build
set WITH_XCOMPILER=ON
set WITH_TVM_AIE_COMPILER=ON
set XRT_DIR=%THIRD_PARTY%\xrt-ipu\xrt

echo "Workspace -> %VAI_RT_WORKSPACE%"
echo "Build -> %VAI_RT_BUILD_DIR%"
echo "Install -> %VAI_RT_PREFIX%"

REM # Let's go into our workspace, and create the build and install directory
mkdir %VAI_RT_WORKSPACE%
mkdir %VAI_RT_PREFIX%
mkdir %VAI_RT_BUILD_DIR%

REM #Build All of VART with tvm-engine
python main.py --type Release --dev-mode --release_file release_file\latest_dev.txt

REM #Install
set "extension=.whl"  REM Change this to the desired extension

for %%f in ("%VAI_RT_BUILD_DIR%\onnxruntime\Release\Release\dist\*%extension%") do (
    echo Processing file: %%f
    pip install --upgrade --force-reinstall %%f
)

for %%f in ("%VAI_RT_BUILD_DIR%\vaip\onnxruntime_vitisai_ep\python\dist\*%extension%") do (
    echo Processing file: %%f
    pip install --upgrade --force-reinstall %%f
)

python ci\onnx_installer\installer.py

endlocal
:end