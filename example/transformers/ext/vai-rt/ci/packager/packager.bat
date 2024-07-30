::
:: Copyright 2022-2023 Advanced Micro Devices Inc.
::
:: Licensed under the Apache License, Version 2.0 (the "License"); you may not
:: use this file except in compliance with the License. You may obtain a copy of
:: the License at
::
:: http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
:: WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
:: License for the specific language governing permissions and limitations under
:: the License.
::

@echo off
set VAI_RT_BUILD_DIR=%1
set VAI_RT_PREFIX=%2
set VAI_RT_PACKAGER=%3
set WORKSPACE=%4
set RELEASE_FILE=%5
set bin_1x4_url=%6
set bin_5x4_url=%7
set INSTALL_RELEASE=%8
set TVM_XCLBINS_URL=%9

set JFROG_CLI="C:\\Users\\xbuild\\jfrog-cli.exe"

echo "%VAI_RT_BUILD_DIR%"
echo "%VAI_RT_PREFIX%"
echo "%VAI_RT_PACKAGER%"
echo "%WORKSPACE%"
echo "%RELEASE_FILE%"
echo "%bin_1x4_url%"
echo "%bin_5x4_url%"
echo "%INSTALL_RELEASE%"

copy "%VAI_RT_BUILD_DIR%"\\onnxruntime\\Release\\Release\\onnxruntime_perf_test.exe "%VAI_RT_PREFIX%"\\bin
python "%VAI_RT_PACKAGER%"\\package.py "%VAI_RT_PREFIX%" "%WORKSPACE%"\\onnx-rt "%VAI_RT_PACKAGER%"\\package_list.txt
if not "%errorlevel%" == "0" (
    echo "failed to run python package for onnx-rt"
    exit 1
)
copy "%VAI_RT_BUILD_DIR%"\\onnxruntime\\Release\\Release\\dist\\onnxruntime_vitisai*.whl "%WORKSPACE%"\\onnx-rt
copy "%VAI_RT_BUILD_DIR%"\\vaip\\onnxruntime_vitisai_ep\\python\\dist\\voe*.whl "%WORKSPACE%"\\onnx-rt
copy release_file\\"%RELEASE_FILE%" "%WORKSPACE%"\\onnx-rt
copy "%WORKSPACE%"\\ci\\onnx_installer\\installer.py "%WORKSPACE%"\\onnx-rt
tar -acf onnx-rt.zip onnx-rt
if not "%errorlevel%" == "0" (
    echo "failed to run tar command for onnx-rt"
    exit 1
)

python "%VAI_RT_PACKAGER%"\\package.py "%VAI_RT_PREFIX%" "%WORKSPACE%"\\"%INSTALL_RELEASE%" "%VAI_RT_PACKAGER%"\\package_release.txt
if not "%errorlevel%" == "0" (
    echo "failed to run python package for install_release"
    exit 1
)
copy "%VAI_RT_BUILD_DIR%"\\onnxruntime\\Release\\Release\\dist\\onnxruntime_vitisai*.whl "%WORKSPACE%"\\"%INSTALL_RELEASE%"
copy "%VAI_RT_BUILD_DIR%"\\vaip\\onnxruntime_vitisai_ep\\python\\dist\\voe*.whl "%WORKSPACE%"\\"%INSTALL_RELEASE%"
for %%F in ("%WORKSPACE%"\\"%INSTALL_RELEASE%"\\"voe*.whl") do set "foldername=%%~nF"
echo "foldername %foldername%"
mkdir "%WORKSPACE%"\\"%INSTALL_RELEASE%"\\"%foldername%"
copy "%WORKSPACE%"\\"%INSTALL_RELEASE%"\\bin\\onnxruntime_vitisai_ep.dll "%WORKSPACE%"\\"%INSTALL_RELEASE%"\\"%foldername%"
copy "%WORKSPACE%"\\"%INSTALL_RELEASE%"\\bin\\onnxruntime.dll "%WORKSPACE%"\\"%INSTALL_RELEASE%"\\"%foldername%"
tar -acf "%foldername%.zip" "%WORKSPACE%"\\"%INSTALL_RELEASE%"\\"%foldername%"
if not "%errorlevel%" == "0" (
    echo "failed to run tar command for %foldername%"
    exit 1
)
copy "%WORKSPACE%"\\"%INSTALL_RELEASE%"\\bin\\vaip_config.json "%WORKSPACE%"\\"%INSTALL_RELEASE%"
rmdir /s /q "%WORKSPACE%"\\"%INSTALL_RELEASE%"\\bin

mkdir "%WORKSPACE%"\\unzippedFiles
mkdir "%WORKSPACE%"\\unzippedFiles\\bin_1x4
cd "%WORKSPACE%"\\unzippedFiles\\bin_1x4
"%JFROG_CLI%" rt dl --flat "%bin_1x4_url%"/binaries.tar.gz
"%JFROG_CLI%" rt dl --flat "%bin_1x4_url%"/info.txt
tar -zxvf binaries.tar.gz -C "%WORKSPACE%"\\unzippedFiles\\bin_1x4
if not "%errorlevel%" == "0" (
    echo "failed to run tar command for bin_1x4"
    exit 1
)
cd ..
mkdir bin_5x4
cd bin_5x4
"%JFROG_CLI%" rt dl --flat "%bin_5x4_url%"/binaries.tar.gz
"%JFROG_CLI%" rt dl --flat "%bin_5x4_url%"/info.txt
tar -zxvf binaries.tar.gz -C "%WORKSPACE%"\\unzippedFiles\\bin_5x4
if not "%errorlevel%" == "0" (
    echo "failed to run tar command for bin_5x4"
    exit 1
)

cd ..
mkdir tvm-xclbins
cd tvm-xclbins
C:\\Users\\xbuild\\jfrog-cli.exe rt dl --flat "%TVM_XCLBINS_URL%"/tvm-xclbins.tar.gz
C:\\Users\\xbuild\\jfrog-cli.exe rt dl --flat "%TVM_XCLBINS_URL%"/info.txt
tar -zxf tvm-xclbins.tar.gz -C "%WORKSPACE%"\\unzippedFiles\\tvm-xclbins

if not "%errorlevel%" == "0" (
    echo "failed to run tar command for tvm-xclbins"
    exit 1
)

cd ../..
copy "%WORKSPACE%"\\unzippedFiles\\bin_1x4\\binaries\\1x4.xclbin "%WORKSPACE%"\\"%INSTALL_RELEASE%"
copy "%WORKSPACE%"\\unzippedFiles\\bin_5x4\\binaries\\5x4.xclbin "%WORKSPACE%"\\"%INSTALL_RELEASE%"
copy "%WORKSPACE%"\\unzippedFiles\\tvm-xclbins\\tvm-xclbins\\aieml_gemm_asr\\asr4x2.xclbin "%WORKSPACE%"\\"%INSTALL_RELEASE%"
copy "%WORKSPACE%"\\unzippedFiles\\tvm-xclbins\\tvm-xclbins\\aieml_gemm_asr_qdq\\asr_qdq_4x2.xclbin "%WORKSPACE%"\\"%INSTALL_RELEASE%"
copy "%WORKSPACE%"\\unzippedFiles\\tvm-xclbins\\tvm-xclbins\\aieml_gemm_vm_phx_4x4\\gemm_vm_phx_4x4.xclbin "%WORKSPACE%"\\"%INSTALL_RELEASE%"
copy "%WORKSPACE%"\\unzippedFiles\\bin_1x4\\info.txt "%WORKSPACE%"\\info_1x4.txt
copy "%WORKSPACE%"\\unzippedFiles\\bin_5x4\\info.txt "%WORKSPACE%"\\info_5x4.txt
copy "%WORKSPACE%"\\unzippedFiles\\tvm-xclbins\\info.txt "%WORKSPACE%"\\info_tvm.txt
copy "%WORKSPACE%"\\ci\\onnx_installer\\installer.py "%WORKSPACE%"\\"%INSTALL_RELEASE%"

rmdir /s /q "%WORKSPACE%"\\unzippedFiles
tar -acf "%INSTALL_RELEASE%".zip "%INSTALL_RELEASE%"
if not "%errorlevel%" == "0" (
    echo "failed to run tar command for install_release"
    exit 1
)
mkdir xrt
copy "%VAI_RT_PREFIX%"\\xrt\\xrt_phxcore.dll "%WORKSPACE%"\\xrt
copy "%VAI_RT_PREFIX%"\\xrt\\xrt_coreutil.dll "%WORKSPACE%"\\xrt
tar -acf xrt.zip xrt
if not "%errorlevel%" == "0" (
    echo "failed to run tar command for xrt"
    exit 1
)
exit 0