REM # Clone the VART Aggregator Repo
cd C:\Users\Administrator\Desktop
git clone https://gitenterprise.xilinx.com/VitisAI/vai-rt.git
 
REM # Use HTTPS method of cloning VART repos
set PREFER_HTTPS_GIT_URL=1
 
REM # ENV for directing the install
set VAI_RT_WORKSPACE=C:\Users\Administrator\Desktop\vai-rt
set VAI_RT_PREFIX=%VAI_RT_WORKSPACE%\install
set VAI_RT_BUILD_DIR=%VAI_RT_WORKSPACE%\build
set XRT_TEST_PACKAGE_DIR=%VAI_RT_WORKSPACE%\test_package
 
REM # Let's go into our workspace, and create the build and install directory
cd %VAI_RT_WORKSPACE%
mkdir build
mkdir install
mkdir test_package
 
REM # ENV for prebuilt XRT zip file
set XRT_ZIP_PATH=\\atg-w-u-2\share\aaronn\install\ipu_stack\
set XRT_ZIP_FILE=test_package.zip
 
REM # Copy the zip locally
xcopy %XRT_ZIP_PATH%%XRT_ZIP_FILE% .
 
REM # Install prebuilt XRT into install directory
for /F %%I IN ('dir /b %XRT_ZIP_FILE%') DO ( tar -xf "%%I" -C %XRT_TEST_PACKAGE_DIR%  )
xcopy /s %XRT_TEST_PACKAGE_DIR%\xrt-ipu\ %VAI_RT_PREFIX%
 
REM # Magically Build All of VART with tvm-engine
python main.py --release_file release_file\tvm.txt --type release
