:: Get current directory
set current_dir=%cd%

:: Go to Desktop directory under admin
cd C:\Users\Transformers
mkdir ipu_stack_rel_silicon

:: Mount network path as a drive Z:\
subst Z: \\atg-w-u-2\share\aaronn\install\ipu_stack\
 
:: ENV for prebuilt XRT zip file
set XRT_ZIP_PATH=Z:\
set XRT_ZIP_FILE=ipu_stack_rel_silicon.zip
set XRT_FOLDER=C:\Users\Transformers\ipu_stack_rel_silicon
 
:: Copy the zip locally, this can take a while, go use the restroom
xcopy %XRT_ZIP_PATH%%XRT_ZIP_FILE% .
 
:: Install prebuilt XRT into install directory
for /F %I IN ('dir /b %XRT_ZIP_FILE%') DO ( tar -xf "%I" -C %XRT_FOLDER% )
 
:: Lets get the test_package now
set XRT_ZIP_FILE=test_package.zip
 
:: Copy the zip locally, this can take a while, go use the restroom
xcopy %XRT_ZIP_PATH%%XRT_ZIP_FILE% .
 
:: Install prebuilt XRT into install directory
for /F %I IN ('dir /b %XRT_ZIP_FILE%') DO ( tar -xf "%I" -C %XRT_FOLDER% )
 
:: Install Driver, this should also remove the old one
cd %XRT_FOLDER%
amd_install_kipudrv.bat

:: Once installed, go back to original directory
cd %current_dir%

set XRT_PATH=C:\Users\Transformers\ipu_stack_rel_silicon\xrt-ipu
