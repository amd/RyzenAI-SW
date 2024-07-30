@REM echo "Hello"
set install_dir=%1
echo %install_dir%

%install_dir%\tests\cpp_tests.exe --gtest_filter=*PSH*
if errorlevel 1 (exit /B 1 %errorlevel%)

%install_dir%\tests\cpp_tests.exe --gtest_filter=*PSF*
if errorlevel 1 (exit /B 1 %errorlevel%)

%install_dir%\tests\cpp_tests.exe --gtest_filter=*PSJ*
if errorlevel 1 (exit /B 1 %errorlevel%)

%install_dir%\tests\cpp_tests.exe --gtest_filter=*MLADF*
if errorlevel 1 (exit /B 1 %errorlevel%)

%install_dir%\tests\cpp_tests.exe --gtest_filter=*EXPTL*
if errorlevel 1 (exit /B 1 %errorlevel%)

@REM python tests\cpp\single_matmul\model.py --dtype int8
@REM %install_dir%\tests\test_single.exe  test_matmul_int8/model_matmul1_meta_int8.json
@REM if errorlevel 1 (exit /B 1 %errorlevel%)

@REM python tests\cpp\single_mha\model.py
@REM %install_dir%\tests\test_mha.exe test_mha\mha.onnx.json
@REM if errorlevel 1 (exit /B 1 %errorlevel%)

@REM python tests\cpp\single_add\model.py
@REM %install_dir%\tests\test_add.exe test_eltadd\add.onnx.json
@REM if errorlevel 1 (exit /B 1 %errorlevel%)

@REM python tests\cpp\single_lrn\model.py
@REM %install_dir%\tests\test_lrn.exe test_lrn\lrn.onnx.json
@REM if errorlevel 1 (exit /B 1 %errorlevel%)

@REM python tests\cpp\single_matmuladdgelu\model.py
@REM %install_dir%\tests\test_mataddgelu.exe test_matmuladdgelu\model_mataddgelu_meta.json
@REM if errorlevel 1 (exit /B 1 %errorlevel%)

@REM python tests\cpp\single_mha\model.py
@REM %install_dir%\tests\test_mha.exe test_mha\mha.onnx.json
@REM if errorlevel 1 (exit /B 1 %errorlevel%)

@REM python tests\cpp\matmul6\model.py 12
@REM %install_dir%\tests\test_matmul6.exe test_matmul6\Matmuls_6x12_for_fusion.onnx.json
@REM if errorlevel 1 (exit /B 1 %errorlevel%)

echo "Tests Finished"
