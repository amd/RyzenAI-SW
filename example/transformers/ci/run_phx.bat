@REM Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.

pushd %PYTORCH_AIE_PATH%

@echo off

goto :run

:run_cpp_tests
    echo Running C++ tests
    set "cpp_tests[0]=Linear*"
    set "cpp_tests[1]=Qlinear_2Testw8a8*"
    set "cpp_tests[2]=Qlinear_2Testw4a16*"
    set "cpp_tests[3]=Qlinear_2Testw3a16*"

    set "i=0"
    :_run_cpp_tests_loop
    if defined cpp_tests[%i%] (
        call tools\utils.bat :EchoExecuteStep ".\build\tests\cpp\Release\cpp_tests.exe --gtest_filter=%%cpp_tests[%i%]%%"
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_cpp_tests_loop
    )

    echo C++ tests pass!
    exit /B 0

:run_pytests
    echo Running python unit test cases
    @REM QLinear & QLinearPerGrp
    set "pytest_tests[0]=.\tests\python\test_qlinear.py"
    set "pytest_tests[1]=--w_bit 3 test_qlinear_pergrp.py"
    set "pytest_tests[2]=--w_bit 4 test_qlinear_pergrp.py"

    @REM Flash Attention
    set "pytest_tests[3]=--quant_mode w4abf16  .\tests\python\test_opt_flash_attention.py"
    set "pytest_tests[4]=--quant_mode w8a8     .\tests\python\test_opt_flash_attention.py"
    set "pytest_tests[5]=--quant_mode w4abf16  .\tests\python\test_llama_flash_attention.py"
    set "pytest_tests[6]=--quant_mode w8a8     .\tests\python\test_llama_flash_attention.py"
    set "pytest_tests[7]=--quant_mode w4abf16  .\tests\python\test_qwen2_flash_attention.py"
    set "pytest_tests[8]=--quant_mode w4abf16  .\tests\python\test_chatglm3_flash_attention.py"
    set "pytest_tests[9]=--quant_mode w4abf16  .\tests\python\test_phi_flash_attention.py"
    set "pytest_tests[10]=--quant_mode w4abf16 .\tests\python\test_mistral_flash_attention.py"

    @REM @REM Fast MLP
    @REM set "pytest_tests[11]=.\tests\python\test_llama_fast_mlp.py --w_bit 3"
    @REM set "pytest_tests[12]=.\tests\python\test_llama_fast_mlp.py --w_bit 4"

    set "i=0"
    :_run_pytests_loop
    if defined pytest_tests[%i%] (
        call tools\utils.bat :EchoExecuteStep "pytest %%pytest_tests[%i%]%%"
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_pytests_loop
    )
    echo Python unit tests pass!
    exit /B 0

:run
    call ci\steps.bat :BuildTransformers
    if errorlevel 1 (exit /B 1 %errorlevel%)
    call ci\steps.bat :BuildAndInstallWheels
    if errorlevel 1 (exit /B 1 %errorlevel%)
    call :run_cpp_tests
    if errorlevel 1 (exit /B 1 %errorlevel%)
    call :run_pytests
    if errorlevel 1 (exit /B 1 %errorlevel%)
    echo All tests pass!!
