pushd %PYTORCH_AIE_PATH%

@echo off

goto :run

:execute_step
    echo Executing %~1
    %~1
    IF errorlevel 1 (echo %~1 failed! & exit /B 1 %errorlevel)
    EXIT /B 0

:build_packages
    call :execute_step "pip install ops\cpp"
    if errorlevel 1 (exit /B 1 %errorlevel%)
    echo All custom packages installed successfully!
    exit /B 0


:build_cpp_tests
    echo Compiling C++ tests
    pushd tests\cpp
    rmdir /S /Q build
    mkdir build
    pushd build
    call :execute_step "cmake .."
    if errorlevel 1 (popd & popd & exit /B 1 %errorlevel%)
    call :execute_step "cmake --build . --config=Release"
    if errorlevel 1 (popd & popd & exit /B 1 %errorlevel%)
    echo C++ tests compiled successfully!
    popd
    popd
    exit /B 0

:run_cpp_tests
    echo Running C++ tests
    pushd .\tests\cpp\build
    call :execute_step ".\Release\cpp_tests.exe"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    echo C++ tests pass!
    popd
    exit /B 0

:run_pytests
    echo Running python unit test cases
    pushd .\tests\python
    call :execute_step "pytest --num_dlls 1 --num_workers 1 test_qlinear.py"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "pytest --num_dlls 1 --num_workers 2 test_qlinear.py"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "pytest --num_dlls 2 --num_workers 1 test_qlinear.py"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "pytest --num_dlls 2 --num_workers 2 test_qlinear.py"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "pytest test_opt_flash_attention.py"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    popd
    echo Python unit tests pass!
    exit /B 0

:run_e2e_tests
    echo Running e2e model tests
    pushd .\models\opt
    call :execute_step "python save_weights.py --action save --model_name opt-1.3b"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python save_weights.py --action save --model_name opt-125m"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-125m --target cpu --quant_mode none --dtype float32 --task benchmark"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-125m --target cpu --quant_mode none --dtype bfloat16 --task benchmark"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-125m --target cpu --quant_mode none --dtype float32 --flash_attention --task benchmark"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-125m --target cpu --quant_mode none --dtype bfloat16 --flash_attention --task benchmark"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-125m --target cpu --quant_mode ptdq --dtype float32 --task benchmark"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-125m --target cpu --quant_mode ptdq --dtype float32 --flash_attention --task benchmark"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-125m --target aie --quant_mode ptdq --dtype float32 --task benchmark"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-125m --target aie --quant_mode ptdq --dtype float32 --smoothquant --load --task benchmark"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-125m --target aie --quant_mode ptdq --dtype bfloat16 --task benchmark"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-125m --target aie --quant_mode ptdq --dtype bfloat16 --task benchmark --flash_attention"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-125m --target aie --quant_mode ptdq --dtype float32 --task benchmark --flash_attention"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-1.3b --target aie --quant_mode ptdq --dtype float32 --task decode"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-1.3b --target aie --quant_mode ptdq --dtype bfloat16 --task decode"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    call :execute_step "python run.py --model_name opt-1.3b --target aie --quant_mode ptdq --dtype bfloat16 --task benchmark"
    if errorlevel 1 (popd & exit /B 1 %errorlevel%)
    popd
    echo E2E model tests pass!
    exit /B 0

:run
    call :build_packages
    if errorlevel 1 (exit /B 1 %errorlevel%)
    call :build_cpp_tests
    if errorlevel 1 (exit /B 1 %errorlevel%)
    call :run_cpp_tests
    if errorlevel 1 (exit /B 1 %errorlevel%)
    call :run_pytests
    if errorlevel 1 (exit /B 1 %errorlevel%)
    call :run_e2e_tests
    if errorlevel 1 (exit /B 1 %errorlevel%)
    echo All tests pass!!
    @echo on