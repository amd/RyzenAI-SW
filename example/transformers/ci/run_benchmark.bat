@REM Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.

pushd %PYTORCH_AIE_PATH%

@echo off

cd %PWD%\models\llm
call :run_all_models

:run_Phi_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "phi_benchmark_tests[0]=python run_awq.py --model_name microsoft/phi-2 --task quantize --algorithm pergrp"
    set "phi_benchmark_tests[1]=python run_awq.py --model_name microsoft/phi-2 --task decode --algorithm pergrp"
    set "phi_benchmark_tests[2]=python run_awq.py --model_name microsoft/phi-2 --task decode --algorithm pergrp --flash_attention_plus"
    set "i=0"
    :_run_phi_benchmark_tests_loop
    if defined phi_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%phi_benchmark_tests[%i%]%% ...
        call %%phi_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_phi_benchmark_tests_loop
    )
    echo Phi Benchmarking All tests Done!
    exit /B 0

:run_Mamba-1.4b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "mamba1.4b_benchmark_tests[0]=python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task quantize --algorithm pergrp --group_size 32"
    set "mamba1.4b_benchmark_tests[1]=python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task quantize --algorithm pergrp --group_size 64"
    set "mamba1.4b_benchmark_tests[2]=python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task decode --algorithm pergrp --group_size 32"
    set "mamba1.4b_benchmark_tests[3]=python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task decode --algorithm pergrp --group_size 64"
    set "i=0"
    :_run_mamba1.4b_benchmark_tests_loop
    if defined mamba1.4b_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%mamba1.4b_benchmark_tests[%i%]%% ...
        call %%mamba1.4b_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_mamba1.4b_benchmark_tests_loop
    )
    echo Mamba-1.4b Benchmarking All tests Done!
    exit /B 0

:run_Mamba-2.8b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "mamba2.8b_benchmark_tests[0]=python run_awq.py --model_name state-spaces/mamba-2.8b-hf --task quantize --algorithm pergrp --group_size 32"
    set "mamba2.8b_benchmark_tests[1]=python run_awq.py --model_name state-spaces/mamba-2.8b-hf --task decode --algorithm pergrp --group_size 32"
    set "i=0"
    :_run_mamba2.8b_benchmark_tests_loop
    if defined mamba2.8b_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%mamba2.8b_benchmark_tests[%i%]%% ...
        call %%mamba2.8b_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_mamba2.8b_benchmark_tests_loop
    )
    echo Mamba-2.8b Benchmarking All tests Done!
    exit /B 0

:run_Gemma-2b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "gemma2b_benchmark_tests[0]=python run_awq.py --task quantize --model_name google/gemma-2b"
    set "gemma2b_benchmark_tests[1]=python run_awq.py --task decode --model_name google/gemma-2b --target aie"
    set "i=0"
    :_run_gemma2b_benchmark_tests_loop
    if defined gemma2b_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%gemma2b_benchmark_tests[%i%]%% ...
        call %%gemma2b_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_gemma2b_benchmark_tests_loop
    )
    echo Gemma-2b Benchmarking All tests Done!
    exit /B 0

:run_Gemma-7b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "gemma7b_benchmark_tests[0]=python run_awq.py --task quantize --model_name google/gemma-7b"
    set "gemma7b_benchmark_tests[1]=python run_awq.py --task decode --model_name google/gemma-7b --target aie"
    set "i=0"
    :_run_gemma7b_benchmark_tests_loop
    if defined gemma7b_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%gemma7b_benchmark_tests[%i%]%% ...
        call %%gemma7b_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_gemma7b_benchmark_tests_loop
    )
    echo Gemma-7b Benchmarking All tests Done!
    exit /B 0

:run_Mistral_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "mistral_benchmark_tests[0]=python run_awq.py --model_name mistralai/Mistral-7B-v0.1 --task quantize --algorithm pergrp"
    set "mistral_benchmark_tests[1]=python run_awq.py --model_name mistralai/Mistral-7B-v0.1 --task decode --target aie --algorithm pergrp"
    set "mistral_benchmark_tests[2]=python run_awq.py --model_name mistralai/Mistral-7B-v0.1 --task decode --target aie --algorithm pergrp --flash_attention_plus"
    set "i=0"
    :_run_mistral_benchmark_tests_loop
    if defined mistral_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%mistral_benchmark_tests[%i%]%% ...
        call %%mistral_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_mistral_benchmark_tests_loop
    )
    echo Mistral Benchmarking All tests Done!
    exit /B 0

:run_Chatglm3-6b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "chatglm3_benchmark_tests[0]=python run_awq.py --model_name THUDM/chatglm3-6b --task quantize --algorithm pergrp"
    set "chatglm3_benchmark_tests[1]=python run_awq.py --model_name THUDM/chatglm3-6b --task decode --target aie --algorithm pergrp"
    set "chatglm3_benchmark_tests[2]=python run_awq.py --model_name THUDM/chatglm3-6b --task decode --target aie --algorithm pergrp --flash_attention_plus"
    set "i=0"
    :_run_chatglm3_benchmark_tests_loop
    if defined chatglm3_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%chatglm3_benchmark_tests[%i%]%% ...
        call %%chatglm3_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_chatglm3_benchmark_tests_loop
    )
    echo Chatglm3-6b Benchmarking All tests Done!
    exit /B 0

:run_Chatglm-6b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "chatglm_benchmark_tests[0]=python run_awq.py --model_name THUDM/chatglm-6b --task quantize --algorithm pergrp"
    set "chatglm_benchmark_tests[1]=python run_awq.py --model_name THUDM/chatglm-6b --task decode --target aie --algorithm pergrp"
    set "i=0"
    :_run_chatglm_benchmark_tests_loop
    if defined chatglm_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%chatglm_benchmark_tests[%i%]%% ...
        call %%chatglm_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_chatglm_benchmark_tests_loop
    )
    echo Chatglm-6b Benchmarking All tests Done!
    exit /B 0

:run_OPT-1.3b-w8a8_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "opt1.3b_w8a8_benchmark_tests[0]=python run_smoothquant.py --model_name facebook/opt-1.3b --task quantize"
    set "opt1.3b_w8a8_benchmark_tests[1]=python run_smoothquant.py --model_name facebook/opt-1.3b --target aie --task benchmark"
    set "opt1.3b_w8a8_benchmark_tests[2]=python run_smoothquant.py --model_name facebook/opt-1.3b --target aie --task benchmark_long"
    set "opt1.3b_w8a8_benchmark_tests[3]=python run_smoothquant.py --model_name facebook/opt-1.3b --target aie --task benchmark --flash_attention_plus"
    set "opt1.3b_w8a8_benchmark_tests[4]=python run_smoothquant.py --model_name facebook/opt-1.3b --target aie --task benchmark_long --flash_attention_plus"
    set "i=0"
    :_run_opt1.3b_w8a8_benchmark_tests_loop
    if defined opt1.3b_w8a8_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%opt1.3b_w8a8_benchmark_tests[%i%]%% ...
        call %%opt1.3b_w8a8_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_opt1.3b_w8a8_benchmark_tests_loop
    )
    echo OPT-1.3b-w8a8 Benchmarking All tests Done!
    exit /B 0

:run_OPT-1.3b-w4abf16_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "opt1.3b_w4abf16_benchmark_tests[0]=python run_awq.py --model_name facebook/opt-1.3b --task quantize"
    set "opt1.3b_w4abf16_benchmark_tests[1]=python run_awq.py --model_name facebook/opt-1.3b --target aie --task benchmark"
    set "opt1.3b_w4abf16_benchmark_tests[2]=python run_awq.py --model_name facebook/opt-1.3b --target aie --task benchmark_long"
    set "opt1.3b_w4abf16_benchmark_tests[3]=python run_awq.py --model_name facebook/opt-1.3b --target aie --task benchmark --flash_attention_plus"
    set "opt1.3b_w4abf16_benchmark_tests[4]=python run_awq.py --model_name facebook/opt-1.3b --target aie --task benchmark_long --flash_attention_plus"
    set "i=0"
    :_run_opt1.3b_w4abf16_benchmark_tests_loop
    if defined opt1.3b_w4abf16_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%opt1.3b_w4abf16_benchmark_tests[%i%]%% ...
        call %%opt1.3b_w4abf16_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_opt1.3b_w4abf16_benchmark_tests_loop
    )
    echo OPT-1.3b-w4abf16 Benchmarking All tests Done!
    exit /B 0

:run_OPT-6.7b-w4abf16_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "opt6.7b_w4abf16_benchmark_tests[0]=python run_awq.py --model_name facebook/opt-6.7b --task quantize"
    set "opt6.7b_w4abf16_benchmark_tests[1]=python run_awq.py --model_name facebook/opt-6.7b --target aie --task benchmark"
    set "opt6.7b_w4abf16_benchmark_tests[2]=python run_awq.py --model_name facebook/opt-6.7b --target aie --task benchmark_long"
    set "opt6.7b_w4abf16_benchmark_tests[3]=python run_awq.py --model_name facebook/opt-6.7b --target aie --task benchmark --flash_attention_plus"
    set "opt6.7b_w4abf16_benchmark_tests[4]=python run_awq.py --model_name facebook/opt-6.7b --target aie --task benchmark_long --flash_attention_plus"
    set "i=0"
    :_run_opt6.7b_w4abf16_benchmark_tests_loop
    if defined opt6.7b_w4abf16_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%opt6.7b_w4abf16_benchmark_tests[%i%]%% ...
        call %%opt6.7b_w4abf16_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_opt6.7b_w4abf16_benchmark_tests_loop
    )
    echo OPT-6.7b-w4abf16 Benchmarking All tests Done!
    exit /B 0

:run_Llama-2-7b-w8a16_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "llama2_w8a16_benchmark_tests[0]=python run_smoothquant.py --model_name llama-2-7b --task quantize"
    set "llama2_w8a16_benchmark_tests[1]=python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie --precision w8a16"
    set "llama2_w8a16_benchmark_tests[2]=python run_smoothquant.py --model_name llama-2-7b --task benchmark_long --target aie --precision w8a16"
    set "llama2_w8a16_benchmark_tests[3]=python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie --flash_attention_plus --precision w8a16"
    set "llama2_w8a16_benchmark_tests[4]=python run_smoothquant.py --model_name llama-2-7b --task benchmark_long --target aie --flash_attention_plus --precision w8a16"
    set "i=0"
    :_run_llama2_w8a16_benchmark_tests_loop
    if defined llama2_w8a16_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%llama2_w8a16_benchmark_tests[%i%]%% ...
        call %%llama2_w8a16_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_llama2_w8a16_benchmark_tests_loop
    )
    echo Llama-2-7b-w8a16 Benchmarking All tests Done!
    exit /B 0

:run_Llama-2-7b-w8a8_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "llama2_w8a8_benchmark_tests[0]=python run_smoothquant.py --model_name llama-2-7b --task quantize"
    set "llama2_w8a8_benchmark_tests[1]=python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie"
    set "llama2_w8a8_benchmark_tests[2]=python run_smoothquant.py --model_name llama-2-7b --task benchmark_long --target aie"
    set "llama2_w8a8_benchmark_tests[3]=python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie --flash_attention_plus"
    set "llama2_w8a8_benchmark_tests[4]=python run_smoothquant.py --model_name llama-2-7b --task benchmark_long --target aie --flash_attention_plus"
    set "i=0"
    :_run_llama2_w8a8_benchmark_tests_loop
    if defined llama2_w8a8_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%llama2_w8a8_benchmark_tests[%i%]%% ...
        call %%llama2_w8a8_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_llama2_w8a8_benchmark_tests_loop
    )
    echo Llama-2-7b-w8a8 Benchmarking All tests Done!
    exit /B 0

:run_Llama-2-7b-chat-w4abf16_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "llama2_w4abf16_benchmark_tests[0]=python run_awq.py --model_name llama-2-7b-chat --task quantize"
    set "llama2_w4abf16_benchmark_tests[1]=python run_awq.py --model_name llama-2-7b-chat --task quantize --algorithm awqplus"
    set "llama2_w4abf16_benchmark_tests[2]=python run_awq.py --model_name llama-2-7b-chat --task benchmark --flash_attention_plus"
    set "llama2_w4abf16_benchmark_tests[3]=python run_awq.py --model_name llama-2-7b-chat --task benchmark_long --flash_attention_plus"
    set "llama2_w4abf16_benchmark_tests[4]=python run_awq.py --model_name llama-2-7b-chat --task benchmark --flash_attention_plus --algorithm awqplus"
    set "llama2_w4abf16_benchmark_tests[5]=python run_awq.py --model_name llama-2-7b-chat --task benchmark_long --flash_attention_plus --algorithm awqplus"
    set "llama2_w4abf16_benchmark_tests[6]=python run_awq.py --model_name llama-2-7b-chat --task benchmark"
    set "llama2_w4abf16_benchmark_tests[7]=python run_awq.py --model_name llama-2-7b-chat --task benchmark_long"
    set "llama2_w4abf16_benchmark_tests[8]=python run_awq.py --model_name llama-2-7b-chat --task benchmark --fast_mlp"
    set "llama2_w4abf16_benchmark_tests[9]=python run_awq.py --model_name llama-2-7b-chat --task benchmark --flash_attention_plus --fast_mlp"
    set "llama2_w4abf16_benchmark_tests[10]=python run_awq.py --model_name llama-2-7b-chat --task benchmark_long --flash_attention_plus --fast_mlp"
    set "i=0"
    :_run_llama2_w4abf16_benchmark_tests_loop
    if defined llama2_w4abf16_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%llama2_w4abf16_benchmark_tests[%i%]%% ...
        call %%llama2_w4abf16_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_llama2_w4abf16_benchmark_tests_loop
    )
    echo Llama-2-7b-chat-w4abf16 Benchmarking All tests Done!
    exit /B 0

:run_Qwen1.5-7b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "qwen_benchmark_tests[0]=python run_awq.py --model_name Qwen/Qwen1.5-7B-Chat --task quantize"
    set "qwen_benchmark_tests[1]=python run_awq.py --model_name Qwen/Qwen1.5-7B-Chat --task quantize --algorithm awqplus"
    set "qwen_benchmark_tests[2]=python run_awq.py --model_name Qwen/Qwen1.5-7B-Chat --task decode"
    set "qwen_benchmark_tests[3]=python run_awq.py --model_name Qwen/Qwen1.5-7B-Chat --task decode --flash_attention_plus"
    set "qwen_benchmark_tests[4]=python run_awq.py --model_name Qwen/Qwen1.5-7B-Chat --task decode --flash_attention_plus --algorithm awqplus"
    set "qwen_benchmark_tests[5]=python run_awq.py --model_name Qwen/Qwen1.5-7B-Chat --task benchmark --flash_attention_plus --algorithm awqplus"
    set "qwen_benchmark_tests[6]=python run_awq.py --model_name Qwen/Qwen1.5-7B-Chat --task benchmark_long --flash_attention_plus --algorithm awqplus"
    set "i=0"
    :_run_qwen_benchmark_tests_loop
    if defined qwen_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%qwen_benchmark_tests[%i%]%% ...
        call %%qwen_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_qwen_benchmark_tests_loop
    )
    echo Qwen1.5-7b Benchmarking All tests Done!
    exit /B 0

:run_Qwen11.5-7b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "qwen11.5_benchmark_tests[0]=python run_awq.py --model_name Qwen/Qwen1.5-7B --task quantize"
    set "qwen11.5_benchmark_tests[1]=python run_awq.py --model_name Qwen/Qwen1.5-7B --task quantize --algorithm awqplus"
    set "qwen11.5_benchmark_tests[2]=python run_awq.py --model_name Qwen/Qwen1.5-7B --task decode --flash_attention_plus"
    set "qwen11.5_benchmark_tests[3]=python run_awq.py --model_name Qwen/Qwen1.5-7B --task decode --flash_attention_plus --algorithm awqplus"
    set "qwen11.5_benchmark_tests[4]=python run_awq.py --model_name Qwen/Qwen1.5-7B --task benchmark --flash_attention_plus --algorithm awqplus"
    set "qwen11.5_benchmark_tests[5]=python run_awq.py --model_name Qwen/Qwen1.5-7B --task benchmark_long --flash_attention_plus --algorithm awqplus"
    set "i=0"
    :_run_qwen11.5_benchmark_tests_loop
    if defined qwen11.5_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%qwen11.5_benchmark_tests[%i%]%% ...
        call %%qwen11.5_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_qwen11.5_benchmark_tests_loop
    )
    echo Qwen11.5-7b Benchmarking All tests Done!
    exit /B 0

:run_CodeLlama-7b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "codellama_benchmark_tests[0]=python run_awq.py --model_name codellama/CodeLlama-7b-hf --task quantize"
    set "codellama_benchmark_tests[1]=python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode"
    set "codellama_benchmark_tests[2]=python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode --flash_attention_plus"
    set "codellama_benchmark_tests[3]=python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode --flash_attention_plus --algorithm awqplus"
    set "i=0"
    :_run_codellama_benchmark_tests_loop
    if defined codellama_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%codellama_benchmark_tests[%i%]%% ...
        call %%codellama_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_codellama_benchmark_tests_loop
    )
    echo CodeLlama-7b Benchmarking All tests Done!
    exit /B 0

:run_Tinyllama_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "Tinyllama_benchmark_tests[0]=python run_awq.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --task quantize --algorithm pergrp"
    set "Tinyllama_benchmark_tests[1]=python run_awq.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --task decode --algorithm pergrp"
    set "i=0"
    :_run_Tinyllama_benchmark_tests_loop
    if defined Tinyllama_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%Tinyllama_benchmark_tests[%i%]%% ...
        call %%Tinyllama_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_Tinyllama_benchmark_tests_loop
    )
    echo Tinyllama Benchmarking All tests Done!
    exit /B 0

:run_Qwen-7b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "qwen7b_benchmark_tests[0]=python run_awq.py --model_name Qwen/Qwen-7b --task quantize"
    set "qwen7b_benchmark_tests[1]=python run_awq.py --model_name Qwen/Qwen-7b --task quantize --algorithm awqplus"
    set "qwen7b_benchmark_tests[2]=python run_awq.py --model_name Qwen/Qwen-7b --task decode --target aie"
    set "qwen7b_benchmark_tests[3]=python run_awq.py --model_name Qwen/Qwen-7b --task decode --target aie --algorithm awqplus"
    set "i=0"
    :_run_qwen7b_benchmark_tests_loop
    if defined qwen7b_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%qwen7b_benchmark_tests[%i%]%% ...
        call %%qwen7b_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_qwen7b_benchmark_tests_loop
    )
    echo Qwen-7b Benchmarking All tests Done!
    exit /B 0

:run_Llama-3-8b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "llama3_benchmark_tests[0]=python run_awq.py --task quantize --model_name Meta-Llama-3-8B-Instruct --algorithm pergrp"
    set "llama3_benchmark_tests[1]=python run_awq.py --model_name Meta-Llama-3-8B-Instruct --task decode --algorithm pergrp"
    set "llama3_benchmark_tests[2]=python run_awq.py --model_name Meta-Llama-3-8B-Instruct --task decode --algorithm pergrp --flash_attention_plus"
    set "i=0"
    :_run_llama3_benchmark_tests_loop
    if defined llama3_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%llama3_benchmark_tests[%i%]%% ...
        call %%llama3_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_llama3_benchmark_tests_loop
    )
    echo Llama-3-8b Benchmarking All tests Done!
    exit /B 0

:run_Phi3_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "phi3_benchmark_tests[0]=python run_awq.py --model_name microsoft/Phi-3-mini-4k-instruct --task quantize --algorithm pergrp"
    set "phi3_benchmark_tests[1]=python run_awq.py --model_name microsoft/Phi-3-mini-4k-instruct --task decode --algorithm pergrp"
    set "i=0"
    :_run_phi3_benchmark_tests_loop
    if defined phi3_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%phi3_benchmark_tests[%i%]%% ...
        call %%phi3_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_phi3_benchmark_tests_loop
    )
    echo Phi3 Benchmarking All tests Done!
    exit /B 0

:run_AGCodeLlama-7b_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "agcodellama_benchmark_tests[0]=python run_awq.py --model_name codellama/CodeLlama-7b-hf --task quantize"
    set "agcodellama_benchmark_tests[1]=python assisted_generation.py --model_name CodeLlama-7b-hf --task decode"
    set "agcodellama_benchmark_tests[2]=python assisted_generation.py --model_name CodeLlama-7b-hf --task decode --assisted_generation"
    set "agcodellama_benchmark_tests[3]=python assisted_generation.py --model_name CodeLlama-7b-hf --task decode --assisted_generation --draft_precision w8af32"
    set "i=0"
    :_run_agcodellama_benchmark_tests_loop
    if defined agcodellama_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%agcodellama_benchmark_tests[%i%]%% ...
        call %%agcodellama_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_agcodellama_benchmark_tests_loop
    )
    echo AGCodeLlama-7b Benchmarking All tests Done!
    exit /B 0

:run_AGLlama-2-7b-chat-w4abf16_benchmark
    echo Starting Benchmarking ...
    cd %PWD%\models\llm
    set "agllama2_w4abf16_benchmark_tests[0]=python run_awq.py --model_name llama-2-7b-chat --task quantize"
    set "agllama2_w4abf16_benchmark_tests[1]=python assisted_generation.py --model_name llama-2-7b-chat --task benchmark"
    set "agllama2_w4abf16_benchmark_tests[2]=python assisted_generation.py --model_name llama-2-7b-chat --task benchmark --assisted_generation"
    set "agllama2_w4abf16_benchmark_tests[3]=python assisted_generation.py --model_name llama-2-7b-chat --task benchmark --assisted_generation --draft_precision w8af32"
    set "i=0"
    :_run_agllama2_w4abf16_benchmark_tests_loop
    if defined agllama2_w4abf16_benchmark_tests[%i%] (
        call echo Starting Test %i%: %%agllama2_w4abf16_benchmark_tests[%i%]%% ...
        call %%agllama2_w4abf16_benchmark_tests[%i%]%%
        if errorlevel 1 (popd & exit /B 1 %errorlevel%)
        set /a "i+=1"
        goto :_run_agllama2_w4abf16_benchmark_tests_loop
    )
    echo AGLlama-2-7b-chat-w4abf16 Benchmarking All tests Done!
    exit /B 0

:run_all_models
    echo Starting All Models Benchmarking on STX ...
    set "models[0]=CodeLlama-7b"
    set "models[1]=Qwen1.5-7b"
    set "models[2]=Qwen11.5-7b"
    set "models[3]=Llama-2-7b-chat-w4abf16"
    set "models[4]=Llama-2-7b-w8a8"
    set "models[5]=Llama-2-7b-w8a16"
    set "models[6]=OPT-1.3b-w4abf16"
    set "models[7]=OPT-1.3b-w8a8"
    set "models[8]=Chatglm-6b"
    set "models[9]=Chatglm3-6b"
    set "models[10]=Mistral"
    set "models[11]=Phi"
    set "models[12]=Phi3"
    set "models[13]=Tinyllama"
    set "models[14]=Qwen-7b"
    set "models[15]=OPT-6.7b-w4abf16"
    set "models[16]=Llama-3-8b"
    set "models[17]=Mamba-1.4b"
    set "models[18]=Mamba-2.8b"
    set "models[19]=Gemma-2b"
    set "models[20]=Gemma-7b"
    set "models[21]=AGCodeLlama-7b"
    set "models[22]=AGLlama-2-7b-chat-w4abf16"
    set "j=0"
    cd %PWD%\models\llm
    :_run_each_model_loop
    if defined models[%j%] (
        call echo %%models[%j%]%% Benchmarking on STX
        call :run_%%models[%j%]%%_benchmark
        if errorlevel 1 (exit /B 1 %errorlevel%)
        call echo %%models[%j%]%% Benchmarking on STX Done!!
        set /a "j+=1"
        goto :_run_each_model_loop
    )
