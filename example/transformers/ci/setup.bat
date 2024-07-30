@REM
@REM Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
@REM

@echo off

@REM Usage: call setup.bat /path/to/root

if "%CONDA_PREFIX%" == "" goto end
goto setup

:end
echo Create and activate the ryzenai-transformers conda env before setup
Exit /b 0

:setup

@REM PWD has a trailing slash
SET "PYTORCH_AIE_PATH=%~1"
SET "PWD=%PYTORCH_AIE_PATH%"
SET "THIRD_PARTY=%PWD%third_party"
SET "DOD_ROOT=%PWD%ext\DynamicDispatch"
CALL %PWD%tools/utils.bat :PrependPathToVar C:\Windows\System32\AMD PATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%ops\python PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%onnx-ops\python PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%tools PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%ext\smoothquant\smoothquant PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%ext\llm-awq PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%ext\llm-awq\awq\quantize PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%ext\llm-awq\awq\utils PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%ext\llm-awq\awq\kernels PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%models\llm\qwen7b PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%models\llm\chatglm PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%models\llm\chatglm3 PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%models\llm\gemma PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%models\llm\mamba PYTHONPATH
CALL %PWD%tools/utils.bat :AppendPathToVar %PWD%models\llm\phi3 PYTHONPATH
SET "AWQ_CACHE=%PWD%ext\awq_cache\"

@REM To avoid a symlink error caused by HuggingFace's .from_pretrained() in pytest
SET "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
