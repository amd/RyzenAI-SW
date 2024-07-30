#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

# Usage ./ci/setup.ps1 /path/to/root

param(
    [string] $root
)

if (!(Test-Path -Path Env:${CONDA_PREFIX})) {
    Write-Warning "Create and activate the ryzenai-transformers conda env before setup"
    Exit 1
}

$env:DOD_ROOT = "$root\ext\DynamicDispatch"
$env:PYTORCH_AIE_PATH = $root

# source common functions
. $root/tools/utils.ps1

$env:THIRD_PARTY = Join-Path $root "third_party"
Add-PathToVar "C:\Windows\System32\AMD" PATH

Add-PathToVar "$root\ops\python" PYTHONPATH -Append
Add-PathToVar "$root\onnx-ops\python" PYTHONPATH -Append
Add-PathToVar "$root\tools" PYTHONPATH -Append
Add-PathToVar "$root\ext\smoothquant\smoothquant" PYTHONPATH -Append
Add-PathToVar "$root\ext\llm-awq" PYTHONPATH -Append
Add-PathToVar "$root\ext\llm-awq\awq\quantize" PYTHONPATH -Append
Add-PathToVar "$root\ext\llm-awq\awq\utils" PYTHONPATH -Append
Add-PathToVar "$root\ext\llm-awq\awq\kernels" PYTHONPATH -Append
Add-PathToVar "$root\models\llm\chatglm" PYTHONPATH -Append
Add-PathToVar "$root\models\llm\qwen7b" PYTHONPATH
Add-PathToVar "$root\models\llm\chatglm" PYTHONPATH
Add-PathToVar "$root\models\llm\chatglm3" PYTHONPATH
Add-PathToVar "$root\models\llm\gemma" PYTHONPATH
$env:AWQ_CACHE = "$root\ext\awq_cache"

# To avoid a symlink error caused by HuggingFace's .from_pretrained() in pytest
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
