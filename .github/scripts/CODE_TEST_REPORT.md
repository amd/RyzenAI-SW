# Code-block test report (per page)

Generated from a full hardware run on the self-hosted **Strix Halo** runner (`ryzen-ai-1.7.1` env, Windows), every runnable block tested (no `notest`). Each page's blocks run sequentially in one sandbox dir (state persists across blocks).

**Totals:** 253 blocks - **20 pass**, **219 fail**, 14 skipped across 33 pages.

## Failure categories

| Count | Category |
|------:|----------|
| 70 | FAIL: placeholder/template (<...>) or missing CLI |
| 58 | FAIL: missing file (needs a prior step / shipped file) |
| 44 | FAIL: runtime error |
| 19 | FAIL: Linux-only (run on Ubuntu later) |
| 18 | FAIL: other |
| 5 | FAIL: needs a model artifact |
| 5 | FAIL: timeout (large download/build) |

> Categories are heuristic hints. `Linux-only` blocks will pass once an Ubuntu runner is added; many `missing file` cases need a prior block (now that blocks share a sandbox) or a small fixture; `placeholder` are `<...>` templates.

## By page

### `docs/README.md`  -  0 pass / 1 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | bash | FAIL: runtime error | 001 instead ⠋ preparing local preview... [2K[1A[2K[1A[2K[Ginfo port 3000 is already in use. trying 3001  |

### `docs/audio/parakeet-tdt.mdx`  -  1 pass / 8 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 1 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 2 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 3 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 4 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 5 | powershell | FAIL: missing file (needs a prior step / shipped file) | ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt' C:\Users\bcon |
| 6 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 11 | powershell | PASS | ['VitisAIExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']  |
| 12 | powershell | FAIL: placeholder/template (<...>) or missing CLI | ffmpeg : The term 'ffmpeg' is not recognized as the name of a cmdlet, function, script file, or operable progr |

### `docs/audio/whisper-asr.mdx`  -  0 pass / 8 fail / 3 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: runtime error |  EnvironmentNameNotFound: Could not find conda environment: ryzen-ai-1.4.0 You can list all discoverable envir |
| 1 | powershell | FAIL: missing file (needs a prior step / shipped file) | ind path 'C:\Users\bconsolv\AppData\Local\Temp\docs-ci-cjudevoy\docs\audio\whisper' because it does not  exist |
| 2 | powershell | FAIL: runtime error | --'. At line:3 char:5 +   --device npu \ +     ~~~~~~ Unexpected token 'device' in expression or statement. At |
| 3 | powershell | FAIL: runtime error | c \ +     ~ Missing expression after unary operator '--'. At line:4 char:5 +   --input mic \ +     ~~~~~ Unexp |
| 4 | powershell | FAIL: runtime error | ine:4 char:5 +   --eval-dir eval_dataset/LibriSpeech-samples \ +     ~~~~~~~~ Unexpected token 'eval-dir' in e |
| 5 | text | skipped: not code (text/json/etc.) | non-runnable lang |
| 6 | json | skipped: not code (text/json/etc.) | non-runnable lang |
| 7 | json | skipped: not code (text/json/etc.) | non-runnable lang |
| 8 | powershell | FAIL: runtime error | At line:2 char:5 +   --model openai/whisper-base \ +     ~~~~~ Unexpected token 'model' in expression or state |
| 9 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 10 | powershell | FAIL: runtime error | --'. At line:4 char:5 +   --device npu \ +     ~~~~~~ Unexpected token 'device' in expression or statement. At |

### `docs/getting-started/inst.mdx`  -  4 pass / 8 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 3 | python | PASS | Test finished  |
| 6 | bash | PASS |  WARNING: apt does not have a stable CLI interface. Use with caution in scripts.   WARNING: apt does not have  |
| 7 | bash | FAIL: other | es not have a stable CLI interface. Use with caution in scripts.  E: Unsupported file ./xrt_202610.2.21.75_24. |
| 8 | bash | FAIL: Linux-only (run on Ubuntu later) | /bin/bash: line 2: /opt/xilinx/xrt/setup.sh: No such file or directory  |
| 9 | bash | FAIL: Linux-only (run on Ubuntu later) | /bin/bash: line 1: xrt-smi: command not found /bin/bash: -c: line 3: syntax error near unexpected token `s' /b |
| 10 | bash | FAIL: missing file (needs a prior step / shipped file) | cp: cannot stat 'ryzen_ai-1.7.1.tgz': No such file or directory tar (child): ryzen_ai-1.7.1.tgz: Cannot open:  |
| 11 | bash | FAIL: Linux-only (run on Ubuntu later) | /bin/bash: line 1: TARGET-PATH: No such file or directory /bin/bash: line 2: TARGET-PATH: No such file or dire |
| 12 | bash | PASS |   |
| 13 | bash | FAIL: Linux-only (run on Ubuntu later) | /bin/bash: line 1: TARGET-PATH: No such file or directory /bin/bash: line 2: python: command not found  |
| 14 | bash | FAIL: Linux-only (run on Ubuntu later) | /bin/bash: line 1: Setting: command not found /bin/bash: line 3: Test: command not found  |
| 15 | bash | FAIL: Linux-only (run on Ubuntu later) | /bin/bash: line 1: TARGET-PATH: No such file or directory  |
| 16 | python | PASS |  |

### `docs/getting-started/model_quantization.mdx`  -  0 pass / 1 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmpesbkwrqc.py", line 1, in <m |

### `docs/getting-started/modelrun.mdx`  -  0 pass / 7 fail / 4 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmp5xvc3gni.py", line 7, in <m |
| 1 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmps9nrlfhj.py", line 2, in <m |
| 2 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmp6mszf6vh.py", line 9, in <m |
| 3 | json | skipped: not code (text/json/etc.) | non-runnable lang |
| 4 | python | FAIL: needs a model artifact | 1\Lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 573, in _create_inference_sess |
| 5 | cpp | skipped: not code (text/json/etc.) | non-runnable lang |
| 6 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmp9nwhww3s.py", line 5, in <m |
| 7 | cpp | skipped: not code (text/json/etc.) | non-runnable lang |
| 8 | python | FAIL: needs a model artifact | -packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 573, in _create_inference_session     se |
| 9 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmpjypkpi_y.py", line 8, in <m |
| 10 | cpp | skipped: not code (text/json/etc.) | non-runnable lang |

### `docs/llms/distilbert-example.mdx`  -  0 pass / 1 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:3 char:4 + cd <RyzenAI-SW>\Transformer-examples\DistilBERT_text_classification_b ... +    ~ The '<' op |

### `docs/llms/high_level_python.mdx`  -  1 pass / 2 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | PASS |  CondaSystemExit: Exiting.  ERROR: pip's dependency resolver does not currently take into account all the pack |
| 1 | powershell | FAIL: runtime error | ers.0.attn.o_proj.MatMulNBits.qweight": tensor(uint8),"model.layers.0.attn.o_proj.MatMulNBits.scales": tensor( |
| 2 | python | FAIL: runtime error | ers.0.attn.o_proj.MatMulNBits.qweight": tensor(uint8),"model.layers.0.attn.o_proj.MatMulNBits.scales": tensor( |

### `docs/llms/hybrid_oga.mdx`  -  3 pass / 12 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: timeout (large download/build) | timeout after 90s |
| 1 | bat | PASS | RyzenAI\1.7.1\deployment\.\onnxruntime_vitisai_ep.dll C:\Program Files\RyzenAI\1.7.1\deployment\.\onnxruntime_ |
| 2 | powershell | FAIL: runtime error | The string is missing the terminator: '.     + CategoryInfo          : ParserError: (:) [], ParentContainsErro |
| 3 | powershell | FAIL: placeholder/template (<...>) or missing CLI |  <path_to_model_dir> -f <prompt_file> -l <lis ... +                                                 ~ The '<'  |
| 4 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:5 char:26 + .\model_benchmark.exe -i <path_to_model_dir> -f amd_genai_prompt_long ... +                |
| 5 | powershell | FAIL: other | File not found - amd_genai_prompt_long.txt  |
| 6 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:26 + .\model_benchmark.exe -i <path_to_model_dir> -f amd_genai_prompt_long ... +                |
| 7 | powershell | FAIL: runtime error | The string is missing the terminator: '.     + CategoryInfo          : ParserError: (:) [], ParentContainsErro |
| 9 | python | PASS |  |
| 10 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmp3ojuaus9.py", line 1, in <m |
| 11 | powershell | FAIL: placeholder/template (<...>) or missing CLI | e\model_chat.py" -m <model_fo ... +                                                                 ~ The '<'  |
| 12 | powershell | FAIL: placeholder/template (<...>) or missing CLI | \vlm\vlm_run.py" -m <model_fo ... +                                                                 ~ The '<'  |
| 13 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:17 + conda create -n <env_name> python=3.12 -y +                 ~ The '<' operator is reserved |
| 14 | powershell | PASS | miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages (from requests->transformers->model-generate==1.7.1) (3.13) R |
| 15 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:2 char:11 + git clone <link_to_model> +           ~ The '<' operator is reserved for future use.     + |

### `docs/llms/llm-sft-deploy.mdx`  -  0 pass / 5 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 1 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:43 + python train.py --lora --lora_qv --hf_dir <HF_username/repo-name> +                        |
| 2 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:88 + ... el --model_name meta-llama/Llama-3.2-1B --adapter_model_dir <adapter  ... +            |
| 3 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 5 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:59 + python inference.py --quark_safetensors --quant_model_dir <path to qu ... +                |

### `docs/llms/llm_linux.mdx`  -  3 pass / 5 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | bash | PASS |  |
| 2 | bash | PASS | /bin/bash: line 2: TARGET-PATH: No such file or directory  |
| 3 | bash | FAIL: Linux-only (run on Ubuntu later) | /bin/bash: - : invalid option Usage:	/bin/bash [GNU long option] [option] ... 	/bin/bash [GNU long option] [op |
| 5 | bash | FAIL: Linux-only (run on Ubuntu later) | /bin/bash: - : invalid option Usage:	/bin/bash [GNU long option] [option] ... 	/bin/bash [GNU long option] [op |
| 6 | bash | PASS |  |
| 7 | bash | FAIL: Linux-only (run on Ubuntu later) | /bin/bash: line 1: ./model_benchmark: No such file or directory  |
| 8 | bash | FAIL: Linux-only (run on Ubuntu later) | /bin/bash: --: invalid option Usage:	/bin/bash [GNU long option] [option] ... 	/bin/bash [GNU long option] [op |
| 9 | bash | FAIL: other | hon application,     it may be easiest to use pipx install xyz, which will manage a     virtual environment fo |

### `docs/llms/oga-inference.mdx`  -  0 pass / 3 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: other | The syntax of the command is incorrect.  |
| 1 | powershell | FAIL: placeholder/template (<...>) or missing CLI | del_chat.py -m <model_path> -pr <prompt_file> -ipl <input_to ... +                                          ~  |
| 2 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |

### `docs/tools/ai_analyzer.mdx`  -  0 pass / 2 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 1 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmp_m_3fpu7.py", line 3, in <m |
| 2 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmpz6l2twb0.py", line 2, in <m |

### `docs/tools/onnx-benchmark.mdx`  -  0 pass / 3 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 1 | powershell | FAIL: runtime error |  EnvironmentNameNotFound: Could not find conda environment: ryzen-ai-1.6.0 You can list all discoverable envir |
| 2 | powershell | FAIL: missing file (needs a prior step / shipped file) | ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements-win.txt'  |

### `docs/tools/quark-quantization.mdx`  -  2 pass / 15 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: runtime error | hannel-urls] [--file FILE]                               [--no-default-packages] [--subdir SUBDIR]             |
| 1 | powershell | FAIL: runtime error | hannel-urls] [--file FILE]                               [--no-default-packages] [--subdir SUBDIR]             |
| 2 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <ryzenai-sw>\docs\models-tutorials\vision\quark_quantization +    ~ The '<' operator is  |
| 3 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <ryzenai-sw>/docs/vision/quark_quantization +    ~ The '<' operator is reserved for futu |
| 4 | python | PASS |  |
| 5 | powershell | FAIL: runtime error | At line:1 char:11 + cd models && python download_ResNet.py +           ~~ The token '&&' is not a valid statem |
| 6 | powershell | FAIL: missing file (needs a prior step / shipped file) | tar.exe: Error opening archive: Failed to open 'val_images.tar.gz' C:\Users\bconsolv\AppData\Local\miniforge3\ |
| 7 | powershell | FAIL: runtime error | At line:1 char:19 + mkdir -p val_data && tar -xzf val_images.tar.gz -C val_data +                   ~~ The tok |
| 8 | python | PASS | led: CPU version of custom ops library compilation failed:Command '['where', 'cl']' returned non-zero exit sta |
| 9 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmp1cvqh397.py", line 1, in <m |
| 10 | python | FAIL: placeholder/template (<...>) or missing CLI | d with `input_tensors` and will be removed in the next release.[0m [33m [QUARK-WARNING]: The custom ops libr |
| 11 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmpdhwsojdm.py", line 2, in <m |
| 12 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmp2giybti4.py", line 1, in <m |
| 13 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 14 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmp7djja3t1.py", line 1, in <m |
| 15 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmpx2qgv5f5.py", line 1, in <m |
| 16 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |

### `docs/tools/xrt_smi.mdx`  -  0 pass / 9 fail / 1 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: Linux-only (run on Ubuntu later) | At line:1 char:28 + xrt-smi examine -f JSON -o <path/to/output.json> +                            ~ The '<' op |
| 1 | powershell | FAIL: Linux-only (run on Ubuntu later) | xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable pro |
| 4 | json | skipped: not code (text/json/etc.) | non-runnable lang |
| 5 | powershell | FAIL: Linux-only (run on Ubuntu later) | xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable pro |
| 7 | powershell | FAIL: Linux-only (run on Ubuntu later) | xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable pro |
| 9 | powershell | FAIL: Linux-only (run on Ubuntu later) | xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable pro |
| 11 | powershell | FAIL: Linux-only (run on Ubuntu later) | xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable pro |
| 13 | powershell | FAIL: Linux-only (run on Ubuntu later) | At line:1 char:27 + xrt-smi configure --pmode <default \| powersaver \| balanced \| performa ... +             |
| 14 | powershell | FAIL: Linux-only (run on Ubuntu later) | xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable pro |
| 15 | powershell | FAIL: Linux-only (run on Ubuntu later) | xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable pro |

### `docs/vision/getstartex.mdx`  -  0 pass / 11 fail / 1 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 2 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmplar7pj8b.py", line 1, in <m |
| 5 | python | FAIL: placeholder/template (<...>) or missing CLI | yzen-ai-1.7.1\Lib\site-packages\quark\onnx\operators\custom_ops\lib\custom_ops.dll does NOT exist.[0m [33m [ |
| 8 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmpdi7nq53p.py", line 1, in <m |
| 11 | powershell | FAIL: timeout (large download/build) | timeout after 90s |
| 12 | powershell | FAIL: runtime error | tion], ItemNotFoundE     xception     + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.Set |
| 13 | powershell | FAIL: placeholder/template (<...>) or missing CLI | devenv : The term 'devenv' is not recognized as the name of a cmdlet, function, script file, or operable progr |
| 14 | powershell | FAIL: other | File not found - resnet_cifar.exe  |
| 15 | powershell | FAIL: other | File not found - *  |
| 16 | powershell | FAIL: placeholder/template (<...>) or missing CLI | resnet_cifar.exe : The term 'resnet_cifar.exe' is not recognized as the name of a cmdlet, function, script fil |
| 17 | powershell | FAIL: placeholder/template (<...>) or missing CLI | : CommandNotFoundException   Predicted : The term 'Predicted' is not recognized as the name of a cmdlet, funct |
| 18 | cpp | skipped: not code (text/json/etc.) | non-runnable lang |
| 19 | powershell | FAIL: placeholder/template (<...>) or missing CLI | resnet_cifar.exe : The term 'resnet_cifar.exe' is not recognized as the name of a cmdlet, function, script fil |

### `docs/vision/getting-started-resnet-bf16.mdx`  -  0 pass / 9 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:3 char:34 + set RYZEN_AI_INSTALLATION_PATH = <path/to/RyzenAI/installation> +                          |
| 1 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 2 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 3 | python | FAIL: runtime error |   File "C:\Users\bconsolv\AppData\Local\Temp\tmpzwl02ehg.py", line 1     cache_dir = Path(__file__).parent.res |
| 4 | powershell | FAIL: runtime error |         ~~ Unexpected token 'AI' in expression or statement. At line:4 char:7 + [Vitis AI EP] No. of Subgraphs |
| 5 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 6 | powershell | FAIL: placeholder/template (<...>) or missing CLI | ErrorId : CommandNotFoundException   Image : The term 'Image' is not recognized as the name of a cmdlet, funct |
| 7 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 8 | powershell | FAIL: placeholder/template (<...>) or missing CLI | ErrorId : CommandNotFoundException   Image : The term 'Image' is not recognized as the name of a cmdlet, funct |

### `docs/vision/hello-world.mdx`  -  0 pass / 4 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | bat | FAIL: other | The syntax of the command is incorrect.  |
| 1 | bat | FAIL: placeholder/template (<...>) or missing CLI | '#' is not recognized as an internal or external command, operable program or batch file.  |
| 2 | powershell | FAIL: missing file (needs a prior step / shipped file) | ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'  |
| 3 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |

### `docs/vision/igpu-getting-started.mdx`  -  0 pass / 9 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | bat | FAIL: other | The syntax of the command is incorrect.  |
| 1 | bat | FAIL: other | (ryzen-ai-1.7.1)  |
| 2 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <RyzenAI-SW>\CNN-examples\iGPU\getting_started +    ~ The '<' operator is reserved for f |
| 3 | powershell | FAIL: runtime error | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: Error while finding module specific |
| 4 | powershell | FAIL: runtime error | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: Error while finding module specific |
| 5 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 6 | powershell | FAIL: timeout (large download/build) | timeout after 90s |
| 7 | powershell | FAIL: placeholder/template (<...>) or missing CLI | t.PowerShell.Commands.SetLocationCommand   compile.bat : The term 'compile.bat' is not recognized as the name  |
| 8 | powershell | FAIL: placeholder/template (<...>) or missing CLI | run.bat : The term 'run.bat' is not recognized as the name of a cmdlet, function, script file, or operable pro |

### `docs/vision/image-classification.mdx`  -  0 pass / 4 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <RyzenAI-SW>\CNN-examples\image_classification +    ~ The '<' operator is reserved for f |
| 1 | powershell | FAIL: missing file (needs a prior step / shipped file) | use it does not exist. At line:1 char:1 + cd models + ~~~~~~~~~     + CategoryInfo          : ObjectNotFound:  |
| 2 | powershell | FAIL: runtime error | At line:1 char:16 + mkdir val_data && tar -xzf val_images.tar.gz -C val_data +                ~~ The token '&& |
| 3 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |

### `docs/vision/nemotron-ocr-v2.mdx`  -  2 pass / 10 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: timeout (large download/build) | timeout after 90s |
| 1 | powershell | PASS | :  40% (4/10) Filtering content:  50% (5/10) Filtering content:  50% (5/10), 8.93 MiB \| 2.85 MiB/s Filtering  |
| 2 | powershell | FAIL: timeout (large download/build) | timeout after 90s |
| 3 | powershell | FAIL: needs a model artifact | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 4 | powershell | FAIL: runtime error |  of a cmdlet, function,  script file, or operable program. Check the spelling of the name, or if a path was in |
| 5 | powershell | FAIL: placeholder/template (<...>) or missing CLI | All : The term 'All' is not recognized as the name of a cmdlet, function, script file, or operable program. Ch |
| 6 | powershell | PASS |  |
| 7 | powershell | FAIL: runtime error | ing expression after unary operator '--'. At line:3 char:5 +   --vai-config vitisai_config.json \ +     ~~~~~~ |
| 8 | powershell | FAIL: runtime error | ing expression after unary operator '--'. At line:4 char:5 +   --vai-config vitisai_config.json \ +     ~~~~~~ |
| 9 | powershell | FAIL: runtime error | ing expression after unary operator '--'. At line:3 char:5 +   --vai-config vitisai_config.json \ +     ~~~~~~ |
| 10 | powershell | FAIL: runtime error | ing expression after unary operator '--'. At line:3 char:5 +   --vai-config vitisai_config.json \ +     ~~~~~~ |
| 11 | powershell | FAIL: runtime error | ne:5 char:5 +   --image "Images\test\test.jpg" \ +     ~~~~~ Unexpected token 'image' in expression or stateme |

### `docs/vision/npu-gpu-pipeline.mdx`  -  1 pass / 7 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | bat | FAIL: other | The syntax of the command is incorrect.  |
| 1 | bat | FAIL: placeholder/template (<...>) or missing CLI |  (ryzen-ai-1.7.1) rem Location of RyzenAI software installation path or default at "C:\Program Files\RyzenAI\< |
| 2 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <RyzenAI-SW>\demo\NPU-GPU-Pipeline +    ~ The '<' operator is reserved for future use.   |
| 3 | powershell | FAIL: missing file (needs a prior step / shipped file) | ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'stable_diffusion\\requirements- |
| 4 | powershell | PASS | %XLNX_VART_FIRMWARE%  |
| 5 | powershell | FAIL: runtime error | Temp\docs-ci-7xo63r59\%RYZEN_AI_INSTALLATION_PATH%\voe-4.0-win _amd64\vaip_config.json' because it does not ex |
| 6 | powershell | FAIL: missing file (needs a prior step / shipped file) | . At line:1 char:1 + cd stable_diffusion + ~~~~~~~~~~~~~~~~~~~     + CategoryInfo          : ObjectNotFound: ( |
| 7 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |

### `docs/vision/super_resolution.mdx`  -  1 pass / 0 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | python | PASS | PSNR @ MSE=100 -> 28.13 dB  |

### `docs/vision/torchvision.mdx`  -  0 pass / 4 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: other | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: No module named ipykernel  |
| 1 | powershell | FAIL: other | The syntax of the command is incorrect.  |
| 2 | powershell | FAIL: runtime error | At line:1 char:18 + mkdir val_images && tar -xzf val_images.tar.gz -C val_images +                  ~~ The tok |
| 3 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |

### `docs/vision/yolov8m.mdx`  -  0 pass / 21 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: runtime error |      [--show-channel-urls] [--file FILE]                               [--no-default-packages] [--subdir SUBDI |
| 1 | powershell | FAIL: runtime error |      [--show-channel-urls] [--file FILE]                               [--no-default-packages] [--subdir SUBDI |
| 2 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <RyzenAI-SW>\docs\models-tutorials\vision\object_detection\yolov8m +    ~ The '<' operat |
| 3 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <RyzenAI-SW>/docs/vision/object_detection/yolov8m +    ~ The '<' operator is reserved fo |
| 4 | powershell | FAIL: needs a model artifact | ause it does not exist. At line:1 char:1 + cd models + ~~~~~~~~~     + CategoryInfo          : ObjectNotFound: |
| 5 | powershell | FAIL: runtime error |  Unexpected token 'output_model_path' in expression or statement. At line:4 char:28 +                          |
| 6 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 7 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 8 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 9 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 10 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 11 | powershell | FAIL: runtime error | nexpected token 'output_model_path' in expression or statement. At line:4 char:28 +                          - |
| 12 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 13 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 14 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 15 | powershell | FAIL: runtime error |                    --exclude_subgraphs "[/model.22/Concat_3], [ ... +                            ~ Missing exp |
| 16 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 17 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 18 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 19 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 20 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |

### `docs/vision/yolov8s-worldv2.mdx`  -  1 pass / 8 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | bat | FAIL: other | The syntax of the command is incorrect.  |
| 1 | powershell | PASS | ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. Thi |
| 2 | powershell | FAIL: placeholder/template (<...>) or missing CLI | .\download.bat : The term '.\download.bat' is not recognized as the name of a cmdlet, function, script file, o |
| 3 | powershell | FAIL: needs a model artifact | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 4 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 5 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 6 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 7 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 8 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |

### `docs/windows-ml/clip.mdx`  -  0 pass / 9 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: missing file (needs a prior step / shipped file) |  CondaSystemExit: Exiting.  ERROR: Could not open requirements file: [Errno 2] No such file or directory: '.\\ |
| 1 | powershell | FAIL: other |  |
| 2 | powershell | FAIL: runtime error |  cmdlet,  function, script file, or operable program. Check the spelling of the name, or if a path was include |
| 3 | powershell | FAIL: placeholder/template (<...>) or missing CLI |  'windowsappruntimeinstall-x86.exe' is not recognized as the name of a  cmdlet, function, script file, or oper |
| 4 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 5 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 6 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 7 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 8 | powershell | FAIL: runtime error |                                         ~ Missing expression after unary operator '-'. At line:36 char:1 + --- |

### `docs/windows-ml/googlebert.mdx`  -  0 pass / 6 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: missing file (needs a prior step / shipped file) |  CondaSystemExit: Exiting.  ERROR: Could not open requirements file: [Errno 2] No such file or directory: '.\\ |
| 1 | powershell | FAIL: other |  |
| 2 | powershell | FAIL: runtime error |  cmdlet,  function, script file, or operable program. Check the spelling of the name, or if a path was include |
| 3 | powershell | FAIL: placeholder/template (<...>) or missing CLI |  'windowsappruntimeinstall-x86.exe' is not recognized as the name of a  cmdlet, function, script file, or oper |
| 4 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 5 | powershell | FAIL: runtime error |                                               ~ An expression was expected after '('. At line:21 char:232 + .. |

### `docs/windows-ml/installation.mdx`  -  0 pass / 4 fail / 1 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:4 char:4 + cd <RyzenAI-SW>\WinML\CNN\ResNet +    ~ The '<' operator is reserved for future use.     +  |
| 1 | powershell | FAIL: other |  |
| 2 | powershell | FAIL: placeholder/template (<...>) or missing CLI |  'windowsappruntimeinstall-x86.exe' is not recognized as the name of a  cmdlet, function, script file, or oper |
| 3 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <RyzenAI-SW>\WinML +    ~ The '<' operator is reserved for future use.     + CategoryInf |
| 4 | text | skipped: not code (text/json/etc.) | non-runnable lang |

### `docs/windows-ml/resnet.mdx`  -  1 pass / 10 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | powershell | FAIL: missing file (needs a prior step / shipped file) |  CondaSystemExit: Exiting.  ERROR: Could not open requirements file: [Errno 2] No such file or directory: '.\\ |
| 1 | powershell | FAIL: other |  |
| 2 | powershell | FAIL: runtime error |  cmdlet,  function, script file, or operable program. Check the spelling of the name, or if a path was include |
| 3 | powershell | FAIL: placeholder/template (<...>) or missing CLI |  'windowsappruntimeinstall-x86.exe' is not recognized as the name of a  cmdlet, function, script file, or oper |
| 4 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <RyzenAI-SW>\WinML\CNN\ResNet\model\ +    ~ The '<' operator is reserved for future use. |
| 5 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <RyzenAI-SW>\WinML\CNN\ResNet\python +    ~ The '<' operator is reserved for future use. |
| 6 | powershell | FAIL: missing file (needs a prior step / shipped file) | C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-1.7.1\python.exe: can't open file 'C:\\Users\\bconsol |
| 8 | python | PASS |  |
| 9 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmpl36dji68.py", line 1, in <m |
| 10 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmpilfeq3k6.py", line 1, in <m |
| 11 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmpm29bo3c6.py", line 1, in <m |

### `docs/windows-ml/winml_ep.mdx`  -  0 pass / 4 fail / 4 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 0 | cpp | skipped: not code (text/json/etc.) | non-runnable lang |
| 1 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmppj78ifsb.py", line 25, in < |
| 2 | cpp | skipped: not code (text/json/etc.) | non-runnable lang |
| 3 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmp6b04ypkv.py", line 3, in <m |
| 4 | cpp | skipped: not code (text/json/etc.) | non-runnable lang |
| 5 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmps4r14ulf.py", line 2, in <m |
| 6 | cpp | skipped: not code (text/json/etc.) | non-runnable lang |
| 7 | python | FAIL: placeholder/template (<...>) or missing CLI | Traceback (most recent call last):   File "C:\Users\bconsolv\AppData\Local\Temp\tmpx7xsk6po.py", line 4, in <m |

### `docs/windows-ml/winml_example.mdx`  -  0 pass / 9 fail / 0 skipped

| Block | Lang | Result | Detail |
|------:|------|--------|--------|
| 1 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:3 char:4 + cd <RyzenAI-SW>\WinML\CNN\ResNet +    ~ The '<' operator is reserved for future use.     +  |
| 2 | powershell | FAIL: other |  |
| 3 | powershell | FAIL: runtime error |  cmdlet,  function, script file, or operable program. Check the spelling of the name, or if a path was include |
| 4 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <RyzenAI-SW>\WinML\CNN\ResNet\model +    ~ The '<' operator is reserved for future use.  |
| 5 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <RyzenAI-SW>\WinML\CNN\ResNet\python +    ~ The '<' operator is reserved for future use. |
| 6 | powershell | FAIL: runtime error | 6 + 287, lynx with confidence of 0.00119624 +      ~~~~ Unexpected token 'lynx' in expression or statement. At |
| 7 | powershell | FAIL: placeholder/template (<...>) or missing CLI | At line:1 char:4 + cd <RyzenAI-SW>\WinML\CNN\cpp\CppResnetBuildDemo\ +    ~ The '<' operator is reserved for f |
| 8 | powershell | FAIL: runtime error | ognized as the name  of a cmdlet, function, script file, or operable program. Check the spelling of the name,  |
| 9 | powershell | FAIL: runtime error | soccer' in expression or statement. At line:8 char:5 + 208,Labrador retriever                0.61% +     ~ Mis |
