# Code-block triage

Full hardware re-run in the `ryzen-ai-ci` conda env (Strix Halo). **197 failing blocks across 32 pages.**

Failures are grouped by page below. Categories are heuristic hints, not automatic fixes - nothing has been tagged `notest`.

## Failure categories

| Count | Category |
|------:|----------|
| 92 | Other (review) |
| 45 | Missing script file (needs example files / cd) |
| 24 | Python error (missing model/var) |
| 18 | Placeholder template (<...>) |
| 7 | Timeout (large clone/build) |
| 6 | Missing requirements.txt (needs prior clone/cd) |
| 5 | Missing file/dir (needs prior step) |

## By page

### `docs/README.md` (1)

- [ ] block#0 (bash) - **Timeout (large clone/build)** - `timeout after 90s`

### `docs/audio/parakeet-tdt.mdx` (8)

- [ ] block#0 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#1 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#2 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#3 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#4 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#5 (powershell) - **Missing requirements.txt (needs prior clone/cd)** - `ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'`
- [ ] block#6 (powershell) - **Other (review)** - `are installed. This behaviour is the source of the following dependency conflicts.`
- [ ] block#12 (powershell) - **Placeholder template (<...>)** - `ffmpeg : The term 'ffmpeg' is not recognized as the name of a cmdlet, function, script file, or operable program.`

### `docs/audio/whisper-asr.mdx` (8)

- [ ] block#0 (powershell) - **Other (review)** - ``
- [ ] block#1 (powershell) - **Other (review)** - `W\docs\audio\docs\models-tutorials\audio\whisper'`
- [ ] block#2 (powershell) - **Other (review)** - `--'.`
- [ ] block#3 (powershell) - **Other (review)** - `c \`
- [ ] block#4 (powershell) - **Other (review)** - `ine:4 char:5`
- [ ] block#8 (powershell) - **Other (review)** - `At line:2 char:5`
- [ ] block#9 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#10 (powershell) - **Other (review)** - `--'.`

### `docs/benchmarking/onnx-benchmark.mdx` (3)

- [ ] block#0 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#1 (powershell) - **Other (review)** - ``
- [ ] block#2 (powershell) - **Missing requirements.txt (needs prior clone/cd)** - `ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements-win.txt'`

### `docs/benchmarking/quark-quantization.mdx` (11)

- [ ] block#5 (powershell) - **Other (review)** - `At line:1 char:11`
- [ ] block#6 (powershell) - **Other (review)** - `goryInfo          : ResourceExists: (C:\Users\bconso...arking\val_data:String) [New-Item], IOException`
- [ ] block#7 (powershell) - **Other (review)** - `At line:1 char:19`
- [ ] block#9 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#10 (python) - **Other (review)** - `aced with `input_tensors` and will be removed in the next release.[0m`
- [ ] block#11 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#12 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#13 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#14 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#15 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#16 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`

### `docs/getting-started/getstartex.mdx` (11)

- [ ] block#2 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#5 (python) - **Other (review)** - `s\ryzen-ai-ci\Lib\site-packages\quark\onnx\operators\custom_ops\lib\custom_ops.dll does NOT exist.[0m`
- [ ] block#8 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#11 (powershell) - **Timeout (large clone/build)** - `timeout after 90s`
- [ ] block#12 (powershell) - **Other (review)** - `ound,Microsoft.PowerShell.Commands.SetLocationCommand`
- [ ] block#13 (powershell) - **Placeholder template (<...>)** - `devenv : The term 'devenv' is not recognized as the name of a cmdlet, function, script file, or operable program.`
- [ ] block#14 (powershell) - **Other (review)** - `File not found - resnet_cifar.exe`
- [ ] block#15 (powershell) - **Other (review)** - `File not found - *`
- [ ] block#16 (powershell) - **Placeholder template (<...>)** - `resnet_cifar.exe : The term 'resnet_cifar.exe' is not recognized as the name of a cmdlet, function, script file, or`
- [ ] block#17 (powershell) - **Other (review)** - `: CommandNotFoundException`
- [ ] block#19 (powershell) - **Placeholder template (<...>)** - `resnet_cifar.exe : The term 'resnet_cifar.exe' is not recognized as the name of a cmdlet, function, script file, or`

### `docs/getting-started/inst.mdx` (1)

- [ ] block#3 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`

### `docs/getting-started/linux.mdx` (8)

- [ ] block#1 (bash) - **Other (review)** - `es not have a stable CLI interface. Use with caution in scripts.`
- [ ] block#2 (bash) - **Missing file/dir (needs prior step)** - `/bin/bash: line 2: /opt/xilinx/xrt/setup.sh: No such file or directory`
- [ ] block#3 (bash) - **Other (review)** - `/bin/bash: line 1: xrt-smi: command not found`
- [ ] block#4 (bash) - **Other (review)** - `mkdir: cannot create directory ‘ryzen_ai-1.7.1’: File exists`
- [ ] block#5 (bash) - **Missing file/dir (needs prior step)** - `/bin/bash: line 1: TARGET-PATH: No such file or directory`
- [ ] block#7 (bash) - **Missing file/dir (needs prior step)** - `/bin/bash: line 1: TARGET-PATH: No such file or directory`
- [ ] block#8 (bash) - **Other (review)** - `/bin/bash: line 1: Setting: command not found`
- [ ] block#9 (bash) - **Missing file/dir (needs prior step)** - `/bin/bash: line 1: TARGET-PATH: No such file or directory`

### `docs/llms/distilbert-example.mdx` (1)

- [ ] block#0 (powershell) - **Other (review)** - `At line:3 char:4`

### `docs/llms/high_level_python.mdx` (3)

- [ ] block#0 (powershell) - **Timeout (large clone/build)** - `timeout after 90s`
- [ ] block#1 (powershell) - **Other (review)** - `ers.0.attn.o_proj.MatMulNBits.qweight": tensor(uint8),"model.layers.0.attn.o_proj.MatMulNBits.scales": tensor(float16),"`
- [ ] block#2 (python) - **Other (review)** - `ers.0.attn.o_proj.MatMulNBits.qweight": tensor(uint8),"model.layers.0.attn.o_proj.MatMulNBits.scales": tensor(float16),"`

### `docs/llms/hybrid_oga.mdx` (13)

- [ ] block#0 (powershell) - **Timeout (large clone/build)** - `timeout after 90s`
- [ ] block#1 (bat) - **Other (review)** - `A subdirectory or file llm_run already exists.`
- [ ] block#2 (powershell) - **Other (review)** - `The string is missing the terminator: '.`
- [ ] block#3 (powershell) - **Placeholder template (<...>)** - `<path_to_model_dir> -f <prompt_file> -l <lis ...`
- [ ] block#4 (powershell) - **Other (review)** - `At line:5 char:26`
- [ ] block#5 (powershell) - **Other (review)** - `File not found - amd_genai_prompt_long.txt`
- [ ] block#6 (powershell) - **Other (review)** - `At line:1 char:26`
- [ ] block#7 (powershell) - **Other (review)** - `The string is missing the terminator: '.`
- [ ] block#10 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#11 (powershell) - **Placeholder template (<...>)** - `e\model_chat.py" -m <model_fo ...`
- [ ] block#12 (powershell) - **Placeholder template (<...>)** - `\vlm\vlm_run.py" -m <model_fo ...`
- [ ] block#13 (powershell) - **Other (review)** - `At line:1 char:17`
- [ ] block#15 (powershell) - **Other (review)** - `At line:2 char:11`

### `docs/llms/llm_linux.mdx` (5)

- [ ] block#3 (bash) - **Other (review)** - `/bin/bash: - : invalid option`
- [ ] block#5 (bash) - **Other (review)** - `/bin/bash: - : invalid option`
- [ ] block#7 (bash) - **Missing file/dir (needs prior step)** - `/bin/bash: line 1: ./model_benchmark: No such file or directory`
- [ ] block#8 (bash) - **Other (review)** - `/bin/bash: --: invalid option`
- [ ] block#9 (bash) - **Other (review)** - `hon application,`

### `docs/llms/oga-inference.mdx` (3)

- [ ] block#0 (powershell) - **Other (review)** - `The syntax of the command is incorrect.`
- [ ] block#1 (powershell) - **Placeholder template (<...>)** - `del_chat.py -m <model_path> -pr <prompt_file> -ipl <input_to ...`
- [ ] block#2 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`

### `docs/multimodal/npu-gpu-pipeline.mdx` (5)

- [ ] block#2 (powershell) - **Other (review)** - `At line:1 char:4`
- [ ] block#3 (powershell) - **Missing requirements.txt (needs prior clone/cd)** - `ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'stable_diffusion\\requirements-common.txt`
- [ ] block#5 (powershell) - **Other (review)** - `AI-SW\docs\multimodal\%RYZEN_AI`
- [ ] block#6 (powershell) - **Other (review)** - `e_diffusion`
- [ ] block#7 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`

### `docs/running-models/model_quantization.mdx` (1)

- [ ] block#0 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`

### `docs/running-models/modelrun.mdx` (7)

- [ ] block#0 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#1 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#2 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#4 (python) - **Other (review)** - `i\Lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 573, in _create_inference_session`
- [ ] block#6 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#8 (python) - **Other (review)** - `-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 573, in _create_inference_session`
- [ ] block#9 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`

### `docs/tools/ai_analyzer.mdx` (2)

- [ ] block#1 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#2 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`

### `docs/tools/xrt_smi.mdx` (9)

- [ ] block#0 (powershell) - **Other (review)** - `At line:1 char:28`
- [ ] block#1 (powershell) - **Placeholder template (<...>)** - `xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable program.`
- [ ] block#5 (powershell) - **Placeholder template (<...>)** - `xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable program.`
- [ ] block#7 (powershell) - **Placeholder template (<...>)** - `xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable program.`
- [ ] block#9 (powershell) - **Placeholder template (<...>)** - `xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable program.`
- [ ] block#11 (powershell) - **Placeholder template (<...>)** - `xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable program.`
- [ ] block#13 (powershell) - **Other (review)** - `At line:1 char:27`
- [ ] block#14 (powershell) - **Placeholder template (<...>)** - `xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable program.`
- [ ] block#15 (powershell) - **Placeholder template (<...>)** - `xrt-smi : The term 'xrt-smi' is not recognized as the name of a cmdlet, function, script file, or operable program.`

### `docs/vision/getting-started-resnet-bf16.mdx` (8)

- [ ] block#0 (powershell) - **Other (review)** - `At line:3 char:34`
- [ ] block#1 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#2 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#4 (powershell) - **Other (review)** - `~~`
- [ ] block#5 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#6 (powershell) - **Other (review)** - `ErrorId : CommandNotFoundException`
- [ ] block#7 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#8 (powershell) - **Other (review)** - `ErrorId : CommandNotFoundException`

### `docs/vision/hello-world.mdx` (2)

- [ ] block#2 (powershell) - **Missing requirements.txt (needs prior clone/cd)** - `ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'`
- [ ] block#3 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`

### `docs/vision/igpu-getting-started.mdx` (7)

- [ ] block#2 (powershell) - **Other (review)** - `At line:1 char:4`
- [ ] block#3 (powershell) - **Other (review)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: Error while finding module specification for 'ol`
- [ ] block#4 (powershell) - **Other (review)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: Error while finding module specification for 'ol`
- [ ] block#5 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#6 (powershell) - **Timeout (large clone/build)** - `timeout after 90s`
- [ ] block#7 (powershell) - **Other (review)** - `t.PowerShell.Commands.SetLocationCommand`
- [ ] block#8 (powershell) - **Placeholder template (<...>)** - `run.bat : The term 'run.bat' is not recognized as the name of a cmdlet, function, script file, or operable program.`

### `docs/vision/image-classification.mdx` (4)

- [ ] block#0 (powershell) - **Other (review)** - `At line:1 char:4`
- [ ] block#1 (powershell) - **Other (review)** - `line:1 char:1`
- [ ] block#2 (powershell) - **Other (review)** - `At line:1 char:16`
- [ ] block#3 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`

### `docs/vision/nemotron-ocr-v2.mdx` (10)

- [ ] block#0 (powershell) - **Timeout (large clone/build)** - `timeout after 90s`
- [ ] block#2 (powershell) - **Timeout (large clone/build)** - `timeout after 90s`
- [ ] block#3 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#4 (powershell) - **Other (review)** - `of a cmdlet, function,`
- [ ] block#5 (powershell) - **Placeholder template (<...>)** - `All : The term 'All' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the`
- [ ] block#7 (powershell) - **Other (review)** - `ing expression after unary operator '--'.`
- [ ] block#8 (powershell) - **Other (review)** - `ing expression after unary operator '--'.`
- [ ] block#9 (powershell) - **Other (review)** - `ing expression after unary operator '--'.`
- [ ] block#10 (powershell) - **Other (review)** - `ing expression after unary operator '--'.`
- [ ] block#11 (powershell) - **Other (review)** - `ne:5 char:5`

### `docs/vision/torchvision.mdx` (3)

- [ ] block#1 (powershell) - **Other (review)** - `The syntax of the command is incorrect.`
- [ ] block#2 (powershell) - **Other (review)** - `At line:1 char:18`
- [ ] block#3 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`

### `docs/vision/yolov8m.mdx` (18)

- [ ] block#2 (powershell) - **Other (review)** - `At line:1 char:4`
- [ ] block#4 (powershell) - **Other (review)** - `t line:1 char:1`
- [ ] block#5 (powershell) - **Other (review)** - ``
- [ ] block#6 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#7 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#8 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#9 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#10 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#11 (powershell) - **Other (review)** - `nexpected token 'output_model_path' in expression or statement.`
- [ ] block#12 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#13 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#14 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#15 (powershell) - **Other (review)** - `--exclude_subgraphs "[/model.22/Concat_3], [ ...`
- [ ] block#16 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#17 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#18 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#19 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#20 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`

### `docs/vision/yolov8s-worldv2.mdx` (7)

- [ ] block#2 (powershell) - **Placeholder template (<...>)** - `.\download.bat : The term '.\download.bat' is not recognized as the name of a cmdlet, function, script file, or`
- [ ] block#3 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#4 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#5 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#6 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#7 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#8 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`

### `docs/winml-examples/clip.mdx` (8)

- [ ] block#0 (powershell) - **Other (review)** - ``
- [ ] block#1 (powershell) - **Other (review)** - ``
- [ ] block#2 (powershell) - **Other (review)** - `cmdlet,`
- [ ] block#4 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#5 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#6 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#7 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#8 (powershell) - **Other (review)** - `~`

### `docs/winml-examples/googlebert.mdx` (5)

- [ ] block#0 (powershell) - **Missing requirements.txt (needs prior clone/cd)** - `ERROR: Could not open requirements file: [Errno 2] No such file or directory: '.\\requirements.txt'`
- [ ] block#1 (powershell) - **Other (review)** - ``
- [ ] block#2 (powershell) - **Other (review)** - `cmdlet,`
- [ ] block#4 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#5 (powershell) - **Other (review)** - `~`

### `docs/winml-examples/resnet.mdx` (9)

- [ ] block#0 (powershell) - **Missing requirements.txt (needs prior clone/cd)** - `ERROR: Could not open requirements file: [Errno 2] No such file or directory: '.\\requirements.txt'`
- [ ] block#1 (powershell) - **Other (review)** - ``
- [ ] block#2 (powershell) - **Other (review)** - `cmdlet,`
- [ ] block#4 (powershell) - **Other (review)** - `At line:1 char:4`
- [ ] block#5 (powershell) - **Other (review)** - `At line:1 char:4`
- [ ] block#6 (powershell) - **Missing script file (needs example files / cd)** - `C:\Users\bconsolv\AppData\Local\miniforge3\envs\ryzen-ai-ci\python.exe: can't open file 'C:\\Users\\bconsolv\\code\\rai_`
- [ ] block#9 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#10 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#11 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`

### `docs/winml/installation.mdx` (3)

- [ ] block#0 (powershell) - **Other (review)** - `At line:4 char:4`
- [ ] block#1 (powershell) - **Other (review)** - ``
- [ ] block#3 (powershell) - **Other (review)** - `At line:1 char:4`

### `docs/winml/winml_ep.mdx` (4)

- [ ] block#1 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#3 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#5 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`
- [ ] block#7 (python) - **Python error (missing model/var)** - `Traceback (most recent call last):`

### `docs/winml/winml_example.mdx` (9)

- [ ] block#1 (powershell) - **Other (review)** - `At line:3 char:4`
- [ ] block#2 (powershell) - **Other (review)** - ``
- [ ] block#3 (powershell) - **Other (review)** - `cmdlet,`
- [ ] block#4 (powershell) - **Other (review)** - `At line:1 char:4`
- [ ] block#5 (powershell) - **Other (review)** - `At line:1 char:4`
- [ ] block#6 (powershell) - **Other (review)** - `6`
- [ ] block#7 (powershell) - **Other (review)** - `At line:1 char:4`
- [ ] block#8 (powershell) - **Other (review)** - `ognized as the name`
- [ ] block#9 (powershell) - **Other (review)** - `soccer' in expression or statement.`
