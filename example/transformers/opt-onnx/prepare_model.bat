@echo off
python run_onnx.py --model_name %1 --onnx
optimum-cli export onnx -m %1_smoothquant\pytorch --task text-generation-with-past %1_smoothquant\onnx  --framework pt --no-post-process
:: Generate Ortquantized models
python run_onnx.py --model_name %1 --quantize
xcopy %1_smoothquant\onnx\generation_config.json %1_ortquantized
python ..\onnx-ops\python\group\matmulint\onnx_group_matmul.py --model %1_ortquantized\model_quantized.onnx --output %1_ortquantized\model_quantized.onnx
