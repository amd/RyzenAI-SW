@echo off

cd %~dp0

del cpu_inference_summary.json > nul 2>&1
del summary.csv > nul 2>&1
del user_events.csv > nul 2>&1
del xrt.run_summary > nul 2>&1
del original-info-signature.txt > nul 2>&1
del original-model-signature.txt > nul 2>&1
del /Q output.log > nul 2>&1

rmdir /S /Q build > nul 2>&1

