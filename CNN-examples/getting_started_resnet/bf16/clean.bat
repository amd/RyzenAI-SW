@echo off

cd %~dp0

del cpu_inference_summary.json > nul 2>&1
del summary.csv > nul 2>&1
del user_events.csv > nul 2>&1
del xrt.run_summary > nul 2>&1
del original-info-signature.txt > nul 2>&1
del original-model-signature.txt > nul 2>&1

rmdir /S /Q models > nul 2>&1
rmdir /S /Q my_cache_dir > nul 2>&1

call %~dp0\app\clean.bat
call %~dp0\app_vaimlUnarchivePath\clean.bat

cd %~dp0


