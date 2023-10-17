@setlocal

@if "%1"=="" (set ENV_NAME="ms-build-demo") else (set ENV_NAME=%1)
call activate %ENV_NAME%
@if %errorlevel% neq 0 exit /b %errorlevel%

python pyapp\main.py

call conda deactivate
@endlocal
