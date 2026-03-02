@echo off

powershell -command "build\Release\app.exe | Tee-object output.log"
:: Check the return code to see if the program returned as expected
if %errorlevel% equ 1 goto error 

:: Check the log to see if final expected message was printed
find /c "Test Done." output.log >NUL
if %errorlevel% equ 1 goto unsuccessful

:: Check the log to see if ops were on CPU or NPU (VAIML)
find /c "[Vitis AI EP] No. of Operators : VAIML" output.log >NUL
if %errorlevel% equ 1 goto no_vaiml
goto success

:success
powershell write-host -fore Green SUCCESS: Model ran on NPU
goto done

:no_vaiml
powershell write-host -fore Red ERROR: Model did not run on NPU
goto done

:unsuccessful
echo.
powershell write-host -fore Red ERROR: Test did not generate the final expected message
goto done

:error
echo.
powershell write-host -fore Red ERROR: Program returned a non-zero value
goto done

:done

