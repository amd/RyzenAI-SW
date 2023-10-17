SET PWD=%~dp0\..\
SET THIRD_PARTY=%PWD%\third_party
set PYTORCH_AIE_PATH=%PWD%
SET PYTHONPATH=%PYTHONPATH%;%PWD%\ext\smoothquant\smoothquant
set DEVICE=aieml

set XLNX_VART_FIRMWARE=%PWD%\xclbin\aieml
set TVM_MODULE_PATH=%PWD%\dll\%DEVICE%\qlinear\libGemmQnnAie_8x2048_2048x2048.dll
