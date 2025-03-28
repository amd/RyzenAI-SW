@echo off
set RYZEN_AI_CONDA_ENV_NAME=ryzen-ai-1.4.0
set RYZEN_AI_INSTALLATION_PATH=C:/Program Files/RyzenAI/1.4.0
set HWINFO_INSTALLATION_PATH=C:/Program Files/HWiNFO64
set DEVICE=strix
set XLNX_TARGET_NAME=AMD_AIE2P_Nx4_Overlay
set XCLBINHOME=%RYZEN_AI_INSTALLATION_PATH%/voe-4.0-win_amd64/xclbins/%DEVICE%
set VAIP_CONFIG_HOME=%RYZEN_AI_INSTALLATION_PATH%/voe-4.0-win_amd64
set PATH=%CONDA_PREFIX%/Lib/site-packages/flexmlrt/lib/;%PATH%
set PATH=%RYZEN_AI_INSTALLATION_PATH%/utils;%PATH%
set XLNX_VART_FIRMWARE=%RYZEN_AI_INSTALLATION_PATH%/voe-4.0-win_amd64/xclbins/%DEVICE%/AMD_AIE2P_Nx4_Overlay.xclbin
echo Environment variables set. Run "call set_env.bat" in your terminal.
