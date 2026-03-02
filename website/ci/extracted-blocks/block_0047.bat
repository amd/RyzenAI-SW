# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\getting-started\manual-installation.mdx:168
cd %RYZEN_AI_INSTALLATION_PATH%\quicktest
python quicktest.py 2>&1 | findstr /i "Operators Subgraphs VITIS_EP_CPU NPU Test"
