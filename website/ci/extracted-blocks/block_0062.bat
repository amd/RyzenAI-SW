# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\llms\hybrid-inference.mdx:93
mkdir llm_run
cd llm_run

:: Copy the sample C++ executable
xcopy /Y "%RYZEN_AI_INSTALLATION_PATH%\LLM\example\model_benchmark.exe" .

:: Copy the sample prompt file
xcopy /Y "%RYZEN_AI_INSTALLATION_PATH%\LLM\example\amd_genai_prompt.txt" .

:: Copy required DLLs
xcopy /Y "%RYZEN_AI_INSTALLATION_PATH%\deployment\." .
