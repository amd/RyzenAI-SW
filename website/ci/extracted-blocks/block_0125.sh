# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\getting_started_resnet\bf16\docs\README_C++.mdx:26
cmake -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -B build -S . -G "Visual Studio 17 2022"
cmake --build .\build --config Release --target ALL_BUILD
