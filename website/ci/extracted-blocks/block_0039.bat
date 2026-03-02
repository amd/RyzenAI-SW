# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\operator-preparation.mdx:79
:: Optional: enable tracing
set DD_PLUGINS_TRACING=1

:: Generate the model
model_generate --hybrid <output hybrid model folder> <dml model folder>
