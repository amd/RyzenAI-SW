# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\llms\python-api.mdx:52
from lemonade.api import from_pretrained

model, tokenizer = from_pretrained(
    "amd/Llama-3.2-1B-Instruct-onnx-ryzenai-hybrid", recipe="oga-hybrid"
)

input_ids = tokenizer("This is my prompt", return_tensors="pt").input_ids
response = model.generate(input_ids, max_new_tokens=30)

print(tokenizer.decode(response[0]))
