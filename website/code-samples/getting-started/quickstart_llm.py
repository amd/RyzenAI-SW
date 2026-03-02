"""Run a simple LLM chat completion on AMD Ryzen AI NPU."""
import os
import sys
import onnxruntime_genai as og

model_path = "models/Llama-3.2-1B-Instruct-onnx-ryzenai-hybrid"
if not os.path.isdir(model_path):
    print(f"Error: Model directory '{model_path}' not found. Run the download step in the quickstart guide first.")
    sys.exit(1)

model = og.Model(model_path)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

params = og.GeneratorParams(model)
params.set_search_options(max_length=256)

prompt = "<|user|>\nWhat is an NPU and how does it differ from a GPU?\n<|assistant|>\n"
input_tokens = tokenizer.encode(prompt)
params.input_ids = input_tokens

generator = og.Generator(model, params)

print("Assistant: ", end="", flush=True)
while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()
    token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(token), end="", flush=True)
print()
