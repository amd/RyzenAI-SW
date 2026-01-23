from transformers import AutoTokenizer, AutoModel
import onnxruntime as ort
import torch

# Load model and tokenizer
model_name = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Prepare dummy inputs with fixed sequence length of 512
dummy_text = "exporting to onnx " * 64  # Ensure enough tokens
dummy_inputs = tokenizer(
    dummy_text,
    return_tensors="pt",
    max_length=512,
    padding="max_length",
    truncation=True
)

# Export to ONNX with static shape (1, 512)
torch.onnx.export(
    model,
    (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
    "bge-large-en-v1.5.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes=None,  # Static input only
    opset_version=17
)

print("ONNX model exported with static input shape (1, 512).")

# Create VitisAI session
# Compile the bge ONNX model and store it to cache named  "modelcachekey_bge" folder.
# Will use this cached compiled model for RAG implementation.
model_path = "bge-large-en-v1.5.onnx"
session = ort.InferenceSession(
    model_path,
    providers=["VitisAIExecutionProvider"],
    provider_options=[{
        "config_file": "custom_embedding/vaiml_config.json",
        "cache_dir": "./",
        "cacheKey": "modelcachekey_bge"
    }]
)

print("NPU session created successfully.")
