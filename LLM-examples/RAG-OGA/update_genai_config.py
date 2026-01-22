import os
import json

# Get the absolute path to the custom ops DLL
custom_ops_path = os.path.abspath("./onnx_custom_ops.dll")

print(custom_ops_path)

# Path to the config file
config_path = "./model/genai_config.json"  # Adjust if needed

# Load the existing config
with open(config_path, "r") as f:
    config = json.load(f)

# Inject the full path to the DLL into the config
config["model"]["decoder"]["session_options"]["custom_ops_library"] = custom_ops_path

# Save the updated config
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Updated config with custom ops path: {custom_ops_path}")
