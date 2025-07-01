import os
import shutil
import argparse
import sys
import pathlib
import onnxruntime
import numpy as np
import pathlib
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Compile BF16 CNN model")
    parser.add_argument("--model", "-i", help="Path to the ONNX model")
    args = parser.parse_args() 

    onnx_model  = args.model
    config_file = 'vitisai_config.json'
    # cache_dir   = 'my_cache_dir'
    
    cache_dir = Path(__file__).parent.resolve()
    cache_dir = os.path.join(cache_dir,'my_cache_dir')

    cache_key   = pathlib.Path(onnx_model).stem
    
    provider_options_dict = {
        "config_file": config_file,
        "cache_dir":   cache_dir,
        "cache_key":   cache_key,
        "enable_cache_file_io_in_mem":0,
    }
   
    print(f"Creating ORT inference session for model {onnx_model}")
    session = onnxruntime.InferenceSession(
        onnx_model,
        providers=["VitisAIExecutionProvider"],
        provider_options=[provider_options_dict]
    )   
 
    print("Done") 
    
 
if __name__ == "__main__":
    main()