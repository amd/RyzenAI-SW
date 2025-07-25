import numpy as np
import subprocess
import onnx
import onnxruntime as ort
import json
import os
import argparse
import shutil

from onnx import shape_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Fix dynamic shapes in ONNX file using onnxruntime")
    parser.add_argument("--input_model", required=True, help="Path to the input ONNX file (encoder.onnx or decoder.onnx)")
    args = parser.parse_args()
    with open('config/params.json', 'r') as f:
        args.params_to_fix = json.load(f)
    return args


def directorycreation(directory_name):
    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
        print(f"Directory '{directory_name}' already exists. It has been deleted.")
    os.mkdir(directory_name)


def generate_dummy_data(input_tensor, name):
    input_shape = input_tensor.shape
    input_type = input_tensor.type
    print(f"Generating dummy data for: {name}")
    if input_type == 'tensor(float)':
        return np.random.randint(0, 64, size=input_shape).astype(np.float32)
    elif input_type == 'tensor(float16)':
        return np.random.rand(*input_shape).astype(np.float32)
    elif input_type == 'tensor(int32)':
        return np.random.randint(0, 100, size=input_shape).astype(np.int32)
    elif input_type == 'tensor(int64)' and name == 'attention_mask':
        return np.random.randint(0, 32, size=input_shape).astype(np.int64)
    elif input_type == 'tensor(int64)' and name == 'token_type_ids':
        return np.random.randint(0, 32, size=input_shape).astype(np.int64)
    elif input_type == 'tensor(int64)':
        return np.random.randint(0, 9, size=input_shape).astype(np.int64)
    elif input_type == 'tensor(bool)':
        return np.ones(input_shape).astype(np.bool_)
    elif input_type == 'tensor(double)':
        return np.random.rand(*input_shape).astype(np.float64)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")


def validate_onnx_model(model):
    try:
        onnx.checker.check_model(model)
        print("ONNX model is valid.")
        return True
    except onnx.onnx_cpp2py_export.checker.ValidationError as e:
        print("ONNX model is not valid.")
        print(e)
        return False


if __name__ == "__main__":
    args = parse_args()
    exported_model_directory = "exported_model_directory"
    tmp_dir = "tmp"

    # Prepare workspace
    directorycreation(tmp_dir)

    # Derive filenames
    input_model_path = args.input_model
    base_filename = os.path.basename(input_model_path)
    tmp_output_path = os.path.join(tmp_dir, base_filename)
    final_output_path = os.path.join(exported_model_directory, base_filename)

    # Fix dynamic shapes
    print("WARNING: You might have to comment out ONNX checker in //onnxruntime/tools/onnx_model_utils.py if model > 2GB")
    command_base = ["python", "-m", "onnxruntime.tools.make_dynamic_shape_fixed"]
    for param, value in args.params_to_fix.items():
        command = command_base + [input_model_path, tmp_output_path, "--dim_param", str(param), "--dim_value", str(value)]
        subprocess.run(command)
        input_model_path = tmp_output_path  # use modified model as next input

    print("Static conversion complete.")

    # Shape inference
    print(f"Inferencing shapes for: {tmp_output_path}")
    model = onnx.load(tmp_output_path)
    if not validate_onnx_model(model):
        exit(1)

    inferred_model = shape_inference.infer_shapes(model, data_prop=True)
    onnx.save_model(inferred_model, final_output_path)
    print(f"Shape inference complete. Overwritten: {final_output_path}")

    # Sanity check (forward pass)
    print("---------- Running forward pass ----------------------")
    ort_session = ort.InferenceSession(final_output_path)
    input_data = {
        tensor.name: generate_dummy_data(tensor, tensor.name)
        for tensor in ort_session.get_inputs()
    }
    outputs = ort_session.run(None, input_data)

    # Cleanup
    shutil.rmtree(tmp_dir)
    print(f"Deleted temporary directory: {tmp_dir}")

    print("Model export successful.")

    
