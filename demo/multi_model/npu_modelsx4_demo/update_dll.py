import os
import shutil

def copy_files(source_dir, destination_dir, file_ext):
    """
    Copies files with the specified extension from source_dir to destination_dir.
    """
    if not os.path.exists(source_dir):
        print(f"ERROR: Source directory '{source_dir}' does not exist.")
        return

    os.makedirs(destination_dir, exist_ok=True)  # Ensure destination exists

    for file_name in os.listdir(source_dir):
        if file_name.lower().endswith(file_ext):  # Filter files by extension
            source_file = os.path.join(source_dir, file_name)
            dest_file = os.path.join(destination_dir, file_name)
            shutil.copy2(source_file, dest_file)
            print(f"Copied: {source_file} -> {dest_file}")

if __name__ == "__main__":
    # Detect latest RyzenAI version
    ryzen_ai_base = os.path.join("C:", os.sep, "Program Files", "RyzenAI")
    versions = [v for v in os.listdir(ryzen_ai_base) if os.path.isdir(os.path.join(ryzen_ai_base, v))]
    
    if not versions:
        print("ERROR: No RyzenAI versions found.")
        exit(1)
    
    latest_version = sorted(versions, reverse=True)[0]
    print(f"Detected RyzenAI version: {latest_version}")

    # Define source paths
    source_dlls = os.path.join(ryzen_ai_base, latest_version, "onnxruntime", "bin")
    source_vaip = os.path.join(ryzen_ai_base, latest_version, "voe-4.0-win_amd64", "vaip_config.json")

    # Define the base path for multi_model
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory (npu_modelsx4_demo)
    multi_model_dir = os.path.abspath(os.path.join(script_dir, ".."))  # Move up to multi_model directory
    destination_bin = os.path.join(multi_model_dir, "bin")  # Target bin directory

    print(f"Copying DLLs from: {source_dlls} to {destination_bin}")
    copy_files(source_dlls, destination_bin, ".dll")

    print(f"Copying vaip_config.json from: {source_vaip} to {destination_bin}")
    if os.path.exists(source_vaip):
        shutil.copy2(source_vaip, destination_bin)
        print(f"Copied: {source_vaip} -> {destination_bin}")
    else:
        print(f"ERROR: '{source_vaip}' not found.")