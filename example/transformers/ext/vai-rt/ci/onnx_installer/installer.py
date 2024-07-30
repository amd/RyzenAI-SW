import os
import shutil
import logging
import platform
import site
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def copy(source_file,dest_path):
    if os.path.exists(source_file):
      logging.info(f"copying {source_file} to {dest_path}")
      shutil.copy(source_file, dest_path)
    else:
      logging.fatal(f"{source_file} does not exist.")

if platform.system() == "Windows":
    source_files = [
        "C:\\Windows\\System32\\AMD\\xrt_core.dll",
        "C:\\Windows\\System32\\AMD\\xrt_coreutil.dll",
        "C:\\Windows\\System32\\AMD\\xrt_phxcore.dll",
        os.path.join(os.getcwd(),"voe-0.1.0-cp39-cp39-win_amd64", "onnxruntime.dll")
    ]
    
    copied = False
    site_package_paths = site.getsitepackages()
    for site_package_path in site_package_paths:
        dest_path = os.path.join(site_package_path + "\\onnxruntime\\capi")
        if os.path.exists(dest_path):
            for source_file in source_files:
                copy(source_file, dest_path)
                copied = True

    if not copied:
        logging.fatal("Installer failed! Please install onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl, \
voe-0.1.0-cp39-cp39-win_amd64.whl and try again.")

else:
    logging.info("This script is intended to run on Windows only.")
