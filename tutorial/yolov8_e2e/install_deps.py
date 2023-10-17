import os
import shutil
import logging
import platform

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def copy(source_file,dist_path):
    if os.path.exists(source_file):
        shutil.copy(source_file, dist_path)
    else:
        logging.fatal(f"{source_file} does not exist.")


if platform.system() == "Windows":
    source_files = [
        "C:\\Windows\\System32\\AMD\\xrt_core.dll",
        "C:\\Windows\\System32\\AMD\\xrt_coreutil.dll"
    ]

    for source_file in source_files:
        copy(source_file, "python\\lib\\site-packages\\onnxruntime\\capi")

    if not os.path.exists('1x4.xclbin'):
        copy("C:\\Windows\\System32\\AMD\\1x4.xclbin", ".")
else:
    logging.info("This script is intended to run on Windows only.")
