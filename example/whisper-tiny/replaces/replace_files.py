import os
import shutil
import sys
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def copy_tree(source_file, dest_path):
    if os.path.exists(dest_path):
        shutil.copytree(source_file, dest_path, dirs_exist_ok=True)
        logging.info(f"{source_file} is copied to {dest_path}")
    else:
        shutil.copytree(source_file, dest_path)
        logging.info(f"{source_file} is copied to {dest_path}")


site_path = sys.exec_prefix
conda_capi = os.path.join(site_path, r"lib\site-packages\onnxruntime\capi")
conda_capi_provider = os.path.join(site_path, r"lib\site-packages\onnxruntime\providers")


if __name__ == "__main__":
    copy_tree("capi", conda_capi)
    copy_tree("providers", conda_capi_provider)