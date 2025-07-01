# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
import sys
import os
import shutil


def get_directories():
    current_dir = Path(__file__).resolve().parent

    # models directory for resnet sample
    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    return current_dir, models_dir

def calib_data_formatting():

    source_folder = 'val_images'
    calib_data_path = 'calib_data'

    if not os.path.exists(source_folder):
        print("The provided data path does not exist.")
        sys.exit(1)

    files = os.listdir(source_folder)

    for filename in files:
        if not filename.startswith('ILSVRC2012_val_') or not filename.endswith(
                '.JPEG'):
            continue

        n_identifier = filename.split('_')[-1].split('.')[0]
        folder_name = n_identifier
        folder_path = os.path.join(source_folder, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(source_folder, filename)
        destination = os.path.join(folder_path, filename)
        shutil.move(file_path, destination)

    print("File organization complete.")

    if not os.path.exists(calib_data_path):
        os.makedirs(calib_data_path)

    destination_folder = calib_data_path

    subfolders = os.listdir(source_folder)

    for subfolder in subfolders:
        source_subfolder = os.path.join(source_folder, subfolder)
        destination_subfolder = os.path.join(destination_folder, subfolder)
        os.makedirs(destination_subfolder, exist_ok=True)

        files = os.listdir(source_subfolder)

        if files:
            file_to_copy = files[0]
            source_file = os.path.join(source_subfolder, file_to_copy)
            destination_file = os.path.join(destination_subfolder, file_to_copy)

            shutil.copy(source_file, destination_file)

    print("Creating calibration dataset complete.")
