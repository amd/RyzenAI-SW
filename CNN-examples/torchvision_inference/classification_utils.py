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
    import os
    import shutil

    source_folder = "val_images"
    calib_data_path = "calib_data"

    if not os.path.isdir(source_folder):
        raise FileNotFoundError(
            f"ImageNet validation directory missing: {source_folder}"
        )

    images = sorted(
        filename for filename in os.listdir(source_folder)
        if filename.startswith("ILSVRC2012_val_")
        and filename.endswith(".JPEG")
        and os.path.isfile(os.path.join(source_folder, filename))
    )

    if len(images) < 100:
        raise RuntimeError(
            f"At least 100 ImageNet validation images required; "
            f"found {len(images)}."
        )

    os.makedirs(calib_data_path, exist_ok=True)

    for filename in images[:100]:
        identifier = filename.split("_")[-1].split(".")[0]

        destination_folder = os.path.join(
            calib_data_path, identifier
        )
        os.makedirs(destination_folder, exist_ok=True)

        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(destination_folder, filename)
        )

    print("Created calibration dataset with 100 images.")
