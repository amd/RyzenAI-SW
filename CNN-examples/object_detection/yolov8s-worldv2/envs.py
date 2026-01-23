# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "images"

COCO_DATA_ROOT = Path("C:\\Users\\Administrator\\Desktop\\max\\datasets\\COCO")

sys.path.append(PROJECT_DIR.as_posix())
