"""
User script for Olive quantization of ConvNeXt model
Provides data loading and preprocessing for calibration
"""

import numpy as np
from pathlib import Path
from PIL import Image
import glob

from olive.data.registry import Registry

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:
        pass


class ConvNeXtDataset(Dataset):
    """
    Dataset class for ConvNeXt calibration

    Uses the same ImageNet preprocessing as ConvNeXt inference:
    resize to 224x224, normalize with ImageNet mean/std.
    """

    def __init__(self, data_dir, max_images=300):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if not data_dir.exists():
            data_dir = Path(__file__).parent.parent / "images"

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(str(data_dir / ext)))
            image_files.extend(glob.glob(str(data_dir / "**" / ext), recursive=True))

        image_files = sorted(list(set(image_files)))

        if len(image_files) > max_images:
            image_files = image_files[:max_images]

        self.image_files = image_files
        print(f"[INFO] ConvNeXtDataset: Loaded {len(self.image_files)} images from {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]

        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        img_array = np.asarray(img).astype(np.float32) / 255.0

        # ImageNet normalization (same as inference)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std  # HWC

        img_array = img_array.transpose((2, 0, 1))  # CHW

        return {"input": img_array}


@Registry.register_dataset()
def convnext_dataset(data_dir=None, **kwargs):
    """Create ConvNeXt dataset for Olive quantization"""
    if data_dir is None:
        data_dir = Path(__file__).parent / "calib_data"
    else:
        data_dir = Path(data_dir)

    max_images = kwargs.get('max_images', 300)
    return ConvNeXtDataset(data_dir, max_images)


@Registry.register_pre_process()
def convnext_preprocess(dataset, **kwargs):
    """Pass through — ConvNeXtDataset handles preprocessing in __getitem__"""
    return dataset
