"""
User script for Olive quantization of DeepLabV3 model.
Provides data loading and ImageNet-normalized preprocessing for calibration.
"""

import glob
import numpy as np
from pathlib import Path
from PIL import Image

from olive.data.registry import Registry

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:
        pass


class DeepLabV3Dataset(Dataset):
    """
    Dataset class for DeepLabV3 calibration.
    Loads images and applies ImageNet normalization to match training preprocessing.
    Implements __len__ and __getitem__ for Olive/Quark compatibility.
    """

    INPUT_SIZE = (520, 520)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, data_dir, max_images=20):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if not data_dir.exists():
            # Fallback to the images/ directory beside the model
            data_dir = Path(__file__).parent.parent / "images"

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(str(data_dir / ext)))
            image_files.extend(glob.glob(str(data_dir / "**" / ext), recursive=True))

        image_files = sorted(set(image_files))[:max_images]
        self.image_files = image_files
        print(f"[INFO] DeepLabV3Dataset: loaded {len(self.image_files)} images from {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img = Image.open(self.image_files[index]).convert("RGB")
        img = img.resize(self.INPUT_SIZE, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.MEAN) / self.STD          # HWC, ImageNet-normalized
        arr = arr.transpose((2, 0, 1))              # CHW
        # Quark expects [C, H, W] (no batch dim) per sample
        return {"input": arr}


@Registry.register_dataset()
def deeplabv3_dataset(data_dir=None, **kwargs):
    """Create DeepLabV3 dataset for Olive quantization."""
    if data_dir is None:
        data_dir = Path(__file__).parent / "calib_data"
    else:
        data_dir = Path(data_dir)
    max_images = kwargs.get("max_images", 20)
    return DeepLabV3Dataset(data_dir, max_images)


@Registry.register_pre_process()
def deeplabv3_preprocess(dataset, **kwargs):
    """
    Pass-through pre-process function.
    All preprocessing is already handled in DeepLabV3Dataset.__getitem__.
    """
    return dataset
