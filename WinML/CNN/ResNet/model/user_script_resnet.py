"""
User script for Olive quantization of ResNet50 model
Provides data loading and preprocessing for calibration
"""

import numpy as np
from pathlib import Path
from PIL import Image
import glob

# Import Olive registry for function registration
from olive.data.registry import Registry

# Import Dataset base class for proper Olive dataset implementation
try:
    from torch.utils.data import Dataset
except ImportError:
    # Fallback if torch not available - create minimal Dataset class
    class Dataset:
        pass


class ResnetDataset(Dataset):
    """
    Dataset class for ResNet50 calibration

    Implements __len__ and __getitem__ for Olive/Quark compatibility
    """

    def __init__(self, data_dir, max_images=300):
        """
        Initialize dataset with image paths

        Args:
            data_dir: Directory containing calibration images
            max_images: Maximum number of images to use
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if not data_dir.exists():
            # Fallback to example images
            data_dir = Path(__file__).parent.parent / "images"

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(str(data_dir / ext)))
            image_files.extend(glob.glob(str(data_dir / "**" / ext), recursive=True))

        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))

        # Limit to max_images
        if len(image_files) > max_images:
            image_files = image_files[:max_images]

        self.image_files = image_files
        print(f"[INFO] ResnetDataset: Loaded {len(self.image_files)} images from {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Get preprocessed image at index

        Returns:
            Dictionary with preprocessed image tensor (CHW, float32)
        """
        img_path = self.image_files[index]

        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        img_array = np.asarray(img).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std  # HWC

        # Convert to CHW (Quark expects [C,H,W] per sample)
        img_array = img_array.transpose((2, 0, 1))

        return {"input": img_array}


@Registry.register_dataset()
def resnet_dataset(data_dir=None, **kwargs):
    """
    Create ResNet dataset for Olive quantization

    Returns a Dataset instance with __len__ and __getitem__
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "calib_data"
    else:
        data_dir = Path(data_dir)

    max_images = kwargs.get('max_images', 300)

    return ResnetDataset(data_dir, max_images)


@Registry.register_pre_process()
def resnet_preprocess(dataset, **kwargs):
    """
    Preprocess dataset for Olive - pass through the dataset

    The dataset (ResnetDataset) already handles preprocessing in __getitem__,
    so we just return it as-is.
    """
    return dataset
