"""
User script for Olive quantization of YOLOv8 model
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


class YOLOv8Dataset(Dataset):
    """
    Dataset class for YOLOv8 calibration

    Implements __len__ and __getitem__ for Olive/Quark compatibility
    """

    def __init__(self, data_dir, max_images=200):
        """
        Initialize dataset with image paths

        Args:
            data_dir: Directory containing calibration images
            max_images: Maximum number of images to use
        """
        # Find all image files
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
        print(f"[INFO] YOLOv8Dataset: Loaded {len(self.image_files)} images from {data_dir}")

    def __len__(self):
        """Return number of images in dataset"""
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Get preprocessed image at index

        Args:
            index: Image index

        Returns:
            Dictionary with preprocessed image tensor
        """
        img_path = self.image_files[index]

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((640, 640), Image.BILINEAR)
        img_array = np.array(img_resized).astype(np.float32)

        # YOLOv8 normalization: [0, 255] -> [0, 1]
        img_array = img_array / 255.0

        # Convert to NCHW format: [H,W,C] -> [C,H,W]
        img_array = img_array.transpose((2, 0, 1))

        # Note: No batch dimension for calibration (Quark expects [C,H,W])

        return {"images": img_array}


@Registry.register_dataset()
def yolov8_dataset(data_dir=None, **kwargs):
    """
    Create YOLOv8 dataset for Olive quantization

    Returns a Dataset class instance with __len__ and __getitem__
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "calib_data"
    else:
        data_dir = Path(data_dir)

    max_images = kwargs.get('max_images', 200)

    return YOLOv8Dataset(data_dir, max_images)


@Registry.register_pre_process()
def yolov8_preprocess(dataset, **kwargs):
    """
    Preprocess dataset for Olive - pass through the dataset

    The dataset (YOLOv8Dataset) already handles preprocessing in __getitem__,
    so we just return it as-is. This must return the dataset object, not iterate it.
    """
    # Return the dataset unchanged - preprocessing happens in __getitem__
    return dataset


# Internal preprocessing function for individual images
def pre_process(image_path, **kwargs):
    """
    Preprocess individual image for YOLOv8 model

    Args:
        image_path: Path to input image or PIL Image

    Returns:
        Dictionary with preprocessed image tensor
    """
    # Load image
    if isinstance(image_path, (str, Path)):
        img = Image.open(image_path).convert('RGB')
    else:
        img = image_path

    # Resize to model input size [640, 640]
    img_resized = img.resize((640, 640), Image.BILINEAR)

    # Convert to numpy array (HWC format)
    img_array = np.array(img_resized).astype(np.float32)

    # YOLOv8 normalization: [0, 255] -> [0, 1]
    img_array = img_array / 255.0

    # Convert to NCHW format: [H,W,C] -> [C,H,W]
    img_array = img_array.transpose((2, 0, 1))

    # Add batch dimension: [C,H,W] -> [1,C,H,W]
    img_array = np.expand_dims(img_array, axis=0)

    # Return as dictionary with input name
    return {"images": img_array}
