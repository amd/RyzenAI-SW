"""
RetinaFace Anchor Generation Utilities

This module provides anchor box generation for RetinaFace face detection model.
RetinaFace uses a feature pyramid network (FPN) with 3 levels to generate
multi-scale anchor boxes.

References:
- RetinaFace: https://arxiv.org/abs/1905.00641
- Feature pyramid anchors for 640x608 input
"""

import numpy as np
from typing import List, Tuple, Dict


# RetinaFace configuration for 640x608 input
RETINAFACE_CONFIG = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],  # Anchor sizes for P3, P4, P5
    'steps': [8, 16, 32],                             # Feature map strides
    'variance': [0.1, 0.2],                           # Variance for bbox decoding
    'clip': False,                                    # Whether to clip anchors to [0,1]
    'image_size': (640, 608)                          # (width, height)
}


class PriorBox:
    """
    Generate prior (anchor) boxes for RetinaFace detection.

    This class implements the anchor generation strategy used in RetinaFace,
    which creates multi-scale anchors across different feature pyramid levels.

    For input size 640x608:
    - P3 (stride 8):  80x76 feature map  -> 12,160 anchors (80*76*2)
    - P4 (stride 16): 40x38 feature map  -> 3,040 anchors  (40*38*2)
    - P5 (stride 32): 20x19 feature map  -> 760 anchors    (20*19*2)
    Total: 15,960 anchors
    """

    def __init__(self, config: Dict = None):
        """
        Initialize PriorBox generator.

        Args:
            config: Configuration dictionary with keys:
                   - min_sizes: List of anchor sizes per level
                   - steps: List of feature map strides
                   - variance: Variance values for bbox decoding
                   - clip: Whether to clip anchors to image bounds
                   - image_size: (width, height) tuple
        """
        self.config = config if config is not None else RETINAFACE_CONFIG
        self.min_sizes = self.config['min_sizes']
        self.steps = self.config['steps']
        self.variance = self.config['variance']
        self.clip = self.config.get('clip', False)
        self.image_size = self.config['image_size']

    def forward(self) -> np.ndarray:
        """
        Generate all anchor boxes for the configured image size.

        Returns:
            anchors: numpy array of shape (num_anchors, 4)
                    Each anchor is [cx, cy, w, h] normalized to [0, 1]
        """
        anchors = []

        for k, (min_size_level, step) in enumerate(zip(self.min_sizes, self.steps)):
            # Calculate feature map dimensions
            # For 640x608 input:
            # P3: (640/8, 608/8) = (80, 76)
            # P4: (640/16, 608/16) = (40, 38)
            # P5: (640/32, 608/32) = (20, 19)
            feature_w = self.image_size[0] // step
            feature_h = self.image_size[1] // step

            # Generate anchors for this feature level
            for i in range(feature_h):
                for j in range(feature_w):
                    # Center coordinates in pixel space
                    cx = (j + 0.5) * step
                    cy = (i + 0.5) * step

                    # Normalize to [0, 1]
                    cx_norm = cx / self.image_size[0]
                    cy_norm = cy / self.image_size[1]

                    # Create anchors with different sizes at this location
                    for min_size in min_size_level:
                        # Width and height in normalized coordinates
                        w_norm = min_size / self.image_size[0]
                        h_norm = min_size / self.image_size[1]

                        anchors.append([cx_norm, cy_norm, w_norm, h_norm])

        anchors = np.array(anchors, dtype=np.float32)

        if self.clip:
            anchors = np.clip(anchors, 0, 1)

        return anchors


def generate_priors(image_size: Tuple[int, int] = (640, 608)) -> np.ndarray:
    """
    Convenience function to generate RetinaFace prior boxes.

    Args:
        image_size: (width, height) tuple for input image

    Returns:
        anchors: numpy array of shape (num_anchors, 4)
                Each anchor is [cx, cy, w, h] normalized to [0, 1]
    """
    config = RETINAFACE_CONFIG.copy()
    config['image_size'] = image_size

    prior_box = PriorBox(config)
    return prior_box.forward()


def decode_boxes(loc: np.ndarray, priors: np.ndarray, variances: List[float] = [0.1, 0.2]) -> np.ndarray:
    """
    Decode bounding box predictions using prior anchors.

    Args:
        loc: Predicted box offsets, shape (num_boxes, 4) as [dx, dy, dw, dh]
        priors: Prior anchor boxes, shape (num_boxes, 4) as [cx, cy, w, h]
        variances: Variance values for decoding [variance_xy, variance_wh]

    Returns:
        boxes: Decoded boxes in format [cx, cy, w, h], shape (num_boxes, 4)
    """
    # Decode center coordinates
    boxes_cx = priors[:, 0] + loc[:, 0] * variances[0] * priors[:, 2]
    boxes_cy = priors[:, 1] + loc[:, 1] * variances[0] * priors[:, 3]

    # Decode width and height
    boxes_w = priors[:, 2] * np.exp(loc[:, 2] * variances[1])
    boxes_h = priors[:, 3] * np.exp(loc[:, 3] * variances[1])

    boxes = np.stack([boxes_cx, boxes_cy, boxes_w, boxes_h], axis=1)
    return boxes


def convert_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Args:
        boxes: Boxes in [cx, cy, w, h] format, shape (num_boxes, 4)

    Returns:
        boxes_xyxy: Boxes in [x1, y1, x2, y2] format, shape (num_boxes, 4)
    """
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    return np.stack([x1, y1, x2, y2], axis=1)


def generate_anchors(image_size: Tuple[int, int] = (640, 608)) -> np.ndarray:
    """
    Generate anchors for RetinaFace (alias for generate_priors).

    Args:
        image_size: (width, height) tuple for input image

    Returns:
        anchors: numpy array of shape (num_anchors, 4)
    """
    return generate_priors(image_size)


# Self-test code
if __name__ == "__main__":
    print("=" * 60)
    print("RetinaFace Anchor Generation Test")
    print("=" * 60)

    # Generate anchors
    print("\nGenerating anchors for 640x608 input...")
    anchors = generate_priors(image_size=(640, 608))

    # Validate anchor count
    print(f"\nAnchor count: {len(anchors)}")
    print(f"Anchor shape: {anchors.shape}")
    print(f"Expected count: 15,960")

    # Check if count matches
    expected_count = 15960
    if len(anchors) == expected_count:
        print("[PASS] Anchor count matches expected!")
    else:
        print(f"[FAIL] Anchor count mismatch! Expected {expected_count}, got {len(anchors)}")

    # Validate normalization
    print(f"\nAnchor value range:")
    print(f"  Min: {anchors.min():.6f}")
    print(f"  Max: {anchors.max():.6f}")

    if anchors.min() >= 0 and anchors.max() <= 1:
        print("[PASS] All anchors normalized to [0, 1]")
    else:
        print("[FAIL] Anchors not properly normalized!")

    # Show sample anchors from each level
    print("\nSample anchors from each feature level:")

    # P3 level (first 2 anchors)
    print(f"\nP3 (stride 8, first location):")
    print(f"  Anchor 0: {anchors[0]}")
    print(f"  Anchor 1: {anchors[1]}")

    # P4 level (anchors after P3)
    p3_count = (640 // 8) * (608 // 8) * 2  # 12,160
    print(f"\nP4 (stride 16, first location):")
    print(f"  Anchor {p3_count}: {anchors[p3_count]}")
    print(f"  Anchor {p3_count + 1}: {anchors[p3_count + 1]}")

    # P5 level (anchors after P3 and P4)
    p4_count = (640 // 16) * (608 // 16) * 2  # 3,040
    p5_start = p3_count + p4_count
    print(f"\nP5 (stride 32, first location):")
    print(f"  Anchor {p5_start}: {anchors[p5_start]}")
    print(f"  Anchor {p5_start + 1}: {anchors[p5_start + 1]}")

    # Verify feature level counts
    print("\nFeature level breakdown:")
    print(f"  P3 (80x76x2): {p3_count} anchors")
    print(f"  P4 (40x38x2): {p4_count} anchors")
    print(f"  P5 (20x19x2): {len(anchors) - p3_count - p4_count} anchors")

    # Test decoding with sample predictions
    print("\n" + "=" * 60)
    print("Testing anchor decoding...")
    print("=" * 60)

    # Create sample predictions (zeros = no offset from anchor)
    sample_loc = np.zeros((len(anchors), 4), dtype=np.float32)
    decoded = decode_boxes(sample_loc, anchors)

    print(f"\nDecoded boxes shape: {decoded.shape}")
    print(f"Sample decoded box (should match anchor): {decoded[0]}")
    print(f"Original anchor: {anchors[0]}")

    if np.allclose(decoded[0], anchors[0], atol=1e-6):
        print("[PASS] Zero-offset decoding works correctly!")
    else:
        print("[FAIL] Decoding issue detected!")

    # Test coordinate conversion
    print("\nTesting coordinate conversion to xyxy format...")
    boxes_xyxy = convert_to_xyxy(anchors[:5])
    print(f"First 5 anchors in xyxy format:")
    for i, box in enumerate(boxes_xyxy):
        print(f"  Anchor {i}: [{box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f}]")

    print("\n" + "=" * 60)
    print("Anchor generation test complete!")
    print("=" * 60)
