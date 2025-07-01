"""
Script to prepare test images for ResNet CIFAR-10 inference.
This script downloads CIFAR-10 dataset and extracts some sample images
for testing with the C++ inference code.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms

# Directory to save the images
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
os.makedirs(output_dir, exist_ok=True)

# Download CIFAR10 dataset (test split)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transforms.ToTensor())

# Extract one image from each class
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

for class_idx in range(10):
    # Find the first image of this class
    for i, (image, label) in enumerate(testset):
        if label == class_idx:
            # Save the raw image bytes in CIFAR-10 binary format (with the label byte)
            filename = os.path.join(output_dir, f"{class_names[class_idx]}.bin")
            
            # Convert tensor back to bytes (CIFAR-10 format: R channel, G channel, B channel)
            raw_data = bytearray([label])  # First byte is the label
            
            # Append image data in CIFAR-10 format (all R, then all G, then all B)
            # Convert from [0,1] float to [0,255] byte
            for c in range(3):  # RGB channels
                for h in range(32):  # Height
                    for w in range(32):  # Width
                        pixel_value = int(image[c][h][w] * 255)
                        raw_data.append(pixel_value)
            
            # Write to binary file
            with open(filename, 'wb') as f:
                f.write(raw_data)
            
            print(f"Saved {filename}")
            break

print("Done preparing test images")
