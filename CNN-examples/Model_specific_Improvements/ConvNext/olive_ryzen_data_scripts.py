#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import numpy as np
from torch.utils.data import Dataset
import cv2
import os

from olive.data.registry import Registry

def preprocess_input(img, transpose):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    
    if(transpose):
        # If transpose is True, convert from HWC to CHW format
        img = np.transpose(img, axes=[2, 0, 1])
        
        norm_img_data = np.zeros(img.shape).astype('float32')
        
        for i in range(img.shape[0]) :
            # for each pixel in each channel, divide the value by 255 to get value between[0, 1] and then normalize
            norm_img_data[i, :, : ] = (img[i, :, : ] / 255 - mean_vec[i]) / stddev_vec[i]
            
    else:
        norm_img_data = np.zeros(img.shape).astype('float32')
        # If transpose is False, keep the image in HWC format
        for i in range(img.shape[2]) :
            # for each pixel in each channel, divide the value by 255 to get value between[0, 1] and then normalize
            norm_img_data[:, :, i] = (img[:, :, i] / 255 - mean_vec[i]) / stddev_vec[i]
    
    return norm_img_data

class MyDataset(Dataset):
    def __init__(self, dataset_path, transpose=False, list_file=None):
        self._transpose = transpose
        self._files = []
        if list_file:
            # Calibration order matters (Entropy histograms are merged incrementally),
            # so the order is pinned by list_file instead of os.listdir().
            with open(list_file) as f:
                names = [line.strip() for line in f if line.strip()]
            for name in names:
                file = os.path.join(dataset_path, name)
                if not os.path.isfile(file):
                    raise FileNotFoundError(f"calibration image listed in {list_file} is missing: {file}")
                self._files.append(file)
            images = [n for n in os.listdir(dataset_path) if cv2.haveImageReader(os.path.join(dataset_path, n))]
            if len(images) != len(names):
                raise ValueError(f"{dataset_path} has {len(images)} images, {list_file} lists {len(names)}")
            print(f"MyDataset: {len(self._files)} calibration images in fixed order from {list_file}")
        else:
            all_files_in_dir = os.listdir(dataset_path)
            for name in all_files_in_dir:
                file = os.path.join(dataset_path, name)
                if cv2.haveImageReader(file):
                    self._files.append(file)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of dataset size")
        
        # Load and pre-process the input
        img = cv2.imread(self._files[index])
        img = preprocess_input(img, self._transpose)
        return {"input": img}, np.array([0])  # returning dummy label, replace with actual label if available

@Registry.register_dataset()
def my_dataset(data_dir, **kwargs):
    transpose = kwargs.get("transpose", False)
    print("Creating MyDataset with transpose =", transpose)
    return MyDataset(data_dir, transpose=transpose,
                     list_file=kwargs.get("list_file"))

@Registry.register_pre_process()
def my_pre_process(input, **kwargs):
    return input

@Registry.register_post_process()
def my_post_process(output, **kwargs):
    return output
