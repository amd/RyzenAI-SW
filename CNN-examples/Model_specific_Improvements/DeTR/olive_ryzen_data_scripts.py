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
    img = cv2.resize(img, (800, 800))
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
    def __init__(self, dataset_path, transpose=False):
        self._transpose = transpose
        self._files = []
        all_files_in_dir = os.listdir(dataset_path)
        for name in all_files_in_dir:
            file = os.path.join(dataset_path, name)
            if os.path.isfile(file):
                if cv2.haveImageReader(file):
                    self._files.append(file)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of dataset size")

        image = cv2.imread(self._files[index]) # read image with OpenCV
        image = preprocess_input(image, transpose=self._transpose)
        return {"input": image}, np.array([0])  # returning dummy label, replace with actual label if available

@Registry.register_dataset()
def my_dataset(data_dir, **kwargs):
    transpose = kwargs.get("transpose", False)
    print("Creating MyDataset with transpose =", transpose)
    return MyDataset(data_dir, transpose=transpose)

@Registry.register_pre_process()
def my_pre_process(input, **kwargs):
    return input

@Registry.register_post_process()
def my_post_process(output, **kwargs):
    return output
