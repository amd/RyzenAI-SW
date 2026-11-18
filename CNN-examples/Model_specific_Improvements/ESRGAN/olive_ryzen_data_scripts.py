#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import numpy as np
from torch.utils.data import Dataset
import cv2
import os

from olive.data.registry import Registry

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def preprocess_input(img, transpose):
    img = center_crop(img, (250, 250))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.array(img).astype('float32')
    
    if transpose:
        img = np.transpose(img, axes=[2, 0, 1])
    return img

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')
MANIFEST_NAME = 'manifest.txt'

def list_calibration_images(dataset_path):
	"""Returns the image file names in dataset_path in calibration order.

	The order changes the quantization result, so it is fixed: if manifest.txt exists (one file
	name per line, '#' = comment), it gives the order and must list exactly the images in the
	folder; otherwise the names are sorted.
	"""
	names = sorted(n for n in os.listdir(dataset_path)
	               if n.lower().endswith(IMAGE_EXTENSIONS) and cv2.haveImageReader(os.path.join(dataset_path, n)))
	manifest = os.path.join(dataset_path, MANIFEST_NAME)
	if not os.path.isfile(manifest):
		return names
	ordered = []
	with open(manifest, encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if line and not line.startswith('#'):
				ordered.append(line)
	if sorted(ordered) != names:
		raise ValueError(f"{manifest} lists {sorted(ordered)} but the folder has {names}")
	return ordered

class MyDataset(Dataset):
    def __init__(self, dataset_path, transpose=False, calib_sz=200):
        self._transpose = transpose
        self.sz = calib_sz
        self._files = [os.path.join(dataset_path, n) for n in list_calibration_images(dataset_path)]
        print(f"Calibration images ({len(self._files)}):", [os.path.basename(f) for f in self._files])

    def __len__(self):
        return min(len(self._files), self.sz)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of dataset size")

        image = cv2.imread(self._files[index]) # read image with OpenCV
        image = preprocess_input(image, transpose=self._transpose)
        return {"input": image}, np.array([0])  # returning dummy label, replace with actual label if available

@Registry.register_dataset()
def my_dataset(data_dir, **kwargs):
    transpose = kwargs.get("transpose", False)
    sz = kwargs.get("calib_sz", 200)
    print("Creating MyDataset with transpose =", transpose)
    return MyDataset(data_dir, transpose=transpose, calib_sz=sz)

@Registry.register_pre_process()
def my_pre_process(input, **kwargs):
    return input

@Registry.register_post_process()
def my_post_process(output, **kwargs):
    return output
