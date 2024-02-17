#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import os
import sys
import numpy as np

import onnx
import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader
import vai_q_onnx


class ImageNetDataset:

    def __init__(
        self,
        data_dir,
        weights,
        **kwargs,
    ):
        super().__init__()
        self.vld_path = data_dir
        self.weights = weights
        self.setup("fit")

    def setup(self, stage: str):
        weights = torchvision.models.get_weight(self.weights)
        preprocessing = weights.transforms()
        self.val_dataset = ImageFolder(self.vld_path, preprocessing)


class GenModelDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_data = sample[0]
        label = sample[1]
        return input_data, label


def create_dataloader(data_dir, weights, batch_size):
    image_dataset = ImageNetDataset(data_dir, weights)
    benchmark_dataloader = DataLoader(GenModelDataset(
        image_dataset.val_dataset),
                                      batch_size=batch_size,
                                      drop_last=True)
    return benchmark_dataloader


class CalibrationDataReader(CalibrationDataReader):

    def __init__(self, data_dir: str, weights: str, batch_size: int = 1):
        super().__init__()
        self.iterator = iter(create_dataloader(data_dir, weights, batch_size))

    def get_next(self) -> dict:
        try:
            return {"input": next(self.iterator)[0].numpy()}
        except Exception:
            return None


def calibration_reader(data_dir, weights, batch_size=1):
    return CalibrationDataReader(data_dir, weights, batch_size=batch_size)


def main():
    # `weights_name` is the name of the weights in the original floating-point Torch model.
    weights_name = sys.argv[1]

    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = sys.argv[2]

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = sys.argv[3]

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = sys.argv[4]

    # `dr` (Data Reader) is an instance of ResNet50DataReader, which is a utility class that
    # reads the calibration dataset and prepares it for the quantization process.
    dr = calibration_reader(calibration_dataset_path, weights_name, 1)
    vai_q_onnx.quantize_static(
        input_model_path,
        output_model_path,
        dr,
        activation_type=vai_q_onnx.QuantType.QUInt8,
        calibrate_method=vai_q_onnx.CalibrationMethod.Percentile,
    )

    print('Calibrated and quantized model saved at:', output_model_path)


if __name__ == '__main__':
    main()
