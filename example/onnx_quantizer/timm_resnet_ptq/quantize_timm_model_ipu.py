#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from timm.data import resolve_data_config
from timm.models import create_model
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
        **kwargs,
    ):
        super().__init__()
        self.train_path = data_dir
        self.vld_path = data_dir
        self.setup("fit")

    def setup(self, stage: str):
        model_in = "resnet50.tv_in1k"
        in_chans = 3
        model = create_model(
            model_in,
            pretrained=False,
            num_classes=1000,
            in_chans=in_chans,
            global_pool=None,
            scriptable=False,
        )

        data_config = resolve_data_config(
            model=model,
            use_test_size=True,
        )
        mean = list(data_config['mean'])
        std = list(data_config['std'])
        normalize = transforms.Normalize(mean=mean, std=std)
        in_size = data_config['input_size'][1]

        self.train_dataset = ImageFolder(
            self.train_path,
            transforms.Compose([
                transforms.RandomResizedCrop(in_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        self.val_dataset = ImageFolder(
            self.vld_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(in_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))


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


def create_dataloader(data_dir, batch_size):
    image_dataset = ImageNetDataset(data_dir)
    benchmark_dataloader = DataLoader(GenModelDataset(
        image_dataset.val_dataset),
                                      batch_size=batch_size,
                                      drop_last=True)
    return benchmark_dataloader


class CalibrationDataReader(CalibrationDataReader):

    def __init__(self, data_dir: str, batch_size: int = 1):
        super().__init__()
        self.iterator = iter(create_dataloader(data_dir, batch_size))

    def get_next(self) -> dict:
        try:
            return {"input": next(self.iterator)[0].numpy()}
        except Exception:
            return None


def calibration_reader(data_dir, batch_size=1):
    return CalibrationDataReader(data_dir, batch_size=batch_size)


def main():
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = sys.argv[1]

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = sys.argv[2]

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = sys.argv[3]

    # `dr` (Data Reader) is an instance of ResNet50DataReader, which is a utility class that
    # reads the calibration dataset and prepares it for the quantization process.
    dr = calibration_reader(calibration_dataset_path, 1)
    vai_q_onnx.quantize_static(
        input_model_path,
        output_model_path,
        dr,
        activation_type=vai_q_onnx.QuantType.QUInt8,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
        enable_dpu=True,
        extra_options={
            'ActivationSymmetric': True,
        })

    print('Calibrated and quantized model saved at:', output_model_path)


if __name__ == '__main__':
    main()
