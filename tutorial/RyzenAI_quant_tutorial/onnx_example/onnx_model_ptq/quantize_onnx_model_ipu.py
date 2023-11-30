#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

import numpy
import os
from PIL import Image
import shutil

import os
import sys
import numpy as np

import onnx
import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader
import vai_q_onnx


def _preprocess_images(images_folder: str,
                       height: int,
                       width: int,
                       size_limit=0,
                       batch_size=100):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_path = os.listdir(images_folder)
    image_names = []
    for image_dir in image_path:
        image_name = os.listdir(os.path.join(images_folder, image_dir))
        image_names.append(os.path.join(image_dir, image_name[0]))
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    batch_data = []
    for index, image_name in enumerate(batch_filenames):
        image_filepath = images_folder + "/" + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        image_array = numpy.array(pillow_img) / 255.0
        mean = numpy.array([0.485, 0.456, 0.406])
        image_array = (image_array - mean)
        std = numpy.array([0.229, 0.224, 0.225])
        nchw_data = image_array / std
        nchw_data = nchw_data.transpose((2, 0, 1))
        nchw_data = numpy.expand_dims(nchw_data, axis=0)
        nchw_data = nchw_data.astype(numpy.float32)
        unconcatenated_batch_data.append(nchw_data)

        if (index + 1) % batch_size == 0:
            one_batch_data = numpy.concatenate(unconcatenated_batch_data,
                                               axis=0)
            unconcatenated_batch_data.clear()
            batch_data.append(one_batch_data)

    return batch_data


class ImageDataReader(CalibrationDataReader):

    def __init__(self, calibration_image_folder: str, model_path: str,
                 batch_size: int):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(calibration_image_folder,
                                                 height,
                                                 width,
                                                 size_limit=0)
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{
                self.input_name: nhwc_data
            } for nhwc_data in self.nhwc_data_list])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

    def reset(self):
        self.enum_data = None


def main():
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = sys.argv[1]

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = sys.argv[2]

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = sys.argv[3]

    # `dr` (Data Reader) is an instance of ResNet50DataReader, which is a utility class that
    # reads the calibration dataset and prepares it for the quantization process.
    dr = ImageDataReader(calibration_dataset_path, input_model_path, 1)

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
