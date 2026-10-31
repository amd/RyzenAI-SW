import os
import subprocess
import shutil

import onnxruntime as ort
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path

def get_npu_info():
    # Run pnputil as a subprocess to enumerate PCI devices
    command = r'pnputil /enum-devices /bus PCI /deviceids '
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # Check for supported Hardware IDs
    npu_type = ''
    if 'PCI\\VEN_1022&DEV_1502&REV_00' in stdout.decode(): npu_type = 'PHX/HPT'
    if 'PCI\\VEN_1022&DEV_17F0&REV_00' in stdout.decode(): npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_10' in stdout.decode(): npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_11' in stdout.decode(): npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_20' in stdout.decode(): npu_type = 'KRK'
    return npu_type

def get_xclbin(npu_device):
    xclbin_file = ''
    if npu_device == 'STX' or npu_device=='KRK':
        xclbin_file = '{}\\voe-4.0-win_amd64\\xclbins\\strix\\AMD_AIE2P_4x4_Overlay.xclbin'.format(os.environ["RYZEN_AI_INSTALLATION_PATH"])
    if npu_device == 'PHX/HPT':
        xclbin_file = '{}\\voe-4.0-win_amd64\\xclbins\\phoenix\\4x4.xclbin'.format(os.environ["RYZEN_AI_INSTALLATION_PATH"])
    return xclbin_file

def reorganize_imagenet_val(val_dir, mapping_file, output_dir):
    # Read the mapping file
    with open(mapping_file, 'r') as f:
        lines = f.readlines()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each line in the mapping file
    for line in lines:
        image_name, class_label = line.strip().split()

        # Create class directory if it doesn't exist
        class_dir = os.path.join(output_dir, class_label)
        os.makedirs(class_dir, exist_ok=True)

        # Move the image to the class directory
        src = os.path.join(val_dir, image_name)
        dst = os.path.join(class_dir, image_name)
        shutil.move(src, dst)

# Example usage
# reorganize_imagenet_val('path/to/val_images', 'path/to/mapping_file.txt', 'path/to/output_dir')
def get_npu_info():
    # Run pnputil as a subprocess to enumerate PCI devices
    command = r'pnputil /enum-devices /bus PCI /deviceids '
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # Check for supported Hardware IDs
    npu_type = ''
    if 'PCI\\VEN_1022&DEV_1502&REV_00' in stdout.decode(): npu_type = 'PHX/HPT'
    if 'PCI\\VEN_1022&DEV_17F0&REV_00' in stdout.decode(): npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_10' in stdout.decode(): npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_11' in stdout.decode(): npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_20' in stdout.decode(): npu_type = 'KRK'
    return npu_type

def evaluate_onnx_model(onnx_model_path, imagenet_data_path, batch_size=1, device='cpu', num_samples=100):
    # Load the ONNX model
    if device == 'npu':
        npu_device = get_npu_info()
        provider = ['VitisAIExecutionProvider']
        cache_dir = Path(__file__).parent.resolve()
        model_name = os.path.basename(onnx_model_path)[:-5]
        print(cache_dir)
        provider_options = [{}]
        if npu_device == 'PHX/HPT':
            provider_options[0]['target'] = 'X1'
            provider_options[0]['xclbin'] = get_xclbin(npu_device)
        session = ort.InferenceSession(onnx_model_path, providers=provider,
                                       provider_options=provider_options)
    else:
        # session = ort.InferenceSession(onnx_model_path)
        if 'BF16' in onnx_model_path:
            from quark.onnx import get_library_path
            sess_options = ort.SessionOptions()
            sess_options.register_custom_ops_library(get_library_path(device='cpu'))
            session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'], sess_options=sess_options)
        else:
            session = ort.InferenceSession(onnx_model_path)

    input_name = session.get_inputs()[0].name

    # Define the preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the ImageNet validation dataset
    imagenet_data = datasets.ImageFolder(root=imagenet_data_path, transform=preprocess)
    if len(imagenet_data) == 0:
        raise ValueError(f"No validation images found in {imagenet_data_path}")
    # Limit to num samples using subnet
    subset_indices = range(min(num_samples, len(imagenet_data)))
    imagenet_subset = Subset(imagenet_data, subset_indices)
    data_loader = DataLoader(imagenet_subset, batch_size=batch_size, shuffle=False)
    # data_loader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=False)

    top1_correct = 0
    top5_correct = 0
    total = 0

    # Evaluate the model
    for images, labels in tqdm(data_loader, desc="Evaluating"):
        # Run inference
        outputs = session.run(None, {input_name: images.numpy()})
        outputs = outputs[0]

        # Calculate top-1 and top-5 predictions
        top1_predictions = np.argmax(outputs, axis=1)
        top5_predictions = np.argsort(outputs, axis=1)[:, -5:]

        # Update top-1 accuracy
        top1_correct += (top1_predictions == labels.numpy()).sum()

        # Update top-5 accuracy
        for i, label in enumerate(labels.numpy()):
            if label in top5_predictions[i]:
                top5_correct += 1

        total += labels.size(0)

    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total

    return top1_accuracy, top5_accuracy

#    print(f"Accuracy: {accuracy * 100:.2f}%")
import numpy
from PIL import Image
import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader


def _preprocess_images(
    images_folder: str,
    height: int,
    width: int,
    size_limit=0,
    batch_size=100
):
    root = Path(images_folder)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Calibration directory not found: {root.resolve()}"
        )

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if size_limit > 0:
        image_files = image_files[:size_limit]

    if not image_files:
        raise ValueError(
            f"No calibration images found in {root.resolve()}"
        )

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    mean = numpy.array([0.485, 0.456, 0.406])
    std = numpy.array([0.229, 0.224, 0.225])

    batches = []
    pending = []

    for image_path in image_files:
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((width, height))
            array = numpy.asarray(img, dtype=numpy.float32) / 255.0

        array = (array - mean) / std
        array = array.transpose((2, 0, 1))
        pending.append(array[numpy.newaxis].astype(numpy.float32))

        if len(pending) == batch_size:
            batches.append(numpy.concatenate(pending, axis=0))
            pending.clear()

    if pending:
        batches.append(numpy.concatenate(pending, axis=0))

    print(f"[CALIB] Path: {root.resolve()}")
    print(f"[CALIB] Images: {len(image_files)}")
    print(f"[CALIB] Batches: {len(batches)}")

    return batches



class ImageDataReader(CalibrationDataReader):
    def __init__(
        self,
        calibration_image_folder: str,
        model_path: str,
        data_size: int,
        batch_size: int
    ):
        self.enum_data = None

        session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"ImageDataReader expects one model input; "
                f"found {len(inputs)}"
            )

        input_shape = inputs[0].shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected NCHW image input, got {input_shape}"
            )

        _, channels, height, width = input_shape

        if channels != 3:
            raise ValueError(
                f"Expected 3-channel RGB input, got {input_shape}"
            )

        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(
                f"Static image dimensions required: {input_shape}"
            )

        self.input_name = inputs[0].name
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder,
            height,
            width,
            size_limit=data_size,
            batch_size=batch_size
        )
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: batch}
                 for batch in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

    def reset(self):
        self.enum_data = None


import torch
from timm.data import create_loader, resolve_data_config, create_dataset
from typing import List, Any, Union
from timm.models import create_model

def post_process_top1(output: torch.tensor) -> float:
    _, preds_top1 = torch.max(output, 1)
    return preds_top1

def getAccuracy_top1(preds: Union[torch.tensor, list], targets: Union[torch.tensor, list]) -> float:
    assert len(preds) == len(targets)
    assert len(preds) > 0
    count = 0
    for i in range(len(preds)):
        pred = preds[i]
        target = targets[i]
        if pred == target:
            count += 1
    return count / len(preds)

global model_name
model_name = "resnet50"

global calibration_dataset_path
calibration_dataset_path = "calib_data"

def top1_accu(results: List[Union[torch.tensor, List[Any]]]) -> float:
    """
    Calculate the top1 accuracy of the model.
    :param results: the result of the model
    :return: the top1 accuracy
    """
    timm_model_name = model_name
    calib_data_path = calibration_dataset_path

    timm_model = create_model(
        timm_model_name,
        pretrained=False,
    )

    data_config = resolve_data_config(model=timm_model, use_test_size=True)

    loader = create_loader(create_dataset('', calib_data_path),
                           input_size=data_config['input_size'],
                           batch_size=20,
                           use_prefetcher=False,
                           interpolation=data_config['interpolation'],
                           mean=data_config['mean'],
                           std=data_config['std'],
                           num_workers=2,
                           crop_pct=data_config['crop_pct'])
    target = []
    for _, labels in loader:
        target.extend(labels.data.tolist())
    outputs_top1 = post_process_top1(torch.tensor(numpy.squeeze(numpy.array(results))))
    top1_acc = getAccuracy_top1(outputs_top1, target)
    return round(top1_acc, 2)
