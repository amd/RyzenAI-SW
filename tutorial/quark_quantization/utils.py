import os  
import shutil  

import onnxruntime as ort  
import numpy as np  
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader  
from tqdm import tqdm  
from pathlib import Path

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

def evaluate_onnx_model(onnx_model_path, imagenet_data_path, batch_size=1, device='cpu'):  
    # Load the ONNX model
    if device == 'npu':
        provider = ['VitisAIExecutionProvider']
        cache_dir = Path(__file__).parent.resolve()
        print(cache_dir)
        provider_options = [{
                        'config_file': 'vaip_config.json',
                        'cacheDir': str(cache_dir),
                        'cacheKey': 'modelcachekey'
                    }]

        session = ort.InferenceSession(onnx_model_path, providers=provider,
                                       provider_options=provider_options)
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
    data_loader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=False)  
  
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

    def __init__(self, calibration_image_folder: str, model_path: str, data_size: int, batch_size: int):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(calibration_image_folder,
                                                 height, width, data_size, batch_size)
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