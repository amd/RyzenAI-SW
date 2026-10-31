#!/bin/python3

import argparse
import numpy as np
import os
import cv2
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path
from resnet_utils import get_npu_info, get_xclbin
import pathlib
import onnxruntime as ort
import onnxruntime
import os
from torchvision.datasets import CIFAR10
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', type=str, default ='cpu',choices = ['cpu','npu'], help='EP backend selection')
    opt = parser.parse_args()

    onnx_model_path = r'./models/resnet_trained_for_cifar10.onnx'
    config_file = 'vitisai_config.json'
    datafile = r'./data/cifar-10-batches-py/test_batch'
    metafile = r'./data/cifar-10-batches-py/batches.meta'
    cache_dir = os.path.abspath('my_cache_dir')
    cache_key   = pathlib.Path(onnx_model_path).stem

    # defualt provider is CPUExecutionProvider
    providers = ['CPUExecutionProvider']
    provider_options_dict = {}

    #NPU/CPU setup
    if opt.ep == 'npu':
        print("execution started on NPU")
        npu_device = get_npu_info()
        providers = ['VitisAIExecutionProvider']

        provider_options_dict = {
                "config_file": config_file,
                "cache_dir": cache_dir,
                "cache_key": cache_key,
                "enable_cache_file_io_in_mem":0,
            }
    elif opt.ep == 'cpu':
        print("execution started on CPU")
    else:
        print("No APU detected. Exiting.")
        exit(-1)

    # Create session options
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 1  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

    # onnxruntime session creation with provided option (either CPU or NPU)
    # corresponding provider option will be created
    session = onnxruntime.InferenceSession(
            onnx_model_path,
            sess_options=session_options,
            providers=providers,
            provider_options=[provider_options_dict]
        )
    
    dataset = CIFAR10(root="./data", train= False, download = False)  
    images = dataset.data.transpose(0, 3, 1, 2)
    labels = np.array(dataset.targets)
    label_names = dataset.classes 

    # create images folder
    dirname = 'images'
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    #Extract and dump first 10 images
    for i in range (0,10):
        im = images[i]
        im  = im.transpose(1,2,0)
        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        im_name = f'./images/image_{i}.png'
        cv2.imwrite(im_name, im)

    correct_predictions = 0

    #Pick dumped images and predict
    for i in range (0,10):
        image_name = f'./images/image_{i}.png'
        image = Image.open(image_name).convert('RGB')
        # Resize the image to match the input size expected by the model
        image = image.resize((32, 32))
        image_array = np.array(image).astype(np.float32)
        image_array = image_array/255

        # Reshape the array to match the input shape expected by the model
        image_array = np.transpose(image_array, (2, 0, 1))

        # Add a batch dimension to the input image
        input_data = np.expand_dims(image_array, axis=0)


        # Run the model
        outputs = session.run(None, {'input': input_data})


        # Process the outputs
        output_array = outputs[0]
        predicted_class = np.argmax(output_array)
        
        actual_class = labels[i]

        predicted_label = label_names[predicted_class]
    
        label = label_names[labels[i]]
        
        print(f'Image {i}: Actual Label {label}, Predicted Label {predicted_label}')

        # Check prediction
        if predicted_class == actual_class:
            correct_predictions += 1

    print(f"\nCorrect predictions: {correct_predictions}/10")

    # PASS if at least 9 out of 10 are correct
    if correct_predictions >= 9:
        print("TEST RESULT: PASS")
        sys.exit(0)
    else:
        print("TEST RESULT: FAIL")
        sys.exit(1)

    #################################################################################
    #License
    #Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
if __name__=="__main__":
    main()