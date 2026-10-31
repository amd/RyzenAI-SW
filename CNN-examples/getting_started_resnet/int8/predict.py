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
from torchvision.datasets import CIFAR10
import sys

quantized_model_path = r'./models/resnet_quantized.onnx'
model = onnx.load(quantized_model_path)


parser = argparse.ArgumentParser()
parser.add_argument('--ep', type=str, default ='cpu',choices = ['cpu','npu'], help='EP backend selection')
opt = parser.parse_args()


providers = ['CPUExecutionProvider']
provider_options = [{}]

#NPU Setup
if opt.ep == 'npu':
   npu_device = get_npu_info()
   providers = ['VitisAIExecutionProvider']
   cache_dir = Path(__file__).parent.resolve()
   provider_options = [{
                'cache_dir': str(cache_dir),
                'log_level':'info',
                'cache_key': 'modelcachekey',
                'enable_cache_file_io_in_mem':'0'
            }]
   # For PHX/HPT, xclbin is mandatory
   if npu_device == 'PHX/HPT':
       provider_options[0]['target'] = 'X1'
       provider_options[0]['xlnx_enable_py3_round'] = 0
       provider_options[0]['xclbin'] = get_xclbin(npu_device)

# Create session options
session_options = ort.SessionOptions()
session_options.log_severity_level = 1  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

session = ort.InferenceSession(model.SerializeToString(),
                               sess_options=session_options,
                               providers=providers,
                               provider_options=provider_options)


dataset = CIFAR10(root="./data", train= False, download = False)  
images = dataset.data.transpose(0, 3, 1, 2)
labels = np.array(dataset.targets)
label_names = dataset.classes   
     
# create images folder
import os
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
