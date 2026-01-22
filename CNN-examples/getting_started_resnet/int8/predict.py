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
       provider_options[0]['xclbin'] = get_xclbin(npu_device)

# Create session options
session_options = ort.SessionOptions()
session_options.log_severity_level = 1  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

session = ort.InferenceSession(model.SerializeToString(),
                               sess_options=session_options,
                               providers=providers,
                               provider_options=provider_options)


def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


datafile = r'./data/cifar-10-batches-py/test_batch'
metafile = r'./data/cifar-10-batches-py/batches.meta'

data_batch_1 = unpickle(datafile)
metadata = unpickle(metafile)

images = data_batch_1['data']
labels = data_batch_1['labels']
images = np.reshape(images,(10000, 3, 32, 32))

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
    predicted_label = metadata['label_names'][predicted_class]
    label = metadata['label_names'][labels[i]]
    print(f'Image {i}: Actual Label {label}, Predicted Label {predicted_label}')


#################################################################################
#License
#Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
