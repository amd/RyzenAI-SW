#!/bin/python3

import argparse
import numpy as np
import cv2
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path


quantized_model_path = r'./quickstart/resnet.qdq.U8S8.onnx'
model = onnx.load(quantized_model_path)


parser = argparse.ArgumentParser()
parser.add_argument('--ep', type=str, default ='cpu',choices = ['cpu','ipu'], help='EP backend selection')
opt = parser.parse_args()


providers = ['CPUExecutionProvider']
provider_options = [{}]


if opt.ep == 'ipu':
   providers = ['VitisAIExecutionProvider']
   cache_dir = Path(__file__).parent.resolve()
   provider_options = [{
                'config_file': 'vaip_config.json',
                'cacheDir': str(cache_dir),
                'cacheKey': 'quickstart_modelcachekey'
            }]

session = ort.InferenceSession(model.SerializeToString(), providers=providers,
                               provider_options=provider_options)

#Pick images and predict
image_name = f'./quickstart/image_{0}.png'
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

#Process the outputs
output_array = outputs[0]
predicted_class = np.argmax(output_array)
label = 'cat'
if predicted_class == 3:
    predicted_label = 'cat'
    print(f'Image {0}: Actual Label {label}, Predicted Label {predicted_label}')

else:
    predicted_label = 'not-cat'
    print(f'Image {0}: Actual Label {label}, Predicted Label {predicted_label}')




#################################################################################  
#License
#Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
