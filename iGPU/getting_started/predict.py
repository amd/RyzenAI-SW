from PIL import Image
import argparse
import numpy as np
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path

model_path = r'./torch_to_onnx-float16_conversion-perf_tuning/gpu-dml_model.onnx'
model = onnx.load(model_path)


providers = ['DmlExecutionProvider']
provider_options = [{"device_id": "0"}]

available_providers = ort.get_available_providers()

session = ort.InferenceSession(model.SerializeToString(), providers=providers,
                               provider_options=provider_options)


image_path = 'cat.jpg'
image = Image.open(image_path)

image = image.resize((224, 224))

image_array = np.array(image).astype(np.float16)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_array = (image_array / 255.0 - mean) / std

image_array = np.transpose(image_array, (2, 0, 1)) 
input_data = np.expand_dims(image_array, axis=0)

input_data = input_data.astype(np.float16)

# Run the model
for i in range(1000): 
    outputs = session.run(None, {'input_image': input_data})

# Process the outputs
output_array = outputs[0]
predicted_class_index = np.argmax(output_array)

with open('imagenet_labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]


print(labels[predicted_class_index])