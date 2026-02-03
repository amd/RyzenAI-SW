import onnxruntime as ort

#   Model               -   size
# mobilenet_v2.onnx     - 25153280
# resnet50.onnx         - 12548096

print("Init the largest model on top");
provider_options = [{
    'config_file': 'vaiml_config.json',
    'cache_dir': 'models_cache',
    'cache_key': 'model1',
    'maxSpillBufferSize': "26000000", # 26MB Larger than the largest model
    "enable_cache_file_io_in_mem": "0"
}]

s1 = ort.InferenceSession("models/mobilenet_v2.onnx", providers=['VitisAIExecutionProvider'], provider_options=provider_options)


print("Init smaller models below");
provider_options = [{
    'config_file': 'vaiml_config.json',
    'cache_dir': 'models_cache',
    'cache_key': 'model2',
    'maxSpillBufferSize': "26000000", # 26MB Larger than the largest model
    "enable_cache_file_io_in_mem": "0"
}]

s2 = ort.InferenceSession("models/resnet50.onnx", providers=['VitisAIExecutionProvider'], provider_options=provider_options)
