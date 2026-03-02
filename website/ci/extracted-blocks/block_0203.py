# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\quark_quantization\README.mdx:238
quant_model = onnx.load(output_model_path)
provider = ['VitisAIExecutionProvider']
cache_dir = Path(__file__).parent.resolve()
print(cache_dir)
provider_options = [{
                'config_file': 'vaip_config.json',
                'cacheDir': str(cache_dir),
                'cacheKey': 'modelcachekey'
            }]

session = ort.InferenceSession(quant_model.SerializeToString(), providers=provider,
                               provider_options=provider_options)

input_data = preprocess_image('test_image.jpg')
outputs = session.run(None, {session.get_inputs()[0].name: input_data})
predicted_class = np.argmax(outputs[0])
print('NPU quantized outputs:')
print(predicted_class)
