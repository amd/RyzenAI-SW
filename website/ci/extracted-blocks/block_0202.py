# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\quark_quantization\README.mdx:221
quant_model = onnx.load(output_model_path)
provider = ['CPUExecutionProvider']

session = ort.InferenceSession(quant_model.SerializeToString(), providers=provider)
input_data = preprocess_image('test_image.jpg')
# run the model in onnxruntime session
outputs = session.run(None, {session.get_inputs()[0].name: input_data})
predicted_class = np.argmax(outputs[0])
print('CPU quantized outputs:')
print(predicted_class)
