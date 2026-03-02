# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\cnn-examples.mdx:56
dummy_inputs = torch.randn(1, 3, 32, 32)
input_names = ['input']
output_names = ['output']
dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
tmp_model_path = str(models_dir / "resnet_trained_for_cifar10.onnx")
torch.onnx.export(
        model,
        dummy_inputs,
        tmp_model_path,
        export_params=True,
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
