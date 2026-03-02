# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\llms\linux-setup.mdx:78
import json

with open('Phi-3.5-mini-instruct-onnx-ryzenai-npu/.cache/MatMulNBits_2_0_meta.json','r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if '.cache' in lines[i]:
            lines[i] = lines[i].replace('\\','/')

with open('Phi-3.5-mini-instruct-onnx-ryzenai-npu/.cache/MatMulNBits_2_0_meta.json','w') as f:
    f.writelines(lines)
