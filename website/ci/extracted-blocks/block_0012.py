# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\model-deployment.mdx:52
input_data = {}
for input in session.get_inputs():
    input_data[input.name] = ...  # Initialize input tensors

outputs = session.run(None, input_data) # Run the model
