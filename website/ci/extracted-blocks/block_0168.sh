# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\object_detection\yolov8m\README.mdx:217
python quantize_quark.py --input_model_path models/yolov8m.onnx \
                         --calib_data_path calib_images \
                         --output_model_path models/yolov8m_XINT8.onnx \
                         --config XINT8
                         --exclude_subgraphs "[/model.22/Concat_3], [/model.22/Concat_10]]"
