# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\quark_quantization\README.mdx:101
mkdir val_data
tar -xzf val_images.tar.gz -C val_data
python prepare_data.py val_data calib_data
