# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\audio\whisper\index.mdx:150
python run_whisper.py \
  --encoder whisper-base-onnx/encoder_model.onnx \
  --decoder whisper-base-onnx/decoder_model.onnx \
  --device npu \
  --input audio_files/1089-134686-0000.wav
