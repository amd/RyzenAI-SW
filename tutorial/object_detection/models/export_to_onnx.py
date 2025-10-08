import torch
import onnx
from onnxsim import simplify

def export_yolov8m_to_onnx():
    # Load the pre-trained YOLOv8m model
    model_path = "yolov8m.pt"
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    # Extract the model from checkpoint
    if 'model' in checkpoint:
        model = checkpoint['model']
    elif 'ema' in checkpoint:
        model = checkpoint['ema']
    else:
        model = checkpoint
    # Set model to evaluation mode
    if hasattr(model, 'float'):
        model = model.float()
    if hasattr(model, 'eval'):
        model.eval()

    # Get number of classes
    if hasattr(model, 'nc'):
        num_classes = model.nc
        print(f"Number of classes: {num_classes}")

    # Define input shape (batch_size, channels, height, width)
    batch_size = 1
    img_size = 640
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    # Export to ONNX
    output_path = "yolov8m.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output0']
    )

    print(f"YOLOv8m exported to {output_path}")

    # Optional: Simplify the ONNX model
    try:
        print("Simplifying ONNX model...")
        onnx_model = onnx.load(output_path)
        model_simp, check = simplify(onnx_model)
        if check:
            onnx.save(model_simp, output_path)
            print("ONNX model simplified successfully")
        else:
            print("Simplification failed, keeping original model")
    except Exception as e:
        print(f"Simplification skipped: {e}")

if __name__ == "__main__":
    export_yolov8m_to_onnx()
