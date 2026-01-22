import torch
import numpy as np
import onnxruntime

class ImageDataReader:
    """Data reader class for ONNX calibration."""
    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)

    def get_next(self):
        try:
            images, _ = next(self.dataloader)  # Ignore labels
            return {"input": images.numpy()}
        except StopIteration:
            return None

def evaluate_onnx_model(onnx_model_path, dataset, batch_size=8):
    """Evaluates ONNX model accuracy using given dataset."""
    session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    
    correct = 0
    total = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for images, labels in dataloader:
        inputs = {"input": images.numpy()}
        outputs = session.run(None, inputs)
        predictions = np.argmax(outputs, axis=1)
        
        correct += (predictions.argmax(axis=1) == labels.numpy()).sum()
        total += labels.size(0)
    
    accuracy = 100 * correct / total
    return accuracy
