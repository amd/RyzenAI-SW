import torch
from onnxruntime.quantization.calibrate import CalibrationDataReader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


import onnx
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod, quantize_static

from quark.onnx.quantization.config import (Config, get_default_config)
from quark.onnx import ModelQuantizer


class CIFAR10DataSet:
    def __init__(
        self,
        data_dir,
        **kwargs,
    ):
        super().__init__()
        self.train_path = data_dir
        self.vld_path = data_dir
        self.setup("fit")

    def setup(self, stage: str):
        transform = transforms.Compose(
            [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
        )
        self.train_dataset = CIFAR10(root=self.train_path, train=True, transform=transform, download=False)
        self.val_dataset = CIFAR10(root=self.vld_path, train=True, transform=transform, download=False)


class PytorchResNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_data = sample[0]
        label = sample[1]
        return input_data, label


def create_dataloader(data_dir, batch_size):
    cifar10_dataset = CIFAR10DataSet(data_dir)
    _, val_set = torch.utils.data.random_split(cifar10_dataset.val_dataset, [49000, 1000])
    benchmark_dataloader = DataLoader(PytorchResNetDataset(val_set), batch_size=batch_size, drop_last=True)
    return benchmark_dataloader


class ResnetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir: str, batch_size: int = 16):
        super().__init__()
        self.iterator = iter(create_dataloader(data_dir, batch_size))

    def get_next(self) -> dict:
        try:
            images, labels = next(self.iterator)
            return {"input": images.numpy()}
        except Exception:
            return None


def resnet_calibration_reader(data_dir, batch_size=16):
    return ResnetCalibrationDataReader(data_dir, batch_size=batch_size)



def main():
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = "models/resnet_trained_for_cifar10.onnx"

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = "models/resnet_quantized.onnx"

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = "data/"

    # `dr` (Data Reader) is an instance of ResNetDataReader, which is a utility class that 
    # reads the calibration dataset and prepares it for the quantization process.
    dr = resnet_calibration_reader(calibration_dataset_path)
    
    
    #Quantization with Quark
    
    # Get quantization configuration
    quant_config = get_default_config("XINT8")
    config = Config(global_quant_config=quant_config)
    print(f"The configuration for quantization is {config}")

    # Create an ONNX quantizer
    quantizer = ModelQuantizer(config)

    # Quantize the ONNX model
    quantizer.quantize_model(input_model_path, output_model_path, dr)
    

if __name__ == '__main__':
    main()



#################################################################################  
#License
#Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
