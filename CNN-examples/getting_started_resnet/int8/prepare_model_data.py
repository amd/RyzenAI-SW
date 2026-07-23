# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import random
import ssl
import sys
import tarfile
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context


def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded / total_size * 100, 100)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar = "#" * int(percent // 2) + "-" * (50 - int(percent // 2))
        sys.stdout.write(f"\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()
        if downloaded >= total_size:
            print()

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet_utils import get_directories
from torchvision.models import ResNet50_Weights, resnet50


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=0)
    parser.add_argument("--train", action='store_true')
    args = parser.parse_args()
    return args


def load_resnet_model():
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)
    resnet.fc = torch.nn.Sequential(torch.nn.Linear(2048, 64), torch.nn.ReLU(inplace=True), torch.nn.Linear(64, 10))
    return resnet


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def prepare_model(num_epochs=0, models_dir="models", data_dir="data"):
    # seed everything to 0
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper-parameters
    num_epochs = num_epochs
    learning_rate = 0.001

    # Image preprocessing modules
    transform = transforms.Compose(
        [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
    )

    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    model = load_resnet_model().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )
        # Decay learning rate
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

    # Test the model
    model.eval()
    if num_epochs:
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("Accuracy of the model on the test images: {} %".format(100 * correct / total))

    # Save the model
    model.to("cpu")
    torch.save(model, str(models_dir / "resnet_trained_for_cifar10.pt"))

def export_to_onnx(model, models_dir): 
    model.to("cpu")
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


def main():
    _, models_dir, data_dir, _ = get_directories()
    args = get_args()

    data_download_path_python = data_dir / "cifar-10-python.tar.gz"
    data_download_path_bin = data_dir / "cifar-10-binary.tar.gz"
    cifar_extracted = data_dir / "cifar-10-batches-py"
    if cifar_extracted.exists():
        print("CIFAR-10 data already exists, skipping download.")
    else:
        for url, path, label in [
            ("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", data_download_path_python, "cifar-10-python.tar.gz"),
            ("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", data_download_path_bin, "cifar-10-binary.tar.gz"),
        ]:
            if path.exists() and not tarfile.is_tarfile(path):
                print(f"{label} is corrupted, re-downloading...")
                path.unlink()
            if not path.exists():
                print(f"Downloading {label}...")
                urllib.request.urlretrieve(url, path, reporthook=download_progress)
            print(f"Extracting {label}...")
            with tarfile.open(path) as f:
                f.extractall(data_dir)
    if args.train:
        prepare_model(args.num_epochs, models_dir, data_dir)
    model = torch.load(str(models_dir / "resnet_trained_for_cifar10.pt"), weights_only=False)
    export_to_onnx(model, models_dir)


if __name__ == "__main__":
    main()
