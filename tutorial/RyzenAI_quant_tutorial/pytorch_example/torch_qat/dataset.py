import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.transforms as transforms

__all__ = ['prepare_data_loaders']

def prepare_data_loaders(data_path,
                         train_batch_size=64,
                         val_batch_size=100,
                         workers=8):
  traindir = os.path.join(data_path, 'train')
  valdir = os.path.join(data_path, 'validation')
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_dataset = datasets.ImageFolder(traindir,
                                       transforms.Compose([
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize,
                                       ]))  # 1281167
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=train_batch_size,
      shuffle=True,
      num_workers=workers,
      pin_memory=True)

  val_dataset = datasets.ImageFolder(valdir,
                                     transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                     ]))  # len 50000

  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=val_batch_size,
      shuffle=False,
      num_workers=workers,
      pin_memory=True)  # len(500)

  return train_loader, val_loader
