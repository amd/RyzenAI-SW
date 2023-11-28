# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim

from dataset import prepare_data_loaders
from pytorch_nndct.nn.modules.quant_stubs import QuantStub, DeQuantStub
from pytorch_nndct.quantization.auto_module import wrap
from torchvision.models import mobilenet_v2
from torchvision.models import resnet18
from utils import AverageMeter, ProgressMeter

from pytorch_nndct import QatProcessor

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='/workspace/dataset/imagenet/pytorch',
    help='Data set directory.')
parser.add_argument(
    '--model_name',
    default='resnet18',
    choices=['MobileNetV2', 'resnet18'],
    help='Model to be used.')
parser.add_argument(
    '--pretrained', default=None, help='Pre trained model weights')
parser.add_argument(
    '--config_file', default=None, help='Quantization config file')
parser.add_argument(
    '--mode',
    default='train',
    choices=['train', 'deploy'],
    help='Running mode.')
parser.add_argument(
    '--workers',
    default=8,
    type=int,
    help='Number of data loading workers to be used.')
parser.add_argument('--epochs', default=3, type=int, help='Training epochs.')
parser.add_argument(
    '--quantizer_lr',
    default=1e-5,
    type=float,
    help='Initial learning rate of quantizer.')
parser.add_argument(
    '--quantizer_lr_decay',
    default=0.5,
    type=int,
    help='Learning rate decay ratio of quantizer.')
parser.add_argument(
    '--weight_lr',
    default=1e-5,
    type=float,
    help='Initial learning rate of network weights.')
parser.add_argument(
    '--weight_lr_decay',
    default=0.94,
    type=int,
    help='Learning rate decay ratio of network weights.')
parser.add_argument(
    '--train_batch_size', default=24, type=int, help='Batch size for training.')
parser.add_argument(
    '--val_batch_size',
    default=100,
    type=int,
    help='Batch size for validation.')
parser.add_argument(
    '--weight_decay', default=1e-4, type=float, help='Weight decay.')
parser.add_argument(
    '--display_freq',
    default=100,
    type=int,
    help='Display training metrics every n steps.')
parser.add_argument(
    '--val_freq', default=1000, type=int, help='Validate model every n steps.')
parser.add_argument(
    '--quantizer_norm',
    default=True,
    type=bool,
    help='Use normlization for quantizer.')
parser.add_argument(
    '--save_dir',
    default='./qat_models',
    help='Directory to save trained models.')
parser.add_argument(
    '--gpus',
    type=str,
    default='0',
    help='gpu ids to be used for training, seperated by commas')
parser.add_argument(
    '--weight_lr_decay_steps',
    type=int,
    default='10000',
    help='adjust learning rate for params')
parser.add_argument(
    '--quantizer_lr_decay_steps',
    type=int,
    default='5000',
    help='adjust learning rate for quantizer params, Only effect in power of 2 mode'
)
parser.add_argument(
    '--output_dir', default='qat_result', help='Directory to save qat result.')
parser.add_argument(
    '--quant_param_l2_penalty',
    default=False,
    type=bool,
    help='Recommand used for TQT quantizer (power of two).')
parser.add_argument(
    '--qat_ckpt',
    default=None,
    help='Checkpoint path used to be exported to deployable model.')
args, _ = parser.parse_known_args()

def train_one_step(model, inputs, criterion, optimizer, step, device):
  images, target = inputs
  images = images.to(device, non_blocking=True)
  target = target.to(device, non_blocking=True)

  # compute output
  output = model(images)
  loss = criterion(output, target)

  # Can add l2 normalization for quantizer param
  if args.quant_param_l2_penalty:
    l2_decay = 1e-4
    l2_norm = 0.0
    q_params = model.quantizer_parameters() if not isinstance(model, nn.DataParallel) \
    else model.module.quantizer_parameters()
    for param in q_params:
      l2_norm += torch.pow(param, 2.0)[0]
    if args.quantizer_norm:
      loss += l2_decay * torch.sqrt(l2_norm)

  # measure accuracy and record loss
  acc1, acc5 = accuracy(output, target, topk=(1, 5))

  # compute gradient and do SGD step
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss, acc1, acc5

def validate(val_loader, model, criterion, device):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

  # switch to evaluate mode
  model.eval()
  if not isinstance(model, nn.DataParallel):
    model = model.to(device)

  with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
      images = images.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)

      # compute output
      output = model(images)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))

  return top1.avg

def mkdir_if_not_exist(x):
  if not x or os.path.isdir(x):
    return
  os.makedirs(x)
  if not os.path.isdir(x):
    raise RuntimeError("Failed to create dir %r" % x)

def save_directory():
  return os.path.join(args.save_dir, args.model_name)

def save_checkpoint(state, is_best, directory):
  mkdir_if_not_exist(directory)

  filepath = os.path.join(directory, 'model.pth')
  torch.save(state, filepath)
  if is_best:
    best_acc1 = state['best_acc1'].item()
    best_filepath = os.path.join(directory, 'model_best_%5.3f.pth' % best_acc1)
    shutil.copyfile(filepath, best_filepath)
    print('Saving best ckpt to {}, acc1: {}'.format(best_filepath, best_acc1))
  return best_filepath if is_best else filepath

def adjust_learning_rate(optimizer, epoch, step):
  """Sets the learning rate to the initial LR decayed by decay ratios"""

  weight_lr_decay_steps = args.weight_lr_decay_steps
  quantizer_lr_decay_steps = args.quantizer_lr_decay_steps
  for param_group in optimizer.param_groups:
    group_name = param_group['name']
    if group_name == 'weight' and step != 0 and step % weight_lr_decay_steps == 0:
      lr = args.weight_lr * (
          args.weight_lr_decay**(step / weight_lr_decay_steps))
      param_group['lr'] = lr
      print(
          'Adjust weight lr at epoch {}, step {}: group_name={}, lr={}'.format(
              epoch, step, group_name, lr))
    if group_name == 'quantizer' and step != 0 and step % quantizer_lr_decay_steps == 0:
      lr = args.quantizer_lr * (
          args.quantizer_lr_decay**(step / quantizer_lr_decay_steps))
      param_group['lr'] = lr
      print('Adjust quantizer lr at epoch {}, step {}: group_name={}, lr={}'
            .format(epoch, step, group_name, lr))

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(model, train_loader, val_loader, criterion, device_ids):
  best_acc1 = 0
  best_filepath = None

  # num_train_batches_per_epoch = int(len(train_loader) / args.train_batch_size)
  if device_ids is not None and len(device_ids) > 0:
    device = f"cuda:{device_ids[0]}"
    model = model.to(device)
    if len(device_ids) > 1:
      model = nn.DataParallel(model, device_ids=device_ids)
  if device_ids is None:
    device = 'cpu'

  batch_time = AverageMeter('Time', ':6.3f')
  data_time = AverageMeter('Data', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')

  param_groups = [{
      'params': model.quantizer_parameters() if not isinstance(model, nn.DataParallel) \
        else model.module.quantizer_parameters(),
      'lr': args.quantizer_lr,
      'name': 'quantizer'
  }, {
      'params': model.non_quantizer_parameters() if not isinstance(model, nn.DataParallel) \
        else model.module.non_quantizer_parameters(),
      'lr': args.weight_lr,
      'name': 'weight'
  }]

  optimizer = torch.optim.Adam(
      param_groups, args.weight_lr, weight_decay=args.weight_decay)

  model.train()
  for epoch in range(args.epochs):
    progress = ProgressMeter(
        len(train_loader) * args.epochs,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch[{}], Step: ".format(epoch))

    for i, (images, target) in enumerate(train_loader):
      end = time.time()
      # measure data loading time
      data_time.update(time.time() - end)

      step = len(train_loader) * epoch + i

      adjust_learning_rate(optimizer, epoch, step)
      loss, acc1, acc5 = train_one_step(model, (images, target), criterion,
                                        optimizer, step, device)

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      if step % args.display_freq == 0:
        progress.display(step)

      if step % args.val_freq == 0:
        # evaluate on validation set
        print('epoch: {}, step: {}'.format(epoch, i))
        acc1 = validate(val_loader, model, criterion, device)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        filepath = save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) \
                  else model.module.state_dict(),
                'best_acc1': best_acc1
            }, is_best,  save_directory())
        if is_best:
          best_filepath = filepath

  return best_filepath

'''
Usage:
python main.py \
  --model_name=[model] \
  --pretrained='resnet18.pth'
  --config_file='./int_float_scale_qatconfig.json' \

model: 'MobileNetV2', 'resnet18'
config_file: e.g ./config_files/int_float_scale_qatconfig.json
'''

name_to_model = {'resnet18': resnet18, 'mobilenet_v2': mobilenet_v2}

class ModelWithQuantStub(nn.Module):

  def __init__(self, model_name, pretrained=None) -> None:
    super().__init__()
    self._model = name_to_model[model_name]()
    if pretrained:
      self._model.load_state_dict(torch.load(pretrained))
    self.quant_stub = QuantStub()
    self.dequant_stub = DeQuantStub()

  def forward(self, imgs):
    imgs = self.quant_stub(imgs)
    out = self._model(imgs)
    return self.dequant_stub(out)

def main():
  print('Used arguments:', args)

  train_loader, val_loader = prepare_data_loaders(args.data_dir,
                                                  args.train_batch_size,
                                                  args.val_batch_size,
                                                  args.workers)

  device_ids = None if args.gpus == "" else [
      int(i) for i in args.gpus.split(",")
  ]
  device = f"cuda:{device_ids[0]}" if device_ids is not None and len(
      device_ids) > 0 else "cpu"
  if device_ids == None:
    device = 'cpu'

  model = ModelWithQuantStub(args.model_name, args.pretrained).to(device)
  criterion = nn.CrossEntropyLoss().to(device)

  # For per-channel use train/val data rather than randn
  inputs = torch.cat([
      torch.unsqueeze(train_loader.dataset[i][0], 0)
      for i in range(args.train_batch_size)
  ]).to(device)  # [args.train_batch_size, 3, 224, 224]

  # Wrap model to replace functional operations with nn.Module objects.
  model = wrap(model, inputs)
  qat_processor = QatProcessor(model, inputs)

  # Step 1: Get quantized model and train it.
  quantized_model = qat_processor.trainable_model(allow_reused_module=True)
  print(quantized_model)

  if args.mode == 'train':
    best_ckpt = train(quantized_model, train_loader, val_loader, criterion,
                      device_ids)
  else:
    quantized_model.load_state_dict(torch.load(args.qat_ckpt)['state_dict'])

    deployable_model = qat_processor.to_deployable(quantized_model,
                                                   args.output_dir)
    validate(val_loader, deployable_model, criterion, device)

    deployable_model = qat_processor.deployable_model(
        args.output_dir, used_for_xmodel=True)
    val_subset = torch.utils.data.Subset(val_loader.dataset, list(range(1)))
    subset_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    # Must forward deployable model at least 1 iteration with batch_size=1
    for images, _ in subset_loader:
      deployable_model(images)
    qat_processor.export_xmodel(args.output_dir)
    qat_processor.export_onnx_model(args.output_dir)

if __name__ == '__main__':
  main()
