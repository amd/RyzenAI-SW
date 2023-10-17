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
#

# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.ops.misc import ConvNormActivation
from pytorch_nndct import nn as nndct_nn
from torch import Tensor
from pytorch_nndct.nn.modules import functional
from pytorch_nndct import QatProcessor
from torchvision.models._utils import _make_divisible

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='imagenet/processed',
    help='Data set directory.')
parser.add_argument(
    '--pretrained',
    default='./mobilenet_v2.pth',
    help='Pre-trained model file path.')
parser.add_argument(
    '--workers',
    default=4,
    type=int,
    help='Number of data loading workers to be used.')
parser.add_argument('--epochs', default=3, type=int, help='Training epochs.')
parser.add_argument(
    '--quantizer_lr',
    default=1e-2,
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
    '--mode',
    default='train',
    choices=['train', 'deploy'],
    help='Running mode.')
parser.add_argument(
    '--save_dir',
    default='./qat_models',
    help='Directory to save trained models.')
parser.add_argument(
    '--output_dir', default='qat_result', help='Directory to save qat result.')
parser.add_argument(
    '--gpus',
    type=str,
    default='0',
    help='gpu ids to be used for training, seperated by commas')
args, _ = parser.parse_known_args()

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(
      in_planes,
      out_planes,
      kernel_size=3,
      stride=stride,
      padding=dilation,
      groups=groups,
      bias=False,
      dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
  """1x1 convolution"""
  return nn.Conv2d(
      in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               downsample=None,
               groups=1,
               base_width=64,
               dilation=1,
               norm_layer=None):
    super(BasicBlock, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64:
      raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu1 = nn.ReLU(inplace=True)
    #self.relu1 = nn.functional.relu
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

    self.skip_add = functional.Add()
    self.relu2 = nn.ReLU(inplace=True)

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out = self.skip_add(out, identity)
    out = self.relu2(out)

    return out

class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               downsample=None,
               groups=1,
               base_width=64,
               dilation=1,
               norm_layer=None):
    super(Bottleneck, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    width = int(planes * (base_width / 64.)) * groups
    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv1x1(inplanes, width)
    self.bn1 = norm_layer(width)
    self.conv2 = conv3x3(width, width, stride, groups, dilation)
    self.bn2 = norm_layer(width)
    self.conv3 = conv1x1(width, planes * self.expansion)
    self.bn3 = norm_layer(planes * self.expansion)
    self.relu1 = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

    self.skip_add = functional.Add()
    self.relu2 = nn.ReLU(inplace=True)
    self.relu3 = nn.ReLU(inplace=True)

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    # The original code was:
    # out += identity
    # Replace '+=' with Add module cause we want to quantize add op.
    out = self.skip_add(out, identity)
    out = self.relu3(out)

    return out

### Model mobilenetV2 defination
class _DeprecatedConvBNAct(ConvNormActivation):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The ConvBNReLU/ConvBNActivation classes are deprecated and will be removed in future versions. "
            "Use torchvision.ops.misc.ConvNormActivation instead.", FutureWarning)
        if kwargs.get("norm_layer", None) is None:
            kwargs["norm_layer"] = nn.BatchNorm2d
        if kwargs.get("activation_layer", None) is None:
            kwargs["activation_layer"] = nn.ReLU6
        super().__init__(*args, **kwargs)


ConvBNReLU = _DeprecatedConvBNAct
ConvBNActivation = _DeprecatedConvBNAct


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                             activation_layer=nn.ReLU6))
        layers.extend([
            # dw
            ConvNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                               activation_layer=nn.ReLU6),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1
        self.skip_add = functional.Add()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            y = self.skip_add(x, self.conv(x))
            return y
            #return x + self.conv(x)
        else:
            return self.conv(x)
class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting=None,
        round_nearest: int = 8,
        block = None,
        norm_layer = None
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer,
                                                        activation_layer=nn.ReLU6)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer,
                                           activation_layer=nn.ReLU6))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.quant_stub = nndct_nn.QuantStub()
        self.dequant_stub = nndct_nn.DeQuantStub()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.quant_stub(x)
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant_stub(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v2(pretrained=None, **kwargs):
  model = MobileNetV2(**kwargs)
  if pretrained:
    model.load_state_dict(torch.load(args.pretrained))
  return model
def train_one_step(model, inputs, criterion, optimizer, step, device):
  # switch to train mode
  model.train()

  images, target = inputs
  if not isinstance(model, nn.DataParallel):
    model = model.to(device)
  images = images.to(device, non_blocking=True)
  target = target.to(device, non_blocking=True)

  # compute output
  output = model(images)
  loss = criterion(output, target)

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
  os.mkdir(x)
  if not os.path.isdir(x):
    raise RuntimeError("Failed to create dir %r" % x)

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

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

class ProgressMeter(object):

  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, step):
  """Sets the learning rate to the initial LR decayed by decay ratios"""

  weight_lr_decay_steps = 3000 * (24 / args.train_batch_size)
  quantizer_lr_decay_steps = 1000 * (24 / args.train_batch_size)

  for param_group in optimizer.param_groups:
    group_name = param_group['name']
    if group_name == 'weight' and step % weight_lr_decay_steps == 0:
      lr = args.weight_lr * (
          args.weight_lr_decay**(step / weight_lr_decay_steps))
      param_group['lr'] = lr
      print('Adjust lr at epoch {}, step {}: group_name={}, lr={}'.format(
          epoch, step, group_name, lr))
    if group_name == 'quantizer' and step % quantizer_lr_decay_steps == 0:
      lr = args.quantizer_lr * (
          args.quantizer_lr_decay**(step / quantizer_lr_decay_steps))
      param_group['lr'] = lr
      print('Adjust lr at epoch {}, step {}: group_name={}, lr={}'.format(
          epoch, step, group_name, lr))

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

  num_train_batches_per_epoch = int(len(train_loader) / args.train_batch_size)
  if device_ids is not None and len(device_ids) > 0:
    device = f"cuda:{device_ids[0]}"
    model = model.to(device)
    if len(device_ids) > 1:
      model = nn.DataParallel(model, device_ids=device_ids)

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
            }, is_best, args.save_dir)
        if is_best:
          best_filepath = filepath

  return best_filepath

def main():
  print('Used arguments:', args)

  traindir = os.path.join(args.data_dir)
  valdir = os.path.join(args.data_dir)

  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  train_dataset = datasets.ImageFolder(
      traindir,
      transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
      ]))

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.train_batch_size,
      shuffle=True,
      num_workers=args.workers,
      pin_memory=True)

  val_dataset = datasets.ImageFolder(
      valdir,
      transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
      ]))

  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=args.val_batch_size,
      shuffle=False,
      num_workers=args.workers,
      pin_memory=True)

  device_ids = None if args.gpus == "" else [int(i) for i in args.gpus.split(",")]

  device = f"cuda:{device_ids[0]}" if device_ids is not None and len(device_ids) > 0 else "cpu"

  model =mobilenet_v2(pretrained=True).to(device)
  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss()

  inputs = torch.randn([args.train_batch_size, 3, 224, 224],
                       dtype=torch.float32,
                       device=device)
  qat_processor = QatProcessor(model, inputs)

  if args.mode == 'train':
    # Step 1: Get quantized model and train it.
    quantized_model = qat_processor.trainable_model()

    criterion = criterion.to(device)
    best_ckpt = train(quantized_model, train_loader, val_loader, criterion, device_ids)

    # Step 2: Get deployable model and test it.
    # There may be some slight differences in accuracy with the quantized model.
    quantized_model.load_state_dict(torch.load(best_ckpt)['state_dict'])
    deployable_model = qat_processor.to_deployable(quantized_model,
                                                   args.output_dir)
    validate(val_loader, deployable_model, criterion, device)
  elif args.mode == 'deploy':
    # Step 3: Export xmodel from deployable model.
    deployable_model = qat_processor.deployable_model(
        args.output_dir, used_for_xmodel=True)
    val_subset = torch.utils.data.Subset(val_dataset, list(range(1)))
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
  else:
    raise ValueError('mode must be one of ["train", "deploy"]')

if __name__ == '__main__':
  main()