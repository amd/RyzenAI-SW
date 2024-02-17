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

import torch
import torch.nn as nn

from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional

__all__ = ['resnet18', 'mobilenetV2']
# https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18
# https://pytorch.org/vision/main/_modules/torchvision/models/mobilenetv2.html#mobilenet_v2

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

class ResNet(nn.Module):

  def __init__(self,
               block,
               layers,
               num_classes=1000,
               zero_init_residual=False,
               groups=1,
               width_per_group=64,
               replace_stride_with_dilation=None,
               norm_layer=None):
    super(ResNet, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer

    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError(
          "replace_stride_with_dilation should be None "
          "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = nn.Conv2d(
        3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(
        block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(
        block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(
        block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    self.quant_stub = nndct_nn.QuantStub()
    self.dequant_stub = nndct_nn.DeQuantStub()

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes * block.expansion, stride),
          norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(
        block(self.inplanes, planes, stride, downsample, self.groups,
              self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              groups=self.groups,
              base_width=self.base_width,
              dilation=self.dilation,
              norm_layer=norm_layer))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.quant_stub(x)

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    x = self.dequant_stub(x)
    return x

def _resnet(arch, block, layers, pretrained):
  model = ResNet(block, layers)
  if pretrained != None:
    model.load_state_dict(torch.load(pretrained))
  return model

def resnet18(pretrained):
  return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained)

'''
MobileNet
'''

def _make_divisible(v, divisor, min_value=None):
  """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v

class ConvBNReLU(nn.Sequential):

  def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
    padding = (kernel_size - 1) // 2
    super(ConvBNReLU, self).__init__(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.1),
        # Replace with ReLU
        nn.ReLU(inplace=False))

class InvertedResidual(nn.Module):

  def __init__(self, inp, oup, stride, expand_ratio):
    super(InvertedResidual, self).__init__()
    self.stride = stride
    assert stride in [1, 2]

    hidden_dim = int(round(inp * expand_ratio))
    self.use_res_connect = self.stride == 1 and inp == oup

    layers = []
    if expand_ratio != 1:
      # pw
      layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
    layers.extend([
        # dw
        ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
        # pw-linear
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup, momentum=0.1),
    ])
    self.conv = nn.Sequential(*layers)
    # Replace torch.add with functional
    self.skip_add = functional.Add()

  def forward(self, x):
    if self.use_res_connect:
      # return self.skip_add.add(x, self.conv(x))
      return self.skip_add(x, self.conv(x))
    else:
      return self.conv(x)

# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L186
class MobileNetV2(nn.Module):

  def __init__(self,
               num_classes=1000,
               width_mult=1.0,
               inverted_residual_setting=None,
               round_nearest=8):
    """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
    super(MobileNetV2, self).__init__()
    block = InvertedResidual
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
    if len(inverted_residual_setting) == 0 or len(
        inverted_residual_setting[0]) != 4:
      raise ValueError(
          "inverted_residual_setting should be non-empty "
          "or a 4-element list, got {}".format(inverted_residual_setting))

    # building first layer
    input_channel = _make_divisible(input_channel * width_mult, round_nearest)
    self.last_channel = _make_divisible(last_channel * max(1.0, width_mult),
                                        round_nearest)
    features = [ConvBNReLU(3, input_channel, stride=2)]
    # building inverted residual blocks
    for t, c, n, s in inverted_residual_setting:
      output_channel = _make_divisible(c * width_mult, round_nearest)
      for i in range(n):
        stride = s if i == 0 else 1
        features.append(
            block(input_channel, output_channel, stride, expand_ratio=t))
        input_channel = output_channel
    # building last several layers
    features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
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
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)

  def forward(self, x):
    x = self.quant_stub(x)
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # x = x.mean([2, 3]) # old
    x = self.classifier(x)
    x = self.dequant_stub(x)
    return x

  # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
  # This operation does not change the numerics
  def fuse_model(self):
    for m in self.modules():
      if type(m) == ConvBNReLU:
        torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
      if type(m) == InvertedResidual:
        for idx in range(len(m.conv)):
          if type(m.conv[idx]) == nn.Conv2d:
            torch.ao.quantization.fuse_modules(
                m.conv, [str(idx), str(idx + 1)], inplace=True)

def mobilenetV2(pretrained):
  model = MobileNetV2()
  if pretrained != None:
    state_dict = torch.load(pretrained)
    model.load_state_dict(state_dict)
  return model

def get_model(name, pretrained=None):
  if not isinstance(name, str):
    raise TypeError(f'Model name must be string, but got {type(name)}')

  globs = globals()
  if name not in globs:
    raise ValueError(f'Unknown model {name}')
  return globs[name](pretrained)
