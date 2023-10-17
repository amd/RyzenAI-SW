# Copyright 2023 Advanced Micro Devices, Inc. on behalf of itself and its subsidiaries and affiliates
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

# PART OF THIS FILE AT ALL TIMES.

GPU_ID=0
WEIGHTS=./../float/yolov8m.pt
BATCH=64
EPOCH=50
cd code

CUDA_VISIBLE_DEVICES=${GPU_ID} yolo detect train data="datasets/coco.yaml" model=${WEIGHTS} pretrained=True sync_bn=True \
    epochs=${EPOCH} batch=${BATCH} optimizer="AdamW" device=${GPU_ID} lr0=0.0001 nndct_quant=True
