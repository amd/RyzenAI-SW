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

ROOT_DIR=${PWD}
WEIGHTS=./../float/yolov8m.pt
cd code

echo "[Calib mode]"
CUDA_VISIBLE_DEVICES=${GPU_ID} yolo detect val data="datasets/coco.yaml" model=${WEIGHTS} \
    nndct_quant=True quant_mode=calib --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish
echo "[Test mode] testing"
CUDA_VISIBLE_DEVICES=${GPU_ID} yolo detect val data="datasets/coco.yaml" model=${WEIGHTS} \
    nndct_quant=True quant_mode=test --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish
