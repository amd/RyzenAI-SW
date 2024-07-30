#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

. ./ci/setup.ps1 $PWD

$env:DEVICE = "phx"

$env:XRT_PATH = Join-Path $env:THIRD_PARTY "xrt-ipu"
$env:XLNX_VART_FIRMWARE = Join-Path $PWD "xclbin\phx"
Add-PathToVar "$env:XRT_PATH\xrt" PATH
