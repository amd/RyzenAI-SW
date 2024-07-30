Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

APU/NPU Check application
======================

Performs basic check to identify current APU/NPU platform.

The application returns 0 if APU/NPU platform found; otherwise non-zero if no platform was detected.

## Build Steps:
- This app is a standalone application and has no SW stack dependencies
- To build app, just need to execute build.bat from the apu_check directory

## Executing:
App supports the following command line arguments
    -h,--help    : Print this help

## Example Output:

Default `apu_check.exe` output:
```
C:\Users\Administrator\Desktop>apu_check.exe
Retrieving APU/NPU platform type...
APU/NPU type: Krackan

```

## Overview/Concepts
In this example code, we identify NPU/APU platform with function getPlatformType.
Function getPlatformType retrieves a list of DEVICE_IDs that is parsed for valid NPU/APU platforms.

Recognized NPU PCI Device IDs:
```
PCI\VEN_1022&DEV_1502 AND REV_00  => PHX
PCI\VEN_1022&DEV_17F0 AND REV_00  => STX_A0
PCI\VEN_1022&DEV_17F0 AND REV_10  => STX_B0
PCI\VEN_1022&DEV_17F0 AND REV_11  => STXH
PCI\VEN_1022&DEV_17F0 AND REV_20  => Krackan
```