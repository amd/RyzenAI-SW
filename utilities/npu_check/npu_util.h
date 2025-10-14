/***********************************************************************************
MIT License

Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
************************************************************************************/

#pragma once

#include <string>
#include <iostream>
#include <vector>

#include <dxgi.h>

#include <Windows.h>

namespace npu_util {

    enum Status {
        OK = 0,
        UNKNOWN,
        NPU_UNRECOGNIZED,
        DRIVER_TOO_OLD,
        EP_TOO_OLD
    };

    struct NPUInfo {
        int device_id;
        std::string device_name;
        DWORDLONG driver_version_number;
        std::string driver_version_string;
        Status check;
    };

    // Return a vector of DXGI adapter descriptions
    std::vector<DXGI_ADAPTER_DESC> enumerateDXGIAdapters();

    // Checks whether the system configuration is compatible for VitisAI EP 1.2
    NPUInfo checkCompatibility_RAI_1_2();

    // Checks whether the system configuration is compatible for VitisAI EP 1.3
    NPUInfo checkCompatibility_RAI_1_3();

    // Checks whether the system configuration is compatible for VitisAI EP 1.3.1
    NPUInfo checkCompatibility_RAI_1_3_1();

    // Checks whether the system configuration is compatible for VitisAI EP 1.4
    NPUInfo checkCompatibility_RAI_1_4();

    // Checks whether the system configuration is compatible for VitisAI EP 1.4.1
    NPUInfo checkCompatibility_RAI_1_4_1();

    // Checks whether the system configuration is compatible for VitisAI EP 1.5
    NPUInfo checkCompatibility_RAI_1_5();

    // Checks whether the system configuration is compatible for VitisAI EP 1.6
    NPUInfo checkCompatibility_RAI_1_6();

} // npu_util

