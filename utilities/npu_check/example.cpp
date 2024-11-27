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

#include <iostream>

#include "npu_util.h"

int main()
{
    npu_util::NPUInfo info;

    std::cout << "System compatibility check for VitisAI EP 1.2..." << std::endl;
    info = npu_util::checkCompatibility_RAI_1_2();
    switch (info.check) {
        case npu_util::Status::OK:
            std::cout << "OK!" << std::endl;
            std::cout << "NPU Device ID     : 0x" << std::hex << info.device_id << std::endl;
            std::cout << "NPU Device Name   : " << info.device_name << std::endl;
            std::cout << "NPU Driver Version: " << info.driver_version_string << std::endl;            
            switch (info.device_id) {
                case 0x1502:
                    std::cout << "PHX/HPT NPU detected. Make sure to load corresponding XCLBIN." << std::endl;
                    break;
                case 0x17F0:
                    std::cout << "STX/KRK NPU detected. Make sure to load corresponding XCLBIN." << std::endl;
                    break;
            }
            break;
        case npu_util::Status::NPU_UNRECOGNIZED:
            std::cout << "NPU type not recognized. Do not use VitisAI EP." << std::endl;
            break;
        case npu_util::Status::DRIVER_TOO_OLD:
            std::cout << "Installed drivers are too old. Do not use VitisAI EP." << std::endl;
            std::cout << "NPU Driver Version: " << info.driver_version_string << std::endl;            
            break;
        case npu_util::Status::EP_TOO_OLD:
            std::cout << "VitisAI EP is too old. Do not use VitisAI EP." << std::endl;
            break;
        default:
            std::cout << "Unknown state. Do not use VitisAI EP." << std::endl;
            break;
    }
    std::cout << std::endl;

    std::cout << "System compatibility check for VitisAI EP 1.3..." << std::endl;
    info = npu_util::checkCompatibility_RAI_1_3();
    switch (info.check) {
        case npu_util::Status::OK:
            std::cout << "OK!" << std::endl;
            std::cout << "NPU Device ID     : 0x" << std::hex << info.device_id << std::endl;
            std::cout << "NPU Device Name   : " << info.device_name << std::endl;
            std::cout << "NPU Driver Version: " << info.driver_version_string << std::endl;            
            switch (info.device_id) {
                case 0x1502:
                    std::cout << "PHX/HPT NPU detected. Make sure to load corresponding XCLBIN." << std::endl;
                    break;
                case 0x17F0:
                    std::cout << "STX/KRK NPU detected. Make sure to load corresponding XCLBIN." << std::endl;
                    break;
            }
            break;
        case npu_util::Status::NPU_UNRECOGNIZED:
            std::cout << "NPU type not recognized. Do not use VitisAI EP." << std::endl;
            break;
        case npu_util::Status::DRIVER_TOO_OLD:
            std::cout << "Installed drivers are too old. Do not use VitisAI EP." << std::endl;
            std::cout << "NPU Driver Version: " << info.driver_version_string << std::endl;            
            break;
        case npu_util::Status::EP_TOO_OLD:
            std::cout << "VitisAI EP is too old. Do not use VitisAI EP." << std::endl;
            break;
        default:
            std::cout << "Unknown state. Do not use VitisAI EP." << std::endl;
            break;
    }
    std::cout << std::endl;

    return 0;
}
