/*****************************************************************************
Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*****************************************************************************/

#include <Windows.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cfgmgr32.h>

#define DEVICE_VENDOR_ID_AMD 0x1022

#define DEVICE_ID_GEN1 0x1502
#define DEVICE_ID_GEN2 0x17F0

#define GENX_REV0   0x00
#define GEN2_REV1   0x10
#define GEN2_REV2   0x11
#define GEN2_REV3   0x20

void printUsage()
{
    std::cout << "APU Platform Check application" <<std::endl;
    std::cout << "Usage: apu_check.exe [OPTIONS]\n" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "-h, --help         Print this help" << std::endl;
}

bool hasDevice(const std::string& devName, const std::string& revNum, std::vector<char> &bar)
{
    bool status_ret = false;

    char* ptr = bar.data();
    while (*ptr)
    {
        std::string id = ptr;
        std::transform(id.cbegin(),
            id.cend(),
            id.begin(),
            [](unsigned char c) { return std::toupper(c); });
        if (id.find(devName) != std::string::npos)
        {
            if (id.find(revNum) != std::string::npos)
            {
                status_ret = true;
            }
        }
        ptr += strlen(ptr) + 1;
    }

    return status_ret;
}

BOOL getPlatformType(int& PlatType)
{
    std::stringstream devStr, revStr;

    CONFIGRET result;
    ULONG bufLen = 0;
    result = CM_Get_Device_ID_List_SizeA(&bufLen, "PCI", CM_GETIDLIST_FILTER_ENUMERATOR);
    if (result != CR_SUCCESS)
    {
        return false;
    }

    std::vector<char> buffer;
    buffer.resize(bufLen);
    result = CM_Get_Device_ID_ListA("PCI", buffer.data(), bufLen, CM_GETIDLIST_FILTER_ENUMERATOR);
    if (result != CR_SUCCESS)
    {
        return false;
    }

#define CHECK_DEV(dev, rev, msg)                                                   \
    devStr = std::stringstream();                                                  \
    revStr = std::stringstream();                                                  \
    devStr << std::hex << std::setfill('0') << std::setw(4) << std::uppercase;     \
    revStr << "&REV_" << std::hex << std::setfill('0') << std::setw(2) << rev;     \
    devStr << "PCI\\VEN_" << DEVICE_VENDOR_ID_AMD << "&DEV_" << dev << "&SUBSYS_"; \
    if (hasDevice(devStr.str(), revStr.str(), buffer))                             \
    {                                                                              \
        std::cout << "APU type: " << msg << std::endl;                             \
        return true;                                                               \
    }

    // 1st gen silicon
    CHECK_DEV(DEVICE_ID_GEN1, GENX_REV0, "PHX");

    // 2nd gen silicon revisions
    CHECK_DEV(DEVICE_ID_GEN2, GENX_REV0, "STX_A0");
    CHECK_DEV(DEVICE_ID_GEN2, GEN2_REV1, "STX_B0");
    CHECK_DEV(DEVICE_ID_GEN2, GEN2_REV2, "STXH");
    CHECK_DEV(DEVICE_ID_GEN2, GEN2_REV3, "Krackan");

    return false;
}

int main(int argc, char** argv) try {
    // parse for help argument
    if (argc == 2)
    {
        std::string arg = argv[1];
        if (arg == "-h" || arg == "--help")
        {
            printUsage();
            return 1;
        }
        else
        {
            std::cout << "Invalid param provided. Printing Usage...\n\n";
            printUsage();
            return 1;
        }
    }

    std::cout << "Checking APU type based on NPU device ID..." << std::endl;

    int platformType;
    BOOL platStatus = getPlatformType(platformType);
    if (!platStatus)
    {
        std::cout << "NPU device is not present." << std::endl;
        return 1;
    }
    
}
catch (std::exception& e) {
    std::cout << "Error: " << e.what() << std::endl;
    std::cout << "Fail." << std::endl;
    return 1;
}
