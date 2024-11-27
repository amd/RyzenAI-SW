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

// compile using: /std:c++latest

#pragma comment(lib, "setupapi.lib")

#include <chrono>
#include <vector>
#include <mutex>

#include <Windows.h>
#include <setupapi.h>
#include <devguid.h>

#include "npu_util.h"


namespace npu_util {

    std::string DriverVersionToString(DWORDLONG ver) {
        std::stringstream string_stream;
        string_stream << ((ver >> 48) & 0xffff) << "." << ((ver >> 32) & 0xffff) << "." << ((ver >> 16) & 0xffff) << "." << ((ver >> 0) & 0xffff);
        return string_stream.str();
    }

    // Extract NPU information
    NPUInfo extractNPUInfo()
    {
        // Make extractNPUInfo thread-safe
        static std::mutex function_mutex;
        std::lock_guard<std::mutex> guard(function_mutex);

        NPUInfo npu_info;
        npu_info.device_id = -1; 
        npu_info.device_name = "";
        npu_info.driver_version_number = -1;
        npu_info.driver_version_string = "";
        npu_info.check = Status::UNKNOWN;

        static const std::vector<std::pair<std::string, int>> PCI_IDS = {
            { "PCI\\VEN_1022&DEV_1502", 0x1502 }, // AIE2
            { "PCI\\VEN_1022&DEV_17F0", 0x17F0 }  // AIE2P
        };

        static const std::vector<const GUID*> DEV_CLASSES = {
            &GUID_DEVCLASS_COMPUTEACCELERATOR,
            &GUID_DEVCLASS_SYSTEM
        };

        for (const auto& devClass : DEV_CLASSES) {
            HDEVINFO deviceInfoSet = SetupDiGetClassDevs(devClass, nullptr, nullptr, DIGCF_PRESENT);
            if (deviceInfoSet == INVALID_HANDLE_VALUE) {
                continue;
            }

            SP_DEVINFO_DATA deviceInfoData = { 0 };
            deviceInfoData.cbSize = sizeof(deviceInfoData);

            DWORD index = 0;
            while (npu_info.device_id == -1 && SetupDiEnumDeviceInfo(deviceInfoSet, index, &deviceInfoData)) {
                DWORD requiredSize = 0;

                SetupDiGetDeviceRegistryPropertyA(deviceInfoSet, &deviceInfoData, SPDRP_HARDWAREID, nullptr, nullptr, 0, &requiredSize);

                std::vector<BYTE> buffer(requiredSize);

                if (SetupDiGetDeviceRegistryPropertyA(deviceInfoSet, &deviceInfoData, SPDRP_HARDWAREID, nullptr, buffer.data(), requiredSize, nullptr)) {
                    std::string hardwareId(reinterpret_cast<const char*>(buffer.data()));

                    for (const auto& entry : PCI_IDS) {
                        if (hardwareId.find(entry.first) != std::string::npos) {
                            npu_info.device_id = entry.second;
                            requiredSize = 0;
                            SetupDiGetDeviceRegistryPropertyA(deviceInfoSet, &deviceInfoData, SPDRP_DEVICEDESC, nullptr, nullptr, 0, &requiredSize);

                            buffer.resize(requiredSize);
                            if (SetupDiGetDeviceRegistryPropertyA(deviceInfoSet, &deviceInfoData, SPDRP_DEVICEDESC, nullptr, buffer.data(), requiredSize, nullptr)) {
                                std::string dev_desc(reinterpret_cast<const char*>(buffer.data()));
                                npu_info.device_name = dev_desc;
                            }
                            SP_DEVINSTALL_PARAMS DeviceInstallParams;
                            ZeroMemory(&DeviceInstallParams, sizeof(DeviceInstallParams));
                            DeviceInstallParams.cbSize = sizeof(SP_DEVINSTALL_PARAMS);
                            DeviceInstallParams.FlagsEx |= (DI_FLAGSEX_INSTALLEDDRIVER | DI_FLAGSEX_ALLOWEXCLUDEDDRVS);
                            if (SetupDiSetDeviceInstallParams(deviceInfoSet, &deviceInfoData, &DeviceInstallParams)) {
                                if (SetupDiBuildDriverInfoList(deviceInfoSet, &deviceInfoData, SPDIT_COMPATDRIVER)) {
                                    SP_DRVINFO_DATA DriverInfoData;
                                    DriverInfoData.cbSize = sizeof(SP_DRVINFO_DATA);
                                    if (SetupDiEnumDriverInfo(deviceInfoSet, &deviceInfoData, SPDIT_COMPATDRIVER, 0, &DriverInfoData)) {
                                        npu_info.driver_version_number = DriverInfoData.DriverVersion;
                                        npu_info.driver_version_string = DriverVersionToString(DriverInfoData.DriverVersion).c_str();
                                    }
                                }
                                SetupDiDestroyDriverInfoList(deviceInfoSet, &deviceInfoData, SPDIT_COMPATDRIVER);
                                break;
                            }
                        }
                    }
                }

                ++index;
            }

            SetupDiDestroyDeviceInfoList(deviceInfoSet);

            if (npu_info.device_id != -1) {
                break;
            }
        }
        return npu_info;
    }

    NPUInfo checkCompatibility(DWORDLONG min_driver_version, std::chrono::year_month_day max_date)
    {
        NPUInfo info = extractNPUInfo();

        // Check if supported NPU is present
        if (info.device_id==-1) {
            info.check = Status::NPU_UNRECOGNIZED; 
            return info;
        }

        // Check if minimum version of driver is installed
        if (info.driver_version_number<min_driver_version) {
            info.check = Status::DRIVER_TOO_OLD; 
            return info;            
        }

        // Check for 3 yr EP/driver compatibility window
        std::chrono::year_month_day current_date{std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now())};;
        if (current_date>max_date) {
            info.check = Status::EP_TOO_OLD; 
            return info;              
        }

        info.check = Status::OK; 
        return info;
    }

    NPUInfo checkCompatibility_RAI_1_2()
    {
        // Min driver: 32.0.201.204
        // Max date  : 2027-07-30 (3 yrs after 2024-07-29, release date of RyzenAI 1.2)
        return checkCompatibility(0x20000000c900cc, { std::chrono::July / 30 / 2027 });
    }

    NPUInfo checkCompatibility_RAI_1_3()
    {
        // Min driver: 32.0.201.237
        // Max date  : 2027-11-26 (3 yrs after 2024-11-25, release date of RyzenAI 1.3)
        return checkCompatibility(0x20000000c900ed, { std::chrono::November / 26 / 2027 });
    }
   
} // npu_util
