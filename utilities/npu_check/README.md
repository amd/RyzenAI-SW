# NPU Check Utilities

A set of APIs to extract information about the NPU and check compatibility of the VitisAI EP with the rest of the environment.

```
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

    // Checks whether the system configuration is compatible for VitisAI EP 1.2
    NPUInfo checkCompatibility_RAI_1_2();

    // Checks whether the system configuration is compatible for VitisAI EP 1.3
    NPUInfo checkCompatibility_RAI_1_3();

}
```

The `checkCompatibility_RAI_1_2()` and `checkCompatibility_RAI_1_3()` implement compatibility checks for specific versions of the VitisAI EP. These APIs return a `NPUInfo` struct containing various information about the environment.

The `check` field of the `NPUInfo` struct is set to `Status::OK` if the following conditions are met:
 - The system has a supported AMD NPU
 - NPU drivers are installed, and the version satisfies the minimum version required by the VitisAI EP being used.
 - The current date is less than 3 years after the release of the VitisAI EP being used. 

If the `check` field is set to a value other than `Status::OK`, the application should not use the VitisAI EP.

The `device_id` field contains the NPU device ID. This value should be used by the application to determine whether to load a PHX or STX xclbin.
 - 0x1502 -> PHX/HPT
 - 0x17F0 -> STX/KRK

The `npu_util.cpp` file must be compiled with the `/std:c++latest` option.

## Disclaimer

The software included in the repository is provided "as is", without warranty of any kind. It is only intended to serve as an example of how to implement the compatibility checks required before using the VitisAI EP.
