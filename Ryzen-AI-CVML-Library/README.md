<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI CVML Library </h1>
    </td>
 </tr>
</table>

<a name="top"></a>
The Ryzen AI Libraries build on top of the Ryzen AI drivers and execution
infrastructure to provide powerful AI capabilities to C++ applications without
having to worry about training specific AI models and integrating them to the
Ryzen AI framework.

Each Ryzen AI CVML library feature offers a simple C++ application programming
interface (API) that can be easily incorporated into existing applications.

[TOC]

Package Contents
================

The Ryzen AI CVML library consists of the following files and folders,

- cmake/

    Packaging info for CMake's find_package function

- include/

    C++ header files

- windows/

    Binary files for Windows, including both compile time .LIB files and runtime .DLL/.GRAPHLIB/.AMODEL files

- linux/

    Binary files for Linux, including compile and runtime .SO files

- samples/

    Individual sample applications

- LICENSE.txt

    License file

[Back to top](#top)

Executing Ryzen AI CVML library enabled applications
====================================================

The Ryzen AI CVML library selects the appropriate hardware (e.g., GPU or NPU) and framework for performing inference operations by default. An API is also available to set the preferred inference backend for those applications that wish to do so.

In order to execute applications that utilize the Ryzen AI CVML library, the appropriate drivers must first be installed on the target system, and the Ryzen AI CVML library files must be included with the application itself.

> Ryzen AI CVML library features that utilize the ONNX backend for NPU operations may experience a longer startup latency the first time they are executed on a device. This increased startup latency does not occur for subsequent runs the feature.

<a name="dependencies"></a>
Prerequisites and dependencies
------------------------------

The AMD Adrenalin and Ryzen AI drivers should be installed before attempting to execute Ryzen AI CVML library applications.

[Back to top](#top)

<a name="windows_setup"></a>
### Windows Setup

The following installations are for Windows OS.  For Linux OS, follow [Ubuntu Setup](#linux_setup) instructions 

### AMD Adrenalin driver

Install either the following Adrenalin driver or a newer one,

https://www.amd.com/en/support/download/drivers.html

[Back to top](#top)

### AMD Ryzen AI driver

Install the latest Ryzen AI NPU driver from,

https://ryzenai.docs.amd.com/en/latest/inst.html

[Back to top](#top)

<a name="linux_setup"></a>
### Ubuntu Setup

The following installations are for Ubuntu.  For Windows OS, follow [Windows Setup](#windows_setup) instructions

Ensure that the following software tools/packages are installed on the development system.

1. OS: Ubuntu 22.04 or Ubuntu 24.04 (recommended 6.11.0-rc2 linux kernel)
2. Install latest Ryzen AI NPU driver from https://ryzenai.docs.amd.com/en/latest/inst.html
3. Vulkan SDK

#### Installing VulkanSDK and 22.04/24.04 specific installs
```
UBUNTU_CODENAME=$(. /etc/os-release; echo "$UBUNTU_CODENAME")
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.296-$UBUNTU_CODENAME.list https://packages.lunarg.com/vulkan/1.3.296/lunarg-vulkan-1.3.296-$UBUNTU_CODENAME.list
sudo apt update
sudo apt install vulkan-sdk
```

#### Additional installation for Ubuntu 22.04: Update MESA Vulkan Drivers
```
sudo apt update && sudo apt upgrade
sudo add-apt-repository ppa:kisak/kisak-mesa -y
sudo apt update
sudo apt upgrade
```

#### Additional installation for Ubuntu 24.04
```
sudo apt install libnsl2 gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly -y

DEP_PKG_LIST="https://launchpad.net/ubuntu/+archive/primary/+files/libmpdec3_2.5.1-2build2_amd64.deb \
    https://launchpad.net/ubuntu/+archive/primary/+files/libpython3.10-minimal_3.10.4-3_amd64.deb \
    https://launchpad.net/ubuntu/+archive/primary/+files/libpython3.10-stdlib_3.10.4-3_amd64.deb \
    https://launchpad.net/ubuntu/+archive/primary/+files/libpython3.10_3.10.4-3_amd64.deb \
    https://launchpad.net/ubuntu/+archive/primary/+files/libprotobuf23_3.12.4-1ubuntu7_amd64.deb \
    https://launchpad.net/ubuntu/+archive/primary/+files/libgoogle-glog0v5_0.5.0+really0.4.0-2_amd64.deb \
    https://launchpad.net/ubuntu/+archive/primary/+files/libtiff5_4.3.0-6_amd64.deb \
    https://launchpad.net/ubuntu/+archive/primary/+files/libilmbase25_2.5.7-2_amd64.deb \
    https://launchpad.net/ubuntu/+archive/primary/+files/libopenexr25_2.5.7-1_amd64.deb"

for pkg in $DEP_PKG_LIST
do
    echo $pkg
    wget $pkg
    sudo dpkg -i *.deb
    rm *.deb
done
```

[Back to top](#top)

Locating Ryzen AI CVML library runtime files for [Windows](#windows_execution) and [Linux](#linux_execution)
--------------------------------------------

When executing applications built against the Ryzen AI CVML library, ensure that:

<a name="windows_execution"></a>
For Windows, either one of the following conditions are met:
1. The Ryzen AI CVML library runtime dll and graphlib files are in the same folder
   as the application executable.
2. The Ryzen AI CVML library's **windows/** folder has been added to the PATH
   environment variable.

<a name="linux_execution"></a>
For Linux, all the following conditions are met.
1. Add location of **linux/** to LD_LIBRARY_PATH
2. Add location of NPU driver libs (/opt/xilinx/xrt/lib) to LD_LIBRARY_PATH

[Back to top](#top)

Programming guide for C++ Applications
======================================

Incorporating the Ryzen AI's optimized features into C++ applications can be
done in a few simple steps, as explained in the following sections.

[Back to top](#top)

Include Ryzen AI CVML library headers
-------------------------------------

The required definitions for compiling each Ryzen AI feature are included in a
corresponding,

    cvml-feature-name.h

header file under the **include/** folder, where `feature-name` is the name
of the desired Ryzen AI feature.

For example, the definitions for the Ryzen AI Depth Estimation feature are
available after adding a line similar to the following example:

    #include <cvml-depth-estimation.h>

Details about each feature's programming interface and expected usage are
provided within their individual include headers.

[Back to top](#top)

Create Ryzen AI CVML library context
------------------------------------

Each Ryzen AI CVML library feature is created against a *CVML context* (see amd::cvml::Context).
The context provides access to common functions for logging, etc. A pointer to a new
context may be obtained by calling the `amd::cvml::CreateContext()` function:

    auto ryzenai_context = amd::cvml::CreateContext();

When no longer needed, the context may be released using its `Release()`
member function:

    ryzenai_context->Release();

[Back to top](#top)

Create Ryzen AI CVML library feature object
-------------------------------------------

The application programming interface for each feature is provided via a
*Ryzen AI CVML library C++ feature object* that may be instantiated once a
Ryzen AI CVML library context has been created.

The following example instantiates a feature object for the depth estimation
library:

    amd::cvml::DepthEstimation ryzenai_depth_estimation(ryzenai_context);

[Back to top](#top)

Encapsulate image buffers
-------------------------

The Ryzen AI CVML library defines its own *Image* class (see amd::cvml::Image) for
representing images and video frame buffers. Each *Image* object is assigned a
specific format and data type on creation. For example, an *Image* to encapsulate
an incoming RGB888 frame buffer can be created with the following code:

    amd::cvml::Image ryzenai_image(amd::cvml::Image::Format::kRGB,
                                   amd::cvml::Image::DataType::kUint8, width,
                                   height, data_pointer);

[Back to top](#top)

Execute the feature
-------------------

To execute a Ryzen AI feature on a provided input, call the appropriate
*execution* member function of the Ryzen AI CVML library feature object.

For example, the following code executes a single instance of the depth
estimation library, using the *ryzenai_image* from the previous section:

    // encapsulate output buffer
    amd::cvml::Image ryzenai_output(amd::cvml::Image::Format::kGrayScale,
                                    amd::cvml::Image::DataType::kFloat32,
                                    output_width, output_height, output_pointer);

    // execute the feature
    ryzenai_depth_estimation.GenerateDepthMap(ryzenai_image, &ryzenai_output);

[Back to top](#top)

Building applications with Ryzen AI CVML Libraries
==================================================

When building applications against the Ryzen AI CVML library, ensure that the
library's,

    include/

folder is part of the compiler's include paths, and that the library's,

    windows/ or linux/

folder has been added to the linker's library paths.

Depending on the application's build environment, it may also be necessary to
explicitly list which of the Ryzen AI CVML library's .LIB files (when building for
Windows applications) need to be linked.

[Back to top](#top)

Building Ryzen AI CVML library applications with CMake
------------------------------------------------------

If CMake is used for the application's build environment, the necessary
include folder and link libraries can be added with the following lines
in the application's CMakeLists.txt file:

    # find Ryzen AI CVML library and set include folders
    find_package(RyzenAILibrary REQUIRED PATHS ${AMD_CVML_SDK_ROOT})

    # add Ryzen AI CVML library linker libraries
    target_link_libraries(${PROJECT_NAME} ${RyzenAILibrary_LIBS})

where `AMD_CVML_SDK_ROOT` defines the location of the Ryzen AI CVML library files and
`PROJECT_NAME` defines the name of the application build target.

[Back to top](#top)

Building Ryzen AI CVML library sample applications
--------------------------------------------------

In addition to general Ryzen AI CVML library prerequisite and dependencies listed
under [Prerequisites and dependencies](#dependencies), the included sample
applications also make use of OpenCV for reading input images/videos/camera
and displaying final output windows. A copy of [OpenCV](https://opencv.org/)
will need to be downloaded to the development system before the samples can
be rebuilt and/or executed.

Ensure the following prerequisites have been set up to build Ryzen AI CVML library sample applications:
- CMake has been installed and is available in the system/user path
- On Windows, Visual Studio's "Desktop development with C++" build tools, or a comparable C++ toolchain, has been installed
- The location of OpenCV libraries has been assigned to the `OPENCV_INSTALL_ROOT` environment variable
- The relative locations of the `include`, `windows`, `linux`, and `samples` folders are unchanged

The following are CMake commands for building samples.

On Windows:
```
cd samples/
mkdir build
cmake -S %CD% -B %CD%\build -DOPENCV_INSTALL_ROOT=$OPENCV_INSTALL_ROOT$
cmake --build %CD%\build --config Release
```

On Linux:
```
cd samples/
mkdir build
cmake -S $PWD -B $PWD/build -DOPENCV_INSTALL_ROOT=$OPENCV_INSTALL_ROOT
cmake --build $PWD/build --config Release
```

[Back to top](#top)

License
=======

Refer to the LICENSE.txt file for the full license text and copyright notice.

[Back to top](#top)

Copyrights and Trademarks
=========================

**2021 Advanced Micro Devices, Inc.** All rights reserved.

The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD's products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale. Any unauthorized copying, alteration, distribution, transmission, performance, display or other use of this material is prohibited.

---

**Trademarks**

AMD, the AMD Arrow logo, AMD AllDay, AMD Virtualization, AMD-V, PowerPlay, Vari-Bright, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Dolby is a trademark of Dolby Laboratories.

HDMI is a trademark of HDMI Licensing, LLC.

HyperTransport is a licensed trademark of the HyperTransport Technology Consortium.

Microsoft, Windows, Windows Vista, and DirectX are registered trademarks of Microsoft Corporation in the US and/or other countries.

PCIe is a registered trademark of PCI-Special Interest Group (PCI-SIG).

USB Type-C &reg; and USB-C &reg; are registered trademarks of USB Implementers Forum.

**Dolby Laboratories, Inc.**

Manufactured under license from Dolby Laboratories.

**Rovi Corporation**

This device is protected by U.S. patents and other intellectual property rights. The use of Rovi Corporation's copy protection technology in the device must be authorized by Rovi Corporation and is intended for home and other limited pay-per-view uses only, unless otherwise authorized in writing by Rovi Corporation.

Reverse engineering or disassembly is prohibited.

USE OF THIS PRODUCT IN ANY MANNER THAT COMPLIES WITH THE MPEG ACTUAL OR DE FACTO VIDEO AND/OR AUDIO STANDARDS IS EXPRESSLY PROHIBITED WITHOUT ALL NECESSARY LICENSES UNDER APPLICABLE PATENTS. SUCH LICENSES MAY BE ACQUIRED FROM VARIOUS THIRD PARTIES INCLUDING, BUT NOT LIMITED TO, IN THE MPEG PATENT PORTFOLIO, WHICH LICENSE IS AVAILABLE FROM MPEG LA, L.L.C., 6312 S. FIDDLERS GREEN CIRCLE, SUITE 400E, GREENWOOD VILLAGE, COLORADO 80111.

**xtensor, xtl, xsimd**

Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht
Copyright (c) 2016, QuantStack
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

[Back to top](#top)

Revision History
================

Date               | Revision  | Notes
------------------ | --------- | -----------------------------
November 30, 2023  |  1.0      | Initial revision
April 2, 2024      |  1.1      | Include driver/copyright info
March 7, 2025      |  1.2      | Minor updates and notes

[Back to top](#top)

---

Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
