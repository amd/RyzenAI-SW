# Ryzen AI CVML library

> import CIStatus from '@site/src/components/CIStatus';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

import CIStatus from '@site/src/components/CIStatus';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Ryzen AI CVML library

<CIStatus validated={false} />

The Ryzen AI CVML libraries build on top of the Ryzen AI drivers and execution infrastructure to provide powerful AI capabilities to C++ applications without having to worry about training specific AI models and integrating them to the Ryzen AI framework.

Each Ryzen AI CVML library feature offers a simple C++ application programming interface (API) that can be easily incorporated into existing applications. The following AI features are currently available,

- **Depth Estimation**: Generates a depth map to assess relative distances within a two-dimensional image.
- **Face Detection**: Identifies and locates faces within an image.
- **Face Mesh**: Constructs a mesh overlay of landmarks for a specified facial image.

The Ryzen AI CVML library is distributed through the RyzenAI-SW Github repository: [Ryzen-AI-CVML-Library](/models-tutorials/vision/cvml)

## Building sample applications

This section describes the steps to build Ryzen AI CVML library sample applications. Before starting, ensure that the following prerequisites are available in the build environment,

- CMake, version 3.18 or newer
- C++ compilation toolchain. On Windows, this may be Visual Studio's "Desktop development with C++" build tools, or a comparable C++ toolchain
- OpenCV, version 4.11 or newer

### Navigate to the folder containing Ryzen AI samples

Download the Ryzen AI CVML sources, and go to the 'samples' sub-folder of the library.

<Tabs>
  <TabItem value="windows" label="Windows (PowerShell)" default>

```powershell
git clone https://github.com/amd/RyzenAI-SW.git -b main --depth-1
cd RyzenAI-SW\Ryzen-AI-CVML-Library\samples
```

</TabItem>
  <TabItem value="linux" label="Linux (Bash)">

```bash
git clone https://github.com/amd/RyzenAI-SW.git -b main --depth-1
cd RyzenAI-SW/Ryzen-AI-CVML-Library/samples
```

</TabItem>
</Tabs>

### OpenCV libraries

Ryzen AI CVML library samples make use of OpenCV, so set an environment variable to let the build scripts know where to find OpenCV.

<Tabs>
  <TabItem value="windows" label="Windows (PowerShell)" default>

```powershell
set OPENCV_INSTALL_ROOT=<location of OpenCV libraries>
```

</TabItem>
  <TabItem value="linux" label="Linux (Bash)">

```bash
export OPENCV_INSTALL_ROOT=<location of OpenCV libraries>
```

</TabItem>
</Tabs>

### Build Instructions

Create a build folder and use CMAKE to build the sample(s).

<Tabs>
  <TabItem value="windows" label="Windows (PowerShell)" default>

```powershell
mkdir build
cmake -S %CD% -B %CD%\build -DOPENCV_INSTALL_ROOT=%OPENCV_INSTALL_ROOT%
cmake --build %CD%\build --config Release
```

</TabItem>
  <TabItem value="linux" label="Linux (Bash)">

```bash
mkdir build
cmake -S $PWD -B $PWD/build -DOPENCV_INSTALL_ROOT=$OPENCV_INSTALL_ROOT
cmake --build $PWD/build --config Release
```

</TabItem>
</Tabs>

The compiled sample applications will be placed in `build/&lt;application&gt;/Release` folders under the `samples` folder.

## Running sample applications

This section describes how to execute Ryzen AI CVML library sample applications.

### Update the console and/or system PATH

Ryzen AI CVML library applications need to be able to find the library files.

<Tabs>
  <TabItem value="windows" label="Windows (PowerShell)" default>

```powershell
set PATH=%PATH%;<location of Ryzen AI CVML library package>\windows
set PATH=%PATH%;%OPENCV_INSTALL_ROOT%\x64\vc16\bin
```

</TabItem>
  <TabItem value="linux" label="Linux (Bash)">

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<location of Ryzen AI CVML library package>/linux
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENCV_INSTALL_ROOT/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/xilinx/xrt/lib
```

</TabItem>
</Tabs>

Adjust the aforementioned commands to match the actual location of Ryzen AI and OpenCV libraries, respectively.

### Select an input source (image or video)

Ryzen AI CVML library samples can accept a variety of image and video input formats, or even open the default camera on the system if "0" is specified as an input.

In this example, a publicly available video file is used for the application's input. The following command downloads a video file and saves it locally as 'dancing.mp4':

```bash
curl -o dancing.mp4 https://videos.pexels.com/video-files/4540332/4540332-hd_1920_1080_25fps.mp4
```

### Execute the sample application

Finally, the previously built sample application may be executed with the selected input source.

<Tabs>
  <TabItem value="windows" label="Windows (PowerShell)" default>

```powershell
build\cvml-sample-depth-estimation\Release\cvml-sample-depth-estimation.exe -i dancing.mp4
```

</TabItem>
  <TabItem value="linux" label="Linux (Bash)">

```bash
build/cvml-sample-depth-estimation/Release/cvml-sample-depth-estimation.exe -i dancing.mp4
```

</TabItem>
</Tabs>

---

Ryzen AI is licensed under MIT License. Refer to the LICENSE file for the full license text and copyright notice.
