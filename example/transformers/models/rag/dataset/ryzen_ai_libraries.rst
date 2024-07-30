.. Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

##################################
Ryzen AI Library Quick Start Guide
##################################

The Ryzen AI Libraries are built on top of the Ryzen AI drivers and execution infrastructure to provide powerful AI capabilities to C++ applications without the need for training specific AI models and integrating them into the Ryzen AI framework.

Each Ryzen AI library feature offers a simple C++ application programming interface (API) that can be easily incorporated into existing applications.

******************
Supported Features
******************
This release of the Ryzen AI Library supports the following features:

- Depth Estimation

****************
Package Contents
****************

The following files are included with the Ryzen AI Library package:

include/
  C++ header files
windows/
  Binary files for Windows, including both compile time .LIB files and runtime .DLL files
thirdparty_lib/
  Additional dependent libraries for sample applications (e.g., OpenCV binaries)
samples/
  Individual sample applications
LICENSE.txt
  License file
README.rst
  This file

**************************************
Programming guide for C++ Applications
**************************************
Incorporating the Ryzen AI's optimized features into C++ applications can be
done in a few simple steps, as explained in the following sections.

Include Ryzen AI Library headers
================================
The required definitions for compiling each Ryzen AI feature are included in a
corresponding,

  cvml-*<feature-name>*.h

header file under the **include/** folder, where *<feature-name>* is the name
of the desired Ryzen AI feature.

For example, the definitions for the Ryzen AI Depth Estimation feature are
available after adding a line similar to the following example::

  #include <cvml-depth-estimation.h>

Details about each feature's programming interface and expected usage are
provided within their individual include headers.

Create Ryzen AI Library context
===============================
Each Ryzen AI Library feature is created against a *CVML context*. The context provides access to common functions for logging and other purposes. A pointer to a new
context may be obtained by calling the *CreateContext()* function::

  auto ryzenai_context = amd::cvml::CreateContext();

When no longer needed, the context may be released using its *Release()*
member function::

  ryzenai_context->Release();

Create Ryzen AI Library feature object
======================================
The application programming interface for each feature is provided through a
*Ryzen AI Library C++ feature object* that may be instantiated afer a
Ryzen AI Library context has been created.

The following example instantiates a feature object for the depth estimation
library::

  amd::cvml::DepthEstimation ryzenai_depth_estimation(ryzenai_context);

Encapsulate image buffers
=========================
The Ryzen AI Library defines its own *Image* class to represent images
and video frame buffers. Each *Image* object is assigned a specific format
and data type on creation. For example, you can use the following code to create an *Image* to encapsulate an incoming
RGB888 frame buffer::

  amd::cvml::Image ryzenai_image(amd::cvml::Image::Format::kRGB,
                                 amd::cvml::Image::DataType::kUint8, width,
                                 height, data_pointer);

Execute the feature
===================
To execute a Ryzen AI feature on a provided input, call the appropriate
*execution* member function of the Ryzen AI Library feature object.

For example, the following code executes a single instance of the depth
estimation library, using the *ryzenai_image* from the previous section::

  // encapsulate output buffer
  amd::cvml::Image ryzenai_output(amd::cvml::Image::Format::kGrayScale,
                                  amd::cvml::Image::DataType::kFloat32,
                                  output_width, output_height, output_pointer);

  // execute the feature
  ryzenai_depth_estimation.GenerateDepthMap(ryzenai_image, &ryzenai_output);

*********************************************
Building applications with Ryzen AI Libraries
*********************************************
When building applications against the Ryzen AI Library, ensure that the
library's,

  include/

folder is part of the compiler's include paths, and that the library's,

  windows/

folder has been added to the linker's library paths.

Depending on the application's build environment, you might also need to
explicitly list which of the Ryzen AI Library's .LIB files (when building for
Windows applications) need to be linked.

***********************************************
Executing Ryzen AI Library enabled applications
***********************************************
When executing Windows applications built against the Ryzen AI Library, ensure
that one of the following conditions is met:

1. The Ryzen AI Library dll's are in the same folder as the application
   executable.
2. The Ryzen AI Library's **windows/** folder has been added to the PATH
   environment variable.

*******
Example
*******

Examples of the Ryzen-AI Library can be found `Ryzen AI Software repo <https://github.com/amd/RyzenAI-SW/tree/main/example/Ryzen-AI-Library>`_


****************
Revision History
****************
+-------------------+----------+------------------+
| Date              | Revision | Notes            |
+===================+==========+==================+
| December 04, 2023 | 1.0      | Initial revision |
+-------------------+----------+------------------+

..
  ------------

  #####################################
  License
  #####################################

  Ryzen AI is licensed under MIT License. Refer to the LICENSE file for the full license text and copyright notice.
