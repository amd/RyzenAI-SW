# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

import os
import platform

if platform.system() == "Windows":
    path = os.getenv("PATH")
    if path:
        paths = [p.strip() for p in path.split(";")]
        paths = [os.path.realpath(p) for p in paths]
        paths = [p for p in paths if os.path.isdir(p)]
    else:
        paths = []

    for path in paths:
        os.add_dll_directory(path)

from ._ryzenai_torch_cpp import *
