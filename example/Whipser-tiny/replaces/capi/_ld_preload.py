# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# This file can be modified by setup.py when building a manylinux2010 wheel
# When modified, it will preload some libraries needed for the python C extension

import warnings

try:
    # This import is necessary in order to delegate the loading of libtvm.so to TVM.
    import tvm
except ImportError as e:
    warnings.warn(
        f"WARNING: Failed to import TVM, libtvm.so was not loaded. More details: {e}"
    )
try:
    # Working between the C++ and Python parts in TVM EP is done using the PackedFunc and
    # Registry classes. In order to use a Python function in C++ code, it must be registered in
    # the global table of functions. Registration is carried out through the JIT interface,
    # so it is necessary to call special functions for registration.
    # To do this, we need to make the following import.
    import onnxruntime.providers.tvm
except ImportError as e:
    warnings.warn(
        f"WARNING: Failed to register python functions to work with TVM EP. More details: {e}"
    )
