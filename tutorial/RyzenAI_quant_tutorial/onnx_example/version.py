#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Define vai_q_onnx version information."""

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION = '1'
_MINOR_VERSION = '16'
_PATCH_VERSION = '0'

# When building releases, we can update this value on the release branch to
# reflect the current release candidate ('rc0', 'rc1') or, finally, the official
# stable release (indicated by `_VERSION_SUFFIX = ''`). Outside the context of a
# release branch, the current version is by default assumed to be a
# 'development' version, labeled 'dev'.
_VERSION_SUFFIX = 'dev'

# Example, '1.16.0'
__version__ = '.'.join([
    _MAJOR_VERSION,
    _MINOR_VERSION,
    _PATCH_VERSION,
])

# The default string could be replaced with a valid string by the build script
__git_version__ = 'COMMIT_SED_MASK'

if ('SED_MASK' not in __git_version__):
    # Example, '1.16.0+12ba01a'
    __version__ = '{}+{}'.format(__version__, __git_version__)
