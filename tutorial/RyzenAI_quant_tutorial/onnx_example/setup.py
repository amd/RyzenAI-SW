#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Install vai_q_onnx."""
import datetime
import os
import sys
import subprocess

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

from version import __version__, _VERSION_SUFFIX  # pylint: disable=g-import-not-at-top

# Get commit id
try:
    commit_version = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
except Exception as e:
    commit_version = "none"

# Update version
version_file = "version.py"

with open(version_file) as ver_file:
    version_file_path = os.path.join("vai_q_onnx", version_file)

    new_file = open(version_file_path, 'w')
    for line in ver_file:
        if "COMMIT_SED_MASK" in line:
            line = line.replace("COMMIT_SED_MASK", commit_version)
        new_file.write(line)
    new_file.close()

# Parse input
if '--release' in sys.argv:
    release = True
    sys.argv.remove('--release')
else:
    # Build a nightly package by default.
    release = False

if release:
    # Add commit id for release
    # '0.0.1+abcdefg'
    project_name = 'vai-q-onnx'
    version_number = __version__ + "+" + commit_version
else:
    # Nightly releases use date-based versioning of the form
    # '0.0.1.dev20180305'
    project_name = 'vai-q-onnx-nightly'
    datestring = datetime.datetime.now().strftime('%Y%m%d')
    version_number = __version__ + "." + _VERSION_SUFFIX + datestring

# Load requirements
with open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "requirements.txt")) as req_file:
    requirements = req_file.read().splitlines()


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return False


setup(
    name=project_name,
    version=version_number,
    description='Xilinx Vitis AI Quantizer for ONNX. '
    'It is customized based on [Quantization Tool](https://github.com/microsoft/onnxruntime/tree/rel-1.14.0/onnxruntime/python/tools/quantization).',
    author='Xiao Sheng',
    author_email='kylexiao@xilinx.com',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    # Add in any packaged data.
    include_package_data=True,
    package_data={'': ['*.so', '*.json']},
    exclude_package_data={'': ['BUILD', '*.h', '*.cc']},
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'pip_pkg': InstallCommandBase,
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='onnx model optimization machine learning',
)
