#!/usr/bin/env python
from setuptools import setup, find_packages

readme = open("README.md", encoding="utf-8").read()
license = open("LICENSE", encoding="utf-8").read()
VERSION = "0.1.0"

requirements = ["onnx==1.15.0", "numpy", "tabulate"]

setup(
    # Metadata
    name="dd_helper",
    version=VERSION,
    author="AMD Inc.",
    description="A tool for ONNX model optimization: A parser, editor and profiler tool for ONNX models.",
    long_description=readme,
    long_description_content_type="text/markdown",
    # TODO: License
    license=license,
    # Package info
    packages=find_packages(),
    # Zip
    zip_safe=True,
    # External reqs
    install_requires=requirements,
    # Classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
