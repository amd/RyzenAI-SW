1. To run the samples(resnet50_pt, perf) directly, just execute related batch file on the root directory.
2. To run with use your own python environment, make sure to change PATH, PYTHONHOME, PYTHONPATH in run*.bat.
And copy related python files(after .) in the python-3.9.7-embed-amd64/python39._pth to your python library directories.

3. To compile the sample, just run build.bat, the result .exe would be created at bin folder.
Note: this script depends on the cmake(min 3.5), MSVC compiler(Visual Studio 16 2019)
Note2: due to safety concerns, the sample code you compiled will not read the environment variable.