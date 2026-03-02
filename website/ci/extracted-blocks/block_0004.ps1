# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\cvml-library.mdx:85
mkdir build
cmake -S %CD% -B %CD%\build -DOPENCV_INSTALL_ROOT=%OPENCV_INSTALL_ROOT%
cmake --build %CD%\build --config Release
