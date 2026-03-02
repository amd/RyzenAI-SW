# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\cvml-library.mdx:94
mkdir build
cmake -S $PWD -B $PWD/build -DOPENCV_INSTALL_ROOT=$OPENCV_INSTALL_ROOT
cmake --build $PWD/build --config Release
