# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\cnn-examples.mdx:244
cd getting_started_resnet\cpp
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -DCMAKE_INSTALL_PREFIX=. -DCMAKE_PREFIX_PATH=. -B build -S resnet_cifar -DOpenCV_DIR="C:/opencv/build" -G "Visual Studio 17 2022"
