

set ROOT=%~dp0

mkdir build
pushd build

cmake -G "Visual Studio 16 2019" -DCMAKE_INSTALL_PREFIX=%ROOT%install -DCMAKE_PREFIX_PATH=%ROOT%install -DBUILD_THIRD_PARTY=ON -DBUILD_GRAPH_ENGINE=ON -DBUILD_SINGLE_LIBRARY=ON -DBUILD_TESTCASES=ON -B. -S..

IF errorlevel 1 (POPD & exit /B %errorlevel%)

cmake --build . --config Release --parallel 4
IF errorlevel 1 (POPD & exit /B %errorlevel%)

popd build
