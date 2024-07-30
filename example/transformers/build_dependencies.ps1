# Set script to stop on error
$ErrorActionPreference = "Stop"

# Check if $env:CONDA_PREFIX is set
if (-not $env:CONDA_PREFIX) {
    Write-Host "Error: CONDA_PREFIX is not set."
    exit 1
}

# Set the root directory for the dependency
$CWD = Split-Path -Parent $MyInvocation.MyCommand.Path
$AIERT_CMAKE_PATH = Join-Path $CWD "ext\aie-rt\driver\src"
$AIECTRL_CMAKE_PATH = Join-Path $CWD "ext\aie_controller"
$DD_CMAKE_PATH = Join-Path $CWD "ext\DynamicDispatch"
$env:XRT_DIR = Join-Path $CWD "third_party\xrt-ipu"
# Check if the directory exists
if (-not (Test-Path $AIERT_CMAKE_PATH)) {
    Write-Host "Error: Directory $AIERT_CMAKE_PATH does not exist."
    exit 1
}

if (-not (Test-Path $AIECTRL_CMAKE_PATH)) {
    Write-Host "Error: Directory $AIECTRL_CMAKE_PATH does not exist."
    exit 1
}

# Invoke cmake to build and install the dependency
try {
    cmake -S $AIERT_CMAKE_PATH -B build_aiert -DXAIENGINE_BUILD_SHARED=OFF -DCMAKE_INSTALL_PREFIX=$env:CONDA_PREFIX
    cmake --build build_aiert --target install --config Release
} catch {
    Write-Host "Error: cmake build or installation failed."
    exit 1
}

try {
    cmake -S $AIECTRL_CMAKE_PATH -B build_aiectrl -DCMAKE_PREFIX_PATH=$env:CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$env:CONDA_PREFIX
    cmake --build build_aiectrl --target install --config Release
} catch {
    Write-Host "Error: cmake build or installation failed."
    exit 1
}

try {
    cmake -S $DD_CMAKE_PATH -B build_dd -DCMAKE_PREFIX_PATH=$env:CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$env:CONDA_PREFIX
    cmake --build build_dd --target install --config Release
} catch {
    Write-Host "Error: cmake build or installation failed."
    exit 1
}

Write-Host "Build and installation completed successfully."
