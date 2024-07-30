# PSx Model Verification

The utilities present in this directory enables the generation of unit subgraphs, reference/golden data for the verification of PSx models at subgraph and full graph level.
Currently, the tests run on CPU only and AIE verification can be enabled by using Custom Ops.

## Setup

Create conda env
```powershell
# Create a conda env
conda create -n dod-compiler-suite python==3.9.13
# Activate
conda activate dod-compiler-suite
```

Install the required packages
```powershell
pip install -r requirements.txt
```

If Vitis-AI EP isn't installed already, install using the wheels
```powershell
# Install Vitis-AI ORT wheel
pip install --force-reinstall <path/to/onnxruntime_vitisai-1.17.0-cp39-cp39-win_amd64.whl>

# Install Vitis-AI VOE wheel
pip install --force-reinstall <path/to/voe-1.2.0-cp39-cp39-win_amd64.whl>
```

:pushpin: Above `.whl` files can be found inside the VitisAI-EP release package. Alternatively, they can also be built by building [vai-rt](https://gitenterprise.xilinx.com/VitisAI/vai-rt) from source.

## CPU Inference of PSx Model and Reference Generation

:pushpin: Input data for PSx models was shared by MS.

Script Usage
```powershell
infer - Version: 1.0, 2024
USAGE:
  infer.bat [Options]
 -----------------------------------------------------------------------
 --help          Shows this help
 --model         Specify onnx model path
 --in-data-dir   Specify input data directory
 --ep            Specify execution provider (Default: cpu) (Optional)
 --config        Specify config file path (needed to vai-ep)
 --xclbin        Specify XCLBIN file path (needed to vai-ep)
 --out-data-dir  Specify output data directory
 --npz           Save output files as .npz (Default: .raw) (Optional)
 --en-cache      Enable cache directory, reuse cached data (Optional)
 --iters         Number of iterations for session run (Default=1) (Optional)
 -----------------------------------------------------------------------
```

Run PSx model inference
```powershell
# Run inference of any PSx Model
### CPU-EP
.\infer.bat --model <path/to/onnx/model> --in-data-dir <path/to/input/data> --ep cpu

### VitiAI-EP
.\infer.bat --model <path/to/onnx/model> --in-data-dir <path/to/input/data> --ep vai --config <path/to/config/json> --xclbin <path/to/xclbin>

### To save output data, add --out-data-dir to the command
.\infer.bat --model <path/to/onnx/model> --in-data-dir <path/to/input/data> --ep cpu --out-data-dir <path/to/output/directory>
```

Generate reference input/output for unit/subgraph tests
```powershell
# To generate intermediate data, we need to update the original model with new output nodes.
python add_output_taps.py --model-in <path/to/original/psx/onnx/model> --model-out <path/to/updated/onnx/model>

# Run inference with updated PSx Model and Generate reference data
.\infer.bat --model <path/to/updated/onnx/model> --in-data-dir <path/to/input/data> --out-data-dir <path/to/output/directory>
```
:pushpin: Output data will be stored in the output data directory as `.raw` files. To save output files in `.npz` format, add `--npz` option to the command.

## Subgraph Generation

To enable testing at subgraph level, we need to extact subgraphs which we have identified to support.

```powershell
# Generate subgraphs (using VitisAI EP, if you have installed it already)
python -m voe.passes.op_fusion --input_model_path <path/to/input/model> --gen_subgraphs
--xclbin <XCLBIN Kernel Precision>

# Generate subgraphs (using onnx-utils)
python ..\OpFusion\op_fusion.py --input_model_path <path/to/input/model> --gen_subgraphs --xclbin <XCLBIN Kernel Precision>

# Note: For PSF, a8w8 XCLBIN is used
python -m voe.passes.op_fusion --input_model_path <path/to/input/model> --gen_subgraphs
--xclbin a8w8
# Note: For PSH/PSJ, a16w8 XCLBIN is used
python -m voe.passes.op_fusion --input_model_path <path/to/input/model> --gen_subgraphs
--xclbin a16w8
```

It generates subgraphs for all the OPs supported. Check the output tree for a given input model `Model_PSF_v1.0.4_a8w8.onnx`.

```powershell
Model_PSF_v1.0.4_a8w8
│
├───QLayerNorm
│   ├───fused_subgraphs
│   └───subgraphs
├───QMatMul
│   ├───fused_subgraphs
│   └───subgraphs
├───QMatMulAdd
│   ├───fused_subgraphs
│   └───subgraphs
├───QMatMulAddGelu
│   ├───fused_subgraphs
│   └───subgraphs
├───QMHAGRPB
│   ├───fused_subgraphs
│   └───subgraphs
└───QSkipAdd
    ├───fused_subgraphs
    └───subgraphs
```

## Run Subgraph Tests

To run subgraph tests, run the `test_subgraph.py` utility.
```powershell
# Check usage
python test_subgraphs.py --help

usage: test_subgraphs.py [-h] --model MODEL [--out-dir OUT_DIR] --data DATA
        [--ep {cpu, CPU, vai, VAI}] [--config CONFIG] [--xclbin XCLBIN]
        [--repeat REPEAT] [--dll DLL] [--filter FILTER] 
        [--reports-dir REPORTS_DIR] [--verbose]
```

Arguments:

| Option        | Type     | Default | Description |
|---------------|----------|---------|-------------|
| -h, --help    | Optional | NA      | Show this help message and exit |
| --model       | Required | NA      | Path to ONNX model *directory or file* |
| --data-dir    | Required | NA      | Input data *directory or file* |
| --out-dir     | Optional | NA      | Output data directory |
| --dll         | Optional | NA      | Custom OP DLL path |
| --filter      | Optional | NA      | Filter by file name charactors |
| --ep          | Optional | CPU     | Execution provider [cpu, vai]|
| --config      | Optional | NA      | Config JSON file path (Required for VAI-EP) |
| --xclbin      | Optional | NA      | XCLBIN file path (Required for VAI-EP) |
| --repeat      | Optional | 1       | Run each session repeat count times, checks output repeatability |
| --reports-dir | Optional | reports | Directory path where the reports should be saved |
| --verbose     | Optional | False   | Enable more prints |

```powershell
# Run all subgraphs

### CPU EP
python test_subgraphs.py --model-dir <path/to/subgraph/directory> --data-dir <path/to/reference/data/directory> --ep cpu

### VAI EP
python test_subgraphs.py --model-dir <path/to/subgraph/directory> --data-dir <path/to/reference/data/directory> --ep vai --config <path/to/config> --xclbin <path/to/xclbin>

### To save output data for subgraphs add "--out-dir" option with output directory path
python test_subgraphs.py --model-dir <path/to/subgraph/directory> --data-dir <path/to/reference/data/directory> --ep vai --config <path/to/config> --xclbin <path/to/xclbin> --out-dir <path/to/output/dir>

### To save reports to specific dir "--reports-dir" option with directory path
### Default reports dir is ".\reports"
python test_subgraphs.py --model-dir <path/to/subgraph/directory> --data-dir <path/to/reference/data/directory> --ep vai --config <path/to/config> --xclbin <path/to/xclbin> --out-dir <path/to/output/dir> --reports-dir <path/to/reports/dir>
```

### Check Test Results

:pushpin: `test_subgraph.py` outputs a table with a entry for each test along with the average L2-Norm over all input data samples. Reports directory contains a consolidated `csv` file with results for all runs inside `summary.csv` and failures in `failures.csv`.


