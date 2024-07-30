# Onnx-utils Repo

## Optimizer

- Find and fuse graph patterns into optimized fused nodes.
- Generate synthetic graphs replicating the shapes in ONNX models.
- Quick analysis of ONNX models.

## Prerequisites

```powershell
# Create a conda env
conda create -n dod-compiler-suite python==3.10

# Activate conda env
conda activate dod-compiler-suite

# Install Suite
pip install -e .
```

## Usage

### Create Fused ONNX Graphs with QOP patterns

- Duplicates the `DequantizeLinear` layer
  - As we are fusing dq---op---op, there should not be any branching(multiple children) near dq layer as it'll be fused with the 1st child node of dq layers and when 2nd pattern appears it cant see the dq layer as it was already fused with the 1st child node.

- Loads the patterns from file (precomuputed)
  - Can be used for any ONNX model

- Fused the patterns and simultaneously saves the subgraphs for each pattern
  - Supported patterns QMatMul, QMatMulAdd, QMatMulAddGelu, QLayerNorm, QMHAGRPB

```powershell
# Generate optimized ONNX model
python -m dd_helper.optimizer.op_fusion --input_model_path <path/to/input/onnx/model> --output_model_path <path/to/output/optimized.model> --xclbin <path/to/xclbin>

# Extract subgraphs
python -m dd_helper.optimizer.op_fusion --input_model_path <path/to/input/onnx/model> --gen_subgraphs --xclbin <path/to/xclbin>
```

### Create synthetic graphs replicating PSx models

- Generalized implementation of ONNX model creation specifically `"N"` headed self attention models, abstracting the creation of ONNX nodes, tensors required for the ONNX graphs.
- Creates `"N"` headed model which replicates the PS models.
- Used by DOD repo to create the synthetic PSx like models to facilitate easy testing
- Supported Unit Models of optypes
  - MatMul
  - MatMulAdd
  - MatMulAddGelu
  - LayerNorm
  - AddSkip
- Can create models with the combination of the above unit nodes
- Can generate full model replicas for PSF, PSJ, PSH models (with the exclusion of unsupported kernels like Tanh, Lpnormalization, Sigmoid outside GRPB etc)

> Note: Please uncomment necessary function to create graphs

```powershell
# Go to utils directory
cd dd_helper/utils

# Generate synthetic model and unit models
python create_model_for_fusion.py --heads 12 --model_name PSK --excel_sheet ..\archive\shapes.csv
```

## TODO

- Generalization of `create_model_for_fusion.py`
- Add ops for supporting dod testing of upcomming kernels
- Creation of models using ONNX Functions for easy cpu-testing of sub graphs (`create_function.py`) - WIP
