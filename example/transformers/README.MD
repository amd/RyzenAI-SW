# OPT & VIT on the IPU

This repository consists of methods to run Transformers in PyTorch and ONNX with operators dispatch to AIE. 

# Setup Transformers
## Step 1: Download repository and create conda environment based on provided yaml file
Open Anaconda prompt on Windows PC.

```
git clone https://github.com/amd/RyzenAI-SW.git
cd RyzenAI-SW\example\transformers
conda env create --file=env.yaml
conda activate ryzenai-transformers
```

## Step 2: Setup environment 
```
setup.bat
```

## Step 3: Build dependencies
```
pip install ops\cpp --force-reinstall
```

## Step 4: Install ONNX EP for running ONNX based flows

Please refer to [OPT-ONNX](opt-onnx/README.MD)

Please refer to [VISION-TRANSFORMER-ONNX](vision-transformer-onnx/README.MD)

# Run Transformers

## 1. **ONNX Runtime Flow**:  [VISION-TRANSFORMER-ONNX](vision-transformer-onnx)

## 2. **ONNX Runtime Flow**:  [OPT-ONNX](opt-onnx)

## 3. **PyTorch Flow**: [OPT-PYTORCH](opt-pytorch)
