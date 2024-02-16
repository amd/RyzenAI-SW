# Create & Run E2E ONNX Models

Examples to create and run ONNX Models with pre/post processing as custom operators.

## Examples

Activate conda env (onnx-vai), if not done already.
```powershell
## This is required if the VAI-EP packages are installed in conda env
## Skip this if you do not have VAI EP installed in conda env
conda activate onnx-vai
```
:pushpin: The voe-package path used in the commands below is the path of the package downloaded earlier from the [link](http://xcoartifactory/artifactory/vai-rt-ipu-prod-local/com/amd/onnx-rt/phx/dev/603/windows/voe-4.0-win_amd64.zip) 

### Yolo-v8

```powershell
## Go to yolov8 directory
cd examples\yolov8

## Base model is available in models directory
ls models

## Create model with pre and post processing nodes
cd genmodel

## The model will be generated based on the input image shape 
python generate.py --img <image-path>

## Output model can be found in models directory
cd ..
ls models

## Test model 
cd test

:caution: **The shape of input image should be same as the image that was used for generating model**

## CPU EP
.\run.bat --e2e --img <image-path> 

# VitisAI EP
.\run.bat --e2e --vai-ep --img <image-path> --voe-path <path to voe package>

## operator level profiling on CPU EP 
.\run.bat --e2e --img <image-path> --op-profile 

# operator level profiling on VitisAI EP
.\run.bat --e2e --vai-ep --img <image-path> --op-profile --voe-path <path to voe package>

# To get the performance numbers for e2e model on VtitisAI EP
# Number of iterations can be given > 100 for generation of profiling numbers
.\run.bat --img <image-path> --vai-ep --e2e --e2e-profile --voe-path <path to voe package> --iterations <Number of iterations>


## Performance
Average latency when E2E model runs with Pre/Post Processing custom operators on CPU: 138.142 ms
Average latency when E2E model runs with Pre/Post Processing custom operators on AIE: 114.99 ms
```
Power Profiling Results for YoloV8

Attribute | Pre-process on CPU | Pre-process on Vitis AI | Difference(%)
--- | --- | --- | --- 
Avg. CPU-GPU Power (W)|3.27|2.26|-30.89
Total Energy Consumption (J)|56.8|42.44|-25.28

### ResNet-50

```powershell
## Go to resnet50 directory
cd examples\resnet50

## Base model is available in models directory
ls models

## Create model with pre and post processing nodes
cd genmodel

## The model will be generated based on the input image shape
python generate.py --img <image-path>

## Output model can be found in models directory
cd ..
ls models

## Test model
cd test

:caution: **The shape of input image should be same as the image that was used for generating model**

## CPU EP
.\run.bat --e2e --img <image-path> 

# VitisAI EP
.\run.bat --e2e --vai-ep --img <image-path> --voe-path <path to voe package>

## operator level profiling on CPU EP 
.\run.bat --e2e --img <image-path> --op-profile 

# operator level profiling on VitisAI EP
.\run.bat --e2e --vai-ep --img <image-path> --op-profile --voe-path <path to voe package>

# To get the performance numbers for e2e model on VtitisAI EP
# Number of iterations can be given > 100 for generation of profiling numbers

.\run.bat --img <image-path> --vai-ep --e2e --e2e-profile --voe-path <path to voe package> --iterations <Number of iterations>


## Performance

Average latency when E2E model runs with Pre/Post Processing custom operators on CPU: 26.6 ms
Average latency when E2E model runs with Pre/Post Processing custom operators on AIE: 28.8 ms
```
Power Profiling Results for ResNet50

Attribute | Pre-process on CPU | Pre-process on Vitis AI | Difference(%)
--- | --- | --- | --- 
Avg. CPU-GPU Power (W)|3.36|2.19|-34.82
Total Energy Consumption (J)|48.54|44.32|-8.69

## Extracting and Visualizing ONNX model info
A python utility to visualize model information like nodes, shapes, attribute of an ONNX model.
For usage, please [README](../tools/README.md)
