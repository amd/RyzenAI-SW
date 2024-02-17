# Create & Run E2E ONNX Models

Examples to create and run ONNX Models with pre/post processing as custom operators.

## Enabling inference of end-to-end models on the IPU

### Using existing pre/post processing OPs from vitis_customop: 

The vitis_customop library supports some common tasks such as resizing, normalization, NMS, etc. To make use of these, the user will need to specify: 
   - pre/postprocessing operations
   - operation-specific parameters 

We will continue to expand the library with more supported operations. The following steps describe how to use the pre and post processor APIs. 

#### Step 1:

Create PreProcessor and PostProcessor instances:

```
from vitis_customop.preprocess import generic_preprocess as pre
input_node_name = "blob.1"
preprocessor = pre.PreProcessor(input_model_path, output_model_path, input_node_name)
output_node_name = "1327"
postprocessor = post.PostProcessor(onnx_pre_model_name, onnx_e2e_model_name, output_node_name)
```

#### Step 2:

Specify the operations to perform, and pass required parameters. 

```
preprocessor.resize(resize_shape)
preprocessor.normalize(mean, std_dev, scale)
preprocessor.set_resnet_params(mean, std_dev, scale)
postprocessor.ResNetPostProcess()
```

#### Step 3:

Generate and save new model

```
preprocessor.build()
postprocessor.build()
```

## Examples

Activate conda env (onnx-vai), if not done already.
```powershell
## This is required if the VAI-EP packages are installed in conda env
## Skip this if you do not have VAI EP installed in conda env
conda activate onnx-vai
```
:pushpin: The voe-package path used in the commands below is the path of the package downloaded earlier from the [link](http://xcoartifactory/artifactory/vai-rt-ipu-prod-local/com/amd/onnx-rt/phx/dev/603/windows/voe-4.0-win_amd64.zip) 

We will use ```resnet50\genmodel\user_generate.py``` and  ```yolov8\genmodel\user_generate.py``` to generate the models with pre or/and post-processing subgraphs. The two scripts demonstrate the use of the APIs mentioned in the previous section. The user can provide the following preprocessing parameters as command-line arguments: 

1. resize_shape: Shape to resize to as a list of 3 integers in HWC format
2. mean: Mean for normalization as a list of 3 floats in HWC format
3. std_dev: Standard deviation for normalization as a list of 3 floats in HWC format
4. scale: Values used to scale the input prior to normalization as a list of 3 floats in HWC format

Default values have been provided for these parameters. 

### ResNet-50

```powershell
## Go to resnet50 directory
cd examples\resnet50

## Base model is available in models directory
ls models

## Create model with pre and post processing nodes
cd genmodel

## The model with preprocessing operations incorporated will be generated based on given parameters
python user_generate.py --img ..\..\test_resnet.jpg --resize_shape 224 224 4 --mean 51.97 58.39 61.84 --std_dev 41.65 41.65 41.65 --scale 1.0 1.0 1.0

## The model with both preprocessing and postprocessing operations incorporated will be generated 
python user_generate.py --img ..\..\test_resnet.jpg --e2e --resize_shape 224 224 4 --mean 51.97 58.39 61.84 --std_dev 41.65 41.65 41.65 --scale 1.0 1.0 1.0

## Output model can be found in models directory
cd ..
ls models

## Test model
cd test

:caution: **The shape of input image should be same as the image that was used for generating model**

## CPU EP
.\run.bat --e2e --img ..\..\test_resnet.jpg

# Run the model with preprocessor on VitisAI EP
.\run.bat --with-pre --vai-ep --img ..\..\test_resnet.jpg --voe-path <path to voe package>

## operator level profiling on CPU EP 
.\run.bat --e2e --img ..\..\test_resnet.jpg --op-profile 

# operator level profiling on VitisAI EP
.\run.bat --with-pre --vai-ep --img ..\..\test_resnet.jpg --op-profile --voe-path <path to voe package>

# To get the performance numbers for e2e model on VtitisAI EP
# Number of iterations can be given > 100 for generation of profiling numbers

.\run.bat --img ..\..\test_resnet.jpg --vai-ep --with-pre --e2e-profile --voe-path <path to voe package> --iterations <Number of iterations>


## Performance

Average latency when E2E model runs with Pre/Post Processing custom operators on CPU: 26.6 ms
Average latency when E2E model runs with Pre/Post Processing custom operators on AIE: 28.8 ms
```
Power Profiling Results for ResNet50

Attribute | Pre-process on CPU | Pre-process on Vitis AI | Difference(%)
--- | --- | --- | --- 
Avg. CPU-GPU Power (W)|3.36|2.19|-34.82
Total Energy Consumption (J)|48.54|44.32|-8.69

### Yolo-v8

```powershell
## Go to yolov8 directory
cd examples\yolov8

## Base model is available in models directory
ls models

## Create model with pre and post processing nodes
cd genmodel

## The model with preprocessing operations incorporated will be generated based on default parameters. These can also be specified as shown in the previous example
python user_generate.py --img ..\..\test.jpg

## Output model can be found in models directory
cd ..
ls models

## Test model 
cd test

:caution: **The shape of input image should be same as the image that was used for generating model**

## CPU EP
.\run.bat --e2e --img ..\..\test.jpg

# VitisAI EP
.\run.bat --e2e --vai-ep --img ..\..\test.jpg --voe-path <path to voe package>

## operator level profiling on CPU EP 
.\run.bat --e2e --img ..\..\test.jpg --op-profile 

# operator level profiling on VitisAI EP
.\run.bat --e2e --vai-ep --img ..\..\test.jpg --op-profile --voe-path <path to voe package>

# To get the performance numbers for e2e model on VtitisAI EP
# Number of iterations can be given > 100 for generation of profiling numbers
.\run.bat --img ..\..\test.jpg --vai-ep --e2e --e2e-profile --voe-path <path to voe package> --iterations <Number of iterations>


## Performance
Average latency when E2E model runs with Pre/Post Processing custom operators on CPU: 138.142 ms
Average latency when E2E model runs with Pre/Post Processing custom operators on AIE: 114.99 ms
```
Power Profiling Results for YoloV8

Attribute | Pre-process on CPU | Pre-process on Vitis AI | Difference(%)
--- | --- | --- | --- 
Avg. CPU-GPU Power (W)|3.27|2.26|-30.89
Total Energy Consumption (J)|56.8|42.44|-25.28


## Extracting and Visualizing ONNX model info
A python utility to visualize model information like nodes, shapes, attribute of an ONNX model.
For usage, please [README](../tools/README.md)
