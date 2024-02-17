<table style="width:100%">
  <tr>

<th width="100%" colspan="6"><img src="https://github.com/Xilinx/Image-Collateral/blob/main/xilinx-logo.png?raw=true" width="30%"/><h1>Ryzen AI Quantization Tutorial</h1>
</th>

  </tr>
  <tr>
    <td width="17%" align="center"><a href=../README.md>1.Introduction</td>
    <td width="17%" align="center"><a href="./ONNX_README.md">2.ONNX Quantization Tutorial</a>  </td>
    <td width="16%" align="center"><a href="./PT_README.md">3. Pytorch Quantization Tutorial</a></td>
    <td width="17%" align="center">4.Tensorflow1.x quantization tutorial</a></td>
    <td width="17%" align="center"><a href="./TF2_README.md"> 5.Tensorflow2.x Quantization Tutorial<a></td>

</tr>

</table>

## 4. Tensorflow1.x Quantization Tutorial
### Introduction

This tutorials takes Resnet50 tensorflow1.x model as an example and shows how to generate quantized onnx model with Ryzen AI quantizer. Then you can run it with onnxruntime on Ryzen AI PCs. 

### Setup

The quantizer is released in the Ryzen AI tool docker. The docker runs on a Linux host. Optionally you can run it on WSL2 of your Windows PCs. 
The quantizer can run on GPU which support ROCm or CUDA as well as X86 CPU. dGPU is recommended because it runs quantization much faster than CPU.

Make sure the Docker is installed on Linux host or WSL2. Please refer to official Docker [documentation](https://docs.docker.com/engine/install) to install.

If you use X86 CPU, run 
  ```shell
  docker pull xilinx/vitis-ai-tensorflow-cpu:latest
  ```
If you use ROCm GPU, run
  ```shell
  docker pull xilinx/vitis-ai-tensorflow-rocm:latest
  ```
If you use CUDA GPU, you need to build the docker 
  ```shell
  cd docker
  ./docker_build.sh -t gpu -f tf1
  ```
and run
  ```shell
  docker pull xilinx/vitis-ai-tensroflow-gpu:latest
  ```

You can now start the Docker using the following command depending on your device:
  ```shell
  ./docker_run.sh xilinx/vitis-ai-tensorflow-<cpu|rocm|gpu>:latest
  ```

### Quick Start in Docker environment

You should be in the conda environment "vitis-ai-tensorflow", in which vai_q_tensorflow package is already installed. 
In this conda environment, python version is 3.6.11, tensorflow version is 1.15.2. 


- Download pre-trained [Resnet50 model]from model zoo(https://www.xilinx.com/bin/public/openDownload?filename=tf_mlperf_resnet50_imagenet_224_224_8.19G_3.0.zip)
  ```shell
  wget https://www.xilinx.com/bin/public/openDownload?filename=tf_mlperf_resnet50_imagenet_224_224_8.19G_3.0.zip -O resnet50.zip
  ```

- To unzip the package, execute the following command:
    ```shell
    unzip resnet50.zip
    ```

- After extracting the package, you will find the following file structure:
    ```shell
    ├── resnet50.zip
    └── tf_mlperf_resnet50_imagenet_224_224_8.19G_3.0
        ├── code
        ├── data
        ├── float
        ├── quantized
        ├── README.md
        ├── requirements.txt
        └── run_eval_pb.sh
    ```
The resnet50.zip file is the package you need to unzip. Once extracted, it will create a directory called tf_mlperf_resnet50_imagenet_224_224_8.19G_3.0. This directory contains the following components:

- code: Contains the code files to quantize and evaluate resnet50 model.
- data: Contains the data files used by the resnet50 package.
- float: Contains the float resnet50 model file for quantize.
- quantized: default folder to store the output quantized model .
- README.md: Provides information and instructions on how to use the resnet50 package.
- requirements.txt: Lists the required dependencies for seting up the environment.
- run_eval_pb.sh: A shell script used for evaluating the performance of the resnet50 .

1.Prepare datset.
  
  ImageNet dataset link: [ImageNet](http://image-net.org/download-images) 
  
  ```
  a.Users need to download the ImageNet dataset by themselves as it needs registration. The script of get_dataset.sh can not automatically download the dataset. 
  b.The downloaded ImageNet dataset needs to be organized as the following format:
    1) Create a folder of "data" along the "code" folder. Put the validation data in +data/validation.
    2) Data from each class for validation set needs to be put in the folder:
    +data/validation
         +validation/n01847000 
         +validation/n02277742
         +validation/n02808304
         +... 
  ```



#### Preparing the Float Model and Related Input Files

Before running vai_q_tensorflow, prepare the frozen inference TensorFlow model in floating-point format and calibration set, including the files listed in the following table.

**Input Files for vai_q_tensorflow**

No. | Name | Description
--- | --- | ---
1 | frozen_graph.pb | Floating-point frozen inference graph. Ensure that the graph is the inference graph rather than the training graph.
2 | calibration dataset | A subset of the training dataset containing 100 to 1000 images.
3 | input_fn | An input function to convert the calibration dataset to the input data of the frozen_graph during quantize calibration. Usually performs data pre-processing and augmentation.

### Generating the Frozen Inference Graph
Enter the quantized script and run the command below to start your quantization.

    ```shell
    $ cd tf_mlperf_resnet50_imagenet_224_224_8.19G_3.0/code/quantize/
    $ source quantize.sh

    ```
The quantized model will be generated and store under the path: `tf_mlperf_resnet50_imagenet_224_224_8.19G_3.0/quantized/quantize_eval_model.pb` once the freeze process complete.

When training a model with TensorFlow 1.x, a folder is created containing a GraphDef file (usually with a .pb or .pbtxt extension) and a set of checkpoint files. For mobile or embedded deployment, you need a single GraphDef file that has been "frozen." Freezing a graph means converting its variables into inline constants, so everything is contained in one file. TensorFlow provides `freeze_graph.py` for this conversion, which is automatically installed with the `vai_q_tensorflow` quantizer.

Here's an example of command-line usage:

```shell
$ freeze_graph \
    --input_graph /tmp/inception_v1_inf_graph.pb \
    --input_checkpoint /tmp/checkpoints/model.ckpt-1000 \
    --input_binary true \
    --output_graph /tmp/frozen_graph.pb \
    --output_node_names InceptionV1/Predictions/Reshape_1
```
The `--input_graph` should be an inference graph other than the training graph. Because the operations of data preprocessing and loss functions are not needed for inference and deployment, the frozen_graph.pb should only include the main part of the model. In particular, the data preprocessing operations should be taken in the Input_fn to generate correct input data for quantize calibration.

### Export the ONNX model
The quantized model is tensorflow protobuf format by default. If you want to get a ONNX format model, just add output_format argument to the vai_q_tensorflow command.

```shell
$vai_q_tensorflow quantize \
--input_frozen_graph frozen_graph.pb \
--input_nodes ${input_nodes} \
--input_shapes ${input_shapes} \
--output_nodes ${output_nodes} \
--input_fn input_fn \
--output_format onnx \
[options]
```

- output_format
    Indicates what format to save the quantized model, 'pb' for saving tensorflow frozen pb, 'onnx' for saving onnx model. The default value is 'pb'.

### (Optional) Dumping the Simulation Results
vai_q_tensorflow dumps the simulation results with the quantize_eval_model.pb generated by the quantizer. This allows you to compare the simulation results on the CPU/GPU with the output values on the DPU.

To dump the quantize simulation results, run the following commands:

```shell
$vai_q_tensorflow dump \
	--input_frozen_graph  quantize_results/quantize_eval_model.pb \
	--input_fn  dump_input_fn \
	--max_dump_batches 1 \
	--dump_float 0 \
	--output_dir quantize_results
```

In this example we can resue the quantize.sh script to do dump only need to change some configuation. Modify the line 48~53 to use the dump command. In the `config.ini` file, the FLOAT_MODEL variable is being modified. Before the change, the variable was set to `../../float/resnet50.pb`. After the change, it is updated to `../../quantized/quantize_eval_model.pb`.

```shell
 48 vai_q_tensorflow dump \
 49   --input_frozen_graph $FLOAT_MODEL \
 50   --input_fn $CALIB_INPUT_FN \
 51   --max_dump_batches 1 \
 52   --dump_float 0 \
 53   --output_dir $QUANTIZE_DIR \

```
config.ini before 
```shell
export FLOAT_MODEL=../../float/resnet50.pb
```
config.ini after
```shell
export FLOAT_MODEL= export FLOAT_MODEL=../../quantized/quantize_eval_model.pb
```
After the two files are modified, to dump the result run the script as below

    ```shell
    $ cd tf_mlperf_resnet50_imagenet_224_224_8.19G_3.0/code/quantize/
    $ source quantize.sh

    ```



#### Handling Training-Specific Operations during Graph Freezing

During the training and inference phases, certain operations, such as dropout and batch normalization, behave differently. To ensure correct behavior when freezing the graph, consider the following:

- Make sure that operations like dropout and batch normalization are in the inference phase when freezing the graph. For instance, when using `tf.layers.dropout` or `tf.layers.batch_normalization`, set the `is_training` flag to `false` to ensure they behave appropriately during inference.

- If you are working with models using `tf.keras`, call `tf.keras.backend.set_learning_phase(0)` before building the graph. This action sets the learning phase to 0, indicating the inference phase, and deactivates any training-specific behaviors during graph freezing.


The estimated input and output nodes cannot be used for quantization if the graph has in-graph pre- and post-processing. This is because some operations are not quantizable and can cause errors when compiled by the Vitis AI compiler if you deploy the quantized model to the DPU.

Another way to get the input and output names of the graph is by visualizing the graph. Both TensorBoard and Netron can do this. See the following example, which uses Netron:


:arrow_forward:**Next Topic:**  [5. Tensorflow2.x Quantization Tutorial](./TF2_README.md)

:arrow_backward:**Previous Topic:**  [3. Pytorch Quantization Tutorial](./PT_README.md)
<hr/>