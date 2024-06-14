# Hello World Example

This folder contains the following files:

- `hello_world.ipynb`: Jupyter Notebook demonstrating how to take a simple ML model and port it to run on the Ryzen AI Neural Processing Unit (NPU)
- `hello_world.py`: This is a Python file version of the Jupyter Notebook. It can be used instead of the Jupyter Notebook to run a simple model on the NPU.
- `README.md`: This file provides an overview of the folder's contents.
- `requirements.txt`: This file contains the necessary dependencies and packages required to run the code in this folder.
- `models\`: This folder is initially empty. After running the hello_world example, the ONNX file of the model and the quantized ONNX file will be stored in this folder.
- `vaip_config.json`: This is the default runtime configuration file. It can also be found in the `ryzen-ai-sw` installation package.

## Getting Started

Before running this example, ensure that you have followed the Ryzen AI Installation instructions found [here](https://ryzenai.docs.amd.com/en/latest/inst.html) and that you have activated the conda environment created during installation.

Install the Python dependencies:

```
pip install -r requirements.txt
```

## Running the Example

### Jupyter Notebook

1. Launch the Jupyter Notebook.
2. Ensure that you've pointed the Jupyter Notebook to the correct Python environment. To do this, in the top-right corner of the notebook, click "Select Kernel" and provide the path to the conda environment.
3. Run all the cells in the notebook.

*Note*: It is recommended to restart the Jupyter Notebook and clear any generated files after making changes to the code to ensure a clean working environment.

### Python Hello World

1. To run the `hello_world.py` script, use the following command:

```
python hello_world.py
```

