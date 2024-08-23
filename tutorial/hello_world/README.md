# Hello World Example

## Introduction
This example demonstrates how to port a simple machine learning model to run on the AMD Ryzen AI Neural Processing Unit (NPU). Leveraging the NPU for inference can accelerate performance while offloading work from the CPU and GPU.

In this "Hello World" example, we'll walk through the process of converting a basic ML model to ONNX format, quantizing it, and running the inference on the Ryzen AI NPU. This is a great place to get started on learning the development process with Ryzen AI Software.

This folder contains the following files:

- `hello_world.ipynb`: Jupyter Notebook demonstrating how to take a simple ML model and port it to run on the Ryzen AI Neural Processing Unit (NPU)
- `hello_world.py`: This is a Python file version of the Jupyter Notebook. It can be used instead of the Jupyter Notebook to run a simple model on the NPU.
- `README.md`: This file provides an overview of the folder's contents.
- `requirements.txt`: This file contains the necessary dependencies and packages required to run the code in this folder.
- `models\`: This folder is initially empty. After running the hello_world example, the ONNX file of the model and the quantized ONNX file will be stored in this folder.

## Getting Started

Before running this example, ensure that you have followed the Ryzen AI Installation instructions found [here](https://ryzenai.docs.amd.com/en/latest/inst.html) and have activated the conda environment created during installation.

Install the Python dependencies:

```
pip install -r requirements.txt
```

## Running the Example

### Jupyter Notebook

There are two ways to open the Jupyter Notebook:

1. **Using an IDE (e.g., VS Code)**:
   - Open the notebook file (`hello_world.ipynb`) in VS Code.
   - VS Code will automatically set up the Jupyter server.
   - Ensure the correct kernel is selected by clicking "Select Kernel" in the top-right corner and choosing the appropriate conda environment.

   *Note*: It's recommended to restart the Jupyter Notebook and clear any generated files after modifying the code to maintain a clean environment.

2. **Using the Command Line**:
   - Follow these steps to launch Jupyter Notebook from the command prompt:

     1. **Ensure the correct environment is selected** by running:
        ```bash
        python -m ipykernel install --user --name <your-env-name> --display-name "Python (<your-display-name>)"
        ```
        _Replace `<your-env-name>` with the actual conda environment name._

     2. **Launch Jupyter Notebook**:
        ```bash
        jupyter notebook
        ```
        This will open a new browser window or tab with the Jupyter interface.

     3. **Open the `hello_world.ipynb` file** from the Jupyter interface.

     4. **Select the correct kernel**:
        - In the top-right corner, click "Kernel" → "Change Kernel" (or "Select Kernel" depending on the version).
        - Choose the conda environment where the necessary dependencies are installed.

     5. **Run all cells**:
        - Click the "Run" button for each cell or go to "Kernel" in the menu and select "Restart & Run All".

   *Note*: If you encounter any errors after modifying the code, it's recommended to restart the Jupyter Notebook and clear any variables or outputs to maintain a clean environment.

### Python Hello World

To run the `hello_world.py` script, use the following command in your terminal:

```bash
python hello_world.py
```