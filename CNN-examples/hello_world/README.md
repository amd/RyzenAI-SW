<table class="sphinxhide" width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Tutorial </h1></td>
  </tr>
</table>

## Introduction

This example demonstrates how to port a simple machine learning model to run on the AMD Ryzen AI Neural Processing Unit (NPU). Using the NPU for inference can improve performance while offloading work from the CPU and GPU.

In this **Hello World** example, you will convert a basic ML model to ONNX format, quantize it, and run inference on the Ryzen AI NPU.

This folder contains the following files:

- `hello_world.ipynb`: Jupyter Notebook showing how to run a simple ML model on the Ryzen AI NPU.
- `hello_world.py`: Python script alternative to the Jupyter Notebook.
- `README.md`: Overview and instructions for this example.
- `requirements.txt`: Python dependencies for this example.
- `models\`: Initially empty; generated ONNX and quantized ONNX model files are saved here after running the example.

## Getting Started

Install Ryzen AI using the [Ryzen AI installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html). The MSI installer creates a Python virtual environment at `C:\Program Files\RyzenAI\ryzen-ai-1.9.0`. The instructions below use Ryzen AI **1.9.0** and Windows Command Prompt (`cmd.exe`). Update the path if your installed version differs.

Open **x64 Native Tools Command Prompt for VS 2022**, **Command Prompt**, or **Miniforge Prompt**.

### Create and Activate Python Virtual Environment

**Step 1: Activate the Ryzen AI installation environment**

```bat
"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat"
set "RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\ryzen-ai-1.9.0"
```

**Step 2: Export the installed Python packages**

```bat
python -m pip freeze > "%TEMP%\ryzenai_1.9_requirements.txt"
```

This creates a requirements file containing the Python packages installed in the Ryzen AI environment.

**Step 3: Create a virtual environment for Hello World**

```bat
cd /d <RyzenAI-SW>\CNN-examples\hello_world
python -m venv hello_world_env
```

**Step 4: Deactivate the Ryzen AI installation environment**

```bat
deactivate
```

**Step 5: Activate the Hello World environment**

```bat
"hello_world_env\Scripts\activate.bat"
```

**Step 6: Install the Ryzen AI Python dependencies**

```bat
python -m pip install -r "%TEMP%\ryzenai_1.9_requirements.txt"
```

**Step 7: Install the example-specific dependencies**

```bat
cd <RyzenAI-SW>\CNN-examples\hello_world
python -m pip install -r requirements.txt
```

The `hello_world_env` virtual environment is now configured for running the Hello World example.

> **Note:** Writing files and creating environments under `C:\Program Files` may require administrator privileges. Reinstalling packages from `pip freeze` is not an exact clone of the MSI environment; native runtime dependencies and package compatibility should be verified.

## Running the Example

### Jupyter Notebook

You can open the Jupyter Notebook in either of the following ways.

#### Option 1: Using an IDE (e.g., VS Code)

1. Open `hello_world.ipynb` in VS Code.
2. Click **Select Kernel** in the top-right corner.
3. Select the interpreter at `<RyzenAI-SW>\CNN-examples\hello_world\hello_world_env\Scripts\python.exe`, or choose the registered **Python (hello_world_env)** kernel.
4. Run all cells in the notebook.

It can help to restart the notebook and clear generated outputs after modifying the code.

#### Option 2: Using the Command Line

**Step 1: Activate the Hello World virtual environment**

```bat
cd /d <RyzenAI-SW>\CNN-examples\hello_world
"hello_world_env\Scripts\activate.bat"
```

**Step 2: Install Jupyter and register the kernel (if needed)**

```bat
python -m pip install notebook ipykernel
python -m ipykernel install --user --name hello_world_env --display-name "Python (hello_world_env)"
```

**Step 3: Launch Jupyter Notebook**

```bat
python -m notebook
```

**Step 4: Open `hello_world.ipynb`** in the Jupyter interface.

**Step 5: Select the kernel** named **Python (hello_world_env)** from the kernel selector.

**Step 6: Run all cells** using the notebook's **Run** controls or **Restart & Run All**.

> **Note:** If the kernel is not visible, restart Jupyter Notebook after registration. Restart the kernel and clear outputs if you encounter stale state after changing the code.

### Python Hello World

Activate the Hello World virtual environment and navigate to the example directory (if not already done):

```bat
cd /d <RyzenAI-SW>\CNN-examples\hello_world
"hello_world_env\Scripts\activate.bat"
set "RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\ryzen-ai-1.9.0"
```

Run the Python script:

```bat
python hello_world.py
```
