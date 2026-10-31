<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Text Classification </h1>
    </td>
  </tr>
</table>

# Running DistilBERT on Ryzen AI

This Ryzen AI example demonstrates how to run the ``distilbert-base-uncased-finetuned-sst-2-english`` model on an NPU. This model is a fine-tuned checkpoint of **DistilBERT-base-uncased**, trained on the **SST-2** dataset for sentiment analysis. Developed by Hugging Face, it can be used for both single-label and multi-label classification.

For more details, refer to the Hugging Face Model Card: [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

## Setup Instructions

Set Up Python Virtual Environment for DistilBERT

Step 1: Activate the Ryzen AI installation environment
```bat
"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat"
set "RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\ryzen-ai-1.9.0"
```

Step 2: Export the installed Python packages
```bat
python -m pip freeze > "%TEMP%\ryzenai_1.9_requirements.txt"
```
This creates a requirements file containing the packages installed in the Ryzen AI environment.

Step 3: Create a virtual environment for DistilBERT
```bat
cd /d "<RyzenAI-SW>\Transformer-examples\DistilBERT_text_classification_bf16"
python -m venv bert
```

Step 4: Deactivate the Ryzen AI installation environment
```bat
deactivate
```

Step 5: Activate the DistilBERT environment
```bat
"bert\Scripts\activate.bat"
```

Step 6: Install the Ryzen AI Python dependencies
```bat
python -m pip install -r "%TEMP%\ryzenai_1.9_requirements.txt"
```

Step 7: Install the example-specific dependencies
```bat
cd "<RyzenAI-SW>\Transformer-examples\DistilBERT_text_classification_bf16"
python -m pip install -r requirements.txt
```
The bert virtual environment is now configured for running the DistilBERT text classification BF16 example.


## Deployment Steps

The following steps outline how to deploy the model on an NPU:

- Download the model from Hugging Face and convert it to ONNX (Opset 17).
- Quantize the model to BF16 using the AMD Quark Quantizer.
- Compile and run the model on an NPU using ONNX Runtime with the Vitis AI Execution Provider.

Details for each step are provided below.

### 1. ONNX Conversion

To download the model and convert it to ONNX, run the following script:

```
python pt_to_onnx.py --output_dir model
```

The script ``pt_to_onnx.py`` downloads the model from the Hugging Face checkpoint and converts it to ONNX format with Opset 17 using the ``torch.onnx.export`` API.

After running this command, the converted model will be saved as: ``model/distilbert-base-uncased-finetuned-sst-2-english.onnx``


### 2. Running Inference

To run inference, execute the following script:

```
python run_inference.py
```

The ONNX Runtime Vitis AI Execution Provider compiles and runs the model on an NPU. A BF16-specific configuration file is passed by the ``config_file`` provider option.

The first-time compilation may take some time, but the compiled model is cached in a directory specified by the ``cache_dir`` and ``cache_key`` provider options.
Subsequent runs bypass the compilation process and directly execute the model on the NPU.

After a successful run, the model will be deployed on the NPU and provide sentiment classification for input texts used in the script

```
**********
Prompt: Hello, my dog is cute
Text classification: POSITIVE
**********
Prompt: Stop talking to me
Text classification: NEGATIVE
**********
Prompt: That painting is ugly
Text classification: NEGATIVE
**********
Prompt: Life is beautiful
Text classification: POSITIVE
**********
```
