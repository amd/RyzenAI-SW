<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI LLM Examples </h1>
    </td>
 </tr>
</table>

# Ryzen AI LLM OGA (Onnx Runtime Generate API) Flow  

Ryzen AI Software supports deploying LLMs on Ryzen AI PCs using the native ONNX Runtime Generate (OGA) C++ or Python API. 

Refer to [OnnxRuntime GenAI (OGA)](oga_api/README.md) or https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html for more details. 

# Pre-optimized VLM

AMD provides a pre-optimized Gemma-3-4b-it multimodal model that is ready to be deployed with Ryzen AI Software. Support for this model is available starting with Ryzen AI 1.7. The model is currently offloaded on the NPU. 

The model can be found here: https://huggingface.co/amd/Gemma-3-4b-it-mm-onnx-ryzenai-npu

# Ryzen AI Installation

Instructions for installing Ryzen AI and its requirements are available on the official Ryzen AI Software documentation page: https://ryzenai.docs.amd.com/en/latest/inst.html

# Steps to Run the VLM

- Create and activate a Conda environment by cloning the existing Ryzen AI environment.
   
   ```
   conda create -n run_VLM --clone ryzen-ai-<version>`  
   conda activate run_VLM
   ```
- Install necessary package.
   ```
   pip install pillow
   ```

- Clone the RyzenAI-SW repo and navigate to the VLM folder.
   
   ```
   git clone https://github.com/amd/RyzenAI-SW.git`  
   cd path\to\RyzenAI-SW\LLM-examples\VLM
   ```

- Download the Gemma-3-4b-it multimodal model for Ryzen AI NPU.
   
   `git clone https://huggingface.co/amd/Gemma-3-4b-it-mm-onnx-ryzenai-npu`

- Execute the run_vision.py script.
   
   `python run_vision.py -m "path\to\Gemma-3-4b-it-mm-onnx-ryzenai-npu"`

[NOTE] Provide the full path to the model with the -m option.

- Sample command:
  
  `python run_vision.py -m "C:\Users\Gemma-3-4b-it-mm-onnx-ryzenai-npu"`
  
- Sample output:

```
Loading model...                                                                                                        
Model loaded
Image Path (comma separated; leave empty if no image): C:\Users\Pictures\australia.jpg
Using image: C:\Users\Pictures\australia.jpg
Prompt: Describe the image
Processing images and prompt...
Generating response...

Here's a description of the image:

**Overall Impression:**

The image is a stunning and dramatic shot of the Sydney Opera House silhouetted against a vibrant, fiery sunset. It evokes a sense of awe and the beauty of a coastal location.


*▁▁▁**Sydney Opera House:** The iconic building is the central focus, dramatically framed by the sunset. It's rendered in silhouette, emphasizing its unique architecture.                                                                      
*▁▁▁**Sunset:** The sky is ablaze with color – deep oranges, reds, and purples. There's a strong, bright sun at the horizon.                                                                                                                    
*▁▁▁**Water:** The water is calm and reflects the colors of the sky, creating a shimmering effect.
*▁▁▁**Clouds:** There are streaks of pink and purple clouds, adding to the overall dramatic feel.

**Color and Tone:**

*▁▁▁The dominant colors are warm – oranges, reds, and yellows – contrasted by the cooler purples and pinks of the clouds
.                                                                                                                       *▁▁▁The image has a high contrast, with the dark silhouette of the Opera House against the bright sky.

**Mood and Atmosphere:**

*▁▁▁The image conveys a sense of peace, wonder, and the grandeur of nature. It feels like a special moment captured.

**Technical Aspects (Based on the image):**

*▁▁▁The image is well-composed, with the Opera House positioned centrally.
*▁▁▁The photographer has used a long exposure to capture the light trails in the water.

Do you want me to focus on a specific aspect of the image, such as the colors, composition, or the feeling it evokes?
Total Time : 44.30
```


# Copyright

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
