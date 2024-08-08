# LLMs on RyzenAI with llama.cpp

[llama-2-7b-chat-alias]: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf
[llama-3-8b-instruct-alias]: https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf
[qwen1_5-7b-chat-alias]: https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF/blob/main/qwen1_5-7b-chat-q4_0.gguf
[Phi-3-mini-4k-instruct-alias]: https://huggingface.co/SanctumAI/Phi-3-mini-4k-instruct-GGUF?show_file_info=phi-3-mini-4k-instruct.Q4_0.gguf 



## Steps to run the models
Assumes Windows CMD shell

### Activate ryzenai-transformers conda-enviornment
```console
cd <transformers>
set TRANSFORMERS_ROOT=%CD%
conda env create --file=env.yaml
conda activate ryzenai-transformers
```

⚠️ **Warning:** Windows has a path length limit that you may hit when building the project or installing the wheels, resulting in cryptic errors.
To work around it, use a virtual drive to shorten the path the repository is cloned to:

*On Command Prompt*
```batch
@REM use any unused drive letter, Z: for example
subst Z: %cd%
@REM switch to the Z: drive
Z:
```

You can remove the virtual drive with:

*On Command Prompt*
```batch
subst /d Z:
```

### Build and Install RyzenAI
```console
setup_phx.bat # or setup_stx.bat

cd %TRANSFORMERS_ROOT%\ops\cpp
cmake -B build\ -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX%
cmake --build build\ --config=Release
cmake --install build\ --config=Release
```

### Build llama.cpp
```console
cd %TRANSFORMERS_ROOT%\ext\llama.cpp
cmake -B build\ -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%;%XRT_PATH%" -DLLAMA_RYZENAI=ON
cmake --build build\ --config=Release
```
Note: To switch between CPU/NPU recompile with compilation flag LLAMA_RYZENAI=OFF/ON

### Download desired model
Download the desired prequantized gguf model from huggingface.
Note: Must be Q4_0 quantized for offload to NPU
Example model: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf
Download the model to:
`%TRANSFORMERS_ROOT%\ext\llama.cpp\models`

### Run
```console
cd %TRANSFORMERS_ROOT%\ext\llama.cpp\build\
bin\Release\main.exe -m ..\models\llama-2-7b-chat.Q4_0.gguf -e -t 1 -n 400 -p "Building a website can be done in 10 simple steps:\nStep 1:"
```

## Example of running Llama3
Command:
```
bin\Release\main.exe -m ..\models\Meta-Llama-3-8b-Instruct.Q4_0.gguf -e -t 4 -n 400 -p "Building a website can be done in 10 simple steps:\nStep 1:"
```
Prompt:
```
Building a website can be done in 10 simple steps:\n
```
Output:
```
Step 1: Choose a Domain Name
Step 2: Select a Web Hosting Service
Step 3: Design Your Website
Step 4: Create the Content
Step 5: Build the Website
Step 6: Add Interactive Elements
Step 7: Optimize for Search Engines
Step 8: Test and Debug
Step 9: Launch and Promote
Step 10: Maintain and Update

Here's a more detailed overview of each step:

Step 1: Choose a Domain Name
Choose a unique and memorable domain name that reflects your brand or website's purpose. Make sure it's available and easy to spell.

Step 2: Select a Web Hosting Service
Choose a reliable web hosting service that meets your needs and budget. Consider factors like storage space, bandwidth, and customer support.

Step 3: Design Your Website
Create a design concept for your website that includes a layout, color scheme, and typography. You can hire a designer or use a website builder like Wix or Squarespace.

Step 4: Create the Content
Write and gather the content for your website, including text, images, and other media. Make sure it's engaging, informative, and easy to read.

Step 5: Build the Website
Use a website builder or coding skills to build the website structure, including the homepage, interior pages, and navigation. Make sure it's responsive and works on different devices.

Step 6: Add Interactive Elements
Add interactive elements like forms, buttons, and menus to enhance user engagement and conversion rates.

Step 7: Optimize for Search Engines
Optimize your website for search engines by using relevant keywords, meta tags, and header tags. This will improve your website's visibility and ranking.

Step 8: Test and Debug
Test your website for usability, functionality, and performance. Fix any bugs or issues that arise during testing.

Step 9: Launch and Promote
Launch your website and promote it through social media,
```

Details about the quantization scheme "Q4_0" implementation on RyzenAI NPU is described in [quantization](./quantization.md).

For technical discussion of the RyzenAI backend in llama.cpp see [background](./background.md).

## Perplexity measurement (Model quality)
The perplexity example within Llama.cpp is used to measure perplexity over a given prompt (lower perplexity is better).

The perplexity measurements in table above are done against the wikitext2 test dataset (https://paperswithcode.com/dataset/wikitext-2), with context length of 512.

```console
cd %TRANSFORMERS_ROOT%\ext\llama.cpp\build\
bin\Release\main.exe -m ..\models\llama-2-7b-chat.Q4_0.gguf -f wikitext-2-raw\wiki.test.raw
```
Output:
```console
perplexity : calculating perplexity over 655 chunks
24.43 seconds per pass - ETA 4.45 hours
[1]4.3306,[2]4.8324,[3]5.4543,[4]6.0606 ...
```

For more details of perplexity measurement on Llama.cpp refer to [perplexity](../../../ext/llama.cpp/README.md)

