# AIG - Internal

The repository is from: https://github.com/kjsman/stable-diffusion-pytorch 

The internal version of this repository is being modified to evaluate the model.

# stable-diffusion-pytorch

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kjsman/stable-diffusion-pytorch/blob/main/demo.ipynb)

Yet another PyTorch implementation of [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release).

I tried my best to make the codebase minimal, self-contained, consistent, hackable, and easy to read. Features are pruned if not needed in Stable Diffusion (e.g. Attention mask at CLIP tokenizer/encoder). Configs are hard-coded (based on Stable Diffusion v1.x). Loops are unrolled when that shape makes more sense.

Despite of my efforts, I feel like [I cooked another sphagetti](https://xkcd.com/927/). Well, help yourself!

Heavily referred to following repositories. Big kudos to them!

* [divamgupta/stable-diffusion-tensorflow](https://github.com/divamgupta/stable-diffusion-tensorflow)
* [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
* [huggingface/transformers](https://github.com/huggingface/transformers)
* [crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion)
* [karpathy/minGPT](https://github.com/karpathy/minGPT)

## Dependencies

* PyTorch
* Numpy
* Pillow
* regex
* tqdm

## How to Install

1. Clone or download this repository.
2. Install dependencies: Run `pip install torch numpy Pillow regex` or `pip install -r requirements.txt`.
3. Download `data.v20221029.tar` from [here](https://huggingface.co/jinseokim/stable-diffusion-pytorch-data/resolve/main/data.v20221029.tar) and unpack in the parent folder of `stable_diffusion_pytorch`. Your folders should be like this:
```
stable-diffusion-pytorch(-main)/
├─ data/
│  ├─ ckpt/
│  ├─ ...
├─ stable_diffusion_pytorch/
│  ├─ samplers/
└  ┴─ ...
```
*Note that checkpoint files included in `data.zip` [have different license](#license) -- you should agree to the license to use checkpoint files.*

## How to Use

Import `stable_diffusion_pytorch` as submodule.

Here's some example scripts. You can also read the docstring of `stable_diffusion_pytorch.pipeline.generate`.

Text-to-image generation:
```py
from stable_diffusion_pytorch import pipeline

prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts)
images[0].save('output.jpg')
```

...with multiple prompts:
```
prompts = [
    "a photograph of an astronaut riding a horse",
    ""]
images = pipeline.generate(prompts)
```

...with unconditional(negative) prompts:
```py
prompts = ["a photograph of an astronaut riding a horse"]
uncond_prompts = ["low quality"]
images = pipeline.generate(prompts, uncond_prompts)
```

...with seed:
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, uncond_prompts, seed=42)
```

Preload models (you will need enough VRAM):
```py
from stable_diffusion_pytorch import model_loader
models = model_loader.preload_models('cuda')

prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, models=models)
```

If you get OOM with above code but have enough RAM (not VRAM), you can move models to GPU when needed
and move back to CPU when not needed:
```py
from stable_diffusion_pytorch import model_loader
models = model_loader.preload_models('cpu')

prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, models=models, device='cuda', idle_device='cpu')
```

Image-to-image generation:
```py
from PIL import Image

prompts = ["a photograph of an astronaut riding a horse"]
input_images = [Image.open('space.jpg')]
images = pipeline.generate(prompts, input_images=images)
```

...with custom strength:
```py
prompts = ["a photograph of an astronaut riding a horse"]
input_images = [Image.open('space.jpg')]
images = pipeline.generate(prompts, input_images=images, strength=0.6)
```

Change [classifier-free guidance](https://arxiv.org/abs/2207.12598) scale:
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, cfg_scale=11)
```

...or disable classifier-free guidance:
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, do_cfg=False)
```

Reduce steps (faster generation, lower quality):
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, n_inference_steps=28)
```

Use different sampler:
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, sampler="k_euler")
# "k_lms" (default), "k_euler", or "k_euler_ancestral" is available
```

Generate image with custom size:
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, height=512, width=768)
```

## LICENSE

All codes on this repository are licensed with MIT License. Please see LICENSE file.

Note that checkpoint files of Stable Diffusion are licensed with [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license) License. It has use-based restriction caluse, so you'd better read it.
