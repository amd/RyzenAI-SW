import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from . import Tokenizer
from . import KLMSSampler, KEulerSampler, KEulerAncestralSampler
from . import util
from . import model_loader


def generate(
        prompts,
        uncond_prompts=None,
        input_images=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        height=512,
        width=512,
        sampler="k_lms",
        n_inference_steps=50,
        models={},
        seed=None,
        device=None,
        idle_device=None
):
    r"""
    Function invoked when calling the pipeline for generation.
    Args:
        prompts (`List[str]`):
            The prompts to guide the image generation.
        uncond_prompts (`List[str]`, *optional*, defaults to `[""] * len(prompts)`):
            The prompts not to guide the image generation. Ignored when not using guidance (i.e. ignored if
            `do_cfg` is False).
        input_images (List[Union[`PIL.Image.Image`, str]]):
            Images which are served as the starting point for the image generation.
        strength (`float`, *optional*, defaults to 0.8):
            Conceptually, indicates how much to transform the reference `input_images`. Must be between 0 and 1.
            `input_images` will be used as a starting point, adding more noise to it the larger the `strength`.
            The number of denoising steps depends on the amount of noise initially added. When `strength` is 1,
            added noise will be maximum and the denoising process will run for the full number of iterations
            specified in `n_inference_steps`. A value of 1, therefore, essentially ignores `input_images`.
        do_cfg (`bool`, *optional*, defaults to True):
            Enable [classifier-free guidance](https://arxiv.org/abs/2207.12598).
        cfg_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale of classifier-free guidance. Ignored when it is disabled (i.e. ignored if
            `do_cfg` is False). Higher guidance scale encourages to generate images that are closely linked
            to the text `prompt`, usually at the expense of lower image quality.
        height (`int`, *optional*, defaults to 512):
            The height in pixels of the generated image. Ignored when `input_images` are provided.
        width (`int`, *optional*, defaults to 512):
            The width in pixels of the generated image. Ignored when `input_images` are provided.
        sampler (`str`, *optional*, defaults to "k_lms"):
            A sampler to be used to denoise the encoded image latents. Can be one of `"k_lms"`, `"k_euler"`,
            or `"k_euler_ancestral"`.
        n_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference. This parameter will be modulated by `strength`.
        models (`Dict[str, torch.nn.Module]`, *optional*):
            Preloaded models. If some or all models are not provided, they will be loaded dynamically.
        seed (`int`, *optional*):
            A seed to make generation deterministic.
        device (`str` or `torch.device`, *optional*):
            PyTorch device which the image generation happens. If not provided, 'cuda' or 'cpu' will be used.
        idle_device (`str` or `torch.device`, *optional*):
            PyTorch device which the models no longer in use are moved to.
    Returns:
        `List[PIL.Image.Image]`:
            The generated images.
    Note:
        This docstring is heavily copied from huggingface/diffusers.
    """
    with torch.no_grad():
        if not isinstance(prompts, (list, tuple)) or not prompts:
            raise ValueError("prompts must be a non-empty list or tuple")

        if uncond_prompts and not isinstance(uncond_prompts, (list, tuple)):
            raise ValueError("uncond_prompts must be a non-empty list or tuple if provided")
        if uncond_prompts and len(prompts) != len(uncond_prompts):
            raise ValueError("length of uncond_prompts must be same as length of prompts")
        uncond_prompts = uncond_prompts or [""] * len(prompts)

        if input_images and not isinstance(uncond_prompts, (list, tuple)):
            raise ValueError("input_images must be a non-empty list or tuple if provided")
        if input_images and len(prompts) != len(input_images):
            raise ValueError("length of input_images must be same as length of prompts")
        if not 0 < strength < 1:
            raise ValueError("strength must be between 0 and 1")

        if height % 8 or width % 8:
            raise ValueError("height and width must be a multiple of 8")

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        tokenizer = Tokenizer()
        clip = models.get('clip') or model_loader.load_clip(device)
        clip.to(device)
        if do_cfg:
            cond_tokens = tokenizer.encode_batch(prompts)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)
            uncond_tokens = tokenizer.encode_batch(uncond_prompts)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.encode_batch(prompts)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        to_idle(clip)
        del tokenizer, clip

        if sampler == "k_lms":
            sampler = KLMSSampler(n_inference_steps=n_inference_steps)
        elif sampler == "k_euler":
            sampler = KEulerSampler(n_inference_steps=n_inference_steps)
        elif sampler == "k_euler_ancestral":
            sampler = KEulerAncestralSampler(n_inference_steps=n_inference_steps,
                                             generator=generator)
        else:
            raise ValueError(
                "Unknown sampler value %s. "
                "Accepted values are {k_lms, k_euler, k_euler_ancestral}"
                % sampler
            )

        noise_shape = (len(prompts), 4, height // 8, width // 8)

        if input_images:
            encoder = models.get('encoder') or model_loader.load_encoder(device)
            encoder.to(device)
            processed_input_images = []
            for input_image in input_images:
                if type(input_image) is str:
                    input_image = Image.open(input_image)

                input_image = input_image.resize((width, height))
                input_image = np.array(input_image)
                input_image = torch.tensor(input_image, dtype=torch.float32)
                input_image = util.rescale(input_image, (0, 255), (-1, 1))
                processed_input_images.append(input_image)
            input_images_tensor = torch.stack(processed_input_images).to(device)
            input_images_tensor = util.move_channel(input_images_tensor, to="first")

            _, _, height, width = input_images_tensor.shape
            noise_shape = (len(prompts), 4, height // 8, width // 8)

            encoder_noise = torch.randn(noise_shape, generator=generator, device=device)
            latents = encoder(input_images_tensor, encoder_noise)

            latents_noise = torch.randn(noise_shape, generator=generator, device=device)
            sampler.set_strength(strength=strength)
            latents += latents_noise * sampler.initial_scale

            to_idle(encoder)
            del encoder, processed_input_images, input_images_tensor, latents_noise
        else:
            latents = torch.randn(noise_shape, generator=generator, device=device)
            latents *= sampler.initial_scale

        diffusion = models.get('diffusion') or model_loader.load_diffusion(device)
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = util.get_time_embedding(timestep).to(device)

            input_latents = latents * sampler.get_input_scale()
            if do_cfg:
                input_latents = input_latents.repeat(2, 1, 1, 1)

            output = diffusion(input_latents, context, time_embedding)
            if do_cfg:
                output_cond, output_uncond = output.chunk(2)
                output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(latents, output)

        to_idle(diffusion)
        del diffusion

        decoder = models.get('decoder') or model_loader.load_decoder(device)
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)
        del decoder

        images = util.rescale(images, (-1, 1), (0, 255), clamp=True)
        images = util.move_channel(images, to="last")
        images = images.to('cpu', torch.uint8).numpy()

        return [Image.fromarray(image) for image in images]
