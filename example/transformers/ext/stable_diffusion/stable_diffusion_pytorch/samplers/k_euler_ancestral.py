import torch
import numpy as np
from .. import util


class KEulerAncestralSampler():
    def __init__(self, n_inference_steps=50, n_training_steps=1000, generator=None):
        timesteps = np.linspace(n_training_steps - 1, 0, n_inference_steps)

        alphas_cumprod = util.get_alphas_cumprod(n_training_steps=n_training_steps)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        log_sigmas = np.log(sigmas)
        log_sigmas = np.interp(timesteps, range(n_training_steps), log_sigmas)
        sigmas = np.exp(log_sigmas)
        sigmas = np.append(sigmas, 0)
        
        self.sigmas = sigmas
        self.initial_scale = sigmas.max()
        self.timesteps = timesteps
        self.n_inference_steps = n_inference_steps
        self.n_training_steps = n_training_steps
        self.step_count = 0
        self.generator = generator

    def get_input_scale(self, step_count=None):
        if step_count is None:
            step_count = self.step_count
        sigma = self.sigmas[step_count]
        return 1 / (sigma ** 2 + 1) ** 0.5

    def set_strength(self, strength=1):
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = np.linspace(self.n_training_steps - 1, 0, self.n_inference_steps)
        self.timesteps = self.timesteps[start_step:]
        self.initial_scale = self.sigmas[start_step]
        self.step_count = start_step

    def step(self, latents, output):
        t = self.step_count
        self.step_count += 1

        sigma_from = self.sigmas[t]
        sigma_to = self.sigmas[t + 1]
        sigma_up = sigma_to * (1 - (sigma_to ** 2 / sigma_from ** 2)) ** 0.5
        sigma_down = sigma_to ** 2 / sigma_from
        latents += output * (sigma_down - sigma_from)
        noise = torch.randn(
            latents.shape, generator=self.generator, device=latents.device)
        latents += noise * sigma_up
        return latents