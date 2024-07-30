
# this file is heavily modified version of pipeline.py in original repo
# these modifications are for internal analysis - Stable_Diffusion.doc
# contact @rajeevp

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from stable_diffusion_pytorch.tokenizer import Tokenizer
from stable_diffusion_pytorch.clip import CLIP
from stable_diffusion_pytorch.encoder import Encoder
from stable_diffusion_pytorch.decoder import Decoder
from stable_diffusion_pytorch.samplers import KLMSSampler, KEulerSampler, KEulerAncestralSampler
from stable_diffusion_pytorch import util
from stable_diffusion_pytorch import model_loader
import time 

import os 

from torchmetrics import StructuralSimilarityIndexMeasure

def calc_ssim(im1, im2):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    res = ssim(im1, im2)
    return res 
    

def run(prompts,
        strength=0.9,
        cfg_scale=7.5,
        height=512,
        width=512,
        sampler="k_euler",
        n_inference_steps=50,
        datatype={'Clip':'float32', 'Diffuser':'float32', 'Decoder':'float32'},
):
    with torch.no_grad():
        if height % 8 or width % 8:
            raise ValueError("height and width must be a multiple of 8")
        uncond_prompts = [""] * len(prompts)        
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu') #Dynamic Quantization only works on CPU

        generator = torch.Generator(device=device)
        generator.manual_seed(123)

        tokenizer = Tokenizer()
        clip = model_loader.load_clip(device)
        #torch.save(clip.state_dict(), "temp.p")
        #size = os.path.getsize("temp.p")
        #print(f"Model size: Clip {size/1e3} KB" )
        # Model size: Clip 492280.855 KB
        
        if datatype['Clip'] == "ptdq":
            start = time.time_ns() 
            clip = torch.ao.quantization.quantize_dynamic(
                        clip, {torch.nn.Linear}, dtype=torch.qint8 )
            end = time.time_ns() 
            print(f"[PROFILE] Clip quantization time: {(end-start)*1e-9}s")
            #torch.save(clip.state_dict(), "temp.p")
            #size = os.path.getsize("temp.p")
            #print(f"Model size: Clip Quantized {size/1e3} KB" )
            # Model size: Clip Quantized 237511.735 KB
        elif datatype['Clip'] == 'bfloat16':
            clip = clip.to(torch.bfloat16)
        else: # float32
            pass 
        clip.to(device)
        
        cond_tokens = tokenizer.encode_batch(prompts)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        cond_context = clip(cond_tokens)
        uncond_tokens = tokenizer.encode_batch(uncond_prompts)
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        
        start = time.time_ns()
        uncond_context = clip(uncond_tokens)
        end = time.time_ns() 
        print(f"[PROFILE] Clip: {(end-start)*1e-9}s")
        
        context = torch.cat([cond_context, uncond_context])
        
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

        latents = torch.randn(noise_shape, generator=generator, device=device)
        latents *= sampler.initial_scale

        diffusion = model_loader.load_diffusion(device)
        #torch.save(diffusion.state_dict(), "temp.p")
        #size = os.path.getsize("temp.p")
        #print(f"Model size: Diffuser: {size/1e3} KB" )
        # Model size: Diffuser: 3438271.477 KB

        if datatype['Diffuser'] == 'ptdq':
            start = time.time_ns() 
            diffusion = torch.ao.quantization.quantize_dynamic(
                        diffusion, {torch.nn.Linear}, dtype=torch.qint8 )
            end = time.time_ns() 
            print(f"[PROFILE] Diffuser quantization time: {(end-start)*1e-9}s")
            #torch.save(diffusion.state_dict(), "temp.p")
            #size = os.path.getsize("temp.p")
            #print(f"Model size: Diffuser Quantized : {size/1e3} KB" )
            # Model size: Diffuser Quantized : 2628241.093 KB
            
        elif datatype['Diffuser'] == 'bfloat16':
            diffusion = diffusion.to(torch.bfloat16)
        else: # float32
            pass 
        diffusion.to(device)
        
        #print("Diffusion Model ******")
        #print(diffusion)

        timesteps = tqdm(sampler.timesteps)
        start = time.time_ns() 
        print(f"[PROFILE] number of diffusion timesteps: {len(timesteps)}")
        for i, timestep in enumerate(timesteps):
            time_embedding = util.get_time_embedding(timestep).to(device)

            input_latents = latents * sampler.get_input_scale()
            input_latents = input_latents.repeat(2, 1, 1, 1)

            if (datatype['Clip'] == 'bfloat16') and (datatype['Diffuser'] == 'float32'):
                time_embedding = time_embedding.to(torch.float)
                input_latents = input_latents.to(torch.float)
                context = context.to(torch.float)
            elif (datatype['Diffuser'] == 'bfloat16'):
                time_embedding = time_embedding.to(torch.bfloat16)
                input_latents = input_latents.to(torch.bfloat16)
                context = context.to(torch.bfloat16)
            else:
                pass 
            output = diffusion(input_latents, context, time_embedding)
            output_cond, output_uncond = output.chunk(2)
            output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(latents, output)
        end = time.time_ns() 
        print(f"[PROFILE] Diffusion for timesteps={timesteps} time: {(end-start)*1e-9}s")
        
        del diffusion

        decoder = model_loader.load_decoder(device)
        #torch.save(decoder.state_dict(), "temp.p")
        #size = os.path.getsize("temp.p")
        #print(f"Model size: decoder : {size/1e3} KB" )
        # Model size: decoder : 197994.835 KB

        if datatype['Decoder'] == 'ptdq':
            start = time.time_ns() 
            decoder = torch.ao.quantization.quantize_dynamic(
                        decoder, {torch.nn.Linear}, dtype=torch.qint8 )
            end = time.time_ns() 
            print(f"[PROFILE] Decoder quantization time: {(end-start)*1e-9}s")
            #torch.save(decoder.state_dict(), "temp.p")
            #size = os.path.getsize("temp.p")
            #print(f"Model size: decoder Quantized : {size/1e3} KB" )
            # Model size: decoder Quantized : 194850.687 KB
        elif datatype['Decoder'] == 'bfloat16':
            decoder = decoder.to(torch.bfloat16)
        else: # float32
            pass 
        decoder.to(device)
        #print("Decoder Model ******")
        #print(decoder)
        
        if (datatype['Diffuser'] == 'bfloat16') and (datatype['Decoder'] == 'float32'):
            latents = latents.to(torch.float)
        elif (datatype['Decoder'] == 'bfloat16'):
            latents = latents.to(torch.bfloat16)          
        else:
            pass
        start = time.time_ns()
        images = decoder(latents)
        end = time.time_ns() 
        print(f"[PROFILE] Decode time (image generation): {(end-start)*1e-9}s")
        
        del decoder
        images = images.to(torch.float)
        images = util.rescale(images, (-1, 1), (0, 255), clamp=True)
        images = util.move_channel(images, to="last")
        images = images.to('cpu', torch.uint8).numpy()

        return [Image.fromarray(image) for image in images]


if __name__ == "__main__":
    prompts = ["albert bierstadt style painting of los angeles"]
    #prompts = ["a man floating in space, large scale, realistic proportions, highly detailed, smooth sharp focus, ray tracing, digital painting, art illustration"]
    
    datatype_list = [   #{'Clip': 'ptdq', 'Diffuser':'ptdq', 'Decoder':'ptdq'},
                        #{'Clip': 'bfloat16', 'Diffuser':'float32', 'Decoder':'float32'},
                        #{'Clip': 'float32', 'Diffuser':'bfloat16', 'Decoder':'float32'},
                        {'Clip': 'float32', 'Diffuser':'float32', 'Decoder':'bfloat16'},
                        {'Clip': 'bfloat16', 'Diffuser':'bfloat16', 'Decoder':'bfloat16'}
                    ] 
    
    # load fp32 image for ssim comparison
    fp32_image = Image.open("output_euler_50_fp32_A.jpg")
    fp32_im_torch = torch.tensor(np.array(fp32_image)).unsqueeze(0)/255.0 
    fp32_im_torch = fp32_im_torch.reshape((1, fp32_im_torch.shape[3], fp32_im_torch.shape[1], fp32_im_torch.shape[2]))
    ssim = calc_ssim(fp32_im_torch, fp32_im_torch)
    print(f"SSIM: FP32: {ssim}")
    
    timesteps = 50
    
    for datatype in datatype_list:
        images = run(prompts, sampler="k_euler", n_inference_steps=timesteps, datatype=datatype)
        import time 
        start = time.time_ns()
        im_torch = torch.tensor(np.array(images[0])).unsqueeze(0)/255.0
        im_torch = im_torch.reshape((1, im_torch.shape[3], im_torch.shape[1], im_torch.shape[2]))
        ssim = calc_ssim(fp32_im_torch, im_torch)
        print(f"SSIM: {ssim}")
        images[0].save('output_euler_%d_clip_%s_diffuser_%s_decoder_%s_ssim_%0.4f.jpg'%(timesteps, datatype['Clip'], datatype['Diffuser'], datatype['Decoder'], ssim))
        end = time.time_ns()
        print(f"Time to generate image with is {(end-start)*1e-9}s")
