import time

import torch
import torch.nn as nn

from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear

from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
    
def get_wikitext2(nsamples=128, seed=0, model="facebook/opt-125m", seqlen=128):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def ppl (model):
    DEV = torch.device("cpu")
    dataloader, testloader = get_wikitext2(nsamples=128, seed=0, model="facebook/" + model_name, seqlen=model.seqlen)
    model = model.to(DEV)
    benchmark = 2048
    input_ids = next(iter(dataloader))[0][:, :benchmark]

    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    
    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    #input(model.config)
    #print('Benchmarking ...')

    loss = nn.CrossEntropyLoss()
    tot = 0.

    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            #input(f"input_ids[:, i].reshape(-1) = {input_ids[:, i].reshape(-1).shape}")
            out = model(
                torch.unsqueeze(input_ids[:, i].reshape(-1), 0),
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            times.append(time.time() - tick)
            #print(i, times[-1])
            if i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        import numpy as np
        #print('Median:', np.median(times))
        print('Perplexity:', torch.exp(tot / (input_ids.numel() - 1)).item())

            
def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model


if __name__ == "__main__":
    model_name = "opt-1.3b"
    from transformers import OPTForCausalLM
    if False:
        model = OPTForCausalLM.from_pretrained("facebook/" + model_name)
        model.seqlen = model.config.max_position_embeddings
        ppl(model) # Perplexity: 12.506213188171387
        # Xeon (sda23): Perplexity: 12.506213188171387
        # Ryzen-AI: Perplexity: 12.506200790405273

    if True: # PTDQ
        model = OPTForCausalLM.from_pretrained("facebook/" + model_name)
        model.seqlen = model.config.max_position_embeddings
        model = torch.ao.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8 )
        ppl(model) # Perplexity: 13.454575538635254 

    if False: 
        model = OPTForCausalLM.from_pretrained("facebook/" + model_name)
        model.seqlen = model.config.max_position_embeddings
        model_w8a8 = quantize_model(model)
        #print(model_w8a8)
        ppl(model_w8a8) 
        # Xeon (sda23): Perplexity: 12.899502754211426
        # Ryzen-AI: Perplexity: 12.827589988708496

    if False:
        model = OPTForCausalLM.from_pretrained("facebook/" + model_name)
        model.seqlen = model.config.max_position_embeddings

        act_scales = torch.load("./act_scales/" + "%s.pt"%model_name)
        smooth_lm(model, act_scales, 0.5)
        model_smoothquant_w8a8 = quantize_model(model)
        print(model_smoothquant_w8a8)
        # Xeon (sda23): Perplexity: 12.556453704833984
        # Ryzen-AI Perplexity: 12.513752937316895


"""
   model_name = "opt-1.3b"
    dtype = "fp16"

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    if dtype=="fp32":
        model = OPTForCausalLM.from_pretrained("facebook/" + model_name, device_map="sequential")    
    else:
        model = OPTForCausalLM.from_pretrained("facebook/" + model_name, torch_dtype=torch.float16,  device_map="sequential")
    
    model.seqlen = model.config.max_position_embeddings
    ppl(model, torch.device("cuda")) 

    # 125M FP32: Perplexity: 27.683439254760742
    # 125M FP16: Perplexity: 27.686040878295900 OPTQ paper: 27.65
    
    # 350M FP32: Perplexity: 21.005409240722656
    # 350M FP16: Perplexity: 21.005483627319336 OPTQ paper: 22.00
    
    # 1.3B FP32: Perplexity: 12.506213188171387
    # 1.3B FP16: Perplexity: 12.507315635681152 OPTQ paper: 14.63
    
    # 13B FP32: Perplexity: 8.351198196411133
    # 13B FP16: Perplexity: can't fit in 1 GPU OPTQ paper: 10.13

    
    
    if False:
        model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8 )
        ppl(model) 
        # 13B : Perplexity: 5501.92333984375
        # 1.3B: Perplexity: 13.454575538635254 
"""