#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import logging 
import time 

from transformers import pipeline, set_seed
from transformers import AutoTokenizer, OPTForCausalLM

from utils import Utils

import random
import numpy as np 


def get_wikitext2(tokenizer, dataset="non-raw", nsamples=128, seqlen=2048):
    """ gptq """
    from datasets import load_dataset
    if dataset == "non-raw":
        traindata = load_dataset('wikitext', 'wikitext-2-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-v1', split='test')
    elif dataset == "raw":
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    else:
        raise ValueError(
                "You are using an unsupported dataset, only support wikitext2-raw-v1 and wikitext2-v1."
                "Using wikitext2-raw-v1 with --dataset=raw and wikitext2-v1 with --dataset=non-raw."
            )

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    dataloader = []
    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = testenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        dataloader.append((inp, tar))
    return dataloader, testenc


def eval_nonuniform(model, input_ids):
    """ from gptq """
    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))
    tot = 0.0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()))
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1,-1)),
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            times.append(time.time() - tick)
            tot += times[-1]
            cache['past'] = list(out.past_key_values)
            del out
        import numpy as np
    return np.median(times), tot


def perplexity(model, tokenizer, dataset, framework="pytorch"):
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)
    print(f"Calculating Perplexity on wikitext2 test set ...")
    model = model#.cuda()
    dataloader, testenc = get_wikitext2(tokenizer, dataset=dataset)
    
    model.seqlen = model.config.max_position_embeddings #2048
    test_enc = testenc.input_ids
    nsamples = test_enc.numel() // model.seqlen
    if framework == "pytorch":
        dtype = next(iter(model.parameters())).dtype

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    if framework == "pytorch":
        for i, layer in enumerate(model.model.decoder.layers):
            layer.register_forward_hook(clear_past(i))

    loss = torch.nn.CrossEntropyLoss()
    nlls = []

    with torch.no_grad():
        attention_mask = torch.ones((1, test_enc.numel()))#.cuda()
        for i in range(nsamples):
            batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]#.cuda()
            if framework == "pytorch":
                out = model(
                    batch,
                    attention_mask=attention_mask[:, (i * model.seqlen):((i + 1) * model.seqlen)].reshape((1, -1))
                )
            else :
                out = model(
                    batch,
                    attention_mask=batch.new_ones(batch.shape)
                )
            shift_labels = test_enc[
                :, (i * model.seqlen):((i + 1) * model.seqlen)
            ][:, 1:]#.cuda()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(out.logits[0][:-1, :], shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        print('Perplexity:', ppl.item())


def calibrate(model, tokenizer):
    print(f"Calculating Perplexity on wikitext2 test set ...")
    model.seqlen = model.config.max_position_embeddings #2048
    nsamples = 128 
    benchmark = 256 
    """ gptq """
    
    def get_dataset(tokenizer, nsamples=nsamples, seqlen=model.seqlen):
        from datasets import load_dataset
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        dataloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            dataloader.append((inp, tar))
        return dataloader, testenc

    dataloader, testloader = get_dataset(tokenizer, nsamples=nsamples, seqlen=model.seqlen)
    input_ids = next(iter(dataloader))[0][:, :benchmark]
    #input(model.config)
    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    loss = torch.nn.CrossEntropyLoss()
    tot = 0.

    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()))
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            # in transformers 4.27.2, torch.unsqueeze() i snot required. 
            # in transformers 4.30.2, batch_size was introduced, so torch.unsqueeze() is needed
            out = model(
                torch.unsqueeze(input_ids[:, i].reshape(-1), dim=0),
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            times.append(time.time() - tick)
            #print(i, times[-1])
            if i != input_ids.numel() - 1:
                tot += loss(out.logits[0], input_ids[:, (i + 1)]).float()
            cache['past'] = list(out.past_key_values)
            del out
        #import numpy as np
        #print('Median:', np.median(times))
        print('Perplexity:', torch.exp(tot / (input_ids.numel() - 1)).item())
    

prompts = [ "What is the meaning of life?",             
            "Tell me something you don't know.",        
            "What does Xilinx do?",                     
            "What is the mass of earth?",                
            "What is a poem?",                          
            "What is recursion?",                        
            "Tell me a one line joke.",                  
            "Who is Gilgamesh?",                         
            "Tell me something about cryptocurrency.",  
            "How did it all begin?"                     
            ]


# For quant_mode = 0
prompt_sets_quant_mode_0 = [
                {   'x_min':-40, 'x_max':40, 'x_scale':40/128,
                    'y_min':-0.22, 'y_max':0.22, 'y_scale':0.22/128,
                    'requantize_in_scale': 1, 'requantize_out_scale': 32,
                    'token_length':30,
                    'prompts':  [   "Who is Gilgamesh?",
                                    "Tell me something about cryptocurrency"
                                ]
                },
                {  'x_min':-43, 'x_max':43, 'x_scale':43/128,
                    'y_min':-0.15, 'y_max':0.15, 'y_scale':0.15/128,
                    'requantize_in_scale': 1, 'requantize_out_scale': 32,
                    'token_length':30,
                    'prompts':  [   "What is the meaning of life?",
                                    "What does Xilinx do?",
                                    "What is a poem?",
                                    "What is recursion?",
                                    "How did it all begin?"
                                ]
                },
                {   'x_min':-32, 'x_max':32, 'x_scale':32/128,
                    'y_min':-0.17, 'y_max':0.17, 'y_scale':0.17/128,
                    'requantize_in_scale': 1, 'requantize_out_scale': 32,
                    'token_length':30,
                    'prompts':  [   "What is the mass of earth?",
                                    "Tell me something you don't know.",
                                    "Tell me a one line joke.",
                                ]
                }
             ]


def warmup(model, tokenizer):
    print(f"Warming up ... ")
    for prompt in prompts[0:1]:
        inputs = tokenizer(prompt, return_tensors="pt") 
        generate_ids = model.generate(inputs.input_ids, max_length=30)
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Warm up DONE!! ")


def decode_prompt(model, tokenizer, prompt, input_ids=None, max_new_tokens=30):
    if input_ids is None:
        print(f"prompt: {prompt}")
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt") 
        end = time.time()
        logging.critical(f"[PROFILE][WARMUP] tokenizer: {end-start}")
    else:
        logging.critical(f"[PROFILE][WARMUP] tokenizer: na") # for logging consistency

    start, end = 0, 0
    prompt_tokens = 0
    if prompt is None:
        start = time.time()
        generate_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
        end = time.time()
        prompt_tokens = input_ids.shape[1]
    else:
        start = time.time()
        generate_ids = model.generate(inputs.input_ids, max_length=max_new_tokens)
        end = time.time()
        prompt_tokens = inputs.input_ids.shape[1]
    num_tokens_out = generate_ids.shape[1]
    new_tokens_generated = num_tokens_out - prompt_tokens
    generate_time = (end - start)
    time_per_token = (generate_time/new_tokens_generated)*1e3
    logging.critical(f"[PROFILE][AIE] generate: {generate_time} for {num_tokens_out} tokens; prompt-tokens: {prompt_tokens}")
    
    start = time.time()
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    end = time.time()
    logging.critical(f"[PROFILE][WARMUP] tokenizer decode: {end-start}")
    
    print(f"response: {response}")
    logging.critical(f"response: {response}")
            

def decode_prompts(model, tokenizer, max_new_tokens=30):
    for prompt in prompts:
        logging.critical("*"*40)
        print("*"*40)
        decode_prompt(model, tokenizer, prompt, max_new_tokens=max_new_tokens)



